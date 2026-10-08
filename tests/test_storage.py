"""Storage abstraction tests: LocalStorage round-trips, atomicity, the
VOSpace CLI argv seam (the agreed one-line adaptation point for CANFAR),
and traversal rejection at the storage boundary."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
import pytest

from xmatcher.exceptions import ConfigError
from xmatcher.storage import (
    LocalStorage,
    VOSpaceStorage,
    all_cache_roots,
    assert_headroom,
    default_cache_root,
    default_output_root,
    open_storage,
)


def test_local_roundtrip(tmp_path: Path) -> None:
    store = open_storage(tmp_path / "cache")
    assert isinstance(store, LocalStorage)

    df = pl.DataFrame({"id": [1, 2], "ra": [10.0, 11.0]})
    store.write_parquet(df, "c/part.parquet")
    assert store.exists("c/part.parquet")
    assert store.size("c/part.parquet") > 0
    assert store.list("c") == ["part.parquet"]
    assert store.read_parquet("c/part.parquet").equals(df)

    out = tmp_path / "stage.parquet"
    store.stage_in("c/part.parquet", out)
    assert out.exists()
    store.stage_out(out, "c/copy.parquet")
    assert store.read_parquet("c/copy.parquet").height == 2

    store.rename("c/part.parquet", "c/moved.parquet")
    assert not store.exists("c/part.parquet")
    assert store.exists("c/moved.parquet")

    store.rm("c/moved.parquet")
    assert not store.exists("c/moved.parquet")
    store.rm("c/moved.parquet")  # missing is a no-op
    store.rm("c")  # directories too


def test_write_parquet_atomic_failure_path(tmp_path: Path, monkeypatch) -> None:
    """A failed write leaves no partial target and no stray tmp file.

    Regression guard: writing directly to the target (no tmp + os.replace)
    would pass the old success-only assertion; here the failure path is
    exercised so the atomic contract actually has teeth.
    """
    store = LocalStorage(tmp_path)
    target = tmp_path / "x" / "y.parquet"
    store.write_parquet(pl.DataFrame({"a": [1]}), "x/y.parquet")
    before = target.read_bytes()

    def explode(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("polars.DataFrame.write_parquet", explode)
    with pytest.raises(OSError, match="disk full"):
        store.write_parquet(pl.DataFrame({"a": [2]}), "x/y.parquet")
    # pre-existing target untouched, no orphaned tmp
    assert target.read_bytes() == before
    assert list((tmp_path / "x").iterdir()) == [target]


def test_write_parquet_preserves_umask_created_permissions(tmp_path: Path) -> None:
    store = LocalStorage(tmp_path)
    previous_umask = os.umask(0o027)
    try:
        store.write_parquet(pl.DataFrame({"a": [1]}), "shared.parquet")
    finally:
        os.umask(previous_umask)

    assert stat.S_IMODE((tmp_path / "shared.parquet").stat().st_mode) == 0o640


def test_stage_out_failure_leaves_no_tmp(tmp_path: Path, monkeypatch) -> None:
    store = LocalStorage(tmp_path)
    src = tmp_path / "payload.dat"
    src.write_bytes(b"data-bytes")
    store.stage_out(src, "c/t.dat")
    assert (tmp_path / "c" / "t.dat").read_bytes() == b"data-bytes"

    def explode(*args, **kwargs):
        raise OSError("copy failed")

    monkeypatch.setattr("shutil.copy2", explode)
    with pytest.raises(OSError, match="copy failed"):
        store.stage_out(src, "c/t.dat")
    # pre-existing target untouched, no orphaned tmp
    assert (tmp_path / "c" / "t.dat").read_bytes() == b"data-bytes"
    assert sorted(p.name for p in (tmp_path / "c").iterdir()) == ["t.dat"]


def test_concurrent_stage_out_uses_distinct_temporary_files(tmp_path: Path, monkeypatch) -> None:
    store = LocalStorage(tmp_path / "cache")
    sources = [tmp_path / "one", tmp_path / "two"]
    sources[0].write_bytes(b"one")
    sources[1].write_bytes(b"two")
    both_opened = threading.Barrier(2)
    both_written = threading.Barrier(2)

    def synchronized_copy(src, dst):
        with Path(dst).open("wb") as handle:
            both_opened.wait(timeout=5)
            handle.write(Path(src).read_bytes())
            both_written.wait(timeout=5)

    monkeypatch.setattr("xmatcher.storage.shutil.copy2", synchronized_copy)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(store.stage_out, src, "same.dat") for src in sources]
        errors = [future.exception(timeout=10) for future in futures]

    assert errors == [None, None]
    assert (store.root / "same.dat").read_bytes() in {b"one", b"two"}


def test_rel_traversal_rejected(tmp_path: Path) -> None:
    """Server-controlled rels must not escape the storage root."""
    store = LocalStorage(tmp_path)
    victim = tmp_path.parent / "victim.parquet"
    victim.write_bytes(b"precious")
    for rel in ("../victim.parquet", "a/../../victim.parquet", ".."):
        with pytest.raises(ValueError, match="escapes the root"):
            store.write_parquet(pl.DataFrame({"a": [1]}), rel)
        with pytest.raises(ValueError, match="escapes the root"):
            store.stage_out(victim, rel)
    assert victim.read_bytes() == b"precious"
    # VOSpace URI building rejects traversal-shaped rels too
    vos = VOSpaceStorage.__new__(VOSpaceStorage)
    vos.root = "vos:x"
    with pytest.raises(ValueError, match="escapes the root"):
        vos._uri("../x.parquet")
    with pytest.raises(ValueError, match="escapes the root"):
        vos._argv("put", "/tmp/local", "a/../../victim.parquet")


def test_rel_symlink_cannot_escape_local_storage_root(tmp_path: Path) -> None:
    store = LocalStorage(tmp_path / "cache")
    outside = tmp_path / "outside"
    outside.mkdir()
    (store.root / "escape").symlink_to(outside, target_is_directory=True)
    payload = tmp_path / "payload.dat"
    payload.write_bytes(b"must stay inside")

    with pytest.raises(ValueError, match="escapes the root"):
        store.stage_out(payload, "escape/payload.dat")

    assert not (outside / "payload.dat").exists()


def test_assert_headroom_uses_binary_gibibytes(tmp_path: Path, monkeypatch) -> None:
    from collections import namedtuple

    from xmatcher.exceptions import CrossMatchError

    DiskUsage = namedtuple("DiskUsage", "total used free")
    monkeypatch.setattr(
        "xmatcher.storage.shutil.disk_usage",
        lambda _: DiskUsage(2**40, 2**40 - 1_000_000_000, 1_000_000_000),
    )
    with pytest.raises(CrossMatchError, match="below the --min-free-gb floor"):
        assert_headroom(tmp_path, 1.0, "test filesystem")

    monkeypatch.setattr(
        "xmatcher.storage.shutil.disk_usage",
        lambda _: DiskUsage(2**40, 0, 1024**3),
    )
    assert_headroom(tmp_path, 1.0, "test filesystem")


def test_rm_unlinks_internal_symlinks_without_removing_their_targets(tmp_path: Path) -> None:
    store = LocalStorage(tmp_path / "cache")
    target = store.root / "target"
    target.mkdir()
    alias = store.root / "alias"
    alias.symlink_to(target, target_is_directory=True)
    dangling = store.root / "dangling"
    dangling.symlink_to(store.root / "missing")

    store.rm("alias")
    store.rm("dangling")

    assert target.is_dir()
    assert not alias.is_symlink()
    assert not dangling.is_symlink()


def _fake_vos_binary(monkeypatch, tmp_path: Path, scripts: dict[str, str] | None = None):
    """Install a fake ``vos`` binary with per-URI stdout scripts.

    ``scripts`` maps the last argv item (the URI or local path) to stdout;
    entries missing from the map emit nothing.
    """
    fake = tmp_path / "bin" / "vos"
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_text("#!/bin/sh\ncat", encoding="utf-8")
    fake.chmod(0o755)
    runs: list[list[str]] = []
    scripts = scripts or {}

    def fake_run(argv, **kw):
        runs.append(argv)
        out = scripts.get(argv[-1], "")
        return subprocess.CompletedProcess(argv, 0, stdout=out, stderr="")

    monkeypatch.setattr("shutil.which", lambda name, **kw: str(fake) if name == "vos" else None)
    monkeypatch.setattr("xmatcher.storage.subprocess.run", fake_run)
    return runs


def test_vospace_argv_seam(monkeypatch, tmp_path) -> None:
    """The vos CLI argv (one-line adaptation point for CANFAR) is pinned.

    Verified here against the reference pyvos console scripts: ``vls`` has no
    ``--recursive``/``-L`` flags and ``vrm`` has no ``--force``, so this test
    would fail loudly if anyone reintroduces those non-portable flags.
    """
    runs = _fake_vos_binary(
        monkeypatch, tmp_path, scripts={"vos:probe/root": "dataset\nproperties\n"}
    )
    store = VOSpaceStorage("vos:probe/root")  # _ensure_root probes vls
    assert runs[-1] == ["vos", "vls", "vos:probe/root"]

    store.list("dataset")
    assert ["vos", "vls", "vos:probe/root/dataset"] in runs
    assert not any("--recursive" in r for r in runs)

    store.list("")
    assert ["vos", "vls", "vos:probe/root"] in runs

    store.size("dataset/Npix=3.parquet")
    assert ["vos", "vls", "-l", "vos:probe/root/dataset/Npix=3.parquet"] in runs
    assert not any("-L" in r for r in runs)

    store.rm("dataset/Npix=3.parquet")
    assert ["vos", "vrm", "vos:probe/root/dataset/Npix=3.parquet"] in runs
    assert not any("--force" in r for r in runs)

    store.stage_in("dataset/Npix=3.parquet", Path("/tmp/go.parquet"))
    assert ["vos", "vcp", "vos:probe/root/dataset/Npix=3.parquet", "/tmp/go.parquet"] in runs

    store.mkdir("dataset")
    assert ["vos", "vmkdir", "-p", "vos:probe/root/dataset"] in runs


def test_vospace_list_parses_basenames_and_sizes(monkeypatch, tmp_path) -> None:
    """vls output (basenames, no trailing slash on dirs) feeds _walk_storage."""
    long_out = (
        "-rw-r--r--  usr  grp  1234  2024-01-01  Npix=3.parquet\n"
        "-rw-r--r--  usr  grp  0  2024-01-01  dataset\n"
    )
    runs = _fake_vos_binary(
        monkeypatch,
        tmp_path,
        scripts={
            "vos:probe/root/data/Npix=3.parquet": long_out,
            "vos:probe/root/data": "Npix=3.parquet\ndataset/\n",
        },
    )
    store = VOSpaceStorage("vos:probe/root")
    # list dir: basename-only entries, trailing slashes stripped
    names = store.list("data")
    assert names == ["Npix=3.parquet", "dataset"]
    # size: the numeric column closest to the name wins (not the year)
    assert store.size("data/Npix=3.parquet") == 1234
    assert ["vos", "vls", "-l", "vos:probe/root/data/Npix=3.parquet"] in runs


def test_vospace_stage_out_mkdirs_parents(monkeypatch, tmp_path) -> None:
    """vcp does not create intermediate containers; stage_out must mkdir first."""
    runs = _fake_vos_binary(monkeypatch, tmp_path)
    store = VOSpaceStorage("vos:probe/root")
    local = tmp_path / "put.parquet"
    local.write_bytes(b"x")
    store.stage_out(local, "name/v1/dataset/Npix=3.parquet")
    mkdir = runs.index(["vos", "vmkdir", "-p", "vos:probe/root/name/v1/dataset"])
    put = runs.index(["vos", "vcp", str(local), "vos:probe/root/name/v1/dataset/Npix=3.parquet"])
    assert mkdir < put


def test_vospace_python_stage_out_reuses_existing_parent(tmp_path: Path) -> None:
    """The python-vos fallback must tolerate repeated mkdirs like `vmkdir -p`."""

    class Node:
        def __init__(self, node_type: str) -> None:
            self.type = node_type

    class Client:
        def __init__(self) -> None:
            self.nodes = {"vos:probe/root": Node("vos:ContainerNode")}
            self.mkdir_calls: list[str] = []
            self.copied: dict[str, bytes] = {}

        def get_node(self, uri: str) -> Node:
            if uri not in self.nodes:
                raise FileNotFoundError(uri)
            return self.nodes[uri]

        def mkdir(self, uri: str) -> None:
            self.mkdir_calls.append(uri)
            if uri in self.nodes:
                raise FileExistsError(uri)
            self.nodes[uri] = Node("vos:ContainerNode")

        def copy(self, source: str, destination: str) -> None:
            self.copied[destination] = Path(source).read_bytes()
            self.nodes[destination] = Node("vos:DataNode")

    store = VOSpaceStorage.__new__(VOSpaceStorage)
    store.root = "vos:probe/root"
    store._binary = None
    store._client = Client()
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    store.stage_out(first, "dataset/first.parquet")
    store.stage_out(second, "dataset/second.parquet")

    assert store._client.copied == {
        "vos:probe/root/dataset/first.parquet": b"first",
        "vos:probe/root/dataset/second.parquet": b"second",
    }
    assert store._client.mkdir_calls == ["vos:probe/root/dataset"]

    store._client.nodes["vos:probe/root/not-a-container"] = Node("vos:DataNode")
    with pytest.raises(OSError, match="exists and is not a container"):
        store.mkdir("not-a-container")


def test_vospace_python_list_normalizes_names() -> None:
    """python-vos 3.7 returns child names, not Node objects, from listdir."""

    class Node:
        uri = "vos:probe/root/catalogue/dataset/"

    class Client:
        def listdir(self, uri: str) -> list[object]:
            assert uri == "vos:probe/root/catalogue"
            return ["properties", Node()]

    store = VOSpaceStorage.__new__(VOSpaceStorage)
    store.root = "vos:probe/root"
    store._binary = None
    store._client = Client()

    assert store.list("catalogue") == ["dataset", "properties"]


def test_open_storage_routing(tmp_path: Path) -> None:
    assert isinstance(open_storage(tmp_path), LocalStorage)
    assert isinstance(open_storage(str(tmp_path)), LocalStorage)


def test_default_cache_root(monkeypatch) -> None:
    key = "XMATCHER_CACHE_ROOT"
    monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("xmatcher.storage.platform_arc_root", lambda: None)
    assert default_cache_root() == str(Path.home() / ".cache" / "xmatcher")
    monkeypatch.setenv(key, "/tmp/xc-root")
    assert default_cache_root() == "/tmp/xc-root"


def test_all_cache_roots_precedence_and_dedup(monkeypatch) -> None:
    """Primary = $XMATCHER_CACHE_ROOT > cache.root > default; replicas follow."""
    key = "XMATCHER_CACHE_ROOT"
    default = str(Path.home() / ".cache" / "xmatcher")
    monkeypatch.delenv(key, raising=False)
    # off-platform probe: tests must not depend on where they run
    monkeypatch.setattr("xmatcher.storage.platform_arc_root", lambda: None)

    # unset everything -> the default root alone (never [])
    assert all_cache_roots() == [default]

    # config root primary, replicas appended, blanks dropped
    cfg = {"root": "/cfg/root", "roots": ["vos:rep-a", "", "vos:rep-a", "vos:rep-b"]}
    assert all_cache_roots(cfg) == ["/cfg/root", "vos:rep-a", "vos:rep-b"]

    # env wins the primary slot; configured replicas still ride along
    monkeypatch.setenv(key, "/env/root")
    assert all_cache_roots(cfg) == ["/env/root", "vos:rep-a", "vos:rep-b"]
    assert all_cache_roots() == ["/env/root"]

    # env identical to a replica does not duplicate it
    assert all_cache_roots({"roots": ["/env/root", "vos:rep-a"]}) == [
        "/env/root",
        "vos:rep-a",
    ]


def test_default_cache_root_prefers_platform_arc(monkeypatch) -> None:
    """On AstroAI/CANFAR sessions the cache defaults to /arc/projects/hats —
    never a home directory — and XMATCHER_CACHE_ROOT still wins."""
    key = "XMATCHER_CACHE_ROOT"
    monkeypatch.delenv(key, raising=False)
    home_default = str(Path.home() / ".cache" / "xmatcher")

    monkeypatch.setattr("xmatcher.storage.platform_arc_root", lambda: "/arc")
    assert default_cache_root() == "/arc/projects/hats"

    monkeypatch.setattr("xmatcher.storage.platform_arc_root", lambda: None)
    assert default_cache_root() == home_default

    # env override beats the platform default
    monkeypatch.setenv(key, "vos:hats")
    monkeypatch.setattr("xmatcher.storage.platform_arc_root", lambda: "/arc")
    assert default_cache_root() == "vos:hats"


def test_default_output_root_platform_and_env(monkeypatch) -> None:
    """Outputs default to /arc/projects/hats/xmatcher on the platform, stay
    cwd-relative off it, and XMATCHER_OUTPUT_ROOT (incl. empty=off) wins."""
    key = "XMATCHER_OUTPUT_ROOT"
    monkeypatch.delenv(key, raising=False)

    monkeypatch.setattr("xmatcher.storage.platform_arc_root", lambda: None)
    assert default_output_root() is None

    monkeypatch.setattr("xmatcher.storage.platform_arc_root", lambda: "/arc")
    assert default_output_root() == "/arc/projects/hats/xmatcher"

    monkeypatch.setenv(key, "vos:hats/xmatcher")
    assert default_output_root() == "vos:hats/xmatcher"

    monkeypatch.setenv(key, "   ")
    assert default_output_root() is None


def test_vospace_requires_backend(monkeypatch) -> None:
    """Without a vos binary or python package, a vos: root is a ConfigError."""
    monkeypatch.setattr("shutil.which", lambda name, **kw: None)
    saved = sys.modules.pop("vos", None)
    try:
        with pytest.raises(ConfigError, match="VOSpace cache root"):
            VOSpaceStorage("vos:some/container")
    finally:
        if saved is not None:
            sys.modules["vos"] = saved
