"""Storage abstraction tests: LocalStorage round-trips, atomicity, the
VOSpace CLI argv seam (the agreed one-line adaptation point for CANFAR),
and traversal rejection at the storage boundary."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

from xmatch.exceptions import ConfigError
from xmatch.storage import (
    LocalStorage,
    VOSpaceStorage,
    all_cache_roots,
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
    monkeypatch.setattr("xmatch.storage.subprocess.run", fake_run)
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
            "vos:probe/root/data": "Npix=3.parquet\ndataset\n",
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


def test_open_storage_routing(tmp_path: Path) -> None:
    assert isinstance(open_storage(tmp_path), LocalStorage)
    assert isinstance(open_storage(str(tmp_path)), LocalStorage)


def test_default_cache_root(monkeypatch) -> None:
    key = "XMATCH_CACHE_ROOT"
    monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("xmatch.storage.platform_arc_root", lambda: None)
    assert default_cache_root() == str(Path.home() / ".cache" / "xmatch")
    monkeypatch.setenv(key, "/tmp/xc-root")
    assert default_cache_root() == "/tmp/xc-root"


def test_all_cache_roots_precedence_and_dedup(monkeypatch) -> None:
    """Primary = $XMATCH_CACHE_ROOT > cache.root > default; replicas follow."""
    key = "XMATCH_CACHE_ROOT"
    default = str(Path.home() / ".cache" / "xmatch")
    monkeypatch.delenv(key, raising=False)
    # off-platform probe: tests must not depend on where they run
    monkeypatch.setattr("xmatch.storage.platform_arc_root", lambda: None)

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
    never a home directory — and XMATCH_CACHE_ROOT still wins."""
    key = "XMATCH_CACHE_ROOT"
    monkeypatch.delenv(key, raising=False)
    home_default = str(Path.home() / ".cache" / "xmatch")

    monkeypatch.setattr("xmatch.storage.platform_arc_root", lambda: "/arc")
    assert default_cache_root() == "/arc/projects/hats"

    monkeypatch.setattr("xmatch.storage.platform_arc_root", lambda: None)
    assert default_cache_root() == home_default

    # env override beats the platform default
    monkeypatch.setenv(key, "vos:hats")
    monkeypatch.setattr("xmatch.storage.platform_arc_root", lambda: "/arc")
    assert default_cache_root() == "vos:hats"


def test_default_output_root_platform_and_env(monkeypatch) -> None:
    """Outputs default to /arc/projects/hats/xmatch on the platform, stay
    cwd-relative off it, and XMATCH_OUTPUT_ROOT (incl. empty=off) wins."""
    key = "XMATCH_OUTPUT_ROOT"
    monkeypatch.delenv(key, raising=False)

    monkeypatch.setattr("xmatch.storage.platform_arc_root", lambda: None)
    assert default_output_root() is None

    monkeypatch.setattr("xmatch.storage.platform_arc_root", lambda: "/arc")
    assert default_output_root() == "/arc/projects/hats/xmatch"

    monkeypatch.setenv(key, "vos:hats/xmatch")
    assert default_output_root() == "vos:hats/xmatch"

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
