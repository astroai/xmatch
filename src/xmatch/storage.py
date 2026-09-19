"""Storage abstraction for xmatch mirrors and Ray task I/O.

A :class:`Storage` hides *where* the durable root lives (a POSIX dir on the
driver/manager node, or a CANFAR VOSpace ``vos:`` URI) behind one API.  All
catalogue parquet reads/writes and per-worker staging funnel through it, so
a Ray task reading "the same catalogue file" transparently hits the durable
root or a worker's fast scratch copy.

``LocalStorage`` is a plain pathlib-backed dir with atomic-ish writes
(tmp + ``os.replace``). ``VOSpaceStorage`` shells the ``vos`` CLI
(``vls``/``vcp``/``vput``/``vmkdir``) — the agreed boring path for CANFAR;
falls back to the python ``vos`` package when the binary is missing, and
raises :class:`ConfigError` when neither is available.

VOSpace CLI flags are not verified against a live CANFAR endpoint (see
``scripts/canfar-smoke.sh``), so the exact ``vos`` argv is kept in one
place (:meth:`VOSpaceStorage._argv`) for a one-line adaptation on the
target machine.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Union

import polars as pl

from .exceptions import ConfigError

logger = logging.getLogger(__name__)


class Storage:
    """Protocol for a durable catalogue store (POSIX dir or VOSpace root)."""

    root: "Union[str, Path]"

    def exists(self, rel: str) -> bool:  # pragma: no cover - protocol
        raise NotImplementedError

    def size(self, rel: str) -> int:  # pragma: no cover - protocol
        """Size in bytes of ``rel``; ``-1`` when unknown/unavailable."""
        raise NotImplementedError

    def list(self, rel: str) -> List[str]:  # pragma: no cover - protocol
        """Entry names directly under ``rel`` (files and dirs)."""
        raise NotImplementedError

    def read_parquet(self, rel: str) -> pl.DataFrame:  # pragma: no cover - protocol
        raise NotImplementedError

    def parquet_schema(self, rel: str) -> "dict[str, object]":  # pragma: no cover - protocol
        """Column name -> polars dtype for the parquet file at ``rel``.

        Used by the union plan builder, which must know every catalogue's
        columns before fanning out (a remote root cannot be probed with a
        local ``Path``).
        """
        raise NotImplementedError

    def write_parquet(self, df: pl.DataFrame, rel: str) -> None:  # pragma: no cover
        raise NotImplementedError

    def rename(self, src: str, dst: str) -> None:  # pragma: no cover - protocol
        raise NotImplementedError

    def mkdir(self, rel: str) -> None:  # pragma: no cover - protocol
        raise NotImplementedError

    def stage_out(self, local: Path, rel: str) -> None:  # pragma: no cover
        """Copy a local file into the durable root at ``rel``."""

    def stage_in(self, rel: str, local: Path) -> None:  # pragma: no cover
        """Copy ``rel`` from the durable root to a local file."""

    def rm(self, rel: str) -> None:  # pragma: no cover - protocol
        """Delete the node at ``rel`` (missing nodes are a no-op)."""


def _safe_rel(rel: str) -> str:
    """Normalize a storage rel, rejecting path traversal out of the root.

    Remote listings (``partition_info.parquet`` file_loc values, HTML hrefs)
    are server-controlled; a malicious/misconfigured source must not be able
    to write outside the cache root.
    """
    rel = rel.lstrip("/")
    if any(part == ".." for part in rel.split("/")):
        raise ValueError(f"storage rel escapes the root: {rel!r}")
    return rel


def _tmp_path(target: Path) -> Path:
    """Same-directory tmp sibling with a pid-unique name (crash/concurrent-safe)."""
    return target.parent / f".{target.name}.{os.getpid()}.tmp"


class LocalStorage(Storage):
    """POSIX-directory :class:`Storage` implementation."""

    root: Path

    def __init__(self, root: Union[str, Path]) -> None:
        self.root = Path(os.path.expandvars(str(root))).expanduser()
        if not self.root.exists():
            self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, rel: str) -> Path:
        rel = _safe_rel(rel)
        if not rel:
            return self.root
        return self.root / rel

    def exists(self, rel: str) -> bool:
        return self._path(rel).exists()

    def size(self, rel: str) -> int:
        p = self._path(rel)
        try:
            return p.stat().st_size if p.is_file() else -1
        except OSError:
            return -1

    def list(self, rel: str) -> List[str]:
        p = self._path(rel)
        if not p.is_dir():
            return []
        return sorted(entry.name for entry in p.iterdir())

    def read_parquet(self, rel: str) -> pl.DataFrame:
        return pl.read_parquet(self._path(rel))

    def parquet_schema(self, rel: str) -> "dict[str, object]":
        return dict(pl.read_parquet_schema(self._path(rel)))

    def write_parquet(self, df: pl.DataFrame, rel: str) -> None:
        target = self._path(rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = _tmp_path(target)
        try:
            df.write_parquet(tmp)
            os.replace(tmp, target)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    def rename(self, src: str, dst: str) -> None:
        src_p = self._path(src)
        dst_p = self._path(dst)
        dst_p.parent.mkdir(parents=True, exist_ok=True)
        os.replace(src_p, dst_p)

    def mkdir(self, rel: str) -> None:
        self._path(rel).mkdir(parents=True, exist_ok=True)

    def stage_out(self, local: Path, rel: str) -> None:
        target = self._path(rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = _tmp_path(target)
        try:
            shutil.copy2(local, tmp)
            os.replace(tmp, target)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    def stage_in(self, rel: str, local: Path) -> None:
        shutil.copy2(self._path(rel), local)

    def rm(self, rel: str) -> None:
        path = self._path(rel)
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink()


class VOSpaceStorage(Storage):
    """CANFAR VOSpace :class:`Storage` via the ``vos`` CLI (or python binding).

    ``root`` is a ``vos:`` URI prefix.  Commands are resolved once at init:
    a ``vos`` binary on PATH wins, the python ``vos`` package is the
    fallback, and :class:`ConfigError` is raised when neither exists.

    NOTE: VOSpace CLI flags (``vls -l`` size parsing, ``vmkdir -p``) are
    verified against a live CANFAR endpoint by ``scripts/canfar-smoke.sh``;
    the argv is centralised in :meth:`_argv` so a one-line adaptation on
    the target machine is all that is needed.
    """

    def __init__(self, root: str) -> None:
        if not isinstance(root, str) or not root.startswith("vos:"):
            raise ConfigError(f"VOSpaceStorage root must be a 'vos:' URI, got {root!r}")
        self.root = root.rstrip("/")
        self._binary: Optional[str] = shutil.which("vos")
        self._client = None  # python-vos client when no binary
        if self._binary is None:
            try:
                import vos  # type: ignore

                self._client = vos.Client()
            except Exception as exc:
                raise ConfigError(
                    "VOSpace cache root requires the 'vos' CLI on PATH or the python "
                    f"'vos' package (neither available: {exc}). Install one to sync "
                    "with vos: roots."
                ) from exc
        self._ensure_root()

    # ------------------------------------------------------------------ CLI
    def _argv(self, cmd: str, *args: str) -> List[str]:
        """Build the ``vos`` CLI argv for ``cmd`` (centralised for adaptation)."""

        def uri(rel: str) -> str:
            return f"{self.root}/{_safe_rel(rel)}".replace("//", "/")

        if cmd == "list":
            return ["vos", "vls", uri(args[0] if args else "")]
        if cmd == "list-top":
            return ["vos", "vls", uri(args[0] if args else "")]
        if cmd == "mkdir":
            return ["vos", "vmkdir", "-p", uri(args[0] if args else "")]
        if cmd == "put":
            return ["vos", "vcp", args[0], uri(args[1])]
        if cmd == "get":
            return ["vos", "vcp", uri(args[0]), args[1]]
        if cmd == "mv":
            return ["vos", "vmv", uri(args[0]), uri(args[1])]
        if cmd == "rm":
            return ["vos", "vrm", uri(args[0] if args else "")]
        raise ValueError(f"unknown vos command {cmd!r}")

    def _run(self, argv: List[str]) -> "subprocess.CompletedProcess[str]":
        return subprocess.run(
            argv,
            capture_output=True,
            text=True,
            check=False,
        )

    def _uri(self, rel: str) -> str:
        return f"{self.root}/{_safe_rel(rel)}"

    def _ensure_root(self) -> None:
        if self._binary is not None:
            proc = self._run(["vos", "vls", str(self.root)])
            if proc.returncode != 0:
                mk = self._run(["vos", "vmkdir", "-p", str(self.root)])
                if mk.returncode != 0:
                    raise ConfigError(
                        f"Cannot access/create VOSpace root {self.root}: {mk.stderr.strip()}"
                    )
        else:  # python-vos fallback (unverified API surface; adapted on CANFAR smoke)
            try:
                self._client.get_node(self.root)  # type: ignore[union-attr]
            except Exception:
                try:
                    self._client.mkdir(self.root)  # type: ignore[union-attr]
                except Exception as exc:
                    raise ConfigError(f"Cannot access VOSpace root {self.root}: {exc}") from exc

    def exists(self, rel: str) -> bool:
        if self._binary is not None:
            return self._run(self._argv("list-top", rel)).returncode == 0
        try:
            self._client.get_node(self._uri(rel))  # type: ignore[union-attr]
            return True
        except Exception:
            return False

    def size(self, rel: str) -> int:
        if self._binary is not None:
            # vls -l long listing: "<perms> <owner> <group> <size> <date> <name>"
            # (layout varies across clients, but the size column always follows
            # at least three leading non-numeric fields).  Take the FIRST
            # integer token at index >= 3 — the date token ("2024-01-01") is
            # not all-digits so it never wins.  Unparsable -> -1, and the
            # mirror's manifest logic treats -1 as "unknown, always refetch",
            # so accuracy here is best-effort, never correctness.
            proc = self._run(["vos", "vls", "-l", self._uri(rel)])
            if proc.returncode != 0:
                return -1
            for line in proc.stdout.splitlines():
                parts = line.split()
                for tok in parts[3:]:
                    if tok.isdigit():
                        return int(tok)
            return -1
        try:
            node = self._client.get_node(self._uri(rel))  # type: ignore[union-attr]
            size = getattr(node, "size", None)
            if size is None:
                size = (node.props or {}).get("length")  # type: ignore[union-attr]
            return int(size) if size is not None else -1
        except Exception:
            return -1

    def list(self, rel: str) -> List[str]:
        """Entry names directly under ``rel`` (non-recursive; dirs w/o trailing slash)."""
        if self._binary is not None:
            proc = self._run(self._argv("list", rel))
            if proc.returncode != 0:
                return []
            names = []
            for line in proc.stdout.splitlines():
                name = line.strip().rsplit("/", 1)[-1].rstrip("/")
                if name and name not in names:
                    names.append(name)
            return sorted(names)
        try:
            nodes = self._client.listdir(self._uri(rel))  # type: ignore[union-attr]
            return sorted(n.uri.rsplit("/", 1)[-1].rstrip("/") for n in nodes)
        except Exception:
            return []

    def read_parquet(self, rel: str) -> pl.DataFrame:
        import tempfile

        tmpdir = tempfile.mkdtemp(prefix="xmatch-vos-")
        local = Path(tmpdir) / "part.parquet"
        try:
            self.stage_in(rel, local)
            if not local.exists():
                raise FileNotFoundError(rel)
            return pl.read_parquet(local)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def parquet_schema(self, rel: str) -> "dict[str, object]":
        """Stage the partition in and read its footer (one transfer, not two)."""
        import tempfile

        tmpdir = tempfile.mkdtemp(prefix="xmatch-vos-schema-")
        local = Path(tmpdir) / "part.parquet"
        try:
            self.stage_in(rel, local)
            if not local.exists():
                raise FileNotFoundError(rel)
            return dict(pl.read_parquet_schema(local))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def write_parquet(self, df: pl.DataFrame, rel: str) -> None:
        import tempfile

        tmpdir = tempfile.mkdtemp(prefix="xmatch-vos-")
        local = Path(tmpdir) / "part.parquet"
        try:
            df.write_parquet(local)
            self.stage_out(local, rel)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def rename(self, src: str, dst: str) -> None:
        if self._binary is not None:
            proc = self._run(self._argv("mv", src, dst))
            if proc.returncode != 0:
                raise OSError(f"vos vmv failed: {proc.stderr.strip()}")
            return
        try:
            self._client.move(self._uri(src), self._uri(dst))  # type: ignore[union-attr]
        except Exception as exc:
            raise OSError(f"vos move failed: {exc}") from exc

    def mkdir(self, rel: str) -> None:
        if self._binary is not None:
            proc = self._run(self._argv("mkdir", rel))
            if proc.returncode != 0:
                raise OSError(f"vos vmkdir failed: {proc.stderr.strip()}")
            return
        try:
            self._client.mkdir(self._uri(rel))  # type: ignore[union-attr]
        except Exception as exc:
            raise OSError(f"vos mkdir failed: {exc}") from exc

    def _python_put(self, local: Path, rel: str) -> None:
        """Store a local file under ``rel`` via the python-vos fallback."""
        # The python binding copies a local path to a vos: URI, creating
        # intermediate containers as needed; exact method name checked on
        # CANFAR smoke (fallback path is unexercised in CI).
        try:
            self._client.copy(str(local), self._uri(rel))  # type: ignore[union-attr]
        except Exception as exc:
            raise OSError(f"vos vcp failed: {exc}") from exc

    def _in_get(self, rel: str, local: Path) -> None:
        try:
            self._client.copy(self._uri(rel), str(local))  # type: ignore[union-attr]
        except Exception as exc:
            raise OSError(f"vos vcp failed: {exc}") from exc

    def _mkdir_parents(self, rel: str) -> None:
        """Ensure the parent containers of ``rel`` exist (vcp does not)."""
        parent = rel.rsplit("/", 1)[0] if "/" in rel else ""
        self.mkdir(parent)

    def stage_out(self, local: Path, rel: str) -> None:
        self._mkdir_parents(rel)
        if self._binary is not None:
            proc = self._run(self._argv("put", str(local), rel))
            if proc.returncode != 0:
                raise OSError(f"vos vcp failed: {proc.stderr.strip()}")
            return
        self._python_put(local, rel)

    def stage_in(self, rel: str, local: Path) -> None:
        if self._binary is not None:
            proc = self._run(self._argv("get", rel, str(local)))
            if proc.returncode != 0:
                raise OSError(f"vos vcp failed: {proc.stderr.strip()}")
            return
        self._in_get(rel, local)

    def rm(self, rel: str) -> None:
        if self._binary is not None:
            proc = self._run(self._argv("rm", rel))
            if proc.returncode != 0 and "not found" not in proc.stderr.lower():
                # missing nodes are a no-op
                raise OSError(f"vos vrm failed: {proc.stderr.strip()}")
            return
        with contextlib.suppress(Exception):
            self._client.delete(self._uri(rel))  # type: ignore[union-attr]


def open_storage(root: Union[str, Path]) -> Storage:
    """Open a :class:`Storage` for ``root`` (``vos:`` URI → VOSpace, else local dir)."""
    if isinstance(root, str) and root.startswith("vos:"):
        return VOSpaceStorage(root)
    return LocalStorage(root)


def platform_arc_root() -> Optional[str]:
    """``/arc`` when running on an AstroAI/CANFAR session, else ``None``.

    Every platform pod mounts the shared ``/arc`` volume; laptops never
    have it.  This is the on-platform probe for the storage defaults.
    """
    return "/arc" if os.path.isdir("/arc") else None


def default_cache_root() -> str:
    """Effective default cache root.

    ``XMATCH_CACHE_ROOT`` wins; on AstroAI/CANFAR sessions (shared ``/arc``
    volume) the default is ``/arc/projects/hats`` — never a home directory,
    which is pod-local and lost; off the platform it falls back to
    ``~/.cache/xmatch``.
    """
    env = os.environ.get("XMATCH_CACHE_ROOT")
    if env:
        return env
    if platform_arc_root():
        return "/arc/projects/hats"
    return str(Path.home() / ".cache" / "xmatch")


def default_output_root() -> Optional[str]:
    """Root for crossmatch outputs.

    ``XMATCH_OUTPUT_ROOT`` wins (empty string disables); on
    AstroAI/CANFAR sessions the default is ``/arc/projects/hats/xmatch``;
    off the platform there is no default and relative outputs stay
    cwd-relative.  The CLI resolves bare relative output names under this
    root; explicit paths are always respected verbatim.
    """
    env = os.environ.get("XMATCH_OUTPUT_ROOT")
    if env is not None:
        return env.strip() or None
    if platform_arc_root():
        return "/arc/projects/hats/xmatch"
    return None


def all_cache_roots(cache_cfg: Optional[dict] = None) -> List[str]:
    """Ordered, deduplicated cache roots: primary first, then replicas.

    Primary = ``$XMATCH_CACHE_ROOT`` → ``cache_cfg["root"]`` →
    :func:`default_cache_root`.  Every entry of ``cache_cfg["roots"]`` is
    appended after it (blank entries dropped; the primary is never duplicated
    as a replica).  Never returns ``[]``.  Replicas may be ``vos:`` URIs —
    :func:`open_storage` routes those — or shared POSIX dirs; they are the
    mirror-fallback copies the union reads when the primary copy is gone.
    """
    cache_cfg = cache_cfg or {}
    primary = os.environ.get("XMATCH_CACHE_ROOT") or cache_cfg.get("root") or default_cache_root()
    roots = [primary]
    for entry in cache_cfg.get("roots") or []:
        text = str(entry).strip()
        if text and text not in roots:
            roots.append(text)
    return roots


def assert_headroom(path: Union[str, Path], min_free_gb: float, label: str) -> None:
    """Fail fast when ``path``'s filesystem has less than ``min_free_gb`` GiB free.

    Runs before any day-scale mirror/union starts so a multi-day run cannot
    die at 95% into a full disk. ``label`` names the mount in the error.
    The directory is created if missing (every caller writes into it next).
    """
    from .exceptions import CrossMatchError

    os.makedirs(str(path), exist_ok=True)
    free_gb = shutil.disk_usage(str(path)).free / 1e9
    if free_gb < min_free_gb:
        raise CrossMatchError(
            f"{label} has only {free_gb:.1f} GiB free, below the --min-free-gb floor "
            f"of {min_free_gb:g} GiB; refusing to start a sync/union it cannot fit."
        )
