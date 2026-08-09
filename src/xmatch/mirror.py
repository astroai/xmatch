"""Remote catalogue mirroring + incremental local cache (TAP and HATS inputs).

Backs the ``engine=ray-union`` pipeline: every remote input (a TAP table, or a
HATS catalogue served over HTTP / ``vos:``) is mirrored into a per-catalogue
cache dir under the cache root (``XMATCH_CACHE_ROOT``, else ``~/.cache/xmatch``),
stored through :class:`xmatch.storage.Storage` (POSIX dir or VOSpace ``vos:``
URI).  Re-syncs are incremental:

* **Remote HATS** — the manifest stores ``{rel: {size}}``; a file whose
  server-reported size matches is skipped (``--force`` / changed size
  refetches).  On HTTP the listing prefers ``partition_info.parquet``
  (authoritative file list + sizes) and falls back to an HTML directory walk
  (``http.server``-style listings) with per-file HEAD probes for sizes.
* **TAP full table** — keyset/``OFFSET`` pages (``--page-size``, default
  100 000).  Re-sync probes each stored page's key window with a cheap
  ``COUNT(*)`` and refetches only the pages whose window moved; the manifest
  is the durable record, and unchanged sources cost zero page downloads.
  A hard ceiling (1.5 x ``estimated_size``, default 500 000) guards against
  accidental full-table pulls of giant TAP tables — those must be ingested
  as HATS.

Rate limiting: one :class:`TokenBucket` per remote host, shared across worker
threads.  HTTP 429/503 → wait ``Retry-After`` (or 1 s) + exponential backoff
(cap 60 s, 3 retries). ``rate_limit_rps=0`` disables throttling.

The local HATS copy mirrors the *original* catalogue untouched: TAP pages are
kept under ``<cache>/<name>/raw/pages/`` (they double as the incremental
store), and the converted HATS catalogue lives under
``<cache>/<name>/<version>/``.
"""

from __future__ import annotations

import hashlib
import io as _io
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import polars as pl

from . import io_utils
from .exceptions import CrossMatchError
from .remote_tap import _quote_id, _select_columns, _table_ref
from .sources import CatalogueSource
from .storage import LocalStorage, Storage, open_storage

logger = logging.getLogger(__name__)

DEFAULT_PAGE_SIZE = 100_000
DEFAULT_ESTIMATED_SIZE = 500_000
_TAP_CEILING_FACTOR = 1.5
_MANIFEST_NAME = "sync.json"
_HTTP_RETRIES = 3
_HTTP_BACKOFF_CAP_S = 60.0


# --------------------------------------------------------------------------- #
# rate limiter
# --------------------------------------------------------------------------- #
class TokenBucket:
    """Per-host rate limiter (token bucket, thread-safe)."""

    def __init__(self, rps: float, burst: int = 10) -> None:
        if rps < 0:
            raise ValueError("rate limit cannot be negative")
        self.rps = float(rps)
        self.burst = max(1, int(burst))
        self._tokens = float(self.burst)
        self._updated = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until a token is available (no-op when ``rps == 0``)."""
        if self.rps <= 0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(self.burst, self._tokens + (now - self._updated) * self.rps)
                self._updated = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self.rps
            time.sleep(wait)


# --------------------------------------------------------------------------- #
# stats + plan
# --------------------------------------------------------------------------- #
@dataclass
class SyncStats:
    """Counters returned by :func:`sync_catalogue`."""

    bytes_downloaded: int = 0
    files_downloaded: int = 0
    files_skipped: int = 0
    failed: int = 0
    pages: int = 0
    converted: bool = False

    def merge(self, other: "SyncStats") -> None:
        self.bytes_downloaded += other.bytes_downloaded
        self.files_downloaded += other.files_downloaded
        self.files_skipped += other.files_skipped
        self.failed += other.failed
        self.pages += other.pages
        self.converted = self.converted or other.converted


def build_sync_plan(
    src: CatalogueSource,
    cache: Storage,
    *,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a cheap per-catalogue sync plan (no data movement).

    Remote HATS: the partition file list known on the server.  TAP: an
    estimated page count from ``estimated_size``.  Local sources: no work.
    """
    cfg = cfg or {}
    if _is_remote_hats(src):
        files = _remote_hats_listing(src)
        return {"type": "hats", "files": files, "host": _rate_host(src)}
    if src.access_method == "tap":
        est = _parse_estimated_size(cfg.get("estimated_size"))
        page_size = int(cfg.get("page_size", DEFAULT_PAGE_SIZE))
        return {
            "type": "tap",
            "pages": max(1, int(est // max(1, page_size))),
            "host": _rate_host(src),
            "ceiling_rows": int(_TAP_CEILING_FACTOR * est),
        }
    if src.is_local or src.access_method == "hats":
        return {"type": "local", "files": []}
    raise CrossMatchError(f"Cannot mirror access method '{src.access_method}'.")


# --------------------------------------------------------------------------- #
# source classification
# --------------------------------------------------------------------------- #
def _is_remote_hats(src: CatalogueSource) -> bool:
    ident = src.access_identifier or ""
    return src.access_method == "hats" and (
        ident.startswith("http://") or ident.startswith("https://") or ident.startswith("vos:")
    )


def _rate_host(src: CatalogueSource) -> Optional[str]:
    if src.access_method == "tap":
        return urllib.parse.urlparse(src.tap_url or "").netloc or None
    ident = src.access_identifier or ""
    if ident.startswith(("http://", "https://")):
        return urllib.parse.urlparse(ident).netloc or None
    return None


def _sha1_str(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name or "catalogue").strip("_")
    return cleaned or "catalogue"


def _version_dir(src: CatalogueSource) -> str:
    """Deterministic cache subdir for a remote source (stable across runs)."""
    if src.access_method == "tap":
        host = urllib.parse.urlparse(src.tap_url or "").netloc or "tap"
        table = (src.access_identifier or "table").replace("/", ".")
        return f"tap-{_safe_name(host)}-{_safe_name(table)}-{_sha1_str(f'{src.tap_url}/{src.access_identifier}', 8)}"
    ident = src.access_identifier or ""
    scheme = "vos" if ident.startswith("vos:") else "http"
    return f"hats-{scheme}-{_sha1_str(ident, 12)}"


def _parse_estimated_size(value: Any) -> int:
    """Catalogue ``estimated_size`` (int, numeric str, or None) -> int rows."""
    if isinstance(value, bool) or value is None:
        return DEFAULT_ESTIMATED_SIZE
    if isinstance(value, (int, float)):
        return int(value) if value > 0 else DEFAULT_ESTIMATED_SIZE
    try:
        parsed = float(str(value))
        return int(parsed) if parsed > 0 else DEFAULT_ESTIMATED_SIZE
    except ValueError:
        return DEFAULT_ESTIMATED_SIZE


# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #
def _http_request(
    url: str,
    bucket: TokenBucket,
    *,
    head: bool = False,
) -> Tuple[Optional[bytes], Optional[int]]:
    """GET (or HEAD) ``url`` under the token bucket with 429/503 backoff.

    Returns ``(body, content_length)`` (body ``None`` for HEAD).  Raises the
    underlying exception after retries are exhausted.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(_HTTP_RETRIES):
        bucket.acquire()
        try:
            req = urllib.request.Request(url, method="HEAD" if head else "GET")
            with urllib.request.urlopen(req, timeout=120) as resp:
                length = resp.headers.get("Content-Length")
                size = int(length) if length and length.isdigit() else None
                return (None, size) if head else (resp.read(), size)
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code in (429, 503):
                retry_after = exc.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after else 1.0
                except ValueError:
                    delay = 1.0
            else:
                raise
        except (urllib.error.URLError, OSError) as exc:
            last_exc = exc
            delay = 1.0
        if attempt < _HTTP_RETRIES - 1:
            time.sleep(min(delay * (2**attempt) + 0.1, _HTTP_BACKOFF_CAP_S))
    assert last_exc is not None
    raise last_exc


def _http_listing(url: str, bucket: TokenBucket) -> List[str]:
    """Entry names from an HTML directory listing (``[]`` when unavailable)."""
    try:
        body, _ = _http_request(url, bucket)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError):
        return []
    if not body:
        return []
    text = body.decode("utf-8", errors="replace")
    names: List[str] = []
    for m in re.finditer(r'href="([^"?#]+)"', text):
        name = m.group(1)
        if name in (".", ".."):
            continue
        names.append(urllib.parse.unquote(name))
    return names


def _http_head(url: str, bucket: TokenBucket) -> Optional[int]:
    try:
        _, size = _http_request(url, bucket, head=True)
        return size
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# remote HATS listing
# --------------------------------------------------------------------------- #
def _remote_hats_listing(
    src: CatalogueSource, bucket: Optional[TokenBucket] = None
) -> List[Dict[str, Any]]:
    """List ``{rel, size}`` pairs for every HATS file on the remote source.

    Prefers ``partition_info.parquet`` (authoritative list + sizes), then an
    HTML walk (``http.server`` listings), then a ``vos:`` Storage walk.
    """
    ident = src.access_identifier or ""
    if ident.startswith("vos:"):
        return _vos_hats_listing(ident)
    if bucket is None:
        bucket = TokenBucket(1.0)
    root = ident.rstrip("/")
    for info_rel in ("dataset/partition_info.parquet", "partition_info.parquet"):
        try:
            body, _ = _http_request(f"{root}/{info_rel}", bucket)
        except (urllib.error.HTTPError, urllib.error.URLError, OSError):
            continue
        if not body:
            continue
        try:
            info = pl.read_parquet(_io.BytesIO(body))
            if "file_loc" not in info.columns:
                continue
            base = "dataset" if info_rel.startswith("dataset/") else ""
            files: List[Dict[str, Any]] = []
            for row in info.rows(named=True):
                loc = str(row.get("file_loc", "")).strip()
                if not loc:
                    continue
                rel = f"{base}/{loc}" if base else loc
                rel = rel.replace("//", "/")
                if rel.endswith(".parquet"):
                    files.append({"rel": rel, "size": _as_size(row.get("file_size"))})
                    continue
                # hats `file_loc` may omit the .parquet suffix (single-file
                # pixel dirs); resolve the real path by probing both shapes.
                for cand in (rel, f"{rel}.parquet"):
                    size = _http_head(f"{root}/{cand}", bucket)
                    if size is not None:
                        files.append({"rel": cand, "size": _as_size(row.get("file_size"))})
                        break
            for extra in ("properties", "dataset/properties", "hats.properties"):
                size = _http_head(f"{root}/{extra}", bucket)
                if size is not None:
                    files.append({"rel": extra, "size": size})
            return files
        except (ImportError, OSError, ValueError):
            continue
    return _http_walk_hats(root, bucket)


def _as_size(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _http_walk_hats(root: str, bucket: TokenBucket) -> List[Dict[str, Any]]:
    """Walk HTML listings under ``root``/``dataset`` collecting partition files."""
    out: List[Dict[str, Any]] = []
    for base in ("", "dataset"):
        url = f"{root}/{base}" if base else root
        for name in _http_listing(url, bucket):
            rel = f"{base}/{name}".strip("/")
            if name.startswith("Norder="):
                out.extend(_walk_order_dir(f"{url}/{name}", bucket, rel))
            elif name.endswith(".parquet") or name in ("properties", "hats.properties"):
                size = _http_head(f"{url}/{name}", bucket)
                out.append({"rel": rel, "size": size})
    return out


def _walk_order_dir(url: str, bucket: TokenBucket, prefix: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for name in _http_listing(url, bucket):
        if name.startswith("Dir="):
            out.extend(_walk_dir(url + "/" + name, bucket, f"{prefix}/{name}"))
    return out


def _walk_dir(url: str, bucket: TokenBucket, prefix: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for name in _http_listing(url, bucket):
        rel = f"{prefix}/{name}" if prefix else name
        if name.lower().endswith(".parquet"):
            size = _http_head(url + "/" + name, bucket)
            out.append({"rel": rel, "size": size})
        elif name.endswith("/"):  # multi-file pixel dir (Npix=NN/…)
            for f in _http_listing(url + "/" + name, bucket):
                if f.lower().endswith(".parquet"):
                    size = _http_head(url + "/" + name + "/" + f, bucket)
                    out.append({"rel": rel + "/" + f, "size": size})
    return out


def _vos_hats_listing(ident: str) -> List[Dict[str, Any]]:
    storage = open_storage(ident)
    rels: List[str] = []
    _walk_storage(storage, "", rels)
    out: List[Dict[str, Any]] = []
    for rel in rels:
        if rel.endswith("/"):
            continue
        out.append({"rel": rel, "size": storage.size(rel)})
    return out


def _walk_storage(storage: Storage, rel: str, acc: List[str], depth: int = 0) -> None:
    """Collect every node under ``rel``: files as paths, containers as dirs.

    ``storage.list`` entries have no type marker (some vos clients append
    ``/`` to directories, LocalStorage does not), so anything that is not a
    data file is treated as a container and recursed into — including wrapper
    dirs like ``dataset/``.  Two guards keep the walk honest against real
    ``vls`` behaviour: a leaf file echoes its own basename when listed, so a
    listing that returns only the current node's name means "not a
    directory"; and a depth cap bounds pathological nests.
    """
    if depth > 12:
        return
    try:
        children = storage.list(rel)
    except ValueError:  # traversal-shaped rel from a hostile listing
        return
    if not children:
        return
    if depth and children == [rel.rsplit("/", 1)[-1]]:
        return  # vls on a leaf file echoes the file's own basename
    for name in children:
        child = f"{rel}/{name}".strip("/")
        if name.endswith(".parquet") or name in ("properties", "hats.properties"):
            acc.append(child)
        else:
            acc.append(child + "/")
            _walk_storage(storage, child, acc, depth + 1)


# --------------------------------------------------------------------------- #
# manifest helpers
# --------------------------------------------------------------------------- #
def _read_json(cache: Storage, rel: str) -> Dict[str, Any]:
    try:
        if not cache.exists(rel):
            return {}
        if isinstance(cache, LocalStorage):
            return json.loads(Path(cache.root / rel).read_text())
        tmpdir = Path(tempfile.mkdtemp(prefix="xmatch-mf-"))
        try:
            local = tmpdir / "manifest.json"
            cache.stage_in(rel, local)
            return json.loads(local.read_text())
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def _write_json(cache: Storage, rel: str, obj: Dict[str, Any]) -> None:
    tmpdir = Path(tempfile.mkdtemp(prefix="xmatch-mf-"))
    local = tmpdir / "manifest.json"
    try:
        local.write_text(json.dumps(obj, indent=2, sort_keys=True))
        if isinstance(cache, LocalStorage):
            target = Path(cache.root) / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.parent / f".{target.name}.{os.getpid()}.tmp"
            try:
                shutil.copy2(local, tmp)
                tmp.replace(target)
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise
        else:
            cache.stage_out(local, rel)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _copy_tree_up(cache: Storage, local_dir: Path, rel: str) -> None:
    """Copy a local directory tree into the cache root at ``rel``.

    The local tree is staged under ``rel + .tmp`` first; the live ``rel`` is
    then moved aside to ``rel + .old`` and the tmp tree renamed into place,
    so a crash never leaves the catalogue *absent* (the previous version is
    recoverable from ``.old`` and the next run self-heals via the properties
    check).
    """
    files = [p for p in sorted(local_dir.rglob("*")) if p.is_file()]
    if isinstance(cache, LocalStorage):
        target = Path(cache.root) / rel
        tmp = Path(cache.root) / (rel.rstrip("/") + ".tmp")
        backup = Path(cache.root) / (rel.rstrip("/") + ".old")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)
        for p in files:
            dest = tmp / p.relative_to(local_dir)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(backup, ignore_errors=True)
        if target.exists():
            os.replace(target, backup)
        try:
            os.replace(tmp, target)
        except BaseException:
            if backup.exists():
                os.replace(backup, target)  # restore the previous version
            raise
        shutil.rmtree(backup, ignore_errors=True)
    else:
        for p in files:
            dest_rel = f"{rel}/{p.relative_to(local_dir)}"
            cache.stage_out(p, dest_rel)


# --------------------------------------------------------------------------- #
# HATS mirroring
# --------------------------------------------------------------------------- #
def _mirror_remote_hats(
    src: CatalogueSource,
    cache: Storage,
    *,
    bucket: TokenBucket,
    force: bool,
    workers: int,
    progress_cb: Optional[Callable[[str], None]],
    stats: SyncStats,
) -> None:
    """Mirror every remote HATS partition file into ``cache/<name>/<version>``."""
    version = _version_dir(src)
    prefix = f"{_safe_name(src.name)}/{version}"
    remote = _remote_hats_listing(src, bucket=bucket)
    manifest_rel = f"{prefix}/{_MANIFEST_NAME}"
    manifest = _read_json(cache, manifest_rel)

    def do_fetch(entry: Dict[str, Any]) -> None:
        rel = entry["rel"]
        tmpdir = Path(tempfile.mkdtemp(prefix="xmatch-"))
        local_tmp = tmpdir / Path(rel).name
        try:
            if _remote_via_storage(src):
                # vos: nodes have no HTTP endpoint; transfer node → temp → cache.
                storage = open_storage((src.access_identifier or "").rstrip("/"))
                storage.stage_in(rel, local_tmp)
            else:
                body, _ = _http_request(_remote_url(src, rel), bucket)
                if body is None:
                    raise CrossMatchError(f"empty body for {rel}")
                local_tmp.write_bytes(body)
            cache.stage_out(local_tmp, f"{prefix}/{rel}")
            # Record the size actually stored (local stat, or vls -l on vos:);
            # an unknown remote size (-1) must never poison the manifest into
            # "unchanged" or the incremental probe goes permanently stale.
            stored = cache.size(f"{prefix}/{rel}")
            stats.files_downloaded += 1
            stats.bytes_downloaded += max(0, stored) if stored >= 0 else 0
            manifest.setdefault("files", {})[rel] = {"size": stored}
        except Exception as exc:  # noqa: BLE001
            stats.failed += 1
            logger.warning("sync %s: failed to fetch %s: %s", src.name, rel, exc)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    todo: List[Dict[str, Any]] = []
    for entry in remote:
        rel = entry["rel"]
        old = manifest.get("files", {}).get(rel)
        old_size = (old or {}).get("size")
        new_size = entry.get("size")
        if (
            not force
            and old is not None
            and old_size is not None
            and old_size >= 0
            and old_size == new_size
        ):
            stats.files_skipped += 1
        else:
            todo.append(entry)
    if progress_cb:
        progress_cb(f"sync {src.name}: {len(todo)} file(s) to fetch")
    if workers > 1 and todo:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(do_fetch, todo))
    else:
        for entry in todo:
            do_fetch(entry)
    manifest["version"] = version
    manifest["source"] = src.access_identifier
    manifest["fetched_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_json(cache, manifest_rel, manifest)
    logger.info(
        "sync %s: %d bytes, %d files (%d skipped, %d failed)",
        src.name,
        stats.bytes_downloaded,
        stats.files_downloaded,
        stats.files_skipped,
        stats.failed,
    )


def _remote_via_storage(src: CatalogueSource) -> bool:
    """True when the remote HATS source is fetched via a ``Storage`` (vos:)."""
    return (src.access_identifier or "").startswith("vos:")


def _remote_url(src: CatalogueSource, rel: str) -> str:
    ident = (src.access_identifier or "").rstrip("/")
    if _remote_via_storage(src):
        raise CrossMatchError("vos: HATS not supported over HTTP fetch")
    return f"{ident}/{rel}"


# --------------------------------------------------------------------------- #
# TAP mirroring
# --------------------------------------------------------------------------- #
def _tap_service(src: CatalogueSource, auth_session: Any):
    from .tap import get_tap_service

    return get_tap_service(src.tap_url, auth_session=auth_session)


def _tap_run(
    src: CatalogueSource, service, query: str, maxrec: Optional[int] = None
) -> pl.DataFrame:
    from .io_utils import astropy_table_to_polars
    from .tap import execute_tap_query

    return astropy_table_to_polars(execute_tap_query(service, query, maxrec=maxrec))


def _tap_page_query(
    src: CatalogueSource,
    *,
    offset: Optional[int] = None,
    window: Optional[Tuple[Any, Any]] = None,
    after_key: Any = None,
    key: Optional[str] = None,
    limit: Optional[int] = None,
) -> str:
    """Build ``SELECT cols FROM tbl AS t [WHERE …] ORDER BY key [LIMIT]``."""
    table_q = _table_ref(src.access_identifier or "", tap_url=src.tap_url or "")
    select = _select_columns(src, None)
    key = key or src.id_column or src.ra_column
    q = f"SELECT {select} FROM {table_q} AS t"
    if window is not None and key:
        lo, hi = window
        q += f" WHERE t.{_quote_id(key)} >= {_tap_literal(lo)} AND t.{_quote_id(key)} <= {_tap_literal(hi)}"
    elif after_key is not None and key:
        q += f" WHERE t.{_quote_id(key)} > {_tap_literal(after_key)}"
    if key:
        q += f" ORDER BY t.{_quote_id(key)}"
    if offset is not None:
        q += f" LIMIT {limit or DEFAULT_PAGE_SIZE} OFFSET {offset}"
    elif limit is not None:
        q += f" LIMIT {limit}"
    return q


def _tap_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def _tap_key_query(src: CatalogueSource, *, max_key: bool = False) -> str:
    key = src.id_column or src.ra_column
    table_q = _table_ref(src.access_identifier or "", tap_url=src.tap_url or "")
    direction = "DESC" if max_key else "ASC"
    return f"SELECT t.{_quote_id(key)} FROM {table_q} AS t ORDER BY t.{_quote_id(key)} {direction} LIMIT 1"


def _tap_count_query(src: CatalogueSource, window: Optional[Tuple[Any, Any]] = None) -> str:
    key = src.id_column or src.ra_column
    table_q = _table_ref(src.access_identifier or "", tap_url=src.tap_url or "")
    if window is not None and key:
        lo, hi = window
        where = f" WHERE t.{_quote_id(key)} >= {_tap_literal(lo)} AND t.{_quote_id(key)} <= {_tap_literal(hi)}"
    else:
        where = ""
    return f"SELECT COUNT(*) AS n FROM {table_q} AS t{where}"


def _file_sha256(cache: Storage, rel: str) -> str:
    """SHA-256 of the parquet bytes stored at ``rel`` (manifest signature)."""
    if isinstance(cache, LocalStorage):
        data = Path(cache.root / rel).read_bytes()
        return hashlib.sha256(data).hexdigest()
    tmpdir = Path(tempfile.mkdtemp(prefix="xmatch-hash-"))
    try:
        local = tmpdir / "page.parquet"
        cache.stage_in(rel, local)
        return hashlib.sha256(local.read_bytes()).hexdigest()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _mirror_tap(
    src: CatalogueSource,
    cache: Storage,
    *,
    bucket: TokenBucket,
    force: bool,
    page_size: int,
    hats_threshold: int,
    estimated_size: int,
    auth_session: Any,
    progress_cb: Optional[Callable[[str], None]],
    stats: SyncStats,
) -> None:
    """Incrementally mirror a TAP table into ``cache/<name>/tap-<version>``.

    Cold sync fetches sequential OFFSET pages; re-sync probes each stored
    page's key window with ``COUNT(*)`` and refetches only moved windows
    (append-only sources additionally fetch the new tail).  The page parquet
    files under ``<cache>/<name>/raw/pages/`` are the incremental store; the
    HATS catalogue under ``<cache>/<name>/<version>/`` is rebuilt whenever a
    page changed.
    """
    name = _safe_name(src.name)
    version = _version_dir(src)
    prefix = f"{name}/{version}"
    raw = f"{name}/raw"
    manifest_rel = f"{raw}/{_MANIFEST_NAME}"
    manifest = _read_json(cache, manifest_rel)
    key = src.id_column or src.ra_column
    ceiling = int(_TAP_CEILING_FACTOR * estimated_size)
    service = _tap_service(src, auth_session)

    def fetch_page(query: str, idx: int) -> pl.DataFrame:
        bucket.acquire()
        df = _tap_run(src, service, query, maxrec=page_size)
        if df.height:
            page_rel = f"{raw}/pages/page_{idx:04d}.parquet"
            cache.write_parquet(df, page_rel)
            stats.pages += 1
            stats.files_downloaded += 1
            stats.bytes_downloaded += cache.size(page_rel)
        return df

    def record_page(idx: int, df: pl.DataFrame) -> None:
        if not df.height:
            return
        page_rel = f"{raw}/pages/page_{idx:04d}.parquet"
        entry: Dict[str, Any] = {"rows": df.height, "sha256": _file_sha256(cache, page_rel)}
        if key:
            keys = df[key].to_list()
            entry["first_key"] = keys[0]
            entry["last_key"] = keys[-1]
        manifest.setdefault("pages", {})[str(idx)] = entry

    pages_manifest = manifest.get("pages")
    if not pages_manifest or force:
        # ---------------- cold (or forced) fetch: OFFSET pages ------------
        if progress_cb:
            progress_cb(f"sync {src.name}: fetching full table")
        offset = 0
        while True:
            df = fetch_page(
                _tap_page_query(src, offset=offset, limit=page_size), offset // page_size
            )
            if df.is_empty():
                break
            record_page(offset // page_size, df)
            offset += page_size
            if offset >= ceiling:
                raise CrossMatchError(
                    f"TAP table '{src.name}' exceeds the sync ceiling of {ceiling} rows "
                    f"(estimated_size {estimated_size} x {_TAP_CEILING_FACTOR}). "
                    "Ingest this catalogue as HATS instead."
                )
            if df.height < page_size:
                break
        changed = True
    else:
        # ---------------- incremental: probe stored windows -------------
        changed = False
        pages = pages_manifest
        if key:
            # tail probe: did the highest key move (append-only sources)?
            tail_df = _tap_run(src, service, _tap_key_query(src, max_key=True), maxrec=1)
            new_last = tail_df[key][0] if tail_df.height else None
            last_idx = str(max(int(i) for i in pages))
            last_key = pages[last_idx].get("last_key")
            tail_changed = last_key is None or new_last != last_key

            for idx_str in sorted(pages, key=int):
                entry = pages[idx_str]
                window = (entry.get("first_key"), entry.get("last_key"))
                if window[0] is None or window[1] is None:
                    continue
                probe = _tap_run(src, service, _tap_count_query(src, window=window), maxrec=1)
                n = probe["n"][0] if probe.height else 0
                if int(n) != int(entry["rows"]):
                    if progress_cb:
                        progress_cb(f"sync {src.name}: page {idx_str} changed")
                    df = fetch_page(
                        _tap_page_query(src, window=window, key=key, limit=page_size),
                        int(idx_str),
                    )
                    if df.height:
                        record_page(int(idx_str), df)
                    else:
                        # window vanished: drop the stored page + registry entry
                        cache.rm(f"{raw}/pages/page_{int(idx_str):04d}.parquet")
                        pages.pop(idx_str, None)
                    changed = True

            if tail_changed:
                # appended rows: keyset continuation past the stored last key
                cont_idx = int(last_idx) + 1
                lo = last_key
                while True:
                    df = fetch_page(
                        _tap_page_query(src, after_key=lo, key=key, limit=page_size),
                        cont_idx,
                    )
                    if df.is_empty():
                        break
                    record_page(cont_idx, df)
                    changed = True
                    cont_idx += 1
                    lo = df[key][-1]
                    if _manifest_total_rows(manifest) > ceiling:
                        raise CrossMatchError(
                            f"TAP table '{src.name}' exceeds the sync ceiling of {ceiling} rows."
                        )
            if not changed and progress_cb:
                progress_cb(f"sync {src.name}: table unchanged, nothing to fetch")
        else:
            probe = _tap_run(src, service, _tap_count_query(src), maxrec=1)
            n = probe["n"][0] if probe.height else 0
            if int(n) != int(manifest.get("total_rows", 0)):
                raise CrossMatchError(
                    f"TAP table '{src.name}' has no ordering key; row count changed from "
                    f"{manifest.get('total_rows')} to {n}. Re-run with --force to re-sync."
                )

    if changed or force or not cache.exists(f"{prefix}/properties"):
        _rebuild_tap_hats(cache, src, raw, prefix, hats_threshold=hats_threshold, stats=stats)
    manifest["version"] = version
    manifest["source"] = src.access_identifier
    manifest["page_size"] = page_size
    manifest["ordering"] = key
    manifest["total_rows"] = _manifest_total_rows(manifest)
    manifest["fetched_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_json(cache, manifest_rel, manifest)
    logger.info(
        "sync %s: %d bytes, %d files (%d skipped, %d failed, %d pages)",
        src.name,
        stats.bytes_downloaded,
        stats.files_downloaded,
        stats.files_skipped,
        stats.failed,
        stats.pages,
    )


def _manifest_total_rows(manifest: Dict[str, Any]) -> int:
    """Rows implied by the page registry (single source of truth)."""
    total = 0
    for entry in (manifest.get("pages") or {}).values():
        total += int(entry.get("rows", 0))
    return total


def _write_hats_native(
    frame: pl.DataFrame,
    out_dir: Path,
    *,
    ra_column: str,
    dec_column: str,
    threshold: int = 100_000,
    max_order: int = 10,
) -> None:
    """Convert ``frame`` into a valid HATS catalogue at ``out_dir`` (no LSDB).

    Adaptive NESTED tiling: any cell holding more than ``threshold`` rows is
    split until ``max_order``.  Layout matches the `hats` reader contract:
    ``dataset/Norder=…/Dir=…/Npix=….parquet`` pixel files, a root
    ``partition_info.csv`` (plus legacy parquet), and ``properties``.

    ponytail: cdshealpix computes every row's pixel per level (O(N·orders));
    fine for mirror-sized frames, replace with an index scan when a frame
    exceeds ~50M rows.
    """
    import cdshealpix  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    from astropy.coordinates import Latitude, Longitude  # noqa: PLC0415

    out_dir = Path(out_dir)
    dataset = out_dir / "dataset"
    dataset.mkdir(parents=True, exist_ok=True)

    segments: List[Tuple[int, int, pl.DataFrame]] = []
    if not frame.is_empty():
        stack: List[Tuple[int, int, pl.DataFrame]] = [(0, 0, frame)]
        while stack:
            order, pix, sub = stack.pop()
            if sub.height <= threshold or order >= max_order:
                if sub.height:
                    segments.append((order, pix, sub))
                continue
            lon = Longitude(sub[ra_column].cast(pl.Float64).to_numpy(), unit="deg")
            lat = Latitude(sub[dec_column].cast(pl.Float64).to_numpy(), unit="deg")
            child = cdshealpix.nested.lonlat_to_healpix(
                lon, lat, np.full(sub.height, order + 1, dtype=np.uint64)
            )
            for cpix in np.unique(child):
                mask = child == cpix
                stack.append(
                    (order + 1, int(cpix), sub.filter(pl.Series("_mask", mask, dtype=pl.Boolean)))
                )

    info: List[Dict[str, Any]] = []
    for order, pix, sub in segments:
        dirtree = (pix // 10000) * 10000
        rel = f"Norder={order}/Dir={dirtree}/Npix={pix}.parquet"
        dest = dataset / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        sub.write_parquet(dest)
        info.append(
            {
                "Norder": int(order),
                "Dir": int(dirtree),
                "Npix": int(pix),
                "Nfiles": 1,
                "file_loc": rel.removesuffix(".parquet"),
                "file_size": dest.stat().st_size,
                "count": sub.height,
            }
        )

    total = frame.height
    info_df = pl.DataFrame(info).select(
        ["Norder", "Dir", "Npix", "Nfiles", "file_loc", "file_size", "count"]
    )
    # hats 0.7.x reads the root copy; the dataset copy is the classic spec.
    info_df.write_csv(dataset / "partition_info.csv")
    info_df.write_csv(out_dir / "partition_info.csv")
    info_df.write_parquet(dataset / "partition_info.parquet")

    props = {
        "obs_collection": out_dir.name or "xmatch",
        "dataproduct_type": "object",
        "hats_nrows": str(total),
        "hats_col_ra": ra_column,
        "hats_col_dec": dec_column,
        "hats_epoch": "J2000",
        "hats_ordering": "NESTED",
        "hats_max_depth": str(max(segments, default=(0, 0, None))[0]),
    }
    lines = "".join(f"{k}={v}\n" for k, v in sorted(props.items()))
    (out_dir / "properties").write_text(lines)
    (dataset / "properties").write_text(lines)
    try:  # schema checkpoint for `hats.read_hats` (silences its warning)
        import pyarrow as pa  # noqa: PLC0415
        import pyarrow.parquet as pq  # noqa: PLC0415

        schemas = [pq.read_schema(dataset / f"{e['file_loc']}.parquet") for e in info]
        unified = pa.unify_schemas(schemas)
        pq.write_metadata(unified, dataset / "_common_metadata")
        pq.write_metadata(unified, dataset / "_metadata")
    except Exception:  # noqa: BLE001
        logger.warning("skipping _metadata for %s (heterogeneous schemas)", out_dir)


def _rebuild_tap_hats(
    cache: Storage,
    src: CatalogueSource,
    raw: str,
    prefix: str,
    *,
    hats_threshold: int,
    stats: SyncStats,
) -> None:
    """Re-convert the page store into the ``cache/<name>/<version>`` HATS cat."""
    frames = []
    for name in sorted(cache.list(f"{raw}/pages"), key=lambda n: n):
        try:
            frame = cache.read_parquet(f"{raw}/pages/{name}")
        except Exception as exc:  # noqa: BLE001
            # An unreadable page (torn write, corrupted transfer) must not
            # silently shrink the converted catalogue while the manifest
            # still counts its rows: surface it and count the failure.
            stats.failed += 1
            logger.warning("sync %s: ignoring unreadable TAP page %s: %s", src.name, name, exc)
            continue
        if frame.height:
            frames.append(frame)
    if not frames:
        cache.mkdir(prefix)
        return
    frame = pl.concat(frames, how="diagonal_relaxed")
    tmpdir = Path(tempfile.mkdtemp(prefix="xmatch-hats-"))
    try:
        _write_hats_native(
            frame,
            tmpdir / "cat",
            ra_column=src.ra_column or "ra",
            dec_column=src.dec_column or "dec",
            threshold=hats_threshold,
        )
        _copy_tree_up(cache, tmpdir / "cat", prefix)
        stats.converted = True
    finally:
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def sync_catalogue(
    src: CatalogueSource,
    *,
    cache_root: Optional[str] = None,
    rate_limit_rps: float = 1.0,
    workers: int = 8,
    force: bool = False,
    progress_cb: Optional[Callable[[str], None]] = None,
    auth_session: Any = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    hats_threshold: int = 100_000,
    estimated_size: Optional[int] = None,
) -> SyncStats:
    """Mirror ``src`` into the cache, incrementally. Returns :class:`SyncStats`."""
    from .storage import default_cache_root

    root = cache_root or default_cache_root()
    cache = open_storage(root)
    stats = SyncStats()
    if src.access_method == "hats" and not _is_remote_hats(src):
        return stats  # already a local HATS catalogue
    if src.access_method not in ("hats", "tap"):
        raise CrossMatchError(
            f"Cannot sync access method '{src.access_method}'; only TAP and HATS are mirrorable."
        )
    bucket = TokenBucket(rate_limit_rps) if rate_limit_rps else TokenBucket(0.0)
    if src.access_method == "hats":
        _mirror_remote_hats(
            src,
            cache,
            bucket=bucket,
            force=force,
            workers=workers,
            progress_cb=progress_cb,
            stats=stats,
        )
    else:
        _mirror_tap(
            src,
            cache,
            bucket=bucket,
            force=force,
            page_size=page_size,
            hats_threshold=hats_threshold,
            estimated_size=estimated_size or DEFAULT_ESTIMATED_SIZE,
            auth_session=auth_session,
            progress_cb=progress_cb,
            stats=stats,
        )
    return stats


def locate_mirrored(src: CatalogueSource, cache_root: Optional[str] = None) -> Tuple[str, str]:
    """Return ``(storage_root, rel)`` of the mirrored HATS copy of ``src``.

    Raises :class:`CrossMatchError` when no mirror exists (run ``xmatch sync``
    or drop ``--no-sync``).
    """
    from .storage import default_cache_root

    root = cache_root or default_cache_root()
    if src.access_method == "hats" and not _is_remote_hats(src):
        path = src.path
        if path is None and src.access_identifier:
            path = Path(src.access_identifier)
        if path is None or not Path(path).is_dir():
            raise CrossMatchError(f"No local HATS copy for '{src.name}'; run 'xmatch sync' first.")
        return str(path), ""
    if src.access_method not in ("hats", "tap"):
        raise CrossMatchError(
            f"Cannot mirror access method '{src.access_method}' for '{src.name}'; "
            "ray-union supports local HATS, remote HATS (http/vos:), and TAP inputs."
        )
    version = _version_dir(src)
    rel = f"{_safe_name(src.name)}/{version}"
    cache = open_storage(root)
    if not cache.exists(f"{rel}/properties") and not cache.exists(
        f"{rel}/dataset/partition_info.parquet"
    ):
        raise CrossMatchError(
            f"Catalogue '{src.name}' is not mirrored yet; run 'xmatch sync {src.name}' "
            "or drop --no-sync."
        )
    return root, rel


def ensure_mirrored(
    src: CatalogueSource,
    *,
    cache_root: Optional[str] = None,
    rate_limit_rps: float = 1.0,
    workers: int = 8,
    force: bool = False,
    progress_cb: Optional[Callable[[str], None]] = None,
    auth_session: Any = None,
    hats_threshold: int = 100_000,
    estimated_size: Optional[int] = None,
) -> CatalogueSource:
    """Mirror a remote source and return a source pointing at the local HATS copy.

    Local HATS catalogues are returned unchanged; other local inputs (plain
    parquet/csv/fits) are converted once into the cache as HATS.
    """
    from .storage import default_cache_root

    root = cache_root or default_cache_root()
    if src.access_method == "hats" and not _is_remote_hats(src):
        return src
    if src.is_local:
        if src.path is None:
            raise CrossMatchError(
                f"ray-union needs file-backed local inputs; '{src.name}' is an in-memory frame."
            )
        rel = f"{_safe_name(src.name)}/local-{_sha1_str(str(src.path), 8)}"
        cache = open_storage(root)
        if force or not cache.exists(f"{rel}/properties"):
            frame = io_utils.scan_frame(src.path).collect()
            tmpdir = Path(tempfile.mkdtemp(prefix="xmatch-hats-"))
            try:
                _write_hats_native(
                    frame,
                    tmpdir / "cat",
                    ra_column=src.ra_column or "ra",
                    dec_column=src.dec_column or "dec",
                    threshold=hats_threshold,
                )
                _copy_tree_up(cache, tmpdir / "cat", rel)
            finally:
                import shutil

                shutil.rmtree(tmpdir, ignore_errors=True)
            logger.info(
                "converted local input '%s' to HATS at %s (threshold=%d)",
                src.name,
                rel,
                hats_threshold,
            )
        return _mirrored_source(src, cache, rel, root)
    stats = sync_catalogue(
        src,
        cache_root=root,
        rate_limit_rps=rate_limit_rps,
        workers=workers,
        force=force,
        progress_cb=progress_cb,
        auth_session=auth_session,
        hats_threshold=hats_threshold,
        estimated_size=estimated_size,
    )
    if stats.failed:
        raise CrossMatchError(
            f"sync {src.name}: {stats.failed} file(s) failed to mirror; refusing to "
            "match over an incomplete catalogue (re-run to retry, or use "
            "'xmatch sync --force' to pin down the failures)."
        )
    rel = f"{_safe_name(src.name)}/{_version_dir(src)}"
    return _mirrored_source(src, open_storage(root), rel, root)


def _mirrored_source(src: CatalogueSource, cache: Storage, rel: str, root: str) -> CatalogueSource:
    """Build the local-HATS ``CatalogueSource`` for a mirrored copy at ``rel``."""
    path = Path(cache.root) / rel if isinstance(cache, LocalStorage) else None
    return CatalogueSource(
        name=src.name,
        is_local=False,
        access_method="hats",
        access_identifier=str(path) if path is not None else f"{root.rstrip('/')}/{rel}",
        path=path,
        hats_cache_rel=rel,
        ra_column=src.ra_column,
        dec_column=src.dec_column,
        id_column=src.id_column,
        epoch=src.epoch,
        epoch_column=src.epoch_column,
        pm_ra_column=src.pm_ra_column,
        pm_dec_column=src.pm_dec_column,
        parallax_column=src.parallax_column,
        radial_velocity_column=src.radial_velocity_column,
        astrometric_covariance_columns=src.astrometric_covariance_columns,
    )
