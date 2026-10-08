"""Remote catalogue mirroring + incremental local cache (TAP and HATS inputs).

Backs the ``engine=ray-union`` pipeline: every remote input (a TAP table, or a
HATS catalogue served over HTTP / ``vos:``) is mirrored into a per-catalogue
cache dir under the cache root (``XMATCHER_CACHE_ROOT``; on AstroAI/CANFAR
sessions ``/arc/projects/hats``, else ``~/.cache/xmatcher``), stored through
:class:`xmatcher.storage.Storage` (POSIX dir or VOSpace ``vos:`` URI).  Re-syncs
are incremental:

* **Remote HATS** — the manifest stores ``{rel: {size}}``; a file whose
  server-reported size matches is skipped (``--force`` / changed size
  refetches).  On HTTP the listing prefers ``partition_info.parquet``
  (authoritative file list + sizes) and falls back to an HTML directory walk
  (``http.server``-style listings) with per-file HEAD probes for sizes.
* **TAP full table** — keyset/``OFFSET`` pages (``--page-size``, default
  100 000).  Incremental key windows require a stable, unique, non-null
  ordering key. Re-sync uses ``COUNT(*)`` probes, which detect row-count
  changes but cannot detect same-count value edits; use ``--force`` for mutable
  tables or pin an immutable upstream release. Appended higher keys are fetched
  from the tail.
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
import math
import os
import re
import secrets
import shutil
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import polars as pl

from . import io_utils
from .exceptions import CrossMatchError, TapError
from .remote_tap import _quote_id, _select_columns, _table_ref
from .sources import CatalogueSource
from .storage import LocalStorage, Storage, open_storage

logger = logging.getLogger(__name__)

DEFAULT_PAGE_SIZE = 100_000
DEFAULT_ESTIMATED_SIZE = 500_000
_TAP_CEILING_FACTOR = 1.5
_MANIFEST_NAME = "sync.json"
_HATS_METADATA_NAMES = frozenset(
    {
        "properties",
        "hats.properties",
        "partition_info.csv",
        "partition_info.parquet",
        "_common_metadata",
        "_metadata",
    }
)
_HTTP_RETRIES = 3
_HTTP_BACKOFF_CAP_S = 60.0
_DEFAULT_MIN_FREE_GB = 10.0


def _min_free_gb(explicit: float | None) -> float:
    """Resolve the free-space floor: explicit > env > 10 GiB default."""
    if explicit is not None:
        return float(explicit)
    env = os.environ.get("XMATCHER_MIN_FREE_GB")
    return float(env) if env else _DEFAULT_MIN_FREE_GB


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

    def merge(self, other: SyncStats) -> None:
        self.bytes_downloaded += other.bytes_downloaded
        self.files_downloaded += other.files_downloaded
        self.files_skipped += other.files_skipped
        self.failed += other.failed
        self.pages += other.pages
        self.converted = self.converted or other.converted


# --------------------------------------------------------------------------- #
# source classification
# --------------------------------------------------------------------------- #
def _is_remote_hats(src: CatalogueSource) -> bool:
    ident = src.access_identifier or ""
    return src.access_method == "hats" and (
        ident.startswith("http://") or ident.startswith("https://") or ident.startswith("vos:")
    )


def _rate_host(src: CatalogueSource) -> str | None:
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
    if isinstance(value, int):
        return value if value > 0 else DEFAULT_ESTIMATED_SIZE
    if isinstance(value, float):
        return int(value) if math.isfinite(value) and value > 0 else DEFAULT_ESTIMATED_SIZE
    try:
        parsed = float(str(value))
        return int(parsed) if math.isfinite(parsed) and parsed > 0 else DEFAULT_ESTIMATED_SIZE
    except (OverflowError, ValueError):
        return DEFAULT_ESTIMATED_SIZE


# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #
def _http_request(
    url: str,
    bucket: TokenBucket,
    *,
    head: bool = False,
    range_probe: bool = False,
) -> tuple[bytes | None, int | None]:
    """GET (or HEAD) ``url`` under the token bucket with 429/503 backoff.

    Returns ``(body, content_length)`` (body ``None`` for HEAD).  Raises the
    underlying exception after retries are exhausted.
    """
    last_exc: Exception | None = None
    for attempt in range(_HTTP_RETRIES):
        bucket.acquire()
        try:
            headers = {"Range": "bytes=0-0"} if range_probe else {}
            req = urllib.request.Request(url, headers=headers, method="HEAD" if head else "GET")
            with urllib.request.urlopen(req, timeout=120) as resp:
                length = resp.headers.get("Content-Length")
                size = int(length) if length and length.isdigit() else None
                if range_probe:
                    content_range = resp.headers.get("Content-Range", "")
                    match = re.fullmatch(r"bytes\s+\d+-\d+/(\d+)", content_range)
                    if match:
                        size = int(match.group(1))
                if head:
                    return None, size
                body = resp.read(1) if range_probe else resp.read()
                return body, size
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


def _http_listing(url: str, bucket: TokenBucket) -> tuple[list[str], bool]:
    """Entry names and whether the HTTP directory listing was retrieved."""
    try:
        body, _ = _http_request(url, bucket)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError):
        return [], False
    if not body:
        return [], False
    text = body.decode("utf-8", errors="replace")
    names: list[str] = []
    for m in re.finditer(r'href="([^"?#]+)"', text):
        name = m.group(1)
        if name in (".", ".."):
            continue
        names.append(urllib.parse.unquote(name))
    return names, True


def _http_probe(url: str, bucket: TokenBucket) -> tuple[bool, int | None]:
    """Return whether a HEAD request succeeded, even when size is unknown."""
    try:
        _, size = _http_request(url, bucket, head=True)
        return True, size
    except urllib.error.HTTPError as exc:
        if exc.code != 405:
            return False, None
        try:
            body, size = _http_request(url, bucket, range_probe=True)
            return body is not None, size
        except (urllib.error.HTTPError, urllib.error.URLError, OSError):
            return False, None
    except (urllib.error.URLError, OSError):
        return False, None


def _http_hats_properties(root: str, bucket: TokenBucket) -> tuple[dict[str, Any], int] | None:
    """Find parseable HATS properties, rejecting generic login/error pages."""
    for rel in ("properties", "dataset/properties", "hats.properties"):
        try:
            body, size = _http_request(f"{root}/{rel}", bucket)
        except (urllib.error.HTTPError, urllib.error.URLError, OSError):
            continue
        if not body:
            continue
        row_count = _hats_property_count(body)
        if row_count is not None:
            return {"rel": rel, "size": size}, row_count
    return None


def _hats_property_count(body: bytes) -> int | None:
    props = {}
    for line in body.decode("utf-8", errors="replace").splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            props[key.strip().lower()] = value.strip()
    try:
        row_count = int(props["hats_nrows"])
    except (KeyError, ValueError):
        return None
    return row_count if row_count >= 0 and any(key.startswith("hats_") for key in props) else None


def _validate_hats_partition_info(info: pl.DataFrame, row_count: int, rel: str) -> None:
    """Reject partition tables that cannot account for the advertised rows."""
    if row_count > 0 and info.height == 0:
        raise OSError(f"{rel} does not agree with hats_nrows; preserving the existing mirror")
    if "count" in info.columns:
        counts = [_as_size(value) for value in info["count"].to_list()]
        if any(count is None or count < 0 for count in counts):
            raise OSError(
                f"{rel} contains invalid partition counts; preserving the existing mirror"
            )
        if sum(int(count) for count in counts if count is not None) != row_count:
            raise OSError(
                f"{rel} counts do not agree with hats_nrows; preserving the existing mirror"
            )


# --------------------------------------------------------------------------- #
# remote HATS listing
# --------------------------------------------------------------------------- #
def _remote_hats_listing(
    src: CatalogueSource, bucket: TokenBucket | None = None
) -> list[dict[str, Any]]:
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
            body, info_size = _http_request(f"{root}/{info_rel}", bucket)
        except (urllib.error.HTTPError, urllib.error.URLError, OSError):
            continue
        if not body:
            continue
        try:
            info = pl.read_parquet(_io.BytesIO(body))
            if "file_loc" not in info.columns:
                continue
            properties_info = _http_hats_properties(root, bucket)
            if properties_info is None:
                continue
            properties, row_count = properties_info
            _validate_hats_partition_info(info, row_count, info_rel)
            base = "dataset" if info_rel.startswith("dataset/") else ""
            files: list[dict[str, Any]] = [{"rel": info_rel, "size": info_size}]
            for row in info.rows(named=True):
                raw_loc = row.get("file_loc")
                if not isinstance(raw_loc, str) or not raw_loc.strip():
                    raise OSError(f"invalid file_loc in {info_rel}; preserving the existing mirror")
                loc = raw_loc.strip().strip("/")
                if any(part in (".", "..") for part in loc.split("/")):
                    raise OSError(f"unsafe file_loc in {info_rel}; preserving the existing mirror")
                rel = f"{base}/{loc}" if base else loc
                rel = rel.replace("//", "/")
                nfiles = _as_size(row.get("Nfiles"))
                if not rel.endswith(".parquet") and nfiles is not None and nfiles > 1:
                    names, complete = _http_listing(f"{root}/{rel}", bucket)
                    parts = [name.strip("/") for name in names if name.lower().endswith(".parquet")]
                    if not complete or len(parts) != nfiles:
                        raise OSError(
                            f"partition {loc!r} advertised by {info_rel} is incomplete; "
                            "preserving the existing mirror"
                        )
                    for part in parts:
                        if Path(part).name != part:
                            raise OSError(f"unsafe partition filename in {info_rel}")
                        candidate = f"{rel}/{part}"
                        found, remote_size = _http_probe(f"{root}/{candidate}", bucket)
                        if not found:
                            raise OSError(
                                f"partition file {candidate!r} advertised by {info_rel} "
                                "is unavailable; preserving the existing mirror"
                            )
                        files.append(
                            {
                                "rel": candidate,
                                "size": (
                                    remote_size
                                    if remote_size is not None
                                    else _as_size(row.get("file_size"))
                                ),
                            }
                        )
                    continue
                candidates = (rel,) if rel.endswith(".parquet") else (rel, f"{rel}.parquet")
                for candidate in candidates:
                    found, remote_size = _http_probe(f"{root}/{candidate}", bucket)
                    if found:
                        listed_size = _as_size(row.get("file_size"))
                        files.append(
                            {
                                "rel": candidate,
                                "size": remote_size if remote_size is not None else listed_size,
                            }
                        )
                        break
                else:
                    raise OSError(
                        f"partition {loc!r} advertised by {info_rel} is unavailable; "
                        "preserving the existing mirror"
                    )

            files.append(properties)
            metadata_files = (
                "dataset/partition_info.parquet",
                "partition_info.parquet",
                "dataset/_common_metadata",
                "dataset/_metadata",
                "_common_metadata",
                "_metadata",
            )
            known = {entry["rel"] for entry in files}
            for extra in metadata_files:
                if extra in known:
                    continue
                found, size = _http_probe(f"{root}/{extra}", bucket)
                if found:
                    files.append({"rel": extra, "size": size})
            schema_files = {
                "dataset/_common_metadata",
                "dataset/_metadata",
                "_common_metadata",
                "_metadata",
            }
            if row_count == 0 and not any(entry["rel"] in schema_files for entry in files):
                raise OSError(f"empty HATS catalogue {root} has no readable schema metadata")
            return files
        except (ImportError, OSError, ValueError):
            # A parsed authoritative table with malformed or missing partitions
            # is incomplete. Falling back to a directory walk could then prune
            # files the table still advertises.
            raise
    files, complete = _http_walk_hats(root, bucket)
    if complete:
        return files
    raise OSError(f"could not obtain a complete HATS listing from {root}")


def _as_size(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _http_walk_hats(root: str, bucket: TokenBucket) -> tuple[list[dict[str, Any]], bool]:
    """Walk accessible HTML listings, returning files and listing completeness."""
    out: list[dict[str, Any]] = []
    root_names, complete = _http_listing(root, bucket)
    if not complete:
        return [], False
    dataset_listed = False
    for name in root_names:
        rel = name.strip("/")
        if rel == "dataset":
            dataset_listed = True
        if rel.startswith("Norder="):
            entries, complete = _walk_order_dir(f"{root}/{rel}", bucket, rel)
            out.extend(entries)
            if not complete:
                return out, False
        elif rel.lower().endswith(".parquet") or Path(rel).name in _HATS_METADATA_NAMES:
            found, size = _http_probe(f"{root}/{rel}", bucket)
            if not found:
                return out, False
            out.append({"rel": rel, "size": size})

    if dataset_listed:
        dataset_names, complete = _http_listing(f"{root}/dataset", bucket)
        if not complete:
            return out, False
        for name in dataset_names:
            child = name.strip("/")
            rel = f"dataset/{child}"
            if child.startswith("Norder="):
                entries, complete = _walk_order_dir(f"{root}/dataset/{child}", bucket, rel)
                out.extend(entries)
                if not complete:
                    return out, False
            elif child.lower().endswith(".parquet") or child in _HATS_METADATA_NAMES:
                found, size = _http_probe(f"{root}/{rel}", bucket)
                if not found:
                    return out, False
                out.append({"rel": rel, "size": size})
    properties_info = _http_hats_properties(root, bucket)
    if properties_info is None or not out:
        return [], False
    properties, row_count = properties_info
    out = [
        entry
        for entry in out
        if entry["rel"] not in ("properties", "dataset/properties", "hats.properties")
    ]
    out.append(properties)
    has_pixels = any("Norder=" in entry["rel"] for entry in out)
    has_partition_info = any(
        Path(entry["rel"]).name in ("partition_info.parquet", "partition_info.csv") for entry in out
    )
    has_schema = any(entry["rel"].endswith(("_common_metadata", "_metadata")) for entry in out)
    if (row_count > 0 and not has_pixels) or (
        row_count == 0 and not (has_partition_info and has_schema)
    ):
        return [], False
    return out, True


def _walk_order_dir(
    url: str, bucket: TokenBucket, prefix: str
) -> tuple[list[dict[str, Any]], bool]:
    out: list[dict[str, Any]] = []
    names, complete = _http_listing(url, bucket)
    if not complete:
        return [], False
    for name in names:
        child = name.strip("/")
        if child.startswith("Dir="):
            entries, complete = _walk_dir(url + "/" + child, bucket, f"{prefix}/{child}")
            out.extend(entries)
            if not complete:
                return out, False
    return out, True


def _walk_dir(url: str, bucket: TokenBucket, prefix: str) -> tuple[list[dict[str, Any]], bool]:
    out: list[dict[str, Any]] = []
    names, complete = _http_listing(url, bucket)
    if not complete:
        return [], False
    for name in names:
        child = name.strip("/")
        rel = f"{prefix}/{child}" if prefix else child
        if child.lower().endswith(".parquet"):
            found, size = _http_probe(url + "/" + child, bucket)
            if not found:
                return out, False
            out.append({"rel": rel, "size": size})
        elif name.endswith("/"):  # multi-file pixel dir (Npix=NN/…)
            files, complete = _http_listing(url + "/" + child, bucket)
            if not complete:
                return out, False
            for f in files:
                child_file = f.strip("/")
                if child_file.lower().endswith(".parquet"):
                    found, size = _http_probe(url + "/" + child + "/" + child_file, bucket)
                    if not found:
                        return out, False
                    out.append({"rel": rel + "/" + child_file, "size": size})
    return out, True


def _vos_hats_listing(ident: str) -> list[dict[str, Any]]:
    storage = open_storage(ident)
    props_rel = next(
        (
            rel
            for rel in ("properties", "dataset/properties", "hats.properties")
            if storage.exists(rel)
        ),
        None,
    )
    if props_rel is None:
        raise OSError(
            f"could not confirm HATS properties at {ident}; preserving the existing mirror"
        )
    with tempfile.TemporaryDirectory(prefix="xmatcher-vos-list-") as directory:
        props_path = Path(directory) / "properties"
        storage.stage_in(props_rel, props_path)
        row_count = _hats_property_count(props_path.read_bytes())
        if row_count is None:
            raise OSError(f"invalid HATS properties at {ident}; preserving the existing mirror")

    info_rel = None
    info = None
    for candidate in ("dataset/partition_info.parquet", "partition_info.parquet"):
        if storage.exists(candidate):
            try:
                info = storage.read_parquet(candidate)
            except Exception:  # noqa: BLE001 - try the canonical CSV copy
                continue
            info_rel = candidate
            break
    if info is None:
        for candidate in ("dataset/partition_info.csv", "partition_info.csv"):
            if not storage.exists(candidate):
                continue
            try:
                with tempfile.TemporaryDirectory(prefix="xmatcher-vos-list-") as directory:
                    local = Path(directory) / "partition_info.csv"
                    storage.stage_in(candidate, local)
                    info = pl.read_csv(local)
            except Exception:  # noqa: BLE001
                continue
            info_rel = candidate
            break
    if info_rel is None:
        raise OSError(
            f"could not confirm a complete HATS listing at {ident}; preserving the existing mirror"
        )
    assert info is not None
    if "file_loc" not in info.columns:
        column_names = {column.lower(): column for column in info.columns}
        if "norder" not in column_names or "npix" not in column_names:
            raise OSError(f"invalid HATS listing at {ident}; preserving the existing mirror")
        file_locs = []
        for row in info.rows(named=True):
            try:
                order = int(row[column_names["norder"]])
                pix = int(row[column_names["npix"]])
            except (TypeError, ValueError, KeyError) as exc:
                raise OSError(
                    f"invalid pixel row in HATS listing at {ident}; preserving the existing mirror"
                ) from exc
            file_locs.append(f"Norder={order}/Dir={(pix // 10000) * 10000}/Npix={pix}")
        info = info.with_columns(pl.Series("file_loc", file_locs))
    if "file_loc" not in info.columns:
        raise OSError(f"invalid HATS listing at {ident}; preserving the existing mirror")
    _validate_hats_partition_info(info, row_count, info_rel)
    base = "dataset" if info_rel.startswith("dataset/") or info_rel.endswith(".csv") else ""
    out: list[dict[str, Any]] = [
        {"rel": props_rel, "size": storage.size(props_rel)},
        {"rel": info_rel, "size": storage.size(info_rel)},
    ]
    for row in info.rows(named=True):
        raw_loc = row.get("file_loc")
        if not isinstance(raw_loc, str) or not raw_loc.strip():
            raise OSError(f"invalid file_loc in {info_rel}; preserving the existing mirror")
        loc = raw_loc.strip().strip("/")
        if any(part in (".", "..") for part in loc.split("/")):
            raise OSError(f"unsafe file_loc in {info_rel}; preserving the existing mirror")
        rel = f"{base}/{loc}" if base else loc
        nfiles = _as_size(row.get("Nfiles")) or 1
        if not rel.endswith(".parquet") and nfiles > 1:
            part_names = storage.list(rel)
            parts = [name for name in part_names if name.endswith(".parquet")]
            if len(parts) != nfiles or any("/" in name or name in (".", "..") for name in parts):
                raise OSError(f"incomplete HATS partition {rel!r}; preserving the existing mirror")
            for part in parts:
                candidate = f"{rel}/{part}"
                if not storage.exists(candidate):
                    raise OSError(
                        f"missing HATS partition file {candidate!r}; preserving the existing mirror"
                    )
                out.append({"rel": candidate, "size": storage.size(candidate)})
            continue
        candidates = (rel,) if rel.endswith(".parquet") else (rel, f"{rel}.parquet")
        for candidate in candidates:
            if storage.exists(candidate):
                size = storage.size(candidate)
                out.append(
                    {
                        "rel": candidate,
                        "size": size if size >= 0 else _as_size(row.get("file_size")),
                    }
                )
                break
        else:
            raise OSError(f"missing HATS partition {rel!r}; preserving the existing mirror")
    metadata_files = (
        "dataset/_common_metadata",
        "dataset/_metadata",
        "_common_metadata",
        "_metadata",
    )
    for extra in metadata_files:
        if storage.exists(extra) and all(entry["rel"] != extra for entry in out):
            out.append({"rel": extra, "size": storage.size(extra)})
    if row_count == 0 and not any(
        entry["rel"].endswith(("_common_metadata", "_metadata")) for entry in out
    ):
        raise OSError(f"empty HATS catalogue {ident} has no readable schema metadata")
    return out


# --------------------------------------------------------------------------- #
# manifest helpers
# --------------------------------------------------------------------------- #
def _read_json(cache: Storage, rel: str) -> dict[str, Any]:
    try:
        if not cache.exists(rel):
            return {}
        if isinstance(cache, LocalStorage):
            return json.loads(Path(cache.root / rel).read_text())
        tmpdir = Path(tempfile.mkdtemp(prefix="xmatcher-mf-"))
        try:
            local = tmpdir / "manifest.json"
            cache.stage_in(rel, local)
            return json.loads(local.read_text())
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def _write_json(cache: Storage, rel: str, obj: dict[str, Any]) -> None:
    tmpdir = Path(tempfile.mkdtemp(prefix="xmatcher-mf-"))
    local = tmpdir / "manifest.json"
    try:
        local.write_text(json.dumps(obj, indent=2, sort_keys=True))
        cache.stage_out(local, rel)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _copy_tree_up(cache: Storage, local_dir: Path, rel: str) -> None:
    """Copy a local directory tree into the cache root at ``rel``.

    Both backends stage the complete tree beside the live path before
    publication. VOSpace therefore needs a backend that supports moving
    containers; file-by-file publication is unsafe. The rename operations
    protect against transfer and promotion errors, while crash atomicity is
    limited to the guarantees of the storage backend.
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
        target_rel = rel.rstrip("/")
        tmp_rel = target_rel + ".tmp"
        backup_rel = target_rel + ".old"

        # Recover a crash between moving the target aside and promoting the
        # completed staging tree. If both paths exist, publication completed
        # and only backup cleanup was interrupted.
        if cache.exists(backup_rel):
            if cache.exists(target_rel):
                _remove_storage_tree(cache, backup_rel)
            else:
                cache.rename(backup_rel, target_rel)
        # A failed cleanup aborts safely with the live tree still advertised.
        if cache.exists(tmp_rel):
            _remove_storage_tree(cache, tmp_rel)

        cache.mkdir(tmp_rel)
        try:
            for path in files:
                dest_rel = f"{tmp_rel}/{path.relative_to(local_dir)}"
                cache.stage_out(path, dest_rel)
        except BaseException:
            try:
                _remove_storage_tree(cache, tmp_rel)
            except Exception:
                logger.warning("Could not remove incomplete staged mirror %s", tmp_rel)
            raise

        had_target = cache.exists(target_rel)
        if had_target:
            try:
                cache.rename(target_rel, backup_rel)
            except BaseException:
                try:
                    _remove_storage_tree(cache, tmp_rel)
                except Exception:
                    logger.warning("Could not remove incomplete staged mirror %s", tmp_rel)
                raise
        try:
            cache.rename(tmp_rel, target_rel)
        except BaseException as publish_error:
            try:
                # A backend may fail after creating the destination. It is
                # safe to remove it here because the prior tree is in backup.
                if had_target and cache.exists(target_rel):
                    _remove_storage_tree(cache, target_rel)
                if had_target:
                    cache.rename(backup_rel, target_rel)
            except Exception as rollback_error:
                raise OSError(
                    f"failed to publish mirror {target_rel!r}; rollback also failed "
                    f"({rollback_error}); previous mirror remains at {backup_rel!r}"
                ) from publish_error
            try:
                if cache.exists(tmp_rel):
                    _remove_storage_tree(cache, tmp_rel)
            except Exception:
                logger.warning("Could not remove incomplete staged mirror %s", tmp_rel)
            raise

        if had_target:
            try:
                _remove_storage_tree(cache, backup_rel)
            except Exception:
                # The new tree is complete and published; stale backup cleanup
                # is best-effort and is retried before the next publication.
                logger.warning("Could not remove old mirror backup %s", backup_rel)


def _remove_storage_tree(cache: Storage, rel: str) -> None:
    """Remove a file or container tree through the storage protocol.

    VOSpace listings may echo a leaf's own basename and do not mark
    containers. Treat that echo or an empty listing as a leaf; failed
    recursive removal remains noisy rather than silently discarding an
    advertised tree.
    """
    children = cache.list(rel)
    if not children or children == [rel.rsplit("/", 1)[-1]]:
        cache.rm(rel)
        return
    for child in children:
        _remove_storage_tree(cache, f"{rel}/{child}")
    cache.rm(rel)


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
    progress_cb: Callable[[str], None] | None,
    stats: SyncStats,
    endpoint: CatalogueSource | None = None,
) -> None:
    """Mirror every remote HATS partition file into ``cache/<name>/<version>``.

    ``endpoint`` (fallback copy) swaps the fetching URL while ``src`` keeps
    the cache identity (name + version dir), so a mirror served by a
    different copy still lands where ``locate_mirrored(src)`` looks.
    """
    fetch = endpoint or src
    version = _version_dir(src)
    prefix = f"{_safe_name(src.name)}/{version}"
    remote = _remote_hats_listing(fetch, bucket=bucket)
    failed_before = stats.failed
    manifest_rel = f"{prefix}/{_MANIFEST_NAME}"
    manifest = _read_json(cache, manifest_rel)
    downloaded_before = stats.files_downloaded

    def do_fetch(entry: dict[str, Any]) -> None:
        rel = entry["rel"]
        tmpdir = Path(tempfile.mkdtemp(prefix="xmatcher-"))
        local_tmp = tmpdir / Path(rel).name
        cache_rel = f"{prefix}/{rel}"
        backup = tmpdir / "previous"
        had_old = False
        backup_ready = False
        upload_started = False
        try:
            if _remote_via_storage(fetch):
                # vos: nodes have no HTTP endpoint; transfer node → temp → cache.
                storage = open_storage((fetch.access_identifier or "").rstrip("/"))
                storage.stage_in(rel, local_tmp)
            else:
                body, _ = _http_request(_remote_url(fetch, rel), bucket)
                if body is None:
                    raise CrossMatchError(f"empty body for {rel}")
                local_tmp.write_bytes(body)
            # LocalStorage publishes each file with os.replace. VOSpace lacks
            # that guarantee, so keep one prior file beside the staged update
            # for rollback if its upload fails.
            had_old = not isinstance(cache, LocalStorage) and cache.exists(cache_rel)
            if had_old:
                cache.stage_in(cache_rel, backup)
                backup_ready = True
            upload_started = True
            cache.stage_out(local_tmp, cache_rel)
            stored = cache.size(cache_rel)
            stats.files_downloaded += 1
            stats.bytes_downloaded += max(0, stored) if stored >= 0 else 0
            manifest.setdefault("files", {})[rel] = {"size": stored}
            if progress_cb:
                progress_cb(f"sync {src.name}: fetched {rel}")
        except Exception as exc:  # noqa: BLE001
            stats.failed += 1
            logger.warning("sync %s: failed to fetch %s: %s", src.name, rel, exc)
            if backup_ready:
                try:
                    cache.stage_out(backup, cache_rel)
                except Exception as rollback_exc:  # noqa: BLE001
                    logger.error(
                        "sync %s: could not restore previous %s: %s",
                        src.name,
                        cache_rel,
                        rollback_exc,
                    )
            elif upload_started and not had_old and not isinstance(cache, LocalStorage):
                # A failed VOS upload may have created a partial node.
                try:
                    cache.rm(cache_rel)
                except Exception as rollback_exc:  # noqa: BLE001
                    logger.error(
                        "sync %s: could not remove partial %s: %s",
                        src.name,
                        cache_rel,
                        rollback_exc,
                    )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    data_todo: list[dict[str, Any]] = []
    metadata_todo: list[dict[str, Any]] = []
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
        elif Path(rel).name in _HATS_METADATA_NAMES:
            metadata_todo.append(entry)
        else:
            data_todo.append(entry)

    def fetch_phase(entries: list[dict[str, Any]]) -> None:
        if workers > 1 and entries:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                list(ex.map(do_fetch, entries))
        else:
            for entry in entries:
                do_fetch(entry)

    if progress_cb:
        progress_cb(f"sync {src.name}: {len(data_todo) + len(metadata_todo)} file(s) to fetch")
    fetch_phase(data_todo)
    data_phase_succeeded = stats.failed == failed_before
    if data_phase_succeeded:
        fetch_phase(metadata_todo)

    files_manifest = manifest.setdefault("files", {})
    pruned = False
    if stats.failed == failed_before:
        remote_rels = {entry["rel"] for entry in remote}
        for stale in sorted(set(files_manifest) - remote_rels):
            try:
                cache.rm(f"{prefix}/{stale}")
            except Exception as exc:  # noqa: BLE001
                # Keep the manifest entry so the next sync retries the removal,
                # and still persist the files already pruned or downloaded.
                stats.failed += 1
                logger.warning("sync %s: failed to remove stale %s: %s", src.name, stale, exc)
                continue
            files_manifest.pop(stale, None)
            pruned = True
        manifest["version"] = version
        manifest["source"] = src.access_identifier
        if stats.failed == failed_before:
            manifest["fetched_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        else:
            manifest.pop("fetched_at", None)
    elif stats.files_downloaded > downloaded_before:
        # Keep successful per-file updates discoverable after a partial sync;
        # metadata and stale-file pruning wait for a complete data phase.
        manifest["version"] = version
        manifest["source"] = src.access_identifier
        manifest.pop("fetched_at", None)
    if stats.files_downloaded > downloaded_before or stats.failed == failed_before or pruned:
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


def _tap_run(src: CatalogueSource, service, query: str, maxrec: int | None = None) -> pl.DataFrame:
    from .io_utils import astropy_table_to_polars
    from .tap import execute_tap_query

    return astropy_table_to_polars(execute_tap_query(service, query, maxrec=maxrec))


def _tap_page_query(
    src: CatalogueSource,
    *,
    offset: int | None = None,
    window: tuple[Any, Any] | None = None,
    after_key: Any = None,
    key: str | None = None,
    limit: int | None = None,
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


def _tap_count_query(fetch: CatalogueSource, window: tuple[Any, Any] | None = None) -> str:
    key = fetch.id_column or fetch.ra_column
    table_q = _table_ref(fetch.access_identifier or "", tap_url=fetch.tap_url or "")
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
    tmpdir = Path(tempfile.mkdtemp(prefix="xmatcher-hash-"))
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
    progress_cb: Callable[[str], None] | None,
    stats: SyncStats,
    endpoint: CatalogueSource | None = None,
) -> None:
    """Incrementally mirror a TAP table into ``cache/<name>/tap-<version>``.

    ``endpoint`` (fallback copy) swaps the service/table identity for the
    queries while ``src`` keeps the cache identity (name + version dir), so
    a mirror served by a different archive still lands where
    ``locate_mirrored(src)`` looks.

    Cold sync fetches sequential OFFSET pages and requires a unique,
    non-null ordering key when one is available. Re-sync probes stored windows
    with ``COUNT(*)`` and refetches changed windows when each still fits one
    page (append-only sources additionally fetch the new tail). Counts cannot
    detect same-count value edits; use ``force`` for mutable tables. The page
    Parquet files under ``<cache>/<name>/raw/pages/`` are the incremental
    store; the HATS catalogue under ``<cache>/<name>/<version>/`` is rebuilt
    whenever a page changed.
    """
    fetch = endpoint or src
    name = _safe_name(src.name)
    version = _version_dir(src)
    prefix = f"{name}/{version}"
    raw = f"{name}/raw"
    manifest_rel = f"{raw}/{_MANIFEST_NAME}"
    manifest = _read_json(cache, manifest_rel)
    key = src.id_column or src.ra_column
    ceiling = int(_TAP_CEILING_FACTOR * estimated_size)
    service = _tap_service(fetch, auth_session)
    fetched_page_files: dict[int, str] = {}

    def page_rel(idx: int, entry: dict[str, Any] | None = None) -> str:
        filename = (entry or {}).get("file", f"page_{idx:04d}.parquet")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise CrossMatchError(f"TAP page registry contains an invalid file name: {filename!r}")
        return f"{raw}/pages/{filename}"

    def fetch_page(query: str, idx: int, *, previous_key: Any = None) -> pl.DataFrame:
        bucket.acquire()
        df = _tap_run(fetch, service, query, maxrec=page_size)
        if df.height:
            if key:
                keys = df[key]
                if keys.null_count() or keys.is_duplicated().any():
                    raise CrossMatchError(
                        f"TAP table '{src.name}' needs a unique, non-null ordering key "
                        "for incremental sync."
                    )
                if previous_key is not None and keys[0] <= previous_key:
                    raise CrossMatchError(
                        f"TAP table '{src.name}' needs a unique, non-null ordering key "
                        "for incremental sync."
                    )
            filename = f"page_{idx:04d}.{secrets.token_hex(8)}.parquet"
            fetched_page_files[idx] = filename
            cache.write_parquet(df, f"{raw}/pages/{filename}")
            stats.pages += 1
            stats.files_downloaded += 1
            stats.bytes_downloaded += cache.size(f"{raw}/pages/{filename}")
            if progress_cb:
                progress_cb(
                    f"sync {src.name}: page {idx:04d} fetched "
                    f"({df.height} rows, {stats.pages} page(s) so far)"
                )
        return df

    def record_page(idx: int, df: pl.DataFrame) -> None:
        if not df.height:
            return
        file = fetched_page_files[idx]
        entry: dict[str, Any] = {
            "rows": df.height,
            "sha256": _file_sha256(cache, f"{raw}/pages/{file}"),
            "file": file,
        }
        if key:
            keys = df[key].to_list()
            entry["first_key"] = keys[0]
            entry["last_key"] = keys[-1]
        manifest.setdefault("pages", {})[str(idx)] = entry

    pages_manifest = manifest.get("pages")
    if not pages_manifest or force:
        # ---------------- cold (or forced) fetch: OFFSET pages ------------
        manifest["pages"] = {}
        if progress_cb:
            progress_cb(f"sync {src.name}: fetching full table")
        offset = 0
        rows_fetched = 0
        previous_key = None
        while True:
            df = fetch_page(
                _tap_page_query(fetch, offset=offset, limit=page_size),
                offset // page_size,
                previous_key=previous_key,
            )
            if df.is_empty():
                break
            rows_fetched += df.height
            if rows_fetched > ceiling:
                for filename in fetched_page_files.values():
                    cache.rm(f"{raw}/pages/{filename}")
                raise CrossMatchError(
                    f"TAP table '{src.name}' exceeds the sync ceiling of {ceiling} rows "
                    f"(fetched at least {rows_fetched}; estimated_size {estimated_size} "
                    f"x {_TAP_CEILING_FACTOR}). Ingest this catalogue as HATS instead."
                )
            record_page(offset // page_size, df)
            if key:
                previous_key = df[key][-1]
            if df.height < page_size:
                break
            offset += page_size
        changed = True
    else:
        # ---------------- incremental: probe stored windows -------------
        changed = False
        pages = pages_manifest
        if key:
            # tail probe: did the highest key move (append-only sources)?
            tail_df = _tap_run(fetch, service, _tap_key_query(fetch, max_key=True), maxrec=1)
            new_last = tail_df[key][0] if tail_df.height else None
            last_idx = str(max(int(i) for i in pages))
            last_key = pages[last_idx].get("last_key")
            tail_changed = last_key is None or new_last != last_key

            for idx_str in sorted(pages, key=int):
                entry = pages[idx_str]
                window = (entry.get("first_key"), entry.get("last_key"))
                if window[0] is None or window[1] is None:
                    continue
                probe = _tap_run(fetch, service, _tap_count_query(fetch, window=window), maxrec=1)
                n = probe["n"][0] if probe.height else 0
                if int(n) != int(entry["rows"]):
                    if int(n) > page_size:
                        raise CrossMatchError(
                            f"TAP table '{src.name}' changed so an ordering-key window now "
                            f"contains more than {page_size} rows; re-run with --force to "
                            "refresh the complete mirror safely."
                        )
                    if progress_cb:
                        progress_cb(f"sync {src.name}: page {idx_str} changed")
                    df = fetch_page(
                        _tap_page_query(fetch, window=window, key=key, limit=page_size),
                        int(idx_str),
                    )
                    if df.height:
                        record_page(int(idx_str), df)
                    else:
                        # window vanished: drop the stored page + registry entry
                        pages.pop(idx_str, None)
                    changed = True

            if tail_changed:
                # appended rows: keyset continuation past the stored last key
                cont_idx = int(last_idx) + 1
                lo = last_key
                while True:
                    df = fetch_page(
                        _tap_page_query(fetch, after_key=lo, key=key, limit=page_size),
                        cont_idx,
                        previous_key=lo,
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
            probe = _tap_run(fetch, service, _tap_count_query(fetch), maxrec=1)
            n = probe["n"][0] if probe.height else 0
            if int(n) != int(manifest.get("total_rows", 0)):
                raise CrossMatchError(
                    f"TAP table '{src.name}' has no ordering key; row count changed from "
                    f"{manifest.get('total_rows')} to {n}. Re-run with --force to re-sync."
                )

    if changed or force or not cache.exists(f"{prefix}/properties"):
        page_rels = [
            page_rel(int(index), entry)
            for index, entry in sorted(
                manifest.get("pages", {}).items(), key=lambda item: int(item[0])
            )
        ]
        _rebuild_tap_hats(
            cache,
            src,
            page_rels,
            prefix,
            hats_threshold=hats_threshold,
            stats=stats,
        )
    manifest["version"] = version
    manifest["source"] = src.access_identifier
    manifest["page_size"] = page_size
    manifest["ordering"] = key
    manifest["total_rows"] = _manifest_total_rows(manifest)
    manifest["fetched_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_json(cache, manifest_rel, manifest)
    live_pages = {
        Path(page_rel(int(index), entry)).name for index, entry in manifest.get("pages", {}).items()
    }
    for old_page in cache.list(f"{raw}/pages"):
        if old_page.endswith(".parquet") and old_page not in live_pages:
            cache.rm(f"{raw}/pages/{old_page}")
    logger.info(
        "sync %s: %d bytes, %d files (%d skipped, %d failed, %d pages)",
        src.name,
        stats.bytes_downloaded,
        stats.files_downloaded,
        stats.files_skipped,
        stats.failed,
        stats.pages,
    )


def _manifest_total_rows(manifest: dict[str, Any]) -> int:
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
    """Convert ``frame`` into a valid HATS catalogue at ``out_dir``.

    Adaptive NESTED tiling: any cell holding more than ``threshold`` rows is
    split until ``max_order``.  Layout matches the `hats` reader contract:
    ``dataset/Norder=…/Dir=…/Npix=….parquet`` pixel files, a root
    ``partition_info.csv`` (and parquet copy), and ``properties``.

    cdshealpix computes every row's pixel per level (O(N·orders));
    fine for mirror-sized frames, replace with an index scan when a frame
    exceeds ~50M rows.
    """
    import cdshealpix  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    from astropy.coordinates import Latitude, Longitude  # noqa: PLC0415

    from .astro_utils import require_finite_coordinates  # noqa: PLC0415

    # Validate once, before any pixellation: one non-finite latitude makes
    # cdshealpix panic (PanicException, a BaseException) and would otherwise
    # kill a cold sync of an entire survey with an unhelpful Rust assertion.
    if not frame.is_empty():
        require_finite_coordinates(
            frame[ra_column].cast(pl.Float64).to_numpy(),
            frame[dec_column].cast(pl.Float64).to_numpy(),
            label=f"catalogue to mirror ({ra_column}/{dec_column})",
        )

    out_dir = Path(out_dir)
    dataset = out_dir / "dataset"
    dataset.mkdir(parents=True, exist_ok=True)

    segments: list[tuple[int, int, pl.DataFrame]] = []
    if not frame.is_empty():
        stack: list[tuple[int, int, pl.DataFrame]] = [(0, 0, frame)]
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
            parts = sub.with_columns(pl.Series("_mask", child)).partition_by("_mask", as_dict=True)
            for k, psub in parts.items():
                cpix = k[0]
                stack.append((order + 1, int(cpix), psub.drop("_mask")))

    info: list[dict[str, Any]] = []
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
    # An empty catalogue is legal (a region / query with no rows) and must
    # still produce a readable HATS dir.  Building from ``info`` alone left the
    # frame with no columns, so ``.select`` raised ColumnNotFoundError.
    info_schema = {
        "Norder": pl.Int64,
        "Dir": pl.Int64,
        "Npix": pl.Int64,
        "Nfiles": pl.Int64,
        "file_loc": pl.Utf8,
        "file_size": pl.Int64,
        "count": pl.Int64,
    }
    info_df = pl.DataFrame(info, schema=info_schema).select(list(info_schema))
    # hats 0.7.x reads the root copy; the dataset copy is the classic spec.
    info_df.write_csv(dataset / "partition_info.csv")
    info_df.write_csv(out_dir / "partition_info.csv")
    info_df.write_parquet(dataset / "partition_info.parquet")

    props = {
        "obs_collection": out_dir.name or "xmatcher",
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

        if info:
            schemas = [pq.read_schema(dataset / f"{e['file_loc']}.parquet") for e in info]
            unified = pa.unify_schemas(schemas)
        else:
            # No partitions: the input frame's own schema is still known.
            unified = frame.to_arrow().schema
        pq.write_metadata(unified, dataset / "_common_metadata")
        pq.write_metadata(unified, dataset / "_metadata")
    except Exception:  # noqa: BLE001
        logger.warning("skipping _metadata for %s (heterogeneous schemas)", out_dir)


def _rebuild_tap_hats(
    cache: Storage,
    src: CatalogueSource,
    page_rels: list[str],
    prefix: str,
    *,
    hats_threshold: int,
    stats: SyncStats,
) -> None:
    """Re-convert the page store into the ``cache/<name>/<version>`` HATS cat."""
    frames = []
    for page_rel in page_rels:
        try:
            frame = cache.read_parquet(page_rel)
        except Exception as exc:  # noqa: BLE001
            # An unreadable page (torn write, corrupted transfer) must not
            # silently shrink the converted catalogue while the manifest
            # still counts its rows: surface it and count the failure.
            stats.failed += 1
            logger.warning("sync %s: ignoring unreadable TAP page %s: %s", src.name, page_rel, exc)
            continue
        if frame.height:
            frames.append(frame)
    if not frames:
        cache.mkdir(prefix)
        return
    frame = pl.concat(frames, how="diagonal_relaxed")
    tmpdir = Path(tempfile.mkdtemp(prefix="xmatcher-hats-"))
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
# endpoint failover + replica roots
# --------------------------------------------------------------------------- #
_ENDPOINT_ERRORS = (urllib.error.URLError, OSError, TimeoutError, TapError)


def _gate_fallback(primary: CatalogueSource, fb: CatalogueSource, label: str) -> None:
    """Refuse a named fallback whose schema differs from the primary's.

    The gate only fires when BOTH sides declare ``default_columns``; raw-URL
    fallbacks have no column metadata and pass through.  A same-schema copy
    is the contract for failover: mixing column layouts mid-sync would
    produce a manifest whose pages do not match the mirror.
    """
    if (
        primary.default_columns
        and fb.default_columns
        and (list(primary.default_columns) != list(fb.default_columns))
    ):
        raise CrossMatchError(
            f"fallback '{label}' for '{primary.name}' has different columns "
            f"than the primary ({fb.default_columns} vs {primary.default_columns}); "
            "configure a same-schema copy as the fallback"
        )


def _fallback_candidates(src: CatalogueSource) -> list[CatalogueSource]:
    """``[src, *endpoint-replaced fallback copies]`` for the failover walk.

    Every candidate keeps the primary's identity (name, columns, cache
    version dir); only the endpoint fields swap, so the mirror always lands
    where ``locate_mirrored(src)`` looks no matter which copy served it.
    """
    out = [src]
    for i, fb in enumerate(src.fallbacks):
        label = f"{src.name}.fallback[{i}]"
        if isinstance(fb, CatalogueSource):
            _gate_fallback(src, fb, fb.name or label)
            out.append(replace(src, tap_url=fb.tap_url, access_identifier=fb.access_identifier))
        else:
            out.append(replace(src, access_identifier=str(fb)))
    return out


def _sync_with_failover(
    src: CatalogueSource,
    cache: Storage,
    *,
    bucket: TokenBucket,
    force: bool,
    workers: int,
    page_size: int,
    hats_threshold: int,
    estimated_size: int,
    auth_session: Any,
    progress_cb: Callable[[str], None] | None,
    stats: SyncStats,
) -> None:
    """Mirror ``src``, retrying endpoint-class failures across its fallbacks.

    Endpoint-class failures only (connection refused/dropped, TAP job
    failure): ``CrossMatchError``-family failures (ceiling, row-count
    mismatch, column gate) always propagate — a mirror that *ran* against a
    different schema is worse than no mirror.  For remote HATS, per-file
    fetch errors are swallowed into ``stats.failed`` (the catalogue is
    mirrored incrementally, so a dead endpoint is only visible as failed
    transfers), so a candidate that failed any file — or moved zero files
    at all (listing or every fetch failed) — is retried against the next
    fallback.
    """
    candidates = _fallback_candidates(src)
    for i, cand in enumerate(candidates):
        if i:
            logger.warning(
                "sync %s: primary endpoint failed; trying fallback %s",
                src.name,
                cand.access_identifier or cand.tap_url or "?",
            )
        try:
            if src.access_method == "hats":
                # stats accumulate across candidates; judge this candidate
                # by the delta it produced.
                dl0, sk0, fa0 = (
                    stats.files_downloaded,
                    stats.files_skipped,
                    stats.failed,
                )
                _mirror_remote_hats(
                    src,
                    cache,
                    bucket=bucket,
                    force=force,
                    workers=workers,
                    progress_cb=progress_cb,
                    stats=stats,
                    endpoint=cand,
                )
                d_fail = stats.failed - fa0
                if d_fail > 0:
                    raise urllib.error.URLError(
                        f"{d_fail} file(s) failed to download from {cand.access_identifier}"
                    )
                if stats.files_downloaded - dl0 == 0 and stats.files_skipped - sk0 == 0:
                    raise urllib.error.URLError(
                        f"no files transferred from {cand.access_identifier}"
                    )
            else:
                _mirror_tap(
                    src,
                    cache,
                    bucket=bucket,
                    force=force,
                    page_size=page_size,
                    hats_threshold=hats_threshold,
                    estimated_size=estimated_size,
                    auth_session=auth_session,
                    progress_cb=progress_cb,
                    stats=stats,
                    endpoint=cand,
                )
            return
        except _ENDPOINT_ERRORS:
            if i == len(candidates) - 1:
                raise
            continue


def _replicate_tree(src_storage: Storage, dst_storage: Storage, rel: str) -> None:
    """Replace the mirrored tree at ``rel`` after staging a complete source copy."""
    rels: list[str] = []
    _walk_all_files(src_storage, rel, rels)
    if not rels:
        raise OSError(f"source tree {rel!r} has no files; preserving replica")
    tmpdir = Path(tempfile.mkdtemp(prefix="xmatcher-repl-"))
    try:
        for r in rels:
            prefix = rel.rstrip("/")
            relative = r[len(prefix) :].lstrip("/") if prefix else r
            local = Path(relative)
            if local.is_absolute() or ".." in local.parts:
                raise ValueError(f"replica source contains an unsafe path: {r!r}")
            local = tmpdir / local
            local.parent.mkdir(parents=True, exist_ok=True)
            src_storage.stage_in(r, local)
        if rel:
            _copy_tree_up(dst_storage, tmpdir, rel)
        else:
            # Empty rel means the caller intentionally targets a storage root;
            # replacing that root could remove unrelated data.
            for local in sorted(path for path in tmpdir.rglob("*") if path.is_file()):
                dst_storage.stage_out(local, local.relative_to(tmpdir).as_posix())
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _walk_all_files(storage: Storage, rel: str, acc: list[str], depth: int = 0) -> None:
    """Collect every FILE under ``rel`` (no filename filter).

    Union outputs also carry ``resume.state`` and ``run.jsonl``; staging
    preserves these alongside catalogue data so a copied union can resume.
    """
    if depth > 12:
        raise OSError(f"storage tree is nested too deeply under {rel!r}")
    try:
        children = storage.list(rel)
    except ValueError as exc:  # traversal-shaped rel from a hostile listing
        raise OSError(f"invalid storage listing path {rel!r}") from exc
    if not children:
        return
    if depth and children == [rel.rsplit("/", 1)[-1]]:
        return  # vls on a leaf file echoes the file's own basename
    for name in children:
        child = f"{rel}/{name}".strip("/")
        if name.endswith("/"):
            _walk_all_files(storage, child, acc, depth + 1)
            continue
        if isinstance(storage, LocalStorage):
            path = Path(storage.root) / child
            if path.is_dir():
                _walk_all_files(storage, child, acc, depth + 1)
            elif path.is_file():
                acc.append(child)
            else:
                raise OSError(f"listed storage node disappeared: {child!r}")
            continue
        node_children = storage.list(child)
        if node_children == [name]:  # VOS vls may echo the listed leaf's basename.
            acc.append(child)
        elif node_children:
            _walk_all_files(storage, child, acc, depth + 1)
        elif storage.exists(child):
            if storage.size(child) >= 0:
                acc.append(child)
            else:
                raise OSError(f"could not classify storage node {child!r}; preserving replica")
        else:
            raise OSError(f"listed storage node is unavailable: {child!r}")


def _copy_tree_files(src_storage: Storage, dst_storage: Storage, rel: str) -> None:
    """Copy every file under ``rel`` onto ``dst_storage`` (unfiltered walk;
    see :func:`_walk_all_files`)."""
    rels: list[str] = []
    _walk_all_files(src_storage, rel, rels)
    tmpdir = Path(tempfile.mkdtemp(prefix="xmatcher-tree-"))
    try:
        for r in rels:
            local = tmpdir / Path(r).name
            src_storage.stage_in(r, local)
            dst_storage.stage_out(local, r)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _replicate_to_roots(cache: Storage, src: CatalogueSource, replica_roots: list[str]) -> None:
    """Best-effort copy of a synced mirror tree onto each replica root.

    Replicas give the union reader a surviving copy when the primary root's
    file is gone (node loss, evicted /scratch).  Only complete mirrors are
    replicated (``stats.failed == 0`` is the caller's gate) and failures here
    never fail the run.
    """
    if not replica_roots:
        return
    prefix = f"{_safe_name(src.name)}/{_version_dir(src)}"
    if not cache.exists(f"{prefix}/properties") and not cache.exists(
        f"{prefix}/dataset/partition_info.parquet"
    ):
        return  # nothing durable was mirrored (fresh gate / no-op)
    for root in replica_roots:
        try:
            _replicate_tree(cache, open_storage(root), prefix)
            logger.info("replicated '%s' mirror to replica root %s", src.name, root)
        except Exception as exc:  # noqa: BLE001 - best-effort by contract
            logger.warning("replica root %s copy failed for '%s': %s", root, src.name, exc)


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def sync_catalogue(
    src: CatalogueSource,
    *,
    cache_root: str | None = None,
    replica_roots: list[str] | None = None,
    rate_limit_rps: float = 1.0,
    workers: int = 8,
    force: bool = False,
    progress_cb: Callable[[str], None] | None = None,
    auth_session: Any = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    hats_threshold: int = 100_000,
    estimated_size: int | None = None,
    fresh_after: float | None = None,
    min_free_gb: float | None = None,
) -> SyncStats:
    """Mirror ``src`` into the cache, incrementally. Returns :class:`SyncStats`.

    ``replica_roots``: after a complete sync, best-effort copies of the
    mirrored HATS tree are written onto each replica root (the union's
    ``--no-sync`` reader probes every root in order and uses the first
    surviving copy).  ``fresh_after`` (days): when the stored mirror
    completed a full sync less than this many days ago (and the HATS
    catalogue exists), skip the change-probe network round-trips entirely
    and trust the copy — the day-scale re-union pattern ("synced Monday,
    union Tuesday…") never re-pays the probe hours.  ``min_free_gb`` floors
    the free space on ``root`` before any fetch begins (default 10 GiB; env
    ``XMATCHER_MIN_FREE_GB`` or config ``cache.min_free_gb`` override).
    """
    from .storage import assert_headroom, default_cache_root

    root = cache_root or default_cache_root()
    cache = open_storage(root)
    stats = SyncStats()
    if src.access_method == "hats" and not _is_remote_hats(src):
        return stats  # already a local HATS catalogue
    if src.access_method not in ("hats", "tap"):
        raise CrossMatchError(
            f"Cannot sync access method '{src.access_method}'; only TAP and HATS are mirrorable."
        )

    min_free = _min_free_gb(min_free_gb)
    assert_headroom(root, min_free, f"cache root '{root}'")

    if fresh_after is not None and not force:
        name = _safe_name(src.name)
        prefix = f"{name}/{_version_dir(src)}"
        manifest_rel = (
            f"{prefix}/{_MANIFEST_NAME}"
            if src.access_method == "hats"
            else f"{name}/raw/{_MANIFEST_NAME}"
        )
        manifest = _read_json(cache, manifest_rel)
        fetched = manifest.get("fetched_at") if manifest else None
        if fetched and cache.exists(f"{prefix}/properties"):
            try:
                fetched_ts = time.mktime(time.strptime(fetched, "%Y-%m-%dT%H:%M:%SZ"))
            except (ValueError, OverflowError):
                fetched_ts = 0.0
            age_days = (time.time() - fetched_ts) / 86400.0
            if 0.0 <= age_days < fresh_after:
                logger.info(
                    "sync %s: mirror fetched %s (%.1f days ago, < fresh-after %.1f); "
                    "skipping the change probe",
                    src.name,
                    fetched,
                    age_days,
                    fresh_after,
                )
                return stats
    bucket = TokenBucket(rate_limit_rps) if rate_limit_rps else TokenBucket(0.0)
    _sync_with_failover(
        src,
        cache,
        bucket=bucket,
        force=force,
        workers=workers,
        page_size=page_size,
        hats_threshold=hats_threshold,
        estimated_size=estimated_size or DEFAULT_ESTIMATED_SIZE,
        auth_session=auth_session,
        progress_cb=progress_cb,
        stats=stats,
    )
    if stats.failed == 0:
        _replicate_to_roots(cache, src, list(replica_roots or []))
    return stats


def locate_mirrored(
    src: CatalogueSource,
    cache_root: str | None = None,
    cache_roots: list[str] | None = None,
) -> tuple[str, str]:
    """Return ``(storage_root, rel)`` of the mirrored HATS copy of ``src``.

    ``cache_roots`` (primary first) probes each root in order and returns the
    first one holding the copy — the union reader's fallback across
    replicated cache roots.  ``cache_root`` still wins when both are given
    (single-root callers are unchanged).  Raises :class:`CrossMatchError`
    when no mirror exists on any probed root (run ``xmatcher sync`` or drop
    ``--no-sync``).
    """
    from .storage import default_cache_root

    if cache_roots is None:
        cache_roots = [cache_root or default_cache_root()]
    if src.access_method == "hats" and not _is_remote_hats(src):
        path = src.path
        if path is None and src.access_identifier:
            path = Path(src.access_identifier)
        if path is None or not Path(path).is_dir():
            raise CrossMatchError(
                f"No local HATS copy for '{src.name}'; run 'xmatcher sync' first."
            )
        return str(path), ""
    if src.access_method not in ("hats", "tap"):
        raise CrossMatchError(
            f"Cannot mirror access method '{src.access_method}' for '{src.name}'; "
            "ray-union supports local HATS, remote HATS (http/vos:), and TAP inputs."
        )
    version = _version_dir(src)
    rel = f"{_safe_name(src.name)}/{version}"
    for root in cache_roots:
        cache = open_storage(root)
        if cache.exists(f"{rel}/properties") or cache.exists(
            f"{rel}/dataset/partition_info.parquet"
        ):
            return root, rel
    raise CrossMatchError(
        f"Catalogue '{src.name}' is not mirrored yet (probed: {', '.join(cache_roots)}); "
        f"run 'xmatcher sync {src.name}' or drop --no-sync."
    )


def ensure_mirrored(
    src: CatalogueSource,
    *,
    cache_root: str | None = None,
    replica_roots: list[str] | None = None,
    rate_limit_rps: float = 1.0,
    workers: int = 8,
    force: bool = False,
    progress_cb: Callable[[str], None] | None = None,
    auth_session: Any = None,
    hats_threshold: int = 100_000,
    estimated_size: int | None = None,
    fresh_after: float | None = None,
    min_free_gb: float | None = None,
) -> CatalogueSource:
    """Mirror a remote source and return a source pointing at the local HATS copy.

    Local HATS catalogues are returned unchanged; other local inputs (plain
    parquet/csv/fits) are converted once into the cache as HATS.
    ``replica_roots`` forwards to :func:`sync_catalogue` (best-effort
    replicated copies for the multi-root union reader).
    """
    from .storage import assert_headroom, default_cache_root

    root = cache_root or default_cache_root()
    assert_headroom(root, _min_free_gb(min_free_gb), f"cache root '{root}'")
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
            tmpdir = Path(tempfile.mkdtemp(prefix="xmatcher-hats-"))
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
        replica_roots=replica_roots,
        rate_limit_rps=rate_limit_rps,
        workers=workers,
        force=force,
        progress_cb=progress_cb,
        auth_session=auth_session,
        hats_threshold=hats_threshold,
        estimated_size=estimated_size,
        fresh_after=fresh_after,
    )
    if stats.failed:
        raise CrossMatchError(
            f"sync {src.name}: {stats.failed} file(s) failed to mirror; refusing to "
            "match over an incomplete catalogue (re-run to retry, or use "
            "'xmatcher sync --force' to pin down the failures)."
        )
    rel = f"{_safe_name(src.name)}/{_version_dir(src)}"
    return _mirrored_source(src, open_storage(root), rel, root)


def _mirrored_source(src: CatalogueSource, cache: Storage, rel: str, root: str) -> CatalogueSource:
    """Build the local-HATS ``CatalogueSource`` for a mirrored copy at ``rel``."""
    path = Path(cache.root) / rel if isinstance(cache, LocalStorage) else None
    return replace(
        src,
        is_local=False,
        access_method="hats",
        access_identifier=str(path) if path is not None else f"{root.rstrip('/')}/{rel}",
        path=path,
        hats_cache_rel=rel,
        hats_cache_root=root,
        _frame=None,
    )
