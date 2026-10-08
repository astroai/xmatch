"""Distributed N-survey full-outer-join (union) crossmatch on a Ray cluster.

Pipeline for the ``engine=ray-union`` route: every input catalogue is
mirrored (see :mod:`xmatcher.mirror`) into the xmatcher cache as a local HATS
catalogue; :func:`build_union_plan` builds a serialisable :class:`UnionPlan`;
Ray worker tasks then produce one row per match set

    centre row + (neighbour row | null) per catalogue after the centre

matching the sequential-union oracle: for a row set ``S`` the oracle's
positions come from the *lowest-indexed* catalogue in ``S`` (catalogue 1
when present, else the smallest member), so we enumerate star-shaped
combinations around every possible centre catalogue: subsets containing
catalogue 1 radiate from the hub row; subsets without it radiate from the
smallest member.  Each output row carries ``sep_arcsec`` = separation of the
closest used edge and a ``_src_cats`` membership string (``"1+2+3"``, or
``"k"`` for singles).  Catalogue 1 columns are unsuffixed; catalogue *k*
columns colliding with an earlier catalogue's names carry ``_k`` (mirroring
the matcher's right-side rename convention).

Chunking: each chunk owns exactly one centre partition (the template of a
HATS output partition).  Candidate rows of each catalogue with a larger
index are the partitions intersecting a cone of radius ``2*(sep + delta)``
degrees around the *centre partition pixel centre* (``sep`` converted
to degrees and enlarged by measured errors/motion; ``delta`` = largest
partition diagonal across all catalogues; the factor-2 margin keeps the
covering conservative, so every mate row lies inside its chunk's pool).
Every centre row belongs to exactly one chunk and its partners all live in
that chunk's pool, so every row set is enumerated exactly once.  Partitions
of catalogue *k* that no earlier catalogue's cone ever covers are emitted
as single-``k`` rows by a "rest" task instead of a centre chunk.

Per-row neighbourhoods are resolved with a scipy :class:`cKDTree` (dense
cone matrices would blow the memory guard); combos per centre row are
capped by ``max_tuples``.

Output: ``<out>/dataset/Norder=…/Dir=…/Npix=….parquet`` — one disjoint NESTED tiling
based on the hub footprint, plus
``dataset/partition_info.parquet``, ``properties`` and
``_metadata``/``_common_metadata``, readable via standard HATS
readers.  Chunk outputs go to ``<out>/chunks/<key>.parquet`` and are skipped
on resume when present.

Ray is imported lazily inside the task factories (repo convention: the
module must import without ray installed).
"""

from __future__ import annotations

import hashlib
import heapq
import itertools
import json
import logging
import math
import os
import shutil
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from . import matchers
from .exceptions import CrossMatchError
from .sources import CatalogueSource
from .storage import LocalStorage, Storage, open_storage

logger = logging.getLogger(__name__)

DEFAULT_TASK_ROWS = 2_000_000
DEFAULT_CHUNK_MEMORY_GB = 8.0
DEFAULT_MAX_TUPLES = 10_000
_BYTES_PER_ROW = (
    64.0  # ponytail: advisory estimate, not a hard memory cap; size wide rows separately
)
_HUB_BLOCK = 50_000  # centre rows per distance batch
_STATE_NAME = "resume.state"
_FIXED_COLS = {"sep_arcsec", "_src_cats", "_union_ra", "_union_dec"}

_LAST_PLAN: UnionPlan | None = None
"""Most recently built :class:`UnionPlan` (inspected by tests / ``doctor``)."""


def last_plan() -> UnionPlan | None:
    """Return the most recently built union plan (``None`` before any run)."""
    return _LAST_PLAN


# --------------------------------------------------------------------------- #
# plan data structures (picklable; ray.put(plan) once, tasks read it on the
# worker via the storage-root strings — no file objects travel)
# --------------------------------------------------------------------------- #
@dataclass
class PartitionPlan:
    """One HATS partition of one catalogue."""

    order: int  # HEALPix order (depth)
    pix: int  # nested pixel index at that order
    rel: str  # storage-relative path of the partition parquet file
    est_rows: int = 0  # footer row count when cheap, else hats_threshold


@dataclass
class CataloguePlan:
    """One mirrored catalogue in the union."""

    name: str
    root: str  # storage root ('vos:'…; local dir otherwise)
    rel: str  # HATS root rel inside that storage
    ra: str  # final (output) RA column name
    dec: str  # final (output) Dec column name
    partitions: list[PartitionPlan] = field(default_factory=list)
    cols: list[str] = field(default_factory=list)  # original column names
    final_cols: list[str] = field(default_factory=list)  # after _k suffixing
    dtypes: dict[str, pl.DataType] = field(default_factory=dict)  # final col -> complete dtype
    ra_err_column: str | None = None
    dec_err_column: str | None = None
    corr_column: str | None = None
    astrometric_covariance_columns: dict[str, str] | None = None
    pos_err_units: str = "arcsec"
    default_pos_error_arcsec: float | None = None
    epoch: float | None = None
    epoch_column: str | None = None
    pm_ra_column: str | None = None
    pm_dec_column: str | None = None
    parallax_column: str | None = None
    radial_velocity_column: str | None = None
    frame: str = "icrs"


@dataclass
class ChunkPlan:
    key: str  # e.g. 'c0-00000' (centre catalogue + partition index)
    center: int  # catalogue index of the centre
    center_idx: int  # partition index inside the centre catalogue
    cand_idx: dict[int, list[int]] = field(default_factory=dict)
    est_rows: int = 0  # pool estimate (rows) for the memory guard


@dataclass
class RestPlan:
    cat: int  # catalogue index
    part_idx: int  # partition index (never covered by an earlier cone)
    cand_idx: dict[int, list[int]] = field(default_factory=dict)
    # partner catalogues k > cat whose partitions intersect the cone around
    # this partition's pixel: the only places a rest row can have a mate.


@dataclass
class UnionPlan:
    catalogues: list[CataloguePlan]
    sep_arcsec: float
    delta_deg: float
    max_tuples: int
    col_names: list[str]  # final output column order (incl. sep_arcsec/_src_cats)
    out_dir: str
    chunks: list[ChunkPlan] = field(default_factory=list)
    rest: list[RestPlan] = field(default_factory=list)
    rest_keys: set = field(default_factory=set)  # {(cat, part_idx)} of rest runs
    matcher: str = "sky"
    max_error: float = 1.0
    target_epoch: float | None = None
    pm_prior: bool = False
    pm_prior_magnitude_column: str | None = None
    fallback_policy: str = "warn"


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def _cdshealpix():
    import cdshealpix  # noqa: PLC0415

    return cdshealpix


def _pixel_diagonal_deg(order: int, pix: int = 0) -> float:
    """Great-circle diameter of the cell's corners, degrees."""
    cds = _cdshealpix()
    lon, lat = cds.vertices(np.asarray([pix], dtype=np.uint64), int(order))
    lon_v = np.asarray(lon.value, dtype=float)[0]
    lat_v = np.asarray(lat.value, dtype=float)[0]
    cos_l = np.cos(lon_v)
    sin_l = np.sin(lon_v)
    cos_b = np.cos(lat_v)
    sin_b = np.sin(lat_v)
    v3 = np.stack([cos_b * cos_l, cos_b * sin_l, sin_b], axis=1)  # (4, 3)
    dots = v3 @ v3.T
    np.fill_diagonal(dots, 1.0)
    return float(np.degrees(np.max(np.arccos(np.clip(dots, -1.0, 1.0)))))


def _pixel_center_deg(order: int, pix: int) -> tuple[float, float]:
    cds = _cdshealpix()
    lon, lat = cds.healpix_to_lonlat(np.asarray([pix], dtype=np.int64), int(order))
    return float(np.degrees(np.asarray(lon.value)[0])), float(np.degrees(np.asarray(lat.value)[0]))


def _cone_ranges(order: int, pix: int, radius_deg: float, depth: int) -> list[tuple[int, int]]:
    """Half-open NESTED intervals at ``depth``; keep covered subtrees compressed."""
    cds = _cdshealpix()
    lon, lat = _pixel_center_deg(order, pix)
    from astropy import units as u  # noqa: PLC0415
    from astropy.coordinates import Latitude, Longitude  # noqa: PLC0415

    ipix, depths, _ = cds.cone_search(
        Longitude(float(lon), unit="deg"),
        Latitude(float(lat), unit="deg"),
        float(radius_deg) * u.deg,
        depth,
    )
    return [
        (int(px) << (2 * (depth - int(d))), (int(px) + 1) << (2 * (depth - int(d))))
        for px, d in zip(ipix, depths, strict=True)
    ]


# --------------------------------------------------------------------------- #
# plan builder
# --------------------------------------------------------------------------- #
def _footer_row_count(storage: Storage, rel: str) -> int | None:
    """Parquet footer row count (None when unavailable, e.g. remote)."""
    if isinstance(storage, LocalStorage):
        try:
            import pyarrow.parquet as pq  # noqa: PLC0415

            md = pq.read_metadata(Path(storage.root) / rel)
            return int(sum(md.row_group(r).num_rows for r in range(md.num_row_groups)))
        except Exception:  # noqa: BLE001
            return None
    return None


def _catalogue_root(src: CatalogueSource, cache_root: str | None) -> tuple[str, str]:
    """(storage root URI, HATS rel) for a local HATS catalogue.

    Mirrored sources (``hats_cache_rel`` set) live under a cache root —
    their own ``hats_cache_root`` when it is known (a replica copy found
    by the no-sync walk), else the cache root argument; plain local HATS
    dirs are opened with the catalogue dir as root.
    """
    if src.hats_cache_rel:
        from .storage import default_cache_root

        return (
            src.hats_cache_root or cache_root or default_cache_root(),
            src.hats_cache_rel,
        )
    return str(src.path or src.access_identifier or ""), ""


def _list_partitions(rel: str, storage: Storage) -> list[PartitionPlan]:
    """Locate the standard HATS layout: ``Norder=*/Dir=*/Npix=*.parquet``.

    Supports both ``<root>/Norder=…`` (flat) and ``<root>/dataset/Norder=…``
    (the layout `hats` tools write) top-level arrangements.
    """
    out: list[PartitionPlan] = []

    def walk_order(order_name: str, order_rel: str) -> None:
        if not order_name.startswith("Norder="):
            return
        try:
            order = int(order_name.split("=")[1].rstrip("/"))
        except ValueError:
            return
        for dir_name in sorted(storage.list(order_rel)):
            if not dir_name.startswith("Dir="):
                continue
            dir_rel = f"{order_rel}/{dir_name}"
            for pix_name in sorted(storage.list(dir_rel)):
                if not pix_name.startswith("Npix="):
                    continue
                pix = int(pix_name.split("=")[1].rstrip("/").split(".")[0])
                out.append(PartitionPlan(order=order, pix=pix, rel=f"{dir_rel}/{pix_name}"))

    root_entries = sorted(storage.list(rel) if rel else storage.list(""))
    bases: list[str] = []
    if any(e.startswith("Norder=") for e in root_entries):
        bases = [rel] if rel else [""]
    elif "dataset" in root_entries:
        bases = [f"{rel}/dataset" if rel else "dataset"]
    else:
        bases = [rel] if rel else [""]
    for base in bases:
        for order_name in sorted(storage.list(base)):
            walk_order(order_name, f"{base}/{order_name}" if base else order_name)
    return out


def _hats_dir(pix: int) -> int:
    """HATS/HiPS ``Dir=`` directory number for a nested pixel.

    The `hats` library derives the directory from the pixel number as
    ``(pixel // 10000) * 10000`` (``HealpixPixel.dir``) and *reconstructs* the
    file path from it, so a ``pix // 10000`` grouping writes partitions a
    conforming reader cannot find.
    """
    return (int(pix) // 10_000) * 10_000


def _catalogue_schema(storage: Storage, rel: str) -> dict[str, pl.DataType]:
    """Column name -> complete polars dtype ('{}' when unknown)."""
    try:
        return dict(storage.parquet_schema(rel))
    except Exception:  # noqa: BLE001 - schema is best-effort; empty means unknown
        return {}


def _est_rows(part: PartitionPlan, storage: Storage, fallback: int) -> int:
    if part.est_rows:
        return part.est_rows
    count = _footer_row_count(storage, part.rel)
    part.est_rows = count if count is not None else fallback
    return part.est_rows


def _partition_index(cat: CataloguePlan, depth: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sorted disjoint tile intervals at the catalogue's deepest order."""
    intervals = sorted(
        (p.pix << (2 * (depth - p.order)), (p.pix + 1) << (2 * (depth - p.order)), i)
        for i, p in enumerate(cat.partitions)
    )
    starts, ends, indices = np.asarray(intervals, dtype=np.int64).reshape(-1, 3).T
    if np.any(starts[1:] < ends[:-1]):
        raise CrossMatchError(f"Catalogue '{cat.name}' has overlapping HATS partitions")
    return starts, ends, indices


def _cone_candidate_idx(
    part: PartitionPlan,
    catalogues: Sequence[CataloguePlan],
    depths: Sequence[int],
    cone_radius: float,
    centre: int | None = None,
    pix_index: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
) -> dict[int, list[int]]:
    """Intersect compressed cone intervals with actual input tiles.

    Expanding a fully covered parent into every deepest-order child can
    exhaust memory on sparse, high-order catalogues. Binary searches on
    disjoint tile intervals preserve every intersecting tile without expansion.
    """
    cand: dict[int, list[int]] = {}
    for j, other in enumerate(catalogues):
        if j == centre:
            continue
        starts, ends, indices = (
            pix_index[j] if pix_index is not None else _partition_index(other, depths[j])
        )
        seen: set[int] = set()
        for lo, hi in _cone_ranges(part.order, part.pix, cone_radius, depths[j]):
            first = np.searchsorted(ends, lo, side="right")
            last = np.searchsorted(starts, hi, side="left")
            seen.update(indices[first:last].tolist())
        if seen:
            cand[j] = sorted(seen)
    return cand


def build_union_plan(
    sources: Sequence[CatalogueSource],
    *,
    sep_arcsec: float,
    hats_threshold: int,
    task_rows: int,
    chunk_memory_gb: float,
    max_tuples: int,
    out_dir: str,
    cache_root: str | None = None,
    matcher: str = "sky",
    max_error: float = 1.0,
    target_epoch: float | None = None,
    pm_prior: bool = False,
    pm_prior_magnitude_column: str | None = None,
    fallback_policy: str = "warn",
) -> UnionPlan:
    """Return a :class:`UnionPlan`; every source must be a local HATS dir."""
    if not sources:
        raise CrossMatchError("union needs at least one catalogue")
    if not math.isfinite(sep_arcsec) or sep_arcsec <= 0:
        raise CrossMatchError(f"radius_arcsec must be positive, got {sep_arcsec}")
    if matcher not in ("sky", "skyerr", "skyellipse"):
        raise CrossMatchError(
            f"engine='ray-union' supports matcher in ('sky', 'skyerr', 'skyellipse'), got matcher='{matcher}'."
        )
    for name, value in (
        ("max_tuples", max_tuples),
        ("task_rows", task_rows),
        ("hats_threshold", hats_threshold),
    ):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0 or int(value) != value:
            raise CrossMatchError(f"{name} must be a positive integer, got {value}")
    for name, limit in (("chunk_memory_gb", chunk_memory_gb), ("max_error", max_error)):
        if not math.isfinite(limit) or limit <= 0:
            raise CrossMatchError(f"{name} must be positive and finite, got {limit}")
    if target_epoch is not None and not math.isfinite(target_epoch):
        raise CrossMatchError("target_epoch must be finite")
    max_tuples = int(max_tuples)
    task_rows = int(task_rows)
    chunk_memory_gb = float(chunk_memory_gb)

    catalogues: list[CataloguePlan] = []
    used: set[str] = set(_FIXED_COLS)
    all_cols: list[str] = []
    max_delta = 0.0

    for ci, src in enumerate(sources):
        root, rel = _catalogue_root(src, cache_root)
        storage = open_storage(root)
        parts = _list_partitions(rel, storage)
        props = _read_hats_properties(storage, rel)
        if not parts:
            empty_info = False
            for name in (
                "dataset/partition_info.parquet",
                "partition_info.parquet",
                "partition_info.csv",
                "dataset/partition_info.csv",
            ):
                info_rel = f"{rel}/{name}" if rel else name
                if not storage.exists(info_rel):
                    continue
                if name.endswith(".parquet"):
                    info = storage.read_parquet(info_rel)
                else:
                    with tempfile.TemporaryDirectory(prefix="xmatcher-empty-") as scratch:
                        local = Path(scratch) / "partition_info.csv"
                        storage.stage_in(info_rel, local)
                        info = pl.read_csv(local)
                empty_info = info.height == 0 and {"Norder", "Npix"} <= set(info.columns)
                break
            if props.get("hats_nrows") != "0" or not empty_info:
                raise CrossMatchError(
                    f"Catalogue '{src.name}' has no HATS partitions under {root!r}/{rel!r}."
                )
        # RING-ordered copies (remote mirrors preserve the source's own
        # hats_ordering property) are converted to NESTED in the plan
        # geometry: every downstream pixel computation (_pixel_center_deg,
        # _cone_ranges, the tile interval shifts, the rest
        # tiling) is NESTED, and the output is always NESTED hub tiling.
        ordering = next((v for k, v in props.items() if k.lower() == "hats_ordering"), "")
        if ordering.upper() == "RING":
            cds = _cdshealpix()
            for part in parts:
                part.pix = int(
                    cds.from_ring(np.asarray([part.pix], dtype=np.uint64), part.order)[0]
                )
            logger.info(
                "catalogue '%s': hats_ordering=RING — converted %d partition pixel(s) to NESTED",
                src.name,
                len(parts),
            )
        parts.sort(key=lambda p: (p.order, p.pix))
        for part in parts:
            _est_rows(part, storage, hats_threshold)
            max_delta = max(max_delta, _pixel_diagonal_deg(part.order, part.pix))
        schema_rel = (
            parts[0].rel
            if parts
            else (f"{rel}/dataset/_common_metadata" if rel else "dataset/_common_metadata")
        )
        schema = _catalogue_schema(storage, schema_rel)
        if not parts and not schema:
            schema_rel = f"{rel}/_common_metadata" if rel else "_common_metadata"
            schema = _catalogue_schema(storage, schema_rel)
        if "file_loc" in schema:  # partition_info-style file is not data
            schema = {}
        cols = list(schema)
        if not cols:
            # Without the partition schema the plan cannot name this
            # catalogue's output columns (or type its all-null blocks), and
            # every chunk would fail hours later.  Fail here instead, with the
            # root named — non-local roots (e.g. `vos:`) need the schema read
            # through Storage.parquet_schema.
            raise CrossMatchError(
                f"Cannot read the HATS schema of catalogue '{src.name}' "
                f"({root!r}/{schema_rel!r}); the union plan needs each catalogue's "
                "column list before it starts. Check the cache copy is readable."
            )
        final_cols: list[str] = []
        for col in cols:
            while col in used:
                col = f"{col}_{ci + 1}"
            used.add(col)
            final_cols.append(col)
        all_cols.extend(final_cols)

        dtypes: dict[str, pl.DataType] = {
            final: schema[orig] for orig, final in zip(cols, final_cols, strict=True)
        }
        col_map = dict(zip(cols, final_cols, strict=True))

        def _map_col(c: str | None, cmap: dict[str, str] = col_map) -> str | None:
            if c is None:
                return None
            return cmap.get(c, c)

        cov_cols = (
            {k: col_map.get(v, v) for k, v in src.astrometric_covariance_columns.items()}
            if src.astrometric_covariance_columns
            else None
        )

        ra_orig = src.ra_column or "ra"
        dec_orig = src.dec_column or "dec"
        ra_final = final_cols[cols.index(ra_orig)] if ra_orig in cols else ra_orig
        dec_final = final_cols[cols.index(dec_orig)] if dec_orig in cols else dec_orig

        catalogues.append(
            CataloguePlan(
                name=src.name,
                root=str(storage.root) if isinstance(storage, LocalStorage) else root,
                rel=rel,
                ra=ra_final,
                dec=dec_final,
                partitions=parts,
                cols=cols,
                final_cols=final_cols,
                dtypes=dtypes,
                ra_err_column=_map_col(src.ra_err_column),
                dec_err_column=_map_col(src.dec_err_column),
                corr_column=_map_col(src.corr_column),
                astrometric_covariance_columns=cov_cols,
                pos_err_units=src.pos_err_units or "arcsec",
                default_pos_error_arcsec=src.default_pos_error_arcsec,
                epoch=src.epoch,
                epoch_column=_map_col(src.epoch_column),
                pm_ra_column=_map_col(src.pm_ra_column),
                pm_dec_column=_map_col(src.pm_dec_column),
                parallax_column=_map_col(src.parallax_column),
                radial_velocity_column=_map_col(src.radial_velocity_column),
                frame=src.frame or "icrs",
            )
        )

    all_cols += ["sep_arcsec", "_src_cats", "_union_ra", "_union_dec"]
    eff_arcsec = float(sep_arcsec)
    max_motion_deg = 0.0
    if matcher != "sky" or target_epoch is not None:
        # Scan one partition at a time: fixed error/motion margins silently
        # miss high-uncertainty or high-proper-motion sources. Use the same
        # epoch propagation and covariance extraction as the workers.
        bounds_plan = UnionPlan(
            catalogues,
            sep_arcsec,
            max_delta,
            max_tuples,
            all_cols,
            out_dir,
            matcher=matcher,
            max_error=float(max_error),
            target_epoch=target_epoch,
            pm_prior=pm_prior,
            pm_prior_magnitude_column=pm_prior_magnitude_column,
            fallback_policy=fallback_policy,
        )
        max_sigma = 0.0
        max_ra_var = max_dec_var = 0.0
        for cat in catalogues:
            for pi in range(len(cat.partitions)):
                original = _cat_frame(cat, [pi])
                aligned = (
                    _cat_frame(cat, [pi], bounds_plan) if target_epoch is not None else original
                )
                if not aligned.height:
                    continue
                if target_epoch is not None:
                    before = matchers._radec_to_xyz(
                        original[cat.ra].to_numpy(), original[cat.dec].to_numpy()
                    )
                    after = matchers._radec_to_xyz(
                        aligned[cat.ra].to_numpy(), aligned[cat.dec].to_numpy()
                    )
                    motion = matchers._chord_to_arcsec(np.linalg.norm(after - before, axis=1))
                    max_motion_deg = max(max_motion_deg, float(np.max(motion)) / 3600.0)
                err = _extract_matcher_err(aligned, cat, matcher)
                if matcher == "skyerr":
                    max_sigma = max(max_sigma, float(np.nanmax(err)))
                elif matcher == "skyellipse":
                    max_ra_var = max(max_ra_var, float(np.nanmax(err[0])))
                    max_dec_var = max(max_dec_var, float(np.nanmax(err[1])))
        if matcher == "skyerr":
            eff_arcsec = max(eff_arcsec, 2.0 * float(max_error) * max_sigma)
        elif matcher == "skyellipse":
            eff_arcsec = max(
                eff_arcsec, float(max_error) * math.sqrt(2.0 * (max_ra_var + max_dec_var))
            )
    sep_deg = eff_arcsec / 3600.0
    cone_radius = min(180.0, 2.0 * (sep_deg + max_delta + max_motion_deg))
    depths = [max((p.order for p in c.partitions), default=0) for c in catalogues]

    chunks: list[ChunkPlan] = []
    covered: list[list[int]] = [[] for _ in catalogues]
    # Open each catalogue's storage once and build each pixel->partition index
    # once: both used to be rebuilt inside the per-partition (O(P)) loops.
    storages = [open_storage(cat.root) for cat in catalogues]
    pix_index = [_partition_index(cat, depths[j]) for j, cat in enumerate(catalogues)]

    for ci, cat in enumerate(catalogues):
        cat_storage = storages[ci]
        for pi, part in enumerate(cat.partitions):
            extra = _est_rows(part, cat_storage, hats_threshold)
            cand = _cone_candidate_idx(
                part, catalogues, depths, cone_radius, centre=ci, pix_index=pix_index
            )
            for j, seen in cand.items():
                extra += sum(
                    _est_rows(catalogues[j].partitions[i], storages[j], hats_threshold)
                    for i in seen
                )
                if j > ci:
                    # only higher catalogues can be *covered* (rest-tracked)
                    covered[j].extend(seen)

            if extra > task_rows:
                raise CrossMatchError(
                    f"Catalogue '{cat.name}' partition {part.rel} drives a pool of ~{extra} rows "
                    f"— exceeds --task-rows={task_rows}; raise --task-rows or shrink --hats-threshold."
                )
            mem = extra * _BYTES_PER_ROW / 1024**3
            if mem > chunk_memory_gb:
                raise CrossMatchError(
                    f"pool of ~{extra} rows (~{mem:,.2f} GiB at {_BYTES_PER_ROW:g} B/row) "
                    f"exceeds --chunk-memory-gb={chunk_memory_gb:g}; raise it or shrink --hats-threshold."
                )
            chunks.append(
                ChunkPlan(
                    key=f"c{ci}-{pi:05d}",
                    center=ci,
                    center_idx=pi,
                    cand_idx=cand,
                    est_rows=extra,
                )
            )

    rest: list[RestPlan] = []
    for ci in range(1, len(catalogues)):
        covered_set = set(covered[ci])
        for pi in range(len(catalogues[ci].partitions)):
            if pi not in covered_set:
                part = catalogues[ci].partitions[pi]
                cand = _cone_candidate_idx(
                    part, catalogues, depths, cone_radius, centre=ci, pix_index=pix_index
                )
                # lower-indexed mates are provably absent (the coverage proof);
                # only higher catalogues can hold a mate of a rest row.
                rest.append(
                    RestPlan(
                        cat=ci, part_idx=pi, cand_idx={k: v for k, v in cand.items() if k > ci}
                    )
                )

    return UnionPlan(
        catalogues=catalogues,
        sep_arcsec=sep_arcsec,
        delta_deg=max_delta,
        max_tuples=max_tuples,
        col_names=all_cols,
        out_dir=out_dir,
        chunks=chunks,
        rest=rest,
        rest_keys={(r.cat, r.part_idx) for r in rest},
        matcher=matcher,
        max_error=float(max_error),
        target_epoch=float(target_epoch) if target_epoch is not None else None,
        pm_prior=bool(pm_prior),
        pm_prior_magnitude_column=pm_prior_magnitude_column,
        fallback_policy=fallback_policy,
    )


# --------------------------------------------------------------------------- #
# task internals (run inside Ray workers; the plan arrives via ray.put)
# --------------------------------------------------------------------------- #
def _part_frame(cat: CataloguePlan, part: PartitionPlan) -> pl.DataFrame:
    storage = open_storage(cat.root)
    return storage.read_parquet(part.rel)


def _check_coords(frame: pl.DataFrame, ra: str, dec: str, what: str) -> None:
    """Reject non-finite or out-of-range coordinates before they reach
    scipy's cKDTree (raises ValueError) or the cdshealpix Rust core (panics
    with pyo3_runtime.PanicException)."""
    if not frame.height:
        return
    ra_v = frame[ra].cast(pl.Float64).to_numpy()
    dec_v = frame[dec].cast(pl.Float64).to_numpy()
    bad = ~(np.isfinite(ra_v) & np.isfinite(dec_v) & (dec_v >= -90.0) & (dec_v <= 90.0))
    if bad.any():
        first = int(np.flatnonzero(bad)[0])
        raise CrossMatchError(
            f"{int(bad.sum())} rows of {what} have non-finite or out-of-range "
            f"coordinates (first at row {first}); clean the inputs before matching"
        )


def _plan_source(cat: CataloguePlan) -> CatalogueSource:
    """Reconstruct a ``CatalogueSource`` matching the renamed partition columns."""
    return CatalogueSource(
        name=cat.name,
        is_local=True,
        ra_column=cat.ra,
        dec_column=cat.dec,
        ra_err_column=cat.ra_err_column,
        dec_err_column=cat.dec_err_column,
        corr_column=cat.corr_column,
        astrometric_covariance_columns=cat.astrometric_covariance_columns,
        pos_err_units=cat.pos_err_units,
        default_pos_error_arcsec=cat.default_pos_error_arcsec,
        epoch=cat.epoch,
        epoch_column=cat.epoch_column,
        pm_ra_column=cat.pm_ra_column,
        pm_dec_column=cat.pm_dec_column,
        parallax_column=cat.parallax_column,
        radial_velocity_column=cat.radial_velocity_column,
        frame=cat.frame,
    )


def _cat_frame(
    cat: CataloguePlan, idxs: Sequence[int], plan: UnionPlan | None = None
) -> pl.DataFrame:
    """Rows of the given partitions, renamed to final output column names."""
    frames = [_part_frame(cat, cat.partitions[i]) for i in idxs]
    # 0-row partitions read fine (polars keeps the schema); dropping them
    # here would lose the column names and crash the rename below.
    frames = [f for f in frames if f is not None]
    frame = (
        pl.concat(frames, how="diagonal_relaxed")
        if len(frames) > 1
        else (frames[0] if frames else pl.DataFrame())
    )
    rename = dict(zip(cat.cols, cat.final_cols, strict=True))
    if rename:
        frame = frame.rename(rename)
    _check_coords(frame, cat.ra, cat.dec, f"catalogue '{cat.name}'")
    if plan is not None and plan.target_epoch is not None and frame.height > 0:
        src_meta = _plan_source(cat)
        target_epoch = float(plan.target_epoch)
        if plan.matcher == "skyerr":
            from .out_of_core import _align_epoch, _validate_sigma  # noqa: PLC0415

            frame = _align_epoch(
                frame.lazy(),
                src_meta,
                target_epoch,
                propagate_covariance=True,
                pm_prior=plan.pm_prior,
                magnitude_column=plan.pm_prior_magnitude_column,
            ).collect()
            _validate_sigma(frame.lazy(), matchers._EPOCH_SIGMA, src_meta)
        else:
            empty_src = CatalogueSource(
                name="_empty", is_local=True, ra_column="_ra", dec_column="_dec"
            )
            empty_df = pl.DataFrame(
                {"_ra": pl.Series([], dtype=pl.Float64), "_dec": pl.Series([], dtype=pl.Float64)}
            )
            frame, _ = matchers._apply_proper_motion(
                frame,
                empty_df,
                src_meta,
                empty_src,
                target_epoch,
                propagate_covariance=plan.matcher == "skyellipse",
                fallback_policy=plan.fallback_policy,
            )
            if plan.pm_prior:
                frame, _, _, _ = matchers._apply_pm_drift_prior(
                    src_meta,
                    empty_src,
                    frame,
                    empty_df,
                    target_epoch,
                    magnitude_column=plan.pm_prior_magnitude_column,
                )
    return frame


def _extract_matcher_err(frame: pl.DataFrame, cat: CataloguePlan, matcher: str) -> Any:
    """Extract per-row positional sigma (``skyerr``) or covariance tuple (``skyellipse``)."""
    if frame.height == 0 or matcher == "sky":
        return None
    src_meta = _plan_source(cat)
    if matcher == "skyerr":
        sig = matchers._pos_sigma_arcsec(frame, src_meta)
        if sig is None:
            raise CrossMatchError(
                f"matcher='skyerr' requires positional errors on catalogue '{cat.name}'"
            )
        return sig
    if matcher == "skyellipse":
        cov = matchers._pos_covariance(frame, src_meta)
        if cov is None:
            raise CrossMatchError(
                f"matcher='skyellipse' requires positional errors on catalogue '{cat.name}'"
            )
        return cov
    return None


def _effective_pair_chord(
    plan: UnionPlan,
    centre_err: Any,
    pool_errs: Sequence[Any],
) -> float:
    """Spatial upper-bound chord for cKDTree pre-filtering in chunk/rest workers."""
    base_chord = matchers._arcsec_to_chord(float(plan.sep_arcsec))
    if plan.matcher == "skyerr" and centre_err is not None:
        c_max = float(np.nanmax(centre_err)) if len(centre_err) else 0.0
        p_max = max(
            (float(np.nanmax(pe)) for pe in pool_errs if pe is not None and len(pe)),
            default=0.0,
        )
        bound_arcsec = float(plan.max_error) * (c_max + p_max)
        if np.isfinite(bound_arcsec) and bound_arcsec > 0.0:
            return max(base_chord, matchers._arcsec_to_chord(bound_arcsec))
    elif plan.matcher == "skyellipse" and centre_err is not None:
        chords = [base_chord]
        for pe in pool_errs:
            if pe is not None and len(pe[0]):
                chords.append(
                    matchers._skyellipse_search_chord_max(centre_err, pe, float(plan.max_error))
                )
        return max(chords)
    return base_chord


def _gather_rows(frame: pl.DataFrame, idx: np.ndarray) -> pl.DataFrame:
    """Frame rows selected by ``idx`` (-1 → all-null row)."""
    if not len(idx):
        return pl.DataFrame()
    if not frame.height:
        return frame
    safe = np.clip(idx, 0, frame.height - 1).astype(np.int64)
    out = frame.gather(pl.Series(safe))
    mask = idx < 0
    if mask.any():
        mask_s = pl.Series(mask, dtype=pl.Boolean)
        out = out.with_columns(
            [pl.when(mask_s).then(None).otherwise(pl.col(c)).alias(c) for c in out.columns]
        )
    return out


def _block_combos(
    centre_ra: np.ndarray,
    centre_dec: np.ndarray,
    centre_row: np.ndarray,  # global centre row ids (same length)
    pools: Sequence[tuple[np.ndarray, np.ndarray]],
    sep_chord: float,
    max_tuples: int,
    centre_label: int,
    cat_labels: Sequence[int],
    centre_cat: int = 0,
    drop_singles: bool = False,
    matcher: str = "sky",
    max_error: float = 1.0,
    centre_err: Any = None,
    pool_errs: Sequence[Any] | None = None,
) -> tuple[dict[int, np.ndarray], np.ndarray, list[str], np.ndarray]:
    """Enumerate star-shaped row sets for one block of centre rows.

    ``pools[k]`` holds (ra, dec) of candidate rows of catalogue ``k``.
    Returns (cat_sel, sep_arcsec, srcs, centre_ids):
      * cat_sel[k] — per-output-row id in pool k (-1 = absent);
      * sep_arcsec — min separation over used edges (NaN = centre only);
      * srcs[i] — membership string like ``"1+3"``;
      * centre_ids — per-output-row centre row id (from ``centre_row``).

    A row set's positions come from the lowest-indexed catalogue in the set,
    so combos selecting a partner from a catalogue *lower*-indexed than the
    centre are dropped (their canonical emission happens in that lower
    centre's chunk), and the centre-only single is dropped for every row
    that has ANY mate — the sequential-union oracle never keeps a singleton
    for a matched row.  ``drop_singles`` (rest partitions) drops all
    centre-only combos: the rest task emits those rows instead.
    """
    from scipy.spatial import cKDTree  # noqa: PLC0415

    n_partner = len(pools)
    sep_chord = float(sep_chord)
    h_xyz = matchers._radec_to_xyz(centre_ra, centre_dec)

    trees: list[Any | None] = []
    for k in range(n_partner):
        ra, dec = pools[k]
        trees.append(cKDTree(matchers._radec_to_xyz(ra, dec)) if len(ra) else None)

    cat_sel: dict[int, list[int]] = {k: [] for k in range(n_partner)}
    centre_out: list[int] = []
    sep_out: list[float] = []
    src_out: list[str] = []

    for bi in range(len(centre_ra)):
        nbrs: list[tuple[np.ndarray, np.ndarray]] = []  # (sorted idx, sep)
        for k in range(n_partner):
            tree = trees[k]
            if tree is None:
                nbrs.append((np.array([], dtype=np.int64), np.array([], dtype=float)))
                continue
            hit = tree.query_ball_point(h_xyz[bi], sep_chord)
            idx = np.sort(np.asarray(hit, dtype=np.int64))
            if idx.size:
                ra, dec = pools[k]
                xyz_c = matchers._radec_to_xyz(ra[idx], dec[idx])
                chord = np.linalg.norm(xyz_c - h_xyz[bi], axis=1)
                seps_k = matchers._chord_to_arcsec(chord)
                if (
                    matcher == "skyerr"
                    and centre_err is not None
                    and pool_errs is not None
                    and pool_errs[k] is not None
                ):
                    c_sig = float(centre_err[bi])
                    p_sig = np.asarray(pool_errs[k], dtype=float)[idx]
                    limit = float(max_error) * (c_sig + p_sig)
                    keep_k = np.isfinite(limit) & np.isfinite(seps_k) & (seps_k <= limit)
                    idx = idx[keep_k]
                    seps_k = seps_k[keep_k]
                elif (
                    matcher == "skyellipse"
                    and centre_err is not None
                    and pool_errs is not None
                    and pool_errs[k] is not None
                ):
                    sra2_c, sde2_c, rho_c = centre_err
                    sra2_p, sde2_p, rho_p = pool_errs[k]
                    mean_dec = 0.5 * (float(centre_dec[bi]) + dec[idx])
                    cos_dec = np.cos(np.radians(mean_dec))
                    delta_ra = (
                        ((float(centre_ra[bi]) - ra[idx] + 180.0) % 360.0 - 180.0)
                        * 3600.0
                        * cos_dec
                    )
                    delta_dec = (float(centre_dec[bi]) - dec[idx]) * 3600.0
                    d2 = matchers._mahalanobis_pairwise(
                        delta_ra,
                        delta_dec,
                        np.full(idx.size, float(sra2_c[bi])),
                        np.full(idx.size, float(sde2_c[bi])),
                        np.full(idx.size, float(rho_c[bi])),
                        sra2_p[idx],
                        sde2_p[idx],
                        rho_p[idx],
                    )
                    keep_k = np.isfinite(d2) & (d2 <= float(max_error) ** 2)
                    idx = idx[keep_k]
                    seps_k = seps_k[keep_k]
                nbrs.append((idx, seps_k))
            else:
                nbrs.append((idx, np.array([], dtype=float)))

        # ---- row-level ownership gate ----------------------------------------
        # A centre row of a non-hub catalogue that also has a LOWER-indexed
        # mate is owned by that lower centre (which feeds its higher pools):
        # skip the row here entirely — otherwise its {j, k} row sets would
        # also star from this chunk and the row would participate twice.
        # Rest rows provably have no lower-indexed mates (the cone-coverage
        # proof), so this only gates non-rest centre chunks; hub chunks
        # (centre_cat == 0) never gate.
        if any(cat_labels[k] - 1 < centre_cat and nbrs[k][0].size for k in range(n_partner)):
            continue

        # ---- combos: (candidate | null) per partner --------------------------
        lists: list[np.ndarray] = []
        for k in range(n_partner):
            idx_k = nbrs[k][0]
            lists.append(
                np.concatenate([np.array([-1], dtype=np.int64), idx_k])
                if idx_k.size
                else np.array([-1], dtype=np.int64)
            )
        total = math.prod(len(each) for each in lists)

        def sep_of(
            combo: tuple[int, ...], nbrs: list[tuple[np.ndarray, np.ndarray]] = nbrs
        ) -> float | None:
            vals = []
            for k in range(n_partner):
                if combo[k] >= 0:
                    pos = int(np.searchsorted(nbrs[k][0], combo[k]))
                    vals.append(float(nbrs[k][1][pos]))
            return min(vals) if vals else None  # None = centre-only row

        always: list[tuple[int, ...]] = []
        if total <= max_tuples:
            for combo in itertools.product(*[each.tolist() for each in lists]):
                if all(c < 0 for c in combo):
                    if drop_singles:
                        continue  # the rest task owns this partition's singles
                    if any(nbrs[k][0].size for k in range(n_partner)):
                        continue  # row has a mate: the oracle keeps no singleton
                    always.append(combo)  # unmatched centre-only rows survive
                    continue
                if any(cat_labels[k] - 1 < centre_cat for k, c in enumerate(combo) if c >= 0):
                    continue  # canonical emission lives in a lower-indexed centre
                sep = sep_of(combo)
                if sep is None:
                    continue
                always.append(combo)
            ordered: list[tuple[int, ...]] = always
        else:
            # The score is the smallest used edge. Sorted axes (null last,
            # with score infinity) make it monotone on the product grid.
            # Best-first traversal returns the exact cap without sampling or
            # materialising the complete product; at most O(cap * N) states.
            sorted_lists = []
            sorted_seps = []
            for idx_k, seps_k in nbrs:
                perm = np.lexsort((idx_k, seps_k))
                sorted_lists.append(np.concatenate([idx_k[perm], [-1]]))
                sorted_seps.append(np.concatenate([seps_k[perm], [np.inf]]))

            def state(pos, sorted_lists=sorted_lists, sorted_seps=sorted_seps):
                combo = tuple(int(sorted_lists[k][p]) for k, p in enumerate(pos))
                score = min(float(sorted_seps[k][p]) for k, p in enumerate(pos))
                return score, combo, pos

            origin = (0,) * n_partner
            frontier = [state(origin)]
            visited = {origin}
            ordered = []
            while frontier and len(ordered) < max_tuples:
                score, combo, pos = heapq.heappop(frontier)
                if np.isfinite(score):
                    ordered.append(combo)
                for k in range(n_partner):
                    if pos[k] + 1 >= len(sorted_lists[k]):
                        continue
                    neighbor = pos[:k] + (pos[k] + 1,) + pos[k + 1 :]
                    if neighbor not in visited:
                        visited.add(neighbor)
                        heapq.heappush(frontier, state(neighbor))

        for combo in ordered:
            centre_out.append(int(centre_row[bi]))
            for k in range(n_partner):
                cat_sel[k].append(int(combo[k]))
            sep = sep_of(combo)
            labels = [centre_label] + [
                int(cat_labels[k]) for k in range(n_partner) if combo[k] >= 0
            ]
            joined = "+".join(map(str, labels))
            src_out.append(joined)
            sep_out.append(sep if sep is not None else float("nan"))

    ksorted = sorted(cat_sel)
    return (
        {k: np.asarray(cat_sel[k], dtype=np.int64) for k in ksorted},
        np.asarray(sep_out, dtype=float),
        src_out,
        np.asarray(centre_out, dtype=np.int64),
    )


def _null_cat_frame(cat: CataloguePlan, m: int) -> pl.DataFrame:
    """Frame of ``m`` all-null rows with this catalogue's final columns."""
    return pl.DataFrame(
        [
            pl.Series(col, [None] * m, dtype=cat.dtypes.get(col), strict=False)
            for col in cat.final_cols
        ]
    )


def _assemble_block(
    plan: UnionPlan,
    centre: int,
    centre_frame: pl.DataFrame,
    cat_frames: dict[int, pl.DataFrame],
    cat_sel: dict[int, np.ndarray],
    seps: np.ndarray,
    srcs: list[str],
    centre_ids: np.ndarray,
) -> pl.DataFrame:
    """Horizontal slice of the output rows for one combination batch.

    Every catalogue contributes one column block: its rows selected by the
    per-row ids (``-1`` → all-null), or all-null when the catalogue is not
    part of the row set.
    """
    m = len(seps)
    tables: list[pl.DataFrame] = []
    for k in range(len(plan.catalogues)):
        if k == centre:
            tables.append(_gather_rows(centre_frame, centre_ids))
        else:
            idx = cat_sel.get(k, np.full(m, -1, dtype=np.int64))
            frame = cat_frames.get(k)
            if frame is None or frame.height == 0:
                frame = _null_cat_frame(plan.catalogues[k], m)
            tables.append(_gather_rows(frame, idx))
    tables.append(pl.Series("sep_arcsec", seps, dtype=pl.Float64).to_frame())
    tables.append(pl.Series("_src_cats", srcs, dtype=pl.String).to_frame())
    return (
        tables[0]
        .hstack([column for frame in tables[1:] for column in frame.get_columns()])
        .with_columns(
            pl.coalesce([pl.col(cat.ra) for cat in plan.catalogues])
            .cast(pl.Float64)
            .alias("_union_ra"),
            pl.coalesce([pl.col(cat.dec) for cat in plan.catalogues])
            .cast(pl.Float64)
            .alias("_union_dec"),
        )
    )


def _empty_union(plan: UnionPlan) -> pl.DataFrame:
    schema: dict[str, pl.DataType | type[pl.DataType]] = {
        col: dtype for cat in plan.catalogues for col, dtype in cat.dtypes.items()
    }
    schema.update(
        sep_arcsec=pl.Float64, _src_cats=pl.String, _union_ra=pl.Float64, _union_dec=pl.Float64
    )
    return pl.DataFrame(schema=schema)


def _run_chunk(plan: UnionPlan, chunk: ChunkPlan) -> dict[str, Any]:
    """Execute one chunk: rows radiating from its centre partition."""
    centre = plan.catalogues[chunk.center]
    centre_frame = _cat_frame(centre, [chunk.center_idx], plan)
    centre_err = _extract_matcher_err(centre_frame, centre, plan.matcher)
    if centre_frame.columns and set(centre.final_cols).issubset(centre_frame.columns):
        centre_frame = centre_frame.select(centre.final_cols)
    n = centre_frame.height
    centre_ra = centre_frame[centre.ra].to_numpy() if n else np.zeros(0, dtype=float)
    centre_dec = centre_frame[centre.dec].to_numpy() if n else np.zeros(0, dtype=float)

    partner_globals = sorted(chunk.cand_idx)
    cat_frames: dict[int, pl.DataFrame] = {}
    pools: list[tuple[np.ndarray, np.ndarray]] = []
    pool_errs: list[Any] = []
    labels: list[int] = []
    for k in partner_globals:
        cat = plan.catalogues[k]
        frame = _cat_frame(cat, chunk.cand_idx[k], plan)
        pool_errs.append(_extract_matcher_err(frame, cat, plan.matcher))
        if frame.columns and set(cat.final_cols).issubset(frame.columns):
            frame = frame.select(cat.final_cols)
        cat_frames[k] = frame
        pools.append(
            (
                frame[cat.ra].to_numpy() if frame.height else np.zeros(0, dtype=float),
                frame[cat.dec].to_numpy() if frame.height else np.zeros(0, dtype=float),
            )
        )
        labels.append(k + 1)

    sep_chord = _effective_pair_chord(plan, centre_err, pool_errs)
    drop_singles = (chunk.center, chunk.center_idx) in plan.rest_keys
    blocks: list[pl.DataFrame] = []
    for b0 in range(0, n, _HUB_BLOCK):
        b1 = min(b0 + _HUB_BLOCK, n)
        block_centre_err: Any = None
        if centre_err is not None:
            if plan.matcher == "skyerr":
                block_centre_err = centre_err[b0:b1]
            elif plan.matcher == "skyellipse":
                block_centre_err = (
                    centre_err[0][b0:b1],
                    centre_err[1][b0:b1],
                    centre_err[2][b0:b1],
                )
        cat_sel, seps, srcs, centre_ids = _block_combos(
            centre_ra[b0:b1],
            centre_dec[b0:b1],
            np.arange(b0, b1, dtype=np.int64),
            pools,
            sep_chord,
            plan.max_tuples,
            centre_label=chunk.center + 1,
            cat_labels=labels,
            centre_cat=chunk.center,
            drop_singles=drop_singles,
            matcher=plan.matcher,
            max_error=plan.max_error,
            centre_err=block_centre_err,
            pool_errs=pool_errs,
        )
        if not len(seps):
            continue
        sel_global = {partner_globals[kk]: cat_sel[kk] for kk in range(len(partner_globals))}
        blocks.append(
            _assemble_block(
                plan, chunk.center, centre_frame, cat_frames, sel_global, seps, srcs, centre_ids
            )
        )

    table = pl.concat(blocks, how="diagonal_relaxed") if blocks else _empty_union(plan)
    dest = Path(plan.out_dir) / "chunks" / f"{chunk.key}.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_parquet(table, dest)
    return {"key": chunk.key, "rows": table.height}


def _atomic_write_parquet(table: pl.DataFrame, dest: Path) -> None:
    """Write ``table`` to ``dest`` via a temp file + rename (crash-safe resume)."""
    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.tmp")
    try:
        table.write_parquet(tmp)
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


def _run_rest(plan: UnionPlan, rest: RestPlan) -> dict[str, Any]:
    """Rows of a catalogue partition no earlier cone covers: single-``k`` rows.

    Every row is emitted inside the *hub's* partition that contains it (the
    output stays a single-tiling HATS tree); rows outside the hub's footprint
    become fine-order partitions of their own, provably disjoint from every
    hub cell.  Lower-indexed mates are impossible (the coverage proof), but a
    rest row CAN pair with a *higher*-indexed catalogue — those ``{j, k}``
    sets are emitted by the centre chunk of this partition, so rows with a
    mate are dropped here.
    """
    cat = plan.catalogues[rest.cat]
    frame = _cat_frame(cat, [rest.part_idx], plan)
    centre_err = _extract_matcher_err(frame, cat, plan.matcher)
    if frame.columns and set(cat.final_cols).issubset(frame.columns):
        frame = frame.select(cat.final_cols)
    m = frame.height
    dest_root = Path(plan.out_dir) / "chunks"
    dest_root.mkdir(parents=True, exist_ok=True)
    dest = dest_root / f"rest-{rest.cat}-{rest.part_idx:05d}.parquet"
    if not m:
        return {"key": f"rest-{rest.cat}-{rest.part_idx:05d}", "rows": 0}

    ra_full = frame[cat.ra].to_numpy() if m else np.zeros(0, dtype=float)
    dec_full = frame[cat.dec].to_numpy() if m else np.zeros(0, dtype=float)

    partner_globals = sorted(rest.cand_idx)
    pools: list[tuple[np.ndarray, np.ndarray]] = []
    pool_errs: list[Any] = []
    for k in partner_globals:
        other = plan.catalogues[k]
        pf = _cat_frame(other, rest.cand_idx[k], plan)
        pool_errs.append(_extract_matcher_err(pf, other, plan.matcher))
        pools.append(
            (
                pf[other.ra].to_numpy() if pf.height else np.zeros(0, dtype=float),
                pf[other.dec].to_numpy() if pf.height else np.zeros(0, dtype=float),
            )
        )

    sep_chord = _effective_pair_chord(plan, centre_err, pool_errs)
    from scipy.spatial import cKDTree  # noqa: PLC0415

    keep = np.ones(m, dtype=bool)
    for b0 in range(0, m, _HUB_BLOCK):
        b1 = min(b0 + _HUB_BLOCK, m)
        h_xyz = matchers._radec_to_xyz(ra_full[b0:b1], dec_full[b0:b1])
        for pk, (ra, dec) in enumerate(pools):
            if not len(ra):
                continue
            p_xyz = matchers._radec_to_xyz(ra, dec)
            tree = cKDTree(p_xyz)
            hits = tree.query_ball_point(h_xyz, sep_chord)
            pe = pool_errs[pk]
            for i, hs in enumerate(hits):
                if not hs:
                    continue
                if plan.matcher == "sky":
                    keep[b0 + i] = False
                elif plan.matcher == "skyerr" and centre_err is not None and pe is not None:
                    idx = np.asarray(hs, dtype=np.int64)
                    chord = np.linalg.norm(p_xyz[idx] - h_xyz[i], axis=1)
                    seps_k = matchers._chord_to_arcsec(chord)
                    limit = float(plan.max_error) * (
                        float(centre_err[b0 + i]) + np.asarray(pe, dtype=float)[idx]
                    )
                    if np.any(np.isfinite(limit) & np.isfinite(seps_k) & (seps_k <= limit)):
                        keep[b0 + i] = False
                elif plan.matcher == "skyellipse" and centre_err is not None and pe is not None:
                    idx = np.asarray(hs, dtype=np.int64)
                    sra2_c, sde2_c, rho_c = centre_err
                    sra2_p, sde2_p, rho_p = pe
                    mean_dec = 0.5 * (float(dec_full[b0 + i]) + dec[idx])
                    cos_dec = np.cos(np.radians(mean_dec))
                    delta_ra = (
                        ((float(ra_full[b0 + i]) - ra[idx] + 180.0) % 360.0 - 180.0)
                        * 3600.0
                        * cos_dec
                    )
                    delta_dec = (float(dec_full[b0 + i]) - dec[idx]) * 3600.0
                    d2 = matchers._mahalanobis_pairwise(
                        delta_ra,
                        delta_dec,
                        np.full(idx.size, float(sra2_c[b0 + i])),
                        np.full(idx.size, float(sde2_c[b0 + i])),
                        np.full(idx.size, float(rho_c[b0 + i])),
                        sra2_p[idx],
                        sde2_p[idx],
                        rho_p[idx],
                    )
                    if np.any(np.isfinite(d2) & (d2 <= float(plan.max_error) ** 2)):
                        keep[b0 + i] = False
                else:
                    keep[b0 + i] = False

    kept_ids = np.flatnonzero(keep)
    if not len(kept_ids):
        return {"key": f"rest-{rest.cat}-{rest.part_idx:05d}", "rows": 0}
    sub = _gather_rows(frame, kept_ids)
    table = _assemble_block(
        plan,
        rest.cat,
        sub,
        {},
        {},
        np.full(sub.height, float("nan"), dtype=float),
        [str(rest.cat + 1)] * sub.height,
        np.arange(sub.height, dtype=np.int64),
    )
    _atomic_write_parquet(table, dest)
    return {"key": f"rest-{rest.cat}-{rest.part_idx:05d}", "rows": table.height}


def _rest_hub_keys(
    frame: pl.DataFrame,
    cat: CataloguePlan,
    hub_parts: list[PartitionPlan],
    max_hub_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row ``(hub partition index, outside pixel)`` for rest rows.

    ``outside pixel`` = the row's pixel at ``max_hub_order`` when no hub
    partition contains it (``-1`` otherwise); hub index ``-1`` marks rows
    outside the hub footprint.
    """
    import cdshealpix  # noqa: PLC0415
    from astropy.coordinates import Latitude, Longitude  # noqa: PLC0415

    ra = frame[cat.ra].cast(pl.Float64).to_numpy()
    dec = frame[cat.dec].cast(pl.Float64).to_numpy()
    if not len(ra):
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    npix = cdshealpix.nested.lonlat_to_healpix(
        Longitude(ra, unit="deg"),
        Latitude(dec, unit="deg"),
        np.full(len(ra), max_hub_order, dtype=np.uint64),
    )
    hub_idx = np.full(len(ra), -1, dtype=np.int64)
    for i, p in sorted(enumerate(hub_parts), key=lambda t: (-t[1].order, t[1].pix)):
        shift = max_hub_order - p.order
        anc = npix >> (2 * shift)
        hit = (anc == p.pix) & (hub_idx < 0)
        hub_idx[hit] = i
    outside = np.where(hub_idx < 0, npix.astype(np.int64), -1)
    return hub_idx, outside


# --------------------------------------------------------------------------- #
# Ray factories (lazy import; repo pattern)
# --------------------------------------------------------------------------- #
def _chunk_task_factory() -> Any:
    import ray  # noqa: PLC0415

    @ray.remote(num_cpus=1.0)
    def chunk_task(plan: UnionPlan, chunk: ChunkPlan) -> dict[str, Any]:
        return _run_chunk(plan, chunk)

    return chunk_task


def _rest_task_factory() -> Any:
    import ray  # noqa: PLC0415

    @ray.remote(num_cpus=1.0)
    def rest_task(plan: UnionPlan, rest: RestPlan) -> dict[str, Any]:
        return _run_rest(plan, rest)

    return rest_task


# --------------------------------------------------------------------------- #
# output assembly
# --------------------------------------------------------------------------- #
def _read_hats_properties(storage: Storage, rel: str) -> dict[str, str]:
    """Parse the ``key=value`` lines of ``<rel>/properties`` (or hats.properties).

    Storage-agnostic (mirror inputs are local dirs or ``vos:`` roots — HTTP
    sources are materialised into the cache before the plan builds).  Any
    read failure returns ``{}``: the HATS spec defaults apply.
    """
    props: dict[str, str] = {}
    for name in ("properties", "hats.properties"):
        full = f"{rel}/{name}" if rel else name
        try:
            if isinstance(storage, LocalStorage):
                p = Path(storage.root) / full
                if not p.is_file():
                    continue
                text = p.read_text()
            else:
                tmpdir = Path(tempfile.mkdtemp(prefix="xmatcher-hatsprops-"))
                try:
                    local = tmpdir / "properties"
                    storage.stage_in(full, local)
                    if not local.exists():
                        continue
                    text = local.read_text()
                finally:
                    shutil.rmtree(tmpdir, ignore_errors=True)
            for line in text.splitlines():
                if "=" in line:
                    k, _, v = line.partition("=")
                    props[k.strip()] = v.strip()
            if props:
                break
        except OSError:
            continue
    return props


def _read_hub_props(plan: UnionPlan) -> dict[str, str]:
    hub = plan.catalogues[0]
    out: dict[str, str] = {}
    try:
        storage = open_storage(hub.root)
        if isinstance(storage, LocalStorage):
            for name in ("properties", "hats.properties"):
                p = Path(storage.root) / hub.rel / name
                if p.is_file():
                    for line in p.read_text().splitlines():
                        if "=" in line:
                            k, _, v = line.partition("=")
                            out[k.strip()] = v.strip()
                    break
    except OSError:
        pass
    out["hats_col_ra"] = "_union_ra"
    out["hats_col_dec"] = "_union_dec"
    out["hats_ordering"] = "NESTED"
    out.setdefault("hats_nested", "True")
    out.setdefault("obs_collection", hub.name)
    return out


def _props_text(props: dict[str, str]) -> str:
    props.setdefault("dataproduct_type", "object")
    return "".join(f"{k}={props[k]}\n" for k in sorted(props))


def _atomic_write_csv(table: pl.DataFrame, dest: Path) -> None:
    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.tmp")
    try:
        table.write_csv(tmp)
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


def _parquet_row_count(path: Path) -> int:
    """Row count from the parquet footer (no data pages read).

    ``pl.read_parquet(dest).height`` re-read and decoded every output
    partition just to size the catalog; for a full-sky union that is a second
    pass over the whole dataset.
    """
    try:
        import pyarrow.parquet as pq  # noqa: PLC0415

        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:  # noqa: BLE001 - fall back to a real read
        return pl.read_parquet(path).height


def _atomic_write_text(text: str, dest: Path) -> None:
    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text)
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


def _assemble(plan: UnionPlan) -> dict[str, Any]:
    """Turn the per-chunk parquets into a HATS catalogue at ``plan.out_dir``."""
    out = Path(plan.out_dir)
    dataset = out / "dataset"
    dataset.mkdir(parents=True, exist_ok=True)
    # Insertion-ordered set: a list made the membership test below a linear
    # scan, i.e. O(P^2) for a full-sky output (P ~ 10^5-10^6 pixels).
    seen: dict[Path, None] = {}

    def add_partition(part: PartitionPlan, src: Path) -> None:
        relf = f"Norder={part.order}/Dir={_hats_dir(part.pix)}/Npix={part.pix}.parquet"
        dest = dataset / relf
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest in seen:
            # several chunks can carry rows of the same output pixel
            # (one per input catalogue); merge them deterministically.
            # ``vertical_relaxed``: a chunk that matched no row of a catalogue
            # contributes all-null columns (Null dtype when the input schema was
            # unreadable, e.g. a vos: cache root), and a strict vertical concat
            # refuses to stack Null next to the typed column another chunk wrote.
            merged = pl.concat(
                [pl.read_parquet(dest), pl.read_parquet(src)], how="vertical_relaxed"
            )
            _atomic_write_parquet(merged, dest)
        else:
            # `src` is our own staging file (rest-routing tmp): the move is
            # the rename, and it consumes the tmp -> no litter.
            os.replace(src, dest)
            seen[dest] = None
        # NOTE: the source chunk parquet is deliberately NOT removed — they
        # are the resume points (a rerun skips existing chunk files).

    # Keep catalogue labels stable when the first input has no footprint.
    hub_parts = next(
        (cat.partitions for cat in plan.catalogues if cat.partitions), [PartitionPlan(0, 0, "")]
    )
    max_hub_order = max(p.order for p in hub_parts)
    routing_cat = CataloguePlan("union", "", "", "_union_ra", "_union_dec")
    files = [out / "chunks" / f"{chunk.key}.parquet" for chunk in plan.chunks]
    files += [out / "chunks" / f"rest-{rest.cat}-{rest.part_idx:05d}.parquet" for rest in plan.rest]
    for src in files:
        if not src.exists():
            continue
        frame = pl.read_parquet(src)
        # Every emitted row uses its lowest-indexed member's coordinates.
        # Route all chunks (including moved sources and secondary-only rows)
        # through the hub tiling; copying each input's tile creates overlapping
        # parent/child output pixels and null advertised coordinate columns.
        hub_idx, outside_pix = _rest_hub_keys(frame, routing_cat, hub_parts, max_hub_order)
        groups = frame.with_columns(pl.Series("_route", hub_idx)).partition_by(
            "_route", as_dict=True
        )
        for key, sub in groups.items():
            h = int(key[0])
            if h < 0:
                continue
            tmp = out / "chunks" / f".{src.stem}-h{h}.tmp"
            try:
                sub.drop("_route").write_parquet(tmp)
                add_partition(hub_parts[h], tmp)
            finally:
                tmp.unlink(missing_ok=True)
        groups = frame.with_columns(pl.Series("_route", outside_pix)).partition_by(
            "_route", as_dict=True
        )
        for key, sub in groups.items():
            px = int(key[0])
            if px < 0:
                continue
            tmp = out / "chunks" / f".{src.stem}-o{px}.tmp"
            try:
                sub.drop("_route").write_parquet(tmp)
                add_partition(PartitionPlan(max_hub_order, px, ""), tmp)
            finally:
                tmp.unlink(missing_ok=True)

    if not seen:
        # A zero-row catalogue still carries a readable schema and HATS metadata.
        tmp = out / "chunks" / ".empty.tmp"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        try:
            _empty_union(plan).write_parquet(tmp)
            add_partition(hub_parts[0], tmp)
        finally:
            tmp.unlink(missing_ok=True)

    if seen:
        info: list[dict[str, Any]] = []
        total_rows = 0
        for dest in seen:
            order = int(dest.parent.parent.name.removeprefix("Norder="))
            pix = int(dest.name.removeprefix("Npix=").removesuffix(".parquet"))
            count = _parquet_row_count(dest)
            total_rows += count
            info.append(
                {
                    "Norder": order,
                    "Dir": int(_hats_dir(pix)),
                    "Npix": int(pix),
                    "Nfiles": 1,
                    "file_loc": f"Norder={order}/Dir={_hats_dir(pix)}/Npix={pix}",
                    "file_size": dest.stat().st_size,
                    "count": count,
                }
            )
        info_df = pl.DataFrame(info).sort(["Norder", "Npix", "file_loc"])
        _atomic_write_parquet(info_df, dataset / "partition_info.parquet")
        # hats 0.7.x reads the root copy; the dataset copy is the classic spec.
        _atomic_write_csv(info_df, dataset / "partition_info.csv")
        _atomic_write_csv(info_df, out / "partition_info.csv")
        props = _read_hub_props(plan)
        # the union's row count, never the hub's inherited hats_nrows
        props["hats_nrows"] = str(total_rows)
        props_text = _props_text(props)
        _atomic_write_text(props_text, out / "properties")
        _atomic_write_text(props_text, dataset / "properties")
        try:
            import pyarrow as pa  # noqa: PLC0415
            import pyarrow.parquet as pq  # noqa: PLC0415

            schemas = [pq.read_schema(f) for f in seen]
            unified = pa.unify_schemas(list(schemas))
            pq.write_metadata(unified, dataset / "_common_metadata")
            pq.write_metadata(unified, dataset / "_metadata")
        except Exception:  # noqa: BLE001
            logger.warning("heterogeneous parquet schemas; skipping _metadata")
    return {"chunks": len(seen), "rows": total_rows if seen else 0}


# --------------------------------------------------------------------------- #
# public driver
# --------------------------------------------------------------------------- #
def _append_event(out: Path, event: str, **fields: Any) -> None:
    """Append one JSON object to ``<out>/run.jsonl`` (best-effort, never fatal).

    The audit trail for day-scale runs: every driver attempt, mirror progress
    message, chunk completion, and the final ``done`` record land here so a
    watched run leaves a verifiable log even when the console is lost.
    """
    try:
        with open(out / "run.jsonl", "a") as fh:
            fh.write(
                json.dumps(
                    {
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "event": event,
                        **fields,
                    },
                    default=str,
                )
                + "\n"
            )
    except OSError:
        pass


def _input_fingerprint(plan: UnionPlan) -> list[dict[str, Any]]:
    """Local stat freshness; content digests for roots without reliable mtimes."""
    files = []
    for cat in plan.catalogues:
        storage = open_storage(cat.root)
        for part in cat.partitions:
            entry: dict[str, Any] = {"root": cat.root, "rel": part.rel}
            if isinstance(storage, LocalStorage):
                stat = (Path(storage.root) / part.rel).stat()
                entry.update(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
            else:
                # Remote stores lack a revision/mtime contract. Read each file
                # once for freshness instead of silently reusing stale chunks.
                with tempfile.TemporaryDirectory(prefix="xmatcher-fingerprint-") as directory:
                    local = Path(directory) / "partition.parquet"
                    storage.stage_in(part.rel, local)
                    with local.open("rb") as stream:
                        entry["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
            files.append(entry)
    return files


def ray_union_match(
    sources: Sequence[CatalogueSource],
    *,
    sep_arcsec: float,
    output_file: str,
    hats_threshold: int = 100_000,
    task_rows: int | None = None,
    chunk_memory_gb: float | None = None,
    max_tuples: int | None = None,
    cache_root: str | None = None,
    progress_cb: Callable[[str], None] | None = None,
    matcher: str = "sky",
    max_error: float = 1.0,
    target_epoch: float | None = None,
    pm_prior: bool = False,
    pm_prior_magnitude_column: str | None = None,
    fallback_policy: str = "warn",
) -> None:
    """Distributed N-way full-outer join over mirrored HATS catalogues.

    Writes the joined HATS catalogue at ``output_file`` (a directory).  The
    per-chunk parquets under ``<out>/chunks/`` are the resume points: reruns
    skip chunks whose file already exists.  The plan's parameters and inputs
    are recorded in ``resume.state``; rerunning with different parameters
    into the same directory wipes the stale chunks first.
    """
    matchers._validate_coordinate_frames(list(sources), target_epoch=target_epoch)
    out = Path(str(output_file))
    for source in sources:
        root, rel = _catalogue_root(source, cache_root)
        if not root.startswith("vos:"):
            input_path = (Path(root) / rel).resolve()
            output_path = out.resolve()
            if output_path.is_relative_to(input_path) or input_path.is_relative_to(output_path):
                raise CrossMatchError("ray-union output must not overlap an input catalogue")
    out.mkdir(parents=True, exist_ok=True)
    task_rows = DEFAULT_TASK_ROWS if task_rows is None else task_rows
    chunk_memory_gb = DEFAULT_CHUNK_MEMORY_GB if chunk_memory_gb is None else chunk_memory_gb
    max_tuples = DEFAULT_MAX_TUPLES if max_tuples is None else max_tuples
    sep_arcsec = float(sep_arcsec)
    global _LAST_PLAN  # noqa: PLW0603 - inspectable by tests / doctor
    plan = build_union_plan(
        list(sources),
        sep_arcsec=sep_arcsec,
        hats_threshold=int(hats_threshold),
        task_rows=task_rows,
        chunk_memory_gb=chunk_memory_gb,
        max_tuples=max_tuples,
        out_dir=str(out),
        cache_root=cache_root,
        matcher=matcher,
        max_error=float(max_error),
        target_epoch=target_epoch,
        pm_prior=pm_prior,
        pm_prior_magnitude_column=pm_prior_magnitude_column,
        fallback_policy=fallback_policy,
    )
    _LAST_PLAN = plan
    fingerprint: dict[str, Any] = {
        "sep_arcsec": sep_arcsec,
        "hats_threshold": int(hats_threshold),
        "task_rows": task_rows,
        "chunk_memory_gb": chunk_memory_gb,
        "max_tuples": max_tuples,
        "plan_version": 3,
        "fallback_policy": fallback_policy,
        "inputs": [
            {**asdict(cat), "dtypes": {col: str(dtype) for col, dtype in cat.dtypes.items()}}
            for cat in plan.catalogues
        ],
        # ponytail: local freshness uses stat, not a full O(bytes) digest;
        # nonlocal files require content hashes because their mtimes are unknown.
        "input_files": _input_fingerprint(plan),
    }
    if matcher != "sky":
        fingerprint["matcher"] = matcher
        fingerprint["max_error"] = float(max_error)
    if target_epoch is not None:
        fingerprint["target_epoch"] = float(target_epoch)
    if pm_prior:
        fingerprint["pm_prior"] = True
        if pm_prior_magnitude_column is not None:
            fingerprint["pm_prior_magnitude_column"] = pm_prior_magnitude_column

    state_path = out / _STATE_NAME
    prior_fp = None
    if state_path.exists():
        try:
            prior_fp = json.loads(state_path.read_text()).get("fingerprint")
        except (OSError, ValueError):
            prior_fp = None
    if prior_fp is None and any((out / sub).exists() for sub in ("chunks", "dataset")):
        raise CrossMatchError(
            "Cannot resume existing output without a valid fingerprint; use a new output directory."
        )
    if prior_fp is not None and prior_fp != fingerprint:
        # different parameters/inputs into the same out dir: stale chunks and
        # an assembled dataset from the old run must not leak into this one.
        import shutil  # noqa: PLC0415

        for sub in ("chunks", "dataset"):
            if (out / sub).exists():
                shutil.rmtree(out / sub)
        for name in ("properties", "partition_info.csv"):
            (out / name).unlink(missing_ok=True)

    if progress_cb:
        progress_cb(
            f"plan: {len(plan.chunks)} chunks, {len(plan.rest)} rest runs, "
            f"delta={plan.delta_deg:.4f} deg, radius={sep_arcsec} arcsec"
        )
    _atomic_write_text(
        json.dumps({"status": "running", "fingerprint": fingerprint}, indent=2), state_path
    )

    import ray  # noqa: PLC0415

    # join a running head (CANFAR: `ray start --head` + RAY_ADDRESS); on a
    # laptop with no cluster, "auto" raises and we fall back to a fresh local
    # cluster (Ray 2.x does not auto-start one for address="auto").
    owns_ray = not ray.is_initialized()
    if owns_ray:
        address = os.environ.get("RAY_ADDRESS")
        try:
            ray.init(address=address or "auto", ignore_reinit_error=True)
        except ConnectionError as exc:
            if address and fallback_policy == "error":
                raise CrossMatchError(
                    "engine='ray-union' could not connect to RAY_ADDRESS under fallback_policy='error'"
                ) from exc
            if address:
                logger.warning("Cannot connect to RAY_ADDRESS; starting a local Ray cluster.")
            # Explicit 'local' bypasses RAY_ADDRESS without changing caller env.
            ray.init(address="local", ignore_reinit_error=True)
    try:
        plan_ref = ray.put(plan)
        chunk_fn = _chunk_task_factory()
        rest_fn = _rest_task_factory()
        todo = []
        key_of = {}
        for chunk in plan.chunks:
            if not (out / "chunks" / f"{chunk.key}.parquet").exists():
                fref = chunk_fn.remote(plan_ref, chunk)
                key_of[fref] = chunk.key
                todo.append(fref)
        for rest in plan.rest:
            fname = f"rest-{rest.cat}-{rest.part_idx:05d}.parquet"
            if not (out / "chunks" / fname).exists():
                fref = rest_fn.remote(plan_ref, rest)
                key_of[fref] = fname.removesuffix(".parquet")
                todo.append(fref)
        total = len(todo)
        if progress_cb:
            progress_cb(f"ray-union: {total} task(s) to run")
        _append_event(out, "start", tasks=total)
        started_at = time.time()
        done = 0
        last_report = 0.0
        while todo:
            ready, todo = ray.wait(todo, num_returns=min(32, len(todo)), timeout=5.0)
            if not ready:
                continue
            ray.get(ready)  # surface task errors (Ray retries each task first)
            done += len(ready)
            for fref in ready:
                _append_event(out, "chunk_done", key=key_of.get(fref))
            now = time.time()
            if now - last_report >= 1.0:
                last_report = now
                rate = done / max(now - started_at, 1e-6)
                eta = int((total - done) / rate) if rate > 0 else None
                eta_s = f", ETA {eta}s" if eta is not None else ""
                if progress_cb:
                    progress_cb(
                        f"ray-union: {done}/{total} tasks "
                        f"({100.0 * done / max(total, 1):.0f}%, {rate:.1f}/s{eta_s})"
                    )
                _atomic_write_text(
                    json.dumps(
                        {
                            "status": "running",
                            "done": done,
                            "total": total,
                            "fingerprint": fingerprint,
                            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        },
                        indent=2,
                    ),
                    state_path,
                )
        assembled = _assemble(plan)
        rows = int(assembled.get("rows", 0))  # assembled total, not the task delta
        _atomic_write_text(
            json.dumps(
                {
                    "status": "done",
                    "chunks": assembled["chunks"],
                    "rows": rows,
                    "fingerprint": fingerprint,
                },
                indent=2,
            ),
            state_path,
        )
        _append_event(out, "done", rows=rows, chunks=assembled["chunks"])
        if progress_cb:
            progress_cb(f"ray-union: {total} tasks, {rows} rows, {assembled['chunks']} partitions")
    finally:
        if owns_ray:
            ray.shutdown()
