"""Distributed N-survey full-outer-join (union) crossmatch on a Ray cluster.

Pipeline for the ``engine=ray-union`` route: every input catalogue is
mirrored (see :mod:`xmatch.mirror`) into the xmatch cache as a local HATS
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
arcsec around the *centre partition pixel centre* (``delta`` = largest
partition diagonal across all catalogues; the factor-2 margin keeps the
covering conservative, so every mate row lies inside its chunk's pool).
Every centre row belongs to exactly one chunk and its partners all live in
that chunk's pool, so every row set is enumerated exactly once.  Partitions
of catalogue *k* that no earlier catalogue's cone ever covers are emitted
as single-``k`` rows by a "rest" task instead of a centre chunk.

Per-row neighbourhoods are resolved with a scipy :class:`cKDTree` (dense
cone matrices would blow the memory guard); combos per centre row are
capped by ``max_tuples``.

Output: ``<out>/dataset/Norder=…/Dir=…/Npix=….parquet`` — one directory
partition per input partition (no re-heap), plus
``dataset/partition_info.parquet``, ``properties`` and
``_metadata``/``_common_metadata``, readable via ``hats`` and LSDB-style
readers.  Chunk outputs go to ``<out>/chunks/<key>.parquet`` and are skipped
on resume when present.

Ray is imported lazily inside the task factories (repo convention: the
module must import without ray installed).
"""

from __future__ import annotations

import heapq
import itertools
import json
import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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
_BYTES_PER_ROW = 64.0  # conservative per-row footprint for the memory guards
_HUB_BLOCK = 50_000  # centre rows per distance batch
_STATE_NAME = "resume.state"
_FIXED_COLS = {"sep_arcsec", "_src_cats"}

_LAST_PLAN: Optional["UnionPlan"] = None
"""Most recently built :class:`UnionPlan` (inspected by tests / ``doctor``)."""


def last_plan() -> Optional["UnionPlan"]:
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
    partitions: List[PartitionPlan] = field(default_factory=list)
    cols: List[str] = field(default_factory=list)  # original column names
    final_cols: List[str] = field(default_factory=list)  # after _k suffixing
    dtypes: Dict[str, str] = field(default_factory=dict)  # final col -> polars dtype str


@dataclass
class ChunkPlan:
    key: str  # e.g. 'c0-00000' (centre catalogue + partition index)
    center: int  # catalogue index of the centre
    center_idx: int  # partition index inside the centre catalogue
    cand_idx: Dict[int, List[int]] = field(default_factory=dict)
    est_rows: int = 0  # pool estimate (rows) for the memory guard


@dataclass
class RestPlan:
    cat: int  # catalogue index
    part_idx: int  # partition index (never covered by an earlier cone)
    cand_idx: Dict[int, List[int]] = field(default_factory=dict)
    # partner catalogues k > cat whose partitions intersect the cone around
    # this partition's pixel: the only places a rest row can have a mate.


@dataclass
class UnionPlan:
    catalogues: List[CataloguePlan]
    sep_arcsec: float
    delta_arcsec: float
    max_tuples: int
    col_names: List[str]  # final output column order (incl. sep_arcsec/_src_cats)
    out_dir: str
    chunks: List[ChunkPlan] = field(default_factory=list)
    rest: List[RestPlan] = field(default_factory=list)
    rest_keys: set = field(default_factory=set)  # {(cat, part_idx)} of rest runs


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


def _pixel_center_deg(order: int, pix: int) -> Tuple[float, float]:
    cds = _cdshealpix()
    lon, lat = cds.healpix_to_lonlat(np.asarray([pix], dtype=np.int64), int(order))
    return float(np.degrees(np.asarray(lon.value)[0])), float(np.degrees(np.asarray(lat.value)[0]))


def _cone_pixels(order: int, pix: int, radius_deg: float, depth: int) -> List[Tuple[int, int]]:
    """Covering pixels at ``depth`` for a cone of ``radius_deg``.

    :func:`cdshealpix.cone_search` returns a *parent* pixel (shallower than
    ``depth``) when a whole subtree is inside the cone; every descendant of
    that parent at the candidate depth is covered, so expand parents to all
    their children (not just the first one).
    """
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
    out: List[Tuple[int, int]] = []
    for ipx, d in zip(np.asarray(ipix).tolist(), np.asarray(depths).tolist(), strict=True):
        dif = depth - int(d)
        if dif <= 0:
            out.append((depth, int(ipx)))
            continue
        base = int(ipx) << (2 * dif)
        for child in range(base, base + (1 << (2 * dif))):
            out.append((depth, child))
    return out


# --------------------------------------------------------------------------- #
# plan builder
# --------------------------------------------------------------------------- #
def _footer_row_count(storage: Storage, rel: str) -> Optional[int]:
    """Parquet footer row count (None when unavailable, e.g. remote)."""
    if isinstance(storage, LocalStorage):
        try:
            import pyarrow.parquet as pq  # noqa: PLC0415

            md = pq.read_metadata(Path(storage.root) / rel)
            return int(sum(md.row_group(r).num_rows for r in range(md.num_row_groups)))
        except Exception:  # noqa: BLE001
            return None
    return None


def _catalogue_root(src: CatalogueSource, cache_root: Optional[str]) -> Tuple[str, str]:
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


def _storage_rel(rel: str, part: str) -> str:
    return f"{rel.rstrip('/')}/{part}" if rel else part


def _list_partitions(rel: str, storage: Storage) -> List[PartitionPlan]:
    """Locate the standard HATS layout: ``Norder=*/Dir=*/Npix=*.parquet``.

    Supports both ``<root>/Norder=…`` (flat) and ``<root>/dataset/Norder=…``
    (the layout `hats` tools write) top-level arrangements.
    """
    out: List[PartitionPlan] = []

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
    bases: List[str] = []
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


def _catalogue_schema(storage: Storage, rel: str) -> Dict[str, str]:
    """Column name -> polars dtype name ('{}' when unknown)."""
    if isinstance(storage, LocalStorage):
        try:
            return {c: str(t) for c, t in pl.read_parquet_schema(Path(storage.root) / rel).items()}
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _est_rows(part: PartitionPlan, storage: Storage, fallback: int) -> int:
    if part.est_rows:
        return part.est_rows
    count = _footer_row_count(storage, part.rel)
    part.est_rows = count if count is not None else fallback
    return part.est_rows


def _c_map(cur_cand: List[Dict[int, List[int]]]) -> Dict[int, List[int]]:
    """Union of per-centre-partition candidate lists (deduped, sorted)."""
    sink: Dict[int, List[int]] = {}
    for d in cur_cand:
        for k, v in d.items():
            for i in v:
                if i not in sink.setdefault(k, []):
                    sink[k].append(i)
    return {k: sorted(v) for k, v in sink.items()}


def _lookup_partition(pix_to_idx: Dict[Tuple[int, int], int], od: int, px: int) -> Optional[int]:
    if (od, px) in pix_to_idx:
        return pix_to_idx[(od, px)]
    for j in range(od - 1, -1, -1):
        key = (j, px >> (2 * (od - j)))
        if key in pix_to_idx:
            return pix_to_idx[key]
    return None


def _cone_candidate_idx(
    part: PartitionPlan,
    catalogues: Sequence[CataloguePlan],
    depths: Sequence[int],
    cone_radius: float,
    centre: Optional[int] = None,
) -> Dict[int, List[int]]:
    """Partitions of each other catalogue intersecting the cone around ``part``."""
    cand: Dict[int, List[int]] = {}
    for j, other in enumerate(catalogues):
        if j == centre:
            continue
        other_pix = {(q.order, q.pix): idx for idx, q in enumerate(other.partitions)}
        seen: set[int] = set()
        for od, px in _cone_pixels(part.order, part.pix, cone_radius, depths[j]):
            if od > depths[j]:
                continue
            idx = _lookup_partition(other_pix, od, px)
            if idx is not None:
                seen.add(idx)
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
    cache_root: Optional[str] = None,
) -> UnionPlan:
    """Return a :class:`UnionPlan`; every source must be a local HATS dir."""
    if not sources:
        raise CrossMatchError("union needs at least one catalogue")
    if sep_arcsec <= 0:
        raise CrossMatchError(f"radius_arcsec must be positive, got {sep_arcsec}")
    max_tuples = max(1, int(max_tuples))
    task_rows = max(1, int(task_rows))
    chunk_memory_gb = max(0.1, float(chunk_memory_gb))

    catalogues: List[CataloguePlan] = []
    used: set[str] = set(_FIXED_COLS)
    all_cols: List[str] = []
    max_delta = 0.0

    for ci, src in enumerate(sources):
        root, rel = _catalogue_root(src, cache_root)
        storage = open_storage(root)
        parts = _list_partitions(rel, storage)
        if not parts:
            raise CrossMatchError(
                f"Catalogue '{src.name}' has no HATS partitions under {root!r}/{rel!r}."
            )
        # RING-ordered copies (remote mirrors preserve the source's own
        # hats_ordering property) are converted to NESTED in the plan
        # geometry: every downstream pixel computation (_pixel_center_deg,
        # _cone_pixels, the _lookup_partition ancestor shifts, the rest
        # tiling) is NESTED, and the output is always NESTED hub tiling.
        props = _read_hats_properties(storage, rel)
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
        _diag_cache: Dict[int, float] = {}
        for part in parts:
            _est_rows(part, storage, hats_threshold)
            diag = _diag_cache.get(part.order)
            if diag is None:
                diag = _pixel_diagonal_deg(part.order)
                _diag_cache[part.order] = diag
            max_delta = max(max_delta, diag)
        schema = _catalogue_schema(storage, parts[0].rel)
        if "file_loc" in schema:  # partition_info-style file is not data
            schema = {}
        cols = list(schema)
        if not cols:
            try:
                cols = list(pl.read_parquet_schema(Path(storage.root) / parts[0].rel))
            except Exception:  # noqa: BLE001
                cols = []
        final_cols: List[str] = []
        for col in cols:
            while col in used:
                col = f"{col}_{ci + 1}"
            used.add(col)
            final_cols.append(col)
        all_cols.extend(final_cols)

        dtypes: Dict[str, str] = {}
        if schema:
            dtypes = {
                final: str(schema[orig]) for orig, final in zip(cols, final_cols, strict=True)
            }
        else:
            try:
                raw = pl.read_parquet_schema(Path(storage.root) / parts[0].rel)
                dtypes = {
                    final: str(raw[orig]) for orig, final in zip(cols, final_cols, strict=True)
                }
            except Exception:  # noqa: BLE001
                dtypes = {}

        ra_orig = src.ra_column or "ra"
        dec_orig = src.dec_column or "dec"
        ra_final = final_cols[cols.index(ra_orig)] if ra_orig in cols else ra_orig
        dec_final = final_cols[cols.index(dec_orig)] if dec_orig in cols else dec_orig

        catalogues.append(
            CataloguePlan(
                name=src.name,
                root=root,
                rel=rel,
                ra=ra_final,
                dec=dec_final,
                partitions=parts,
                cols=cols,
                final_cols=final_cols,
                dtypes=dtypes,
            )
        )

    all_cols += ["sep_arcsec", "_src_cats"]
    sep_deg = sep_arcsec / 3600.0
    cone_radius = 2.0 * (sep_deg + max_delta)
    depths = [max((p.order for p in c.partitions), default=0) for c in catalogues]

    chunks: List[ChunkPlan] = []
    covered: List[List[int]] = [[] for _ in catalogues]

    for ci, cat in enumerate(catalogues):
        cat_storage = open_storage(cat.root)
        for pi, part in enumerate(cat.partitions):
            extra = _est_rows(part, cat_storage, hats_threshold)
            cand = _cone_candidate_idx(part, catalogues, depths, cone_radius, centre=ci)
            for j, seen in cand.items():
                extra += sum(
                    _est_rows(
                        catalogues[j].partitions[i],
                        open_storage(catalogues[j].root),
                        hats_threshold,
                    )
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
            mem = extra * _BYTES_PER_ROW / 1e9
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
                    cand_idx=_c_map([cand]),
                    est_rows=extra,
                )
            )

    rest: List[RestPlan] = []
    for ci in range(1, len(catalogues)):
        covered_set = set(covered[ci])
        for pi in range(len(catalogues[ci].partitions)):
            if pi not in covered_set:
                part = catalogues[ci].partitions[pi]
                cand = _cone_candidate_idx(part, catalogues, depths, cone_radius, centre=ci)
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
        delta_arcsec=max_delta,
        max_tuples=max_tuples,
        col_names=all_cols,
        out_dir=out_dir,
        chunks=chunks,
        rest=rest,
        rest_keys={(r.cat, r.part_idx) for r in rest},
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


def _cat_frame(cat: CataloguePlan, idxs: Sequence[int]) -> pl.DataFrame:
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
    return frame


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
    pools: Sequence[Tuple[np.ndarray, np.ndarray]],
    sep_chord: float,
    max_tuples: int,
    centre_label: int,
    cat_labels: Sequence[int],
    centre_cat: int = 0,
    drop_singles: bool = False,
) -> Tuple[Dict[int, np.ndarray], np.ndarray, List[str], np.ndarray]:
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

    trees: List[Optional[Any]] = []
    for k in range(n_partner):
        ra, dec = pools[k]
        trees.append(cKDTree(matchers._radec_to_xyz(ra, dec)) if len(ra) else None)

    cat_sel: Dict[int, List[int]] = {k: [] for k in range(n_partner)}
    centre_out: List[int] = []
    sep_out: List[float] = []
    src_out: List[str] = []

    for bi in range(len(centre_ra)):
        nbrs: List[Tuple[np.ndarray, np.ndarray]] = []  # (sorted idx, sep)
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
                chord = np.sqrt(np.clip(2.0 - 2.0 * (h_xyz[bi] @ xyz_c.T), 0.0, 4.0))
                nbrs.append((idx, matchers._chord_to_arcsec(chord)))
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
        lists: List[np.ndarray] = []
        for k in range(n_partner):
            idx_k = nbrs[k][0]
            lists.append(
                np.concatenate([np.array([-1], dtype=np.int64), idx_k])
                if idx_k.size
                else np.array([-1], dtype=np.int64)
            )
        total = int(np.prod([len(each) for each in lists]))

        def sep_of(
            combo: Tuple[int, ...], nbrs: List[Tuple[np.ndarray, np.ndarray]] = nbrs
        ) -> Optional[float]:
            vals = []
            for k in range(n_partner):
                if combo[k] >= 0:
                    pos = int(np.searchsorted(nbrs[k][0], combo[k]))
                    vals.append(float(nbrs[k][1][pos]))
            return min(vals) if vals else None  # None = centre-only row

        always: List[Tuple[int, ...]] = []
        heap: List[Tuple[float, Tuple[int, ...]]] = []  # (-sep, combo)
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
            ordered: List[Tuple[int, ...]] = always
        else:
            # Product exceeds max_tuples: keep the max_tuples combos with the
            # smallest sep.  Sort each partner's neighbours by separation and
            # scan the product row-major over the sorted lists (the best
            # combos — every partner near its closest — cluster at the scan
            # start), bounded to a small multiple of the cap so pathological
            # pools stay cheap; the heap bounds memory and the final sort
            # makes the cut deterministic.  Combo ids stay original pool ids,
            # so sep_of/cat_sel downstream need no conversion.  A row here
            # has mates (total > 1), so no centre-only combo can survive —
            # `always` stays empty and the single falls to the oracle rules.
            perm_k: List[Optional[np.ndarray]] = []
            sorted_lists: List[np.ndarray] = []
            for k in range(n_partner):
                idx_k = nbrs[k][0]
                if idx_k.size:
                    perm = np.argsort(nbrs[k][1], kind="stable")
                    perm_k.append(perm)
                    sorted_lists.append(np.concatenate([np.array([-1]), idx_k[perm]]))
                else:
                    perm_k.append(None)
                    sorted_lists.append(np.array([-1], dtype=np.int64))
            budget = max(1 << 20, max_tuples * 64)
            for scanned, combo in enumerate(
                itertools.product(*[each.tolist() for each in sorted_lists])
            ):
                if scanned >= budget:
                    break
                if all(c < 0 for c in combo):
                    continue  # centre-only: `always` carries it (empty here)
                vals = [
                    float(nbrs[k][1][perm_k[k][combo[k] - 1]])  # type: ignore[index]
                    for k in range(n_partner)
                    if combo[k] >= 0
                ]
                sep = min(vals)
                if len(heap) < max_tuples:
                    heapq.heappush(heap, (-sep, combo))
                elif sep < -heap[0][0]:
                    heapq.heapreplace(heap, (-sep, combo))
            ordered = [c for _, c in sorted(heap, key=lambda t: (t[0], t[1]))]
            if always:
                ordered.insert(0, always[0])

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
    cols = []
    for col in cat.final_cols:
        dtype = None
        dt = cat.dtypes.get(col)
        if dt:
            try:
                dtype = getattr(pl, dt.split("(", 1)[0].strip())
            except AttributeError:
                dtype = None
        cols.append(pl.Series(col, [None] * m, dtype=dtype, strict=False))
    return pl.DataFrame(cols)


def _assemble_block(
    plan: UnionPlan,
    centre: int,
    centre_frame: pl.DataFrame,
    cat_frames: Dict[int, pl.DataFrame],
    cat_sel: Dict[int, np.ndarray],
    seps: np.ndarray,
    srcs: List[str],
    centre_ids: np.ndarray,
) -> pl.DataFrame:
    """Horizontal slice of the output rows for one combination batch.

    Every catalogue contributes one column block: its rows selected by the
    per-row ids (``-1`` → all-null), or all-null when the catalogue is not
    part of the row set.
    """
    m = len(seps)
    tables: List[pl.DataFrame] = []
    for k in range(len(plan.catalogues)):
        if k == centre:
            tables.append(_gather_rows(centre_frame, centre_ids))
        else:
            idx = cat_sel.get(k, np.full(m, -1, dtype=np.int64))
            frame = cat_frames.get(k, _null_cat_frame(plan.catalogues[k], m))
            if frame.height == 0:
                frame = _null_cat_frame(plan.catalogues[k], m)  # empty pool
            tables.append(_gather_rows(frame, idx))
    tables.append(pl.Series("sep_arcsec", seps, dtype=pl.Float64).to_frame())
    tables.append(pl.Series("_src_cats", srcs, dtype=pl.String).to_frame())
    return pl.concat(tables, how="horizontal")


def _empty_named(col_names: Sequence[str]) -> pl.DataFrame:
    return pl.DataFrame({c: pl.Series(c, [], dtype=pl.Null) for c in col_names})


def _run_chunk(plan: UnionPlan, chunk: ChunkPlan) -> Dict[str, Any]:
    """Execute one chunk: rows radiating from its centre partition."""
    centre = plan.catalogues[chunk.center]
    centre_frame = _cat_frame(centre, [chunk.center_idx])
    n = centre_frame.height
    centre_ra = centre_frame[centre.ra].to_numpy() if n else np.zeros(0, dtype=float)
    centre_dec = centre_frame[centre.dec].to_numpy() if n else np.zeros(0, dtype=float)

    partner_globals = sorted(chunk.cand_idx)
    cat_frames: Dict[int, pl.DataFrame] = {}
    pools: List[Tuple[np.ndarray, np.ndarray]] = []
    labels: List[int] = []
    for k in partner_globals:
        cat = plan.catalogues[k]
        frame = _cat_frame(cat, chunk.cand_idx[k])
        cat_frames[k] = frame
        pools.append(
            (
                frame[cat.ra].to_numpy() if frame.height else np.zeros(0, dtype=float),
                frame[cat.dec].to_numpy() if frame.height else np.zeros(0, dtype=float),
            )
        )
        labels.append(k + 1)

    sep_chord = matchers._arcsec_to_chord(float(plan.sep_arcsec))
    drop_singles = (chunk.center, chunk.center_idx) in plan.rest_keys
    blocks: List[pl.DataFrame] = []
    for b0 in range(0, n, _HUB_BLOCK):
        b1 = min(b0 + _HUB_BLOCK, n)
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
        )
        if not len(seps):
            continue
        sel_global = {partner_globals[kk]: cat_sel[kk] for kk in range(len(partner_globals))}
        blocks.append(
            _assemble_block(
                plan, chunk.center, centre_frame, cat_frames, sel_global, seps, srcs, centre_ids
            )
        )

    table = pl.concat(blocks, how="diagonal_relaxed") if blocks else _empty_named(plan.col_names)
    if not table.height:
        return {"key": chunk.key, "rows": 0}
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


def _run_rest(plan: UnionPlan, rest: RestPlan) -> Dict[str, Any]:
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
    frame = _cat_frame(cat, [rest.part_idx])
    m = frame.height
    dest_root = Path(plan.out_dir) / "chunks"
    dest_root.mkdir(parents=True, exist_ok=True)
    dest = dest_root / f"rest-{rest.cat}-{rest.part_idx:05d}.parquet"
    if not m:
        return {"key": f"rest-{rest.cat}-{rest.part_idx:05d}", "rows": 0}

    ra_full = frame[cat.ra].to_numpy() if m else np.zeros(0, dtype=float)
    dec_full = frame[cat.dec].to_numpy() if m else np.zeros(0, dtype=float)

    partner_globals = sorted(rest.cand_idx)
    pools: List[Tuple[np.ndarray, np.ndarray]] = []
    for k in partner_globals:
        other = plan.catalogues[k]
        pf = _cat_frame(other, rest.cand_idx[k])
        pools.append(
            (
                pf[other.ra].to_numpy() if pf.height else np.zeros(0, dtype=float),
                pf[other.dec].to_numpy() if pf.height else np.zeros(0, dtype=float),
            )
        )

    sep_chord = matchers._arcsec_to_chord(float(plan.sep_arcsec))
    from scipy.spatial import cKDTree  # noqa: PLC0415

    keep = np.ones(m, dtype=bool)
    for b0 in range(0, m, _HUB_BLOCK):
        b1 = min(b0 + _HUB_BLOCK, m)
        h_xyz = matchers._radec_to_xyz(ra_full[b0:b1], dec_full[b0:b1])
        for ra, dec in pools:
            if not len(ra):
                continue
            tree = cKDTree(matchers._radec_to_xyz(ra, dec))
            hits = tree.query_ball_point(h_xyz, sep_chord)
            for i, hs in enumerate(hits):
                if hs:
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
    hub_parts: List[PartitionPlan],
    max_hub_order: int,
) -> Tuple[np.ndarray, np.ndarray]:
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
    def chunk_task(plan: UnionPlan, chunk: ChunkPlan) -> Dict[str, Any]:
        return _run_chunk(plan, chunk)

    return chunk_task


def _rest_task_factory() -> Any:
    import ray  # noqa: PLC0415

    @ray.remote(num_cpus=1.0)
    def rest_task(plan: UnionPlan, rest: RestPlan) -> Dict[str, Any]:
        return _run_rest(plan, rest)

    return rest_task


# --------------------------------------------------------------------------- #
# output assembly
# --------------------------------------------------------------------------- #
def _read_hats_properties(storage: Storage, rel: str) -> Dict[str, str]:
    """Parse the ``key=value`` lines of ``<rel>/properties`` (or hats.properties).

    Storage-agnostic (mirror inputs are local dirs or ``vos:`` roots — HTTP
    sources are materialised into the cache before the plan builds).  Any
    read failure returns ``{}``: the HATS spec defaults apply.
    """
    props: Dict[str, str] = {}
    for name in ("properties", "hats.properties"):
        full = f"{rel}/{name}" if rel else name
        try:
            if isinstance(storage, LocalStorage):
                p = Path(storage.root) / full
                if not p.is_file():
                    continue
                text = p.read_text()
            else:
                tmpdir = Path(tempfile.mkdtemp(prefix="xmatch-hatsprops-"))
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


def _read_hub_props(plan: UnionPlan) -> Dict[str, str]:
    hub = plan.catalogues[0]
    out: Dict[str, str] = {}
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
    out.setdefault("hats_col_ra", hub.ra)
    out.setdefault("hats_col_dec", hub.dec)
    out.setdefault("hats_nested", "True")
    out.setdefault("obs_collection", hub.name)
    return out


def _props_text(props: Dict[str, str]) -> str:
    props.setdefault("dataproduct_type", "object")
    return "".join(f"{k}={props[k]}\n" for k in sorted(props))


def _atomic_write_csv(table: pl.DataFrame, dest: Path) -> None:
    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.tmp")
    try:
        table.write_csv(tmp)
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_write_text(text: str, dest: Path) -> None:
    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text)
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


def _assemble(plan: UnionPlan) -> Dict[str, Any]:
    """Turn the per-chunk parquets into a HATS catalogue at ``plan.out_dir``."""
    out = Path(plan.out_dir)
    dataset = out / "dataset"
    dataset.mkdir(parents=True, exist_ok=True)
    seen: List[Path] = []

    def add_partition(part: PartitionPlan, src: Path, keep: bool = True) -> None:
        relf = f"Norder={part.order}/Dir={part.pix // 10000}/Npix={part.pix}.parquet"
        dest = dataset / relf
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest in seen:
            # several chunks can carry rows of the same output pixel
            # (one per input catalogue); merge them deterministically.
            merged = pl.concat([pl.read_parquet(dest), pl.read_parquet(src)])
            _atomic_write_parquet(merged, dest)
        elif keep:
            tmp = dest.with_name(f".{dest.name}.{os.getpid()}.tmp")
            try:
                import shutil  # noqa: PLC0415

                shutil.copy2(src, tmp)
                os.replace(tmp, dest)
            finally:
                tmp.unlink(missing_ok=True)
            seen.append(dest)
        else:
            # `src` is our own staging file (rest-routing tmp): the move is
            # the rename, and it consumes the tmp -> no litter.
            os.replace(src, dest)
            seen.append(dest)
        # NOTE: the source chunk parquet is deliberately NOT removed — they
        # are the resume points (a rerun skips existing chunk files).

    for chunk in plan.chunks:
        src = out / "chunks" / f"{chunk.key}.parquet"
        if src.exists():
            part = plan.catalogues[chunk.center].partitions[chunk.center_idx]
            add_partition(part, src)
    for rest in plan.rest:
        src = out / "chunks" / f"rest-{rest.cat}-{rest.part_idx:05d}.parquet"
        if not src.exists():
            continue
        frame = pl.read_parquet(src)
        cat = plan.catalogues[rest.cat]
        hub_parts = plan.catalogues[0].partitions
        max_hub_order = max(p.order for p in hub_parts)
        hub_idx, outside_pix = _rest_hub_keys(frame, cat, hub_parts, max_hub_order)

        hub_parts_dict = frame.with_columns(pl.Series("_k", hub_idx)).partition_by(
            "_k", as_dict=True
        )
        for (h,), sub_df in hub_parts_dict.items():
            if h < 0:
                continue
            tmp = out / "chunks" / f".rest-{rest.cat}-{rest.part_idx:05d}-h{int(h):05d}.tmp"
            try:
                sub_df.drop("_k").write_parquet(tmp)
                add_partition(plan.catalogues[0].partitions[int(h)], tmp, keep=False)
            finally:
                tmp.unlink(missing_ok=True)

        outside_parts_dict = frame.with_columns(pl.Series("_o", outside_pix)).partition_by(
            "_o", as_dict=True
        )
        for (px,), sub_df in outside_parts_dict.items():
            if px < 0:
                continue
            tmp = (
                out
                / "chunks"
                / f".rest-{rest.cat}-{rest.part_idx:05d}-o{max_hub_order:02d}_{int(px)}.tmp"
            )
            try:
                sub_df.drop("_o").write_parquet(tmp)
                add_partition(
                    PartitionPlan(order=max_hub_order, pix=int(px), rel=""), tmp, keep=False
                )
            finally:
                tmp.unlink(missing_ok=True)

    if seen:
        info: List[Dict[str, Any]] = []
        total_rows = 0
        for dest in seen:
            order = int(dest.parent.parent.name.removeprefix("Norder="))
            pix = int(dest.name.removeprefix("Npix=").removesuffix(".parquet"))
            count = pl.read_parquet(dest).height
            total_rows += count
            info.append(
                {
                    "Norder": order,
                    "Dir": int(pix // 10000),
                    "Npix": int(pix),
                    "Nfiles": 1,
                    "file_loc": f"Norder={order}/Dir={pix // 10000}/Npix={pix}",
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


def ray_union_match(
    sources: Sequence[CatalogueSource],
    *,
    sep_arcsec: float,
    output_file: str,
    hats_threshold: int = 100_000,
    task_rows: Optional[int] = None,
    chunk_memory_gb: Optional[float] = None,
    max_tuples: Optional[int] = None,
    cache_root: Optional[str] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> None:
    """Distributed N-way full-outer join over mirrored HATS catalogues.

    Writes the joined HATS catalogue at ``output_file`` (a directory).  The
    per-chunk parquets under ``<out>/chunks/`` are the resume points: reruns
    skip chunks whose file already exists.  The plan's parameters and inputs
    are recorded in ``resume.state``; rerunning with different parameters
    into the same directory wipes the stale chunks first.
    """
    out = Path(str(output_file))
    out.mkdir(parents=True, exist_ok=True)
    task_rows = int(task_rows or DEFAULT_TASK_ROWS)
    chunk_memory_gb = float(chunk_memory_gb or DEFAULT_CHUNK_MEMORY_GB)
    max_tuples = int(max_tuples or DEFAULT_MAX_TUPLES)
    sep_arcsec = float(sep_arcsec)
    fingerprint = {
        "sep_arcsec": sep_arcsec,
        "hats_threshold": int(hats_threshold),
        "task_rows": task_rows,
        "chunk_memory_gb": chunk_memory_gb,
        "max_tuples": max_tuples,
        "inputs": [
            {
                "name": s.name,
                "root": (cache_root or "") if s.hats_cache_rel else str(s.path or ""),
                "rel": s.hats_cache_rel or "",
            }
            for s in sources
        ],
    }

    state_path = out / _STATE_NAME
    prior_fp = None
    if state_path.exists():
        try:
            prior_fp = json.loads(state_path.read_text()).get("fingerprint")
        except (OSError, ValueError):
            prior_fp = None
    if prior_fp is not None and prior_fp != fingerprint:
        # different parameters/inputs into the same out dir: stale chunks and
        # an assembled dataset from the old run must not leak into this one.
        import shutil  # noqa: PLC0415

        for sub in ("chunks", "dataset"):
            shutil.rmtree(out / sub, ignore_errors=True)

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
    )
    _LAST_PLAN = plan
    if progress_cb:
        progress_cb(
            f"plan: {len(plan.chunks)} chunks, {len(plan.rest)} rest runs, "
            f"delta={plan.delta_arcsec:.4f} deg, radius={sep_arcsec} arcsec"
        )
    state_path.write_text(json.dumps({"status": "running", "fingerprint": fingerprint}, indent=2))
    import ray  # noqa: PLC0415

    # join a running head (CANFAR: `ray start --head` + RAY_ADDRESS); on a
    # laptop with no cluster, "auto" raises and we fall back to a fresh local
    # cluster (Ray 2.x does not auto-start one for address="auto").
    try:
        ray.init(address=os.environ.get("RAY_ADDRESS") or "auto", ignore_reinit_error=True)
    except ConnectionError:
        # ray.init honours $RAY_ADDRESS for address=None too, so a dead
        # cluster address must be cleared before the local fallback.
        os.environ.pop("RAY_ADDRESS", None)
        ray.init(address=None, ignore_reinit_error=True)
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
                state_path.write_text(
                    json.dumps(
                        {
                            "status": "running",
                            "done": done,
                            "total": total,
                            "fingerprint": fingerprint,
                            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        },
                        indent=2,
                    )
                )
        assembled = _assemble(plan)
        rows = int(assembled.get("rows", 0))  # assembled total, not the task delta
        (out / _STATE_NAME).write_text(
            json.dumps(
                {
                    "status": "done",
                    "chunks": assembled["chunks"],
                    "rows": rows,
                    "fingerprint": fingerprint,
                },
                indent=2,
            )
        )
        _append_event(out, "done", rows=rows, chunks=assembled["chunks"])
        if progress_cb:
            progress_cb(f"ray-union: {total} tasks, {rows} rows, {assembled['chunks']} partitions")
    finally:
        ray.shutdown()
