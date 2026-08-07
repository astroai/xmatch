"""Optional Ray-based distributed match engine.

Requires ``ray`` (``pip install ray``).  When available, pixel-batch matching
in the HEALPix zone engine is fanned out across Ray workers.  When Ray is
unavailable, the engine falls back to the single-machine :func:`_zone_match`
transparently.

Usage
-----
The public entry point :func:`ray_zone_match` is called from
:func:`xmatch.matchers.sky_match` when ``engine="ray"`` is selected.

Architecture
------------
Each HEALPix pixel group (left-side points co-located in one pixel) is
treated as a stateless Ray task.  Right-side pixel data is placed in Ray's
object store with ``ray.put()`` so workers can read it zero-copy.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

from .exceptions import CrossMatchError

logger = logging.getLogger(__name__)

# Lazy-initialised Ray remote function (avoids import-time @ray.remote failure).
_RAY_PIXEL_BATCH = None


def ray_available() -> bool:
    """Return ``True`` when Ray can be imported."""
    try:
        import ray  # noqa: F401

        return True
    except Exception:
        return False


def _get_ray_pixel_batch():
    """Return the ``@ray.remote``-decorated pixel-batch function.

    Only initialised once Ray is confirmed available, so the module imports
    cleanly even without ``ray`` installed.
    """
    global _RAY_PIXEL_BATCH
    if _RAY_PIXEL_BATCH is not None:
        return _RAY_PIXEL_BATCH

    import ray

    @ray.remote  # type: ignore[name-defined]
    def _ray_pixel_batch(
        l_pix_int: int,
        indices_arr: np.ndarray,
        l_xyz_ref,
        r_xyz_refs: dict,
        r_groups_ref: dict,
        spec,
        radius_deg: float,
        chord_max: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process one left HEALPix pixel batch as a Ray task.

        Builds a merged margin-cached cKDTree for all neighbouring right
        pixels and queries it once, returning
        ``(left_idx, right_idx, seps)`` for this batch only.
        """
        import cdshealpix as hp
        from scipy.spatial import cKDTree

        from .matchers import _chord_to_arcsec

        empty = (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=float),
        )

        l_xyz = l_xyz_ref  # already numpy array via object store
        batch_xyz = l_xyz[indices_arr]

        # Determine neighbouring right pixels via cone search.
        mid = len(indices_arr) // 2
        rep_xyz = batch_xyz[mid]
        rep_lon = float(np.arctan2(rep_xyz[1], rep_xyz[0]))
        rep_lat = float(np.arcsin(np.clip(rep_xyz[2], -1.0, 1.0)))
        npix = hp.cone_search_lonlat(
            lon=rep_lon,
            lat=rep_lat,
            radius=float(np.radians(radius_deg)),
            depth=5,
        )

        # Build merged margin tree from all neighbouring right pixels.
        margin_xyz_parts: list = []
        margin_global_parts: list = []
        for rpix in npix:
            rpix_int = int(rpix)
            if rpix_int not in r_xyz_refs:
                continue
            margin_xyz_parts.append(r_xyz_refs[rpix_int])
            margin_global_parts.append(r_groups_ref[rpix_int])

        if not margin_xyz_parts:
            return empty

        margin_xyz = np.vstack(margin_xyz_parts)
        margin_global = np.concatenate(margin_global_parts)
        margin_tree = cKDTree(margin_xyz)

        if spec.find == "best":
            dist, local_idx = margin_tree.query(
                batch_xyz,
                k=1,
                distance_upper_bound=chord_max,
                workers=-1,
            )
            valid = np.isfinite(dist) & (local_idx < margin_tree.n)
            if not np.any(valid):
                return empty
            left_idx = indices_arr[valid].astype(np.int64)
            right_idx = margin_global[local_idx[valid]].astype(np.int64)
            seps = _chord_to_arcsec(dist[valid])
            return left_idx, right_idx, np.asarray(seps, dtype=float)

        # find == "all"
        idx_lists = margin_tree.query_ball_point(
            batch_xyz,
            r=chord_max,
            workers=-1,
        )
        l_parts, r_parts, sep_parts = [], [], []
        for k, neighbors in enumerate(idx_lists):
            if not neighbors:
                continue
            k_idx = int(indices_arr[k])
            nb = np.asarray(neighbors, dtype=np.int64)
            global_r = margin_global[nb].astype(np.int64)
            chords = np.linalg.norm(margin_xyz[nb] - batch_xyz[k], axis=-1)
            seps = _chord_to_arcsec(chords)
            l_parts.append(np.full(len(neighbors), k_idx, dtype=np.int64))
            r_parts.append(global_r)
            sep_parts.append(np.asarray(seps, dtype=float))

        if not l_parts:
            return empty
        return (
            np.concatenate(l_parts),
            np.concatenate(r_parts),
            np.concatenate(sep_parts),
        )

    _RAY_PIXEL_BATCH = _ray_pixel_batch
    return _RAY_PIXEL_BATCH


def _ray_gather_with_progress(futures, total: int):
    """Gather Ray task results with a polling progress indicator.

    Uses ``ray.wait()`` to collect results one at a time and logs progress
    at INFO level with batch count, percentage, and throughput.
    """
    import time

    import ray

    remaining = list(futures)
    results: list = []
    start_time = time.time()
    last_log = start_time

    while remaining:
        ready, remaining = ray.wait(remaining, num_returns=1, timeout=None)
        results.extend(ray.get(ready))
        done = len(results)
        now = time.time()
        # Log at most once per second so the progress bar doesn't spam.
        if done == total or now - last_log >= 1.0:
            elapsed = max(now - start_time, 1e-6)
            rate = done / elapsed
            eta = (total - done) / rate if rate > 0 else 0
            logger.info(
                "Ray progress: %d/%d batches (%.0f%%, %.1f/s, ETA %.0fs)",
                done,
                total,
                100.0 * done / total,
                rate,
                eta,
            )
            last_log = now

    elapsed = time.time() - start_time
    logger.info(
        "Ray gather complete: %d batches in %.1fs (%.1f/s).",
        total,
        elapsed,
        total / max(elapsed, 1e-6),
    )
    return results


def ray_zone_match(
    left,
    right,
    left_src,
    right_src,
    spec,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ray-parallelised HEALPix pixel-batch zone match.

    Falls back to :func:`xmatch.matchers._zone_match` when Ray is unavailable
    or when the Ray cluster has only one node.

    Parameters are identical to :func:`xmatch.matchers._scipy_match`.
    """
    from .matchers import _scipy_match, _zone_match

    # N-dimensional extra_distance_cols are delegated to _scipy_match
    # (single-machine cKDTree with N-d ranking).
    if spec.extra_distance_cols:
        if spec.fallback_policy == "error":
            raise CrossMatchError(
                "engine='ray' cannot honor extra-distance semantics; use engine='fast'"
            )
        logger.info("extra_distance_cols set; using Tier 1 cKDTree for N-d matching.")
        return _scipy_match(left, right, left_src, right_src, spec)

    if not ray_available():
        if spec.fallback_policy == "error":
            raise CrossMatchError("engine='ray' requires Ray under fallback_policy='error'")
        logger.info("Ray unavailable; using single-machine zone engine.")
        return _zone_match(left, right, left_src, right_src, spec)

    import ray

    if not ray.is_initialized():
        try:
            ray.init(ignore_reinit_error=True, logging_level=logging.WARNING)
        except Exception as exc:
            if spec.fallback_policy == "error":
                raise CrossMatchError(
                    "engine='ray' could not initialize Ray under fallback_policy='error'"
                ) from exc
            logger.warning("Ray init failed (%s); using single-machine zone engine.", exc)
            return _zone_match(left, right, left_src, right_src, spec)

    # --- partition left-side data by HEALPix pixel ------------------------
    from .matchers import _radec_to_xyz

    l_ra = left[left_src.ra_column].to_numpy().astype(float)
    l_dec = left[left_src.dec_column].to_numpy().astype(float)
    r_ra = right[right_src.ra_column].to_numpy().astype(float)
    r_dec = right[right_src.dec_column].to_numpy().astype(float)

    try:
        import cdshealpix as hp

        NSIDE = 32
        l_pix = np.asarray(
            hp.lonlat_to_healpix(np.radians(l_ra), np.radians(l_dec), NSIDE),
            dtype=int,
        )
        r_pix = np.asarray(
            hp.lonlat_to_healpix(np.radians(r_ra), np.radians(r_dec), NSIDE),
            dtype=int,
        )
    except ImportError as exc:
        if spec.fallback_policy == "error":
            raise CrossMatchError(
                "engine='ray' requires optional cdshealpix under fallback_policy='error'"
            ) from exc
        logger.warning("cdshealpix unavailable; falling back to scipy cKDTree.")
        return _scipy_match(left, right, left_src, right_src, spec)

    # Group left by pixel.
    # O(N log N) vectorized grouping to replace slow native Python loop over millions of points.
    l_sort = np.argsort(l_pix)
    l_unique_pix, l_start_idx = np.unique(l_pix[l_sort], return_index=True)
    l_splits = np.split(l_sort, l_start_idx[1:])
    l_by_pix: dict[int, list] = {int(pix): split.tolist() for pix, split in zip(l_unique_pix, l_splits)}

    # Group right by pixel.
    # O(N log N) grouping optimization to replace O(N * K) boolean masking.
    r_sort = np.argsort(r_pix)
    unique_pix, r_start_idx = np.unique(r_pix[r_sort], return_index=True)
    r_splits = np.split(r_sort, r_start_idx[1:])
    r_groups = {int(pix): split for pix, split in zip(unique_pix, r_splits)}
    r_xyz_by_pix = {int(pix): _radec_to_xyz(r_ra[idx], r_dec[idx]) for pix, idx in r_groups.items()}

    # Place right-side pixel data in Ray's object store (zero-copy for workers).
    r_xyz_refs: dict[int, ray.ObjectRef] = {}
    for pix, xyz in r_xyz_by_pix.items():
        r_xyz_refs[pix] = ray.put(xyz)

    # Compute chord_max and radius_deg for the match.
    from .matchers import _arcsec_to_chord

    if spec.matcher == "sky":
        radius_deg = spec.radius_arcsec / 3600.0
        chord_max = _arcsec_to_chord(spec.radius_arcsec)
    else:
        from .matchers import _pos_sigma_arcsec

        lsig = _pos_sigma_arcsec(left, left_src)
        rsig = _pos_sigma_arcsec(right, right_src)
        if lsig is None or rsig is None:
            return (np.array([], int), np.array([], int), np.array([], float))
        search_radius = spec.max_error * (float(np.nanmax(lsig)) + float(np.nanmax(rsig)))
        radius_deg = max(search_radius, 0.0) / 3600.0
        chord_max = _arcsec_to_chord(max(search_radius, 0.0))

    # Put immutable data in object store.
    l_xyz_ref = ray.put(_radec_to_xyz(l_ra, l_dec))
    r_groups_ref = ray.put({int(k): v for k, v in r_groups.items()})
    spec_ref = ray.put(spec)
    radius_deg_ref = ray.put(radius_deg)
    chord_max_ref = ray.put(chord_max)

    # Submit one task per left pixel group.
    pixel_batch_fn = _get_ray_pixel_batch()
    futures: list = []
    pixel_items = list(l_by_pix.items())
    total_batches = len(pixel_items)
    for l_pix_int, left_indices in pixel_items:
        indices_arr = np.asarray(left_indices, dtype=np.int64)
        future = pixel_batch_fn.remote(
            l_pix_int,
            indices_arr,
            l_xyz_ref,
            r_xyz_refs,
            r_groups_ref,
            spec_ref,
            radius_deg_ref,
            chord_max_ref,
        )
        futures.append(future)

    logger.info("Submitted %d pixel batches to Ray workers.", total_batches)

    # Gather results with progress reporting.
    results = _ray_gather_with_progress(futures, total_batches)

    # Concatenate per-pixel batch results.
    l_parts, r_parts, sep_parts = [], [], []
    for l_idx, r_idx, seps in results:
        if l_idx.size == 0:
            continue
        l_parts.append(l_idx)
        r_parts.append(r_idx)
        sep_parts.append(seps)

    empty = (np.array([], int), np.array([], int), np.array([], float))
    if not l_parts:
        return empty
    return (
        np.concatenate(l_parts),
        np.concatenate(r_parts),
        np.concatenate(sep_parts),
    )
