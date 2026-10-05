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

import dataclasses
import logging
import os

import numpy as np

from .exceptions import CrossMatchError
from .matchers import _first_occurrence_indices

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
        chord_max: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process one left HEALPix pixel batch as a Ray task.

        ``r_xyz_refs``/``r_groups_ref`` hold *only* this batch's neighbouring
        right pixels (the driver scopes them with a cone search), so a worker
        never materialises the whole right catalogue.  Builds one merged
        margin-cached cKDTree and queries it once, returning
        ``(left_idx, right_idx, seps)`` for this batch only.
        """
        from scipy.spatial import cKDTree

        from .matchers import _chord_to_arcsec

        empty = (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=float),
        )

        l_xyz = l_xyz_ref  # already numpy array via object store
        batch_xyz = l_xyz[indices_arr]

        # Nested ObjectRefs are NOT resolved by Ray 2.x when passed inside a
        # dict argument — resolve this batch's neighbour buffers here.
        import ray  # noqa: PLC0415

        margin_xyz_parts: list = []
        margin_global_parts: list = []
        for rpix_int, ref in r_xyz_refs.items():
            margin_xyz_parts.append(ray.get(ref))
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ray-parallelised HEALPix pixel-batch zone match.

    Falls back to :func:`xmatch.matchers._zone_match` when Ray is unavailable
    or when the Ray cluster has only one node.

    Parameters are identical to :func:`xmatch.matchers._scipy_match`.
    """
    from .matchers import _pixellate, _scipy_match, _skyerr_pair_filter, _zone_match

    # N-dimensional extra_distance_cols are delegated to _scipy_match
    # (single-machine cKDTree with N-d ranking).
    if spec.extra_distance_cols:
        if spec.fallback_policy == "error":
            raise CrossMatchError(
                "engine='ray' cannot honor extra-distance semantics; use engine='fast'"
            )
        logger.info("extra_distance_cols set; using Tier 1 cKDTree for N-d matching.")
        return _scipy_match(left, right, left_src, right_src, spec)

    # The distributed tasks score candidates on an isotropic per-row radius only,
    # so a full 2-D Mahalanobis match cannot be expressed here.  Say so instead
    # of silently matching a different criterion (torchsky does the same).
    if spec.matcher == "skyellipse":
        message = (
            "engine='ray' does not implement matcher='skyellipse'; "
            "use engine='zone' or engine='fast'"
        )
        if spec.fallback_policy == "error":
            raise CrossMatchError(message)
        logger.warning("%s — falling back to the single-machine zone engine.", message)
        return _zone_match(left, right, left_src, right_src, spec)

    if not ray_available():
        if spec.fallback_policy == "error":
            raise CrossMatchError("engine='ray' requires Ray under fallback_policy='error'")
        logger.info("Ray unavailable; using single-machine zone engine.")
        return _zone_match(left, right, left_src, right_src, spec)

    import ray

    if not ray.is_initialized():
        # Join the cluster in $RAY_ADDRESS when one is running (CANFAR:
        # a ray-manager session — see scripts/canfar-ray-job.sh); with the env unset this
        # starts a fresh local cluster exactly as before.  A dead address
        # (ConnectionError) falls back to local — same pattern as
        # ray_union.ray_union_match.
        try:
            ray.init(
                address=os.environ.get("RAY_ADDRESS") or None,
                ignore_reinit_error=True,
                logging_level=logging.WARNING,
            )
        except ConnectionError:
            # ray.init honours $RAY_ADDRESS for address=None too, so a dead
            # cluster address must be cleared before the local fallback.
            os.environ.pop("RAY_ADDRESS", None)
            ray.init(address=None, ignore_reinit_error=True, logging_level=logging.WARNING)
        except Exception as exc:
            if spec.fallback_policy == "error":
                raise CrossMatchError(
                    "engine='ray' could not initialize Ray under fallback_policy='error'"
                ) from exc
            logger.warning("Ray init failed (%s); using single-machine zone engine.", exc)
            return _zone_match(left, right, left_src, right_src, spec)

    # --- partition left-side data by HEALPix pixel ------------------------
    from .astro_utils import require_finite_coordinates
    from .matchers import _radec_to_xyz

    l_ra = left[left_src.ra_column].to_numpy().astype(float)
    l_dec = left[left_src.dec_column].to_numpy().astype(float)
    r_ra = right[right_src.ra_column].to_numpy().astype(float)
    r_dec = right[right_src.dec_column].to_numpy().astype(float)
    require_finite_coordinates(l_ra, l_dec, label=f"catalogue '{left_src.name}'")
    require_finite_coordinates(r_ra, r_dec, label=f"catalogue '{right_src.name}'")

    try:
        import cdshealpix as hp

        DEPTH = 5  # nside = 2 ** DEPTH == 32
        l_pix = _pixellate(hp, l_ra, l_dec, DEPTH, label=f"catalogue '{left_src.name}'")
        r_pix = _pixellate(hp, r_ra, r_dec, DEPTH, label=f"catalogue '{right_src.name}'")
    except ImportError as exc:
        if spec.fallback_policy == "error":
            raise CrossMatchError(
                "engine='ray' requires optional cdshealpix under fallback_policy='error'"
            ) from exc
        logger.warning("cdshealpix unavailable; falling back to scipy cKDTree.")
        return _scipy_match(left, right, left_src, right_src, spec)

    # Group left by pixel.
    l_sort_idx = np.argsort(l_pix, kind="stable")
    l_sorted_pix = l_pix[l_sort_idx]
    l_unique_indices = _first_occurrence_indices(l_sorted_pix)
    l_unique_pix = l_sorted_pix[l_unique_indices]
    l_splits = np.split(l_sort_idx, l_unique_indices[1:])
    # Keep numpy arrays (not Python int lists) — a full-sky catalogue has
    # millions of row indices and ``.tolist()`` triples their footprint.
    l_by_pix: dict[int, np.ndarray] = {
        int(k): v for k, v in zip(l_unique_pix, l_splits, strict=False)
    }

    # Group right by pixel.
    r_sort_idx = np.argsort(r_pix, kind="stable")
    r_sorted_pix = r_pix[r_sort_idx]
    r_unique_indices = _first_occurrence_indices(r_sorted_pix)
    unique_pix = r_sorted_pix[r_unique_indices]
    r_splits = np.split(r_sort_idx, r_unique_indices[1:])
    r_groups = {int(k): v for k, v in zip(unique_pix, r_splits, strict=False)}
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

    # Put immutable data in object store.  ``skyerr`` is a per-row criterion, so
    # the workers must hand back every candidate inside the global bound; the
    # driver applies the N-sigma filter and the best-per-primary reduction once
    # the batch results are gathered.
    task_spec = dataclasses.replace(spec, find="all") if spec.matcher == "skyerr" else spec
    l_xyz = _radec_to_xyz(l_ra, l_dec)
    l_xyz_ref = ray.put(l_xyz)
    spec_ref = ray.put(task_spec)
    chord_max_ref = ray.put(chord_max)

    # Scope every task's right-side inputs to the pixels inside its cone.
    # Passing the full ``r_xyz_refs`` dict made each worker resolve and
    # materialise the ENTIRE right catalogue (O(P) object-store reads per
    # task, O(P^2) overall — and on a multi-node cluster the whole catalogue
    # over the network per task).  The neighbour set is identical to the one
    # the worker used to compute locally, so results do not change.
    from .matchers import _cone_search_pixels

    # Submit one task per left pixel group.
    pixel_batch_fn = _get_ray_pixel_batch()
    futures: list = []
    total_batches = 0
    for l_pix_int, left_indices in l_by_pix.items():
        indices_arr = np.asarray(left_indices, dtype=np.int64)
        rep = l_xyz[indices_arr[indices_arr.shape[0] // 2]]
        npix = _cone_search_pixels(
            hp,
            float(np.arctan2(rep[1], rep[0])),
            float(np.arcsin(np.clip(rep[2], -1.0, 1.0))),
            float(np.radians(radius_deg)),
            DEPTH,
        )
        task_refs = {int(p): r_xyz_refs[int(p)] for p in npix if int(p) in r_xyz_refs}
        task_groups = {int(p): r_groups[int(p)] for p in npix if int(p) in r_groups}
        future = pixel_batch_fn.remote(
            l_pix_int,
            indices_arr,
            l_xyz_ref,
            task_refs,
            task_groups,
            spec_ref,
            chord_max_ref,
        )
        futures.append(future)
        total_batches += 1

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
    left_idx = np.concatenate(l_parts)
    right_idx = np.concatenate(r_parts)
    seps = np.concatenate(sep_parts)
    if spec.matcher == "skyerr":
        left_idx, right_idx, seps = _skyerr_pair_filter(
            left_idx,
            right_idx,
            seps,
            lsig,
            rsig,
            spec.max_error,
            find=spec.find,
        )
    return left_idx, right_idx, seps
