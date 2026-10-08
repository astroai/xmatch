"""Bounded-memory pairwise matching for local CSV/Parquet catalogues."""

from __future__ import annotations

import dataclasses
import logging
import math
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import polars as pl

from . import io_utils
from .exceptions import CrossMatchError
from .matchers import (
    _EPOCH_SIGMA,
    _PROPAGATED_COV_EE,
    _PROPAGATED_COV_NN,
    MatchSpec,
    _apply_pm_drift_prior,
    _apply_proper_motion,
    _astrometric_covariance_mas,
    _pos_sigma_arcsec,
    _validate_coordinate_frames,
    sky_match,
)
from .request import MatchRequest, SideOverrides
from .sources import CatalogueSource, position_error_to_arcsec_factor

logger = logging.getLogger(__name__)

_INTERNAL_PREFIX = "_xmatcher_spill_"
_LEFT_ROW = f"{_INTERNAL_PREFIX}left_row"
_RIGHT_ROW = f"{_INTERNAL_PREFIX}right_row"
_ZONE = f"{_INTERNAL_PREFIX}zone"
_LEFT_SIGMA = f"{_INTERNAL_PREFIX}left_sigma"
_RIGHT_SIGMA = f"{_INTERNAL_PREFIX}right_sigma"
_RANK = f"{_INTERNAL_PREFIX}rank"
_RESULT_GROUP = f"{_INTERNAL_PREFIX}result_group"
_SUPPORTED_INPUT_SUFFIXES = {".csv", ".parquet"}
_SUPPORTED_OUTPUT_SUFFIXES = {".csv", ".parquet"}
_MIN_BUDGET = 1024


def preflight_request(req: MatchRequest) -> None:
    """Validate spill controls and reject large unsupported files before scanning."""
    if req.memory_budget_bytes is None:
        return
    budget = int(req.memory_budget_bytes)
    if budget < _MIN_BUDGET:
        raise CrossMatchError(
            f"memory_budget_bytes must be at least {_MIN_BUDGET} bytes; got {budget}."
        )
    _resolve_order(req.partition_order, 1, budget)
    paths: list[Path] = []
    for value in (req.cat1, req.cat2):
        if isinstance(value, (str, Path)):
            path = Path(value)
            if path.is_file():
                paths.append(path)
    if len(paths) == 2 and sum(path.stat().st_size for path in paths) > budget:
        suffixes = {path.suffix.lower() for path in paths}
        if not suffixes <= _SUPPORTED_INPUT_SUFFIXES:
            raise CrossMatchError(
                "Bounded-memory local matching currently supports pairwise CSV/Parquet inputs "
                f"only; got {sorted(suffixes)}. Convert large FITS inputs to Parquet first."
            )


def projected_input_bytes(
    src: CatalogueSource,
    overrides: SideOverrides,
    spec: MatchSpec,
) -> int:
    """Conservative on-disk proxy for the lazily projected input size."""
    if src.path is None:
        return 0
    total = src.path.stat().st_size
    schema_names = src.lazy().collect_schema().names()
    selected = _required_columns(schema_names, src, overrides, spec)
    return max(1, math.ceil(total * len(selected) / max(1, len(schema_names))))


def spill_required(req: MatchRequest, src1: CatalogueSource, src2: CatalogueSource) -> bool:
    """Return whether this request should take the bounded-memory route."""
    if req.memory_budget_bytes is None:
        return False
    budget = int(req.memory_budget_bytes)
    if budget < _MIN_BUDGET:
        raise CrossMatchError(
            f"memory_budget_bytes must be at least {_MIN_BUDGET} bytes; got {budget}."
        )
    paths = (src1.path, src2.path)
    if any(path is None for path in paths):
        return False
    estimate = projected_input_bytes(src1, req.side1, req.spec) + projected_input_bytes(
        src2, req.side2, req.spec
    )
    if estimate <= budget:
        return False
    suffixes = {path.suffix.lower() for path in paths if path is not None}
    if not suffixes <= _SUPPORTED_INPUT_SUFFIXES:
        raise CrossMatchError(
            "Bounded-memory local matching currently supports pairwise CSV/Parquet inputs only; "
            f"got {sorted(suffixes)}. Convert large FITS inputs to Parquet first."
        )
    return True


def match_to_output(
    req: MatchRequest,
    src1: CatalogueSource,
    src2: CatalogueSource,
    *,
    right_suffix: str = "_2",
    source_tags: bool = False,
) -> dict[str, int]:
    """Partition, batch-match, canonicalize, and atomically stream one result."""
    if req.output_file is None:
        raise CrossMatchError(
            "A bounded-memory match requires output_file so results can be streamed without "
            "materialising the final table."
        )
    if req.lazy:
        raise CrossMatchError("lazy=True is not supported for a bounded-memory output match.")
    out = Path(req.output_file)
    if out.suffix.lower() not in _SUPPORTED_OUTPUT_SUFFIXES:
        raise CrossMatchError(
            f"Bounded-memory output currently supports .parquet and .csv only; got '{out.suffix}'."
        )
    _validate_semantics(req, src1, src2)

    budget = int(req.memory_budget_bytes or 0)
    estimated = projected_input_bytes(src1, req.side1, req.spec) + projected_input_bytes(
        src2, req.side2, req.spec
    )
    order = _resolve_order(req.partition_order, estimated, budget)
    n_dec = 1 << order
    n_ra = 1 << (order + 1)

    scratch_parent = Path(req.scratch_dir) if req.scratch_dir else None
    if scratch_parent is not None:
        scratch_parent.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{out.stem}.xmatcher-", suffix=out.suffix, dir=out.parent
    )
    os.close(fd)
    temp_out = Path(temp_name)
    temp_out.unlink()

    stats = {
        "memory_budget_bytes": budget,
        "estimated_input_bytes": estimated,
        "partition_order": order,
        "partitions_matched": 0,
        "batch_pairs": 0,
        "peak_proxy_bytes": 0,
        "observed_batch_bytes": 0,
        "minimum_batch_proxy_bytes": 0,
    }
    try:
        with tempfile.TemporaryDirectory(prefix="xmatcher-spill-", dir=scratch_parent) as td:
            root = Path(td)
            left_dir = root / "left"
            right_dir = root / "right"
            candidates_dir = root / "candidates"
            candidates_dir.mkdir()

            effective_spec, left_lf, right_lf, max_halo_arcsec = _prepare_inputs(req, src1, src2)
            fragment = 0
            left_count = left_lf.select(pl.len()).collect(engine="streaming").item()
            right_count = right_lf.select(pl.len()).collect(engine="streaming").item()
            if left_count and (right_count or req.spec.join_type in {"1or2", "all"}):
                _stage(left_lf, left_dir, src1, _LEFT_ROW, n_dec, n_ra)
            if right_count and (left_count or req.spec.join_type in {"1or2", "all"}):
                _stage(right_lf, right_dir, src2, _RIGHT_ROW, n_dec, n_ra)

            if left_count and right_count:
                left_zones = _zone_paths(left_dir)
                right_zones = _zone_paths(right_dir)
                halo_bands = math.ceil((max_halo_arcsec / 3600.0) / (180.0 / n_dec))
                for left_zone, left_path in left_zones.items():
                    left_band = left_zone // n_ra
                    # scan every occupied RA cell in the declination halo. This is
                    # exact but can over-read dense bands; upgrade to HEALPix adjacency when
                    # that profiling ceiling is reached.
                    relevant = [
                        (zone, path)
                        for zone, path in right_zones.items()
                        if abs(zone // n_ra - left_band) <= halo_bands + 1
                    ]
                    if not relevant:
                        continue
                    stats["partitions_matched"] += 1
                    for _, right_path in relevant:
                        fragment = _match_partition_pair(
                            left_path,
                            right_path,
                            candidates_dir,
                            fragment,
                            req,
                            effective_spec,
                            src1,
                            src2,
                            max_halo_arcsec,
                            budget,
                            right_suffix,
                            stats,
                        )

            if fragment == 0:
                _write_empty_candidate(
                    left_lf,
                    right_lf,
                    candidates_dir / "00000000.parquet",
                    src1,
                    src2,
                    effective_spec,
                    max_halo_arcsec,
                    right_suffix,
                )

            canonical = _canonical_pairs(
                pl.scan_parquet(candidates_dir / "*.parquet"), effective_spec
            )
            if effective_spec.join_type in {"1or2", "all"}:
                canonical_path = root / "canonical.parquet"
                canonical.sink_parquet(canonical_path, engine="streaming")
                result_lf = _outer_result(
                    pl.scan_parquet(canonical_path),
                    left_lf,
                    right_lf,
                    left_dir,
                    right_dir,
                    root,
                    right_suffix,
                    source_tags=source_tags,
                )
            else:
                result_lf = _public_result(canonical)
            io_utils.write_frame(result_lf, temp_out)
            os.replace(temp_out, out)
    finally:
        temp_out.unlink(missing_ok=True)
    logger.info(
        "Bounded-memory match wrote %s (order=%d, peak proxy=%d/%d bytes).",
        out,
        order,
        stats["peak_proxy_bytes"],
        budget,
    )
    return stats


def _validate_semantics(req: MatchRequest, src1: CatalogueSource, src2: CatalogueSource) -> None:
    _validate_coordinate_frames([src1, src2], target_epoch=req.spec.target_epoch)
    if req.id_join:
        raise CrossMatchError(
            "Bounded-memory ID joins are not part of the first local spill tranche."
        )
    if req.spec.join_type not in {"1and2", "1or2", "all"}:
        raise CrossMatchError(
            "Bounded-memory matching currently supports join_type='1and2' or '1or2' only."
        )
    if req.spec.matcher not in {"sky", "skyerr"}:
        raise CrossMatchError(
            "Bounded-memory matching currently supports matcher='sky' and 'skyerr' only."
        )
    if (
        req.probabilistic
        or req.spec.prior_columns
        or req.spec.extra_distance_cols
        or req.spec.filter_expr
    ):
        raise CrossMatchError(
            "Bounded-memory matching does not yet support priors, extra-distance columns, "
            "or filter expressions."
        )
    if req.spec.pm_prior:
        raise CrossMatchError(
            "Bounded-memory matching does not yet support a missing-motion prior."
        )
    if req.spec.target_epoch is not None and (
        not math.isfinite(req.spec.target_epoch) or req.spec.target_epoch <= 0
    ):
        raise CrossMatchError("target_epoch must be a finite positive Julian year.")
    for src in (src1, src2):
        if not src.ra_column or not src.dec_column:
            raise CrossMatchError(f"RA/Dec columns unknown for '{src.name}'.")


def _resolve_order(value: str | int, estimated: int, budget: int) -> int:
    if value == "auto":
        target_cells = max(1, math.ceil(estimated / max(1, budget // 2)))
        return min(10, max(0, math.ceil(math.log(max(1, target_cells / 2), 4))))
    try:
        order = int(value)
    except (TypeError, ValueError) as exc:
        raise CrossMatchError("partition_order must be 'auto' or a non-negative integer.") from exc
    if order < 0 or order > 12:
        raise CrossMatchError("partition_order must be between 0 and 12.")
    return order


def _required_columns(
    schema_names: list[str],
    src: CatalogueSource,
    overrides: SideOverrides,
    spec: MatchSpec,
) -> list[str]:
    requested = list(overrides.columns) if overrides.columns else list(schema_names)
    required: list[str | None] = [src.ra_column, src.dec_column]
    if spec.matcher == "skyerr":
        required.extend((src.ra_err_column, src.dec_err_column))
    if spec.target_epoch is not None:
        required.extend(
            (
                src.epoch_column,
                src.pm_ra_column,
                src.pm_dec_column,
                src.parallax_column,
                src.radial_velocity_column,
            )
        )
        if spec.matcher == "skyerr" and src.astrometric_covariance_columns:
            required.extend(src.astrometric_covariance_columns.values())
    selected_set = set(requested) | {name for name in required if name}
    missing = selected_set - set(schema_names)
    if missing:
        raise CrossMatchError(f"Requested columns not found in '{src.name}': {sorted(missing)}")
    selected = [name for name in schema_names if name in selected_set]
    reserved = [name for name in selected if name.startswith(_INTERNAL_PREFIX)]
    if reserved:
        raise CrossMatchError(f"Input uses reserved bounded-memory columns: {reserved}")
    return selected


def _sigma_expr(src: CatalogueSource, alias: str) -> pl.Expr:
    factor = position_error_to_arcsec_factor(src.pos_err_units)
    floor = (
        float(src.default_pos_error_arcsec) * math.sqrt(2.0)
        if src.default_pos_error_arcsec is not None
        else None
    )
    if src.ra_err_column and src.dec_err_column:
        sigma = (
            (pl.col(src.ra_err_column).cast(pl.Float64) * factor) ** 2
            + (pl.col(src.dec_err_column).cast(pl.Float64) * factor) ** 2
        ).sqrt()
        if floor is not None:
            sigma = sigma.fill_nan(floor).fill_null(floor).clip(lower_bound=floor)
        return sigma.alias(alias)
    if floor is not None:
        return pl.lit(floor, dtype=pl.Float64).alias(alias)
    return pl.lit(None, dtype=pl.Float64).alias(alias)


def _has_errors(src: CatalogueSource) -> bool:
    return bool(src.ra_err_column and src.dec_err_column) or (
        src.default_pos_error_arcsec is not None
    )


def _prepare_inputs(
    req: MatchRequest,
    src1: CatalogueSource,
    src2: CatalogueSource,
) -> tuple[MatchSpec, pl.LazyFrame, pl.LazyFrame, float]:
    spec = req.spec
    matcher = spec.matcher
    if matcher == "skyerr" and not (_has_errors(src1) and _has_errors(src2)):
        if spec.fallback_policy == "error" or spec.target_epoch is not None:
            raise CrossMatchError(
                "matcher='skyerr' requires positional errors for the requested science"
            )
        logger.warning("skyerr lacks positional errors; bounded-memory matching falls back to sky.")
        spec = dataclasses.replace(spec, matcher="sky")

    names1 = src1.lazy().collect_schema().names()
    names2 = src2.lazy().collect_schema().names()
    lf1 = src1.lazy().select(_required_columns(names1, src1, req.side1, spec))
    lf2 = src2.lazy().select(_required_columns(names2, src2, req.side2, spec))
    _validate_coordinates(lf1, src1)
    _validate_coordinates(lf2, src2)
    if spec.target_epoch is not None:
        # Repartition the moved positions, rather than guessing a motion halo.
        lf1 = _align_epoch(
            lf1, src1, spec.target_epoch, propagate_covariance=spec.matcher == "skyerr"
        )
        lf2 = _align_epoch(
            lf2, src2, spec.target_epoch, propagate_covariance=spec.matcher == "skyerr"
        )
    if spec.matcher == "sky":
        return spec, lf1, lf2, float(spec.radius_arcsec)

    lf1 = lf1.with_columns(
        pl.col(_EPOCH_SIGMA).alias(_LEFT_SIGMA)
        if spec.target_epoch is not None
        else _sigma_expr(src1, _LEFT_SIGMA)
    )
    lf2 = lf2.with_columns(
        pl.col(_EPOCH_SIGMA).alias(_RIGHT_SIGMA)
        if spec.target_epoch is not None
        else _sigma_expr(src2, _RIGHT_SIGMA)
    )
    _validate_sigma(lf1, _LEFT_SIGMA, src1)
    _validate_sigma(lf2, _RIGHT_SIGMA, src2)
    max_left = lf1.select(pl.col(_LEFT_SIGMA).max()).collect(engine="streaming").item()
    max_right = lf2.select(pl.col(_RIGHT_SIGMA).max()).collect(engine="streaming").item()
    max_halo = spec.max_error * (float(max_left or 0.0) + float(max_right or 0.0))
    return spec, lf1, lf2, max_halo


def _align_epoch(
    lf: pl.LazyFrame,
    src: CatalogueSource,
    epoch: float,
    *,
    propagate_covariance: bool = False,
    pm_prior: bool = False,
    magnitude_column: str | None = None,
) -> pl.LazyFrame:
    """Align batches with measured motion and optional evaluated uncertainty.

    ``skyerr`` uses sqrt(trace(C_position)) from the existing local Jacobian.
    Five-parameter physical covariance conditions on a declared deterministic
    radial velocity; uncertain RV is rejected instead of silently dropping its
    variance. Reference-epoch rows retain their declared positional errors.
    Explicit ``pm_prior`` supports the existing radial RMS drift model on
    sides with no measured PM columns; it does not replace invalid measured PM.
    """
    if src.frame.lower() != "icrs":
        raise CrossMatchError("Epoch alignment requires ICRS coordinates.")
    if not math.isfinite(epoch) or epoch <= 0:
        raise CrossMatchError("target_epoch must be a finite positive Julian year.")
    assert src.ra_column and src.dec_column
    lf = lf.with_columns(pl.col(src.ra_column, src.dec_column).cast(pl.Float64))
    if propagate_covariance:
        lf = lf.with_columns(_sigma_expr(src, _EPOCH_SIGMA))
    if src.epoch_column:
        reference = pl.col(src.epoch_column).cast(pl.Float64)
    elif src.epoch is not None:
        reference = pl.repeat(float(src.epoch), pl.len(), dtype=pl.Float64)
    else:
        raise CrossMatchError(f"'{src.name}' needs a known epoch for proper motion alignment.")
    invalid_epoch = reference.is_null() | ~reference.is_finite() | (reference <= 0)
    if lf.select(invalid_epoch.sum()).collect(engine="streaming").item():
        raise CrossMatchError(f"'{src.name}' contains invalid proper motion reference epochs.")
    moving = reference != epoch
    prior_only = pm_prior and not (src.pm_ra_column or src.pm_dec_column)
    if src.pm_ra_column and src.pm_dec_column:
        pmra = pl.col(src.pm_ra_column).cast(pl.Float64)
        pmdec = pl.col(src.pm_dec_column).cast(pl.Float64)
        invalid_motion = moving & (
            pmra.is_null() | pmdec.is_null() | ~pmra.is_finite() | ~pmdec.is_finite()
        )
    else:
        invalid_motion = moving & pl.lit(not prior_only)
    if lf.select(invalid_motion.sum()).collect(engine="streaming").item():
        raise CrossMatchError(
            f"'{src.name}' needs finite proper motion for rows away from target_epoch; "
            "supply measured motion or an explicitly stationary model."
        )
    if not lf.select(moving.any()).collect(engine="streaming").item():
        return lf

    def propagate(batch: pl.DataFrame) -> pl.DataFrame:
        changed = batch.select(moving).to_series().to_numpy()
        if not changed.any():
            return batch
        if prior_only:
            inflated = _apply_pm_drift_prior(
                src, src, batch, batch.head(0), epoch, magnitude_column=magnitude_column
            )[0]
            sigma = _pos_sigma_arcsec(inflated, src)
            if sigma is None:
                raise CrossMatchError(f"'{src.name}' needs positional errors for a motion prior.")
            return batch.with_columns(pl.Series(_EPOCH_SIGMA, sigma))
        subset = batch.filter(pl.Series(changed))
        if propagate_covariance:
            covariance = _astrometric_covariance_mas(subset, src)
            if covariance is None:
                raise CrossMatchError(
                    f"'{src.name}' needs measured astrometric covariance for target-epoch skyerr."
                )
            _, valid_physical, valid_angular = covariance
            complete = np.zeros(subset.height, dtype=bool)
            if src.parallax_column and src.radial_velocity_column:
                parallax = subset[src.parallax_column].to_numpy().astype(float)
                rv = subset[src.radial_velocity_column].to_numpy().astype(float)
                complete = np.isfinite(parallax) & (parallax > 0) & np.isfinite(rv)
            if (
                complete.any()
                and src.release_metadata.get("radial_velocity_uncertainty") != "deterministic"
            ):
                raise CrossMatchError(
                    "target-epoch skyerr radial-velocity uncertainty must be explicitly deterministic; five-parameter covariance omits RV uncertainty."
                )
            if not np.where(complete, valid_physical, valid_angular).all():
                raise CrossMatchError(
                    f"'{src.name}' contains missing or invalid target-epoch astrometric covariance."
                )
        moved = _apply_proper_motion(
            subset,
            subset.head(0),
            src,
            src,
            epoch,
            propagate_covariance=propagate_covariance,
            fallback_policy="error",
        )[0]
        columns = []
        for name in (src.ra_column, src.dec_column):
            values = batch[name].to_numpy().copy()
            values[changed] = moved[name].to_numpy()
            columns.append(pl.Series(name, values))
        if propagate_covariance:
            if _PROPAGATED_COV_EE not in moved or _PROPAGATED_COV_NN not in moved:
                raise CrossMatchError(
                    "Torchsky target-epoch covariance propagation is unavailable."
                )
            east = moved[_PROPAGATED_COV_EE].to_numpy()
            north = moved[_PROPAGATED_COV_NN].to_numpy()
            if not (np.isfinite(east) & np.isfinite(north) & (east >= 0) & (north >= 0)).all():
                raise CrossMatchError("target-epoch positional covariance is invalid.")
            sigma = batch[_EPOCH_SIGMA].to_numpy().copy()
            evaluated = np.sqrt(east + north)
            if src.default_pos_error_arcsec is not None:
                evaluated = np.maximum(evaluated, src.default_pos_error_arcsec * math.sqrt(2))
            sigma[changed] = evaluated
            columns.append(pl.Series(_EPOCH_SIGMA, sigma))
        return batch.with_columns(columns)

    # This operation preserves row order, schema and identity. Blocking predicate
    # pushdown ensures every motion row is validated before spatial filtering.
    return lf.map_batches(
        propagate,
        schema=lf.collect_schema(),
        streamable=True,
        predicate_pushdown=False,
        projection_pushdown=False,
        slice_pushdown=False,
    )


def _validate_coordinates(lf: pl.LazyFrame, src: CatalogueSource) -> None:
    assert src.ra_column and src.dec_column
    ra = pl.col(src.ra_column).cast(pl.Float64)
    dec = pl.col(src.dec_column).cast(pl.Float64)
    invalid = (
        lf.select(
            (
                ra.is_null()
                | dec.is_null()
                | ~ra.is_finite()
                | ~dec.is_finite()
                | (dec < -90.0)
                | (dec > 90.0)
            ).sum()
        )
        .collect(engine="streaming")
        .item()
    )
    if invalid:
        raise CrossMatchError(
            f"'{src.name}' contains {invalid} invalid RA/Dec row(s); coordinates must be "
            "finite and declination must lie in [-90, 90] degrees."
        )


def _validate_sigma(lf: pl.LazyFrame, column: str, src: CatalogueSource) -> None:
    sigma = pl.col(column)
    invalid = (
        lf.select((sigma.is_null() | ~sigma.is_finite() | (sigma < 0.0)).sum())
        .collect(engine="streaming")
        .item()
    )
    if invalid:
        raise CrossMatchError(
            f"'{src.name}' contains {invalid} invalid positional-uncertainty row(s)."
        )


def _stage(
    lf: pl.LazyFrame,
    target: Path,
    src: CatalogueSource,
    row_column: str,
    n_dec: int,
    n_ra: int,
) -> None:
    assert src.ra_column and src.dec_column
    dec_width = 180.0 / n_dec
    ra_width = 360.0 / n_ra
    dec_bin = (
        ((pl.col(src.dec_column).cast(pl.Float64) + 90.0) / dec_width)
        .floor()
        .clip(0, n_dec - 1)
        .cast(pl.Int64)
    )
    ra_bin = (
        ((pl.col(src.ra_column).cast(pl.Float64) % 360.0) / ra_width)
        .floor()
        .clip(0, n_ra - 1)
        .cast(pl.Int64)
    )
    staged = lf.with_row_index(row_column).with_columns((dec_bin * n_ra + ra_bin).alias(_ZONE))
    staged.sink_parquet(
        pl.PartitionBy(target, key=_ZONE, include_key=True),
        mkdir=True,
        maintain_order=True,
        engine="streaming",
    )


def _zone_paths(root: Path, zone_column: str = _ZONE) -> dict[int, Path]:
    return {
        int(path.name.split("=", 1)[1]): path
        for path in sorted(root.glob(f"{zone_column}=*"))
        if path.is_dir()
    }


def _scan_partition(path: Path) -> pl.LazyFrame:
    return pl.scan_parquet(path / "*.parquet")


def _row_count(path: Path) -> int:
    return int(_scan_partition(path).select(pl.len()).collect(engine="streaming").item())


def _bytes_per_row(path: Path, rows: int) -> int:
    size = sum(file.stat().st_size for file in path.glob("*.parquet"))
    return max(32, math.ceil(size / max(1, rows)))


def _slices(rows: int, batch_rows: int) -> Iterable[tuple[int, int]]:
    for offset in range(0, rows, batch_rows):
        yield offset, min(batch_rows, rows - offset)


def _match_partition_pair(
    left_path: Path,
    right_path: Path,
    candidates_dir: Path,
    fragment: int,
    req: MatchRequest,
    spec: MatchSpec,
    src1: CatalogueSource,
    src2: CatalogueSource,
    max_halo_arcsec: float,
    budget: int,
    right_suffix: str,
    stats: dict[str, int],
) -> int:
    left_rows = _row_count(left_path)
    right_rows = _row_count(right_path)
    if not left_rows or not right_rows:
        return fragment
    left_bpr = _bytes_per_row(left_path, left_rows)
    right_bpr = _bytes_per_row(right_path, right_rows)
    result_bpr = left_bpr + right_bpr + 24

    if spec.find == "all" or spec.matcher == "skyerr":
        max_pairs = max(1, budget // max(1, 2 * result_bpr))
        side = max(1, math.isqrt(max_pairs))
        left_batch_rows = min(left_rows, side)
        right_batch_rows = min(right_rows, side)
        candidate_find = "all"
    else:
        left_batch_rows = min(left_rows, max(1, budget // max(1, 4 * left_bpr)))
        right_batch_rows = min(right_rows, max(1, budget // max(1, 4 * right_bpr)))
        candidate_find = "best"

    def planned_bytes(left_count: int, right_count: int) -> int:
        result_count = left_count * right_count if candidate_find == "all" else left_count
        return left_count * left_bpr + right_count * right_bpr + result_count * result_bpr

    while planned_bytes(left_batch_rows, right_batch_rows) > budget and (
        left_batch_rows > 1 or right_batch_rows > 1
    ):
        if right_batch_rows > left_batch_rows and right_batch_rows > 1:
            right_batch_rows = max(1, right_batch_rows // 2)
        elif left_batch_rows > 1:
            left_batch_rows = max(1, left_batch_rows // 2)
        else:
            right_batch_rows = max(1, right_batch_rows // 2)
    stats["minimum_batch_proxy_bytes"] = max(
        stats["minimum_batch_proxy_bytes"], planned_bytes(1, 1)
    )

    candidate_spec = dataclasses.replace(
        spec,
        matcher="sky",
        radius_arcsec=max_halo_arcsec,
        find=candidate_find,
        join_type="1and2",
        prior_columns=[],
        batch_size=None,
        target_epoch=None,  # Positions were already aligned before spatial staging.
    )
    engine = "fast" if req.engine == "auto" else req.engine
    for left_offset, left_len in _slices(left_rows, left_batch_rows):
        left = _scan_partition(left_path).slice(left_offset, left_len).collect(engine="streaming")
        for right_offset, right_len in _slices(right_rows, right_batch_rows):
            right = (
                _scan_partition(right_path)
                .slice(right_offset, right_len)
                .collect(engine="streaming")
            )
            stats["batch_pairs"] += 1
            observed = int(left.estimated_size() + right.estimated_size())
            potential_results = left_len * right_len if candidate_find == "all" else left_len
            pair_proxy = (
                left_len * left_bpr + right_len * right_bpr + potential_results * result_bpr
            )
            stats["peak_proxy_bytes"] = max(stats["peak_proxy_bytes"], pair_proxy)
            result = sky_match(
                src1,
                src2,
                left.lazy(),
                right.lazy(),
                candidate_spec,
                engine=engine,
                right_suffix=right_suffix,
            ).collect()
            if result.height == 0:
                stats["observed_batch_bytes"] = max(stats["observed_batch_bytes"], observed)
                continue
            if spec.matcher == "skyerr":
                limit = spec.max_error * (pl.col(_LEFT_SIGMA) + pl.col(_RIGHT_SIGMA))
                result = result.filter(pl.col("sep_arcsec") <= limit).with_columns(
                    (pl.col("sep_arcsec") / limit).alias(_RANK)
                )
            else:
                result = result.with_columns(pl.col("sep_arcsec").alias(_RANK))
            observed += int(result.estimated_size())
            stats["observed_batch_bytes"] = max(stats["observed_batch_bytes"], observed)
            if result.height:
                result.write_parquet(candidates_dir / f"{fragment:08d}.parquet")
                fragment += 1
    return fragment


def _write_empty_candidate(
    left_lf: pl.LazyFrame,
    right_lf: pl.LazyFrame,
    output: Path,
    src1: CatalogueSource,
    src2: CatalogueSource,
    spec: MatchSpec,
    max_halo_arcsec: float,
    right_suffix: str,
) -> None:
    empty_left = (
        left_lf.with_row_index(_LEFT_ROW)
        .with_columns(pl.lit(0, dtype=pl.Int64).alias(_ZONE))
        .head(0)
    )
    empty_right = (
        right_lf.with_row_index(_RIGHT_ROW)
        .with_columns(pl.lit(0, dtype=pl.Int64).alias(_ZONE))
        .head(0)
    )
    empty_spec = dataclasses.replace(
        spec, matcher="sky", radius_arcsec=max_halo_arcsec, find="all", join_type="1and2"
    )
    empty = sky_match(
        src1,
        src2,
        empty_left,
        empty_right,
        empty_spec,
        engine="fast",
        right_suffix=right_suffix,
    ).collect()
    empty.with_columns(pl.lit(None, dtype=pl.Float64).alias(_RANK)).write_parquet(output)


def _canonical_pairs(lf: pl.LazyFrame, spec: MatchSpec) -> pl.LazyFrame:
    pair_order = [_LEFT_ROW, _RIGHT_ROW, _RANK, "sep_arcsec"]
    lf = lf.sort(pair_order).unique(subset=[_LEFT_ROW, _RIGHT_ROW], keep="first")
    if spec.find == "best":
        lf = lf.sort([_LEFT_ROW, _RANK, "sep_arcsec", _RIGHT_ROW]).unique(
            subset=[_LEFT_ROW], keep="first"
        )
    return lf.sort([_LEFT_ROW, _RIGHT_ROW])


def _public_result(lf: pl.LazyFrame) -> pl.LazyFrame:
    internal = [name for name in lf.collect_schema().names() if name.startswith(_INTERNAL_PREFIX)]
    return lf.drop(internal)


def _outer_result(
    matched: pl.LazyFrame,
    left_lf: pl.LazyFrame,
    right_lf: pl.LazyFrame,
    left_dir: Path,
    right_dir: Path,
    root: Path,
    right_suffix: str,
    *,
    source_tags: bool,
) -> pl.LazyFrame:
    """Append deterministic source-order islands without global row-ID state."""
    right_zone = f"{_ZONE}{right_suffix}"
    left_ids_dir = root / "matched-left"
    right_ids_dir = root / "matched-right"
    _stage_matched_ids(matched, left_ids_dir, _LEFT_ROW, _ZONE)
    _stage_matched_ids(matched, right_ids_dir, _RIGHT_ROW, right_zone)

    left_only_dir = root / "left-only"
    right_only_dir = root / "right-only"
    left_only_dir.mkdir()
    right_only_dir.mkdir()
    _write_unmatched(left_dir, left_ids_dir, left_only_dir, _LEFT_ROW)

    left_columns = set(left_lf.collect_schema().names()) | {_LEFT_ROW, _ZONE}
    right_rename = {name: f"{name}{right_suffix}" for name in left_columns}
    _write_unmatched(
        right_dir,
        right_ids_dir,
        right_only_dir,
        _RIGHT_ROW,
        matched_zone_column=right_zone,
        rename=right_rename,
    )

    empty_left = (
        left_lf.with_row_index(_LEFT_ROW)
        .with_columns(pl.lit(None, dtype=pl.Int64).alias(_ZONE))
        .head(0)
    )
    empty_right = (
        right_lf.with_row_index(_RIGHT_ROW)
        .with_columns(pl.lit(None, dtype=pl.Int64).alias(_ZONE))
        .head(0)
    )
    overlaps = set(empty_right.collect_schema().names()) & set(right_rename)
    empty_right = empty_right.rename({name: right_rename[name] for name in overlaps})
    parts = [
        matched.with_columns(pl.lit(0, dtype=pl.Int8).alias(_RESULT_GROUP)),
        _scan_or_empty(left_only_dir, empty_left).with_columns(
            pl.lit(1, dtype=pl.Int8).alias(_RESULT_GROUP)
        ),
        _scan_or_empty(right_only_dir, empty_right).with_columns(
            pl.lit(2, dtype=pl.Int8).alias(_RESULT_GROUP)
        ),
    ]
    result = pl.concat(parts, how="diagonal_relaxed")
    if source_tags:
        result = result.with_columns(
            pl.when(pl.col(_RESULT_GROUP) == 0)
            .then(pl.lit("1+2"))
            .when(pl.col(_RESULT_GROUP) == 1)
            .then(pl.lit("1"))
            .otherwise(pl.lit("2"))
            .alias("_src_cats")
        )
    return _public_result(result.sort([_RESULT_GROUP, _LEFT_ROW, _RIGHT_ROW]))


def _stage_matched_ids(
    matched: pl.LazyFrame,
    target: Path,
    row_column: str,
    zone_column: str,
) -> None:
    target.mkdir()
    matched.select(row_column, zone_column).sink_parquet(
        pl.PartitionBy(target, key=zone_column, include_key=True),
        mkdir=True,
        maintain_order=True,
        engine="streaming",
    )


def _write_unmatched(
    source_dir: Path,
    matched_ids_dir: Path,
    target: Path,
    row_column: str,
    *,
    matched_zone_column: str = _ZONE,
    rename: dict[str, str] | None = None,
) -> None:
    matched_zones = _zone_paths(matched_ids_dir, matched_zone_column)
    for zone, source_path in _zone_paths(source_dir).items():
        source = _scan_partition(source_path)
        matched_path = matched_zones.get(zone)
        if matched_path is not None:
            # the anti-join holds IDs for one sky zone; if a single zone
            # exceeds RAM, subpartition these markers by original row-ID range.
            matched_ids = _scan_partition(matched_path).select(row_column).unique()
            source = source.join(matched_ids, on=row_column, how="anti")
        if rename:
            overlaps = set(source.collect_schema().names()) & set(rename)
            source = source.rename({name: rename[name] for name in overlaps})
        source.sink_parquet(target / f"{zone:08d}.parquet", engine="streaming")


def _scan_or_empty(root: Path, empty: pl.LazyFrame) -> pl.LazyFrame:
    paths = sorted(root.glob("*.parquet"))
    return pl.scan_parquet(paths) if paths else empty
