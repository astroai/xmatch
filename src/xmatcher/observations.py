"""Lazy measurement ledgers and immutable, source-preserving Parquet releases.

Mappings describe upstream columns; they do not assert physical identity or
reduce conflicting evidence. No survey conventions are guessed here.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import sqlite3
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import polars as pl

from .sources import CatalogueSource

# Persisted v1 wire names stay fixed across the package rename to preserve IDs.
OBSERVATION_RELEASE_SCHEMA_VERSION = "xmatch.observations.release.v1"
SOURCE_ID_SCHEMA_VERSION = "xmatch.source.v1"
_FLUX_FACTORS = {"Jy": 1.0, "mJy": 1e-3, "uJy": 1e-6, "nanomaggy": 3.631e-6}
_STATUSES = {
    "measured",
    "upper_limit",
    "missing",
    "masked",
    "outside_footprint",
    "not_detected",
    "ambiguous",
    "invalid",
}
_CATEGORIES = {
    "measured",
    "inferred_spectroscopy",
    "inferred_photometry_astrometry",
    "literature_assertion",
    "prediction",
}
_PHOTOMETRY_SCHEMA = {
    "source_id": pl.String,
    "release_namespace": pl.String,
    "measurement_id": pl.String,
    "passband": pl.String,
    "native_value": pl.Float64,
    "native_error": pl.Float64,
    "native_unit": pl.String,
    "flux_jy": pl.Float64,
    "flux_error_jy": pl.Float64,
    "flux_error_lower_jy": pl.Float64,
    "flux_error_upper_jy": pl.Float64,
    "upper_limit_jy": pl.Float64,
    "status": pl.String,
    "measurement_method": pl.String,
    "calibration_id": pl.String,
    "observation_time": pl.Float64,
    "time_format": pl.String,
    "time_scale": pl.String,
    "quality": pl.String,
    "flags": pl.List(pl.String),
    "passband_version": pl.String,
    "passband_uri": pl.String,
    "limit_sigma": pl.Float64,
    "limit_confidence": pl.Float64,
    "limit_convention": pl.String,
}
_EVIDENCE_SCHEMA = {
    "source_id": pl.String,
    "release_namespace": pl.String,
    "evidence_id": pl.String,
    "property": pl.String,
    "native_value": pl.String,
    "native_value_type": pl.String,
    "native_error": pl.Float64,
    "native_unit": pl.String,
    "category": pl.String,
    "method": pl.String,
    "quality": pl.String,
    "reference_frame": pl.String,
    "doppler_convention": pl.String,
    "observation_time": pl.Float64,
    "time_format": pl.String,
    "time_scale": pl.String,
    "model_id": pl.String,
    "input_features_json": pl.String,
    "association_id": pl.String,
    "upstream_evidence_id": pl.String,
}


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty string without surrounding whitespace")
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def namespaced_source_id(release_namespace: str, native_id: str | int) -> str:
    """Stable identity over the explicit release namespace and typed native ID."""
    _text(release_namespace, "release_namespace")
    if isinstance(native_id, bool) or not isinstance(native_id, (str, int)):
        raise ValueError("native identifier must be a string or integer")
    if isinstance(native_id, str):
        _text(native_id, "native identifier")
    return f"{SOURCE_ID_SCHEMA_VERSION}:{_digest([release_namespace, type(native_id).__name__, native_id])}"


def _identity(source: CatalogueSource) -> tuple[pl.LazyFrame, pl.Expr]:
    _text(source.release_namespace, "release_namespace")
    _text(source.id_column, "id_column")
    frame = source.lazy()
    schema = frame.collect_schema()
    if source.id_column not in schema:
        raise ValueError(f"native id_column {source.id_column!r} is missing")
    dtype = schema[source.id_column]
    if dtype != pl.String and not dtype.is_integer():
        raise ValueError("native id_column must have string or integer dtype")
    # SHA-256 is a lazy Python UDF; use a compiled stable hash only
    # if profiling shows this boundary dominates ingestion. Never use row IDs.
    identity = pl.col(source.id_column).map_elements(
        lambda value: namespaced_source_id(source.release_namespace, value),
        return_dtype=pl.String,
        skip_nulls=False,
    )
    return frame, identity


def _column(
    frame: pl.LazyFrame, name: str | None, dtype: pl.DataType | type[pl.DataType] = pl.Float64
) -> pl.Expr:
    if name is None:
        return pl.lit(None, dtype=dtype)
    if name not in frame.collect_schema():
        raise ValueError(f"mapped column {name!r} is missing")
    return pl.col(name).cast(dtype)


def source_inventory(source: CatalogueSource) -> pl.LazyFrame:
    """One row per native source, including sources with no photometry.

    All original fields are retained in ``native``. Missing astrometry is null;
    epoch is the explicitly declared astrometric reference epoch, not equinox.
    """
    frame, identity = _identity(source)
    return frame.select(
        identity.alias("source_id"),
        pl.lit(source.release_namespace).alias("release_namespace"),
        pl.col(source.id_column).cast(pl.String).alias("native_id"),
        pl.lit(str(frame.collect_schema()[source.id_column])).alias("native_id_type"),
        _column(frame, source.ra_column).alias("ra_deg"),
        _column(frame, source.dec_column).alias("dec_deg"),
        (
            _column(frame, source.epoch_column)
            if source.epoch_column
            else pl.lit(source.epoch, dtype=pl.Float64)
        ).alias("epoch_jyear"),
        pl.struct(pl.all()).alias("native"),
    )


def required_observation_columns(source: CatalogueSource) -> list[str]:
    """Explicit acquisition selection for source, photometry and evidence fields.

    This does not modify matching defaults or download additional data.
    """
    columns = [
        getattr(source, name)
        for name in (
            "id_column",
            "ra_column",
            "dec_column",
            "epoch_column",
            "pm_ra_column",
            "pm_dec_column",
            "parallax_column",
            "radial_velocity_column",
            "ra_err_column",
            "dec_err_column",
            "corr_column",
        )
    ]
    columns.extend((source.astrometric_covariance_columns or {}).values())
    for mapping in (*source.photometry, *source.property_evidence):
        columns.extend(value for key, value in mapping.items() if key.endswith("_column"))
    return list(
        dict.fromkeys(_text(column, "mapped column") for column in columns if column is not None)
    )


def _mapping(
    raw: Mapping[str, Any], allowed: set[str], required: tuple[str, ...]
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("measurement mappings must be dictionaries")
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown measurement mapping fields: {sorted(unknown)}")
    for name in required:
        _text(raw.get(name), name)
    result = dict(raw)
    for name in ("observation_time", "zeropoint_jy", "limit_sigma", "limit_confidence"):
        if result.get(name) is not None:
            value = result[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
    if result.get("zeropoint_jy") is not None and result["zeropoint_jy"] <= 0:
        raise ValueError("zeropoint_jy must be positive")
    if result.get("limit_sigma") is not None and result["limit_sigma"] <= 0:
        raise ValueError("limit_sigma must be positive")
    if result.get("limit_confidence") is not None and not 0 < result["limit_confidence"] < 1:
        raise ValueError("limit_confidence must be between zero and one")
    if (
        result.get("observation_time_column") is not None
        or result.get("observation_time") is not None
    ):
        _text(result.get("time_format"), "time_format")
        _text(result.get("time_scale"), "time_scale")
    return result


def _time(frame: pl.LazyFrame, mapping: Mapping[str, Any]) -> pl.Expr:
    return (
        _column(frame, mapping["observation_time_column"])
        if mapping.get("observation_time_column")
        else pl.lit(mapping.get("observation_time"), dtype=pl.Float64)
    )


def _nullable_text(value: Any) -> pl.Expr:
    return pl.lit(value, dtype=pl.String)


def _record_id(identity: pl.Expr, mapping: Mapping[str, Any], prefix: str) -> pl.Expr:
    mapping_hash = _digest(mapping)
    return identity.map_elements(
        lambda value: f"{prefix}:{_digest([value, mapping_hash])}", return_dtype=pl.String
    )


def normalize_photometry(source: CatalogueSource) -> pl.LazyFrame:
    """Long native photometry with supported flux-density conversions to Jy.

    Unknown units and Vega magnitudes without a supplied zero point remain
    native-only. Upper limits are never generated from missing rows.
    """
    frame, identity = _identity(source)
    rows = []
    seen = set()
    allowed = {
        "passband",
        "passband_version",
        "passband_uri",
        "value_column",
        "error_column",
        "unit",
        "method",
        "calibration_id",
        "zeropoint_jy",
        "status_column",
        "status_values",
        "quality_column",
        "observation_time_column",
        "observation_time",
        "time_format",
        "time_scale",
        "limit_sigma",
        "limit_confidence",
        "limit_convention",
    }
    for raw in source.photometry:
        mapping = _mapping(raw, allowed, ("passband", "value_column", "unit"))
        mapping_id = _digest(mapping)
        if mapping_id in seen:
            raise ValueError("duplicate photometry mapping creates duplicate measurement identity")
        seen.add(mapping_id)
        value = _column(frame, mapping["value_column"])
        error = _column(frame, mapping.get("error_column"))
        valid_value = value.is_not_null() & value.is_finite()
        valid_error = error.is_not_null() & error.is_finite() & (error >= 0)
        if mapping.get("status_column"):
            status_values = mapping.get("status_values")
            if (
                not isinstance(status_values, dict)
                or not status_values
                or not set(status_values.values()) <= _STATUSES
            ):
                raise ValueError(
                    "status_column requires explicit status_values using canonical statuses"
                )
            status = _column(frame, mapping["status_column"], pl.String).replace_strict(
                status_values, default="invalid", return_dtype=pl.String
            )
        else:
            status = (
                pl.when(value.is_null())
                .then(pl.lit("missing"))
                .when(valid_value)
                .then(pl.lit("measured"))
                .otherwise(pl.lit("invalid"))
            )
        status = (
            pl.when(status.is_in(["measured", "upper_limit"]) & ~valid_value)
            .then(pl.lit("invalid"))
            .otherwise(status)
        )
        unit = mapping["unit"]
        factor = _FLUX_FACTORS.get(unit)
        zero = 3631.0 if unit == "ABmag" else mapping.get("zeropoint_jy")
        magnitude = unit in {"ABmag", "Vegamag"} and zero is not None
        if factor is not None:
            flux = value * factor
            sigma = error * factor
            lower, upper = sigma, sigma
        elif magnitude:
            log_zero = math.log10(zero)
            flux = pl.lit(10.0).pow(log_zero - 0.4 * value)
            sigma = pl.lit(None, dtype=pl.Float64)
            lower = flux - pl.lit(10.0).pow(log_zero - 0.4 * (value + error))
            upper = pl.lit(10.0).pow(log_zero - 0.4 * (value - error)) - flux
        else:
            flux = sigma = lower = upper = pl.lit(None, dtype=pl.Float64)
        conversion_nonfinite = valid_value & flux.is_not_null() & ~flux.is_finite()
        uncertainty_nonfinite = (
            valid_value
            & valid_error
            & (factor is not None or magnitude)
            & (
                (lower.is_not_null() & ~lower.is_finite())
                | (upper.is_not_null() & ~upper.is_finite())
            )
        )
        flux = pl.when(flux.is_finite()).then(flux).otherwise(None)
        invalid_upper_limit = (status == "upper_limit") & flux.is_not_null() & (flux <= 0)
        status = pl.when(invalid_upper_limit).then(pl.lit("invalid")).otherwise(status)
        measured = status == "measured"
        flags = pl.concat_list(
            pl.when(error.is_not_null() & ~valid_error)
            .then(pl.lit("invalid_uncertainty"))
            .otherwise(pl.lit(None, dtype=pl.String)),
            pl.when(valid_value & (factor is None) & (not magnitude))
            .then(pl.lit("native_only"))
            .otherwise(pl.lit(None, dtype=pl.String)),
            pl.when(conversion_nonfinite)
            .then(pl.lit("conversion_nonfinite"))
            .otherwise(pl.lit(None, dtype=pl.String)),
            pl.when(uncertainty_nonfinite)
            .then(pl.lit("conversion_uncertainty_nonfinite"))
            .otherwise(pl.lit(None, dtype=pl.String)),
            pl.when(invalid_upper_limit)
            .then(pl.lit("invalid_upper_limit"))
            .otherwise(pl.lit(None, dtype=pl.String)),
        ).list.drop_nulls()
        rows.append(
            frame.select(
                identity.alias("source_id"),
                pl.lit(source.release_namespace).alias("release_namespace"),
                _record_id(identity, mapping, "xmatch.photometry.v1").alias("measurement_id"),
                pl.lit(mapping["passband"]).alias("passband"),
                value.alias("native_value"),
                error.alias("native_error"),
                pl.lit(unit).alias("native_unit"),
                pl.when(measured).then(flux).otherwise(None).cast(pl.Float64).alias("flux_jy"),
                pl.when(measured & valid_error & sigma.is_finite())
                .then(sigma)
                .otherwise(None)
                .cast(pl.Float64)
                .alias("flux_error_jy"),
                pl.when(measured & valid_error & lower.is_finite())
                .then(lower)
                .otherwise(None)
                .cast(pl.Float64)
                .alias("flux_error_lower_jy"),
                pl.when(measured & valid_error & upper.is_finite())
                .then(upper)
                .otherwise(None)
                .cast(pl.Float64)
                .alias("flux_error_upper_jy"),
                pl.when(status == "upper_limit")
                .then(flux)
                .otherwise(None)
                .cast(pl.Float64)
                .alias("upper_limit_jy"),
                status.alias("status"),
                pl.lit(mapping.get("method", "unknown")).alias("measurement_method"),
                _nullable_text(mapping.get("calibration_id")).alias("calibration_id"),
                _time(frame, mapping).alias("observation_time"),
                _nullable_text(mapping.get("time_format")).alias("time_format"),
                _nullable_text(mapping.get("time_scale")).alias("time_scale"),
                _column(frame, mapping.get("quality_column"), pl.String).alias("quality"),
                flags.alias("flags"),
                _nullable_text(mapping.get("passband_version")).alias("passband_version"),
                _nullable_text(mapping.get("passband_uri")).alias("passband_uri"),
                pl.lit(mapping.get("limit_sigma"), dtype=pl.Float64).alias("limit_sigma"),
                pl.lit(mapping.get("limit_confidence"), dtype=pl.Float64).alias("limit_confidence"),
                _nullable_text(mapping.get("limit_convention")).alias("limit_convention"),
            )
        )
    return pl.concat(rows) if rows else pl.DataFrame(schema=_PHOTOMETRY_SCHEMA).lazy()


def normalize_property_evidence(source: CatalogueSource) -> pl.LazyFrame:
    """Preserve raw classifications/properties and their declared provenance.

    Values are typed native strings, including categorical codes; the complete
    native row remains in source_inventory. No unit/frame transformation occurs.
    """
    frame, identity = _identity(source)
    rows = []
    seen = set()
    allowed = {
        "property",
        "value_column",
        "error_column",
        "unit",
        "category",
        "method",
        "quality_column",
        "reference_frame",
        "doppler_convention",
        "observation_time_column",
        "observation_time",
        "time_format",
        "time_scale",
        "model_id",
        "input_features",
        "association_id_column",
        "upstream_evidence_id_column",
    }
    for raw in source.property_evidence:
        mapping = _mapping(raw, allowed, ("property", "value_column", "category", "method"))
        mapping_id = _digest(mapping)
        if mapping_id in seen:
            raise ValueError("duplicate property evidence mapping")
        seen.add(mapping_id)
        if mapping["category"] not in _CATEGORIES:
            raise ValueError(f"unsupported evidence category: {mapping['category']}")
        if mapping["category"] == "prediction":
            _text(mapping.get("model_id"), "prediction model_id")
        value_column = mapping["value_column"]
        if value_column not in frame.collect_schema():
            raise ValueError(f"mapped column {value_column!r} is missing")
        dtype = frame.collect_schema()[value_column]
        if dtype.is_nested():
            native_value = pl.col(value_column).map_elements(
                lambda value: json.dumps(
                    value.to_list() if isinstance(value, pl.Series) else value
                ),
                return_dtype=pl.String,
            )
        else:
            native_value = _column(frame, value_column, pl.String)
        rows.append(
            frame.select(
                identity.alias("source_id"),
                pl.lit(source.release_namespace).alias("release_namespace"),
                _record_id(identity, mapping, "xmatch.property.evidence.v1").alias("evidence_id"),
                pl.lit(mapping["property"]).alias("property"),
                native_value.alias("native_value"),
                pl.lit(str(frame.collect_schema()[mapping["value_column"]])).alias(
                    "native_value_type"
                ),
                _column(frame, mapping.get("error_column")).alias("native_error"),
                _nullable_text(mapping.get("unit")).alias("native_unit"),
                pl.lit(mapping["category"]).alias("category"),
                pl.lit(mapping["method"]).alias("method"),
                _column(frame, mapping.get("quality_column"), pl.String).alias("quality"),
                pl.lit(mapping.get("reference_frame", "unknown")).alias("reference_frame"),
                pl.lit(mapping.get("doppler_convention", "unknown")).alias("doppler_convention"),
                _time(frame, mapping).alias("observation_time"),
                _nullable_text(mapping.get("time_format")).alias("time_format"),
                _nullable_text(mapping.get("time_scale")).alias("time_scale"),
                _nullable_text(mapping.get("model_id")).alias("model_id"),
                pl.lit(json.dumps(mapping.get("input_features", []), sort_keys=True)).alias(
                    "input_features_json"
                ),
                _column(frame, mapping.get("association_id_column"), pl.String).alias(
                    "association_id"
                ),
                _column(frame, mapping.get("upstream_evidence_id_column"), pl.String).alias(
                    "upstream_evidence_id"
                ),
            )
        )
    return pl.concat(rows) if rows else pl.DataFrame(schema=_EVIDENCE_SCHEMA).lazy()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def validate_source_inventory(path: str | Path) -> None:
    """Disk-backed uniqueness check with a bounded Arrow batch/cache size."""
    import pyarrow.parquet as pq

    path = Path(path)
    with tempfile.TemporaryDirectory(prefix=".identity-", dir=path.parent) as directory:
        database = Path(directory) / "ids.sqlite"
        with sqlite3.connect(database) as connection:
            connection.execute("PRAGMA cache_size = -8192")
            connection.execute("CREATE TABLE ids (id TEXT PRIMARY KEY) WITHOUT ROWID")
            for batch in pq.ParquetFile(path).iter_batches(batch_size=65536, columns=["source_id"]):
                try:
                    connection.executemany(
                        "INSERT INTO ids VALUES (?)",
                        ((value,) for value in batch.column(0).to_pylist()),
                    )
                except sqlite3.IntegrityError as exc:
                    raise ValueError("duplicate native identity in release_namespace") from exc


def write_observation_release(
    path: str | Path,
    sources: Iterable[CatalogueSource],
    *,
    release_id: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Stream per-namespace Parquet shards and atomically publish a manifest.

    Never replace a release. Duplicate namespaces/native IDs fail publication.
    Provenance must supply real acquisition/checksum/software facts; this
    function records it without claiming independent verification of upstreams.
    """
    _text(release_id, "release_id")
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.parent / f".{target.name}.lock"
    lock.mkdir()  # exclusive among writers; no partial directory is published
    temporary = None
    try:
        temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
        manifest: dict[str, Any] = {
            "schema_version": OBSERVATION_RELEASE_SCHEMA_VERSION,
            "release_id": release_id,
            "provenance": dict(provenance),
            "sources": [],
        }
        namespaces: set[str] = set()
        for index, source in enumerate(sources):
            namespace = _text(source.release_namespace, "release_namespace")
            if namespace in namespaces:
                raise ValueError(
                    "each release_namespace must occur once; concatenate same-release inputs explicitly"
                )
            namespaces.add(namespace)
            shard = temporary / f"source-{index:04d}"
            shard.mkdir()
            entry = {
                "name": source.name,
                "release_namespace": namespace,
                "release_metadata": source.release_metadata,
                "source_metadata": {
                    name: getattr(source, name)
                    for name in (
                        "id_column",
                        "ra_column",
                        "dec_column",
                        "epoch",
                        "epoch_column",
                        "frame",
                        "pm_ra_column",
                        "pm_dec_column",
                        "parallax_column",
                        "radial_velocity_column",
                        "ra_err_column",
                        "dec_err_column",
                        "corr_column",
                        "astrometric_covariance_columns",
                        "pos_err_units",
                        "default_pos_error_arcsec",
                    )
                },
                "photometry": source.photometry,
                "property_evidence": source.property_evidence,
            }
            inventory = source_inventory(source)
            for kind, frame in (
                ("inventory", inventory),
                ("photometry", normalize_photometry(source)),
                ("evidence", normalize_property_evidence(source)),
            ):
                destination = shard / f"{kind}.parquet"
                frame.sink_parquet(destination, engine="streaming")
                if kind == "inventory":
                    validate_source_inventory(destination)
                count = pl.scan_parquet(destination).select(pl.len()).collect().item()
                entry[f"{kind}_path"] = str(destination.relative_to(temporary))
                entry[f"{kind}_count"] = count
                entry[f"{kind}_sha256"] = _file_sha256(destination)
                entry[f"{kind}_schema"] = {
                    name: str(dtype)
                    for name, dtype in pl.scan_parquet(destination).collect_schema().items()
                }
            entry["source_count"] = entry["inventory_count"]
            manifest["sources"].append(entry)
        if not manifest["sources"]:
            raise ValueError("sources must contain at least one explicit catalogue")
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
        temporary.rename(target)
        return manifest
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
        lock.rmdir()


def verify_observation_release(path: str | Path) -> dict[str, Any]:
    """Verify local shard checksums, row counts and recorded Parquet schemas."""
    root = Path(path).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != OBSERVATION_RELEASE_SCHEMA_VERSION:
        raise ValueError("unsupported observation release schema")
    _text(manifest.get("release_id"), "release_id")
    for source in manifest["sources"]:
        for kind in ("inventory", "photometry", "evidence"):
            relative = Path(source[f"{kind}_path"])
            candidate = (root / relative).resolve()
            if relative.is_absolute() or not candidate.is_relative_to(root):
                raise ValueError("observation shard path escapes release directory")
            if _file_sha256(candidate) != source[f"{kind}_sha256"]:
                raise ValueError(f"checksum mismatch for {relative}")
            frame = pl.scan_parquet(candidate)
            schema = {name: str(dtype) for name, dtype in frame.collect_schema().items()}
            if (
                schema != source[f"{kind}_schema"]
                or frame.select(pl.len()).collect().item() != source[f"{kind}_count"]
            ):
                raise ValueError(f"schema/count mismatch for {relative}")
    return manifest
