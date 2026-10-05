"""Source-conserving sparse candidates and per-source association alternatives.

Candidate connectivity is not physical identity. Fixed-radius searches describe
only the declared search region; callers supply uncertainty/motion bounds and
scientific likelihoods before adopting associations.
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

import polars as pl

from .exceptions import CrossMatchError
from .io_utils import FrameLike, to_lazy
from .matchers import MatchSpec
from .out_of_core import match_to_output
from .request import MatchRequest
from .sources import CatalogueSource

_SCHEMA = "xmatch.candidates.release.v1"
_PAIR_SCHEMA = {
    "source_id": pl.String,
    "candidate_id": pl.String,
    "source_namespace": pl.String,
    "candidate_namespace": pl.String,
    "sep_arcsec": pl.Float64,
    "evaluation_epoch_jyear": pl.Float64,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_candidate_release(
    directory: str | Path,
    sources: Sequence[CatalogueSource],
    *,
    radius_arcsec: float = 1.0,
    target_epoch: float | None = None,
    memory_budget_bytes: int = 64 * 1024 * 1024,
    partition_order: str | int = "auto",
    survey_pairs: Iterable[tuple[str, str]] | None = None,
    provenance: dict | None = None,
) -> dict:
    """Publish all admissible pairs plus an inventory including isolated sources.

    Inputs must be local and release-namespaced. Matching reuses the spill
    engine's partition/batch memory proxy; the budget is not an RSS guarantee
    for Polars scans/sorts. Target-epoch searches require explicit finite motion
    on rows whose reference epoch differs. Raw positions remain in the source
    inventory. No probability or adopted physical identity is manufactured.
    """
    from .observations import source_inventory, validate_source_inventory

    output = Path(directory)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if not sources or any(not source.is_local for source in sources):
        raise ValueError("sources must contain local CatalogueSource inputs")
    if any(source.frame.lower() != "icrs" for source in sources):
        raise ValueError(
            "candidate releases require ICRS coordinates; transform other frames first"
        )
    if not math.isfinite(radius_arcsec) or not 0 < radius_arcsec <= 180 * 3600:
        raise ValueError("radius_arcsec must be finite and in (0, 648000]")
    if isinstance(memory_budget_bytes, bool) or memory_budget_bytes < 1024:
        raise ValueError("memory_budget_bytes must be at least 1024")
    if target_epoch is not None and (not math.isfinite(target_epoch) or target_epoch <= 0):
        raise ValueError("target_epoch must be a finite positive Julian year")
    raw_namespaces = [source.release_namespace for source in sources]
    if any(not isinstance(value, str) or not value.strip() for value in raw_namespaces):
        raise ValueError("every source needs a release_namespace")
    namespaces = [str(value) for value in raw_namespaces]
    if len(set(namespaces)) != len(namespaces):
        raise ValueError("release_namespace must be unique across sources")
    ordered = sorted(zip(namespaces, sources, strict=True), key=lambda item: item[0])
    namespaces = [name for name, _ in ordered]
    combinations = list(itertools.combinations(namespaces, 2))
    if survey_pairs is not None:
        requested = {tuple(sorted(pair)) for pair in survey_pairs}
        if not requested <= set(combinations):
            raise ValueError("survey_pairs must name distinct registered release namespaces")
        combinations = sorted(requested)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / f".{output.name}.lock"
    lock.mkdir()
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        scratch = temporary / "scratch"
        scratch.mkdir()
        normalized: dict[str, CatalogueSource] = {}
        inventories = []
        for index, (namespace, source) in enumerate(ordered):
            inventory = source_inventory(source)
            coords = inventory.select(
                "source_id",
                "release_namespace",
                "native_id",
                "native_id_type",
                "ra_deg",
                "dec_deg",
                "epoch_jyear",
            )
            invalid = (
                coords.select(
                    (
                        pl.col("ra_deg").is_null()
                        | ~pl.col("ra_deg").is_finite()
                        | pl.col("dec_deg").is_null()
                        | ~pl.col("dec_deg").is_finite()
                        | ~pl.col("dec_deg").is_between(-90, 90)
                    ).sum()
                )
                .collect(engine="streaming")
                .item()
            )
            if invalid:
                raise ValueError(f"{namespace} contains invalid coordinates")
            inventories.append(coords)
            motion_fields = {
                "pm_ra_column": source.pm_ra_column,
                "pm_dec_column": source.pm_dec_column,
                "parallax_column": source.parallax_column,
                "radial_velocity_column": source.radial_velocity_column,
            }
            motion_aliases = {
                key: f"_candidate_{key}" for key, name in motion_fields.items() if name
            }
            projected = inventory.select(
                "source_id",
                "release_namespace",
                "ra_deg",
                "dec_deg",
                "epoch_jyear",
                *[
                    pl.col("native").struct.field(name).alias(motion_aliases[key])
                    for key, name in motion_fields.items()
                    if name
                ],
            )
            path = scratch / f"input-{index}.parquet"
            projected.sink_parquet(path, engine="streaming")
            validate_source_inventory(path)
            normalized[namespace] = dataclasses.replace(
                source,
                path=path,
                _frame=None,
                id_column="source_id",
                ra_column="ra_deg",
                dec_column="dec_deg",
                epoch_column="epoch_jyear"
                if source.epoch_column or source.epoch is not None
                else None,
                **motion_aliases,
            )
        pair_paths = []
        statistics = []
        for index, (left_namespace, right_namespace) in enumerate(combinations):
            left, right = normalized[left_namespace], normalized[right_namespace]
            raw_path = scratch / f"matches-{index}.parquet"
            request = MatchRequest(
                left.path,
                right.path,
                spec=MatchSpec(
                    radius_arcsec=radius_arcsec,
                    find="all",
                    target_epoch=target_epoch,
                    fallback_policy="error",
                ),
                engine="fast",
                output_file=raw_path,
                memory_budget_bytes=memory_budget_bytes,
                partition_order=partition_order,
                scratch_dir=scratch,
            )
            try:
                statistics.append(match_to_output(request, left, right))
            except CrossMatchError as exc:
                raise ValueError(str(exc)) from exc
            pair_path = scratch / f"pairs-{index}.parquet"
            pl.scan_parquet(raw_path).select(
                "source_id",
                pl.col("source_id_2").alias("candidate_id"),
                pl.col("release_namespace").alias("source_namespace"),
                pl.col("release_namespace_2").alias("candidate_namespace"),
                "sep_arcsec",
                pl.lit(target_epoch, dtype=pl.Float64).alias("evaluation_epoch_jyear"),
            ).sink_parquet(pair_path, engine="streaming")
            pair_paths.append(pair_path)
        pairs = (
            pl.concat([pl.scan_parquet(path) for path in pair_paths])
            if pair_paths
            else pl.LazyFrame(schema=_PAIR_SCHEMA)
        )
        pairs.sort("source_id", "candidate_id").sink_parquet(
            temporary / "candidates.parquet", engine="streaming"
        )
        pairs = pl.scan_parquet(temporary / "candidates.parquet")
        endpoints = (
            pl.concat(
                [pairs.select("source_id"), pairs.select(pl.col("candidate_id").alias("source_id"))]
            )
            .group_by("source_id")
            .agg(pl.len().cast(pl.UInt64).alias("candidate_count"))
        )
        pl.concat(inventories).join(endpoints, on="source_id", how="left").with_columns(
            pl.col("candidate_count").fill_null(0)
        ).sort("source_id").sink_parquet(temporary / "sources.parquet", engine="streaming")
        files = {
            name: {
                "sha256": _sha256(temporary / name),
                "rows": pl.scan_parquet(temporary / name)
                .select(pl.len())
                .collect(engine="streaming")
                .item(),
            }
            for name in ("sources.parquet", "candidates.parquet")
        }
        manifest = {
            "schema_version": _SCHEMA,
            "source_count": files["sources.parquet"]["rows"],
            "candidate_count": files["candidates.parquet"]["rows"],
            "input_release_ids": namespaces,
            "files": files,
            "parameters": {
                "radius_arcsec": radius_arcsec,
                "target_epoch": target_epoch,
                "survey_pairs": combinations,
            },
            "score_semantics": "ranking_score",
            "association_state": "unevaluated_candidates",
            "provenance": provenance or {},
            "execution": statistics,
        }
        manifest["release_id"] = (
            f"{_SCHEMA}:"
            + hashlib.sha256(
                json.dumps(
                    {key: value for key, value in manifest.items() if key != "execution"},
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode()
            ).hexdigest()
        )
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n"
        )
        shutil.rmtree(scratch)
        manifest = verify_candidate_release(temporary)
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        os.rename(temporary, output)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
        lock.rmdir()


def verify_candidate_release(directory: str | Path) -> dict:
    """Verify immutable payload hashes, counts and endpoint referential integrity."""
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") != _SCHEMA:
        raise ValueError("unsupported candidate release schema")
    if set(manifest["files"]) != {"sources.parquet", "candidates.parquet"}:
        raise ValueError("unexpected candidate release files")
    for name, metadata in manifest["files"].items():
        if _sha256(root / name) != metadata["sha256"]:
            raise ValueError(f"checksum mismatch for {name}")
        if (
            pl.scan_parquet(root / name).select(pl.len()).collect(engine="streaming").item()
            != metadata["rows"]
        ):
            raise ValueError(f"row count mismatch for {name}")
    sources, pairs = (
        pl.scan_parquet(root / "sources.parquet"),
        pl.scan_parquet(root / "candidates.parquet"),
    )
    if (
        manifest["source_count"] != manifest["files"]["sources.parquet"]["rows"]
        or manifest["candidate_count"] != manifest["files"]["candidates.parquet"]["rows"]
    ):
        raise ValueError("candidate release summary count mismatch")
    radius = manifest["parameters"]["radius_arcsec"]
    if (
        not isinstance(radius, (int, float))
        or not math.isfinite(radius)
        or not 0 < radius <= 648000
    ):
        raise ValueError("invalid candidate release radius")
    endpoints = pl.concat(
        [pairs.select("source_id"), pairs.select(pl.col("candidate_id").alias("source_id"))]
    )
    if (
        endpoints.join(sources.select("source_id"), on="source_id", how="anti")
        .select(pl.len())
        .collect(engine="streaming")
        .item()
    ):
        raise ValueError("unknown candidate release endpoint")
    joined = pairs.join(
        sources.select("source_id", pl.col("release_namespace").alias("_actual_source")),
        on="source_id",
        how="left",
    ).join(
        sources.select(
            pl.col("source_id").alias("candidate_id"),
            pl.col("release_namespace").alias("_actual_candidate"),
        ),
        on="candidate_id",
        how="left",
    )
    invalid = (
        joined.select(
            (
                pl.col("source_id").is_null()
                | pl.col("candidate_id").is_null()
                | pl.col("source_namespace").is_null()
                | pl.col("candidate_namespace").is_null()
                | (pl.col("source_id") == pl.col("candidate_id"))
                | (pl.col("source_namespace") != pl.col("_actual_source"))
                | (pl.col("candidate_namespace") != pl.col("_actual_candidate"))
                | (pl.col("source_namespace") == pl.col("candidate_namespace"))
                | pl.col("sep_arcsec").is_null()
                | ~pl.col("sep_arcsec").is_finite()
                | ~pl.col("sep_arcsec").is_between(0, radius)
            ).any()
        )
        .collect(engine="streaming")
        .item()
    )
    if invalid:
        raise ValueError("invalid candidate separation or endpoint namespace")
    if (
        pairs.select(pl.struct("source_id", "candidate_id").is_duplicated().any())
        .collect(engine="streaming")
        .item()
    ):
        raise ValueError("duplicate candidate endpoint pair")
    counts = endpoints.group_by("source_id").agg(pl.len().cast(pl.UInt64).alias("_actual_count"))
    invalid_count = (
        sources.join(counts, on="source_id", how="left")
        .select(
            (
                pl.col("candidate_count").is_null()
                | (pl.col("candidate_count") != pl.col("_actual_count").fill_null(0))
            ).any()
        )
        .collect(engine="streaming")
        .item()
    )
    if invalid_count:
        raise ValueError("source candidate_count mismatch")
    identity = {
        key: value for key, value in manifest.items() if key not in {"release_id", "execution"}
    }
    expected = (
        f"{_SCHEMA}:"
        + hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    )
    if manifest["release_id"] != expected:
        raise ValueError("candidate release manifest identity mismatch")
    return manifest


def normalize_candidate_hypotheses(
    sources: FrameLike,
    candidates: FrameLike,
    *,
    candidate_namespace: str,
    log_weight_column: str = "log_weight",
    no_match_log_weight: float = 0.0,
) -> pl.LazyFrame:
    """Normalize explicitly supplied log weights with a no-match alternative.

    These are alternatives within one named target survey, not a global
    one-to-one assignment or a blend model. Release pairs are oriented against
    that target; observations from distinct surveys must not compete as if only
    one survey could observe a source. Pair-only inputs without namespace columns
    must already represent exactly the declared target population.
    Weights must incorporate the caller's declared likelihood and
    priors (including search/coverage selection). No-match means no counterpart
    in that declared candidate population. Normalization is not calibration.
    """
    if not isinstance(candidate_namespace, str) or not candidate_namespace.strip():
        raise ValueError("candidate_namespace must explicitly identify the target population")
    inventory = to_lazy(sources)
    raw_pairs = to_lazy(candidates)
    namespace_columns = {"source_namespace", "candidate_namespace"}
    names = set(raw_pairs.collect_schema().names())
    if namespace_columns & names:
        if not namespace_columns <= names or "release_namespace" not in inventory.collect_schema():
            raise ValueError(
                "namespaced pairs require complete namespace columns and inventory namespaces"
            )
        forward = raw_pairs.filter(pl.col("candidate_namespace") == candidate_namespace).select(
            "source_id", "candidate_id", log_weight_column
        )
        reverse = raw_pairs.filter(pl.col("source_namespace") == candidate_namespace).select(
            pl.col("candidate_id").alias("source_id"),
            pl.col("source_id").alias("candidate_id"),
            log_weight_column,
        )
        raw_pairs = pl.concat([forward, reverse])
        inventory = inventory.filter(pl.col("release_namespace") != candidate_namespace)
    inventory = inventory.select("source_id")
    pairs = raw_pairs.select(
        "source_id", "candidate_id", pl.col(log_weight_column).cast(pl.Float64).alias("log_weight")
    )
    if math.isnan(no_match_log_weight) or no_match_log_weight == math.inf:
        raise ValueError("no-match log weight must be finite or negative infinity")
    if (
        inventory.select(
            pl.col("source_id").is_null().any() | pl.col("source_id").is_duplicated().any()
        )
        .collect(engine="streaming")
        .item()
    ):
        raise ValueError("source IDs must be unique and non-null")
    invalid = (
        pairs.select(
            (
                pl.col("log_weight").is_null()
                | pl.col("log_weight").is_nan()
                | (pl.col("log_weight") == math.inf)
                | pl.col("source_id").is_null()
                | pl.col("candidate_id").is_null()
            ).any()
        )
        .collect(engine="streaming")
        .item()
    )
    if invalid:
        raise ValueError("candidate endpoints and log weights must be valid")
    if (
        pairs.join(inventory, on="source_id", how="anti")
        .select(pl.len())
        .collect(engine="streaming")
        .item()
    ):
        raise ValueError("candidate references an unknown source")
    if (
        pairs.select(pl.struct("source_id", "candidate_id").is_duplicated().any())
        .collect(engine="streaming")
        .item()
    ):
        raise ValueError("duplicate candidate hypothesis")
    no_match = inventory.with_columns(
        pl.lit(None, dtype=pl.String).alias("candidate_id"),
        pl.lit(no_match_log_weight, dtype=pl.Float64).alias("log_weight"),
    )
    alternatives = pl.concat([pairs, no_match]).with_columns(
        pl.col("log_weight").max().over("source_id").alias("_max_weight")
    )
    if (
        alternatives.select((pl.col("_max_weight") == -math.inf).any())
        .collect(engine="streaming")
        .item()
    ):
        raise ValueError("every source needs at least one finite log weight")
    alternatives = alternatives.with_columns(
        (pl.col("log_weight") - pl.col("_max_weight")).exp().alias("_weight")
    )
    return (
        alternatives.with_columns(
            (pl.col("_weight") / pl.col("_weight").sum().over("source_id")).alias("probability"),
            pl.when(pl.col("candidate_id").is_null())
            .then(pl.lit("no_match"))
            .otherwise(pl.lit("counterpart"))
            .alias("hypothesis_kind"),
            pl.lit("assumed_prior_posterior").alias("score_semantics"),
            pl.lit(candidate_namespace).alias("candidate_namespace"),
        )
        .drop("_max_weight", "_weight")
        .sort("source_id", "candidate_id")
    )
