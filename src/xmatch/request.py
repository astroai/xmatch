"""Typed request objects for xmatch crossmatch operations.

The :class:`MatchRequest` dataclass consolidates all match parameters into a
fully-typed specification. Side-level column overrides are bundled into
:class:`SideOverrides` so the orchestrator never needs to re-derive the same
dict twice.

.. code-block:: python

    from xmatch import CrossMatch, MatchRequest, MatchSpec, SideOverrides

    cm = CrossMatch()
    req = MatchRequest(
        cat1="gaia",
        cat2="wise.parquet",
        spec=MatchSpec(radius_arcsec=2.0, matcher="skyerr"),
        side2=SideOverrides(ra_column="RAJ2000", dec_column="DEJ2000"),
    )
    cm.crossmatch_request(req)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from .matchers import MatchSpec

FrameInput = str | Path | pl.DataFrame | pl.LazyFrame


@dataclass
class SideOverrides:
    """Per-catalogue column overrides supplied by the user.

    Every field is optional — when ``None`` the resolver falls back to auto-
    detection or the YAML config defaults.
    """

    ra_column: str | None = None
    """Right-ascension column name on this side."""

    dec_column: str | None = None
    """Declination column name on this side."""

    id_column: str | None = None
    """Identifier column name on this side."""

    ra_err_column: str | None = None
    """Right-ascension uncertainty column for local ``skyerr`` matching."""

    dec_err_column: str | None = None
    """Declination uncertainty column for local ``skyerr`` matching."""

    corr_column: str | None = None
    """RA/Dec uncertainty correlation column for ``skyellipse`` matching."""

    astrometric_covariance_columns: dict[str, str] | None = None
    """Canonical Gaia-style five-parameter error/correlation column mapping."""

    pos_err_units: str | None = None
    """Units for local positional-error columns (arcsec, mas, arcmin, or deg)."""

    default_pos_error_arcsec: float | None = None
    """Fallback one-axis positional error for local rows, in arcseconds."""

    epoch: float | None = None
    """Catalogue-level Julian-year epoch."""

    epoch_column: str | None = None
    """Per-row Julian-year epoch column."""

    pm_ra_column: str | None = None
    """Cosine-weighted right-ascension proper-motion column, in mas/yr."""

    pm_dec_column: str | None = None
    """Declination proper-motion column, in mas/yr."""

    parallax_column: str | None = None
    """Parallax column, in mas, for complete space-motion propagation."""

    radial_velocity_column: str | None = None
    """Barycentric radial-velocity column, in km/s."""

    columns: list[str] | None = None
    """Columns to select on this side (defaults to ``CatalogueSource.default_columns``)."""

    endpoint: str | None = None
    """TAP endpoint short-name for ad-hoc table-id resolution (vizier, noirlab, …)."""

    frame: str | None = None
    """Declared coordinate frame; mixed-frame matching requires prior conversion."""

    def as_dict(self) -> dict[str, Any]:
        """Return a dict suitable for ``resolve_source(…, overrides=…)``."""
        d: dict[str, Any] = {}
        for field_name in (
            "ra_column",
            "dec_column",
            "id_column",
            "ra_err_column",
            "dec_err_column",
            "corr_column",
            "astrometric_covariance_columns",
            "pos_err_units",
            "default_pos_error_arcsec",
            "epoch",
            "epoch_column",
            "pm_ra_column",
            "pm_dec_column",
            "parallax_column",
            "radial_velocity_column",
            "endpoint",
            "frame",
        ):
            val = getattr(self, field_name)
            if val is not None:
                d[field_name] = val
        return d


@dataclass
class MatchRequest:
    """All typed parameters for a single catalogue crossmatch.

    Populate this directly or build one from keyword parameters via
    :meth:`from_params`. The ``CrossMatch`` orchestrator resolves the sources
    and dispatches to the appropriate backend.
    """

    cat1: FrameInput
    """First catalogue: file path, configured name, or in-memory frame."""

    cat2: FrameInput
    """Second catalogue: file path, configured name, or in-memory frame."""

    spec: MatchSpec = field(default_factory=MatchSpec)
    """Match criteria (radius, matcher, join type, find mode, Bayes priors)."""

    # ------------------------------------------------------------------ output
    output_file: str | Path | None = None
    """If set, stream results to this file (``.parquet`` / ``.csv`` / ``.fits``)."""

    lazy: bool = False
    """Return a ``pl.LazyFrame`` instead of collecting eagerly."""

    memory_budget_bytes: int | None = None
    """Enable bounded-memory local matching when projected inputs exceed this many bytes."""

    scratch_dir: str | Path | None = None
    """Parent directory for temporary partitioned Parquet datasets."""

    partition_order: str | int = "auto"
    """Sky-zone order used by bounded-memory matching, or ``"auto"``."""

    # ---------------------------------------------------------------- matching
    engine: str = "auto"
    """Sky-match engine: ``"auto"`` → STILTS if available else fast,
    ``"astropy"``, ``"fast"`` (cKDTree), ``"torchsky"`` (tensor-native
    nearest-neighbour), ``"zone"`` (HEALPix sharded), or ``"stilts"``."""

    id_join: bool = False
    """Switch from sky-matching to a pure polars relational ID join."""

    id_column_1: str | None = None
    """ID column on catalogue 1 (for ``id_join=True``)."""

    id_column_2: str | None = None
    """ID column on catalogue 2 (for ``id_join=True``)."""

    # ---------------------------------------------------------- column overrides
    side1: SideOverrides = field(default_factory=SideOverrides)
    """Per-catalogue overrides for the first catalogue."""

    side2: SideOverrides = field(default_factory=SideOverrides)
    """Per-catalogue overrides for the second catalogue."""

    # ---------------------------------------------------------- remote region
    ra: float | None = None
    """Region centre RA (deg) for remote downloads."""

    dec: float | None = None
    """Region centre Dec (deg) for remote downloads."""

    radius_deg: float | None = None
    """Region radius (deg) for remote downloads."""

    # --------------------------------------------------------------- Bayesian
    probabilistic: bool = False
    """Signal that Tier-3 Bayesian qualification should be attempted."""

    # ----------------------------------------------------------------- factory
    @classmethod
    def from_params(
        cls,
        cat1: FrameInput,
        cat2: FrameInput,
        output_file: str | Path | None = None,
        *,
        lazy: bool = False,
        **params: Any,
    ) -> MatchRequest:
        """Build a ``MatchRequest`` from keyword arguments."""
        prior: list[str] = list(params.get("prior_columns") or [])
        extra_distance: dict[str, float] = dict(params.get("extra_distance_cols") or {})
        # Normalise extra_distance_cols values to float (may arrive as str/int).
        if extra_distance:
            extra_distance = {str(k): float(v) for k, v in extra_distance.items()}
        spec = MatchSpec(
            radius_arcsec=float(params.get("radius_arcsec", 1.0)),
            matcher=params.get("matcher") or "sky",
            max_error=float(params.get("max_error", 3.0)),
            join_type=params.get("join_type", "1and2"),
            find=params.get("find", "best"),
            prior_columns=prior,
            target_epoch=params.get("target_epoch"),
            filter_expr=params.get("filter_expr"),
            extra_distance_cols=extra_distance,
            batch_size=params.get("batch_size"),
            lr_magnitude_column=params.get("lr_magnitude_column"),
            lr_q=float(params.get("lr_q", 0.8)),
            ml_color_columns=list(params.get("ml_color_columns") or []),
            ml_model_path=params.get("ml_model_path"),
            xgb_model_path=params.get("xgb_model_path"),
            macauff_flux_columns=list(params.get("macauff_flux_columns") or []),
            pm_prior=bool(params.get("pm_prior")),
            pm_prior_magnitude_column=params.get("pm_prior_magnitude_column"),
            fallback_policy=params.get("fallback_policy", "warn"),
        )
        return cls(
            cat1=cat1,
            cat2=cat2,
            spec=spec,
            output_file=output_file,
            lazy=lazy,
            memory_budget_bytes=params.get("memory_budget_bytes"),
            scratch_dir=params.get("scratch_dir"),
            partition_order=params.get("partition_order", "auto"),
            engine=params.get("engine", "auto"),
            id_join=bool(params.get("id_join")),
            id_column_1=params.get("id_column_1"),
            id_column_2=params.get("id_column_2"),
            side1=SideOverrides(
                ra_column=params.get("ra_column_1"),
                dec_column=params.get("dec_column_1"),
                id_column=params.get("id_column_1"),
                ra_err_column=params.get("ra_err_column_1"),
                dec_err_column=params.get("dec_err_column_1"),
                corr_column=params.get("corr_column_1"),
                astrometric_covariance_columns=params.get("astrometric_covariance_columns_1"),
                pos_err_units=params.get("pos_err_units_1"),
                default_pos_error_arcsec=params.get("default_pos_error_arcsec_1"),
                epoch=params.get("epoch_1"),
                epoch_column=params.get("epoch_column_1"),
                pm_ra_column=params.get("pm_ra_column_1"),
                pm_dec_column=params.get("pm_dec_column_1"),
                parallax_column=params.get("parallax_column_1"),
                radial_velocity_column=params.get("radial_velocity_column_1"),
                columns=_parse_columns(params.get("columns_1")),
                endpoint=params.get("endpoint"),
                frame=params.get("frame_1"),
            ),
            side2=SideOverrides(
                ra_column=params.get("ra_column_2"),
                dec_column=params.get("dec_column_2"),
                id_column=params.get("id_column_2"),
                ra_err_column=params.get("ra_err_column_2"),
                dec_err_column=params.get("dec_err_column_2"),
                corr_column=params.get("corr_column_2"),
                astrometric_covariance_columns=params.get("astrometric_covariance_columns_2"),
                pos_err_units=params.get("pos_err_units_2"),
                default_pos_error_arcsec=params.get("default_pos_error_arcsec_2"),
                epoch=params.get("epoch_2"),
                epoch_column=params.get("epoch_column_2"),
                pm_ra_column=params.get("pm_ra_column_2"),
                pm_dec_column=params.get("pm_dec_column_2"),
                parallax_column=params.get("parallax_column_2"),
                radial_velocity_column=params.get("radial_velocity_column_2"),
                columns=_parse_columns(params.get("columns_2")),
                endpoint=params.get("endpoint"),
                frame=params.get("frame_2"),
            ),
            ra=params.get("ra"),
            dec=params.get("dec"),
            radius_deg=params.get("radius_deg"),
            probabilistic=bool(params.get("probabilistic")),
        )


def _parse_columns(value: Any) -> list[str] | None:
    """Normalise a ``columns_*`` parameter into ``List[str] | None``."""
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        items = [c.strip() for c in value.split(",") if c.strip()]
        return items or None
    return None
