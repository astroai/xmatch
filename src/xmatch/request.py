"""Typed request objects for xmatch crossmatch operations.

The :class:`MatchRequest` dataclass consolidates all match parameters that were
previously passed as ad-hoc ``**kwargs``, giving mypy (and IDEs) a fully-typed
view of what every match call requires. Side-level column overrides are bundled
into :class:`SideOverrides` so the orchestrator never needs to re-derive the
same dict twice.

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
from typing import Any, Dict, List, Optional, Union

import polars as pl

from .matchers import MatchSpec

FrameInput = Union[str, Path, pl.DataFrame, pl.LazyFrame]


@dataclass
class SideOverrides:
    """Per-catalogue column overrides supplied by the user.

    Every field is optional — when ``None`` the resolver falls back to auto-
    detection or the YAML config defaults.
    """

    ra_column: Optional[str] = None
    """Right-ascension column name on this side."""

    dec_column: Optional[str] = None
    """Declination column name on this side."""

    id_column: Optional[str] = None
    """Identifier column name on this side."""

    columns: Optional[List[str]] = None
    """Columns to select on this side (defaults to ``CatalogueSource.default_columns``)."""

    def as_dict(self) -> Dict[str, Any]:
        """Return a dict suitable for ``resolve_source(…, overrides=…)``."""
        d: Dict[str, Any] = {}
        for field_name in ("ra_column", "dec_column", "id_column"):
            val = getattr(self, field_name)
            if val is not None:
                d[field_name] = val
        return d


@dataclass
class MatchRequest:
    """All typed parameters for a single catalogue crossmatch.

    Populate this directly (preferred for new code) or build one from a legacy
    ``**params`` dict via :meth:`from_legacy`. The ``CrossMatch`` orchestrator
    resolves the sources and dispatches to the appropriate backend.
    """

    cat1: FrameInput
    """First catalogue: file path, configured name, or in-memory frame."""

    cat2: FrameInput
    """Second catalogue: file path, configured name, or in-memory frame."""

    spec: MatchSpec = field(default_factory=MatchSpec)
    """Match criteria (radius, matcher, join type, find mode, Bayes priors)."""

    # ------------------------------------------------------------------ output
    output_file: Optional[Union[str, Path]] = None
    """If set, stream results to this file (``.parquet`` / ``.csv`` / ``.fits``)."""

    lazy: bool = False
    """Return a ``pl.LazyFrame`` instead of collecting eagerly."""

    # ---------------------------------------------------------------- matching
    engine: str = "auto"
    """Sky-match engine: ``"auto"`` → STILTS if available else fast,
    ``"astropy"``, ``"fast"`` (cKDTree), ``"torchsky"`` (tensor-native
    nearest-neighbour), ``"zone"`` (HEALPix sharded), or ``"stilts"``."""

    id_join: bool = False
    """Switch from sky-matching to a pure polars relational ID join."""

    id_column_1: Optional[str] = None
    """ID column on catalogue 1 (for ``id_join=True``)."""

    id_column_2: Optional[str] = None
    """ID column on catalogue 2 (for ``id_join=True``)."""

    # ---------------------------------------------------------- column overrides
    side1: SideOverrides = field(default_factory=SideOverrides)
    """Per-catalogue overrides for the first catalogue."""

    side2: SideOverrides = field(default_factory=SideOverrides)
    """Per-catalogue overrides for the second catalogue."""

    # ---------------------------------------------------------- remote region
    ra: Optional[float] = None
    """Region centre RA (deg) for remote downloads."""

    dec: Optional[float] = None
    """Region centre Dec (deg) for remote downloads."""

    radius_deg: Optional[float] = None
    """Region radius (deg) for remote downloads."""

    # --------------------------------------------------------------- Bayesian
    probabilistic: bool = False
    """Signal that Tier-3 Bayesian qualification should be attempted."""

    # ----------------------------------------------------------------- factory
    @classmethod
    def from_legacy(
        cls,
        cat1: FrameInput,
        cat2: FrameInput,
        output_file: Optional[Union[str, Path]] = None,
        *,
        lazy: bool = False,
        **params: Any,
    ) -> "MatchRequest":
        """Build a ``MatchRequest`` from the legacy ``**params`` dict.

        Used internally by :meth:`CrossMatch.crossmatch` so the old spread-args
        API keeps working while the internals migrate to typed dataclasses.
        """
        prior: List[str] = list(params.get("prior_columns") or [])
        extra_distance: Dict[str, float] = dict(params.get("extra_distance_cols") or {})
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
            engine=params.get("engine", "auto"),
            id_join=bool(params.get("id_join")),
            id_column_1=params.get("id_column_1"),
            id_column_2=params.get("id_column_2"),
            side1=SideOverrides(
                ra_column=params.get("ra_column_1"),
                dec_column=params.get("dec_column_1"),
                id_column=params.get("id_column_1"),
                columns=_parse_columns(params.get("columns_1")),
            ),
            side2=SideOverrides(
                ra_column=params.get("ra_column_2"),
                dec_column=params.get("dec_column_2"),
                id_column=params.get("id_column_2"),
                columns=_parse_columns(params.get("columns_2")),
            ),
            ra=params.get("ra"),
            dec=params.get("dec"),
            radius_deg=params.get("radius_deg"),
            probabilistic=bool(params.get("probabilistic")),
        )


def _parse_columns(value: Any) -> Optional[List[str]]:
    """Normalise a ``columns_*`` parameter into ``List[str] | None``."""
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        items = [c.strip() for c in value.split(",") if c.strip()]
        return items or None
    return None
