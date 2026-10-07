"""Uniform catalogue-source abstraction.

A :class:`CatalogueSource` describes *where* a catalogue comes from (a local
file/frame, a TAP service, the CDS XMatch service, or a HATS catalogue) together
with the metadata needed to match it (coordinate / id / error columns, epoch).

Local sources can produce a :class:`polars.LazyFrame` directly; remote sources
are materialised by the relevant backend (TAP/CDS/HATS) during execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from . import io_utils
from .exceptions import CrossMatchError

_POSITION_ERROR_UNIT_TO_ARCSEC = {
    "arcsec": 1.0,
    "mas": 1e-3,
    "deg": 3600.0,
    "arcmin": 60.0,
}


def position_error_to_arcsec_factor(units: object) -> float:
    """Return the conversion factor for supported positional-error units."""
    if units is None:
        unit = "arcsec"
    elif isinstance(units, str):
        unit = units.strip().lower()
    else:
        unit = ""
    try:
        return _POSITION_ERROR_UNIT_TO_ARCSEC[unit]
    except KeyError as exc:
        raise CrossMatchError(
            f"Unsupported positional-error unit {units!r}; use arcsec, mas, arcmin, or deg."
        ) from exc


ASTROMETRIC_COVARIANCE_KEYS = (
    "ra_error",
    "dec_error",
    "parallax_error",
    "pmra_error",
    "pmdec_error",
    "ra_dec_corr",
    "ra_parallax_corr",
    "ra_pmra_corr",
    "ra_pmdec_corr",
    "dec_parallax_corr",
    "dec_pmra_corr",
    "dec_pmdec_corr",
    "parallax_pmra_corr",
    "parallax_pmdec_corr",
    "pmra_pmdec_corr",
)


@dataclass
class CatalogueSource:
    name: str
    is_local: bool
    ra_column: str | None = None
    dec_column: str | None = None
    id_column: str | None = None

    # Positional error metadata (for skyerr / skyellipse).
    ra_err_column: str | None = None
    dec_err_column: str | None = None
    corr_column: str | None = None
    astrometric_covariance_columns: dict[str, str] | None = None
    pos_err_units: str = "arcsec"
    default_pos_error_arcsec: float | None = None

    # Epoch / proper-motion metadata.
    epoch: float | None = None
    epoch_column: str | None = None
    pm_ra_column: str | None = None
    pm_dec_column: str | None = None
    parallax_column: str | None = None
    radial_velocity_column: str | None = None
    frame: str = "icrs"

    # Remote access metadata.
    access_method: str | None = None  # None | "tap" | "cds_xmatch" | "hats"
    archive: str | None = None
    access_identifier: str | None = None
    tap_url: str | None = None
    default_columns: list[str] | None = None

    # Mirror location inside the xmatch cache root (set by ensure_mirrored).
    hats_cache_rel: str | None = None

    # Cache root that actually holds the mirrored copy at ``hats_cache_rel``
    # (a replica root when the primary copy is gone; set by _mirrored_source).
    hats_cache_root: str | None = None

    # Alternate endpoints for the same data (mirror archives / mirrored
    # copies other data centres keep).  Entries are resolved CatalogueSource
    # objects (same-schema copies in xmatch.yaml) or raw http(s)/vos: HATS
    # URLs for remote-HATS sources.  Used by the sync failover walk.
    fallbacks: list[str | CatalogueSource] = field(default_factory=list)

    # Local payload (exactly one is set for local sources).
    path: Path | None = None
    _frame: pl.LazyFrame | None = field(default=None, repr=False)

    # Explicit upstream release/measurement metadata. Append fields to preserve
    # existing positional constructors; never infer survey conventions.
    release_namespace: str | None = None
    release_metadata: dict[str, Any] = field(default_factory=dict)
    photometry: list[dict[str, Any]] = field(default_factory=list)
    property_evidence: list[dict[str, Any]] = field(default_factory=list)

    def lazy(self) -> pl.LazyFrame:
        """Return the catalogue as a LazyFrame (local sources only)."""
        if self._frame is not None:
            return self._frame
        if self.path is not None:
            return io_utils.scan_frame(self.path)
        raise ValueError(f"Source '{self.name}' is not local; cannot produce a frame directly.")

    def with_frame(self, frame: pl.LazyFrame) -> CatalogueSource:
        """Return a copy of this source backed by an in-memory frame (post-download)."""
        import copy

        clone = copy.copy(self)
        clone._frame = frame
        clone.path = None
        clone.is_local = True
        return clone

    def columns(self) -> list[str]:
        if self.is_local:
            return io_utils.frame_columns(self.lazy())
        return list(self.default_columns or [])
