"""Uniform catalogue-source abstraction.

A :class:`CatalogueSource` describes *where* a catalogue comes from (a local
file/frame, a TAP service, the CDS XMatch service, or a HATS catalogue) together
with the metadata needed to match it (coordinate / id / error columns, epoch).

Local sources can produce a :class:`polars.LazyFrame` directly; remote sources
are materialised by the relevant backend (TAP/CDS/HATS) during execution.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import polars as pl

from . import io_utils

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
    ra_column: Optional[str] = None
    dec_column: Optional[str] = None
    id_column: Optional[str] = None

    # Positional error metadata (for skyerr / skyellipse).
    ra_err_column: Optional[str] = None
    dec_err_column: Optional[str] = None
    corr_column: Optional[str] = None
    astrometric_covariance_columns: Optional[Dict[str, str]] = None
    pos_err_units: str = "arcsec"
    default_pos_error_arcsec: Optional[float] = None

    # Epoch / proper-motion metadata.
    epoch: Optional[float] = None
    epoch_column: Optional[str] = None
    pm_ra_column: Optional[str] = None
    pm_dec_column: Optional[str] = None
    parallax_column: Optional[str] = None
    radial_velocity_column: Optional[str] = None

    # Remote access metadata.
    access_method: Optional[str] = None  # None | "tap" | "cds_xmatch" | "hats"
    archive: Optional[str] = None
    access_identifier: Optional[str] = None
    tap_url: Optional[str] = None
    default_columns: Optional[List[str]] = None

    # Mirror location inside the xmatch cache root (set by ensure_mirrored).
    hats_cache_rel: Optional[str] = None

    # Local payload (exactly one is set for local sources).
    path: Optional[Path] = None
    _frame: Optional[pl.LazyFrame] = field(default=None, repr=False)

    def lazy(self) -> pl.LazyFrame:
        """Return the catalogue as a LazyFrame (local sources only)."""
        if self._frame is not None:
            return self._frame
        if self.path is not None:
            return io_utils.scan_frame(self.path)
        raise ValueError(f"Source '{self.name}' is not local; cannot produce a frame directly.")

    def with_frame(self, frame: pl.LazyFrame) -> "CatalogueSource":
        """Return a copy of this source backed by an in-memory frame (post-download)."""
        import copy

        clone = copy.copy(self)
        clone._frame = frame
        clone.path = None
        clone.is_local = True
        return clone

    def columns(self) -> List[str]:
        if self.is_local:
            return io_utils.frame_columns(self.lazy())
        return list(self.default_columns or [])
