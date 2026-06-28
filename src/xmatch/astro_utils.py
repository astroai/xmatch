"""Astronomy helpers: coordinate-column detection, validation and sky extent.

These functions operate on column-name lists and numpy arrays so they are
agnostic to whether the data lives in a polars or pandas frame.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_RA_PATTERNS = [
    "ra",
    "raj2000",
    "ra_icrs",
    "ra_deg",
    "alpha",
    "rightascension",
    "ra_j2000",
    "_raj2000",
]
_DEC_PATTERNS = [
    "dec",
    "dej2000",
    "de_icrs",
    "dec_deg",
    "delta",
    "declination",
    "dec_j2000",
    "_dej2000",
    "de",
]


def find_coord_columns(columns: Sequence[str]) -> Tuple[Optional[str], Optional[str]]:
    """Best-effort detection of RA/Dec columns from a list of column names.

    Matching is case-insensitive and tolerant of separators (``RA (deg)`` →
    ``ra``). Returns ``(None, None)`` if no confident match is found.
    """

    def normalise(name: str) -> str:
        return "".join(ch for ch in name.lower() if ch.isalnum())

    norm_map: Dict[str, str] = {}
    for original in columns:
        norm_map.setdefault(normalise(original), original)

    ra_col = next((norm_map[normalise(p)] for p in _RA_PATTERNS if normalise(p) in norm_map), None)
    dec_col = next(
        (norm_map[normalise(p)] for p in _DEC_PATTERNS if normalise(p) in norm_map), None
    )

    if ra_col and dec_col:
        logger.info("Auto-detected coordinate columns: RA='%s', Dec='%s'", ra_col, dec_col)
    else:
        logger.debug("Could not auto-detect RA/Dec columns from %s", list(columns))
    return ra_col, dec_col


def validate_coordinates(ra: np.ndarray, dec: np.ndarray) -> None:
    """Raise ``ValueError`` if RA/Dec arrays contain NaNs or out-of-range values."""
    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)
    if np.isnan(ra).any() or np.isnan(dec).any():
        raise ValueError("RA/Dec contain NaN values.")
    eps = 1e-9
    if ((ra < -eps) | (ra > 360 + eps)).any():
        raise ValueError("RA values outside [0, 360].")
    if ((dec < -90 - eps) | (dec > 90 + eps)).any():
        raise ValueError("Dec values outside [-90, 90].")


def sky_extent(ra: np.ndarray, dec: np.ndarray) -> Optional[Dict[str, float]]:
    """Return the center and radius (degrees) of a cone covering all points.

    Uses a unit-vector mean so it is correct across the RA=0/360 boundary and
    near the poles.
    """
    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)
    if ra.size == 0:
        return None
    validate_coordinates(ra, dec)

    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)
    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)
    mean = np.array([x.mean(), y.mean(), z.mean()])
    norm = np.linalg.norm(mean)
    if norm < 1e-12:  # antipodal spread; fall back to whole sky
        return {
            "ra_center_deg": float(ra.mean()),
            "dec_center_deg": float(dec.mean()),
            "radius_deg": 180.0,
        }
    mean /= norm
    center_dec = np.degrees(np.arcsin(np.clip(mean[2], -1.0, 1.0)))
    center_ra = np.degrees(np.arctan2(mean[1], mean[0])) % 360.0

    # Max angular separation from the center (great-circle).
    cos_sep = np.sin(np.radians(center_dec)) * np.sin(dec_rad) + np.cos(
        np.radians(center_dec)
    ) * np.cos(dec_rad) * np.cos(ra_rad - np.radians(center_ra))
    radius_deg = float(np.degrees(np.arccos(np.clip(cos_sep, -1.0, 1.0))).max())
    return {
        "ra_center_deg": float(center_ra),
        "dec_center_deg": float(center_dec),
        "radius_deg": max(radius_deg, 1e-6),
    }


def propagate_proper_motion(
    ra: np.ndarray,
    dec: np.ndarray,
    pm_ra_cosdec: np.ndarray,
    pm_dec: np.ndarray,
    source_epoch: np.ndarray,
    target_epoch: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Propagate ICRS coordinates to ``target_epoch`` (Julian year).

    Proper motions are in mas/yr (``pm_ra_cosdec`` already includes cos(Dec)).
    Returns ``(ra_deg, dec_deg)``. NaN proper motions are treated as zero.
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.time import Time

    pm_ra_cosdec = np.nan_to_num(np.asarray(pm_ra_cosdec, dtype=float))
    pm_dec = np.nan_to_num(np.asarray(pm_dec, dtype=float))
    source_epoch = np.where(np.isnan(source_epoch), target_epoch, source_epoch)

    coords = SkyCoord(
        ra=np.asarray(ra, dtype=float) * u.deg,
        dec=np.asarray(dec, dtype=float) * u.deg,
        pm_ra_cosdec=pm_ra_cosdec * u.mas / u.yr,
        pm_dec=pm_dec * u.mas / u.yr,
        obstime=Time(source_epoch, format="jyear", scale="tcb"),
        frame="icrs",
    )
    moved = coords.apply_space_motion(new_obstime=Time(target_epoch, format="jyear", scale="tcb"))
    return moved.ra.deg, moved.dec.deg


def coord_arrays(frame, ra_col: str, dec_col: str) -> Tuple[np.ndarray, np.ndarray]:
    """Extract RA/Dec numpy arrays from a polars frame (lazy or eager)."""
    import polars as pl

    sel = frame.select([ra_col, dec_col])
    df = sel.collect() if isinstance(sel, pl.LazyFrame) else sel
    return df[ra_col].to_numpy(), df[dec_col].to_numpy()


def sky_extent_from_frame(
    frame,
    ra_col: str,
    dec_col: str,
) -> Optional[Dict[str, float]]:
    """Compute the bounding cone of a polars frame without materialising RA/Dec.

    Expresses the great-circle mean and max-separation entirely as polars
    aggregations so a TB-scale local LazyFrame stays lazy; only the resulting
    scalars (mean-x/y/z, max-cos-sep) are collected. Returns ``None`` for an
    empty frame and the same dict shape as :func:`sky_extent`.
    """
    import polars as pl

    lf = frame.lazy() if isinstance(frame, pl.DataFrame) else frame
    schema = lf.collect_schema()
    if ra_col not in schema or dec_col not in schema:
        raise ValueError(f"Frame missing {ra_col} or {dec_col} (has: {schema.names()}).")

    if lf.select(pl.len()).collect().item() == 0:
        return None

    # Phase 1: mean-x/y/z unit-vector (handles RA=0/360 wrap and the poles).
    means = lf.select(
        (pl.col(dec_col).radians().cos() * pl.col(ra_col).radians().cos()).mean().alias("x"),
        (pl.col(dec_col).radians().cos() * pl.col(ra_col).radians().sin()).mean().alias("y"),
        pl.col(dec_col).radians().sin().mean().alias("z"),
    ).collect()

    mx, my, mz = float(means["x"][0]), float(means["y"][0]), float(means["z"][0])
    norm = math.sqrt(mx * mx + my * my + mz * mz)
    if norm < 1e-12:  # antipodal spread; fall back to whole-sky equator mean
        ra_mean = lf.select(pl.col(ra_col).drop_nulls().mean()).collect().item()
        dec_mean = lf.select(pl.col(dec_col).drop_nulls().mean()).collect().item()
        if ra_mean is None or dec_mean is None:
            return None
        return {
            "ra_center_deg": float(ra_mean),
            "dec_center_deg": float(dec_mean),
            "radius_deg": 180.0,
        }
    mx, my, mz = mx / norm, my / norm, mz / norm

    center_dec = math.degrees(math.asin(max(-1.0, min(1.0, mz))))
    center_ra = math.degrees(math.atan2(my, mx)) % 360.0

    # Phase 2: max great-circle separation from the centre, again all in polars.
    sin_d = math.sin(math.radians(center_dec))
    cos_d = math.cos(math.radians(center_dec))
    ra0 = math.radians(center_ra)
    cos_sep_expr = (
        sin_d * pl.col(dec_col).radians().sin()
        + cos_d * pl.col(dec_col).radians().cos() * (pl.col(ra_col).radians() - ra0).cos()
    )
    cos_sep_max = lf.select(cos_sep_expr.max()).collect().item()
    radius_deg = math.degrees(math.acos(max(-1.0, min(1.0, float(cos_sep_max)))))
    return {
        "ra_center_deg": float(center_ra),
        "dec_center_deg": float(center_dec),
        "radius_deg": max(radius_deg, 1e-6),
    }


def list_or_none(value) -> Optional[List[str]]:
    if value is None:
        return None
    return list(value)
