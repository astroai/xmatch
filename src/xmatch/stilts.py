"""Thin wrapper around the STILTS ``tmatch2`` command for sky crossmatching.

Frames are handed off as temporary FITS files; the result is read back as a
polars ``DataFrame``. This module no longer contains any Python re-implementation
of sky matching — see :mod:`xmatch.matchers` for the astropy fallback engine.
"""

import logging
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from .exceptions import StiltsError
from .sources import position_error_to_arcsec_factor

logger = logging.getLogger(__name__)

# STILTS join keyword per xmatch join_type.
_STILTS_JOIN = {
    "1and2": "1and2",
    "1or2": "1or2",
    "all": "1or2",
    "all1": "all1",
    "all2": "all2",
    "1not2": "1not2",
    "2not1": "2not1",
}


# Well-known locations of the ``stilts`` wrapper shipped with TOPCAT.
_TOPCAT_STILTS_CANDIDATES = (
    "/Applications/TOPCAT.app/Contents/Resources/app/stilts",
    "~/Applications/TOPCAT.app/Contents/Resources/app/stilts",
)


def _resolve_base_command(stilts_cmd_base: str | None) -> list[str] | None:
    """Return the argv prefix to invoke STILTS, or None if unavailable."""
    if stilts_cmd_base:
        try:
            return shlex.split(stilts_cmd_base)
        except ValueError:
            return None
    if shutil.which("stilts"):
        return ["stilts"]
    jar = os.getenv("STILTS_JAR")
    if jar and Path(jar).is_file() and shutil.which("java"):
        return ["java", "-jar", jar]
    for candidate in _TOPCAT_STILTS_CANDIDATES:
        path = Path(candidate).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return [str(path)]
    return None


def stilts_available(stilts_cmd_base: str | None = None) -> bool:
    return _resolve_base_command(stilts_cmd_base) is not None


def _build_command(
    base: list[str],
    task: str,
    params: dict[str, Any],
    java_opts: str | None,
    tmpdir: str | None,
) -> list[str]:
    cmd = list(base)
    if base[0] == "java":
        opts = shlex.split(java_opts) if java_opts else []
        if tmpdir:
            opts.append(f"-Djava.io.tmpdir={tmpdir}")
        cmd[1:1] = opts
    cmd.append(task)
    for key, value in params.items():
        if value is not None:
            cmd.append(f"{key}={value}")
    return cmd


def _run_stilts(
    base: list[str],
    task: str,
    params: dict[str, Any],
    java_opts: str | None = None,
    tmpdir: str | None = None,
) -> None:
    cmd = _build_command(base, task, params, java_opts, tmpdir)
    logger.debug("Running STILTS: %s", " ".join(shlex.quote(a) for a in cmd))
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, errors="replace")
    except FileNotFoundError as exc:
        raise StiltsError(f"STILTS executable not found: {exc}") from exc
    except subprocess.CalledProcessError as exc:
        raise StiltsError(
            f"STILTS task '{task}' failed (exit {exc.returncode}):\n{(exc.stderr or '').strip()}"
        ) from exc


def _write_fits(df: pl.DataFrame, path: Path) -> None:
    from .io_utils import polars_to_astropy

    polars_to_astropy(df).write(path, overwrite=True)


def _great_circle_arcsec(ra1, dec1, ra2, dec2) -> np.ndarray:
    """Vectorised great-circle separation in arcsec (NaN where any input is NaN)."""
    lon1, lat1, lon2, lat2 = (
        np.radians(np.asarray(a, dtype=float)) for a in (ra1, dec1, ra2, dec2)
    )
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    haversine = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0)))) * 3600.0


def _add_true_separation(result: pl.DataFrame, l_ra, l_dec, r_ra, r_dec) -> pl.DataFrame:
    """Replace STILTS' internal score columns with a true ``sep_arcsec`` column.

    STILTS reports angular separation for ``sky`` but a normalised score for
    ``skyerr``; recompute the real arcsec separation so both engines agree.
    """
    for meta in ("Separation", "GroupID", "GroupSize"):
        if meta in result.columns:
            result = result.drop(meta)
    if not all(c in result.columns for c in (l_ra, l_dec, r_ra, r_dec)):
        return result.with_columns(pl.lit(None, dtype=pl.Float64).alias("sep_arcsec"))
    sep = _great_circle_arcsec(
        result[l_ra].to_numpy(),
        result[l_dec].to_numpy(),
        result[r_ra].to_numpy(),
        result[r_dec].to_numpy(),
    )
    return result.with_columns(pl.Series("sep_arcsec", sep))


def _error_value_expr(src, max_error: float) -> str:
    """STILTS expression for the n-sigma positional error radius (arcsec).

    Mirrors :func:`xmatch.matchers._pos_sigma_arcsec`: the per-row error is
    ``hypot(ra_err, dec_err)`` (unit-converted, floored), scaled by ``max_error``
    so that STILTS' ``sep <= err1 + err2`` rule becomes
    ``sep <= max_error * (e1 + e2)``.
    """
    factor = position_error_to_arcsec_factor(src.pos_err_units)
    if src.ra_err_column and src.dec_err_column:
        core = f"hypot({src.ra_err_column},{src.dec_err_column})*{factor}"
    else:
        core = "0"
    if src.default_pos_error_arcsec is not None:
        floor = float(src.default_pos_error_arcsec) * (2**0.5)
        core = f"max({core},{floor})"
    return f"{max_error}*({core})"


def _values_expr(src, spec) -> str:
    ra, dec = src.ra_column, src.dec_column
    if spec.matcher == "sky":
        return f"{ra} {dec}"
    # skyerr uses a circular n-sigma error radius. skyellipse is rejected at
    # the STILTS entrypoint because this backend cannot honor its covariance.
    return f"{ra} {dec} {_error_value_expr(src, spec.max_error)}"


def stilts_sky_match(
    left_src,
    right_src,
    left: pl.DataFrame,
    right: pl.DataFrame,
    spec,
    *,
    stilts_cmd_base=None,
    java_opts=None,
    tmpdir=None,
    right_suffix: str = "_2",
) -> pl.DataFrame:
    unsupported = []
    if spec.matcher not in {"sky", "skyerr"}:
        unsupported.append(f"matcher={spec.matcher!r}")
    if spec.extra_distance_cols:
        unsupported.append("extra_distance_cols")
    if spec.prior_columns:
        unsupported.append("prior_columns")
    if spec.probabilistic:
        unsupported.append("probabilistic scoring")
    if spec.filter_expr:
        unsupported.append("filter_expr")
    if unsupported:
        raise StiltsError("STILTS cannot honor " + ", ".join(unsupported))

    base = _resolve_base_command(stilts_cmd_base)
    if base is None:
        raise StiltsError("No STILTS command available.")

    if spec.matcher == "sky":
        stilts_matcher = "sky"
        params_value = str(spec.radius_arcsec)
    else:
        # skyerr: 'params' is only a binning scale; size it from the data so no
        # genuine match is ever binned out.
        from .matchers import _pos_sigma_arcsec

        lsig = _pos_sigma_arcsec(left, left_src)
        rsig = _pos_sigma_arcsec(right, right_src)
        if lsig is None or rsig is None:
            raise StiltsError(
                "skyerr matcher requires positional error columns on both "
                "catalogues, or a default_pos_error_arcsec in the config."
            )
        scale = spec.max_error * (float(np.nanmax(lsig)) + float(np.nanmax(rsig)))
        stilts_matcher = "skyerr"
        params_value = str(max(scale, 1e-6))

    with tempfile.TemporaryDirectory(prefix="xmatch_stilts_") as tmp:
        in1 = Path(tmp) / "in1.fits"
        in2 = Path(tmp) / "in2.fits"
        from astropy.table import Table

        from .io_utils import astropy_table_to_polars

        out = Path(tmp) / "out.fits"
        _write_fits(left, in1)
        _write_fits(right, in2)

        params = {
            "in1": str(in1),
            "in2": str(in2),
            "ifmt1": "fits",
            "ifmt2": "fits",
            "matcher": stilts_matcher,
            "values1": _values_expr(left_src, spec),
            "values2": _values_expr(right_src, spec),
            "params": params_value,
            "join": _STILTS_JOIN.get(spec.join_type, "1and2"),
            # xmatch selects per primary; STILTS "best" instead enforces 1:1.
            "find": "best1" if spec.find == "best" else spec.find,
            # Match the astropy engine's schema: left columns keep their names,
            # right-hand collisions get a custom suffix.
            "fixcols": "dups",
            "suffix1": "",
            "suffix2": right_suffix,
            "out": str(out),
            "ofmt": "fits",
        }
        _run_stilts(base, "tmatch2", params, java_opts, tmpdir)
        result = astropy_table_to_polars(Table.read(out))

        collide = set(left.columns) & set(right.columns)
        r_ra = right_src.ra_column + (right_suffix if right_src.ra_column in collide else "")
        r_dec = right_src.dec_column + (right_suffix if right_src.dec_column in collide else "")
        return _add_true_separation(result, left_src.ra_column, left_src.dec_column, r_ra, r_dec)
