"""HATS-native crossmatch via xmatch matchers.

One task per left HATS pixel; right side loads the same pixel plus HEALPix
neighbours for boundary-safe matching. Engines ``fast``/``zone``/``ray``/…
and matchers ``sky``/``skyerr``/``skyellipse`` are honored. Outer joins
(``1or2``, …) materialize both sides then call :func:`sky_match` once.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import polars as pl

from .exceptions import CrossMatchError
from .matchers import MatchSpec, sky_match
from .sources import CatalogueSource

logger = logging.getLogger(__name__)


def _hats_root(src: CatalogueSource) -> Path:
    path = src.access_identifier or (str(src.path) if src.path else None)
    if not path:
        raise CrossMatchError(f"No HATS path for '{src.name}'.")
    root = Path(path).expanduser()
    if not root.exists():
        raise CrossMatchError(f"HATS path does not exist: {root}")
    return root


def _read_properties(root: Path) -> dict[str, str]:
    props: dict[str, str] = {}
    for name in ("properties", "hats.properties"):
        p = root / name
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                props[k.strip()] = v.strip()
        break
    return props


def _ra_dec_columns(src: CatalogueSource, root: Path) -> tuple[str, str]:
    ra = src.ra_column
    dec = src.dec_column
    if ra and dec:
        return ra, dec
    props = _read_properties(root)
    ra = ra or props.get("hats_col_ra") or props.get("ra_col") or "ra"
    dec = dec or props.get("hats_col_dec") or props.get("dec_col") or "dec"
    return ra, dec


def list_hats_pixels(root: Path) -> list[tuple[int, int, Path]]:
    """Return ``(order, pixel, path)`` for each Npix partition under root."""
    dataset = root / "dataset"
    base = dataset if dataset.exists() else root
    found: list[tuple[int, int, Path]] = []
    for npix_dir in sorted(base.glob("Norder=*/Dir=*/Npix=*")):
        if npix_dir.is_dir():
            try:
                order = int(npix_dir.parent.parent.name.split("=")[1])
                pix = int(npix_dir.name.split("=")[1])
            except Exception:
                continue
            found.append((order, pix, npix_dir))
            continue
        if npix_dir.suffix == ".parquet":
            try:
                order = int(npix_dir.parent.parent.name.split("=")[1])
                pix = int(npix_dir.stem.split("=")[1])
            except Exception:
                continue
            found.append((order, pix, npix_dir))
    if not found:
        for parquet in sorted(base.glob("Norder=*/Dir=*/Npix=*.parquet")):
            try:
                order = int(parquet.parent.parent.name.split("=")[1])
                pix = int(parquet.stem.split("=")[1])
            except Exception:
                continue
            found.append((order, pix, parquet))
    return found


def _load_pixel_path(path: Path) -> pl.DataFrame:
    if path.is_file():
        return pl.read_parquet(path)
    files = sorted(path.glob("*.parquet"))
    if not files:
        return pl.DataFrame()
    return pl.read_parquet(files)


def load_hats_all(src: CatalogueSource) -> pl.DataFrame:
    """Materialize an entire HATS catalogue (outer-join / small-catalog path)."""
    root = _hats_root(src)
    frames = [_load_pixel_path(p) for _, _, p in list_hats_pixels(root)]
    frames = [f for f in frames if f is not None and f.height > 0]
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


def _healpix_neighbors(order: int, center_pix: int, *, radius_arcsec: float = 0.0) -> list[int]:
    """Return nested HEALPix pixels covering a cap around ``center_pix``."""
    pix = int(center_pix)
    nside = 2 ** int(order)
    try:
        import torch
        from torchsky.catalogs import cap_covering_pixels
        from torchsky.sphere import healpix as ts_healpix
    except ImportError:
        pass
    else:
        center = torch.tensor([pix], dtype=torch.int64)
        lon, lat = ts_healpix.pix2ang(nside, center, nest=True, lonlat=True)
        pixrad = ts_healpix.max_pixrad(nside, degrees=False)
        radius_rad = math.radians(max(float(radius_arcsec), 0.0) / 3600.0) + (2.0 * pixrad)
        covered = cap_covering_pixels(nside, lon, lat, radius_rad, nest=True)
        return sorted({int(p) for p in covered.tolist()})
    out = {pix}
    try:
        import healpy as hp

        for n in hp.get_all_neighbours(nside, pix, nest=True):
            if int(n) >= 0:
                out.add(int(n))
        return sorted(out)
    except Exception:
        pass
    try:
        import cdshealpix as chp
        import numpy as np

        lon, lat = chp.healpix_to_lonlat(np.asarray([pix], dtype=np.uint64), depth=int(order))
        if hasattr(lon, "to_value"):
            lon = lon.to_value("rad")
            lat = lat.to_value("rad")
        pix_scale = np.sqrt(4.0 * np.pi / (12.0 * nside * nside))
        from .matchers import _cone_search_pixels  # noqa: PLC0415

        neigh = _cone_search_pixels(
            chp,
            float(lon[0]),
            float(lat[0]),
            float(1.5 * pix_scale),
            int(order),
        )
        for n in np.atleast_1d(neigh).astype(int).ravel():
            if n >= 0:
                out.add(int(n))
    except Exception:
        logger.warning(
            "HEALPix neighbour lookup unavailable; margin is same-pixel only "
            "(install healpy or cdshealpix for boundary-safe HATS matches)"
        )
    return sorted(out)


def margin_pixels(
    order: int,
    center_pix: int,
    radius_arcsec: float,
    right_pixels: Sequence[int],
) -> list[int]:
    """Right pixels for a left pixel match: covering of the match cap.

    Intersection with ``right_pixels`` is O(cover size), not O(N_right).
    """
    right_set = {int(p) for p in right_pixels}
    if not right_set:
        return []
    return [
        p
        for p in _healpix_neighbors(order, int(center_pix), radius_arcsec=float(radius_arcsec))
        if p in right_set
    ]


def _frame_source(base: CatalogueSource, df: pl.DataFrame, name: str) -> CatalogueSource:
    ra = base.ra_column or "ra"
    dec = base.dec_column or "dec"
    return CatalogueSource(
        name=name,
        is_local=True,
        ra_column=ra,
        dec_column=dec,
        id_column=base.id_column,
        ra_err_column=base.ra_err_column,
        dec_err_column=base.dec_err_column,
        corr_column=base.corr_column,
        astrometric_covariance_columns=base.astrometric_covariance_columns,
        pos_err_units=base.pos_err_units,
        default_pos_error_arcsec=base.default_pos_error_arcsec,
        epoch=base.epoch,
        epoch_column=base.epoch_column,
        pm_ra_column=base.pm_ra_column,
        pm_dec_column=base.pm_dec_column,
        parallax_column=base.parallax_column,
        radial_velocity_column=base.radial_velocity_column,
        _frame=df.lazy(),
    )


def _match_frames(
    left_src: CatalogueSource,
    right_src: CatalogueSource,
    left_df: pl.DataFrame,
    right_df: pl.DataFrame,
    spec: MatchSpec,
    engine: str,
    right_suffix: str,
) -> pl.DataFrame:
    if left_df.is_empty() and right_df.is_empty():
        return pl.DataFrame()
    lsrc = _frame_source(left_src, left_df, left_src.name)
    rsrc = _frame_source(right_src, right_df, right_src.name)
    if left_src.access_method == "hats":
        ra, dec = _ra_dec_columns(left_src, _hats_root(left_src))
        lsrc.ra_column, lsrc.dec_column = ra, dec
    if right_src.access_method == "hats":
        ra, dec = _ra_dec_columns(right_src, _hats_root(right_src))
        rsrc.ra_column, rsrc.dec_column = ra, dec
    eng = engine if engine not in {"auto", "ray"} else "fast"
    result = sky_match(
        lsrc,
        rsrc,
        lsrc.lazy(),
        rsrc.lazy(),
        spec,
        engine=eng,
        right_suffix=right_suffix,
    )
    if isinstance(result, pl.LazyFrame):
        return result.collect()
    return result


def _pixel_task_payload(
    order: int,
    pix: int,
    left_path: str,
    right_index: dict[int, str],
    right_hats: bool,
    right_payload,
    src1: CatalogueSource,
    src2: CatalogueSource,
    spec: MatchSpec,
    right_suffix: str,
    engine: str = "fast",
) -> pl.DataFrame | None:
    left_df = _load_pixel_path(Path(left_path))
    if left_df.is_empty():
        return None
    if right_hats:
        marg = margin_pixels(order, pix, float(spec.radius_arcsec), list(right_index))
        frames = [_load_pixel_path(Path(right_index[p])) for p in marg if p in right_index]
        frames = [f for f in frames if f.height > 0]
        right_df = pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()
    else:
        right_df = right_payload
    part = _match_frames(src1, src2, left_df, right_df, spec, engine, right_suffix)
    return part if part is not None and part.height > 0 else None


def hats_native_crossmatch(
    src1: CatalogueSource,
    src2: CatalogueSource,
    spec: MatchSpec,
    *,
    engine: str = "fast",
    local_lf1: pl.LazyFrame | None = None,
    local_lf2: pl.LazyFrame | None = None,
    right_suffix: str = "_2",
) -> pl.DataFrame:
    """Crossmatch with real matchers; HATS sides read as parquet pixels."""
    left_hats = src1.access_method == "hats"
    right_hats = src2.access_method == "hats"
    if not left_hats and not right_hats:
        raise CrossMatchError("hats_native_crossmatch requires at least one HATS source")

    eng = (engine or "fast").strip().lower()

    if spec.join_type != "1and2":
        logger.info(
            "HATS native outer/anti join (%s): materializing both catalogues",
            spec.join_type,
        )
        left_df = load_hats_all(src1) if left_hats else (local_lf1 or src1.lazy()).collect()
        right_df = load_hats_all(src2) if right_hats else (local_lf2 or src2.lazy()).collect()
        return _match_frames(src1, src2, left_df, right_df, spec, eng, right_suffix)

    if not left_hats:
        left_df = (local_lf1 or src1.lazy()).collect()
        right_df = load_hats_all(src2) if right_hats else (local_lf2 or src2.lazy()).collect()
        return _match_frames(src1, src2, left_df, right_df, spec, eng, right_suffix)

    left_root = _hats_root(src1)
    left_pixels = list_hats_pixels(left_root)
    if not left_pixels:
        return pl.DataFrame()

    right_index: dict[int, Path] = {}
    right_all = None
    if right_hats:
        right_root = _hats_root(src2)
        for _order, pix, path in list_hats_pixels(right_root):
            right_index[int(pix)] = path
    else:
        right_all = (local_lf2 or src2.lazy()).collect()

    results: list[pl.DataFrame] = []
    right_idx_str = {k: str(v) for k, v in right_index.items()}

    if eng == "ray":
        try:
            import ray as ray_mod

            if not ray_mod.is_initialized():
                ray_mod.init(ignore_reinit_error=True)
            remote_fn: Any = ray_mod.remote(_pixel_task_payload)
            futures = [
                remote_fn.remote(
                    order,
                    pix,
                    str(path),
                    right_idx_str,
                    right_hats,
                    right_all,
                    src1,
                    src2,
                    spec,
                    right_suffix,
                    eng,
                )
                for order, pix, path in left_pixels
            ]
            for part in ray_mod.get(futures):
                if part is not None:
                    results.append(part)
            if results:
                return pl.concat(results, how="diagonal_relaxed")
            return pl.DataFrame()
        except Exception as exc:
            logger.warning("HATS native Ray unavailable (%s); serial pixel tasks", exc)

    for order, pix, path in left_pixels:
        part = _pixel_task_payload(
            order,
            pix,
            str(path),
            right_idx_str,
            right_hats,
            right_all,
            src1,
            src2,
            spec,
            right_suffix,
            eng,
        )
        if part is not None:
            results.append(part)

    if not results:
        return pl.DataFrame()
    return pl.concat(results, how="diagonal_relaxed")
