"""HATS-native pixel matcher tests."""

from pathlib import Path

import polars as pl
import pytest

from xmatcher import CatalogueSource, MatchSpec, hats_native
from xmatcher.exceptions import CrossMatchError


def _write_hats(root: Path, rows: list[dict], order: int = 0, pix: int = 0) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "properties").write_text("hats_col_ra=ra\nhats_col_dec=dec\nhats_npix_suffix=/\n")
    pix_dir = root / "dataset" / f"Norder={order}" / "Dir=0" / f"Npix={pix}"
    pix_dir.mkdir(parents=True)
    pl.DataFrame(rows).write_parquet(pix_dir / "catalog.parquet")
    return root


def test_list_and_load_hats_pixels(tmp_path: Path):
    root = _write_hats(
        tmp_path / "cat",
        [{"ra": 10.0, "dec": 5.0, "id": 1}, {"ra": 10.0001, "dec": 5.0001, "id": 2}],
    )
    pixels = hats_native.list_hats_pixels(root)
    assert len(pixels) == 1
    assert pixels[0][1] == 0
    src = CatalogueSource(
        name="c",
        is_local=False,
        access_method="hats",
        access_identifier=str(root),
        ra_column="ra",
        dec_column="dec",
    )
    df = hats_native.load_hats_all(src)
    assert df.height == 2


def test_hats_native_inner_match(tmp_path: Path):
    left = _write_hats(
        tmp_path / "left",
        [{"ra": 10.0, "dec": 5.0, "name": "a"}],
        pix=0,
    )
    right = _write_hats(
        tmp_path / "right",
        [{"ra": 10.0001, "dec": 5.00005, "name": "b"}],
        pix=0,
    )
    src1 = CatalogueSource(
        name="L",
        is_local=False,
        access_method="hats",
        access_identifier=str(left),
        ra_column="ra",
        dec_column="dec",
    )
    src2 = CatalogueSource(
        name="R",
        is_local=False,
        access_method="hats",
        access_identifier=str(right),
        ra_column="ra",
        dec_column="dec",
    )
    out = hats_native.hats_native_crossmatch(
        src1,
        src2,
        MatchSpec(radius_arcsec=2.0, join_type="1and2"),
        engine="fast",
    )
    assert out.height >= 1
    assert "sep_arcsec" in out.columns


def test_hats_native_torchsky_engine(tmp_path: Path):
    pytest.importorskip("torch")
    pytest.importorskip("torchsky")
    left = _write_hats(
        tmp_path / "left",
        [{"ra": 10.0, "dec": 5.0, "name": "a"}],
        pix=0,
    )
    right = _write_hats(
        tmp_path / "right",
        [{"ra": 10.0001, "dec": 5.00005, "name": "b"}],
        pix=0,
    )
    src1 = CatalogueSource(
        name="L",
        is_local=False,
        access_method="hats",
        access_identifier=str(left),
        ra_column="ra",
        dec_column="dec",
    )
    src2 = CatalogueSource(
        name="R",
        is_local=False,
        access_method="hats",
        access_identifier=str(right),
        ra_column="ra",
        dec_column="dec",
    )
    out = hats_native.hats_native_crossmatch(
        src1,
        src2,
        MatchSpec(radius_arcsec=2.0, join_type="1and2", fallback_policy="error"),
        engine="torchsky",
    )
    assert out.height >= 1
    assert "sep_arcsec" in out.columns


def test_hats_native_outer_join(tmp_path: Path):
    left = _write_hats(
        tmp_path / "left",
        [{"ra": 10.0, "dec": 5.0}, {"ra": 20.0, "dec": 5.0}],
    )
    right = _write_hats(
        tmp_path / "right",
        [{"ra": 10.0001, "dec": 5.00005}],
    )
    src1 = CatalogueSource(
        name="L",
        is_local=False,
        access_method="hats",
        access_identifier=str(left),
        ra_column="ra",
        dec_column="dec",
    )
    src2 = CatalogueSource(
        name="R",
        is_local=False,
        access_method="hats",
        access_identifier=str(right),
        ra_column="ra",
        dec_column="dec",
    )
    out = hats_native.hats_native_crossmatch(
        src1,
        src2,
        MatchSpec(radius_arcsec=2.0, join_type="1or2"),
        engine="fast",
    )
    assert out.height >= 2


def test_hats_native_requires_hats_side():
    src = CatalogueSource(name="a", is_local=True, ra_column="ra", dec_column="dec")
    with pytest.raises(CrossMatchError, match="at least one HATS"):
        hats_native.hats_native_crossmatch(
            src,
            src,
            MatchSpec(),
            engine="fast",
            local_lf1=pl.DataFrame({"ra": [0.0], "dec": [0.0]}).lazy(),
            local_lf2=pl.DataFrame({"ra": [0.0], "dec": [0.0]}).lazy(),
        )


def test_margin_pixels_includes_healpix_neighbors():
    pytest.importorskip("healpy")
    import healpy as hp

    order = 5
    nside = 2**order
    pix = 1000
    neigh = [int(n) for n in hp.get_all_neighbours(nside, pix, nest=True) if n >= 0]
    right = [pix] + neigh + [0, 1, 2, 9999]
    marg = hats_native.margin_pixels(order, pix, 1.5, right)
    assert pix in marg
    assert any(n in marg for n in neigh)
    assert 9999 not in marg


@pytest.mark.parametrize("join_type", ["1and2", "all1", "1not2", "1or2"])
def test_hats_native_ray_engine_joins(tmp_path: Path, join_type: str):
    ray = pytest.importorskip("ray")
    left = _write_hats(
        tmp_path / "left",
        [
            {"id": 1, "ra": 10.0, "dec": 5.0, "rae": 0.2, "dee": 0.2},
            {"id": 2, "ra": 25.0, "dec": 5.0, "rae": 0.2, "dee": 0.2},
        ],
        pix=0,
    )
    right = _write_hats(
        tmp_path / "right",
        [
            {"id": 10, "ra": 10.00005, "dec": 5.0, "rae": 0.2, "dee": 0.2},
            {"id": 20, "ra": 35.0, "dec": 5.0, "rae": 0.2, "dee": 0.2},
        ],
        pix=0,
    )
    src1 = CatalogueSource(
        name="L",
        is_local=False,
        access_method="hats",
        access_identifier=str(left),
        ra_column="ra",
        dec_column="dec",
        ra_err_column="rae",
        dec_err_column="dee",
    )
    src2 = CatalogueSource(
        name="R",
        is_local=False,
        access_method="hats",
        access_identifier=str(right),
        ra_column="ra",
        dec_column="dec",
        ra_err_column="rae",
        dec_err_column="dee",
    )
    spec = MatchSpec(radius_arcsec=2.0, join_type=join_type, matcher="skyerr", max_error=3.0)
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, logging_level=40)
    fast_out = hats_native.hats_native_crossmatch(src1, src2, spec, engine="fast")
    ray_out = hats_native.hats_native_crossmatch(src1, src2, spec, engine="ray")
    assert ray_out.height == fast_out.height
    assert set(ray_out.columns) == set(fast_out.columns)


@pytest.mark.parametrize("matcher", ["skyellipse", "lr", "ml", "auf", "macauff"])
def test_hats_native_ray_engine_matchers(tmp_path: Path, matcher: str):
    ray = pytest.importorskip("ray")
    left_rows = [
        {
            "id": i,
            "ra": 10.0 + i * 0.01,
            "dec": 5.0,
            "rae": 0.2,
            "dee": 0.2,
            "corr": 0.1,
            "mag": 18.0 + 0.1 * i,
            "color": 0.5 + 0.05 * i,
        }
        for i in range(12)
    ]
    right_rows = [
        {
            "id": 100 + i,
            "ra": 10.0 + i * 0.01 + 0.2 / 3600.0,
            "dec": 5.0,
            "rae": 0.2,
            "dee": 0.2,
            "corr": 0.1,
            "mag": 18.02 + 0.1 * i,
            "color": 0.51 + 0.05 * i,
        }
        for i in range(12)
    ]
    left = _write_hats(tmp_path / f"left_{matcher}", left_rows, pix=0)
    right = _write_hats(tmp_path / f"right_{matcher}", right_rows, pix=0)
    src1 = CatalogueSource(
        name="L",
        is_local=False,
        access_method="hats",
        access_identifier=str(left),
        ra_column="ra",
        dec_column="dec",
        ra_err_column="rae",
        dec_err_column="dee",
        corr_column="corr",
    )
    src2 = CatalogueSource(
        name="R",
        is_local=False,
        access_method="hats",
        access_identifier=str(right),
        ra_column="ra",
        dec_column="dec",
        ra_err_column="rae",
        dec_err_column="dee",
        corr_column="corr",
    )
    spec = MatchSpec(
        radius_arcsec=2.0,
        matcher=matcher,
        max_error=3.0,
        lr_magnitude_column="mag" if matcher == "lr" else None,
        ml_color_columns=["mag", "color"] if matcher == "ml" else None,
        macauff_flux_columns=["mag"] if matcher == "macauff" else None,
    )
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, logging_level=40)
    fast_out = hats_native.hats_native_crossmatch(src1, src2, spec, engine="fast").sort("id")
    ray_out = hats_native.hats_native_crossmatch(src1, src2, spec, engine="ray").sort("id")
    assert ray_out.height == fast_out.height == 12
    assert ray_out["id_2"].to_list() == fast_out["id_2"].to_list()
