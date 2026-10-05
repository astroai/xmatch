"""Epoch spill uses evaluated uncertainties and preserves astrometric precision."""

import numpy as np
import polars as pl
import pytest

from xmatch import CatalogueSource, MatchRequest, MatchSpec, SideOverrides, astro_utils
from xmatch.exceptions import CrossMatchError
from xmatch.out_of_core import _align_epoch, match_to_output
from xmatch.sources import ASTROMETRIC_COVARIANCE_KEYS


def _source(frame, **metadata):
    return CatalogueSource(
        name="motion",
        is_local=True,
        _frame=frame.lazy(),
        id_column="id",
        ra_column="ra",
        dec_column="dec",
        epoch=2000.0,
        pm_ra_column="pmra",
        pm_dec_column="pmdec",
        **metadata,
    )


def _covariance_frame():
    row = {
        "id": [1],
        "ra": [10.0],
        "dec": [0.0],
        "pmra": [0.0],
        "pmdec": [0.0],
        "ra_error": [1.0],
        "dec_error": [1.0],
        "parallax_error": [1.0],
        "pmra_error": [50.0],
        "pmdec_error": [50.0],
    }
    row.update({key: [0.0] for key in ASTROMETRIC_COVARIANCE_KEYS[5:]})
    return pl.DataFrame(row)


def _covariance_source(frame, **metadata):
    return _source(
        frame,
        ra_err_column="ra_error",
        dec_err_column="dec_error",
        pos_err_units="mas",
        astrometric_covariance_columns={key: key for key in ASTROMETRIC_COVARIANCE_KEYS},
        **metadata,
    )


def _zero_motion_jacobian(ra, dec, pmra, pmdec, epochs, target):
    """Exact tangent derivative at zero PM: position + dt times PM."""
    assert np.all(np.asarray(pmra) == 0) and np.all(np.asarray(pmdec) == 0)
    n = len(ra)
    jacobian = np.zeros((n, 2, 4))
    jacobian[:, 0, 0] = jacobian[:, 1, 1] = 1.0
    jacobian[:, 0, 2] = jacobian[:, 1, 3] = target - np.asarray(epochs)
    return np.asarray(ra), np.asarray(dec), jacobian


@pytest.mark.parametrize("dtype", [pl.Int64, pl.Float32])
def test_epoch_alignment_accepts_integer_and_float32_coordinates(dtype):
    frame = pl.DataFrame(
        {"id": [1], "ra": [10], "dec": [0], "pmra": [100.0], "pmdec": [0.0]}
    ).with_columns(pl.col("ra", "dec").cast(dtype))
    aligned = _align_epoch(frame.lazy(), _source(frame), 2010.0).collect()
    assert aligned.schema["ra"] == aligned.schema["dec"] == pl.Float64
    assert aligned["ra"].item() == pytest.approx(10.0 + 1.0 / 3600, abs=1e-9)


def test_skyerr_epoch_spill_inflates_uncertainty_and_halo(tmp_path, monkeypatch):
    monkeypatch.setattr(astro_utils, "propagate_proper_motion_with_jacobian", _zero_motion_jacobian)
    left = _covariance_frame()
    right = pl.DataFrame(
        {"id": [2], "ra": [10.0 + 0.4 / 3600], "dec": [0.0], "ra_error": [1.0], "dec_error": [1.0]}
    )
    left_path, right_path, output = (
        tmp_path / "left.parquet",
        tmp_path / "right.parquet",
        tmp_path / "pairs.parquet",
    )
    left.write_parquet(left_path)
    right.write_parquet(right_path)
    source_left = _covariance_source(left)
    source_left.path, source_left._frame = left_path, None
    source_right = CatalogueSource(
        name="reference",
        is_local=True,
        path=right_path,
        id_column="id",
        ra_column="ra",
        dec_column="dec",
        epoch=2010.0,
        ra_err_column="ra_error",
        dec_err_column="dec_error",
        pos_err_units="mas",
    )
    request = MatchRequest(
        left_path,
        right_path,
        spec=MatchSpec(
            matcher="skyerr",
            target_epoch=2010.0,
            max_error=1.0,
            find="all",
            fallback_policy="error",
        ),
        side1=SideOverrides(columns=("id", "ra", "dec")),
        side2=SideOverrides(columns=("id", "ra", "dec")),
        output_file=output,
        engine="fast",
        memory_budget_bytes=2048,
        partition_order=4,
    )
    match_to_output(request, source_left, source_right)
    result = pl.read_parquet(output)
    assert result["id"].to_list() == [1] and result["id_2"].to_list() == [2]
    assert result["sep_arcsec"].item() == pytest.approx(0.4, abs=1e-8)
    assert not any(name.startswith("_xmatch_spill_") for name in result.columns)


def test_epoch_covariance_alignment_rejects_missing_invalid_or_unavailable_covariance(monkeypatch):
    frame = _covariance_frame()
    no_mapping = _source(
        frame, ra_err_column="ra_error", dec_err_column="dec_error", pos_err_units="mas"
    )
    with pytest.raises(CrossMatchError, match="covariance"):
        _align_epoch(frame.lazy(), no_mapping, 2010.0, propagate_covariance=True).collect()
    invalid = frame.with_columns(pl.lit(None, dtype=pl.Float64).alias("pmra_error"))
    with pytest.raises(CrossMatchError, match="covariance"):
        _align_epoch(
            invalid.lazy(), _covariance_source(invalid), 2010.0, propagate_covariance=True
        ).collect()
    monkeypatch.setattr(astro_utils, "propagate_proper_motion_with_jacobian", lambda *args: None)
    with pytest.raises(CrossMatchError, match="Torchsky"):
        _align_epoch(
            frame.lazy(), _covariance_source(frame), 2010.0, propagate_covariance=True
        ).collect()


def test_epoch_covariance_alignment_keeps_static_rows_without_motion_or_covariance():
    frame = pl.DataFrame(
        {"id": [1], "ra": [10.0], "dec": [0.0], "ra_error": [1.0], "dec_error": [1.0]}
    )
    source = CatalogueSource(
        name="static",
        is_local=True,
        _frame=frame.lazy(),
        id_column="id",
        ra_column="ra",
        dec_column="dec",
        epoch=2010.0,
        ra_err_column="ra_error",
        dec_err_column="dec_error",
        pos_err_units="mas",
    )
    result = _align_epoch(frame.lazy(), source, 2010.0, propagate_covariance=True).collect()
    assert result["ra"].to_list() == [10.0]


def test_epoch_covariance_alignment_refuses_undeclared_rv_uncertainty():
    frame = _covariance_frame().with_columns(
        pl.lit(20.0).alias("parallax"), pl.lit(10.0).alias("rv")
    )
    source = _covariance_source(frame, parallax_column="parallax", radial_velocity_column="rv")
    with pytest.raises(CrossMatchError, match="radial.velocity.*deterministic"):
        _align_epoch(frame.lazy(), source, 2010.0, propagate_covariance=True).collect()


def test_epoch_covariance_mixed_rows_preserve_static_uncertainty_and_identity(monkeypatch):
    monkeypatch.setattr(astro_utils, "propagate_proper_motion_with_jacobian", _zero_motion_jacobian)
    moving = _covariance_frame().with_columns(pl.lit(2000.0).alias("epoch"))
    static = moving.with_columns(
        pl.lit(2**60 + 1).alias("id"),
        pl.lit(2010.0).alias("epoch"),
        pl.lit(None, dtype=pl.Float64).alias("pmra"),
        pl.lit(None, dtype=pl.Float64).alias("pmra_error"),
    )
    frame = pl.concat([static, moving])
    source = _covariance_source(frame, epoch_column="epoch")
    result = _align_epoch(frame.lazy(), source, 2010.0, propagate_covariance=True).collect()
    assert result["id"].to_list() == [2**60 + 1, 1]
    assert result["_xmatch_spill_epoch_sigma"].to_list() == pytest.approx(
        [
            np.sqrt(2) / 1000,
            np.sqrt(500002) / 1000,
        ]
    )


def test_epoch_covariance_physical_path_accepts_explicit_deterministic_rv(monkeypatch):
    def zero_motion_space_jacobian(ra, dec, pmra, pmdec, parallax, rv, epochs, target):
        assert np.all(np.asarray(rv) == 0) and np.all(np.asarray(pmra) == 0)
        jacobian = np.broadcast_to(np.eye(6), (len(ra), 6, 6)).copy()
        jacobian[:, 0, 2] = jacobian[:, 1, 3] = target - np.asarray(epochs)
        return np.asarray(ra), np.asarray(dec), jacobian

    monkeypatch.setattr(
        astro_utils, "propagate_space_motion_with_jacobian", zero_motion_space_jacobian
    )
    frame = _covariance_frame().with_columns(
        pl.lit(20.0).alias("parallax"), pl.lit(0.0).alias("rv")
    )
    source = _covariance_source(
        frame,
        parallax_column="parallax",
        radial_velocity_column="rv",
        release_metadata={"radial_velocity_uncertainty": "deterministic"},
    )
    result = _align_epoch(frame.lazy(), source, 2010.0, propagate_covariance=True).collect()
    assert result["_xmatch_spill_epoch_sigma"].item() == pytest.approx(np.sqrt(500002) / 1000)


def test_epoch_covariance_matches_real_torchsky_zero_motion_jacobian():
    pytest.importorskip("torchsky.wcs")
    frame = _covariance_frame()
    result = _align_epoch(
        frame.lazy(), _covariance_source(frame), 2010.0, propagate_covariance=True
    ).collect()
    sigma = result["_xmatch_spill_epoch_sigma"].item()
    assert sigma == pytest.approx(np.sqrt(2 * (1 + 10**2 * 50**2)) / 1000, rel=1e-10)


def test_epoch_covariance_scalar_epoch_broadcasts_over_multiple_rows(monkeypatch):
    monkeypatch.setattr(astro_utils, "propagate_proper_motion_with_jacobian", _zero_motion_jacobian)
    frame = pl.concat(
        [
            _covariance_frame(),
            _covariance_frame().with_columns(pl.lit(2, dtype=pl.Int64).alias("id")),
        ]
    )
    result = _align_epoch(
        frame.lazy(), _covariance_source(frame), 2010.0, propagate_covariance=True
    ).collect()
    assert result["id"].to_list() == [1, 2]
    assert result["_xmatch_spill_epoch_sigma"].to_list() == pytest.approx(
        [np.sqrt(500002) / 1000] * 2
    )


@pytest.mark.parametrize("find", ["best", "all"])
def test_target_epoch_skyerr_matches_with_and_without_spill(tmp_path, monkeypatch, find):
    from xmatch.matchers import sky_match

    monkeypatch.setattr(astro_utils, "propagate_proper_motion_with_jacobian", _zero_motion_jacobian)
    left = _covariance_frame()
    right = pl.DataFrame(
        {
            "id": [2, 3, 4],
            "ra": [10 + offset / 3600 for offset in (0.4, 0.6, 0.9)],
            "dec": [0.0] * 3,
            "ra_error": [1.0] * 3,
            "dec_error": [1.0] * 3,
        }
    )
    left_path, right_path = tmp_path / "left.parquet", tmp_path / "right.parquet"
    left.write_parquet(left_path)
    right.write_parquet(right_path)
    source_left = _covariance_source(left)
    source_left.path, source_left._frame = left_path, None
    source_right = CatalogueSource(
        name="right",
        is_local=True,
        path=right_path,
        id_column="id",
        ra_column="ra",
        dec_column="dec",
        epoch=2010.0,
        ra_err_column="ra_error",
        dec_err_column="dec_error",
        pos_err_units="mas",
    )
    spec = MatchSpec(
        matcher="skyerr", target_epoch=2010.0, max_error=1.0, find=find, fallback_policy="warn"
    )
    eager = sky_match(
        source_left, source_right, left.lazy(), right.lazy(), spec, engine="fast"
    ).collect()
    request = MatchRequest(
        left_path,
        right_path,
        spec=spec,
        engine="fast",
        memory_budget_bytes=2048,
        partition_order=4,
        output_file=tmp_path / "spill.parquet",
    )
    match_to_output(request, source_left, source_right)
    spill = pl.read_parquet(request.output_file)
    expected = [2] if find == "best" else [2, 3]
    assert eager["id_2"].sort().to_list() == spill["id_2"].sort().to_list() == expected
    assert (
        eager.select("id", "id_2", "sep_arcsec")
        .sort("id_2")
        .equals(spill.select("id", "id_2", "sep_arcsec").sort("id_2"))
    )
    assert not any(name.startswith("_xmatch_spill_epoch_sigma") for name in eager.columns)


@pytest.mark.parametrize("fallback", ["warn", "error"])
@pytest.mark.parametrize("pm_prior", [False, True])
def test_eager_target_epoch_skyerr_fails_on_missing_covariance_even_with_warn(fallback, pm_prior):
    from xmatch.matchers import sky_match

    frame = _covariance_frame()
    source = _source(
        frame, ra_err_column="ra_error", dec_err_column="dec_error", pos_err_units="mas"
    )
    with pytest.raises(CrossMatchError, match="covariance"):
        sky_match(
            source,
            source,
            frame.lazy(),
            frame.lazy(),
            MatchSpec(
                matcher="skyerr", target_epoch=2010.0, fallback_policy=fallback, pm_prior=pm_prior
            ),
            engine="fast",
        )


def test_eager_epoch_skyerr_outer_rows_do_not_leak_evaluated_uncertainty(monkeypatch):
    from xmatch.matchers import sky_match

    monkeypatch.setattr(astro_utils, "propagate_proper_motion_with_jacobian", _zero_motion_jacobian)
    left = _covariance_frame()
    right = pl.DataFrame(
        {"id": [2], "ra": [50.0], "dec": [0.0], "ra_error": [1.0], "dec_error": [1.0]}
    )
    source_right = CatalogueSource(
        name="right",
        is_local=True,
        _frame=right.lazy(),
        id_column="id",
        ra_column="ra",
        dec_column="dec",
        epoch=2010.0,
        ra_err_column="ra_error",
        dec_err_column="dec_error",
        pos_err_units="mas",
    )
    result = sky_match(
        _covariance_source(left),
        source_right,
        left.lazy(),
        right.lazy(),
        MatchSpec(matcher="skyerr", target_epoch=2010.0, join_type="1or2"),
        engine="fast",
    ).collect()
    assert result.height == 2
    assert not any(name.startswith("_xmatch_spill_epoch_sigma") for name in result.columns)


def test_epoch_uncertainty_prior_uses_existing_radial_rms_convention(monkeypatch):
    from xmatch import matchers

    monkeypatch.setattr(matchers, "_galactic_latitude", lambda ra, dec: np.zeros(len(ra)))
    frame = pl.DataFrame(
        {"id": [1], "ra": [10.0], "dec": [0.0], "ra_error": [3.0], "dec_error": [4.0]}
    )
    source = CatalogueSource(
        name="prior",
        is_local=True,
        _frame=frame.lazy(),
        ra_column="ra",
        dec_column="dec",
        epoch=1900.0,
        ra_err_column="ra_error",
        dec_err_column="dec_error",
    )
    evaluated = _align_epoch(
        frame.lazy(), source, 2000.0, propagate_covariance=True, pm_prior=True
    ).collect()
    # Reference radial RMS is 5 arcsec; the declared model adds 10 mas/yr
    # * 100 years = 1 arcsec radial RMS, once: sqrt(5**2 + 1**2).
    assert evaluated["_xmatch_spill_epoch_sigma"].item() == pytest.approx(np.sqrt(26))
    assert evaluated["ra"].item() == 10.0


@pytest.mark.parametrize("kind", ["unknown_motion", "missing_errors", "stilts"])
def test_eager_epoch_skyerr_refuses_unknown_or_unsupported_science(kind):
    from xmatch.matchers import sky_match

    frame = _covariance_frame()
    source = _covariance_source(frame)
    if kind == "unknown_motion":
        source.pm_ra_column = source.pm_dec_column = None
    elif kind == "missing_errors":
        source.ra_err_column = source.dec_err_column = None
    with pytest.raises(CrossMatchError):
        sky_match(
            source,
            source,
            frame.lazy(),
            frame.lazy(),
            MatchSpec(matcher="skyerr", target_epoch=2010.0, fallback_policy="warn"),
            engine="stilts" if kind == "stilts" else "fast",
        )


def test_epoch_alignment_preserves_empty_catalogue_without_imputing_motion():
    frame = pl.DataFrame(
        schema={
            "ra": pl.Float64,
            "dec": pl.Float64,
            "ra_error": pl.Float64,
            "dec_error": pl.Float64,
        }
    )
    source = CatalogueSource(
        name="empty",
        is_local=True,
        _frame=frame.lazy(),
        ra_column="ra",
        dec_column="dec",
        epoch=2000.0,
        ra_err_column="ra_error",
        dec_err_column="dec_error",
    )
    result = _align_epoch(frame.lazy(), source, 2010.0, propagate_covariance=True).collect()
    assert result.is_empty()


def test_epoch_skyerr_refuses_physically_inconsistent_velocity(monkeypatch):
    monkeypatch.setattr(astro_utils, "propagate_proper_motion_with_jacobian", _zero_motion_jacobian)
    frame = _covariance_frame().with_columns(
        pl.lit(20.0).alias("parallax"), pl.lit(200000.0).alias("rv")
    )
    source = _covariance_source(
        frame,
        parallax_column="parallax",
        radial_velocity_column="rv",
        release_metadata={"radial_velocity_uncertainty": "deterministic"},
    )
    with pytest.raises(CrossMatchError, match="physically"):
        _align_epoch(frame.lazy(), source, 2010.0, propagate_covariance=True).collect()


@pytest.mark.parametrize("target_epoch", [None, 2000.0])
def test_direct_sky_match_checks_declared_coordinate_frames(target_epoch):
    from xmatch.matchers import sky_match

    frame = _covariance_frame()
    left = _covariance_source(frame)
    right = _covariance_source(frame)
    right.frame = "galactic"
    with pytest.raises(CrossMatchError, match="frame|ICRS"):
        sky_match(
            left,
            right,
            frame.lazy(),
            frame.lazy(),
            MatchSpec(target_epoch=target_epoch),
            engine="fast",
        )
    left.frame = "galactic"
    if target_epoch is None:
        assert (
            sky_match(left, right, frame.lazy(), frame.lazy(), MatchSpec(), engine="fast")
            .collect()
            .height
            == 1
        )
    else:
        with pytest.raises(CrossMatchError, match="ICRS"):
            sky_match(
                left,
                right,
                frame.lazy(),
                frame.lazy(),
                MatchSpec(target_epoch=target_epoch),
                engine="fast",
            )


def test_direct_spill_match_rejects_mixed_coordinate_frames(tmp_path):
    frame = pl.DataFrame({"id": [1], "ra": [10.0], "dec": [0.0]})
    path = tmp_path / "source.parquet"
    frame.write_parquet(path)
    left = CatalogueSource(
        name="left", is_local=True, path=path, ra_column="ra", dec_column="dec", frame="icrs"
    )
    right = CatalogueSource(
        name="right", is_local=True, path=path, ra_column="ra", dec_column="dec", frame="galactic"
    )
    request = MatchRequest(
        path, path, output_file=tmp_path / "result.parquet", memory_budget_bytes=2048, engine="fast"
    )
    with pytest.raises(CrossMatchError, match="frame"):
        match_to_output(request, left, right)
    assert not request.output_file.exists()
