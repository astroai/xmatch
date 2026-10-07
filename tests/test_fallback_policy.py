"""Strict engine-fallback contracts for partition-native matching."""

from __future__ import annotations

import builtins
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from xmatch import CatalogueSource, CrossMatchError, MatchSpec
from xmatch.matchers import _torchsky_match, _zone_match, sky_match


def _source(name: str, **kwargs) -> CatalogueSource:
    return CatalogueSource(
        name=name,
        is_local=True,
        ra_column="ra",
        dec_column="dec",
        **kwargs,
    )


def test_match_spec_rejects_unknown_fallback_policy() -> None:
    with pytest.raises(ValueError, match="fallback_policy"):
        MatchSpec(fallback_policy="silent")


def test_ray_strict_policy_fails_when_ray_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("xmatch.ray_engine.ray_available", lambda: False)
    left = pl.DataFrame({"ra": [10.0], "dec": [5.0]})
    right = pl.DataFrame({"ra": [10.00005], "dec": [5.00005]})
    with pytest.raises(CrossMatchError, match="requires Ray"):
        sky_match(
            _source("left"),
            _source("right"),
            left.lazy(),
            right.lazy(),
            MatchSpec(fallback_policy="error"),
            engine="ray",
        ).collect()


def test_zone_strict_policy_fails_when_healpix_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def no_healpix(name: str, *args: object, **kwargs: object):
        if name == "cdshealpix":
            raise ImportError("test missing cdshealpix")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_healpix)
    left = pl.DataFrame({"ra": [10.0], "dec": [5.0]})
    right = pl.DataFrame({"ra": [10.00005], "dec": [5.00005]})
    with pytest.raises(CrossMatchError, match="requires optional cdshealpix"):
        _zone_match(
            left,
            right,
            _source("left"),
            _source("right"),
            MatchSpec(fallback_policy="error"),
        )


def test_auto_stilts_falls_back_for_unsupported_match_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xmatch import stilts

    monkeypatch.setattr(stilts, "stilts_available", lambda _command: True)

    def stilts_must_not_run(*args: object, **kwargs: object):
        raise AssertionError("STILTS cannot implement this matcher")

    monkeypatch.setattr(stilts, "stilts_sky_match", stilts_must_not_run)
    left = pl.DataFrame({"ra": [10.0], "dec": [5.0], "rae": [0.1], "dee": [0.1]})
    right = pl.DataFrame({"ra": [10.00005], "dec": [5.0], "rae": [0.1], "dee": [0.1]})
    source_left = _source("left", ra_err_column="rae", dec_err_column="dee")
    source_right = _source("right", ra_err_column="rae", dec_err_column="dee")

    result = sky_match(
        source_left,
        source_right,
        left.lazy(),
        right.lazy(),
        MatchSpec(matcher="skyellipse"),
        engine="auto",
    ).collect()

    assert result.height == 1


def test_stilts_strict_policy_rejects_unsupported_match_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xmatch import stilts

    monkeypatch.setattr(stilts, "stilts_available", lambda _command: True)
    left = pl.DataFrame({"ra": [10.0], "dec": [5.0], "rae": [0.1], "dee": [0.1]})
    right = pl.DataFrame({"ra": [10.00005], "dec": [5.0], "rae": [0.1], "dee": [0.1]})
    source_left = _source("left", ra_err_column="rae", dec_err_column="dee")
    source_right = _source("right", ra_err_column="rae", dec_err_column="dee")

    with pytest.raises(CrossMatchError, match="cannot honor matcher='skyellipse'"):
        sky_match(
            source_left,
            source_right,
            left.lazy(),
            right.lazy(),
            MatchSpec(matcher="skyellipse", fallback_policy="error"),
            engine="stilts",
        )


def test_torchsky_engine_adapts_nearest_match_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")
    captured = {}

    def fake_crossmatch(left_ra, left_dec, right_ra, right_dec, *, radius_arcsec, find, **_kwargs):
        captured["inputs"] = (
            left_ra,
            left_dec,
            right_ra,
            right_dec,
            radius_arcsec,
            find,
        )
        return SimpleNamespace(
            left_index=np.array([0], dtype=np.int64),
            right_index=np.array([1], dtype=np.int64),
            separation_arcsec=np.array([0.25]),
        )

    monkeypatch.setattr("xmatch.matchers._load_torchsky_crossmatch", lambda: fake_crossmatch)
    left = pl.DataFrame({"ra": [10.0], "dec": [5.0]})
    right = pl.DataFrame({"ra": [50.0, 10.00005], "dec": [5.0, 5.00005]})

    left_idx, right_idx, separation = _torchsky_match(
        left,
        right,
        _source("left"),
        _source("right"),
        MatchSpec(radius_arcsec=1.5),
    )

    assert left_idx.tolist() == [0]
    assert right_idx.tolist() == [1]
    assert separation.tolist() == [0.25]
    assert captured["inputs"][-2:] == (1.5, "best")


def test_torchsky_skyerr_nan_sigma_is_not_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")

    def fake_crossmatch(left_ra, left_dec, right_ra, right_dec, *, radius_arcsec, find, **_kwargs):
        return SimpleNamespace(
            left_index=np.array([0], dtype=np.int64),
            right_index=np.array([0], dtype=np.int64),
            separation_arcsec=np.array([0.0]),
        )

    monkeypatch.setattr("xmatch.matchers._load_torchsky_crossmatch", lambda: fake_crossmatch)
    left = pl.DataFrame({"ra": [10.0], "dec": [5.0], "rae": [np.nan], "dee": [np.nan]})
    right = pl.DataFrame({"ra": [10.0], "dec": [5.0], "rae": [0.1], "dee": [0.1]})
    with pytest.raises(CrossMatchError, match="non-finite or negative RA position error"):
        _torchsky_match(
            left,
            right,
            _source("left", ra_err_column="rae", dec_err_column="dee"),
            _source("right", ra_err_column="rae", dec_err_column="dee"),
            MatchSpec(matcher="skyerr", max_error=3.0),
        )


@pytest.mark.parametrize(
    "spec",
    [
        MatchSpec(extra_distance_cols={"mag": 1.0}, fallback_policy="error"),
        MatchSpec(matcher="skyellipse", fallback_policy="error"),
    ],
)
def test_torchsky_strict_policy_rejects_unsupported_semantics(spec: MatchSpec) -> None:
    left = pl.DataFrame({"ra": [10.0], "dec": [5.0]})
    right = pl.DataFrame({"ra": [10.00005], "dec": [5.00005]})

    with pytest.raises(CrossMatchError, match="currently supports only"):
        sky_match(
            _source("left", default_pos_error_arcsec=0.1),
            _source("right", default_pos_error_arcsec=0.1),
            left.lazy(),
            right.lazy(),
            spec,
            engine="torchsky",
        ).collect()


def test_torchsky_missing_package_fails_loud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_torchsky():
        raise CrossMatchError("engine='torchsky' requires the optional torchsky package")

    monkeypatch.setattr("xmatch.matchers._load_torchsky_crossmatch", missing_torchsky)
    left = pl.DataFrame({"ra": [10.0], "dec": [5.0]})
    right = pl.DataFrame({"ra": [10.00005], "dec": [5.00005]})

    with pytest.raises(CrossMatchError, match="requires the optional torchsky"):
        sky_match(
            _source("left"),
            _source("right"),
            left.lazy(),
            right.lazy(),
            MatchSpec(),
            engine="torchsky",
        ).collect()
