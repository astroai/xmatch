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


def test_torchsky_engine_adapts_nearest_match_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_crossmatch(left_ra, left_dec, right_ra, right_dec, *, radius_arcsec, find):
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


@pytest.mark.parametrize(
    "spec",
    [
        MatchSpec(matcher="skyerr", fallback_policy="error"),
        MatchSpec(extra_distance_cols={"mag": 1.0}, fallback_policy="error"),
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


def test_torchsky_warn_policy_falls_back_to_fast(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def missing_torchsky():
        raise CrossMatchError("engine='torchsky' requires the optional torchsky package")

    monkeypatch.setattr("xmatch.matchers._load_torchsky_crossmatch", missing_torchsky)
    left = pl.DataFrame({"ra": [10.0], "dec": [5.0]})
    right = pl.DataFrame({"ra": [10.00005], "dec": [5.00005]})

    out = sky_match(
        _source("left"),
        _source("right"),
        left.lazy(),
        right.lazy(),
        MatchSpec(),
        engine="torchsky",
    ).collect()

    assert out.height == 1
    assert "falling back to fast engine" in caplog.text
