"""Strict engine-fallback contracts for partition-native matching."""

from __future__ import annotations

import builtins

import polars as pl
import pytest

from xmatch import CatalogueSource, CrossMatchError, MatchSpec
from xmatch.matchers import _zone_match, sky_match


def _source(name: str) -> CatalogueSource:
    return CatalogueSource(name=name, is_local=True, ra_column="ra", dec_column="dec")


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
