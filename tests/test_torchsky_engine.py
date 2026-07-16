"""Cross-repository acceptance checks for the optional Torchsky engine."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from xmatch.matchers import MatchSpec, sky_match
from xmatch.sources import CatalogueSource

pytest.importorskip("torchsky")


def _source(name: str, id_column: str) -> CatalogueSource:
    return CatalogueSource(
        name=name,
        is_local=True,
        id_column=id_column,
        ra_column="ra",
        dec_column="dec",
    )


def test_torchsky_engine_matches_fast_nearest_neighbours() -> None:
    rng = np.random.default_rng(271828)
    left_ra = rng.uniform(0.0, 360.0, 64)
    left_dec = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, 64)))
    right_ra = rng.uniform(0.0, 360.0, 96)
    right_dec = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, 96)))
    right_ra[:24] = np.mod(left_ra[:24] + rng.uniform(-0.15, 0.15, 24) / 3600.0, 360.0)
    right_dec[:24] = left_dec[:24] + rng.uniform(-0.15, 0.15, 24) / 3600.0

    left = pl.DataFrame({"left_id": np.arange(64), "ra": left_ra, "dec": left_dec})
    right = pl.DataFrame({"right_id": np.arange(96), "ra": right_ra, "dec": right_dec})
    spec = MatchSpec(radius_arcsec=1.0, fallback_policy="error")

    outputs = {
        engine: sky_match(
            _source("left", "left_id"),
            _source("right", "right_id"),
            left.lazy(),
            right.lazy(),
            spec,
            engine=engine,
        )
        .collect()
        .sort("left_id")
        for engine in ("fast", "torchsky")
    }

    assert outputs["torchsky"]["left_id"].to_list() == outputs["fast"]["left_id"].to_list()
    assert outputs["torchsky"]["right_id"].to_list() == outputs["fast"]["right_id"].to_list()
    np.testing.assert_allclose(
        outputs["torchsky"]["sep_arcsec"].to_numpy(),
        outputs["fast"]["sep_arcsec"].to_numpy(),
        rtol=0.0,
        atol=1e-6,
    )
