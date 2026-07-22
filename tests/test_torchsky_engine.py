"""Cross-repository acceptance checks for the optional Torchsky engine."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from xmatch.astro_utils import propagate_space_motion_with_jacobian
from xmatch.matchers import MatchSpec, sky_match
from xmatch.sources import CatalogueSource

pytest.importorskip("torch")
pytest.importorskip("torchsky")


def _source(name: str, id_column: str) -> CatalogueSource:
    return CatalogueSource(
        name=name,
        is_local=True,
        id_column=id_column,
        ra_column="ra",
        dec_column="dec",
    )


def test_torchsky_space_motion_jacobian_adapter() -> None:
    import torchsky.wcs

    if not hasattr(torchsky.wcs, "propagate_space_motion_with_jacobian"):
        pytest.skip("installed Torchsky predates the local phase-space Jacobian API")
    result = propagate_space_motion_with_jacobian(
        np.array([269.452075]),
        np.array([4.693391]),
        np.array([-801.551]),
        np.array([10_362.394]),
        np.array([548.31]),
        np.array([-110.6]),
        np.array([2000.0]),
        2025.0,
    )

    assert result is not None
    ra, dec, jacobian = result
    np.testing.assert_allclose(ra, [269.44648069352934], atol=1e-10)
    np.testing.assert_allclose(dec, [4.765463779109472], atol=1e-10)
    assert jacobian.shape == (1, 6, 6)
    assert np.isfinite(jacobian).all()


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


def test_torchsky_engine_matches_fast_all_candidates() -> None:
    left = pl.DataFrame({"left_id": [0, 1], "ra": [10.0, 20.0], "dec": [0.0, 0.0]})
    right = pl.DataFrame(
        {
            "right_id": [0, 1, 2, 3],
            "ra": [10.0002, 10.0001, 20.0001, 80.0],
            "dec": [0.0, 0.0, 0.0, 0.0],
        }
    )
    spec = MatchSpec(radius_arcsec=2.0, find="all", fallback_policy="error")

    outputs = {
        engine: sky_match(
            _source("left", "left_id"),
            _source("right", "right_id"),
            left.lazy(),
            right.lazy(),
            spec,
            engine=engine,
        ).collect()
        for engine in ("fast", "torchsky")
    }

    outputs = {
        engine: output.sort(["left_id", "sep_arcsec", "right_id"], maintain_order=True)
        for engine, output in outputs.items()
    }
    assert outputs["torchsky"]["left_id"].to_list() == outputs["fast"]["left_id"].to_list()
    assert outputs["torchsky"]["right_id"].to_list() == outputs["fast"]["right_id"].to_list()
    np.testing.assert_allclose(
        outputs["torchsky"]["sep_arcsec"].to_numpy(),
        outputs["fast"]["sep_arcsec"].to_numpy(),
        rtol=0.0,
        atol=1e-6,
    )
