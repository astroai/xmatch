"""Large-catalogue crossmatch tests against the :mod:`xmatch.stilts` backend.

The existing unit tests in :mod:`tests.test_matchers` cover STILTS with rows in
the single digits. They exercise the *math* but say nothing about what happens
when the backend is fed ``O(10_000)`` rows on each side: tempdir cleanup, the
polars-to-astropy FITS round-trip, the result partitioning, and the engine
plumbing all behave differently at scale.

These tests build two deterministic ~10k-row synthetic catalogues and check
that the STILTS invocation

* finds **exactly** the matches we planted, no more and no fewer;
* reports separations consistent with an independent great-circle calculation;
* produces the right schema when invoked through the high-level
  :class:`xmatch.CrossMatch` API and writes an output file;
* does not silently degrade into the :mod:`xmatch.astropy` fallback if STILTS
  ever fails to launch in CI.

The synthetic catalogues are designed so the bulk of each catalogue lives on a
wide rectangular grid (Dec ∈ [-30, 30], 0.5–0.6° spacing) far coarser than the
1.5 arcsec match radius; the only "matches" are a small number of deliberate
pairs injected outside the grid's bounding box.
"""

import time

import numpy as np
import polars as pl
import pytest

from xmatch import CrossMatch, stilts
from xmatch.matchers import MatchSpec
from xmatch.sources import CatalogueSource

# Skip the entire module when STILTS isn't on the path. Mirrors the convention
# used in tests/test_matchers.py so a missing-Java environment doesn't
# generate spurious failures.
pytestmark = [
    pytest.mark.slow,  # filter with `pytest -m "not slow"` on small CI runners
    pytest.mark.skipif(
        not stilts.stilts_available(), reason="STILTS executable not available on PATH"
    ),
]  # slow listed first so its reason surfaces ahead of the STILTS skipif in pytest -rs output


# --------------------------------------------------------------------------- #
# Catalogue construction
# --------------------------------------------------------------------------- #
_N_BASE = 10_000  # rows per catalogue
_INJECT = 50  # deliberate close pairs appended to each catalogue
_RADIUS_ARCSEC = 1.5  # match radius used in these tests
_INJECT_SEP_ARCSEC = 0.7  # intended angular separation of each injected pair


def _grid_positions(n: int, *, ra_phase: float) -> tuple[np.ndarray, np.ndarray]:
    """Return ``n`` (ra, dec) pairs evenly spread on a wide grid.

    The grid spans Dec ∈ [-29.4°, 30°] stepping by 0.6° and RA ∈ [0°, 49.5°]
    stepping by 0.5° (100 cells per axis). The right-hand catalogue shifts its
    RA cells by ``ra_phase * 0.25°`` so no row coincides with a row of the left
    catalogue (the left grid lives on multiples of 0.5° in RA; the right grid
    lives at offsets {0.25° + k·0.5°}). With Dec capped at |30°|,
    ``cos(dec) ≥ 0.866`` and the minimum great-circle separation between *any*
    left-grid row and *any* right-grid row is at least ``~1559 arcsec`` --
    guaranteed ≫ the 1.5″ match radius.
    """
    idx = np.arange(n)
    ra = (idx % 100) * 0.5 + ra_phase * 0.25  # 0/0.25° → 49.5/49.75°
    dec = 30.0 - (idx // 100) * 0.6  # 30° → -29.4°
    return ra.astype(np.float64), dec.astype(np.float64)


def _build_catalogue(
    seed: int,
    *,
    inject_ras: np.ndarray,
    inject_decs: np.ndarray,
    ra_phase: float,
) -> pl.DataFrame:
    """Assemble a catalogue of ``_N_BASE`` rows.

    Rows 0..(_N_BASE - _INJECT - 1) come from the wide grid; the last
    ``_INJECT`` rows are the deliberate "injected" positions supplied via the
    keyword arguments.
    """
    n_grid = _N_BASE - _INJECT
    ra_grid, dec_grid = _grid_positions(n_grid, ra_phase=ra_phase)
    ra = np.concatenate([ra_grid, np.asarray(inject_ras, dtype=np.float64)])
    dec = np.concatenate([dec_grid, np.asarray(inject_decs, dtype=np.float64)])
    return pl.DataFrame(
        {
            "source_id": np.arange(ra.size, dtype=np.int64),
            "ra": ra,
            "dec": dec,
        }
    )


@pytest.fixture(scope="module")
def pair() -> tuple[pl.DataFrame, pl.DataFrame, list[int]]:
    """Build a deterministic left/right pair with exactly ``_INJECT`` matches."""
    rng = np.random.default_rng(42)
    base_ras = rng.uniform(60.0, 80.0, _INJECT)
    base_decs = rng.uniform(-30.0, -10.0, _INJECT)
    # RA offset such that great-circle separation is _INJECT_SEP_ARCSEC,
    # irrespective of dec (ΔDec = 0 ⇒ sep = ΔRA × cos(dec)).
    ra_off = (_INJECT_SEP_ARCSEC / 3600.0) / np.cos(np.radians(base_decs))
    left = _build_catalogue(
        42,
        inject_ras=base_ras,
        inject_decs=base_decs,
        ra_phase=0.0,
    )
    right = _build_catalogue(
        43,
        inject_ras=base_ras + ra_off,
        inject_decs=base_decs,
        ra_phase=1.0,  # → right grid is offset by 0.5° in RA
    )
    injected_ids = list(range(_N_BASE - _INJECT, _N_BASE))
    return left, right, injected_ids


def _great_circle_arcsec(
    ra1: np.ndarray, dec1: np.ndarray, ra2: np.ndarray, dec2: np.ndarray
) -> np.ndarray:
    """Vectorised great-circle separation in arcsec (independent reference)."""
    lon1, lat1, lon2, lat2 = (
        np.radians(np.asarray(a, dtype=float)) for a in (ra1, dec1, ra2, dec2)
    )
    dlon, dlat = lon2 - lon1, lat2 - lat1
    hav = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0)))) * 3600.0


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
# Call the STILTS backend directly so a STILTS failure surfaces as a real test
# failure -- matchers.sky_match() would catch any exception and silently
# substitute the astropy engine, which would mask the regression.
def _stilts(left: pl.DataFrame, right: pl.DataFrame, spec: MatchSpec) -> pl.DataFrame:
    return stilts.stilts_sky_match(
        CatalogueSource(name="L", is_local=True, ra_column="ra", dec_column="dec"),
        CatalogueSource(name="R", is_local=True, ra_column="ra", dec_column="dec"),
        left,
        right,
        spec,
    )


def test_stilts_sky_match_finds_only_injected_pairs(pair):
    """STILTS on 10k×10k @ 1.5″ returns exactly the 50 injected pairs."""
    left, right, injected_ids = pair
    spec = MatchSpec(radius_arcsec=_RADIUS_ARCSEC, matcher="sky", find="best")

    out = _stilts(left, right, spec)

    assert out.height == _INJECT, f"expected exactly {_INJECT} injected matches, got {out.height}"
    assert sorted(out["source_id"].to_list()) == sorted(injected_ids)
    # STILTS suffixes overlapping right-side columns with "_2"; verify both
    # points of each pair landed on the same injection id range.
    right_ids = out["source_id_2"].to_list()
    assert sorted(right_ids) == sorted(injected_ids)


def test_stilts_sky_match_reported_separations_match_reference(pair):
    """The reported ``sep_arcsec`` reproduces an independent great-circle
    computation to within numerical precision."""
    left, right, _ = pair
    spec = MatchSpec(radius_arcsec=_RADIUS_ARCSEC, matcher="sky", find="best")

    out = _stilts(left, right, spec).sort("source_id")

    expected = _great_circle_arcsec(
        out["ra"].to_numpy(),
        out["dec"].to_numpy(),
        out["ra_2"].to_numpy(),
        out["dec_2"].to_numpy(),
    )
    assert np.allclose(out["sep_arcsec"].to_numpy(), expected, atol=1e-3)
    # And every reported separation should sit inside the match radius.
    assert (out["sep_arcsec"].to_numpy() <= _RADIUS_ARCSEC).all()


def test_stilts_sky_match_outer_join_keeps_unmatched_left(pair):
    """``join_type=1or2`` must surface every left row plus the 50 matched pairs."""
    left, right, _ = pair
    spec = MatchSpec(
        radius_arcsec=_RADIUS_ARCSEC,
        matcher="sky",
        find="best",
        join_type="1or2",
    )

    out = _stilts(left, right, spec)

    # The outer-join semantics must surface every left source_id (matched or
    # unmatched).
    assert out["source_id"].drop_nulls().n_unique() == _N_BASE
    matched = out.drop_nulls(subset=["source_id", "source_id_2"])
    if matched["source_id_2"].dtype.is_float():
        matched = matched.filter(matched["source_id_2"].is_nan().not_())
    if matched["source_id"].dtype.is_float():
        matched = matched.filter(matched["source_id"].is_nan().not_())
    assert matched.height == _INJECT
    # Cast to int for comparison if it was read as float
    matched_ids = matched["source_id_2"].cast(pl.Int64).to_list()
    assert sorted(matched_ids) == list(range(_N_BASE - _INJECT, _N_BASE))


def test_stilts_high_level_crossmatch_streams_to_parquet(pair, tmp_path, monkeypatch):
    """``LazyFrame`` inputs → :meth:`CrossMatch.crossmatch(lazy=True)` →
    ``sink_parquet(engine='streaming')``.

    Exercises the polars streaming pipeline that backs xmatch's "results
    stream straight to disk so large matches never need to fit in memory" claim
    (see README.md). The match output is delivered as a ``LazyFrame`` and
    sinked via polars' streaming executor; the result is never collected in
    RAM.
    """
    left, right, _ = pair
    left_lf = left.lazy()
    right_lf = right.lazy()
    cm = CrossMatch()

    # Patch out the astropy engine so any silent fallback raises.
    import xmatch.matchers as _matchers

    def _raise_if_called(*_a, **_kw):  # pragma: no cover - only on regression
        raise AssertionError("xmatch fell back to the astropy engine; STILTS did not run")

    monkeypatch.setattr(_matchers, "_astropy_match", _raise_if_called)

    result_lf = cm.crossmatch(
        left_lf,
        right_lf,
        radius_arcsec=_RADIUS_ARCSEC,
        matcher="sky",
        find="best",
        engine="stilts",
        lazy=True,
    )

    # Sanity-check that a streaming execution plan can be produced — polars
    # raises ComputeError on plans that cannot be streamed, so a successful
    # plan implies the result is streamable.
    plan = result_lf.explain(engine="streaming")
    assert plan.strip()

    # Sink straight to disk using polars' streaming executor explicitly; this
    # is stricter than xmatch's own io_utils.write_frame (which lets polars
    # pick 'auto') so a future default change can't silently regress this test.
    out_path = tmp_path / "result.parquet"
    result_lf.sink_parquet(out_path, engine="streaming")
    assert out_path.exists() and out_path.stat().st_size > 0

    result = pl.read_parquet(out_path)
    assert result.height == _INJECT
    assert "sep_arcsec" in result.columns
    assert "source_id_2" in result.columns


def test_stilts_sky_match_runs_in_reasonable_time(pair):
    """10k×10k @ 1.5″ should comfortably finish in under a minute. Generous
    bound keep this test from being flaky on slow CI but it would still
    regress if, e.g., someone disabled STILTS' HEALPix-style partitioning."""
    left, right, _ = pair
    spec = MatchSpec(radius_arcsec=_RADIUS_ARCSEC, matcher="sky", find="best")

    t0 = time.perf_counter()
    _stilts(left, right, spec)
    elapsed = time.perf_counter() - t0
    assert elapsed < 60.0, f"STILTS took {elapsed:.1f}s, expected < 60s"


# --------------------------------------------------------------------------- #
# skyerr matcher: n-sigma positional error criterion at scale
# --------------------------------------------------------------------------- #
_SKYERR_SIGMA = 0.5  # arcsec; per-side sigma = hypot(rae, dee)
_SKYERR_ERROR_AXIS = _SKYERR_SIGMA / np.sqrt(2)  # so hypot(axis, axis) = SKYERR_SIGMA
_SKYERR_BIN_SCALE_ENGINEER_ERR = 10.0  # arcsec per axis; inflates the max-sigma


@pytest.fixture(scope="module")
def skyerr_pair():
    """Two ~10k-row catalogues with ``rae``/``dee`` on both sides and 50
    injected pairs at sep = ~0.7 arcsec.

    One "bin-scale engineer" row sits at the end of the left catalogue with
    ``rae = dee = 10"``. Without it, STILTS computes its HEALPix binning
    scale from the per-row sigma maxima, and a tight ``max_error`` would drop
    injected candidates during coarse binning rather than rejecting them via
    the per-row n-sigma expression. The engineer row inflates the binning
    scale by an order of magnitude, so STILTS always inspects the candidates
    and the test genuinely exercises the per-row criterion at every
    ``max_error`` it tries.
    """
    rng = np.random.default_rng(43)
    base_ras = rng.uniform(60.0, 80.0, _INJECT)
    base_decs = rng.uniform(-30.0, -10.0, _INJECT)
    _SEP_ARCSEC = _INJECT_SEP_ARCSEC
    ra_off = (_SEP_ARCSEC / 3600.0) / np.cos(np.radians(base_decs))

    n_grid_left = _N_BASE - _INJECT - 1  # room for one engineer row
    n_grid_right = _N_BASE - _INJECT
    ra_l_grid, dec_l_grid = _grid_positions(n_grid_left, ra_phase=0.0)
    ra_r_grid, dec_r_grid = _grid_positions(n_grid_right, ra_phase=1.0)

    # Engineer row: far away from the injected zone and the grid, with a
    # deliberately large per-axis error so its sigma dominates max(left_sig,
    # right_sig) for the binning scale calculation.
    eng_ra, eng_dec = 100.0, -50.0

    left = pl.DataFrame(
        {
            "source_id": np.arange(_N_BASE, dtype=np.int64),
            "ra": np.concatenate([ra_l_grid, base_ras, [eng_ra]]).astype(np.float64),
            "dec": np.concatenate([dec_l_grid, base_decs, [eng_dec]]).astype(np.float64),
            "rae": np.concatenate(
                [
                    np.full(n_grid_left, _SKYERR_ERROR_AXIS),
                    np.full(_INJECT, _SKYERR_ERROR_AXIS),
                    [_SKYERR_BIN_SCALE_ENGINEER_ERR],
                ]
            ),
            "dee": np.concatenate(
                [
                    np.full(n_grid_left, _SKYERR_ERROR_AXIS),
                    np.full(_INJECT, _SKYERR_ERROR_AXIS),
                    [_SKYERR_BIN_SCALE_ENGINEER_ERR],
                ]
            ),
        }
    )
    right = pl.DataFrame(
        {
            "source_id": np.arange(_N_BASE, dtype=np.int64),
            "ra": np.concatenate([ra_r_grid, base_ras + ra_off]).astype(np.float64),
            "dec": np.concatenate([dec_r_grid, base_decs]).astype(np.float64),
            "rae": np.full(_N_BASE, _SKYERR_ERROR_AXIS),
            "dee": np.full(_N_BASE, _SKYERR_ERROR_AXIS),
        }
    )
    injected_ids = list(range(n_grid_left, n_grid_left + _INJECT))
    return left, right, injected_ids, _SEP_ARCSEC, _SKYERR_SIGMA


def _src_err(name: str = "cat") -> CatalogueSource:
    return CatalogueSource(
        name=name,
        is_local=True,
        ra_column="ra",
        dec_column="dec",
        ra_err_column="rae",
        dec_err_column="dee",
        pos_err_units="arcsec",
    )


def _stilts_skyerr(left: pl.DataFrame, right: pl.DataFrame, *, max_error: float) -> pl.DataFrame:
    spec = MatchSpec(radius_arcsec=1.0, matcher="skyerr", max_error=max_error)
    return stilts.stilts_sky_match(_src_err("L"), _src_err("R"), left, right, spec)


def test_stilts_skyerr_matches_within_n_sigma(skyerr_pair):
    """sep ``= 0.7"``, sigma ``= 0.5"`` each side, ``max_error = 3.0`` ⇒
    threshold ``= 3.0·(0.5+0.5) = 3.0 > 0.7``: all 50 injected pairs match."""
    left, right, injected_ids, _, _ = skyerr_pair
    out = _stilts_skyerr(left, right, max_error=3.0)
    assert out.height == _INJECT
    assert sorted(out["source_id"].to_list()) == sorted(injected_ids)
    right_injected_ids = list(range(_N_BASE - _INJECT, _N_BASE))
    assert sorted(out["source_id_2"].to_list()) == right_injected_ids


def test_stilts_skyerr_rejects_outside_n_sigma(skyerr_pair):
    """sep ``= 0.7"``, sigma ``= 0.5"`` each side, ``max_error = 0.5`` ⇒
    threshold ``= 0.5·(0.5+0.5) = 0.5 < 0.7``: zero matches.

    The bin-scale engineer guarantees STILTS actually inspects the
    candidates rather than binning them out via the spatial pre-filter
    (``params = max_error · (max_lsig + max_rsig) = 0.5 · (14.14 + 0.5) = 7.32 > 0.7``).
    """
    left, right, _, _, _ = skyerr_pair
    out = _stilts_skyerr(left, right, max_error=0.5)
    assert out.height == 0


def test_stilts_skyerr_separations_match_reference(skyerr_pair):
    """``sep_arcsec`` reproduces an independent great-circle reference to
    numerical precision."""
    left, right, _, _, _ = skyerr_pair
    out = _stilts_skyerr(left, right, max_error=3.0).sort("source_id")
    expected = _great_circle_arcsec(
        out["ra"].to_numpy(),
        out["dec"].to_numpy(),
        out["ra_2"].to_numpy(),
        out["dec_2"].to_numpy(),
    )
    assert np.allclose(out["sep_arcsec"].to_numpy(), expected, atol=1e-3)
    # Every reported separation sits inside the per-row n-sigma threshold.
    assert (out["sep_arcsec"].to_numpy() <= 3.0 * 2 * _SKYERR_SIGMA).all()


def test_stilts_skyerr_end_to_end_via_crossmatch_streams_to_parquet(
    skyerr_pair, tmp_path, monkeypatch
):
    """LazyFrame inputs → ``CrossMatch.crossmatch(lazy=True, matcher='skyerr',
    max_error=3.0)`` → ``sink_parquet(engine='streaming')```."""
    left, right, injected_ids, _, _ = skyerr_pair
    left_lf = left.lazy()
    right_lf = right.lazy()
    cm = CrossMatch()

    import xmatch.matchers as _matchers

    def _raise(*_a, **_kw):  # pragma: no cover - only on regression
        raise AssertionError("xmatch fell back to astropy; STILTS did not run")

    monkeypatch.setattr(_matchers, "_astropy_match", _raise)

    result_lf = cm.crossmatch(
        left_lf,
        right_lf,
        matcher="skyerr",
        max_error=3.0,
        engine="stilts",
        lazy=True,
    )
    assert result_lf.explain(streaming=True).strip()

    out_path = tmp_path / "skyerr.parquet"
    result_lf.sink_parquet(out_path, engine="streaming")

    result = pl.read_parquet(out_path)
    assert result.height == _INJECT
    assert sorted(result["source_id"].to_list()) == sorted(injected_ids)
    assert "sep_arcsec" in result.columns
    assert "source_id_2" in result.columns
