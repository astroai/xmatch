"""Streaming tests for the pure polars ``id_join`` machinery in xmatcher.

The STILTS-backed spatial tests in :mod:`tests.test_large_crossmatch` cover
streaming end-to-end on a 10k-row scale, but they exercise the spatial
matcher (which shells out to STILTS via a FITS round-trip). This file
exercises the parallel polars-only path — a relational id-join executed
entirely inside polars — so it doesn't need Java or STILTS to run.

These tests are deliberately **not** marked ``slow`` and not gated on STILTS
availability: they exist specifically to catch streaming regressions on
every CI invocation, even on small runners.
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from xmatcher import CrossMatch

_N_BASE = 10_000  # rows per parquet catalogue
_INJECT = 50  # deliberate id overlap (left's tail == right's head)


@pytest.fixture
def parquet_catalogues(tmp_path: Path) -> tuple[Path, Path]:
    """Two 10k-row parquet files written to ``tmp_path`` with 50 deliberately
    overlapping ``oid`` values.

    The overlap range is ``[_N_BASE - _INJECT, _N_BASE)``: left carries
    ``oid`` values 0..10_000-1 and right carries 9_950..19_949, so an inner
    id-join produces exactly ``_INJECT = 50`` matched rows.

    Both catalogues use identical column names (``oid``, ``ra``, ``dec``,
    ``mag``) on purpose: the names need to be ``ra``/``dec`` so xmatcher's
    local-source resolver can satisfy its (rather eager) RA/Dec-detection
    step on both sides, and naming collisions on the right catalogue are
    exactly what we want — they exercise polars's suffix-on-collision
    rename to ``_2`` (this project's ``matchers._RIGHT_SUFFIX``).
    """
    rng = np.random.default_rng(101)
    left = pl.DataFrame(
        {
            "oid": np.arange(_N_BASE, dtype=np.int64),
            "ra": rng.uniform(0.0, 360.0, _N_BASE).astype(np.float64),
            "dec": rng.uniform(-30.0, 30.0, _N_BASE).astype(np.float64),
            "mag": rng.uniform(15.0, 22.0, _N_BASE).astype(np.float64),
        }
    )
    right = pl.DataFrame(
        {
            "oid": np.arange(_N_BASE - _INJECT, 2 * _N_BASE - _INJECT, dtype=np.int64),
            "ra": rng.uniform(0.0, 360.0, _N_BASE).astype(np.float64),
            "dec": rng.uniform(-30.0, 30.0, _N_BASE).astype(np.float64),
            "mag": rng.uniform(15.0, 22.0, _N_BASE).astype(np.float64),
        }
    )
    a_path = tmp_path / "left.parquet"
    b_path = tmp_path / "right.parquet"
    left.write_parquet(a_path)
    right.write_parquet(b_path)
    return a_path, b_path


def test_id_join_lazy_streams_to_parquet(parquet_catalogues, tmp_path):
    """parquet scan → ``cm.crossmatch(id_join=True, lazy=True)`` →
    ``sink_parquet(engine='streaming')``.

    Same recipe as the STILTS streaming test, but with the polars-only
    ``id_join`` path so it can run on every CI runner without Java + STILTS.
    Verifies that the streaming executor handles a pure polars pipeline
    (scan + join + projection) and that the result contains exactly the 50
    pre-planned matched ids.
    """
    a_path, b_path = parquet_catalogues

    cm = CrossMatch()
    result_lf = cm.crossmatch(
        a_path,
        b_path,
        id_join=True,
        id_column_1="oid",
        id_column_2="oid",
        join_type="1and2",
        lazy=True,
    )
    assert isinstance(result_lf, pl.LazyFrame)

    # Streaming plan must be producible — polars raises ComputeError on plans
    # that cannot be streamed, so a successful plan implies the result is
    # streamable end-to-end.
    plan = result_lf.explain(engine="streaming")
    assert plan.strip()

    # Sink via the streaming executor; the result is never materialised.
    out = tmp_path / "result.parquet"
    result_lf.sink_parquet(out, engine="streaming")
    assert out.exists() and out.stat().st_size > 0

    result = pl.read_parquet(out)
    assert result.height == _INJECT
    # polars merges the join-key column (``oid`` is on both sides) into a
    # single column; non-key collisions get the ``_2`` suffix on the right.
    # Asserting on a non-key column (``ra_2``) verifies the suffix-on-collision
    # rename contract end-to-end.
    assert "oid" in result.columns
    assert "ra_2" in result.columns
    # Every matched id must come from the deliberate overlap range.
    expected = sorted(range(_N_BASE - _INJECT, _N_BASE))
    assert sorted(result["oid"].to_list()) == expected


def test_id_join_lazy_outer_join_streams_to_parquet(parquet_catalogues, tmp_path):
    """Outer id-join streamed via the same polars streaming engine.

    Exercises the streaming writer against a non-inner join (``all1`` keeps
    every left row, paired with right data when an ``oid`` matches). Catches
    regressions where streaming fails specifically on ``how="left"`` or full
    outer joins — the diagonal_relaxed style used in spatial matches is not
    used here, so this is a complementary streaming path.
    """
    a_path, b_path = parquet_catalogues

    cm = CrossMatch()
    result_lf = cm.crossmatch(
        a_path,
        b_path,
        id_join=True,
        id_column_1="oid",
        id_column_2="oid",
        join_type="all1",
        lazy=True,
    )
    assert result_lf.explain(engine="streaming").strip()

    out = tmp_path / "all1.parquet"
    result_lf.sink_parquet(out, engine="streaming")

    result = pl.read_parquet(out)
    # ``all1`` must surface every left id (matched or unmatched). Use the
    # non-key right column ``ra_2`` as the match indicator (null iff unmapped
    # left row).
    assert result["oid"].n_unique() == _N_BASE
    assert result["ra_2"].is_null().sum() == _N_BASE - _INJECT
    assert result["ra_2"].is_not_null().sum() == _INJECT
    # And the 50 matched ids must come from the deliberate overlap range.
    matched = result.filter(result["ra_2"].is_not_null())
    assert sorted(matched["oid"].to_list()) == sorted(range(_N_BASE - _INJECT, _N_BASE))
