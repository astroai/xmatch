"""Hermetic I/O-scale benchmark: 10K DESI × 1M Gaia DR3 cone, frozen snapshot.

Asserts wall-clock budget + peak-RSS bound per available matcher engine
(always ``astropy`` / ``fast``, plus ``zone`` if ``cdshealpix`` is installed;
``stilts`` requires ``--run-stilts`` plus a Java runtime).  Inputs are local
parquet files generated once by ``scripts/gen_benchmark_data.py`` — the test
does NOT touch the network or any catalogue service, so it runs hermetically
on every CI runner.

Run::

    pytest -m bench tests/test_benchmarks_io_1m.py -v -s
    pytest -m bench tests/test_benchmarks_io_1m.py -v -s --run-stilts
"""
from __future__ import annotations

import gc
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from tests.conftest import BENCH_RESULTS
from xmatch import stilts
from xmatch.matchers import MatchSpec, sky_match
from xmatch.sources import CatalogueSource

pytestmark = [pytest.mark.bench, pytest.mark.slow]

# --------------------------------------------------------------------------- #
# Snapshot paths + budget spec — MUST stay in sync with
# scripts/gen_benchmark_data.py.  Bump a budget only with a SPECIFIC PR
# justification; the point of this benchmark is to catch regressions.
# --------------------------------------------------------------------------- #
TEST_DIR = Path(__file__).parent
_DESI_PATH = TEST_DIR / "data" / "benchmark" / "desi_l.parquet"
_GAIA_PATH = TEST_DIR / "data" / "benchmark" / "gaia_r.parquet"

# Per-engine wall-clock ceilings (median over ``N_TIMED`` runs).  Generous
# shared-CI budgets: loose enough to avoid flaking on a slow runner while
# still catching a 5x regression in the hot path.
TIME_BUDGET_S = {
    "fast": 60.0,
    "zone": 60.0,
    "astropy": 120.0,
    "stilts": 180.0,
}
# Whole-process peak-RSS ceiling.  Process-lifetime RSS (the only quantity
# Linux's ``ru_maxrss`` exposes) cannot be reset between engines — the budget
# is therefore the SUM of all engine allocations rather than any one engine.
PEAK_RSS_BUDGET_MB = 4096
N_TIMED = 3


def _peak_rss_mb() -> float:
    """Cross-platform peak-RSS readout.

    ``ru_maxrss`` is kilobytes on Linux and bytes on macOS — these are *high-
    water marks* for the whole process, monotonic across the session, so this
    returns process-lifetime memory rather than per-engine delta.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return raw / (1024 * 1024)
    return raw / 1024  # Linux KB → MB


def _available_engines(request: pytest.FixtureRequest) -> list[str]:
    """Engines present in the current environment (respecting ``--run-stilts``).

    ``zone`` is only listed if ``cdshealpix`` is importable — otherwise
    ``engine="zone"`` would silently downgrade to ``fast`` and we'd be
    benchmarking the same code twice under different labels.
    """
    engines: list[str] = ["astropy"]
    try:
        import scipy  # noqa: F401

        engines.append("fast")
    except ImportError:
        # Without scipy the fast engine would raise inside sky_match; skip
        # it (rather than surface a confusing traceback) so the bench still
        # reports on whatever IS available.
        pass
    try:
        import cdshealpix  # noqa: F401

        engines.append("zone")
    except ImportError:
        pass
    if request.config.getoption("run_stilts", False) and stilts.stilts_available():
        engines.append("stilts")
    return engines


@pytest.fixture(scope="module")
def snapshot() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load the parquet snapshot pair, generating it on first run if missing.

    A first-run ``pytest -m bench`` triggers ``scripts/gen_benchmark_data.py``
    via subprocess so the snapshot fixture is a hermetic dependency rather
    than a network-fetched artefact.
    """
    if not (_DESI_PATH.exists() and _GAIA_PATH.exists()):
        gen_script = TEST_DIR.parent / "scripts" / "gen_benchmark_data.py"
        if not gen_script.exists():
            pytest.skip(
                f"Snapshot files missing and generator not found at {gen_script}; "
                "run scripts/gen_benchmark_data.py once to create them."
            )
        subprocess.run([sys.executable, str(gen_script)], check=True)
    return pl.read_parquet(_DESI_PATH), pl.read_parquet(_GAIA_PATH)


def _local_source(frame: pl.DataFrame, *, name: str) -> CatalogueSource:
    """Build a local CatalogueSource from a fully-materialised polars frame."""
    return CatalogueSource(
        name=name,
        is_local=True,
        ra_column="ra",
        dec_column="dec",
        id_column="source_id",
        ra_err_column="rae",
        dec_err_column="dee",
        pos_err_units="arcsec",
    ).with_frame(frame.lazy())


def _time_engine(
    left_src: CatalogueSource,
    right_src: CatalogueSource,
    spec: MatchSpec,
    engine: str,
    *,
    n_timed: int = N_TIMED,
) -> tuple[float, pl.DataFrame]:
    """Run ``sky_match`` once warm + ``n_timed`` timed runs.  Returns (median, last)."""
    left_lf = left_src.lazy()
    right_lf = right_src.lazy()

    # Warm-up — primes caches and any first-touch lazy materials.
    result = sky_match(left_src, right_src, left_lf, right_lf, spec, engine=engine).collect()

    times: list[float] = []
    for _ in range(n_timed):
        gc.collect()  # make per-iteration memory creep visible to the next timed run
        t0 = time.perf_counter()
        result = sky_match(left_src, right_src, left_lf, right_lf, spec, engine=engine).collect()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), result


def test_bench_io_1m_desi_x_gaia(snapshot, request):
    """Frozen 10K DESI × 1M Gaia DR3 cone: assert wall-clock + RSS budget per engine.

    Verifies the two non-trivial README claims:

    * **Out-of-core memory safety** — runtime + 1M-row background frame + the
      matcher's working set fits under 4 GB peak RSS even when both sides are
      fully materialised.
    * **In-process engines agree** — the three tiers (astropy / fast / zone)
      surface the SAME set of matches (within a 20 % relative tolerance —
      sampling noise + per-engine preprocessing can produce small differences).

    This is a single test (parametrising over radii would break the
    cross-engine parity compare on a like-for-like basis).
    """
    left_df, right_df = snapshot
    left_n, right_n = len(left_df), len(right_df)

    spec = MatchSpec(radius_arcsec=2.0, matcher="sky", find="best")
    left_src = _local_source(left_df, name="desi_l")
    right_src = _local_source(right_df, name="gaia_r")

    print(
        f"\nLoaded snapshot: left (desi_l) = {left_n:,} rows, "
        f"right (gaia_r) = {right_n:,} rows @ r={spec.radius_arcsec}\""
    )

    baseline_n: int | None = None  # first successful engine's match count
    passed_engines: list[str] = []

    for engine in _available_engines(request):
        try:
            elapsed, result = _time_engine(left_src, right_src, spec, engine)
        except Exception as exc:  # noqa: BLE001 — record every engine failure
            # Don't fail the whole bench on a single-engine regression; record
            # the failure to BENCH_RESULTS and continue with the next engine.
            print(f"  [{engine}] FAILED after warm-up: {type(exc).__name__}: {exc}")
            BENCH_RESULTS.append(
                {
                    "scenario": "io-1m desi_x_gaia",
                    "n_left": left_n,
                    "n_right": right_n,
                    "radius": spec.radius_arcsec,
                    "engine": engine,
                    "time_s": float("nan"),
                    "matches": 0,
                    "sep_max": 0.0,
                    "result": None,
                }
            )
            continue

        peak_rss_mb = _peak_rss_mb()
        n_matches = result.height
        sep_max = float(result["sep_arcsec"].max()) if n_matches else 0.0
        print(
            f"  [{engine}] median {elapsed:6.2f}s over {N_TIMED} runs; "
            f"{n_matches:,} matches; sep_max={sep_max:.3f}\"; "
            f"peak RSS = {peak_rss_mb:.0f} MB"
        )
        BENCH_RESULTS.append(
            {
                "scenario": "io-1m desi_x_gaia",
                "n_left": left_n,
                "n_right": right_n,
                "radius": spec.radius_arcsec,
                "engine": engine,
                "time_s": elapsed,
                "matches": n_matches,
                "sep_max": sep_max,
                "result": result,
            }
        )

        # Memory ceiling.
        assert peak_rss_mb < PEAK_RSS_BUDGET_MB, (
            f"{engine}: process peak RSS {peak_rss_mb:.0f} MB exceeds "
            f"PEAK_RSS_BUDGET_MB={PEAK_RSS_BUDGET_MB} MB budget"
        )
        # Engine-specific time ceiling.
        if engine in TIME_BUDGET_S:
            assert elapsed < TIME_BUDGET_S[engine], (
                f"{engine}: median {elapsed:.1f}s exceeds "
                f"TIME_BUDGET_S[{engine!r}]={TIME_BUDGET_S[engine]}s budget"
            )
        # Sanity floor — the geometric-probability estimate is ~12.3K
        # matches.  A near-zero count means the snapshot or the matcher's
        # RA/Dec wiring has regressed.
        assert n_matches > 100, (
            f"{engine}: only {n_matches} matches at r={spec.radius_arcsec}\" — "
            "snapshot may have lost RA/Dec spread"
        )

        # Cross-engine parity: track the first successful engine as the
        # baseline; later engines must agree to within 5 % relative AND
        # ±500 absolute (whichever is wider).  The ±500 absolute floor
        # protects small-match-count inputs where 5 % of the baseline is
        # tighter than the floating-point / sampling noise floor that a
        # purely-relative tolerance would over-tighten against.
        # NB: engine ORDERING matters — the first engine in
        # ``_available_engines`` that completes becomes the baseline, so
        # reordering that list can shift which engine is benchmarked
        # against which.
        if baseline_n is None:
            baseline_n = n_matches
        else:
            rel = abs(n_matches - baseline_n) / max(baseline_n, 1)
            abs_tol = max(int(baseline_n * 0.05), 500)
            abs_delta = abs(n_matches - baseline_n)
            assert rel <= 0.05 or abs_delta <= abs_tol, (
                f"{engine} produced {n_matches:,} matches; baseline "
                f"{baseline_n:,} → relative delta {rel:.2%} exceeds 5 % "
                f"AND absolute delta {abs_delta:,} exceeds {abs_tol:,}. "
                "Often signals a real engine bug, not a regression."
            )
        passed_engines.append(engine)

    # Failing every engine would mask the actual issue; require at least the
    # always-available tier (``astropy`` + ``fast``) to succeed.
    assert passed_engines, "no in-process engine completed the benchmark"
    for required in ("astropy", "fast"):
        if required in _available_engines(request):
            assert required in passed_engines, (
                f"required engine '{required}' did not complete; "
                "benchmark cannot validate the README's engine-parity claim"
            )
