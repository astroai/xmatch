"""Real-catalogue benchmark suite comparing xmatch matcher engines.

Runs the three xmatch in-process tiers (astropy, fast/cKDTree, zone/HEALPix)
plus STILTS (when ``--run-stilts`` is passed) against real astronomical
catalogues.  Measures wall-clock time, match counts, and separation agreement
between engines.

Benchmarks are opt-in (``pytest -m bench``) and gated on network + astroquery.
Results are printed as a compact terminal table at the end of the suite.

=== Quick start ===

    pytest -m bench tests/test_benchmarks.py -v -s
    pytest -m bench tests/test_benchmarks.py -v -s --run-stilts
    pytest -m bench tests/test_benchmarks.py -k "gaia_self" --durations=0
"""

from __future__ import annotations

import time

import numpy as np
import polars as pl
import pytest

from tests.conftest import BENCH_RESULTS  # shared accumulator
from xmatch import stilts
from xmatch.matchers import MatchSpec, sky_match
from xmatch.sources import CatalogueSource

# --------------------------------------------------------------------------- #
# Imports from the real-catalogue test module (tests/ is a package).
# --------------------------------------------------------------------------- #
from .test_real_catalogues import (  # noqa: E402
    _HAVE_ASTROQUERY,
    _allwise_source,
    _gaia_source,
    _usno_source,
)

# --------------------------------------------------------------------------- #
# Marker: all benchmarks are bench + slow.
# --------------------------------------------------------------------------- #
pytestmark = [pytest.mark.bench, pytest.mark.slow]

# --------------------------------------------------------------------------- #
# Engine registry (built on first access to avoid import errors at collect time)
# --------------------------------------------------------------------------- #
_known_engines: list[str] = []


def _available_engines() -> list[str]:
    """Return engine names available in the current environment."""
    if _known_engines:
        return _known_engines
    engines: list[str] = ["astropy"]  # always available
    try:
        import scipy  # noqa: F401

        engines.append("fast")
        engines.append("zone")  # falls back to fast if cdshealpix missing
    except ImportError:
        pass
    _known_engines[:] = engines
    return _known_engines


def _resolve_engines(request: pytest.FixtureRequest) -> list[str]:
    """Resolve which engines to test, respecting --run-stilts flag."""
    engines = list(_available_engines())
    if request.config.getoption("run_stilts", False) and stilts.stilts_available():
        engines.append("stilts")
    return engines


# --------------------------------------------------------------------------- #
# Benchmark helpers
# --------------------------------------------------------------------------- #
def _time_match(
    left_src: CatalogueSource,
    right_src: CatalogueSource,
    spec: MatchSpec,
    engine: str,
    *,
    n_warmup: int = 1,
    n_timed: int = 3,
) -> tuple[float, pl.DataFrame, int, int]:
    """Run ``sky_match`` *n_timed* times after *n_warmup* warm-up calls.

    Returns ``(median_seconds, result_frame, n_left_rows, n_right_rows)``.
    """
    # Collect source frames once so _record doesn't re-materialise.
    left = left_src.lazy().collect()
    right = right_src.lazy().collect()
    left_n, right_n = left.height, right.height

    # Re-wrap as lazy for sky_match (it calls .collect() internally).
    left_lf, right_lf = left.lazy(), right.lazy()

    result = None
    for _ in range(n_warmup):
        result = sky_match(left_src, right_src, left_lf, right_lf, spec, engine=engine).collect()

    times = []
    for _ in range(n_timed):
        t0 = time.perf_counter()
        result = sky_match(left_src, right_src, left_lf, right_lf, spec, engine=engine).collect()
        times.append(time.perf_counter() - t0)

    return float(np.median(times)), result, left_n, right_n  # type: ignore[return-value]


def _raw_astropy_match(
    left_src: CatalogueSource,
    right_src: CatalogueSource,
    spec: MatchSpec,
    *,
    n_warmup: int = 1,
    n_timed: int = 3,
) -> tuple[float, int]:
    """Raw astropy ``match_to_catalog_sky`` benchmark (no polars overhead)."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    left = left_src.lazy().collect()
    right = right_src.lazy().collect()
    lcoord = SkyCoord(
        left[left_src.ra_column].to_numpy() * u.deg,
        left[left_src.dec_column].to_numpy() * u.deg,
    )
    rcoord = SkyCoord(
        right[right_src.ra_column].to_numpy() * u.deg,
        right[right_src.dec_column].to_numpy() * u.deg,
    )
    spec.radius_arcsec * u.arcsec

    for _ in range(n_warmup):
        idx, sep2d, _ = lcoord.match_to_catalog_sky(rcoord)

    times = []
    for _ in range(n_timed):
        t0 = time.perf_counter()
        idx, sep2d, _ = lcoord.match_to_catalog_sky(rcoord)
        times.append(time.perf_counter() - t0)

    n_matches = int((sep2d.arcsec <= spec.radius_arcsec).sum())
    return float(np.median(times)), n_matches


# NOTE: BENCH_RESULTS is imported from tests.conftest as session-scoped shared state.
# It is safe under sequential pytest execution (the default). Forked/multi-process
# runners (pytest-xdist, --forked) will each have their own copy; benchmark results
# will only be printed for the master process.


def _record(
    scenario: str,
    left_src: CatalogueSource,
    right_src: CatalogueSource,
    engine: str,
    elapsed: float,
    result: pl.DataFrame,
    left_n: int,
    right_n: int,
    spec: MatchSpec,
) -> None:
    n = result.height
    sep_max = float(result["sep_arcsec"].max()) if n else 0.0
    BENCH_RESULTS.append(
        {
            "scenario": scenario,
            "n_left": left_n,
            "n_right": right_n,
            "radius": spec.radius_arcsec,
            "engine": engine,
            "time_s": elapsed,
            "matches": n,
            "sep_max": sep_max,
            "result": result,
        }
    )


def _record_raw(
    scenario: str,
    spec: MatchSpec,
    engine: str,
    elapsed: float,
    n_matches: int,
) -> None:
    BENCH_RESULTS.append(
        {
            "scenario": scenario,
            "n_left": 0,
            "n_right": 0,
            "radius": spec.radius_arcsec,
            "engine": engine,
            "time_s": elapsed,
            "matches": n_matches,
            "sep_max": 0.0,
            "result": None,
        }
    )


# --------------------------------------------------------------------------- #
# Benchmarks
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAVE_ASTROQUERY, reason="astroquery not installed")
@pytest.mark.parametrize("radius_arcsec,find", [(0.5, "best"), (1.5, "best"), (3.0, "best")])
def test_bench_gaia_self(radius_arcsec: float, find: str, gaia_csv, request):
    """Gaia DR3 self-match: all rows self-pair at sep≈0."""
    src = _gaia_source(gaia_csv)
    spec = MatchSpec(radius_arcsec=radius_arcsec, find=find)

    for engine in _resolve_engines(request):
        elapsed, result, left_n, right_n = _time_match(src, src, spec, engine)
        _record("Gaia self", src, src, engine, elapsed, result, left_n, right_n, spec)

    try:
        elapsed, n = _raw_astropy_match(src, src, spec)
        _record_raw("Gaia self", spec, "astropy-raw", elapsed, n)
    except Exception:
        pass


@pytest.mark.skipif(not _HAVE_ASTROQUERY, reason="astroquery not installed")
@pytest.mark.parametrize("radius_arcsec", [1.0, 3.0, 5.0])
def test_bench_gaia_x_allwise(radius_arcsec: float, gaia_csv, allwise_csv, request):
    """Gaia × AllWISE cross-catalogue: optical × mid-IR on the same sky patch."""
    left_src = _gaia_source(gaia_csv)
    right_src = _allwise_source(allwise_csv)
    spec = MatchSpec(radius_arcsec=radius_arcsec, find="best")

    for engine in _resolve_engines(request):
        elapsed, result, left_n, right_n = _time_match(left_src, right_src, spec, engine)
        _record(
            "Gaia x AllWISE",
            left_src,
            right_src,
            engine,
            elapsed,
            result,
            left_n,
            right_n,
            spec,
        )

    try:
        elapsed, n = _raw_astropy_match(left_src, right_src, spec)
        _record_raw("Gaia x AllWISE", spec, "astropy-raw", elapsed, n)
    except Exception:
        pass


@pytest.mark.skipif(not _HAVE_ASTROQUERY, reason="astroquery not installed")
def test_bench_gaia_x_usno(gaia_csv, usno_csv, request):
    """Gaia × USNO-B1.0: optical catalogues with proper-motion columns."""
    left_src = _gaia_source(gaia_csv)
    right_src = _usno_source(usno_csv)
    spec = MatchSpec(radius_arcsec=2.0, find="best")

    for engine in _resolve_engines(request):
        elapsed, result, left_n, right_n = _time_match(left_src, right_src, spec, engine)
        _record(
            "Gaia x USNO-B", left_src, right_src, engine, elapsed, result, left_n, right_n, spec
        )

    try:
        elapsed, n = _raw_astropy_match(left_src, right_src, spec)
        _record_raw("Gaia x USNO-B", spec, "astropy-raw", elapsed, n)
    except Exception:
        pass


@pytest.mark.skipif(not _HAVE_ASTROQUERY, reason="astroquery not installed")
def test_bench_allwise_self(allwise_csv, request):
    """AllWISE self-match: mid-IR catalogue, string IDs."""
    src = _allwise_source(allwise_csv)
    spec = MatchSpec(radius_arcsec=0.5, find="best")

    for engine in _resolve_engines(request):
        elapsed, result, left_n, right_n = _time_match(src, src, spec, engine)
        _record("AllWISE self", src, src, engine, elapsed, result, left_n, right_n, spec)

    try:
        elapsed, n = _raw_astropy_match(src, src, spec)
        _record_raw("AllWISE self", spec, "astropy-raw", elapsed, n)
    except Exception:
        pass
