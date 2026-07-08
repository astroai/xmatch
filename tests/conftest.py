"""Shared pytest fixtures and hooks for the xmatch test suite."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pytest

from .test_real_catalogues import allwise_csv, gaia_csv, usno_csv  # noqa: F401

# Global accumulator for benchmark results, populated by test_benchmarks.py.
BENCH_RESULTS: List[Dict] = []


# --------------------------------------------------------------------------- #
# Custom CLI options
# --------------------------------------------------------------------------- #
def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-stilts",
        action="store_true",
        default=False,
        help="Include STILTS engine in benchmark comparisons.",
    )


# --------------------------------------------------------------------------- #
# Pretty-print benchmark table at session end
# --------------------------------------------------------------------------- #
def _aggregate(results: List[Dict]) -> Dict:
    groups: Dict = {}
    for r in results:
        key = (r["scenario"], r["radius"])
        groups.setdefault(key, []).append(r)
    return groups


def _sep_max_deviation(results: List[Dict]) -> str:
    baseline_seps = None
    for r in results:
        if r["engine"] == "astropy" and r.get("result") is not None:
            baseline_seps = r["result"]["sep_arcsec"].to_numpy()
            break
    if baseline_seps is None:
        return ""

    parts = []
    for r in results:
        if r["engine"] == "astropy" or r.get("result") is None or not r["matches"]:
            continue
        engine_seps = r["result"]["sep_arcsec"].to_numpy()
        if len(engine_seps) != len(baseline_seps):
            parts.append(f"{r['engine']}:count-diff")
            continue
        diff_arcsec = np.abs(engine_seps - baseline_seps)
        max_uas = float(diff_arcsec.max()) * 1e6
        parts.append(f"{r['engine']}:{max_uas:.1f}µas")
    return " | ".join(parts) if parts else ""


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if not BENCH_RESULTS:
        return

    groups = _aggregate(BENCH_RESULTS)
    astropy_times: Dict = {}
    for key, entries in groups.items():
        for e in entries:
            if e["engine"] == "astropy":
                astropy_times[key] = e["time_s"]
                break

    sep = "=" * 120
    print("\n" + sep)
    print("BENCHMARK RESULTS — xmatch matcher engine comparison")
    print(sep)
    header = (
        f"{'Scenario':<22} {'N₁':>6} {'N₂':>6} {'r(″)':>5} {'Engine':>10} "
        f"{'Time(s)':>8} {'SpdUp':>6} {'Matches':>8} {'sep_max(″)':>10}"
    )
    print(header)
    print("-" * 120)

    for key in sorted(groups):
        entries = groups[key]
        scenario, radius = key
        astro_time = astropy_times.get(key)
        for e in sorted(entries, key=lambda x: x["engine"]):
            if astro_time and astro_time > 0:
                speedup = f"{astro_time / e['time_s']:.1f}x"
            else:
                speedup = "—"
            sep_max_str = f"{e['sep_max']:.3f}" if e["sep_max"] else "—"
            n_left = e.get("n_left", "?")
            n_right = e.get("n_right", "?")
            print(
                f"{scenario:<22} {str(n_left):>6} {str(n_right):>6} "
                f"{radius:>5.1f} {e['engine']:>10} "
                f"{e['time_s']:>8.4f} {speedup:>6} {e['matches']:>8} {sep_max_str:>10}"
            )

    print()
    print("Separation agreement (max |sep - sep_astropy| vs astropy baseline):")
    for key in sorted(groups):
        scenario, radius = key
        dev = _sep_max_deviation(groups[key])
        if dev:
            print(f"  {scenario:<22} r={radius:.1f}\u2033: {dev}")

    print(sep)
    print(f"Total benchmarks: {len(BENCH_RESULTS)}")
    print("Run with: pytest -m bench tests/test_benchmarks.py -v -s")
    print("Include STILTS: pytest -m bench tests/test_benchmarks.py -v -s --run-stilts")
    print(sep + "\n")
