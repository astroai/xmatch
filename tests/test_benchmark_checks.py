"""Offline checks that benchmark validation detects incorrect results."""

import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

from tests.test_benchmarks import _assert_match_agreement


def test_benchmark_agreement_ignores_result_order():
    reference = pl.DataFrame({"_bench_left_row": [0, 1], "sep_arcsec": [0.1, 0.2]})
    _assert_match_agreement(reference, reference.reverse())


@pytest.mark.parametrize(
    "wrong",
    [
        {"_bench_left_row": [0, 2], "sep_arcsec": [0.1, 0.2]},
        {"_bench_left_row": [0, 1], "sep_arcsec": [0.1, 0.3]},
        {"_bench_left_row": [0, 0], "sep_arcsec": [0.1, 0.2]},
    ],
)
def test_benchmark_agreement_rejects_wrong_matches(wrong):
    reference = pl.DataFrame({"_bench_left_row": [0, 1], "sep_arcsec": [0.1, 0.2]})
    with pytest.raises(AssertionError):
        _assert_match_agreement(reference, pl.DataFrame(wrong))


def test_union_benchmark_refuses_disabled_checks():
    script = Path(__file__).resolve().parents[1] / "scripts/bench_ray_union.py"
    result = subprocess.run(
        [sys.executable, "-O", str(script), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode != 0
    assert "requires enabled assertions" in result.stderr
