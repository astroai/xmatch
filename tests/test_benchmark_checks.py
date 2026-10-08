"""Offline checks that benchmark validation detects incorrect results."""

import runpy
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


def test_scaling_summary_rejects_under_provisioned_worker_count():
    """A configured 2-worker point is invalid if only one worker executed."""
    script = Path(__file__).resolve().parents[1] / "scripts/analyse_canfar_scaling.py"
    summarise = runpy.run_path(str(script))["summarise"]
    rows = [
        {
            "rows_per_input": "100",
            "configured_workers": str(workers),
            "iteration": str(iteration),
            "head_task_count": "0",
            "failed_attempts": "0",
            "actual_worker_nodes": str(1 if workers == 2 else workers),
            "planned_tasks": "10",
            "finished_tasks": "10",
            "output_rows": "150",
            "output_partitions": "3",
            "union_seconds": "2.0",
            "planner_seconds": "0.1",
            "assembly_seconds": "0.1",
            "task_span_seconds": "1.0",
        }
        for workers in (1, 2)
        for iteration in range(4)
    ]

    with pytest.raises(ValueError):
        summarise(rows)
