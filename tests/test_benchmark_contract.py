"""Tests for the deterministic xmatch benchmark contract."""

from scripts.benchmark_contract import run_benchmark, synthetic_catalogues


def test_synthetic_catalogues_are_deterministic() -> None:
    first = synthetic_catalogues(4, 8)
    second = synthetic_catalogues(4, 8)
    assert first[0].equals(second[0])
    assert first[1].equals(second[1])


def test_benchmark_reports_global_id_parity() -> None:
    result = run_benchmark(4, 8, timed_runs=1)
    assert result["benchmark"] == "xmatch-global-id-v1"
    assert result["global_id_parity"] is True
    assert result["candidate_count"] >= 4
    assert result["peak_rss_bytes"] > 0
