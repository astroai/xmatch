"""Tests for the deterministic xmatch benchmark contract."""

from scripts.benchmark_contract import (
    run_benchmark,
    run_matrix,
    synthetic_catalogues,
)


def test_synthetic_catalogues_are_deterministic() -> None:
    first = synthetic_catalogues(4, 8)
    second = synthetic_catalogues(4, 8)
    assert first[0].equals(second[0])
    assert first[1].equals(second[1])


def test_sparse_synthetic_catalogues_are_deterministic() -> None:
    first = synthetic_catalogues(4, 8, layout="sparse")
    second = synthetic_catalogues(4, 8, layout="sparse")
    assert first[0].equals(second[0])
    assert first[1].equals(second[1])


def test_benchmark_reports_global_id_parity() -> None:
    result = run_benchmark(4, 8, timed_runs=1)
    assert result["benchmark"] == "xmatch-global-id-v1"
    assert result["global_id_parity"] is True
    assert result["candidate_count"] >= 4
    assert result["peak_rss_bytes"] > 0


def test_engine_matrix_reports_dense_and_sparse_fast_parity() -> None:
    result = run_matrix(
        4,
        8,
        1,
        engines=("fast",),
        layouts=("dense", "sparse"),
        finds=("best", "all"),
    )

    assert result["benchmark"] == "xmatch-engine-matrix-v1"
    assert len(result["rows"]) == 4
    assert result["all_requested_engines_succeeded"] is True
    assert result["all_successful_engines_match_fast"] is True
    dense_all = next(
        row for row in result["rows"] if row["layout"] == "dense" and row["find"] == "all"
    )
    sparse_all = next(
        row for row in result["rows"] if row["layout"] == "sparse" and row["find"] == "all"
    )
    assert dense_all["candidate_count"] > sparse_all["candidate_count"]
