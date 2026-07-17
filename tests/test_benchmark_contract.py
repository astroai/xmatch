"""Tests for the deterministic xmatch benchmark contract."""

from copy import deepcopy

from scripts.benchmark_contract import (
    evaluate_promotion,
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
    assert result["timed_runs"] == 1


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


def _promotion_matrix() -> dict:
    rows = []
    for layout in ("dense", "sparse"):
        for find in ("best", "all"):
            for engine, wall, rss in (("fast", 1.0, 100), ("torchsky", 1.1, 120)):
                rows.append(
                    {
                        "engine": engine,
                        "layout": layout,
                        "find": find,
                        "status": "ok",
                        "left_rows": 10_000,
                        "right_rows": 100_000,
                        "timed_runs": 3,
                        "global_id_parity": True,
                        "pair_parity_with_fast": True,
                        "wall_seconds_median": wall,
                        "peak_rss_bytes": rss,
                    }
                )
    return {"benchmark": "xmatch-engine-matrix-v1", "rows": rows}


def test_promotion_requires_parity_scale_and_resource_thresholds() -> None:
    decision = evaluate_promotion(_promotion_matrix(), "torchsky")

    assert decision["policy"] == "xmatch-engine-promotion-v1"
    assert decision["eligible_for_automatic_selection"] is True
    assert len(decision["cases"]) == 4
    assert decision["reasons"] == []


def test_promotion_reports_every_failed_gate() -> None:
    matrix = deepcopy(_promotion_matrix())
    candidate = next(row for row in matrix["rows"] if row["engine"] == "torchsky")
    candidate["left_rows"] = 32
    candidate["pair_parity_with_fast"] = False
    candidate["wall_seconds_median"] = 2.0
    candidate["peak_rss_bytes"] = 200

    decision = evaluate_promotion(matrix, "torchsky")

    assert decision["eligible_for_automatic_selection"] is False
    assert any("left_rows" in reason for reason in decision["reasons"])
    assert any("pair hash" in reason for reason in decision["reasons"])
    assert any("wall ratio" in reason for reason in decision["reasons"])
    assert any("RSS ratio" in reason for reason in decision["reasons"])
