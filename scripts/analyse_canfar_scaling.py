#!/usr/bin/env python
"""Plot a correctness-checked CANFAR sweep from its curated repeat-level CSV.

Requires Matplotlib only when plotting; use an existing plotting environment.
Each successful point must contain warmup iteration 0 and measured iterations
1, 2 and 3. Task span is occupied wall time, not measured CPU utilization.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import median


def summarise(
    rows: list[dict[str, str]],
) -> dict[tuple[int, int], dict[str, tuple[float, float, float]]]:
    groups: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    workload_shapes: dict[int, tuple[int, int]] = {}
    metrics = ("union_seconds", "planner_seconds", "assembly_seconds", "task_span_seconds")
    for row in rows:
        key = (int(row["rows_per_input"]), int(row["configured_workers"]))
        if min(key) < 1 or int(row["head_task_count"]) != 0:
            raise ValueError("Positive input/worker counts and a task-free head are required")
        if int(row["failed_attempts"]) != 0:
            raise ValueError("Failed task attempts require separate analysis")
        if int(row["actual_worker_nodes"]) != key[1]:
            raise ValueError("Executing worker count must agree with the configured cluster")
        if not 0 < int(row["planned_tasks"]) == int(row["finished_tasks"]):
            raise ValueError("Every planned task must have a recorded successful completion")
        if int(row["output_rows"]) != 2 * key[0] - key[0] // 2:
            raise ValueError("Output row count disagrees with the benchmark's exact-ID oracle")
        if int(row["output_partitions"]) < 1:
            raise ValueError("Missing output partitions")
        shape = (int(row["planned_tasks"]), int(row["output_partitions"]))
        if workload_shapes.setdefault(key[0], shape) != shape:
            raise ValueError("Task and output partition counts changed within one input size")
        values = {metric: float(row[metric]) for metric in metrics}
        if not all(math.isfinite(value) and value >= 0 for value in values.values()):
            raise ValueError("Stage times must be finite and nonnegative")
        if values["union_seconds"] <= 0:
            raise ValueError("Union elapsed time must be positive")
        if (
            values["planner_seconds"] + values["assembly_seconds"] > values["union_seconds"] + 0.01
            or values["task_span_seconds"] > values["union_seconds"] + 0.01
        ):
            raise ValueError("Stage duration exceeds its enclosing union duration")
        groups[key].append(row)
    result = {}
    for key, group in groups.items():
        if sorted(int(row["iteration"]) for row in group) != [0, 1, 2, 3]:
            raise ValueError(f"{key}: expected one warmup and three unique measured iterations")
        timed = [row for row in group if int(row["iteration"]) > 0]
        point = {}
        for metric in (*metrics, "serial_seconds"):
            samples = [
                float(row["planner_seconds"]) + float(row["assembly_seconds"])
                if metric == "serial_seconds"
                else float(row[metric])
                for row in timed
            ]
            point[metric] = (median(samples), min(samples), max(samples))
        result[key] = point
    if not result or any((size, 1) not in result for size, _ in result):
        raise ValueError("Every input size requires a successful one-worker baseline")
    return result


def self_check() -> None:
    rows = [
        {
            "rows_per_input": "100",
            "configured_workers": str(workers),
            "iteration": str(iteration),
            "head_task_count": "0",
            "failed_attempts": "0",
            "actual_worker_nodes": str(workers),
            "planned_tasks": "10",
            "finished_tasks": "10",
            "output_rows": "150",
            "output_partitions": "3",
            "union_seconds": str(seconds / workers),
            "planner_seconds": "0.1",
            "assembly_seconds": "0.2",
            "task_span_seconds": str(seconds / workers - 0.4),
        }
        for workers in (1, 2)
        for iteration, seconds in enumerate((1000, 9, 10, 11))
    ]
    summary = summarise(rows)
    if summary[100, 1]["union_seconds"] != (10.0, 9.0, 11.0) or summary[100, 2][
        "union_seconds"
    ] != (5.0, 4.5, 5.5):
        raise AssertionError("Warmup exclusion or median/range aggregation failed")
    for broken in (
        rows + [rows[0]],
        [{**row, "union_seconds": "nan"} for row in rows],
        [{**row, "head_task_count": "1"} for row in rows],
        [{**row, "assembly_seconds": "10000"} for row in rows],
        [{**row, "finished_tasks": "9"} for row in rows],
        [{**row, "output_rows": "149"} for row in rows],
        [{**row, "output_partitions": "4"} if i == 0 else row for i, row in enumerate(rows)],
        [
            {**row, "actual_worker_nodes": "1"} if row["configured_workers"] == "2" else row
            for row in rows
        ],
    ):
        try:
            summarise(broken)
        except ValueError:
            continue
        raise AssertionError("Invalid scaling evidence was accepted")
    print("Scaling analysis self-check passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, nargs="?")
    parser.add_argument("--plot", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
    if args.csv is None:
        if not args.self_check:
            parser.error("provide a CSV or --self-check")
        return
    with args.csv.open(newline="") as stream:
        summary = summarise(list(csv.DictReader(stream)))
    print("| Rows/input | Workers | Median union s | Range s | Speedup | Efficiency | Serial s |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for (size, workers), point in sorted(summary.items()):
        center, low, high = point["union_seconds"]
        speedup = summary[size, 1]["union_seconds"][0] / center
        print(
            f"| {size:,} | {workers} | {center:.3f} | {low:.3f}–{high:.3f} | "
            f"{speedup:.2f}× | {speedup / workers:.1%} | {point['serial_seconds'][0]:.3f} |"
        )
    if args.plot is None:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = sorted({size for size, _ in summary})
    fig, axes = plt.subplots(3, len(sizes), figsize=(6 * len(sizes), 10), squeeze=False)
    for column, size in enumerate(sizes):
        worker_counts = sorted(count for rows, count in summary if rows == size)
        for metric, label, style in (
            ("union_seconds", "Total union", "o-"),
            ("task_span_seconds", "First–last task span", "s--"),
            ("serial_seconds", "Planner + assembly", "^-"),
        ):
            values = [summary[size, count][metric] for count in worker_counts]
            axes[0, column].errorbar(
                worker_counts,
                [value[0] for value in values],
                yerr=[
                    [value[0] - value[1] for value in values],
                    [value[2] - value[0] for value in values],
                ],
                fmt=style,
                capsize=3,
                label=label,
            )
        baseline, baseline_low, baseline_high = summary[size, 1]["union_seconds"]
        speedups = [baseline / summary[size, count]["union_seconds"][0] for count in worker_counts]
        bounds = [
            (1.0, 1.0)
            if count == 1
            else (
                baseline_low / summary[size, count]["union_seconds"][2],
                baseline_high / summary[size, count]["union_seconds"][1],
            )
            for count in worker_counts
        ]
        errors = [
            [center - low for center, (low, _) in zip(speedups, bounds, strict=True)],
            [high - center for center, (_, high) in zip(speedups, bounds, strict=True)],
        ]
        axes[1, column].errorbar(worker_counts, speedups, yerr=errors, fmt="o-", capsize=3)
        axes[1, column].plot(worker_counts, worker_counts, ":", color="gray", label="Ideal")
        axes[2, column].errorbar(
            worker_counts,
            [100 * speedup / count for speedup, count in zip(speedups, worker_counts, strict=True)],
            yerr=[
                [100 * error / count for error, count in zip(side, worker_counts, strict=True)]
                for side in errors
            ],
            fmt="o-",
            capsize=3,
        )
        axes[2, column].axhline(100, linestyle=":", color="gray")
        axes[0, column].set_title(f"{size:,} rows per input")
        axes[0, column].set_ylabel("Elapsed seconds")
        axes[0, column].legend(fontsize=9)
        axes[1, column].set_ylabel("Speedup vs one worker")
        axes[1, column].legend(fontsize=9)
        axes[2, column].set_ylabel("Speedup / workers (%)")
        for ax in axes[:, column]:
            ax.set_xscale("log", base=2)
            ax.set_xticks(worker_counts, labels=[str(value) for value in worker_counts])
            ax.set_xlabel("CANFAR workers (1 CPU, 4 GiB each)")
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.2)
    fig.suptitle("CANFAR Ray union scaling · median and range · n=3, warmup=1")
    fig.tight_layout()
    fig.savefig(args.plot, dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
