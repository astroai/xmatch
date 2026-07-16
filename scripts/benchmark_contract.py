#!/usr/bin/env python
"""Deterministic xmatch scale-contract benchmark.

The benchmark deliberately reports global source-ID validity and repeatability
alongside timing and RSS. Larger sizes belong on CANFAR ``/scratch``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from xmatch.matchers import MatchSpec, sky_match
from xmatch.sources import CatalogueSource


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def synthetic_catalogues(
    left_rows: int,
    right_rows: int,
    *,
    seed: int = 20260712,
    layout: str = "dense",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return deterministic catalogues with one nearby right row per left row."""
    if left_rows < 1 or right_rows < left_rows:
        raise ValueError("right_rows must be >= left_rows >= 1")
    if layout not in ("dense", "sparse"):
        raise ValueError("layout must be 'dense' or 'sparse'")
    rng = np.random.default_rng(seed)
    if layout == "dense":
        left_ra = rng.uniform(150.0, 150.5, left_rows)
        left_dec = rng.uniform(1.5, 2.0, left_rows)
        right_ra = rng.uniform(150.0, 150.5, right_rows)
        right_dec = rng.uniform(1.5, 2.0, right_rows)
    else:
        left_ra = rng.uniform(0.0, 360.0, left_rows)
        left_dec = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, left_rows)))
        right_ra = rng.uniform(0.0, 360.0, right_rows)
        right_dec = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, right_rows)))
    right_ra[:left_rows] = left_ra + 1.0e-7
    right_dec[:left_rows] = left_dec - 1.0e-7
    if layout == "dense" and right_rows >= 2 * left_rows:
        right_ra[left_rows : 2 * left_rows] = left_ra + 2.0e-7
        right_dec[left_rows : 2 * left_rows] = left_dec - 2.0e-7
    left = pl.DataFrame(
        {
            "source_id": np.arange(left_rows, dtype=np.int64),
            "ra": left_ra,
            "dec": left_dec,
        }
    )
    right = pl.DataFrame(
        {
            "source_id": np.arange(10_000_000, 10_000_000 + right_rows, dtype=np.int64),
            "ra": right_ra,
            "dec": right_dec,
        }
    )
    return left, right


def run_benchmark(
    left_rows: int,
    right_rows: int,
    timed_runs: int = 3,
    *,
    engine: str = "fast",
    layout: str = "dense",
    find: str = "all",
) -> dict[str, Any]:
    """Run one matcher case and return a JSON-serializable evidence record."""
    if timed_runs < 1:
        raise ValueError("timed_runs must be positive")
    left, right = synthetic_catalogues(left_rows, right_rows, layout=layout)
    left_src = CatalogueSource(
        name="left",
        is_local=True,
        ra_column="ra",
        dec_column="dec",
        id_column="source_id",
        default_pos_error_arcsec=0.1,
    )
    right_src = CatalogueSource(
        name="right",
        is_local=True,
        ra_column="ra",
        dec_column="dec",
        id_column="source_id",
        default_pos_error_arcsec=0.1,
    )
    left_lf, right_lf = left.lazy(), right.lazy()
    spec = MatchSpec(radius_arcsec=1.0, find=find, fallback_policy="error")
    for _ in range(1):
        sky_match(left_src, right_src, left_lf, right_lf, spec, engine=engine).collect()
    timings: list[float] = []
    result: pl.DataFrame | None = None
    for _ in range(timed_runs):
        start = time.perf_counter()
        result = sky_match(left_src, right_src, left_lf, right_lf, spec, engine=engine).collect()
        timings.append(time.perf_counter() - start)
    assert result is not None

    right_ids = result.get_column("source_id_2").to_numpy()
    parity = bool(
        len(right_ids) > 0
        and np.all(right_ids >= 10_000_000)
        and np.all(right_ids < 10_000_000 + right_rows)
    )
    canonical = result.sort(["source_id", "source_id_2"]).to_dicts()
    checksum = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, default=str).encode()
    ).hexdigest()
    pair_rows = result.select("source_id", "source_id_2").sort(["source_id", "source_id_2"]).rows()
    pair_checksum = hashlib.sha256(
        json.dumps(pair_rows, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "benchmark": "xmatch-global-id-v1",
        "engine": engine,
        "layout": layout,
        "find": find,
        "left_rows": left_rows,
        "right_rows": right_rows,
        "candidate_count": result.height,
        "wall_seconds_median": float(np.median(timings)),
        "rows_per_second": left_rows / float(np.median(timings)),
        "peak_rss_bytes": _peak_rss_bytes(),
        "global_id_parity": parity,
        "pair_sha256": pair_checksum,
        "output_sha256": checksum,
    }


def run_matrix(
    left_rows: int,
    right_rows: int,
    timed_runs: int,
    *,
    engines: tuple[str, ...],
    layouts: tuple[str, ...],
    finds: tuple[str, ...],
) -> dict[str, Any]:
    """Run an engine/layout matrix and compare pair identity with ``fast``."""
    rows: list[dict[str, Any]] = []
    for layout in layouts:
        for find in finds:
            for engine in engines:
                try:
                    row = run_benchmark(
                        left_rows,
                        right_rows,
                        timed_runs,
                        engine=engine,
                        layout=layout,
                        find=find,
                    )
                    row["status"] = "ok"
                except Exception as exc:  # noqa: BLE001 - benchmark evidence records failures
                    row = {
                        "benchmark": "xmatch-global-id-v1",
                        "engine": engine,
                        "layout": layout,
                        "find": find,
                        "left_rows": left_rows,
                        "right_rows": right_rows,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                rows.append(row)

    baselines = {
        (row["layout"], row["find"]): row["pair_sha256"]
        for row in rows
        if row["status"] == "ok" and row["engine"] == "fast"
    }
    for row in rows:
        baseline = baselines.get((row["layout"], row["find"]))
        row["pair_parity_with_fast"] = (
            baseline is not None and row["status"] == "ok" and row["pair_sha256"] == baseline
        )
    return {
        "benchmark": "xmatch-engine-matrix-v1",
        "rows": rows,
        "all_requested_engines_succeeded": all(row["status"] == "ok" for row in rows),
        "all_successful_engines_match_fast": all(
            row["pair_parity_with_fast"] for row in rows if row["status"] == "ok"
        ),
    }


def _csv_values(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-rows", type=int, default=10_000)
    parser.add_argument("--right-rows", type=int, default=100_000)
    parser.add_argument("--timed-runs", type=int, default=3)
    parser.add_argument("--engines", type=_csv_values, default=("fast",))
    parser.add_argument("--layouts", type=_csv_values, default=("dense",))
    parser.add_argument("--finds", type=_csv_values, default=("all",))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if len(args.engines) == len(args.layouts) == len(args.finds) == 1:
        result = run_benchmark(
            args.left_rows,
            args.right_rows,
            args.timed_runs,
            engine=args.engines[0],
            layout=args.layouts[0],
            find=args.finds[0],
        )
    else:
        result = run_matrix(
            args.left_rows,
            args.right_rows,
            args.timed_runs,
            engines=args.engines,
            layouts=args.layouts,
            finds=args.finds,
        )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
