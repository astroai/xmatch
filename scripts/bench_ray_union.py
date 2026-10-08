#!/usr/bin/env python
"""Correctness-checked HATS/Ray union benchmark with planted pairs and singles.

Run with Pixi, e.g. ``pixi run python scripts/bench_ray_union.py --rows 100000``.
Half the rows have exact partners; the others lie in disjoint sky patches.
Every output identity is checked, not just the total row count. Timings include
planning, worker execution and HATS assembly, but exclude input construction.
"""

import argparse
import math
import os
import resource
import sys
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xmatcher import hats_native, ray_union  # noqa: E402
from xmatcher.mirror import _write_hats_native  # noqa: E402
from xmatcher.sources import CatalogueSource  # noqa: E402


def main() -> None:
    if not __debug__:
        raise SystemExit("Benchmark validation requires enabled assertions; omit -O/PYTHONOPTIMIZE")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100_000, help="Rows per input catalogue.")
    parser.add_argument("--threshold", type=int, default=10_000)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--cpus", type=int, default=2, help="Local Ray CPU slots.")
    parser.add_argument("--work-dir", type=Path, help="Keep input/output artifacts here.")
    args = parser.parse_args()
    if min(args.rows, args.threshold, args.repeat, args.cpus) < 1 or args.warmup < 0:
        parser.error("rows, threshold, repeat and cpus must be positive; warmup nonnegative")
    width = math.ceil(math.sqrt(args.rows))
    minimum_sep = (
        math.degrees(2 * math.asin(math.cos(math.radians(50)) * math.sin(math.radians(50 / width))))
        * 3600
    )
    if minimum_sep <= 1:
        parser.error("grid rows are too dense to guarantee only planted 1-arcsec pairs")

    import ray

    owns_ray = not ray.is_initialized()
    if owns_ray:
        address = os.environ.get("RAY_ADDRESS")
        ray.init(
            address=address or None,
            include_dashboard=False,
            logging_level=40,
            **({"num_cpus": args.cpus} if not address else {}),
        )
    durations = []
    try:
        with tempfile.TemporaryDirectory(prefix="xmatcher-union-bench-") as temporary:
            root = args.work_dir or Path(temporary)
            root.mkdir(parents=True, exist_ok=True)
            # A unique run directory preserves any caller-owned work-dir data.
            with (
                nullcontext(tempfile.mkdtemp(prefix="run-", dir=root))
                if args.work_dir
                else tempfile.TemporaryDirectory(prefix="run-", dir=root)
            ) as run:
                run_root = Path(run)
                ids = np.arange(args.rows, dtype=np.int64)
                split = args.rows // 2
                left = pl.DataFrame(
                    {
                        "id": ids,
                        "ra": 10 + ids % width * (100 / width),
                        "dec": -50 + ids // width * (100 / width),
                    }
                )
                right = left.with_columns(
                    pl.when(pl.col("id") >= split)
                    .then(pl.col("ra") + 150)
                    .otherwise(pl.col("ra"))
                    .alias("ra")
                )
                sources = []
                prepared = time.perf_counter()
                for name, frame in (("left", left), ("right", right)):
                    path = run_root / name
                    _write_hats_native(
                        frame, path, ra_column="ra", dec_column="dec", threshold=args.threshold
                    )
                    sources.append(
                        CatalogueSource(
                            name=name, is_local=True, path=path, ra_column="ra", dec_column="dec"
                        )
                    )
                print(f"Input tiling: {time.perf_counter() - prepared:.3f}s", flush=True)
                for iteration in range(args.warmup + args.repeat):
                    output = run_root / f"output-{iteration}"
                    started = time.perf_counter()
                    ray_union.ray_union_match(
                        sources,
                        sep_arcsec=1,
                        output_file=str(output),
                        hats_threshold=args.threshold,
                        max_tuples=10,
                        fallback_policy="error",
                    )
                    elapsed = time.perf_counter() - started
                    pixels = hats_native.list_hats_pixels(output)
                    result = pl.concat([pl.read_parquet(path) for _, _, path in pixels])
                    pairs = result.drop_nulls(["id", "id_2"]).sort("id")
                    left_only = result.filter(pl.col("id_2").is_null()).sort("id")
                    right_only = result.filter(pl.col("id").is_null()).sort("id_2")
                    assert pairs["id"].to_list() == pairs["id_2"].to_list() == list(range(split))
                    assert (
                        left_only["id"].to_list()
                        == right_only["id_2"].to_list()
                        == list(range(split, args.rows))
                    )
                    assert result.height == args.rows * 2 - split
                    assert (
                        result["_union_ra"].null_count() == result["_union_dec"].null_count() == 0
                    )
                    assert np.all(pairs["sep_arcsec"].to_numpy() < 1e-6)
                    if iteration >= args.warmup:
                        durations.append(elapsed)
                    print(
                        f"Iteration {iteration}: {elapsed:.3f}s; {result.height:,} verified rows, {len(pixels)} output partitions",
                        flush=True,
                    )
                print(
                    f"Median {np.median(durations):.3f}s; range {min(durations):.3f}–{max(durations):.3f}s; n={args.repeat}; warmup={args.warmup}"
                )
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
                    1024**2 if sys.platform == "darwin" else 1024
                )
                print(f"Driver peak RSS {rss:.1f} MiB (excludes worker processes)")
    finally:
        if owns_ray:
            ray.shutdown()


if __name__ == "__main__":
    main()
