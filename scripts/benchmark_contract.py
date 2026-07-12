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
    left_rows: int, right_rows: int, *, seed: int = 20260712
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return deterministic catalogues with one nearby right row per left row."""
    if left_rows < 1 or right_rows < left_rows:
        raise ValueError("right_rows must be >= left_rows >= 1")
    rng = np.random.default_rng(seed)
    left_ra = rng.uniform(150.0, 150.5, left_rows)
    left_dec = rng.uniform(1.5, 2.0, left_rows)
    right_ra = rng.uniform(150.0, 150.5, right_rows)
    right_dec = rng.uniform(1.5, 2.0, right_rows)
    right_ra[:left_rows] = left_ra + 1.0e-7
    right_dec[:left_rows] = left_dec - 1.0e-7
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


def run_benchmark(left_rows: int, right_rows: int, timed_runs: int = 3) -> dict[str, Any]:
    """Run the fast matcher and return a JSON-serializable evidence record."""
    if timed_runs < 1:
        raise ValueError("timed_runs must be positive")
    left, right = synthetic_catalogues(left_rows, right_rows)
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
    spec = MatchSpec(radius_arcsec=1.0, find="all", fallback_policy="error")
    for _ in range(1):
        sky_match(left_src, right_src, left_lf, right_lf, spec, engine="fast").collect()
    timings: list[float] = []
    result: pl.DataFrame | None = None
    for _ in range(timed_runs):
        start = time.perf_counter()
        result = sky_match(left_src, right_src, left_lf, right_lf, spec, engine="fast").collect()
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
    return {
        "benchmark": "xmatch-global-id-v1",
        "left_rows": left_rows,
        "right_rows": right_rows,
        "candidate_count": result.height,
        "wall_seconds_median": float(np.median(timings)),
        "rows_per_second": left_rows / float(np.median(timings)),
        "peak_rss_bytes": _peak_rss_bytes(),
        "global_id_parity": parity,
        "output_sha256": checksum,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-rows", type=int, default=10_000)
    parser.add_argument("--right-rows", type=int, default=100_000)
    parser.add_argument("--timed-runs", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = (
        json.dumps(
            run_benchmark(args.left_rows, args.right_rows, args.timed_runs),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
