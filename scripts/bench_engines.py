#!/usr/bin/env python
"""Correctness-checked throughput benchmark for spatial matching engines.

The synthetic left and right catalogues share identical positions and stable
row IDs. This is a deliberate self-match workload: every engine must return
exactly the identity pair for each source before its timing is reported. It is
useful for bounded regression checks, not a substitute for representative
survey workloads.

Examples::

    pixi run python scripts/bench_engines.py --sizes 100,1k --n-warmup 0 --n-timed 1
    pixi run python scripts/bench_engines.py --sizes 1k,10k --extra-cols g:0.5,r:0.3
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from xmatcher.matchers import MatchSpec, sky_match  # noqa: E402
from xmatcher.sources import CatalogueSource  # noqa: E402


def _src() -> CatalogueSource:
    return CatalogueSource(name="synthetic", is_local=True, ra_column="ra", dec_column="dec")


def _make_catalogue(
    n_stars: int,
    *,
    seed: int,
    density: float,
    phot_cols: dict[str, float],
) -> pl.DataFrame:
    """Build a reproducible, valid-declination square patch with stable IDs."""
    rng = np.random.default_rng(seed)
    # Keep Dec in [-60, 60] even for large requested catalogues.
    side_deg = min(math.sqrt(n_stars / density), 120.0)
    return pl.DataFrame(
        {
            "id": np.arange(n_stars, dtype=np.int64),
            "ra": rng.uniform(0.0, side_deg, n_stars),
            "dec": rng.uniform(-side_deg / 2.0, side_deg / 2.0, n_stars),
            **{name: rng.normal(15.0, 1.5, n_stars) for name in phot_cols},
        }
    )


def _pair_signature(result: pl.DataFrame) -> tuple[tuple[int, int], ...]:
    if not {"id", "id_2"}.issubset(result.columns):
        raise AssertionError(f"match result has no stable ID pair columns: {result.columns}")
    return tuple(
        sorted((int(left), int(right)) for left, right in result.select("id", "id_2").iter_rows())
    )


def _time_engine(
    engine: str,
    frame: pl.LazyFrame,
    source: CatalogueSource,
    spec: MatchSpec,
    expected: tuple[tuple[int, int], ...],
    n_warmup: int,
    n_timed: int,
) -> tuple[float, float, float, int]:
    """Time repeated matches and reject missing, duplicate, or incorrect pairs."""
    durations: list[float] = []
    count = 0
    for iteration in range(n_warmup + n_timed):
        started = time.perf_counter()
        result = sky_match(source, source, frame, frame, spec, engine=engine).collect()
        elapsed = time.perf_counter() - started
        signature = _pair_signature(result)
        if signature != expected:
            raise AssertionError(
                f"{engine} returned {len(signature)} pairs; expected {len(expected)} identity pairs"
            )
        count = result.height
        if iteration >= n_warmup:
            durations.append(elapsed)
    return float(np.median(durations)), min(durations), max(durations), count


def _parse_sizes(raw: str) -> list[int]:
    sizes = []
    for token in raw.split(","):
        value = token.strip().lower()
        multiplier = 1
        if value.endswith("k"):
            multiplier, value = 1_000, value[:-1]
        elif value.endswith("m"):
            multiplier, value = 1_000_000, value[:-1]
        try:
            size = int(value) * multiplier
        except ValueError as exc:
            raise ValueError(f"invalid size token: {token!r}") from exc
        if size <= 0:
            raise ValueError(f"sizes must be positive: {token!r}")
        sizes.append(size)
    if not sizes:
        raise ValueError("provide at least one size")
    return sizes


def _parse_extra_cols(raw: str | None) -> dict[str, float]:
    result: dict[str, float] = {}
    if raw is None:
        return result
    for token in raw.split(","):
        name, separator, weight_text = token.strip().partition(":")
        if not name or name in result:
            raise ValueError(f"empty or duplicate extra column: {token!r}")
        try:
            weight = float(weight_text) if separator else 1.0
        except ValueError as exc:
            raise ValueError(f"invalid weight for {name!r}: {weight_text!r}") from exc
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"weight for {name!r} must be finite and positive")
        result[name] = weight
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes", default="1k,5k,20k,100k", help="Comma-separated sizes; k/m suffixes accepted."
    )
    parser.add_argument("--radius", type=float, default=1.0, help="Match radius in arcseconds.")
    parser.add_argument("--n-warmup", type=int, default=1, help="Warm-up iterations per engine.")
    parser.add_argument("--n-timed", type=int, default=3, help="Timed iterations per engine.")
    parser.add_argument("--chunk-size", type=int, help="Override N-D matching chunk size.")
    parser.add_argument("--extra-cols", help="N-D benchmark columns, e.g. 'g:0.5,r:0.3'.")
    parser.add_argument(
        "--density", type=float, default=10.0, help="Synthetic source density per square degree."
    )
    args = parser.parse_args()

    try:
        sizes = _parse_sizes(args.sizes)
        phot_cols = _parse_extra_cols(args.extra_cols)
    except ValueError as exc:
        parser.error(str(exc))
    if not math.isfinite(args.radius) or args.radius <= 0:
        parser.error("--radius must be finite and positive")
    if args.n_warmup < 0 or args.n_timed < 1:
        parser.error("--n-warmup must be nonnegative and --n-timed must be positive")
    if not math.isfinite(args.density) or args.density <= 0:
        parser.error("--density must be finite and positive")
    if args.chunk_size is not None and args.chunk_size < 1:
        parser.error("--chunk-size must be positive")

    from xmatcher import matchers  # noqa: PLC0415

    saved_chunk_size = matchers._ND_CHUNK_SIZE
    if args.chunk_size is not None:
        matchers._ND_CHUNK_SIZE = args.chunk_size

    import ray  # noqa: PLC0415

    engines = ["fast", "zone", "ray"]
    started_ray = not ray.is_initialized()
    if started_ray:
        address = os.environ.get("RAY_ADDRESS")
        ray.init(
            address=address or None,
            include_dashboard=False,
            logging_level=40,
            **({"num_cpus": 2} if not address else {}),
        )

    source = _src()
    spec = MatchSpec(radius_arcsec=args.radius, find="best", fallback_policy="error")
    nd_spec = (
        MatchSpec(
            radius_arcsec=args.radius,
            find="best",
            extra_distance_cols=phot_cols,
            fallback_policy="error",
        )
        if phot_cols
        else None
    )
    failures = 0
    try:
        print("Synthetic identity self-match; every timed run is checked against exact source IDs.")
        print(f"Median and range in seconds; n={args.n_timed}; warmup={args.n_warmup}")
        print(
            f"{'Size':>9}  {'Engine':>16}  {'Median':>10}  {'Range':>19}  {'Pairs':>9}  {'Rows/s':>12}"
        )
        print("-" * 66)
        for size in sizes:
            frame = _make_catalogue(
                size,
                seed=size,
                density=args.density,
                phot_cols=phot_cols,
            ).lazy()
            expected = tuple((index, index) for index in range(size))
            for engine in engines:
                try:
                    elapsed, minimum, maximum, count = _time_engine(
                        engine, frame, source, spec, expected, args.n_warmup, args.n_timed
                    )
                    print(
                        f"{size:>9,}  {engine:>16}  {elapsed:>9.4f}s  {minimum:>8.4f}–{maximum:<8.4f}s  {count:>9,}  {size / elapsed:>12,.0f}"
                    )
                except Exception as exc:
                    failures += 1
                    print(f"{size:>9,}  {engine:>16}  FAILED: {exc}", file=sys.stderr)

            if nd_spec is not None:
                candidates = {matchers._ND_CHUNK_SIZE, min(5_000, size), size}
                for chunk in sorted(c for c in candidates if c > 0):
                    matchers._ND_CHUNK_SIZE = chunk
                    try:
                        elapsed, minimum, maximum, count = _time_engine(
                            "fast", frame, source, nd_spec, expected, args.n_warmup, args.n_timed
                        )
                        label = f"N-D chunk={chunk:,}"
                        print(
                            f"{size:>9,}  {label:>16}  {elapsed:>9.4f}s  {minimum:>8.4f}–{maximum:<8.4f}s  {count:>9,}  {size / elapsed:>12,.0f}"
                        )
                    except Exception as exc:
                        failures += 1
                        print(f"{size:>9,}  N-D chunk={chunk:,} FAILED: {exc}", file=sys.stderr)
            matchers._ND_CHUNK_SIZE = args.chunk_size or saved_chunk_size
        print("-" * 66)
        print(
            f"Engines checked: {', '.join(engines)}; radius={args.radius:g} arcsec; failures={failures}"
        )
    finally:
        matchers._ND_CHUNK_SIZE = saved_chunk_size
        if started_ray and "ray" in sys.modules:
            import ray  # noqa: PLC0415

            ray.shutdown()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
