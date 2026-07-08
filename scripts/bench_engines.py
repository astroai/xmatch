#!/usr/bin/env python
"""Benchmark xmatch engines: fast, zone, ray on synthetic catalogues.

Generates random star catalogues at sizes [1k, 5k, 20k, 100k] and runs each
available engine with warm-up + timed iterations.  Ray is benchmarked only
when Ray is installed and a local cluster can be started.

When --extra-cols is passed, synthetic photometry columns are added and each
size also benchmarks N-dimensional cKDTree matching (spatial + photometry)
against the spatial-only baseline, showing the overhead.

Output: a formatted table of median times (seconds) and match counts.

Usage::

    python scripts/bench_engines.py              # default sizes
    python scripts/bench_engines.py --sizes 1k,10k,100k,500k
    python scripts/bench_engines.py --n-timed 5 --radius 2.0
    python scripts/bench_engines.py --extra-cols g:0.5,r:0.3 --chunk-size 5000
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl

# Add the project root to sys.path so `xmatch` is importable.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from xmatch.matchers import MatchSpec, sky_match  # noqa: E402
from xmatch.sources import CatalogueSource  # noqa: E402

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _src(name="synth", ra="ra", dec="dec"):
    return CatalogueSource(name=name, is_local=True, ra_column=ra, dec_column=dec)


def _make_catalogue(
    n_stars: int,
    seed: int = 42,
    density: float = 10.0,
    phot_cols: Optional[Dict[str, float]] = None,
) -> pl.DataFrame:
    """Generate *n_stars* random stars in a square patch (degrees).

    *density* controls spatial star density: the patch side length is
    ``sqrt(n_stars / density)`` degrees.

    When *phot_cols* is provided, synthetic photometry columns are added
    with normally-distributed values around a fixed mean (useful for
    N-d matching benchmarks).
    """
    rng = np.random.default_rng(seed)
    side_deg = np.sqrt(float(n_stars) / max(density, 1e-6))
    half = side_deg * 0.5

    cols = {
        "ra": rng.uniform(-half, half, n_stars),
        "dec": rng.uniform(-half, half, n_stars),
    }
    if phot_cols:
        for col_name in phot_cols:
            # Random magnitudes in a realistic range (10–20 mag).
            cols[col_name] = rng.normal(15.0, 1.5, n_stars)
    return pl.DataFrame(cols)


def _time_engine(
    engine: str,
    left_lf: pl.LazyFrame,
    right_lf: pl.LazyFrame,
    left_src: CatalogueSource,
    right_src: CatalogueSource,
    spec: MatchSpec,
    n_warmup: int,
    n_timed: int,
) -> Tuple[float, int]:
    """Return ``(median_seconds, n_matches)`` for *engine*."""
    times: list = []
    n_matches = 0
    for _ in range(n_warmup):
        out = sky_match(left_src, right_src, left_lf, right_lf, spec, engine=engine).collect()
        n_matches = out.height

    for _ in range(n_timed):
        t0 = time.perf_counter()
        out = sky_match(left_src, right_src, left_lf, right_lf, spec, engine=engine).collect()
        times.append(time.perf_counter() - t0)
        n_matches = out.height

    return float(np.median(times)), n_matches


def _format_time(sec: float) -> str:
    if sec < 0.001:
        return f"{sec * 1e6:.0f} µs"
    if sec < 1.0:
        return f"{sec * 1e3:.1f} ms"
    return f"{sec:.3f} s"


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark xmatch engines on synthetic catalogues."
    )
    parser.add_argument(
        "--sizes",
        default="1k,5k,20k,100k",
        help="Comma-separated catalogue sizes (k=×1000).",
    )
    parser.add_argument("--radius", type=float, default=1.0, help="Match radius in arcsec.")
    parser.add_argument("--n-warmup", type=int, default=1, help="Warm-up iterations per engine.")
    parser.add_argument("--n-timed", type=int, default=3, help="Timed iterations per engine.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Override _scipy_match_nd chunk size (default: 50000). "
        "Only affects N-d matching via the fast engine.",
    )
    parser.add_argument(
        "--extra-cols",
        default=None,
        help="Comma-separated col:weight pairs for N-d matching benchmark "
        "(e.g. 'g:0.5,r:0.3').  When set, each size also benchmarks "
        "N-d cKDTree matching vs spatial-only baseline.",
    )
    args = parser.parse_args()

    # Apply chunk-size override to the matchers module before any matching.
    from xmatch import matchers  # noqa: E402

    if args.chunk_size is not None:
        matchers._ND_CHUNK_SIZE = args.chunk_size

    # Parse sizes.
    sizes: List[int] = []
    for tok in args.sizes.split(","):
        tok = tok.strip().lower()
        mult = 1
        if tok.endswith("k"):
            mult = 1_000
            tok = tok[:-1]
        elif tok.endswith("m"):
            mult = 1_000_000
            tok = tok[:-1]
        try:
            sizes.append(int(float(tok) * mult))
        except ValueError:
            print(f"Invalid size token: {tok!r}", file=sys.stderr)
            return 1
    if not sizes:
        print("No sizes provided.", file=sys.stderr)
        return 1

    spec = MatchSpec(radius_arcsec=args.radius, find="best")
    left_src = right_src = _src()

    # Parse extra-cols into dict.
    phot_cols: Dict[str, float] = {}
    if args.extra_cols:
        for pair in args.extra_cols.split(","):
            pair = pair.strip()
            if not pair:
                continue
            if ":" in pair:
                col, _, w = pair.partition(":")
                with contextlib.suppress(ValueError):
                    phot_cols[col.strip()] = float(w.strip())
            else:
                phot_cols[pair] = 1.0

    nd_spec: Optional[MatchSpec] = None
    if phot_cols:
        nd_spec = MatchSpec(
            radius_arcsec=args.radius,
            find="best",
            extra_distance_cols=phot_cols,
        )

    # Determine available engines.
    engines = ["fast"]
    try:
        import cdshealpix  # noqa: F401

        engines.append("zone")
    except ImportError:
        pass

    ray_ok = False
    try:
        import ray  # noqa: F401

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, logging_level=40)
        ray_ok = True
        engines.append("ray")
    except Exception:
        pass

    # Header.
    len(phot_cols)
    hdr_engine = f"{'Engine':>18}" if nd_spec else f"{'Engine':>6}"
    print()
    print(f"{'Size':>8}  {hdr_engine}  {'Time':>10}  {'Matches':>8}  {'rate':>10}")
    print("-" * (61 + (6 if nd_spec else 0)))

    for n in sizes:
        df = _make_catalogue(n, seed=n, phot_cols=phot_cols)
        lf = df.lazy()
        indent = "" if not nd_spec else "      "
        print(f"{n:>8,}  " + " " * (38 + (6 if nd_spec else 0)))

        for eng in engines:
            try:
                elapsed, n_match = _time_engine(
                    eng,
                    lf,
                    lf,
                    left_src,
                    right_src,
                    spec,
                    args.n_warmup,
                    args.n_timed,
                )
                rate = n / elapsed if elapsed > 0 else 0
                print(
                    f"{indent:>8}  {eng:>6}  {_format_time(elapsed):>10}  "
                    f"{n_match:>8,}  {rate:>8,.0f}/s"
                )
            except Exception as exc:
                print(f"{indent:>8}  {eng:>6}  {'FAILED':>10}  ({exc})")

        # N-d matching benchmark (fast engine only, at several chunk sizes).
        # Spatial-only baseline is the "fast" engine row already printed above.
        if nd_spec:
            # Build chunk-size candidates; the configured size always appears.
            cand = {matchers._ND_CHUNK_SIZE}  # user config / default
            cand.add(min(5_000, n))  # small (many chunks)
            if n > 20_000:
                cand.add(n)  # no chunking
            below = sorted(c for c in cand if 0 < c < n)  # distinct sizes < n
            if any(c >= n for c in cand):
                below.append(n)  # one "full" row
            chunk_sizes = below

            _saved = matchers._ND_CHUNK_SIZE  # restore after loop
            try:
                for cs in chunk_sizes:
                    matchers._ND_CHUNK_SIZE = cs
                    if cs >= n:
                        label = "N-d (full: 1 chunk)"
                    else:
                        label = f"N-d ({cs // 1000}k chunk)"
                    try:
                        elapsed_nd, n_match_nd = _time_engine(
                            "fast",
                            lf,
                            lf,
                            left_src,
                            right_src,
                            nd_spec,
                            args.n_warmup,
                            args.n_timed,
                        )
                        rate_nd = n / elapsed_nd if elapsed_nd > 0 else 0
                        print(
                            f"{indent:>8}  {label:>18}  "
                            f"{_format_time(elapsed_nd):>10}  "
                            f"{n_match_nd:>8,}  {rate_nd:>8,.0f}/s"
                        )
                    except Exception as exc:
                        print(f"{indent:>8}  {label:>18}  {'FAILED':>10}  ({exc})")
            finally:
                matchers._ND_CHUNK_SIZE = _saved

        # Blank line between sizes.
        if n != sizes[-1]:
            print()

    print("-" * (61 + (6 if nd_spec else 0)))
    print(f"Engines: {', '.join(engines)}")
    print(f"Radius: {args.radius} arcsec, warmup={args.n_warmup}, timed={args.n_timed}")
    if phot_cols:
        print(f"N-d columns: {', '.join(f'{k}:{v}' for k, v in phot_cols.items())}")
    print(f"N-d chunk size: {matchers._ND_CHUNK_SIZE:,}")
    if ray_ok:
        import ray  # noqa: F811

        ray.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
