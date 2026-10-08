#!/usr/bin/env python
"""Bounded benchmark for exact capped Ray-union tuple enumeration.

Measures the same pure-Python/NumPy block-combination kernel used by Ray tasks;
it does not launch a Ray cluster. Before timing, an independent full Cartesian
product oracle verifies unique tuple identities and the lowest separation
prefix. Equal-score ordering is intentionally unspecified, and cap-boundary
ties may select any combinations with the same score.
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from xmatcher import matchers, ray_union  # noqa: E402


def _angular_sep_arcsec(ra_deg: float, dec_deg: float) -> float:
    """Independent haversine great-circle separation from (0, 0)."""
    ra, dec = math.radians(ra_deg), math.radians(dec_deg)
    hav = math.sin(dec / 2.0) ** 2 + math.cos(dec) * math.sin(ra / 2.0) ** 2
    return math.degrees(2.0 * math.asin(math.sqrt(min(1.0, hav)))) * 3600.0


def _oracle_combinations(
    pools: list[tuple[np.ndarray, np.ndarray]], radius: float
) -> list[tuple[float, tuple[int, ...]]]:
    separations = [
        [_angular_sep_arcsec(float(ra), float(dec)) for ra, dec in zip(*pool, strict=True)]
        for pool in pools
    ]
    candidate_ids = [
        [index for index, sep in enumerate(pool_seps) if sep <= radius] for pool_seps in separations
    ]
    result = []
    for combo in itertools.product(*[[-1, *indices] for indices in candidate_ids]):
        used = [separations[k][index] for k, index in enumerate(combo) if index >= 0]
        if used:
            result.append((min(used), combo))
    return sorted(result)


def _capped_combinations(
    pools: list[tuple[np.ndarray, np.ndarray]], radius: float, cap: int
) -> tuple[list[tuple[int, ...]], np.ndarray]:
    selections, separations, _, _ = ray_union._block_combos(
        np.array([0.0]),
        np.array([0.0]),
        np.array([0], dtype=np.int64),
        pools,
        matchers._arcsec_to_chord(radius),
        cap,
        centre_label=1,
        cat_labels=list(range(2, len(pools) + 2)),
    )
    combinations = [
        tuple(int(selections[k][row]) for k in sorted(selections))
        for row in range(len(separations))
    ]
    return combinations, separations


def _assert_capped_matches_oracle(
    actual: tuple[list[tuple[int, ...]], np.ndarray],
    oracle: list[tuple[float, tuple[int, ...]]],
    cap: int,
) -> None:
    combinations, separations = actual
    if len(combinations) != cap or len(separations) != cap:
        raise AssertionError(f"capped kernel returned {len(combinations)} rows; expected {cap}")
    if len(set(combinations)) != len(combinations):
        raise AssertionError("capped kernel emitted duplicate candidate tuples")

    oracle_scores = {combo: score for score, combo in oracle}
    if len(oracle_scores) != len(oracle):
        raise AssertionError("independent oracle generated duplicate candidate tuples")
    try:
        tuple_scores = np.asarray([oracle_scores[combo] for combo in combinations], dtype=float)
    except KeyError as exc:
        raise AssertionError(
            f"capped kernel emitted a tuple absent from the full product: {exc}"
        ) from exc

    # ponytail: equal-score tuple ordering is intentionally unspecified; only
    # the score prefix is unique when a cap cuts through a tie.
    expected_scores = np.asarray([score for score, _ in oracle[:cap]], dtype=float)
    if np.any(np.diff(separations) < -1e-10):
        raise AssertionError("capped kernel did not emit tuples in nondecreasing score order")
    if not np.allclose(tuple_scores, separations, rtol=0.0, atol=1e-8):
        raise AssertionError("candidate tuple identities disagree with their oracle scores")
    if not np.allclose(separations, expected_scores, rtol=0.0, atol=1e-8):
        raise AssertionError("capped kernel scores disagree with the lowest-score oracle prefix")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pool-size", type=int, default=20, help="Candidate rows per secondary pool."
    )
    parser.add_argument(
        "--cap", type=int, default=10, help="Maximum emitted tuple count per centre."
    )
    parser.add_argument("--repeat", type=int, default=3, help="Capped-kernel timing repetitions.")
    args = parser.parse_args()
    if args.pool_size < 1 or args.cap < 1 or args.repeat < 1:
        parser.error("pool-size, cap, and repeat must be positive")

    radius = 10.0
    rng = np.random.default_rng(20261006)
    pools = []
    for _ in range(2):
        ra_arcsec = rng.uniform(-5.0, 5.0, args.pool_size)
        dec_arcsec = rng.uniform(-5.0, 5.0, args.pool_size)
        pools.append((ra_arcsec / 3600.0, dec_arcsec / 3600.0))

    oracle = _oracle_combinations(pools, radius)
    if args.cap >= len(oracle):
        parser.error(f"cap must be below the {len(oracle)} valid product rows in this case")
    actual = _capped_combinations(pools, radius, args.cap)
    _assert_capped_matches_oracle(actual, oracle, args.cap)

    capped_times = []
    full_times = []
    for _ in range(args.repeat):
        started = time.perf_counter()
        actual = _capped_combinations(pools, radius, args.cap)
        capped_times.append(time.perf_counter() - started)
        full_started = time.perf_counter()
        oracle = _oracle_combinations(pools, radius)
        full_times.append(time.perf_counter() - full_started)
        _assert_capped_matches_oracle(actual, oracle, args.cap)

    print(f"Candidate pool sizes: {args.pool_size} × {args.pool_size}")
    print(f"Complete product rows: {len(oracle):,}; requested cap: {args.cap:,}")
    print(f"Median and range in seconds; n={args.repeat}; warmup=1")
    for label, durations in (
        ("Full-product oracle", full_times),
        ("Exact capped kernel", capped_times),
    ):
        print(
            f"{label}: {np.median(durations):.6f}s; range {min(durations):.6f}–{max(durations):.6f}s"
        )
    print(
        "Capped tuple identities are unique and match the lowest-score oracle prefix; "
        "equal-score order and boundary-tie selection are interchangeable."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
