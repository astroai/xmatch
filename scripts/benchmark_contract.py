#!/usr/bin/env python
"""Deterministic xmatch scale-contract benchmark.

The benchmark deliberately reports global source-ID validity and repeatability
alongside timing and RSS. Larger sizes belong on CANFAR ``/scratch``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import resource
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np
import polars as pl

from xmatch.matchers import MatchSpec, sky_match
from xmatch.sources import CatalogueSource

_PROMOTION_LAYOUTS = ("dense", "sparse")
_PROMOTION_FINDS = ("best", "all")


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
        "timed_runs": timed_runs,
        "candidate_count": result.height,
        "wall_seconds_median": float(np.median(timings)),
        "rows_per_second": left_rows / float(np.median(timings)),
        "peak_rss_bytes": _peak_rss_bytes(),
        "global_id_parity": parity,
        "pair_sha256": pair_checksum,
        "output_sha256": checksum,
    }


def _run_benchmark_isolated(
    left_rows: int,
    right_rows: int,
    timed_runs: int,
    *,
    engine: str,
    layout: str,
    find: str,
) -> dict[str, Any]:
    """Run one case in a fresh process so peak RSS is case-local."""
    with tempfile.TemporaryDirectory(prefix="xmatch-benchmark-") as tmpdir:
        output = Path(tmpdir) / "result.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--left-rows",
            str(left_rows),
            "--right-rows",
            str(right_rows),
            "--timed-runs",
            str(timed_runs),
            "--engines",
            engine,
            "--layouts",
            layout,
            "--finds",
            find,
            "--output",
            str(output),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(detail[-2000:] or f"benchmark exited {completed.returncode}")
        row = json.loads(output.read_text())
        row.pop("provenance", None)
        return row


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
                    row = _run_benchmark_isolated(
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


def evaluate_promotion(
    matrix: dict[str, Any],
    candidate_engine: str,
    *,
    max_wall_ratio: float = 1.25,
    max_rss_ratio: float = 1.5,
    min_left_rows: int = 10_000,
    min_right_rows: int = 100_000,
    min_timed_runs: int = 3,
) -> dict[str, Any]:
    """Evaluate a candidate without changing automatic engine dispatch."""
    if max_wall_ratio <= 0.0 or max_rss_ratio <= 0.0:
        raise ValueError("promotion ratios must be positive")
    if candidate_engine == "fast":
        raise ValueError("candidate engine must differ from the fast baseline")
    rows = matrix.get("rows", [])
    by_case = {(row.get("engine"), row.get("layout"), row.get("find")): row for row in rows}
    reasons: list[str] = []
    cases: list[dict[str, Any]] = []
    for layout, find in product(_PROMOTION_LAYOUTS, _PROMOTION_FINDS):
        baseline = by_case.get(("fast", layout, find))
        candidate = by_case.get((candidate_engine, layout, find))
        label = f"{layout}/{find}"
        if baseline is None or baseline.get("status") != "ok":
            reasons.append(f"{label}: missing successful fast baseline")
            continue
        if candidate is None or candidate.get("status") != "ok":
            reasons.append(f"{label}: missing successful {candidate_engine} result")
            continue
        case_reasons: list[str] = []
        for key in ("left_rows", "right_rows", "timed_runs"):
            if candidate.get(key) != baseline.get(key):
                case_reasons.append(f"candidate {key} differs from fast baseline")
        if candidate.get("left_rows", 0) < min_left_rows:
            case_reasons.append(f"left_rows < {min_left_rows}")
        if candidate.get("right_rows", 0) < min_right_rows:
            case_reasons.append(f"right_rows < {min_right_rows}")
        if candidate.get("timed_runs", 0) < min_timed_runs:
            case_reasons.append(f"timed_runs < {min_timed_runs}")
        if not candidate.get("global_id_parity", False):
            case_reasons.append("global source-ID parity failed")
        if not candidate.get("pair_parity_with_fast", False):
            case_reasons.append("candidate pair hash differs from fast")
        baseline_wall = baseline.get("wall_seconds_median", 0.0)
        baseline_rss = baseline.get("peak_rss_bytes", 0)
        if baseline_wall <= 0.0 or baseline_rss <= 0:
            reasons.append(f"{label}: fast baseline has invalid resource measurements")
            continue
        candidate_wall = candidate.get("wall_seconds_median", 0.0)
        candidate_rss = candidate.get("peak_rss_bytes", 0)
        if candidate_wall <= 0.0 or candidate_rss <= 0:
            reasons.append(f"{label}: candidate has invalid resource measurements")
            continue
        wall_ratio = candidate_wall / baseline_wall
        rss_ratio = candidate_rss / baseline_rss
        if wall_ratio > max_wall_ratio:
            case_reasons.append(f"wall ratio {wall_ratio:.3f} > {max_wall_ratio:.3f}")
        if rss_ratio > max_rss_ratio:
            case_reasons.append(f"RSS ratio {rss_ratio:.3f} > {max_rss_ratio:.3f}")
        reasons.extend(f"{label}: {reason}" for reason in case_reasons)
        cases.append(
            {
                "layout": layout,
                "find": find,
                "wall_ratio_vs_fast": wall_ratio,
                "rss_ratio_vs_fast": rss_ratio,
                "eligible": not case_reasons,
            }
        )
    return {
        "policy": "xmatch-engine-promotion-v1",
        "candidate_engine": candidate_engine,
        "eligible_for_automatic_selection": not reasons and len(cases) == 4,
        "thresholds": {
            "max_wall_ratio_vs_fast": max_wall_ratio,
            "max_rss_ratio_vs_fast": max_rss_ratio,
            "min_left_rows": min_left_rows,
            "min_right_rows": min_right_rows,
            "min_timed_runs": min_timed_runs,
            "required_layouts": list(_PROMOTION_LAYOUTS),
            "required_finds": list(_PROMOTION_FINDS),
        },
        "cases": cases,
        "reasons": reasons,
    }


def _torchsky_provenance() -> dict[str, str]:
    """Identify the optional Torchsky engine and its editable source revision."""
    try:
        distribution = importlib.metadata.distribution("torchsky")
    except importlib.metadata.PackageNotFoundError:
        return {"torchsky": "not-installed", "torchsky_source_sha": "not-installed"}

    result = {"torchsky": distribution.version, "torchsky_source_sha": "unavailable"}
    direct_url = distribution.read_text("direct_url.json")
    if not direct_url:
        return result
    try:
        source = json.loads(direct_url)
        url = str(source["url"])
    except (KeyError, TypeError, json.JSONDecodeError):
        return result
    result["torchsky_source_url"] = url
    parsed = urlparse(url)
    if parsed.scheme != "file" or not source.get("dir_info", {}).get("editable"):
        return result
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(unquote(parsed.path)),
        capture_output=True,
        text=True,
        check=False,
    )
    if git.returncode == 0:
        result["torchsky_source_sha"] = git.stdout.strip()
    return result


def benchmark_provenance() -> dict[str, str]:
    """Return enough environment identity to reproduce an archived report."""
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    result = {
        "created_utc": datetime.now(UTC).isoformat(),
        "git_sha": git.stdout.strip() if git.returncode == 0 else "unknown",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "xmatch": importlib.metadata.version("xmatch"),
        "numpy": np.__version__,
        "polars": pl.__version__,
    }
    result.update(_torchsky_provenance())
    return result


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
    parser.add_argument("--promotion-engine")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.promotion_engine and len(args.engines) == len(args.layouts) == len(args.finds) == 1:
        parser.error("--promotion-engine requires a matrix with multiple engines/layouts/finds")
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
        if args.promotion_engine:
            result["promotion"] = evaluate_promotion(result, args.promotion_engine)
    result["provenance"] = benchmark_provenance()
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
