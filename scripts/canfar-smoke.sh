#!/usr/bin/env bash
# scripts/canfar-smoke.sh
#
# End-to-end smoke for the mirror + distributed-union data plane on the
# CANFAR lab (or any scratch box with the pixi env):
#
#   1. build two tiny synthetic parquet catalogues in a scratch dir,
#   2. `xmatch match --engine ray-union` them end to end (local parquet ->
#      mirrored HATS in the cache -> Ray union match -> HATS output),
#   3. verify the output is a readable, non-empty HATS dataset.
#
# With optional positional catalogue names (configured in xmatch.yaml) the
# script first runs `xmatch sync NAME1 NAME2` so the real TAP/HATS remote
# data plane is covered too; the ray-union match then runs against the
# mirrored copies. This remote smoke checks only that the result is readable
# and non-empty; live catalogues do not have a fixed row-count oracle.
#
# Usage:
#   scripts/canfar-smoke.sh                 # synthetic catalogues only
#   scripts/canfar-smoke.sh allwise gaia    # sync two configured catalogues first
#
# Exit code 0 on success; every failure aborts loudly.

set -euo pipefail

if [ "$#" -eq 1 ]; then
    echo "Error: pass no arguments for the synthetic smoke or at least two configured catalogue names." >&2
    echo "Usage: $0 [CATALOGUE1 CATALOGUE2 ...]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCRATCH="${TMP_SCRATCH_DIR:-$(mktemp -d /tmp/xmatch-canfar-smoke.XXXXXX)}"
CACHE_ROOT="${XMATCH_CACHE_ROOT:-${SCRATCH}/cache}"
OUT_DIR="${SCRATCH}/out"
mkdir -p "${OUT_DIR}"

cd "${REPO_DIR}"
command -v pixi >/dev/null 2>&1 || { echo "Error: pixi not on PATH" >&2; exit 1; }

REMOTE_CATALOGUES=0
if [ "$#" -ge 2 ]; then
    REMOTE_CATALOGUES=1
    echo "==> syncing ${*} into ${CACHE_ROOT}"
    pixi run xmatch sync "$@" --cache-root "${CACHE_ROOT}"
    MATCH_INPUTS=("$@")
else
    echo "==> building synthetic catalogues in ${SCRATCH}"
    XS_SCRATCH="${SCRATCH}" pixi run python - <<'PY'
import os
import polars as pl

def cat(n, ra0, dec0, dx=0.001):
    return pl.DataFrame({
        "id": list(range(n)),
        "ra": [ra0 + (i % 25) * dx for i in range(n)],
        "dec": [dec0 + (i % 12) * dx for i in range(n)],
    })

scratch = os.environ["XS_SCRATCH"]
cat(60, 10.0, -5.0).write_parquet(f"{scratch}/cat1.parquet")
cat(45, 11.0, -5.0).write_parquet(f"{scratch}/cat2.parquet")
PY
    MATCH_INPUTS=("${SCRATCH}/cat1.parquet" "${SCRATCH}/cat2.parquet")
fi

echo "==> ray-union match: ${MATCH_INPUTS[*]}"
pixi run xmatch match "${MATCH_INPUTS[@]}" \
    --engine ray-union \
    --cache-root "${CACHE_ROOT}" \
    -o "${OUT_DIR}/union.hats"

echo "==> verifying output HATS dataset"
XS_OUT_DIR="${OUT_DIR}" XS_REMOTE_CATALOGUES="${REMOTE_CATALOGUES}" pixi run python - <<'PY'
import os
from pathlib import Path

import hats
import polars as pl
from xmatch import hats_native

def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"Error: {message}")


out = Path(os.environ["XS_OUT_DIR"]) / "union.hats"
require(out.is_dir(), f"expected HATS output directory, got {out}")
hats_ds = hats.read_hats(out)
require(len(hats_ds.partition_info.as_dataframe()) > 0, "no pixel partition")
partition_paths = [p for _, _, p in hats_native.list_hats_pixels(out)]
require(bool(partition_paths), "no pixel data files")

if os.environ["XS_REMOTE_CATALOGUES"] == "1":
    # Keep the live smoke bounded: read only the membership marker and hold
    # one partition at a time instead of materializing the full-sky result.
    n = sum(pl.read_parquet(path, columns=["_src_cats"]).height for path in partition_paths)
    require(n > 0, "union output is empty")
    print(
        f"OK: remote smoke output {out} is readable and non-empty ({n} rows); "
        "live catalogue results have no fixed row-count oracle"
    )
else:
    frames = [pl.read_parquet(path) for path in partition_paths]
    result = pl.concat(frames, how="diagonal_relaxed")
    n = result.height
    require(n > 0, "union output is empty")
    rows = result.to_dicts()
    left = [row for row in rows if row["_src_cats"] == "1"]
    right = [row for row in rows if row["_src_cats"] == "2"]
    require(len(left) == 60, f"expected 60 left singletons, got {len(left)}")
    require(len(right) == 45, f"expected 45 right singletons, got {len(right)}")
    require(
        {row["id"] for row in left} == set(range(60)),
        "left singleton IDs do not match the synthetic input",
    )
    require(
        {row["id_2"] for row in right} == set(range(45)),
        "right singleton IDs do not match the synthetic input",
    )
    require(all(row["id_2"] is None for row in left), "left rows include a right ID")
    require(all(row["id"] is None for row in right), "right rows include a left ID")
    require(len(rows) == 105, f"expected 105 unmatched singleton rows, got {len(rows)}")
    print(f"OK: synthetic union {out} preserves all 105 singleton IDs across partitions")
PY

echo "canfar-smoke OK (cache: ${CACHE_ROOT})"
