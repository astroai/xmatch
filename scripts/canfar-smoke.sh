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
# mirrored copies.
#
# Usage:
#   scripts/canfar-smoke.sh                 # synthetic catalogues only
#   scripts/canfar-smoke.sh allwise gaia    # sync two configured catalogues first
#
# Exit code 0 on success; every failure aborts loudly.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCRATCH="${TMP_SCRATCH_DIR:-$(mktemp -d /tmp/xmatch-canfar-smoke.XXXXXX)}"
CACHE_ROOT="${XMATCH_CACHE_ROOT:-${TMP_SCRATCH_DIR:-$HOME/.cache}/xmatch-canfar-smoke}"
OUT_DIR="${SCRATCH}/out"
mkdir -p "${OUT_DIR}"

cd "${REPO_DIR}"
command -v pixi >/dev/null 2>&1 || { echo "Error: pixi not on PATH" >&2; exit 1; }

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
cat(45, 10.02, -4.99).write_parquet(f"{scratch}/cat2.parquet")
PY

if [ "$#" -ge 2 ]; then
    echo "==> syncing ${*} into ${CACHE_ROOT}"
    pixi run xmatch sync "$@" --cache-root "${CACHE_ROOT}"
    MATCH_INPUTS=("$@")
else
    MATCH_INPUTS=("${SCRATCH}/cat1.parquet" "${SCRATCH}/cat2.parquet")
fi

echo "==> ray-union match: ${MATCH_INPUTS[*]}"
pixi run xmatch match "${MATCH_INPUTS[@]}" \
    --engine ray-union \
    --cache-root "${CACHE_ROOT}" \
    -o "${OUT_DIR}/union.parquet"

echo "==> verifying output HATS dataset"
pixi run python - <<PY
from pathlib import Path
import hats
import polars as pl
from xmatch import hats_native

out = Path("${OUT_DIR}/union.parquet")
assert out.stat().st_size > 0, "empty output file"
hats_ds = hats.read_hats(out)
assert len(hats_ds.partition_info.as_dataframe()) > 0, "no pixel partition"
n = sum(
    pl.read_parquet(p).height
    for _, _, p in hats_native.list_hats_pixels(out)
)
# cat1 (60 rows) and cat2 (45 rows) sit ~72 arcsec apart vs the 1.0 arcsec
# radius, so the union is exactly 105 singleton rows; an exact count proves
# neither input catalogue was lost.
assert n == 105, f"expected exactly 105 rows (60+45 singles), got {n}"
print(f"OK: union output {out} readable, {n} rows across partitions")
PY

echo "canfar-smoke OK (cache: ${CACHE_ROOT})"
