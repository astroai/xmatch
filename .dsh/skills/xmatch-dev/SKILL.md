---
name: xmatch-dev
description: Develop xmatch (catalogue cross-matching) — polars pipeline, engine matrix, deterministic benchmark contract, HATS output
metadata:
  project: xmatch
  stack: python, polars, pixi
---

# xmatch development

Cross-match astronomical catalogues: local files (Parquet, CSV, FITS), HATS
catalogue directories, or remote catalogues (TAP services, CDS XMatch) with a
single command. All data flows through polars `LazyFrame`s; pairwise local
matches can use an explicit memory budget for staged, streaming deterministic
partitions. Outputs: Parquet, CSV, FITS, or HATS.

## Deterministic scale evidence (the core gate)

`pixi run python scripts/benchmark_contract.py` reports fixed catalogue sizes,
candidate counts, wall time, peak RSS, output checksums, and global-ID parity.
Use `/scratch` for large CANFAR runs. Engine matrix runs compare engines
(`fast`, `torchsky`) across layouts (`dense`, `sparse`):

```bash
pixi run python scripts/benchmark_contract.py \
  --left-rows 10000 --right-rows 100000 --timed-runs 3 \
  --engines fast,torchsky --layouts dense,sparse --finds best,all \
  --promotion-engine torchsky --output benchmark_results/xmatch_engine_matrix.json
```

Rules:

- Unavailable or unsupported engines are recorded as explicit errors; a
  benchmark failure NEVER silently falls back to another engine.
- Successful engines report pair-ID parity against `fast`; the matrix
  distinguishes "all engines completed" from "completed engines agree".
- `--promotion-engine` evaluates but does not change automatic engine dispatch.
- Provenance records the torchsky version under evaluation.

## Gates

- `pixi run preflight-push` (ruff + format --check + compileall)
- `pixi run ci-local` (preflight + `tests/test_ci_smoke.py`)
- `pixi run test`

Algorithm reference: `CROSSMATCH_ALGORITHMS.md`; audits in `docs/audits/`.
