# xmatcher v0.5.1

## Breaking rename

- Python imports and the command-line executable change from `xmatch` to
  `xmatcher`.
- The bundled and user configuration file changes from `xmatch.yaml` to
  `xmatcher.yaml`.
- Configuration environment variables change from the `XMATCH_` prefix to
  `XMATCHER_`.
- The default cache path changes from `~/.cache/xmatch` to
  `~/.cache/xmatcher`. The documented CANFAR working/output root is
  `/arc/projects/hats/xmatcher`.

The `xmatch.*` association schema identifiers remain stable. Existing v0.5.0
GitHub release assets and historical provenance such as
`software_version: xmatch:0.5.0` are unchanged.

## Migration

To reuse an existing user configuration, copy
`~/.config/xmatch/xmatch.yaml` to `~/.config/xmatcher/xmatcher.yaml`.
The `adopt` command writes catalogue entries to this new user configuration.
To reuse the previous local cache, set
`XMATCHER_CACHE_ROOT=~/.cache/xmatch`. The shared CANFAR cache base
`/arc/projects/hats` is unchanged; the xmatcher working/output directory is
`/arc/projects/hats/xmatcher`.

## Installation

```bash
python -m pip install "xmatcher==0.5.1"
python -m pip install "xmatcher[cds,hats-ray,torchfits,ml]==0.5.1"
```

## Verification

The release source passed the full Pixi suite: 760 passed, 10 skipped, and 13
warnings in 523.18 seconds, including all 32 benchmark checks. The live ESA
Gaia TAP integration test passed. The pre-push gate passed with 727 passed,
10 skipped, and 33 slow/benchmark tests deselected; lint, formatting, typing,
and compilation checks also passed.

The separate optional-dependency run passed all 70 selected IO, Torchsky,
native HATS, Healpy, fallback-policy, Astropy scoring, and score-filtering checks,
using Torchfits 1.0.0 and
XGBoost 3.4.2. All 16 best/all Ray parity cases passed with the optional ML
dependencies installed. The scaling analyser self-check passed normally and under
`python -O`. The [CANFAR scaling report](canfar-scaling-2026-10-07.md) records
the earlier eight-worker results from the frozen `xmatch` 0.5.0 candidate, not
this `xmatcher` wheel. A fresh eight-worker smoke, with one independent CANFAR
session per worker, could not start: `canfar-cluster status` reported no
manager, and `canfar ps` failed its TLS connection to the CANFAR session
service. No sessions or jobs were created for this attempt, so the 0.5.1
release does not claim a new CANFAR run.

## Audit findings

Fixed defects cover masked FITS round trips, covariance inflation and singular
ellipse pairs, Astropy radius candidates, fallback candidate retention,
post-filter score alignment, catalogue-name resolution, archive authentication,
CDS row-key collisions, candidate namespaces and lock cleanup, incomplete HATS
refreshes, VOS replica safety, TAP row limits, and empty distributed inputs.
Existing coverage omitted the triggering boundary cases: masked fast-reader
success, nonzero cross covariance, singular pairs, score-bearing filters,
partial transfers, nested VOS listings, and schema-bearing empty catalogues.
The HTML listing fallback also now retains standard CSV partition metadata and
empty-catalogue schemas; the prior HTTP tests used Parquet partition metadata.
New regressions failed against the previous implementation: the combined
mutation run had 60 failures; six additional storage regressions failed against
an intermediate implementation before restoration. Reverting the CSV metadata
fix also made both new HTTP regressions fail.

The duplicate Astropy ellipse calculation now uses the existing shared filter.
The obsolete recursive storage walker was removed after checking all readers
and side effects; it had no external callers, so no test can exercise it.
No dependency was added.

Rejected hypotheses included a plain-radius boundary error and temporary-file
basename corruption: their reproductions preserved the expected results.
Moving filters before best-match selection was rejected because filtering is
explicitly a post-match operation; the misleading test was corrected instead.
Whole-cache transactions were deferred in favour of the existing per-file
transfer contract with metadata published last. Extra hardening of synthetic
wheel test fixtures is deferred; the actual wheel installation remains a
required CI gate.

The audit changed an earlier assumption that reader fallback alone preserved
FITS nulls: the writer also needed explicit mask serialization. It also caught
and corrected intermediate VOS backup and empty-partition regressions before
the frozen suites ran.

Pushing the tag `v0.5.1` runs the release workflow. That workflow publishes
these notes, the tested wheel and sdist, and `SHA256SUMS`, then uploads the
same archives to PyPI.
