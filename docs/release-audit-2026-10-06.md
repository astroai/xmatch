# Release audit — 2026-10-06

This audit covers the repository at `e2e4581`, the distributed crossmatching
implementation, and the initial untracked `stellar_catalogue.md` note. Three
Luna agents reviewed catalogue integrity, local/remote matching, and release
documentation/tooling; the parent reviewed distributed execution and numerical
correctness. The release candidate version is 0.5.0. Extended integration and
performance verification continues below; publication awaits the release target
decision because the PyPI name `xmatch` belongs to an unrelated project.

## Scope

| Area | Files reviewed |
| --- | --- |
| Matching and orchestration | All matcher, Bayes, astrometry, request, source, crossmatch, Ray, HATS, spill, CLI and entry-point modules; corresponding tests |
| Remote services | TAP/CDS clients, TAP query generation, STILTS adapter, discovery and authentication; fake TAP server and remote/fallback tests |
| Data integrity | Association, observations, candidates, I/O, storage, mirroring and user configuration; all contract, release, component, delta, equivalence and storage tests |
| Examples and guidance | README, contributing guide, changelog, every documentation file, catalogue note and shell/Python scripts |
| Release inputs | Python/Pixi metadata, lockfile, source manifest, hooks, ignore rules, license, catalogue YAML, association JSON fixture and test CSV inputs |
| Verification | Every existing test module, benchmark helpers and shared fixtures; new distributed and benchmark-validation regressions |

Reviewed files without a demonstrated defect were retained. The lockfile changed
only to remove the unpublished Torchsky package extra; dependency versions and
catalogue/contract fixtures were retained.

## Main corrections

- Distributed tuple caps select the exact lowest-separation prefix, preserve
  original row indices, and avoid integer overflow. Independent full-product
  oracles check selection; there is no sampled candidate shortcut.
- Distributed HATS planning intersects compressed HEALPix intervals and measures
  actual error/motion bounds. Mixed orders and RING inputs produce disjoint
  NESTED output tiles with non-null routing coordinates. Empty outputs retain
  schema, including nested and timezone-bearing dtypes.
- Resume validates input metadata and freshness, rejects unknown/corrupt
  checkpoints, protects input/output overlap, and preserves caller-owned Ray.
  Remote partition contents are hashed; local freshness uses size and mtime.
- Local feature ranking considers every spatial candidate. Ellipse RA wrapping,
  tiny-angle geometry, coordinate units, covariance extraction and motion
  inflation now share consistent validated assumptions. FoF includes links
  between secondary catalogues while retaining primary-anchored output.
- Bayesian scoring uses consistent per-axis errors, corrected N-way
  normalization, complete-source photometric KDEs and explicit finite-value
  validation. Unknown errors require an explicit floor. Scores remain conditional
  on their documented model and prior assumptions.
- Remote service shortcuts run only when they implement the requested semantics.
  Other requests use regional downloads and the canonical local matcher.
  Uncertainty/epoch queries require an explicit remote search cone; projections
  retain required geometry, scoring and motion fields.
- TAP mirrors reject unsafe ordering keys and publish manifest-backed page
  generations. Failed forced refreshes preserve the prior usable generation.
  VOSpace publication stages complete trees and has promotion/rollback/recovery
  fault-injection checks.
- Storage/config writes use collision-safe temporary files and atomic replacement.
  Symlink containment, dangling destinations, scientific input validation,
  authentication aliases and discovery metadata handling have regressions.
- Examples use the typed request API correctly. Documentation distinguishes
  distributed Ray-union planning from pairwise native HATS materialization,
  states real memory limits, and labels AUF/macauff implementations as heuristics.
- The release gate runs the complete offline suite. Benchmarks assert correctness
  before reporting timings; old exception swallowing and order-dependent
  separation comparisons were removed.
- Source archives now include shared test helpers, input fixtures, documentation,
  scripts and reproducible environment metadata. The previous archive shipped
  tests that could not run. A stdlib artifact checker compares packaged bytes
  with the current source tree and validates wheel metadata.

The N-way model was checked against the small-angle approximation in
[Budavári & Szalay (2008), equations 16–18](https://arxiv.org/html/0707.1611).
Its equal prior odds and heuristic photometric terms do not establish empirical
population calibration.

## Initial verification evidence

These results precede the extended live-service and scale checks below.
Intermediate red/green logs under `/tmp/xmatch-*.log` are session evidence,
not shipped release inputs. Runnable regressions remain in the repository.

| Check | Result |
| --- | --- |
| Initial complete offline `pixi run ci-local -ra --durations=15` | 606 passed, 9 skipped, 32 deselected in 401.39 s |
| Initial preflight: Ruff, formatting, Mypy, bytecode compilation | Passed; 76 formatted Python files |
| Mypy over source, tests and scripts | Passed, 64 files; configured untyped-function bodies are not checked |
| Parent distributed/remote targeted suite | 75 passed |
| Catalogue integrity targeted suite | 160 passed, 1 skipped; later mirror suite 26 passed |
| Local/remote targeted suite | 213 passed, 2 skipped; Bayesian boundary checks 18 passed |
| Targeted distributed and benchmark validation regressions | 49 passed, including 4 newly added benchmark checks |
| Final edited TAP/Ray fixture checks | 3 passed against real local servers/workers |
| Real STILTS large-crossmatch module | 10 passed; fallback disabled and uncertainty columns explicitly declared |
| Bounded engine benchmark | Fast, zone and real Ray; 100 and 1,000 exact identity pairs, zero failures; feature-ranking chunk checks pass |
| Exact tuple-cap benchmark | 40,400 oracle tuples; capped 100 agree; five kernel repetitions |
| Initial wheel/source archive contents | 29 wheel files and 95 source files matched the source bytes at that check; metadata passed |
| Installed wheel smoke | Exports, every module import, YAML/JSON resources, metadata, exact-ID crossmatch and CLI help pass |
| Tests extracted from the source archive, using the installed wheel | CLI, configuration, benchmark guards and I/O: 86 passed, 1 missing-Torchfits skip |

The bounded tuple benchmark used `--pool-size 200 --cap 100 --repeat 5`:
mean capped-kernel time 0.002352 s, independent full-product oracle 0.028715 s.
This is a small synthetic workload, not a full-survey throughput or memory claim.

All 13 warnings in the offline gate were ERFA distance-override warnings from
synthetic motion cases with unspecified or zero distance. The nine skips cover
missing Torch, Torchsky, Torchfits, Healpy and XGBoost integrations. They were
not introduced to bypass failures. Targeted checks cover benchmark validation
and type/fixture corrections; their counts overlap the final complete gate and
should not be summed as distinct tests.

Final rebuilt artifacts will be `dist/xmatch-0.5.0-py3-none-any.whl` and
`dist/xmatch-0.5.0.tar.gz`, with SHA-256 digests in `dist/SHA256SUMS`.
Reproduce archive verification with `pixi run python scripts/check_release.py`.
The source archive includes `MANIFEST.in`, all test modules and their fixtures,
release documentation and tooling. Live service calls are not part of the wheel
smoke test.

## Extended release verification

The user explicitly authorized the remaining full, live and performance checks.
The first complete run collected 654 tests: 644 passed, nine optional dependency
skips and two failures in the Gaia/AllWISE STILTS benchmarks. A real-STILTS
two-primary/one-secondary regression reproduced the lost primary. Mapping
xmatch `find="best"` to STILTS `best1` fixed it; all eight live benchmarks then
passed (40 engine measurements), with fallback disabled.

Further fixes cover invalid SQL filters, missing/non-finite feature values,
feature normalization overflow, pooled catalogue feature statistics and nested
SciPy worker pools in one-CPU Ray tasks. ML negative sampling now uses a local
fixed seed after a strict Ray/XGBoost check exposed nonrepeatable training data
and caller random-state mutation. Failed explicit Ray addresses now obey
strict fallback policy and preserve the caller's environment. Storage and chunk
memory budgets use binary GiB as documented, and block assembly no longer
allocates unused null columns. New regressions retain these checks.

A late review found the planner's memory-budget rejection accidentally nested
after an unconditional exception. The corrected regression now checks both
acceptance at the GiB boundary and rejection below it. Mypy unreachable-code
warnings are enabled for source, tests and scripts to catch this failure class.
The same check removed an obsolete HATS return branch; positional-error units
retain runtime validation of inputs outside their documented type.

The next complete live run passed 669 tests with nine skips and exposed two
stale Ray-address tests. They assumed the old environment-clearing fallback.
The tests now verify the documented warning/local fallback policies, preserved
caller environment and exact output identities; all five targeted address
checks passed. The fresh complete run after the memory-budget correction
finished with 670 passed, nine skipped and one failure in the live ESA
remote-first test: the archive closed a job-metadata connection. An unchanged
retry timed out. These are unresolved live transport failures, not successful
release checks. A new offline three-catalogue regression exercises the same
remote-first path through real PyVO and a local UWS server, checking exact
identities and job deletion. That regression and the four TAP lifecycle checks
passed together (five tests, 12.54 s). The live test now anchors its cone and
expected identity to an actual Gaia DR3 sample; this revised live assertion has
not yet passed against ESA.

Fresh 500-row Gaia DR3, AllWISE and USNO-B1 samples passed all 12 live catalogue
checks. Optional Torchsky/Torchfits checks used the sibling Torchsky checkout
and published Torchfits 1.0.0; this does not establish an independently published
Torchsky distribution. Healpy and actual LR, Random Forest, XGBoost, AUF and
macauff scoring integrations also passed with strict fallback.

TAP execution now uses PyVO's native asynchronous-job context manager. Four
real-client tests against the local UWS protocol reproduced orphaned jobs before
the fix, then verified exact-job cleanup on success, server error, callback
failure and cancellation while preserving an unrelated job. The affected offline
TAP, mirror, CLI-sync and remote slice passed all 64 tests. Native cleanup may
still wait on an unresponsive archive, and failed submission before PyVO returns
a job object remains an upstream lifecycle limitation.

The live ARC VOS test used `python-vos` 3.7 and actual backend moves. It exposed
and fixed non-idempotent parent creation and incorrect `listdir()` result
handling. Upload/readback, transfer interruption with old bytes preserved,
promotion failure/rollback, replacement and obsolete-partition removal passed.
The unique test namespace was deleted and verified absent. Regression tests also
cover trailing-slash directory names and refusal to reuse a data node as a
container.

The HATS/Ray union benchmark validates every output identity, including matched
pairs and both sets of unmatched rows. At 100,000 rows per input, all four runs
produced the expected 150,000 rows across 29 output partitions. After one warmup,
the three timed runs had median 13.541 s and range 12.428–20.563 s. Driver peak
RSS was 241.6 MiB, excluding workers. These are synthetic scale measurements,
not survey throughput or total cluster memory claims.

At 1,000,000 rows per input, one warmup and three timed runs each verified
1,500,000 output rows across 286 partitions. Median match and assembly
time was 520.134 s, with range 330.063–541.328 s; input tiling took 14.163 s
and was excluded, as was the subsequent identity validation. Driver peak RSS
was 548.2 MiB, including validation but excluding workers. The broad
timing spread and background machine activity limit throughput interpretation.
This run began before the final null-allocation and GiB/Polars compatibility
fixes; its identity proof does not measure those later changes' performance.

The same 100,000-row workload also ran on an actual CANFAR Ray manager with two
separate one-CPU, 4-GiB workers. Ray task records confirm execution on both worker
nodes. Every warmup and measured run verified 150,000 output identities across
29 partitions. The three timed runs had median 6.330 s and range 6.149–6.820 s;
driver peak RSS was 264.4 MiB, excluding workers. Inputs and outputs used shared
ARC storage. Different hosts and workload conditions prevent a controlled
speedup claim against the local measurement.

A bounded repeat before the late memory-budget correction passed every identity check again:
100,000 rows per input, 150,000 output rows, 29 partitions, one warmup and three
timed runs. Median time was 10.438 s, with range 9.759–11.379 s; driver peak RSS
was 247.9 MiB, excluding workers. This is not a controlled before/after comparison.

The final capped-tuple benchmark checks unique tuple identities, each tuple's
independent score, nondecreasing ordering and the complete product's lowest-score
prefix. Equal-score order and boundary-tie selection may vary. With two 200-row
pools, 40,400 full-product tuples and cap 100, five timed runs after one warmup
had median 0.003814 s for the capped kernel and 0.120979 s for the full-product
oracle. These methods emit different row counts; the measurements establish
correctness and bounded kernel cost, not an equivalent-workload speedup.

The CANFAR environment used Python 3.13.15, Ray 2.58.0, NumPy 2.4.6, Polars
1.44.2, PyArrow 25.0.0, SciPy 1.18.1, Astropy 8.0.1, PyVO 1.9.1, HATS 0.11.0
and CDSHealpix 0.8.1. The three owned cluster sessions were removed after testing.
Following explicit cleanup authorization, the helper-created VOS configuration
was removed before starting the larger worker sweep. The controlled sweep
completed both input sizes with one, two, four and eight workers, using the same
frozen matching package and input hashes. Every point has one warmup and three
timed repetitions, with output identity checks and complete worker task records
saved locally. The one-million-row median fell from 283.778 s with one worker
to 60.004 s with eight workers (4.73 times faster, 59.1% parallel efficiency).
Launching worker sixteen failed with a manager HTTP 400 wrapping a CANFAR
session-service HTTP 500; no sixteen-worker timing exists. The user removed the
manager and workers after the sweep stopped. All seventeen owned manager,
bootstrap and worker IDs subsequently returned HTTP 404 on direct checks.
The stale manager configuration is also absent. Temporary ARC storage cleanup
is being verified separately after preserving all eight completed points.

Separate Python 3.14.8 / Polars 2.0 compatibility checks passed 206 tests with
24 optional dependency skips. This is a targeted compatibility check, not the
complete distributed release gate. The locked release runtime remains Python
3.13. Optional ML integrations exercised actual classifiers with strict fallback.

All configured pre-commit hooks passed. Mypy now uses the locked Pixi scientific
environment; the former isolated hook lacked dependencies and misclassified
Polars type aliases. Codespell passes with a reduced list of scientific tokens.

The tightened archive checker rejects undeclared wheel payloads and unexpected
generated source-archive metadata. Five malformed-archive cases failed before
the fix; all twelve checker tests now pass. The scaling analyzer self-check also
passes with Python assertions disabled. The latest fast gate passes Ruff,
formatting, Mypy and compilation. Final artifacts still need rebuilding from
the completed report and source tree.

The review candidate is [draft PR #204](https://github.com/astroai/xmatch/pull/204).
Its configured commit and pre-push offline gates passed. The first Ubuntu job
caught incompatible setup-pixi settings (`cache=true`, `run-install=false`);
the workflow now performs the locked installation through the setup action,
removing the redundant separate install step. The corrected workflow passed on
Ubuntu at commit `aafb077`: 639 tests passed, twelve skipped and 33 live tests
deselected, plus lint, typing, compilation, artifact checks and installed-wheel
CLI verification. [The successful run](https://github.com/astroai/xmatch/actions/runs/37637370937)
validates that commit; the subsequent report and checker edits still need the
final pre-push gate and a fresh CI run.

## Compatibility and limits from the initial audit

- Invalid uncertainties and non-finite prior measurements now fail explicitly.
  Bayesian users must declare positional uncertainty or an intentional floor;
  prior columns must exist in every input with comparable units and meaning.
- Unsupported remote/Ray-union options fail instead of being silently ignored.
  Remote uncertainty and target-epoch matching requires an explicit cone.
- The locked release runtime is Python 3.13. The extended checks above cover
  Linux/CANFAR, Python 3.14, optional integrations and live archives separately;
  targeted checks do not establish complete coverage of every dependency version.
- Local resume fingerprints can miss edits that preserve both size and mtime.
  Mutable TAP values can evade count-only refresh probes; use `--force` or pinned
  immutable upstream releases. Remote resume hashing reads every input partition.
- Pairwise native HATS adaptive/error cases can materialize all rows on the
  coordinator. Ray-union task memory estimates are advisory, not hard limits;
  tuple/output cardinality can still grow rapidly. Workers need shared inputs.
- The extended VOS check exercises backend moves and injected client failures;
  it does not simulate a server or machine crash. Multi-node CANFAR execution is
  verified separately above.
- Full and live tests were run only after explicit user authorization. No
  survey-scale performance claim follows from the synthetic checks.

Per-slice evidence is retained in `.cursor/harness/trajectories/`.
