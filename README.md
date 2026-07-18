# xmatch

Cross-match astronomical catalogues with a single command:

```bash
xmatch cat1 cat2
```

`cat1`/`cat2` can each be a **local file** (Parquet, CSV, FITS), a **HATS catalogue
directory**, or the **name of a configured remote catalogue** (TAP service or the
CDS XMatch service). xmatch auto-detects the type, the coordinate columns, and the
matching strategy.

Internally all catalogue data flows through [polars](https://pola.rs) `LazyFrame`s;
results stream straight to disk so large matches never need to fit in memory.
Results can be saved as Parquet, CSV, FITS, or **HATS** (hierarchical tiling)
for efficient spatial queries on massive catalogues.

For deterministic scale evidence, run
`pixi run python scripts/benchmark_contract.py`. It reports fixed catalogue
sizes, candidate counts, wall time, peak RSS, output checksums, and global-ID
parity; use `/scratch` for large CANFAR runs.

To compare the in-process catalog engines across dense-patch and sparse
full-sky layouts, run:

```bash
pixi run python scripts/benchmark_contract.py \
  --left-rows 10000 \
  --right-rows 100000 \
  --timed-runs 3 \
  --engines fast,torchsky \
  --layouts dense,sparse \
  --finds best,all \
  --promotion-engine torchsky \
  --output benchmark_results/xmatch_engine_matrix.json
```

Unavailable or semantically unsupported engines are recorded as explicit
errors. Successful engines report pair-ID parity against `fast`; benchmark
failures never silently fall back to another engine. The matrix distinguishes
whether every requested engine completed from whether the successful engines
agree with the `fast` pair set.

`--promotion-engine` evaluates, but does not change, automatic engine dispatch.
Run the campaign in an environment containing the exact Torchsky wheel under
evaluation; a missing optional engine is recorded as an error, never replaced
by a fallback. Provenance records the Torchsky version and, for editable source
installs, its direct URL and Git revision.
The versioned `xmatch-engine-promotion-v1` policy requires successful
dense/sparse × best/all cases at 10,000-by-100,000 scale with at least three
timed runs, exact global-ID and pair-hash parity, wall time no more than 1.25×
`fast`, and isolated peak RSS no more than 1.5× `fast` in every case. Each
matrix case runs in a fresh subprocess so its peak RSS is not inherited from a
previous engine. The JSON records environment provenance and every failed gate;
keep Torchsky out of `engine="auto"` until an archived representative report is
eligible under this policy.

The SHA-pinned 10,000-by-100,000 macOS/arm64 preliminary report is archived at
[`benchmark_results/xmatch_engine_promotion_10k_100k_macos_arm64.json`](benchmark_results/xmatch_engine_promotion_10k_100k_macos_arm64.json)
(SHA-256 `8a63161fd8f9485206881a90fab1c41409ab06b132abc524f88f3880b8a3a0c9`).
The follow-up against Xmatch `b18eb97` and merged Torchsky `4113f94` is
[`benchmark_results/xmatch_engine_promotion_10k_100k_torchsky_4113f94_macos_arm64.json`](benchmark_results/xmatch_engine_promotion_10k_100k_torchsky_4113f94_macos_arm64.json)
(SHA-256 `584f183f3bab472b375513f59bc8f04131041f6b5c122ba954c3d02cf2ad0d11`).
Correctness parity passes in all four cases. Torchsky is faster on both sparse
cases, but dense wall time and the fixed PyTorch RSS footprint remain above the
policy thresholds, so automatic selection remains unchanged. Repeat the same
command in the pinned CANFAR release environment before treating the hardware
measurements as publication-grade evidence.

## Features

- One simple command: `xmatch cat1 cat2`.
- Inputs: local Parquet/CSV/FITS files, polars/pandas frames (Python API), remote
  TAP catalogues, the CDS XMatch service, and HATS catalogues (optional).
- Outputs: Parquet, CSV, FITS, and **HATS** (hierarchical tiling with
  configurable row-per-pixel threshold via `--hats-threshold`). Ideal for
  union catalogues and N-way results that grow into millions of rows.
- Spatial matching engines (`--engine NAME`, default `auto`):
  - **STILTS** `tmatch2` (`sky`/`skyerr`/`skyellipse`) when a `stilts` command is
    available (`engine=stilts`).
  - **astropy** KD-tree matcher as a pure-Python fallback (`engine=astropy`).
  - **fast** – Tier 1: in-process `scipy.spatial.cKDTree` on 3-D Cartesian
    unit-sphere embeddings; drop-in replacement for astropy (~3–5× faster,
    no Java dependency).
  - **torchsky** – optional tensor-native nearest-neighbour matching with
    coarse nested-HEALPix candidate pruning. It supports `matcher=sky`,
    `find=best` or `find=all`, and no extra-distance columns.
  - **zone** – Tier 2: HEALPix-sharded cone match via `cdshealpix`. When
    `cdshealpix` is not importable the engine transparently falls back to
    Tier 1 with a logged warning, so the API contract is unchanged.
- Bayesian qualification (engine-agnostic): `--probabilistic --priors g,r`
  appends a Budavári-style assumed-prior posterior `p_match` column in `[0, 1]`
  to every matched pair. It is not a calibrated probability. Priors are fitted
  on the unconditional union of both catalogues and combined with the joint
  positional kernel.
- ID joins (relational joins on identifier columns) via polars.
- All standard join modes: inner, outer, left/right-outer, and anti joins.
- Returns an eager polars `DataFrame` by default, or a `LazyFrame` on request.

## Installation

Requires Python >= 3.10. For STILTS-backed matching you also need Java and a
`stilts` command (or set `STILTS_JAR`); otherwise the fast fallback is used
automatically.

```bash
pip install .          # from source
pip install -e ".[dev]"  # development
pip install -e ".[hats]" # optional HATS/LSDB support
```

## Command line

The CLI is **subcommand-driven**.  The legacy `xmatch CAT1 CAT2` flat form
keeps working exactly as before, but you'll get a tidier experience with the
explicit subcommands.

| Goal                              | Command                                               |
|-----------------------------------|-------------------------------------------------------|
| Match two catalogues              | `xmatch match cat1 cat2 [-o out.parquet -r 1.5]`      |
| Match a local file vs a remote    | `xmatch my_sources.csv gaia -o my_gaia.parquet -r 2`  |
| N-way or union match              | `xmatch match gaia allwise.csv twomass.csv --union`   |
| List configured catalogues        | `xmatch list`                                         |
| Show a catalogue's columns        | `xmatch describe gaia`                                |
| Search remote TAP services        | `xmatch search gaia`                                  |
| Browse a remote endpoint          | `xmatch discover noirlab`                             |
| Schema + YAML hint for a table    | `xmatch discover noirlab --schema nsc_dr2.object`     |

```bash
# Two local files (coordinate columns auto-detected); CSV result on stdout
xmatch a.parquet b.csv

# Write the result to a file (.parquet/.csv/.fits/.hats)
xmatch a.parquet b.csv -o matches.parquet -r 1.5

# Save a union catalogue as a HATS directory for spatial queries
xmatch match gaia allwise.csv twomass.csv --union -r 1.5 -o master.hats --hats-threshold 50000

# Local file vs a configured remote catalogue (downloaded around the local footprint)
xmatch my_sources.csv gaia -o my_gaia.parquet -r 2.0

# Error-ellipse match (needs error columns in the catalogue config)
xmatch match gaia desils_noao --matcher skyerr --max-error 3.0 -o out.parquet

# Pick the in-process cKDTree engine (no Java) and stream matches to disk
xmatch match a.parquet b.parquet --engine fast -r 1.0 -o matches.parquet

# Compute an assumed-prior p_match posterior from a photometric prior
xmatch match a.parquet b.parquet --engine fast --probabilistic --priors g_mag,r_mag -o matches.parquet

# Keep all matches within the radius, and use an outer join
xmatch match a.csv b.csv --find all --join 1or2

# ID join instead of sky position
xmatch match a.csv b.csv --id-join --id1 source_id --id2 source_id

# Inspect the catalogue configuration
xmatch list
xmatch describe gaia
xmatch --list            # legacy alias kept for backwards compatibility
xmatch --describe gaia   # legacy alias kept for backwards compatibility
```

### Help, colour, and discovery

* Every command has rich, grouped `--help`: `xmatch --help` shows the five
  subcommands and a one-screen example block; `xmatch match --help` shows
  only the match options, grouped into **Output / Geometry / Match algorithm
  / ID join / Probabilistic / Proper motion / Advanced filters /
  Matcher-specific / Region** sections.
* Output is **auto-colourised** when stdout and stderr are TTYs, and
  automatically disabled otherwise (CI logs, pipes, `pytest -s`, etc.).
  Override with `--no-color`, or set `NO_COLOR=1` (or `XMATCH_NO_COLOR=1`)
  in your shell — see https://no-color.org/.
* Errors include a **"Did you mean: …"** hint when a catalogue name is
  close to one in the config (e.g. `xmatch --describe gaiaesa` suggests
  `gaia_esa`).

Run `xmatch --help` for the full top-level summary, and `xmatch <cmd> --help`
for command-specific options.

`--matcher` / `--max-error`: with `skyerr`/`skyellipse` a pair matches when
`separation ≤ max_error · (err₁ + err₂)` using the catalogues' positional errors
(`--radius` applies only to `sky`). `skyellipse` currently behaves like `skyerr`
(correlation is not yet modelled).

### Shell tab completion

`xmatch completion <bash|zsh|fish>` emits a self-contained completion
script. Catalogue names and TAP-endpoint short-names are pulled from the
active `xmatch.yaml` at emission time and embedded into the script &mdash;
so `xmatch match <TAB>`, `xmatch describe <TAB>`, and `xmatch discover <TAB>`
complete against the catalogues **you** have configured. A short fallback
list is bundled so completion never breaks if the config can't be loaded.

| Shell | Install                                                                                       |
|-------|-----------------------------------------------------------------------------------------------|
| bash  | `eval "$(xmatch completion bash)"` *(current shell only)*                                     |
| bash  | `xmatch completion bash > ~/.local/share/bash-completion/completions/xmatch` *(persistent)*    |
| zsh   | `xmatch completion zsh > "${ZDOTDIR:-$HOME}/.zsh/completions/_xmatch"`                        |
| fish  | `xmatch completion fish > ~/.config/fish/completions/xmatch.fish`                             |

```bash
# Preview a generated script before installing
xmatch completion bash | less

# Compose with other init scripts: shell-only `eval`
eval "$(xmatch completion bash)"
```

Refresh the script whenever you add or rename a catalogue in your
`xmatch.yaml` so TAB-complete reflects the new set.

### Configuration drift — `xmatch doctor`

`xmatch doctor` compares your active `xmatch.yaml` against the bundled
baseline and reports any drift. Three categories are tracked:

* **OUTDATED FIELDS** — the same catalogue exists in both configs but a
  structural field (`ra_column`, `dec_column`, `id_column`, `default_columns`,
  `archive`, `service_id`, `access_identifier`, `epoch`, …) differs.
  This is the kind of change that may silently break a downstream query,
  so it flips the exit code to **1**.
* **INFORMATIONAL** — `description`, `estimated_size`, `release` — cosmetic
  drift; reported but exit code stays 0.
* **MISSING IN USER / USER-ONLY** — intentional local trims or additions;
  reported but exit code stays 0 unless `--strict` is passed.

| Goal                                                          | Command                                  |
|---------------------------------------------------------------|------------------------------------------|
| Human-readable drift report                                    | `xmatch doctor`                          |
| Machine-readable JSON (for CI / dashboards)                  | `xmatch doctor --json`                   |
| Strict — exit 1 on **any** drift                              | `xmatch doctor --strict`                 |
| Quiet — single-line summary, ideal for scripts                | `xmatch doctor --quiet`                  |
| Inspect a custom config (skip auto-detected user config)      | `xmatch doctor --config ~/.config/xmatch/xmatch.yaml` |

Exit codes: **0** matches baseline (or only informational drift), **1**
drift detected, **2** cannot parse user/bundled config. The bundled baseline
is loaded from the package via `importlib.resources` so it tracks the
exact baseline shipped with your installed `xmatch` version — run
`xmatch doctor` after every upgrade to see what changed.

## Python API

```python
import polars as pl
from xmatch import CrossMatch

cm = CrossMatch()

# Local frame vs a configured remote catalogue
sources = pl.read_csv("my_sources.csv")
matches = cm.crossmatch(sources, "gaia", radius_arcsec=1.5)   # -> polars.DataFrame
print(matches.head())

# Lazy result (you call .collect() / .sink_parquet() yourself)
lazy = cm.crossmatch("a.parquet", "b.parquet", radius_arcsec=1.0, lazy=True)
lazy.sink_parquet("out.parquet")

# ID join
cm.crossmatch(a, b, id_join=True, id_column_1="id", id_column_2="id",
              join_type="1and2")
```

`crossmatch()` returns a polars `DataFrame` by default, a `LazyFrame` when
`lazy=True`, and `None` when `output_file=` is given (the result is streamed to
that file). Use `.to_pandas()` if you need a pandas frame.

### Multi-catalogue matching

```python
# N-catalogue pairwise intersection (cat1 × cat2 → result × cat3 → …)
result = cm.crossmatch_multi(
    ["gaia", "allwise.csv", "twomass.csv"],
    radius_arcsec=1.5,
    output_file="matches.parquet",
)

# Union catalogue: full outer join across all catalogues
# Every row from every catalogue survives — ideal for building
# a master catalogue of all measurements of all sources on the sky.
# A `_src_cats` column shows which catalogue(s) contributed (e.g. "1+2+3").
master = cm.union_match(
    ["gaia", "allwise.csv", "twomass.csv"],
    ra=180, dec=-30, radius_deg=0.5, radius_arcsec=1.5,
)
print(master["_src_cats"].value_counts())

# Bayesian N-way simultaneous crossmatching (Budavári & Szalay 2008)
# Scores tuples from ALL catalogues at once, producing a p_match column.
nway = cm.nway_match(
    ["gaia", "allwise.csv", "usno_b1.csv"],
    radius_arcsec=2.0,
    prior_columns=["phot_g_mean_mag"],
)
```

## Matcher tiers

Spatial sky matching has three tiers, each a drop-in replacement. They all share
the same result schema (left columns + right columns with collisions suffixed
`_2` + `sep_arcsec`); only the engine flag differs.

| Tier | Flag | Implementation | Notes |
|------|------|----------------|-------|
| 1    | `--engine fast` | in-process `scipy.spatial.cKDTree` on 3-D Cartesian unit-sphere embeddings | Drop-in replacement for astropy, ~3–5× faster, no Java. |
| 1T   | `--engine torchsky` | tensor-native matching with nested-HEALPix candidate pruning | Optional `xmatch[torchsky]` extra; exact for `matcher=sky`, `find=best` or `find=all`, with no extra-distance columns. |
| 2    | `--engine zone` | HEALPix-sharded cone match via `cdshealpix` | Falls back to Tier 1 with a logged warning if `cdshealpix` is not importable. |
| (default) | `--engine auto` → `stilts` → `fast` | automatic in-process fallback | Default when no `--engine` is passed. STILTS needs Java. |
| 3    | `--probabilistic --priors c1,c2,…` | Budavári hierarchical Bayes factor over the matched pairs | Engine-agnostic: layers an assumed-prior posterior `p_match ∈ [0, 1]` on top of whichever engine produced the pairs; it is not a calibrated probability. |

The engines behind `--engine` are mutually exclusive (sky-match engines).
Tier 3 is orthogonal: pass `--probabilistic --priors g,r` on top of any engine
to add a Bayesian qualification column.

Versioned downstream association releases use the
[`xmatch.association.v1`](docs/association-v1.md) candidate-record contract and
the atomic, streaming-verifiable `xmatch.association.release.v1` directory
format. Release IDs cover the canonical JSONL checksum, record count, shared
provenance, and optional parent release. The companion
`xmatch.association.component.release.v1` format publishes exact, namespaced
component membership and explicit merge/split lineage without treating a
component ID as stable across releases. Deterministic constructors require
explicit endpoint namespaces and decision policy; content-addressed component
delta releases classify created, continued, merged, split, and retired topology
from exact membership overlap.

## HATS output

When the output path ends with `.hats`, xmatch writes the result as a
**HATS (Hierarchical Adaptive Tiling Scheme) catalogue** — a directory of
HEALPix-partitioned Parquet files indexed for fast spatial queries. This is
especially valuable for:

- **Union catalogues** (`--union`) that merge millions of rows via outer joins.
- **N-way results** from `crossmatch_multi`, `union_match`, or `nway_match`.
- Any output large enough that you later plan to query it by position.

```bash
# Build a massive union catalogue, then query it efficiently later
xmatch gaia allwise_dl twomass_dl --union -r 1.5 -o master.hats --hats-threshold 50000
```

**Requirements:** HATS output needs the optional `lsdb` package (`pip install lsdb`).
The `--hats-threshold` flag (default 100 000) controls the maximum rows per
HEALPix pixel — lower values give finer spatial partitioning at the cost of
more files.

From Python, any crossmatch method that accepts `output_file=` supports `.hats`:

```python
# Union catalogue → HATS directory
cm.union_match(
    ["gaia", "allwise.csv", "twomass.csv"],
    output_file="master.hats",
    hats_threshold=50_000,
    ra=180, dec=-30, radius_deg=0.5, radius_arcsec=1.5,
)

# N-way Bayesian match → HATS directory
cm.nway_match(
    ["gaia", "allwise.csv", "usno_b1.csv"],
    output_file="nway_result.hats",
    radius_arcsec=2.0,
    prior_columns=["phot_g_mean_mag"],
    hats_threshold=25_000,
)
```

Later, read the HATS catalogue back for spatial queries with LSDB:

```python
import lsdb
cat = lsdb.read_hats("master.hats")
# Crossmatch against a new catalogue
result = cat.crossmatch(lsdb.read_hats("new_data.hats"), radius_arcsec=1.0)
```

## Strategy selection

| cat1 / cat2                         | strategy                                            |
|-------------------------------------|-----------------------------------------------------|
| local / local                       | STILTS or astropy sky match (or polars id join)     |
| local / remote TAP                  | cone-download remote around the local footprint, then local match |
| local / remote CDS XMatch           | CDS XMatch service (results rejoined by surrogate id) |
| remote TAP / remote TAP (same svc)  | ADQL spatial self-join on the service               |
| remote / remote (other)             | download both for a region, then local match        |
| any HATS                            | LSDB partition-aware crossmatch                      |

## Configuration

- `xmatch.yaml` defines archives (TAP/CDS services) and catalogues (access
  identifiers, coordinate / id / error columns, epoch). The packaged config is
  used by default; override with `--config` or `CrossMatch(config_file=...)`.
- `auth.py` loads credentials for authenticated TAP services via `keyring` or the
  `XMATCH_<SERVICE>_USER` / `XMATCH_<SERVICE>_PASSWORD` environment variables.

## License

MIT.
