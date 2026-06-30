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
  - **zone** – Tier 2: HEALPix-sharded cone match via `cdshealpix`. When
    `cdshealpix` is not importable the engine transparently falls back to
    Tier 1 with a logged warning, so the API contract is unchanged.
- Probabilistic qualification (engine-agnostic): `--probabilistic --priors g,r`
  appends a Budavári-style hierarchical Bayes factor `p_match` column in
  [0, 1] to every matched pair. Priors are fitted on the unconditional union
  of both catalogues and combined with the joint positional kernel.
- ID joins (relational joins on identifier columns) via polars.
- All standard join modes: inner, outer, left/right-outer, and anti joins.
- Returns an eager polars `DataFrame` by default, or a `LazyFrame` on request.

## Installation

Requires Python >= 3.10. For STILTS-backed matching you also need Java and a
`stilts` command (or set `STILTS_JAR`); otherwise the astropy fallback is used
automatically.

```bash
pip install .          # from source
pip install -e ".[dev]"  # development
pip install -e ".[hats]" # optional HATS/LSDB support
```

## Command line

```bash
# Two local files (coordinate columns auto-detected); CSV result on stdout
xmatch a.parquet b.csv

# Write the result to a file (.parquet/.csv/.fits/.hats)
xmatch a.parquet b.csv -o matches.parquet -r 1.5

# Save a union catalogue as a HATS directory for spatial queries
xmatch gaia allwise.csv twomass.csv --union -r 1.5 -o master.hats --hats-threshold 50000

# Local file vs a configured remote catalogue (downloaded around the local footprint)
xmatch my_sources.csv gaia -o my_gaia.parquet -r 2.0

# Error-ellipse match (needs error columns in the catalogue config)
xmatch gaia desils_noao --matcher skyerr --max-error 3.0 -o out.parquet

# Pick the in-process cKDTree engine (no Java) and stream matches to disk
xmatch a.parquet b.parquet --engine fast -r 1.0 -o matches.parquet

# Compute a probabilistic p_match column from a photometric prior
xmatch a.parquet b.parquet --engine fast --probabilistic --priors g_mag,r_mag -o matches.parquet

# Keep all matches within the radius, and use an outer join
xmatch a.csv b.csv --find all --join 1or2

# ID join instead of sky position
xmatch a.csv b.csv --id-join --id1 source_id --id2 source_id

# Inspect the catalogue configuration
xmatch --list
xmatch --describe gaia
```

Run `xmatch --help` for the full option list.

`--matcher` / `--max-error`: with `skyerr`/`skyellipse` a pair matches when
`separation ≤ max_error · (err₁ + err₂)` using the catalogues' positional errors
(`--radius` applies only to `sky`). `skyellipse` currently behaves like `skyerr`
(correlation is not yet modelled).

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
| 2    | `--engine zone` | HEALPix-sharded cone match via `cdshealpix` | Falls back to Tier 1 with a logged warning if `cdshealpix` is not importable. |
| (default) | `--engine auto` → `stilts` → `astropy` | unchanged from prior releases | Default when no `--engine` is passed. STILTS needs Java. |
| 3    | `--probabilistic --priors c1,c2,…` | Budavári hierarchical Bayes factor over the matched pairs | Engine-agnostic: layers a `p_match ∈ [0, 1]` column on top of whichever engine produced the pairs. |

The three engines behind `--engine` are mutually exclusive (sky-match engines).
Tier 3 is orthogonal: pass `--probabilistic --priors g,r` on top of any engine
to add a probabilistic qualification column.

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
