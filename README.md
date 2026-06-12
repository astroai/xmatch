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

## Features

- One simple command: `xmatch cat1 cat2`.
- Inputs: local Parquet/CSV/FITS files, polars/pandas frames (Python API), remote
  TAP catalogues, the CDS XMatch service, and HATS catalogues (optional).
- Spatial matching engines:
  - **STILTS** `tmatch2` (`sky`/`skyerr`/`skyellipse`) when a `stilts` command is
    available (default).
  - **astropy** KD-tree matcher as a pure-Python fallback (no Java required).
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

# Write the result to a file (.parquet/.csv/.fits)
xmatch a.parquet b.csv -o matches.parquet -r 1.5

# Local file vs a configured remote catalogue (downloaded around the local footprint)
xmatch my_sources.csv gaia -o my_gaia.parquet -r 2.0

# Error-ellipse match (needs error columns in the catalogue config)
xmatch gaia desils_noao --matcher skyerr --max-error 3.0 -o out.parquet

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
