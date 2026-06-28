# xmatch public API

Stable surface only. Anything not exported from `xmatch.__init__` is internal.

---

## Top-level entry points

### `xmatch.CrossMatch`

The orchestrator. Holds the YAML config and dispatches
`local × local`, `local × remote`, `remote × remote` (incl. HATS) work to the
right backend.

```python
from xmatch import CrossMatch
cm = CrossMatch()  # uses bundled xmatch.yaml
result = cm.crossmatch("a.parquet", "b.csv", radius_arcsec=1.0)
```

Methods:

| Method | Purpose |
|---|---|
| `crossmatch(cat1, cat2, output_file=None, lazy=False, **params)` | Run the match. Returns `pl.DataFrame` (eager, default), `pl.LazyFrame` (`lazy=True`), or `None` (`output_file` set, streamed). |
| `resolve_source(value, overrides)` | Build a `CatalogueSource` from a path / frame / configured name. |
| `resolve_name(name)` | Apply catalogue-alias indirection. |
| `get_catalogue_config(name)` | Resolve a `name` (incl. alias) to a fully-merged config dict. |
| `list_catalogues(...)` / `describe(...)` | CLI helpers; iterate from `cm.catalogues_config` directly for programmatic use. |

### Match parameters (`**params`)

| Key | Type | Default | Notes |
|---|---|---|---|
| `radius_arcsec` | `float` | `1.0` | Search radius for `matcher="sky"`. |
| `matcher` | `"sky"` \| `"skyerr"` \| `"skyellipse"` | `"sky"` | `skyellipse` currently behaves like `skyerr` (correlation not yet modelled). |
| `max_error` | `float` | `3.0` | N-sigma cap for `skyerr`/`skyellipse`. |
| `join_type` | `"1and2"`, `"1or2"`, `"all1"`, `"all2"`, `"1not2"`, `"2not1"`, `"all"` | `"1and2"` | STILTS + polars id-join aligned. |
| `find` | `"best"` \| `"all"` | `"best"` | Keep the closest or all matches within the radius. |
| `engine` | `"auto"` \| `"stilts"` \| `"astropy"` | `"auto"` | Stub for STILTS when `STILTS_JAR` is on PATH; otherwise astropy. |
| `id_join` | `bool` | `False` | Switch from sky match to polars id join. |
| `id_column_1`, `id_column_2` | `str` | from config | ID column on each side. |
| `ra_column_1/2`, `dec_column_1/2` | `str` | auto-detected | Override coordinate column names. |
| `output_file` | `Path` | `None` | If set, result is streamed to disk via `sink_*`. |

---

## Data classes

```python
@dataclass
class CatalogueSource:
    name: str
    is_local: bool
    ra_column: Optional[str]
    dec_column: Optional[str]
    id_column: Optional[str]
    ra_err_column: Optional[str]     # for skyerr
    dec_err_column: Optional[str]    # for skyerr
    corr_column: Optional[str]       # for skyerr (not yet wired)
    pos_err_units: str = "arcsec"
    default_pos_error_arcsec: Optional[float]
    epoch: Optional[float]
    access_method: Optional[str]     # "tap" | "cds_xmatch" | "hats"
    ...
```

```python
@dataclass
class MatchSpec:
    radius_arcsec: float = 1.0
    matcher: str = "sky"
    max_error: float = 3.0
    join_type: str = "1and2"
    find: str = "best"
```

---

## I/O helpers (Arrow interop)

| Symbol | Signature | Notes |
|---|---|---|
| `FrameLike` | `Union[pl.DataFrame, pl.LazyFrame]` | Type alias exported for user code. |
| `is_hats_dir` | `(path) -> bool` | True if the directory looks like a HATS catalogue. |
| `scan_frame` | `(path) -> pl.LazyFrame` | Lazy scan of `.parquet` (predicate push-down), `.csv` (push-down), eager for `.fits`/`.fit`. |
| `write_frame` | `(frame, output_file)` | Streams `.parquet`/`.csv` via `sink_*`. FITS is eager. |
| `astropy_table_to_polars` | `(table) -> pl.DataFrame` | Arrow bridge (bytes → utf8 auto). |
| `polars_to_astropy` | `(df_or_lf) -> astropy.Table` | Arrow bridge; lazy frames are collected first. |

Performance note: the bridge reads/writes **polars' internal Arrow buffers**
directly, so a round-trip doesn't re-serialise through pandas — it just changes
the Python-level container.

---

## Exceptions (all inherit `CrossMatchError`)

| Class | When |
|---|---|
| `ConfigError` | YAML config missing, malformed, or invalid. |
| `InputError` | File/frame unreadable, unsupported extension, or auto-detection failure for local inputs. |
| `TapError` | TAP/ADQL failures (network, parse, async job failed). |
| `StiltsError` | STILTS not found or subprocess failed. |

Catch `CrossMatchError` for the full family.

---

## What is *not* public

* `matchers.sky_match`/`id_join` — implement ADQL/STILTS dialog; expose via
  `CrossMatch.crossmatch` instead.
* `remote_tap.download_from_tap`, `remote_cds.cds_xmatch_local_remote` — same.
* `astropy_utils.sky_extent`, `find_coord_columns` — workhorse helpers; may
  become public on demand.
* The `cfg["stilts_config"]` block — use the `stilts_cmd_base`, `java_opts`,
  `tmpdir` kwargs on `CrossMatch(...)`.

---

## CLI

```bash
xmatch --help
xmatch --list                       # print configured catalogues
xmatch --describe gaia              # print merged config
xmatch a.csv b.csv -o out.parquet -r 1.5
xmatch --matcher skyerr --max-error 3.0 a.csv b.csv
xmatch --id-join --id1 objid --id2 objid a.csv b.csv
```

See `README.md` for full workflow examples.
