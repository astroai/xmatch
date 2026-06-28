"""Polars-first I/O helpers for reading and writing astronomical catalogues.

All catalogue data flows through :class:`polars.LazyFrame` internally. This
module centralises the conversions to/from the formats xmatch understands
(Parquet, CSV, FITS) and the interop with astropy ``Table`` objects returned by
TAP/CDS services.

Polars' storage layer is Apache Arrow, and a polars ``DataFrame`` can be
materialised as a ``pyarrow.Table`` via ``df.to_arrow()`` with zero copy.
Wherever possible (e.g. when streaming to a parquet sink) this module takes
that fast path. Conversions involving astropy ``Table`` go through a
``pyarrow.Table`` bridge *if* the installed astropy exposes ``Table.to_arrow``
(introduced in 7.x+) and otherwise fall back to a ``pandas`` round-trip. The
pandas path keeps everything working today; the Arrow path becomes zero-copy
once astropy's interface stabilises.

Bytes-string columns from VOTable responses (e.g. raw ``bytes`` for ``char``
TVP fields) are decoded to UTF-8 so polars sees ``pl.Utf8`` instead of
``pl.Binary``. Multi-D table columns (no polars equivalent) are dropped.
Masked values become nulls automatically.
"""

import logging
from pathlib import Path
from typing import Any, List, Union

import polars as pl

from .exceptions import InputError

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".parquet", ".csv", ".fits", ".fit"}

# Files that, when present in a directory, identify a HATS / HiPSCat catalogue.
_HATS_MARKERS = ("properties", "hats.properties", "catalog_info.json", "_metadata")

FrameLike = Union["pl.DataFrame", "pl.LazyFrame"]


def is_hats_dir(path: Union[str, Path]) -> bool:
    """Return True if *path* looks like a HATS / HiPSCat catalogue directory."""
    p = Path(path)
    if not p.is_dir():
        return False
    return any((p / marker).exists() for marker in _HATS_MARKERS)


def _decode_pandas_bytes_columns(pdf) -> Any:
    """Decode ``bytes`` columns in a pandas DataFrame to UTF-8 strings."""

    for col in pdf.columns:
        if pdf[col].dtype == object and len(pdf) and isinstance(pdf[col].iloc[0], bytes):
            pdf[col] = pdf[col].str.decode("utf-8", errors="replace")
    return pdf


def astropy_table_to_polars(table) -> pl.DataFrame:
    """Convert an astropy ``Table`` to a polars ``DataFrame`` (Arrow if available).

    Multi-D columns are dropped (no polars equivalent) and ``bytes`` string
    columns are decoded to UTF-8. Masked values become nulls.

    The function tries Apache Arrow for zero-copy round-trips when the installed
    astropy exposes ``Table.to_arrow`` (7.x+) and falls back to pandas.
    """

    keep = [name for name in table.colnames if getattr(table[name], "ndim", 1) == 1]
    dropped = set(table.colnames) - set(keep)
    if dropped:
        logger.debug(
            "astropy_table_to_polars: dropping %d multi-dimensional column(s): %s",
            len(dropped),
            sorted(dropped),
        )
    if not keep:
        return pl.DataFrame()
    table = table[keep]

    # Fast path: astropy ≥ 7.x with native arrow bridge.
    if hasattr(table, "to_arrow"):
        try:
            import pyarrow as pa

            pa_table = table.to_arrow()
            if any(pa.types.is_binary(f.type) for f in pa_table.schema):
                import pyarrow.compute as pc

                new_fields, new_columns = [], []
                for field, column in zip(pa_table.schema, pa_table.columns, strict=False):
                    if pa.types.is_binary(field.type):
                        decoded = pc.fill_null(pc.utf8_decode(column, replace_invalid=True), "")
                        field = pa.field(field.name, pa.string())
                        column = decoded
                    new_fields.append(field)
                    new_columns.append(column)
                pa_table = pa.table.from_arrays(new_columns, schema=pa.schema(new_fields))
            df = pl.from_arrow(pa_table)
            logger.debug(
                "Loaded astropy table via Arrow: %d rows, %d columns.",
                df.height,
                df.width,
            )
            return df
        except Exception as exc:  # pragma: no cover - depends on astropy version
            logger.debug("astropy Arrow bridge failed (%s), falling back to pandas.", exc)

    # Fallback path: astropy before 7.x / no native arrow bridge.
    pdf = table[keep].to_pandas() if keep else table.to_pandas()
    pdf = _decode_pandas_bytes_columns(pdf)
    df = pl.from_pandas(pdf)
    logger.debug(
        "Loaded astropy table via pandas fallback: %d rows, %d columns.",
        df.height,
        df.width,
    )
    return df


def polars_to_astropy(frame: FrameLike):
    """Convert a polars frame (eager or lazy) to an astropy ``Table``.

    Lazy frames are collected first. The function prefers the astropy Arrow
    bridge when present; otherwise it falls back to pandas.
    """
    from astropy.table import Table

    if isinstance(frame, pl.LazyFrame):
        frame = frame.collect()

    if hasattr(Table, "from_arrow"):
        try:
            return Table.from_arrow(frame.to_arrow())
        except Exception as exc:  # pragma: no cover - depends on astropy version
            logger.debug("astropy Table.from_arrow failed (%s), falling back to pandas.", exc)
    return Table.from_pandas(frame.to_pandas())


def _read_fits(path: Path) -> pl.DataFrame:
    from astropy.table import Table

    last_err = None
    for hdu in (1, 0):
        try:
            table = Table.read(path, hdu=hdu)
            return astropy_table_to_polars(table)
        except Exception as exc:  # try next HDU
            last_err = exc
    raise InputError(f"Could not read a table from FITS file {path}: {last_err}")


def scan_frame(path: Union[str, Path]) -> pl.LazyFrame:
    """Return a ``LazyFrame`` for a local catalogue file.

    Parquet and CSV are scanned lazily (projection/predicate push-down). FITS
    reads are inherently eager through astropy and then exposed as a lazy frame.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        return pl.scan_parquet(p)
    if suffix == ".csv":
        return pl.scan_csv(p)
    if suffix in (".fits", ".fit"):
        return _read_fits(p).lazy()
    raise InputError(f"Unsupported file format '{suffix}'. Supported: {sorted(SUPPORTED_SUFFIXES)}")


def frame_columns(frame: FrameLike) -> List[str]:
    """Return column names without materialising data."""
    if isinstance(frame, pl.LazyFrame):
        return frame.collect_schema().names()
    return frame.columns


def to_lazy(frame: FrameLike) -> pl.LazyFrame:
    return frame.lazy() if isinstance(frame, pl.DataFrame) else frame


def write_frame(frame: FrameLike, output_file: Union[str, Path]) -> None:
    """Write a frame to ``.parquet`` / ``.csv`` / ``.fits`` based on the suffix.

    Parquet and CSV are streamed via ``sink_*`` with the polars streaming engine
    so large results never need to be fully materialised in memory.
    """
    out = Path(output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    suffix = out.suffix.lower()
    lf = to_lazy(frame)

    if suffix == ".parquet":
        lf.sink_parquet(out, engine="streaming")
    elif suffix == ".csv":
        lf.sink_csv(out, engine="streaming")
    elif suffix in (".fits", ".fit"):
        polars_to_astropy(lf).write(out, overwrite=True)
    else:
        raise InputError(f"Unsupported output format '{suffix}'.")
    logger.info("Wrote results to %s", out)
