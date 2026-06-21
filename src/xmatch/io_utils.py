"""Polars-first I/O helpers for reading and writing astronomical catalogues.

All catalogue data flows through :class:`polars.LazyFrame` internally. This module
centralises the conversions to/from the formats xmatch understands (Parquet, CSV,
FITS) and the interop with astropy ``Table`` objects returned by TAP/CDS services.
"""

import logging
from pathlib import Path
from typing import List, Union

import polars as pl

from .exceptions import InputError

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".parquet", ".csv", ".fits", ".fit"}

# Files that, when present in a directory, identify a HATS / HiPSCat catalogue.
_HATS_MARKERS = ("properties", "hats.properties", "catalog_info.json", "_metadata")

FrameLike = Union[pl.DataFrame, pl.LazyFrame]


def is_hats_dir(path: Union[str, Path]) -> bool:
    """Return True if *path* looks like a HATS / HiPSCat catalogue directory."""
    p = Path(path)
    if not p.is_dir():
        return False
    return any((p / marker).exists() for marker in _HATS_MARKERS)


def astropy_table_to_polars(table) -> pl.DataFrame:
    """Convert an astropy ``Table`` to a polars ``DataFrame`` (via pandas).

    Multi-dimensional columns are dropped (no polars equivalent) and byte-string
    columns are decoded to UTF-8. Masked values become nulls.
    """
    keep = [name for name in table.colnames if getattr(table[name], "ndim", 1) == 1]
    dropped = set(table.colnames) - set(keep)
    if dropped:
        logger.warning("Dropping multi-dimensional columns: %s", sorted(dropped))
    pdf = table[keep].to_pandas() if keep else table.to_pandas()

    # ⚡ Bolt Optimization: Avoid iterating over all columns and checking dtype individually.
    # We use pdf.columns[pdf.dtypes == "object"] to identify object columns efficiently,
    # and .iat[0] for fast scalar access instead of .iloc[0].
    if len(pdf):
        for col in pdf.columns[pdf.dtypes == "object"]:
            if isinstance(pdf[col].iat[0], bytes):
                pdf[col] = pdf[col].str.decode("utf-8", "replace")
    return pl.from_pandas(pdf)


def polars_to_astropy(df: pl.DataFrame):
    """Convert a polars ``DataFrame`` to an astropy ``Table`` (via Arrow)."""
    from astropy.table import Table

    return Table.from_pandas(df.to_pandas())


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

    Parquet and CSV are scanned lazily (projection/predicate push-down). FITS is
    read eagerly through astropy and then exposed as a lazy frame.
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

    Parquet and CSV are streamed via ``sink_*`` so large results never need to be
    fully materialised in memory.
    """
    out = Path(output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    suffix = out.suffix.lower()
    lf = to_lazy(frame)

    if suffix == ".parquet":
        lf.sink_parquet(out)
    elif suffix == ".csv":
        lf.sink_csv(out)
    elif suffix in (".fits", ".fit"):
        polars_to_astropy(lf.collect()).write(out, overwrite=True)
    else:
        raise InputError(f"Unsupported output format '{suffix}'.")
    logger.info("Wrote results to %s", out)
