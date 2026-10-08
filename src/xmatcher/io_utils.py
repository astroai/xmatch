"""Polars-first I/O helpers for reading and writing astronomical catalogues.

All catalogue data flows through :class:`polars.LazyFrame` internally. This
module centralises the conversions to/from the formats xmatcher understands
(Parquet, CSV, FITS) and the interop with astropy ``Table`` objects returned by
TAP/CDS services.

Polars' storage layer is Apache Arrow, and a polars ``DataFrame`` can be
materialised as a ``pyarrow.Table`` via ``df.to_arrow()`` with zero copy.
Wherever possible (e.g. when streaming to a parquet sink) this module takes
that fast path. Conversions involving astropy ``Table`` go directly through
``pyarrow`` and ``numpy`` column buffers.

Bytes-string columns from VOTable responses (e.g. raw ``bytes`` for ``char``
TVP fields) are decoded to UTF-8 so polars sees ``pl.String`` instead of
``pl.Binary``. Multi-D table columns (no polars equivalent) are dropped.
Masked values become nulls automatically.
"""

import logging
from pathlib import Path

import numpy as np
import polars as pl

from .exceptions import InputError

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".parquet", ".csv", ".tsv", ".tab", ".fits", ".fit", ".hats"}

# Files that, when present in a directory, identify a HATS / HiPSCat catalogue.
_HATS_MARKERS = ("properties", "hats.properties", "catalog_info.json", "_metadata")

FrameLike = pl.DataFrame | pl.LazyFrame


def is_hats_dir(path: str | Path) -> bool:
    """Return True if *path* looks like a HATS / HiPSCat catalogue directory."""
    p = Path(path)
    if not p.is_dir():
        return False
    return any((p / marker).exists() for marker in _HATS_MARKERS)


def astropy_table_to_polars(table) -> pl.DataFrame:
    """Convert an astropy ``Table`` to a polars ``DataFrame`` via Arrow/NumPy.

    Multi-D columns are dropped (no polars equivalent) and ``bytes`` string
    columns are decoded to UTF-8. Masked values become nulls.
    """
    import pyarrow as pa

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

    pa_cols = {}
    for name in keep:
        col = table[name]
        raw_mask = getattr(col, "mask", None)
        mask = (
            np.asarray(raw_mask, dtype=bool)
            if raw_mask is not None and raw_mask is not np.ma.nomask and np.any(raw_mask)
            else None
        )
        arr = np.asarray(col)
        if arr.dtype.byteorder not in ("=", "|"):
            arr = arr.astype(arr.dtype.newbyteorder("="))
        if arr.dtype.kind == "S":
            arr = np.strings.decode(arr, "utf-8", errors="replace")
        elif arr.dtype.kind == "O" and len(arr):
            first = next((x for x in arr if x is not None and x is not np.ma.masked), None)
            if isinstance(first, (bytes, bytearray, np.bytes_)):
                arr = np.array(
                    [
                        x.decode("utf-8", errors="replace")
                        if isinstance(x, (bytes, bytearray, np.bytes_))
                        else x
                        for x in arr
                    ],
                    dtype=object,
                )
        pa_cols[name] = pa.array(arr, mask=mask)

    df = pl.DataFrame(pa.table(pa_cols))
    logger.debug(
        "Loaded astropy table via Arrow: %d rows, %d columns.",
        df.height,
        df.width,
    )
    return df


def polars_to_astropy(frame: FrameLike):
    """Convert a polars frame (eager or lazy) to an astropy ``Table``.

    Lazy frames are collected first. Nullable columns are converted to
    ``MaskedColumn`` instances.
    """
    from astropy.table import MaskedColumn, Table

    if isinstance(frame, pl.LazyFrame):
        frame = frame.collect()
    if frame.width == 0:
        return Table()

    cols = {}
    for s in frame:
        if s.null_count() > 0:
            mask = s.is_null().to_numpy()
            if s.dtype == pl.String:
                vals = np.asarray(s.fill_null("").to_list(), dtype=str)
            elif s.dtype == pl.Boolean:
                vals = s.fill_null(False).to_numpy()
            elif s.dtype.is_integer():
                vals = s.fill_null(0).to_numpy()
            elif s.dtype.is_float():
                vals = s.fill_null(np.nan).to_numpy()
            else:
                vals = np.asarray(s.to_list())
            cols[s.name] = MaskedColumn(vals, name=s.name, mask=mask)
        else:
            if s.dtype == pl.String:
                vals = np.asarray(s.to_list(), dtype=str)
            else:
                vals = s.to_numpy()
            cols[s.name] = vals
    return Table(cols)


def _read_fits(path: Path) -> pl.DataFrame:
    """Read a local FITS table with Torchfits, falling back to Astropy.

    Torchfits keeps the local catalogue path Arrow/Polars-native. Astropy
    serves as the fallback for FITS variants not supported by Torchfits or
    when the optional ``torchfits`` extra is not installed.
    """
    last_err = None
    try:
        from torchfits import table as torchfits_table
    except ImportError:
        torchfits_table = None

    if torchfits_table is not None:
        for hdu in (1, 0):
            try:
                return torchfits_table.read_polars(str(path), hdu=hdu).frame
            except Exception as exc:  # try next HDU, then Astropy fallback
                last_err = exc
        logger.debug("Torchfits could not read %s (%s); falling back to Astropy.", path, last_err)

    from astropy.table import Table

    for hdu in (1, 0):
        try:
            table = Table.read(path, hdu=hdu)
            return astropy_table_to_polars(table)
        except Exception as exc:  # try next HDU
            last_err = exc
    raise InputError(f"Could not read a table from FITS file {path}: {last_err}")


def scan_frame(path: str | Path) -> pl.LazyFrame:
    """Return a ``LazyFrame`` for a local catalogue file.

    Parquet and CSV are scanned lazily (projection/predicate push-down). FITS
    reads are eager and then exposed as a lazy frame; they use optional
    Torchfits first and Astropy as a fallback.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        return pl.scan_parquet(p)
    if suffix == ".csv":
        return pl.scan_csv(p)
    if suffix in (".tsv", ".tab"):
        return pl.scan_csv(p, separator="\t")
    if suffix in (".fits", ".fit"):
        return _read_fits(p).lazy()
    raise InputError(f"Unsupported file format '{suffix}'. Supported: {sorted(SUPPORTED_SUFFIXES)}")


def frame_columns(frame: FrameLike) -> list[str]:
    """Return column names without materialising data."""
    if isinstance(frame, pl.LazyFrame):
        return frame.collect_schema().names()
    return frame.columns


def to_lazy(frame: FrameLike) -> pl.LazyFrame:
    return frame.lazy() if isinstance(frame, pl.DataFrame) else frame


def write_hats(
    frame: FrameLike,
    output_dir: str | Path,
    ra_column: str = "ra",
    dec_column: str = "dec",
    threshold: int = 100_000,
) -> None:
    """Write a frame as a HATS catalogue directory.

    Partitions the catalogue into an adaptive hierarchical HEALPix tiling
    scheme and writes one Parquet file per pixel, making the catalogue
    efficient for spatial queries on massive datasets.

    Parameters
    ----------
    frame:
        Polars DataFrame or LazyFrame to write.
    output_dir:
        Destination directory for the HATS catalogue.
    ra_column:
        Right-ascension column name in ``frame``.
    dec_column:
        Declination column name in ``frame``.
    threshold:
        Maximum rows per HEALPix pixel (default 100 000).

    Raises
    ------
    CrossMatchError
        If ``cdshealpix`` is not installed.
    """
    try:
        import cdshealpix  # noqa: F401
    except ImportError as exc:
        from .exceptions import CrossMatchError

        raise CrossMatchError(
            "HATS output requires the optional 'cdshealpix' package. "
            "Install it with `pip install xmatcher[hats]`."
        ) from exc

    if isinstance(frame, pl.LazyFrame):
        frame = frame.collect()

    from .mirror import _write_hats_native

    _write_hats_native(
        frame,
        Path(output_dir),
        ra_column=ra_column,
        dec_column=dec_column,
        threshold=threshold,
    )
    logger.info(
        "Wrote HATS catalogue to %s (%d rows, threshold=%d)",
        output_dir,
        frame.height,
        threshold,
    )


def _write_frame_vospace(
    frame: FrameLike,
    output_file: str,
    *,
    ra_column: str,
    dec_column: str,
    hats_threshold: int,
) -> None:
    """Write a frame to a ``vos:`` node: VOSpace has no POSIX path, so the
    frame is written to a local staging area and uploaded through the storage
    layer (single files via ``stage_out``; HATS trees file-by-file via
    :func:`xmatcher.mirror._replicate_tree`).
    """
    import shutil
    import tempfile

    from .storage import LocalStorage, open_storage

    root, _, rel = str(output_file).rpartition("/")
    if not root or not rel:
        raise InputError(f"Invalid vos: output '{output_file}' (need a container and a name).")
    storage = open_storage(root)
    suffix = Path(rel).suffix.lower()
    if suffix == ".hats":
        staging = Path(tempfile.mkdtemp(prefix="xmatcher-hats-vos-"))
        try:
            write_hats(
                frame,
                staging / Path(rel).name,
                ra_column=ra_column,
                dec_column=dec_column,
                threshold=hats_threshold,
            )
            from .mirror import _replicate_tree

            _replicate_tree(LocalStorage(str(staging)), storage, "")
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        logger.info("Wrote HATS catalogue to %s", output_file)
        return

    staging = Path(tempfile.mkdtemp(prefix="xmatcher-out-vos-"))
    try:
        local = staging / Path(rel).name
        lf = to_lazy(frame)
        if suffix == ".parquet":
            lf.sink_parquet(local, engine="streaming")
        elif suffix == ".csv":
            lf.sink_csv(local, engine="streaming")
        elif suffix in (".tsv", ".tab"):
            lf.sink_csv(local, separator="\t")
        elif suffix in (".fits", ".fit"):
            polars_to_astropy(lf).write(local, overwrite=True)
        else:
            raise InputError(f"Unsupported output format '{suffix}'.")
        storage.stage_out(local, rel)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    logger.info("Wrote results to %s", output_file)


def write_frame(
    frame: FrameLike,
    output_file: str | Path,
    *,
    ra_column: str = "ra",
    dec_column: str = "dec",
    hats_threshold: int = 100_000,
) -> None:
    """Write a frame to ``.parquet`` / ``.csv`` / ``.fits`` / ``.hats`` based on the suffix.

    Parquet and CSV are streamed via ``sink_*`` with the polars streaming engine
    so large results never need to be fully materialised in memory.

    When the suffix is ``.hats`` the result is written as a HATS catalogue
    directory (requires ``cdshealpix``).  ``ra_column`` and ``dec_column`` identify
    the spatial columns for partitioning; ``hats_threshold`` controls the
    maximum rows per HEALPix pixel.

    A ``vos:`` output (e.g. ``vos:hats/xmatcher/full.hats``) is staged locally
    and uploaded through the storage layer.
    """
    if isinstance(output_file, str) and output_file.startswith("vos:"):
        _write_frame_vospace(
            frame,
            output_file,
            ra_column=ra_column,
            dec_column=dec_column,
            hats_threshold=hats_threshold,
        )
        return
    out = Path(output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    suffix = out.suffix.lower()
    lf = to_lazy(frame)

    if suffix == ".parquet":
        lf.sink_parquet(out, engine="streaming")
    elif suffix == ".csv":
        lf.sink_csv(out, engine="streaming")
    elif suffix in (".tsv", ".tab"):
        lf.sink_csv(out, separator="\t")
    elif suffix in (".fits", ".fit"):
        polars_to_astropy(lf).write(out, overwrite=True)
    elif suffix == ".hats":
        write_hats(frame, out, ra_column=ra_column, dec_column=dec_column, threshold=hats_threshold)
        return
    else:
        raise InputError(f"Unsupported output format '{suffix}'.")
    logger.info("Wrote results to %s", out)
