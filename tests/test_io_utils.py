import polars as pl
import pytest

from xmatch import io_utils
from xmatch.exceptions import InputError


@pytest.fixture
def sample() -> pl.DataFrame:
    return pl.DataFrame({"id": [1, 2], "ra": [10.0, 20.0], "dec": [5.0, 6.0]})


def test_roundtrip_parquet(tmp_path, sample):
    path = tmp_path / "x.parquet"
    io_utils.write_frame(sample, path)
    back = io_utils.scan_frame(path).collect()
    assert back.sort("id").equals(sample.sort("id"))


def test_roundtrip_csv(tmp_path, sample):
    path = tmp_path / "x.csv"
    io_utils.write_frame(sample, path)
    assert io_utils.frame_columns(io_utils.scan_frame(path)) == ["id", "ra", "dec"]


def test_roundtrip_fits(tmp_path, sample):
    path = tmp_path / "x.fits"
    io_utils.write_frame(sample, path)
    back = io_utils.scan_frame(path).collect()
    assert set(back.columns) == {"id", "ra", "dec"}
    assert back.height == 2


def test_frame_columns_lazy_no_collect(tmp_path, sample):
    path = tmp_path / "x.parquet"
    io_utils.write_frame(sample, path)
    assert io_utils.frame_columns(io_utils.scan_frame(path)) == ["id", "ra", "dec"]


def test_unsupported_format(tmp_path):
    with pytest.raises(InputError):
        io_utils.scan_frame(tmp_path / "x.weird")


def test_is_hats_dir(tmp_path):
    assert not io_utils.is_hats_dir(tmp_path)
    (tmp_path / "properties").write_text("hats")
    assert io_utils.is_hats_dir(tmp_path)


def test_astropy_table_roundtrip(sample):
    table = io_utils.polars_to_astropy(sample)
    back = io_utils.astropy_table_to_polars(table)
    assert set(back.columns) == {"id", "ra", "dec"}


def test_arrow_roundtrip_preserves_schema(sample):
    """Polars <-> astropy round-trip preserves schema via io_utils helpers.

    Works on every astropy version: helpers go via Arrow when 7.x+ exposes
    ``Table.to_arrow`` and via pandas otherwise.
    """
    table = io_utils.polars_to_astropy(sample)
    back = io_utils.astropy_table_to_polars(table)
    assert back.columns == sample.columns
    assert back.height == sample.height
    assert back.dtypes == sample.dtypes


def test_bytes_columns_decode_to_utf8():
    """VOTable-style bytes columns become Utf8 in polars, not pl.Binary."""
    from astropy.table import Table

    table = Table({"src": [b"GDR3", b"Gaia"], "mag": [10.5, 12.3]})
    df = io_utils.astropy_table_to_polars(table)
    assert df.schema["src"] == pl.Utf8
    assert df["src"].to_list() == ["GDR3", "Gaia"]


def test_multi_d_columns_are_dropped():
    """Multi-dimensional columns are skipped entirely."""
    import numpy as np
    from astropy.table import Table

    t = Table({"vec": [np.array([[1.0, 2.0]]), np.array([[3.0, 4.0]])]})
    df = io_utils.astropy_table_to_polars(t)
    assert df.width == 0


def test_polars_to_astropy_accepts_lazyframe():
    """polars_to_astropy must collect a LazyFrame internally."""
    lf = pl.LazyFrame({"id": [1, 2], "ra": [10.0, 20.0], "dec": [5.0, 6.0]})
    table = io_utils.polars_to_astropy(lf)
    assert len(table) == 2
    assert set(table.colnames) == {"id", "ra", "dec"}
