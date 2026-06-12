import polars as pl
import pytest

from xmatch import io_utils
from xmatch.exceptions import InputError


@pytest.fixture
def sample():
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
