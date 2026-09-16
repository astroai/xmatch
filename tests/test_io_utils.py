import sys
from types import SimpleNamespace
from unittest import mock

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


def test_roundtrip_tsv(tmp_path, sample):
    path = tmp_path / "x.tsv"
    io_utils.write_frame(sample, path)
    assert io_utils.frame_columns(io_utils.scan_frame(path)) == ["id", "ra", "dec"]
    back = io_utils.scan_frame(path).collect()
    assert back.sort("id").equals(sample.sort("id"))


def test_roundtrip_fits(tmp_path, sample):
    path = tmp_path / "x.fits"
    io_utils.write_frame(sample, path)
    back = io_utils.scan_frame(path).collect()
    assert set(back.columns) == {"id", "ra", "dec"}
    assert back.height == 2


def test_fits_read_prefers_torchfits_polars(tmp_path, sample, monkeypatch):
    """The optional Torchfits backend feeds its Polars frame through unchanged."""
    path = tmp_path / "x.fits"
    fake_table = mock.Mock()
    fake_table.read_polars.return_value = SimpleNamespace(frame=sample)
    monkeypatch.setitem(sys.modules, "torchfits", SimpleNamespace(table=fake_table))

    back = io_utils.scan_frame(path).collect()

    fake_table.read_polars.assert_called_once_with(str(path), hdu=1)
    assert back.equals(sample)


def test_roundtrip_fits_with_torchfits_when_installed(tmp_path, sample):
    """Exercise the real optional backend in environments that provide it."""
    torchfits = pytest.importorskip("torchfits")
    path = tmp_path / "x.fits"
    io_utils.write_frame(sample, path)
    with mock.patch.object(
        torchfits.table, "read_polars", wraps=torchfits.table.read_polars
    ) as read:
        back = io_utils.scan_frame(path).collect()

    read.assert_called_once_with(str(path), hdu=1)
    assert back.sort("id").equals(sample.sort("id"))


def test_fits_read_falls_back_to_astropy_when_torchfits_rejects(tmp_path, sample, monkeypatch):
    """Unsupported Torchfits input retains the established Astropy path."""
    path = tmp_path / "x.fits"
    io_utils.write_frame(sample, path)
    fake_table = mock.Mock()
    fake_table.read_polars.side_effect = RuntimeError("unsupported FITS variant")
    monkeypatch.setitem(sys.modules, "torchfits", SimpleNamespace(table=fake_table))

    back = io_utils.scan_frame(path).collect()

    assert fake_table.read_polars.call_count == 2
    assert back.sort("id").equals(sample.sort("id"))


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


def test_decode_pandas_bytes_columns_skips_non_object_columns():
    """Vectorised byte-column decode skips non-object columns up-front."""
    import pandas as pd

    pdf = pd.DataFrame(
        {
            "id": [1, 2, 3],  # int64 — skipped
            "mag": [10.5, 12.0, 14.2],  # float64 — skipped
            "src": [b"GDR3", b"Gaia", b"DESI"],  # object bytes — decoded
            "label": ["alpha", "beta", "gamma"],  # object str — left alone
            "epoch": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-03"]
            ),  # datetime — skipped
        }
    )
    out = io_utils._decode_pandas_bytes_columns(pdf.copy())
    assert out["src"].tolist() == ["GDR3", "Gaia", "DESI"]
    assert out["label"].tolist() == ["alpha", "beta", "gamma"]
    # Non-object columns must be unchanged.
    assert out["id"].dtype == pdf["id"].dtype
    assert out["mag"].dtype == pdf["mag"].dtype
    assert out["epoch"].dtype == pdf["epoch"].dtype


def test_decode_pandas_bytes_columns_empty_dataframe():
    """Empty DataFrame with object columns does not raise."""
    import pandas as pd

    pdf = pd.DataFrame({"src": pd.Series([], dtype=object)})
    out = io_utils._decode_pandas_bytes_columns(pdf)
    assert out["src"].tolist() == []


def test_decode_pandas_bytes_columns_all_non_bytes_objects():
    """Object columns that contain only strings skip decode cleanly."""
    import pandas as pd

    pdf = pd.DataFrame({"label": ["a", "b", "c"], "name": ["x", "y", "z"]})
    out = io_utils._decode_pandas_bytes_columns(pdf.copy())
    assert out["label"].tolist() == ["a", "b", "c"]
    assert out["name"].tolist() == ["x", "y", "z"]


def test_decode_pandas_bytes_columns_skips_string_dtype():
    """``exclude='str'`` skips pandas ``StringDtype`` columns cleanly.

    Pins the contract introduced by the pandas 4 migration fix: a column
    with dtype ``string`` (modern ``pd.StringDtype``) never carries bytes,
    so it is filtered out up-front and ``_decode_pandas_bytes_columns``
    leaves it untouched.

    Regression guard: a future pandas that narrows or widens what
    ``select_dtypes(exclude='str')`` matches will trip this test instead
    of silently decoding (or silently breaking) ``StringDtype`` columns.
    """
    import pandas as pd

    pdf = pd.DataFrame(
        {
            "src": [b"GDR3", b"Gaia"],  # object bytes — decoded
            "tag": pd.array(["x", "y"], dtype="string"),  # str — skipped
            "label": ["a", "b"],  # object str — left alone
        }
    )
    out = io_utils._decode_pandas_bytes_columns(pdf.copy())
    assert out["src"].tolist() == ["GDR3", "Gaia"]
    assert out["tag"].tolist() == ["x", "y"]  # untouched
    assert str(out["tag"].dtype) == "string"
    assert out["label"].tolist() == ["a", "b"]


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


# --------------------------------------------------------------------------- #
# HATS write tests (mocked — no actual lsdb required)
# --------------------------------------------------------------------------- #


def test_write_frame_hats_suffix_routes_to_write_hats(tmp_path):
    """write_frame with .hats suffix dispatches to write_hats (mocked)."""
    sample = pl.DataFrame({"ra": [10.0, 20.0], "dec": [5.0, 10.0], "mag": [10.0, 12.0]})
    hats_dir = tmp_path / "result.hats"

    with mock.patch("xmatch.io_utils.write_hats") as mock_write_hats:
        io_utils.write_frame(sample, hats_dir)
        mock_write_hats.assert_called_once()
        _, kwargs = mock_write_hats.call_args
        assert kwargs["ra_column"] == "ra"
        assert kwargs["dec_column"] == "dec"
        assert kwargs["threshold"] == 100_000


def test_write_frame_hats_respects_threshold(tmp_path):
    """write_frame passes hats_threshold through to write_hats."""
    sample = pl.DataFrame({"ra": [10.0], "dec": [5.0]})
    hats_dir = tmp_path / "big.hats"

    with mock.patch("xmatch.io_utils.write_hats") as mock_write_hats:
        io_utils.write_frame(sample, hats_dir, hats_threshold=42_000)
        _, kwargs = mock_write_hats.call_args
        assert kwargs["threshold"] == 42_000


def test_write_frame_hats_custom_ra_dec(tmp_path):
    """write_frame threads ra_column/dec_column through to HATS output."""
    sample = pl.DataFrame({"alpha": [10.0], "delta": [5.0]})
    hats_dir = tmp_path / "custom.hats"

    with mock.patch("xmatch.io_utils.write_hats") as mock_write_hats:
        io_utils.write_frame(
            sample,
            hats_dir,
            ra_column="alpha",
            dec_column="delta",
        )
        _, kwargs = mock_write_hats.call_args
        assert kwargs["ra_column"] == "alpha"
        assert kwargs["dec_column"] == "delta"


def test_write_hats_unavailable_raises():
    """write_hats raises clean CrossMatchError when lsdb is missing."""
    import builtins

    sample = pl.DataFrame({"ra": [10.0], "dec": [5.0]})
    _original_import = builtins.__import__

    def _raise_on_lsdb(name, *args, **kwargs):
        if name == "lsdb":
            raise ImportError("No module named 'lsdb'")
        return _original_import(name, *args, **kwargs)

    with mock.patch("builtins.__import__", side_effect=_raise_on_lsdb):
        with pytest.raises(Exception) as exc_info:
            io_utils.write_hats(sample, "/tmp/fake.hats")
        assert "lsdb" in str(exc_info.value).lower()


def test_supported_suffixes_includes_hats():
    """.hats is a recognized output format."""
    assert ".hats" in io_utils.SUPPORTED_SUFFIXES


def test_write_hats_collects_lazy_frame(tmp_path):
    """write_hats must collect LazyFrame before passing to lsdb."""
    lf = pl.LazyFrame({"ra": [10.0, 20.0], "dec": [5.0, 10.0]})
    hats_dir = tmp_path / "lazy.hats"
    mock_lsdb = mock.MagicMock()
    fake_catalog = mock_lsdb.from_dataframe.return_value

    with mock.patch.dict("sys.modules", {"lsdb": mock_lsdb}):
        io_utils.write_hats(lf, hats_dir)
        # from_dataframe received a collected (eager) DataFrame, not LazyFrame.
        call_args, _ = mock_lsdb.from_dataframe.call_args
        assert not isinstance(call_args[0], pl.LazyFrame)
        assert isinstance(call_args[0], pl.DataFrame)
        fake_catalog.to_hats.assert_called_once_with(str(hats_dir))
