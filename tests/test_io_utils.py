import sys
from types import SimpleNamespace
from unittest import mock

import polars as pl
import pytest

from xmatcher import io_utils
from xmatcher.exceptions import InputError


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
    io_utils.write_frame(sample, path)
    fake_table = mock.Mock()
    fake_table.read_polars.return_value = SimpleNamespace(frame=sample)
    monkeypatch.setitem(sys.modules, "torchfits", SimpleNamespace(table=fake_table))

    back = io_utils.scan_frame(path).collect()

    fake_table.read_polars.assert_called_once_with(str(path), hdu=1)
    assert back.equals(sample)


def test_fits_read_preserves_nulls_when_torchfits_loses_masks(tmp_path, monkeypatch):
    """A lossy Torchfits result must fall back to the FITS null masks."""
    source = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["ok", "", None],
            "mag": [12.3, 12.4, None],
            "count": [7, 8, None],
        }
    )
    path = tmp_path / "nullable.fits"
    io_utils.write_frame(source, path)

    # Emulate Torchfits exposing masked values as ordinary data and omitting mask metadata.
    lossy = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["ok", "", ""],
            "mag": [12.3, 12.4, float("nan")],
            "count": [7, 8, 9],
        }
    )
    fake_table = mock.Mock()
    fake_table.read_polars.return_value = SimpleNamespace(frame=lossy)
    monkeypatch.setitem(sys.modules, "torchfits", SimpleNamespace(table=fake_table))

    back = io_utils.scan_frame(path).collect()

    assert back["name"].to_list() == ["ok", "", None]
    assert back["mag"].to_list() == [12.3, 12.4, None]
    assert back["count"].to_list() == [7, 8, None]


def test_roundtrip_nullable_fits(tmp_path):
    """The default FITS writer and reader preserve masks and unmasked empty strings."""
    source = pl.DataFrame(
        {
            "count": [7, None],
            "name": ["", None],
            "mag": [12.3, None],
        }
    )
    path = tmp_path / "nullable.fits"
    io_utils.write_frame(source, path)

    back = io_utils.scan_frame(path).collect()

    assert back["count"].to_list() == [7, None]
    assert back["name"].to_list() == ["", None]
    assert back["mag"].to_list() == [12.3, None]


def test_roundtrip_integer_only_nullable_fits(tmp_path):
    """Serialized mask metadata preserves integer nulls without sentinel columns."""
    source = pl.DataFrame({"count": [7, None]})
    path = tmp_path / "integer-nullable.fits"
    io_utils.write_frame(source, path)

    back = io_utils.scan_frame(path).collect()

    assert back["count"].to_list() == [7, None]


def test_fits_read_preserves_external_tnull_without_other_null_markers(tmp_path, monkeypatch):
    """An external FITS TNULL column still triggers the Astropy fallback."""
    from astropy.io import fits
    from astropy.table import MaskedColumn, Table

    source = Table({"count": MaskedColumn([7, 8, 9], mask=[False, True, False])})
    path = tmp_path / "external-tnull.fits"
    source.write(path, format="fits", overwrite=True)
    with fits.open(path) as hdus:
        null_count = hdus[1].header["TNULL1"]

    fake_table = mock.Mock()
    fake_table.read_polars.return_value = SimpleNamespace(
        frame=pl.DataFrame({"count": [7, null_count, 9]})
    )
    monkeypatch.setitem(sys.modules, "torchfits", SimpleNamespace(table=fake_table))

    back = io_utils.scan_frame(path).collect()

    assert back["count"].to_list() == [7, None, 9]


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


def test_roundtrip_nullable_fits_with_torchfits_when_installed(tmp_path):
    """The real optional reader must preserve masked FITS cells end to end."""
    pytest.importorskip("torchfits")
    source = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["ok", "", None],
            "mag": [12.3, 12.4, None],
            "count": [7, 8, None],
        }
    )
    path = tmp_path / "nullable.fits"
    io_utils.write_frame(source, path)

    back = io_utils.scan_frame(path).collect()

    assert back["name"].to_list() == ["ok", "", None]
    assert back["mag"].to_list() == [12.3, 12.4, None]
    assert back["count"].to_list() == [7, 8, None]


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
    """Polars <-> astropy round-trip preserves schema via io_utils helpers."""
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


def test_astropy_table_to_polars_handles_masked_and_object_bytes():
    """Masked numeric/string/bytes columns and object-bytes columns convert cleanly."""
    import numpy as np
    from astropy.table import MaskedColumn, Table

    table = Table(
        {
            "id": MaskedColumn([1, 2, 3], mask=[False, True, False]),
            "mag": MaskedColumn([10.5, 12.0, 14.2], mask=[False, False, True]),
            "src": MaskedColumn([b"GDR3", b"Gaia", b"DESI"], mask=[False, True, False]),
            "obj_bytes": np.array([b"alpha\xff", b"beta", b"gamma"], dtype=object),
            "big_endian": np.array([1.5, 2.5, 3.5], dtype=">f8"),
        }
    )
    out = io_utils.astropy_table_to_polars(table)
    assert out["id"].to_list() == [1, None, 3]
    assert out["mag"].to_list() == [10.5, 12.0, None]
    assert out["src"].to_list() == ["GDR3", None, "DESI"]
    assert out["obj_bytes"].to_list() == ["alpha\ufffd", "beta", "gamma"]
    assert out["big_endian"].to_list() == [1.5, 2.5, 3.5]


def test_polars_to_astropy_nullable_columns():
    """Nullable Polars columns become MaskedColumn in Astropy Table."""
    from astropy.table import MaskedColumn

    df = pl.DataFrame(
        {
            "id": [1, None],
            "mag": [10.5, None],
            "name": ["GDR3", None],
            "flag": [True, None],
        }
    )
    table = io_utils.polars_to_astropy(df)
    for col in ("id", "mag", "name", "flag"):
        assert isinstance(table[col], MaskedColumn)
        assert list(table[col].mask) == [False, True]
    back = io_utils.astropy_table_to_polars(table)
    assert back["id"].to_list() == [1, None]
    assert back["mag"].to_list() == [10.5, None]
    assert back["name"].to_list() == ["GDR3", None]
    assert back["flag"].to_list() == [True, None]


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
# HATS write tests
# --------------------------------------------------------------------------- #


def test_write_frame_hats_suffix_routes_to_write_hats(tmp_path):
    """write_frame with .hats suffix dispatches to write_hats (mocked)."""
    sample = pl.DataFrame({"ra": [10.0, 20.0], "dec": [5.0, 10.0], "mag": [10.0, 12.0]})
    hats_dir = tmp_path / "result.hats"

    with mock.patch("xmatcher.io_utils.write_hats") as mock_write_hats:
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

    with mock.patch("xmatcher.io_utils.write_hats") as mock_write_hats:
        io_utils.write_frame(sample, hats_dir, hats_threshold=42_000)
        _, kwargs = mock_write_hats.call_args
        assert kwargs["threshold"] == 42_000


def test_write_frame_hats_custom_ra_dec(tmp_path):
    """write_frame threads ra_column/dec_column through to HATS output."""
    sample = pl.DataFrame({"alpha": [10.0], "delta": [5.0]})
    hats_dir = tmp_path / "custom.hats"

    with mock.patch("xmatcher.io_utils.write_hats") as mock_write_hats:
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
    """write_hats raises clean CrossMatchError when cdshealpix is missing."""
    import builtins

    sample = pl.DataFrame({"ra": [10.0], "dec": [5.0]})
    _original_import = builtins.__import__

    def _raise_on_cdshealpix(name, *args, **kwargs):
        if name == "cdshealpix":
            raise ImportError("No module named 'cdshealpix'")
        return _original_import(name, *args, **kwargs)

    with mock.patch("builtins.__import__", side_effect=_raise_on_cdshealpix):
        with pytest.raises(Exception) as exc_info:
            io_utils.write_hats(sample, "/tmp/fake.hats")
        assert "cdshealpix" in str(exc_info.value).lower()


def test_supported_suffixes_includes_hats():
    """.hats is a recognized output format."""
    assert ".hats" in io_utils.SUPPORTED_SUFFIXES


def test_write_hats_collects_lazy_frame(tmp_path):
    """write_hats collects a LazyFrame and writes a native HATS directory."""
    lf = pl.LazyFrame({"ra": [10.0, 20.0], "dec": [5.0, 10.0]})
    hats_dir = tmp_path / "lazy.hats"

    with (
        mock.patch.dict("sys.modules", {"cdshealpix": mock.MagicMock()}),
        mock.patch("xmatcher.mirror._write_hats_native") as mock_native,
    ):
        io_utils.write_hats(lf, hats_dir)
        call_args, kwargs = mock_native.call_args
        assert isinstance(call_args[0], pl.DataFrame)
        assert call_args[1] == hats_dir
        assert kwargs["ra_column"] == "ra"
        assert kwargs["dec_column"] == "dec"
