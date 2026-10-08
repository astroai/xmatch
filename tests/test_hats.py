"""Tests for HATS source resolution and CrossMatch dispatch."""

from unittest import mock

import polars as pl
import pytest

from xmatcher import CrossMatch, hats_native, io_utils


@pytest.fixture
def cm():
    return CrossMatch()


def test_resolve_source_hats_dir(cm, tmp_path):
    """resolve_source detects a HATS directory via is_hats_dir."""
    hats_dir = tmp_path / "my_hats_cat"
    hats_dir.mkdir()
    (hats_dir / "properties").write_text("hats")

    src = cm.resolve_source(str(hats_dir), {})
    assert src.access_method == "hats"
    assert src.access_identifier == str(hats_dir)
    assert not src.is_local


def test_resolve_source_hats_dir_with_overrides(cm, tmp_path):
    """HATS dir resolution accepts ra/dec/id overrides."""
    hats_dir = tmp_path / "my_hats_cat"
    hats_dir.mkdir()
    (hats_dir / "properties").write_text("hats")

    src = cm.resolve_source(
        str(hats_dir),
        {
            "ra_column": "raj2000",
            "dec_column": "dej2000",
            "id_column": "source_id",
        },
    )
    assert src.ra_column == "raj2000"
    assert src.dec_column == "dej2000"
    assert src.id_column == "source_id"


def test_crossmatch_multi_hats_at_position_3_routes_via_hats_native(cm, tmp_path):
    """crossmatch_multi routes a HATS catalogue at position 3 to hats_native_crossmatch."""
    hats1 = tmp_path / "hats_cat1"
    hats1.mkdir()
    (hats1 / "properties").write_text("hats")

    hats3 = tmp_path / "hats_cat3"
    hats3.mkdir()
    (hats3 / "properties").write_text("hats")

    local2_path = tmp_path / "local2.csv"
    pl.DataFrame({"ra": [10.0, 20.0], "dec": [5.0, 6.0]}).write_csv(local2_path)

    with mock.patch.object(
        hats_native,
        "hats_native_crossmatch",
        return_value=pl.DataFrame(
            {
                "ra": [10.0, 20.0],
                "dec": [5.0, 6.0],
                "ra_2": [10.0001, 20.0001],
                "dec_2": [5.0001, 6.0001],
                "sep_arcsec": [0.5, 0.7],
            }
        ),
    ) as mock_hats_xm:
        cm.crossmatch_multi(
            [str(hats1), str(local2_path), str(hats3)],
            radius_arcsec=1.0,
            ra=10.0,
            dec=5.0,
            radius_deg=0.01,
        )

    assert mock_hats_xm.call_count == 2
    second_call_kwargs = mock_hats_xm.call_args_list[1].kwargs
    assert second_call_kwargs["right_suffix"] == "_3"


def test_crossmatch_two_hats_via_dispatch(cm, tmp_path):
    """Two HATS catalogues route through _dispatch -> hats_native_crossmatch."""
    hats1 = tmp_path / "hats1"
    hats1.mkdir()
    (hats1 / "properties").write_text("hats")
    hats2 = tmp_path / "hats2"
    hats2.mkdir()
    (hats2 / "properties").write_text("hats")

    with mock.patch.object(
        hats_native,
        "hats_native_crossmatch",
        return_value=pl.DataFrame(
            {
                "ra": [10.0],
                "dec": [5.0],
                "ra_2": [10.0001],
                "dec_2": [5.0001],
                "sep_arcsec": [0.5],
            }
        ),
    ) as mock_hats_xm:
        cm.crossmatch_multi(
            [str(hats1), str(hats2)],
            radius_arcsec=1.0,
        )

    mock_hats_xm.assert_called_once()
    kwargs = mock_hats_xm.call_args.kwargs
    assert kwargs["right_suffix"] == "_2"


def test_is_hats_dir_multiple_markers(tmp_path):
    """is_hats_dir detects any of the known HATS markers."""
    for marker in io_utils._HATS_MARKERS:
        d = tmp_path / f"test_{marker}"
        d.mkdir()
        (d / marker).write_text("x")
        assert io_utils.is_hats_dir(d), f"marker {marker!r} not detected"

    d = tmp_path / "empty"
    d.mkdir()
    assert not io_utils.is_hats_dir(d)
