"""Tests for HATS/LSDB integration (mocked — no actual lsdb required)."""

from unittest import mock

import polars as pl
import pytest

from xmatch import CatalogueSource, CrossMatch, MatchSpec, hats_source, io_utils
from xmatch.exceptions import CrossMatchError
from xmatch.request import MatchRequest


# ---------------------------------------------------------------------------
# lsdb availability helpers
# ---------------------------------------------------------------------------
def test_lsdb_available_false():
    """lsdb_available() returns False when lsdb is not installed."""
    assert hats_source.lsdb_available() is False


def test_require_lsdb_raises_with_helpful_message():
    """_require_lsdb() raises CrossMatchError with pip install hint."""
    with pytest.raises(CrossMatchError, match="pip install lsdb"):
        hats_source._require_lsdb()


# ---------------------------------------------------------------------------
# read_hats
# ---------------------------------------------------------------------------
def test_read_hats_no_path():
    """read_hats raises when source has no path and no access_identifier."""
    src = CatalogueSource(name="test", is_local=False)
    with (
        mock.patch.object(hats_source, "_require_lsdb", return_value=mock.MagicMock()),
        pytest.raises(CrossMatchError, match="No HATS path"),
    ):
        hats_source.read_hats(src)


# ---------------------------------------------------------------------------
# hats_crossmatch — validation before lsdb call
# ---------------------------------------------------------------------------
def test_hats_crossmatch_unsupported_join_type_raises():
    """hats_crossmatch rejects join types other than '1and2'."""
    src1 = CatalogueSource(
        name="hats1", is_local=False, access_method="hats", access_identifier="/tmp/fake"
    )
    src2 = CatalogueSource(
        name="hats2", is_local=False, access_method="hats", access_identifier="/tmp/fake"
    )
    # patch _require_lsdb to return a dummy so we get past the import gate
    with (
        mock.patch.object(hats_source, "_require_lsdb", return_value=mock.MagicMock()),
        pytest.raises(CrossMatchError, match="only supports join_type='1and2'"),
    ):
        hats_source.hats_crossmatch(
            src1,
            src2,
            MatchSpec(radius_arcsec=1.0, join_type="all"),
        )


def test_hats_target_epoch_fails_instead_of_matching_unpropagated_coordinates():
    src1 = CatalogueSource(
        name="hats1", is_local=False, access_method="hats", access_identifier="/tmp/fake"
    )
    src2 = CatalogueSource(
        name="hats2", is_local=False, access_method="hats", access_identifier="/tmp/fake"
    )
    request = MatchRequest("hats1", "hats2", spec=MatchSpec(radius_arcsec=1.0, target_epoch=2025.0))

    with pytest.raises(CrossMatchError, match="target-epoch propagation is not yet supported"):
        CrossMatch()._dispatch(src1, src2, request)


# ---------------------------------------------------------------------------
# hats_crossmatch — full mocked integration
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_lsdb():
    """Return a mock lsdb module with enough surface for hats_crossmatch."""
    lsdb = mock.MagicMock()

    # mock lsdb.read_hats(path) → a MagicMock catalog
    lsdb.read_hats = mock.MagicMock(return_value=_make_fake_catalog())

    # mock lsdb.from_dataframe(pdf, ra_column=..., dec_column=...) → a MagicMock cat
    lsdb.from_dataframe = mock.MagicMock(return_value=_make_fake_catalog())
    return lsdb


def _make_fake_catalog():
    """Return a MagicMock catalog that computes to the fake result."""
    cat = mock.MagicMock()
    cat.crossmatch.return_value.compute.return_value = _make_fake_result_df()
    return cat


def _make_fake_result_df():
    """Return a minimal pandas DataFrame mimicking an LSDB crossmatch result."""
    import pandas as pd

    return pd.DataFrame(
        {
            "ra": [10.0, 20.0],
            "dec": [5.0, 6.0],
            "mag": [15.0, 16.0],
            "ra_2": [10.0001, 20.0001],
            "dec_2": [5.0001, 6.0001],
            "mag_2": [15.1, 16.1],
            "_dist_arcsec": [0.5, 0.7],
        }
    )


def test_hats_crossmatch_passes_n_neighbors_best(fake_lsdb):
    """find='best' → LSDB's n_neighbors=1."""
    src1 = CatalogueSource(
        name="hats1",
        is_local=False,
        access_method="hats",
        access_identifier="/tmp/fake",
        ra_column="ra",
        dec_column="dec",
    )
    src2 = CatalogueSource(
        name="hats2",
        is_local=False,
        access_method="hats",
        access_identifier="/tmp/fake2",
        ra_column="ra",
        dec_column="dec",
    )

    with mock.patch.object(hats_source, "_require_lsdb", return_value=fake_lsdb):
        result = hats_source.hats_crossmatch(
            src1,
            src2,
            MatchSpec(radius_arcsec=1.0, find="best"),
        )

    # Assert LSDB was called with correct parameters.
    # Both sides are HATS, so read_hats is called twice.
    assert fake_lsdb.read_hats.call_count == 2

    # Get the first catalog's crossmatch call args
    cat1 = fake_lsdb.read_hats.return_value
    cat1.crossmatch.assert_called_once()
    _, kwargs = cat1.crossmatch.call_args
    assert kwargs["radius_arcsec"] == 1.0
    assert kwargs["n_neighbors"] == 1
    assert kwargs["suffixes"] == ("", "_2")

    # Result is a polars DataFrame with sep_arcsec (renamed from _dist_arcsec)
    assert isinstance(result, pl.DataFrame)
    assert result.height == 2
    assert "sep_arcsec" in result.columns
    assert "_dist_arcsec" not in result.columns


def test_hats_crossmatch_passes_n_neighbors_all(fake_lsdb):
    """find='all' → LSDB's n_neighbors=None (unlimited)."""
    src1 = CatalogueSource(
        name="hats1",
        is_local=False,
        access_method="hats",
        access_identifier="/tmp/fake",
        ra_column="ra",
        dec_column="dec",
    )
    src2 = CatalogueSource(
        name="hats2",
        is_local=False,
        access_method="hats",
        access_identifier="/tmp/fake2",
        ra_column="ra",
        dec_column="dec",
    )

    with mock.patch.object(hats_source, "_require_lsdb", return_value=fake_lsdb):
        hats_source.hats_crossmatch(
            src1,
            src2,
            MatchSpec(radius_arcsec=1.0, find="all"),
        )

    cat1 = fake_lsdb.read_hats.return_value
    _, kwargs = cat1.crossmatch.call_args
    assert kwargs["n_neighbors"] is None


def test_hats_crossmatch_find_all_multi_row_rename(fake_lsdb):
    """find='all' with multiple matches per left row → _dist_arcsec renamed to
    sep_arcsec correctly on every row, and row count exceeds left input count."""
    import pandas as pd

    # 2 left stars: star-A matches 2 right stars, star-B matches 1.
    cat_multi = mock.MagicMock()
    cat_multi.crossmatch.return_value.compute.return_value = pd.DataFrame(
        {
            "ra": [10.0, 10.0, 20.0],  # left star-A (×2), star-B
            "dec": [5.0, 5.0, 6.0],
            "mag": [15.0, 15.0, 16.0],
            "ra_2": [10.0001, 10.0002, 20.0001],
            "dec_2": [5.0001, 5.0002, 6.0001],
            "mag_2": [15.1, 15.2, 16.1],
            "_dist_arcsec": [0.5, 0.9, 0.7],
        }
    )
    fake_lsdb.read_hats = mock.MagicMock(return_value=cat_multi)

    src1 = CatalogueSource(
        name="hats1",
        is_local=False,
        access_method="hats",
        access_identifier="/tmp/fake",
        ra_column="ra",
        dec_column="dec",
    )
    src2 = CatalogueSource(
        name="hats2",
        is_local=False,
        access_method="hats",
        access_identifier="/tmp/fake2",
        ra_column="ra",
        dec_column="dec",
    )

    with mock.patch.object(hats_source, "_require_lsdb", return_value=fake_lsdb):
        result = hats_source.hats_crossmatch(
            src1,
            src2,
            MatchSpec(radius_arcsec=1.0, find="all"),
        )

    # 3 output rows (2 matches for star-A + 1 for star-B)
    assert isinstance(result, pl.DataFrame)
    assert result.height == 3

    # Every row has sep_arcsec; _dist_arcsec is gone.
    assert "sep_arcsec" in result.columns
    assert "_dist_arcsec" not in result.columns
    assert list(result["sep_arcsec"].to_list()) == [0.5, 0.9, 0.7]

    # Left-star columns are duplicated across multi-match rows.
    assert list(result["ra"].to_list()) == [10.0, 10.0, 20.0]

    # LSDB was called with n_neighbors=None (unlimited).
    _, kwargs = cat_multi.crossmatch.call_args
    assert kwargs["n_neighbors"] is None


def test_hats_crossmatch_custom_right_suffix(fake_lsdb):
    """right_suffix='_3' → LSDB suffixes=('', '_3')."""
    src1 = CatalogueSource(
        name="hats1",
        is_local=False,
        access_method="hats",
        access_identifier="/tmp/fake",
        ra_column="ra",
        dec_column="dec",
    )
    src2 = CatalogueSource(
        name="hats2",
        is_local=False,
        access_method="hats",
        access_identifier="/tmp/fake2",
        ra_column="ra",
        dec_column="dec",
    )

    with mock.patch.object(hats_source, "_require_lsdb", return_value=fake_lsdb):
        hats_source.hats_crossmatch(
            src1,
            src2,
            MatchSpec(radius_arcsec=1.0),
            right_suffix="_3",
        )

    cat1 = fake_lsdb.read_hats.return_value
    _, kwargs = cat1.crossmatch.call_args
    assert kwargs["suffixes"] == ("", "_3")


def test_hats_crossmatch_warns_on_missing_dist_arcsec(fake_lsdb, caplog):
    """When LSDB result lacks _dist_arcsec, a warning is logged."""
    src1 = CatalogueSource(
        name="hats1",
        is_local=False,
        access_method="hats",
        access_identifier="/tmp/fake",
        ra_column="ra",
        dec_column="dec",
    )
    src2 = CatalogueSource(
        name="hats2",
        is_local=False,
        access_method="hats",
        access_identifier="/tmp/fake2",
        ra_column="ra",
        dec_column="dec",
    )

    # Override fake_lsdb.read_hats to return a catalog without _dist_arcsec.
    import pandas as pd

    cat_no_dist = mock.MagicMock()
    cat_no_dist.crossmatch.return_value.compute.return_value = pd.DataFrame(
        {
            "ra": [10.0],
            "dec": [5.0],
        }
    )
    fake_lsdb.read_hats = mock.MagicMock(return_value=cat_no_dist)

    with mock.patch.object(hats_source, "_require_lsdb", return_value=fake_lsdb):
        hats_source.hats_crossmatch(
            src1,
            src2,
            MatchSpec(radius_arcsec=1.0),
        )

    assert "missing expected '_dist_arcsec'" in caplog.text


def test_hats_crossmatch_with_local_left_frame(fake_lsdb):
    """A non-HATS left catalogue is converted via lsdb.from_dataframe."""
    src1 = CatalogueSource(name="local", is_local=True, ra_column="ra", dec_column="dec")
    src2 = CatalogueSource(
        name="hats",
        is_local=False,
        access_method="hats",
        access_identifier="/tmp/fake",
        ra_column="ra",
        dec_column="dec",
    )

    local_lf = pl.DataFrame(
        {
            "ra": [10.0, 20.0],
            "dec": [5.0, 6.0],
            "mag": [15.0, 16.0],
        }
    ).lazy()

    with mock.patch.object(hats_source, "_require_lsdb", return_value=fake_lsdb):
        hats_source.hats_crossmatch(
            src1,
            src2,
            MatchSpec(radius_arcsec=1.0),
            local_lf1=local_lf,
        )

    # from_dataframe was called for the local side
    fake_lsdb.from_dataframe.assert_called_once()
    _, kwargs = fake_lsdb.from_dataframe.call_args
    assert kwargs["ra_column"] == "ra"
    assert kwargs["dec_column"] == "dec"
    # read_hats called once for the HATS side
    fake_lsdb.read_hats.assert_called_once()


def test_hats_crossmatch_warns_on_prior_columns(fake_lsdb, caplog):
    """Bayesian prior_columns trigger a warning (unsupported via LSDB)."""
    src1 = CatalogueSource(
        name="hats1",
        is_local=False,
        access_method="hats",
        access_identifier="/tmp/fake",
        ra_column="ra",
        dec_column="dec",
    )
    src2 = CatalogueSource(
        name="hats2",
        is_local=False,
        access_method="hats",
        access_identifier="/tmp/fake2",
        ra_column="ra",
        dec_column="dec",
    )

    with mock.patch.object(hats_source, "_require_lsdb", return_value=fake_lsdb):
        hats_source.hats_crossmatch(
            src1,
            src2,
            MatchSpec(radius_arcsec=1.0, prior_columns=["phot_g_mean_mag"]),
        )

    assert "probabilistic qualification" in caplog.text


# ---------------------------------------------------------------------------
# HATS source resolution via CrossMatch
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# crossmatch_multi — HATS routing at position 3+
# ---------------------------------------------------------------------------
def test_crossmatch_multi_hats_at_position_3_routes_via_hats_crossmatch(cm, tmp_path):
    """crossmatch_multi routes a HATS catalogue at position 3 to hats_crossmatch."""
    # Create HATS dirs for cat1 and cat3
    hats1 = tmp_path / "hats_cat1"
    hats1.mkdir()
    (hats1 / "properties").write_text("hats")

    hats3 = tmp_path / "hats_cat3"
    hats3.mkdir()
    (hats3 / "properties").write_text("hats")

    # Cat 2 is a local CSV
    local2_path = tmp_path / "local2.csv"
    pl.DataFrame({"ra": [10.0, 20.0], "dec": [5.0, 6.0]}).write_csv(local2_path)

    # Mock hats_crossmatch to avoid needing actual LSDB.
    # The first _dispatch call routes to hats_crossmatch (cat1=HATS, cat2=local).
    # Both the first-pair and the position-3 HATS branch call hats_crossmatch.
    with (
        mock.patch.object(hats_source, "_require_lsdb", return_value=mock.MagicMock()),
        mock.patch.object(
            hats_source,
            "hats_crossmatch",
            return_value=pl.DataFrame(
                {
                    "ra": [10.0, 20.0],
                    "dec": [5.0, 6.0],
                    "ra_2": [10.0001, 20.0001],
                    "dec_2": [5.0001, 6.0001],
                    "sep_arcsec": [0.5, 0.7],
                }
            ),
        ) as mock_hats_xm,
    ):
        cm.crossmatch_multi(
            [str(hats1), str(local2_path), str(hats3)],
            radius_arcsec=1.0,
            ra=10.0,
            dec=5.0,
            radius_deg=0.01,
        )

    # hats_crossmatch should be called exactly twice:
    # 1st call: cat1 (HATS) × cat2 (local) → _dispatch → hats_crossmatch
    # 2nd call: accumulator × cat3 (HATS) → crossmatch_multi HATS branch
    assert mock_hats_xm.call_count == 2

    # Verify 2nd call uses right_suffix='_3'
    second_call_kwargs = mock_hats_xm.call_args_list[1].kwargs
    assert second_call_kwargs["right_suffix"] == "_3"


def test_crossmatch_two_hats_via_dispatch(cm, tmp_path):
    """Two HATS catalogues route through _dispatch → hats_crossmatch."""
    hats1 = tmp_path / "hats1"
    hats1.mkdir()
    (hats1 / "properties").write_text("hats")
    hats2 = tmp_path / "hats2"
    hats2.mkdir()
    (hats2 / "properties").write_text("hats")

    with mock.patch.object(
        hats_source,
        "hats_crossmatch",
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

    # _dispatch routes to hats_crossmatch once for the pair
    mock_hats_xm.assert_called_once()
    kwargs = mock_hats_xm.call_args.kwargs
    # Default right_suffix for first pair is "_2"
    assert kwargs["right_suffix"] == "_2"


# ---------------------------------------------------------------------------
# is_hats_dir — additional edge cases
# ---------------------------------------------------------------------------
def test_is_hats_dir_multiple_markers(tmp_path):
    """is_hats_dir detects any of the known HATS markers."""
    for marker in io_utils._HATS_MARKERS:
        d = tmp_path / f"test_{marker}"
        d.mkdir()
        (d / marker).write_text("x")
        assert io_utils.is_hats_dir(d), f"marker {marker!r} not detected"

    # Empty dir is not a HATS dir
    d = tmp_path / "empty"
    d.mkdir()
    assert not io_utils.is_hats_dir(d)
