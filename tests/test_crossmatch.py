import polars as pl
import pytest

from xmatch import CrossMatch
from xmatch.exceptions import ConfigError, CrossMatchError, InputError


@pytest.fixture
def cm():
    return CrossMatch()  # uses the bundled xmatch.yaml


# ------------------------------------------------------------------ config
def test_bundled_config_loads_and_validates(cm):
    assert "gaia_esa" in cm.catalogues_config
    assert cm.resolve_name("gaia") == "gaia_esa"


def test_get_catalogue_config_merges_service(cm):
    cfg = cm.get_catalogue_config("gaia_esa")
    assert cfg["_catalogue_name"] == "gaia_esa"
    assert cfg["access_method"] == "tap"  # inherited from the archive service
    assert cfg["ra_column"] == "ra"


def test_get_catalogue_config_unknown(cm):
    with pytest.raises(CrossMatchError, match="not found"):
        cm.get_catalogue_config("does_not_exist")


def test_validate_rejects_missing_keys(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("archives: {}\n")  # missing 'catalogues'
    with pytest.raises(ConfigError):
        CrossMatch(config_file=bad)


# ------------------------------------------------------------------ sources
def test_resolve_source_local_frame(cm):
    df = pl.DataFrame({"ra": [1.0], "dec": [2.0]})
    src = cm.resolve_source(df, {})
    assert src.is_local and src.ra_column == "ra" and src.dec_column == "dec"


def test_resolve_source_remote_alias(cm):
    src = cm.resolve_source("gaia", {})
    assert not src.is_local
    assert src.access_method == "tap"
    assert src.name == "gaia_esa"


def test_resolve_source_unknown_raises(cm):
    with pytest.raises(InputError):
        cm.resolve_source("totally_unknown_thing", {})


def test_resolve_source_needs_coords_when_undetectable(cm):
    df = pl.DataFrame({"x": [1.0], "y": [2.0]})
    with pytest.raises(InputError):
        cm.resolve_source(df, {})
    # explicit override works
    src = cm.resolve_source(df, {"ra_column": "x", "dec_column": "y"})
    assert src.ra_column == "x"


# --------------------------------------------------------------- crossmatch
@pytest.fixture
def two_frames():
    a = pl.DataFrame({"id": [1, 2, 3], "ra": [10.0, 20.0, 30.0], "dec": [5.0, 6.0, 7.0]})
    b = pl.DataFrame(
        {"id": [1, 2, 3], "ra": [10.00005, 20.5, 30.00002], "dec": [5.00005, 6.5, 7.00001]}
    )
    return a, b


def test_crossmatch_returns_eager_dataframe(cm, two_frames):
    a, b = two_frames
    out = cm.crossmatch(a, b, radius_arcsec=1.0)
    assert isinstance(out, pl.DataFrame)
    assert out.height == 2


def test_crossmatch_lazy_opt_in(cm, two_frames):
    a, b = two_frames
    out = cm.crossmatch(a, b, radius_arcsec=1.0, lazy=True)
    assert isinstance(out, pl.LazyFrame)
    assert out.collect().height == 2


def test_crossmatch_writes_output_file(cm, two_frames, tmp_path):
    a, b = two_frames
    out_path = tmp_path / "result.parquet"
    res = cm.crossmatch(a, b, radius_arcsec=1.0, output_file=out_path)
    assert res is None
    assert pl.read_parquet(out_path).height == 2


def test_crossmatch_id_join(cm):
    a = pl.DataFrame({"oid": [1, 2, 3], "ra": [10.0, 20.0, 30.0], "dec": [5.0, 6.0, 7.0]})
    b = pl.DataFrame({"oid": [2, 3, 4], "ra": [1.0, 2.0, 3.0], "dec": [1.0, 2.0, 3.0]})
    out = cm.crossmatch(a, b, id_join=True, id_column_1="oid", id_column_2="oid")
    assert out.height == 2


def test_crossmatch_id_join_missing_columns_errors(cm):
    a = pl.DataFrame({"ra": [10.0], "dec": [5.0]})
    b = pl.DataFrame({"ra": [10.0], "dec": [5.0]})
    with pytest.raises(CrossMatchError, match="id columns"):
        cm.crossmatch(a, b, id_join=True)
