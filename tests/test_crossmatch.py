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


def test_resolve_source_datalab_alias(cm):
    """Data Lab catalogue aliases resolve to TAP-backed CatalogueSource."""
    for alias, expected_name, expected_table, expect_ra, expect_dec in [
        ("nsc", "nsc_noao", "nsc_dr2.object", "ra", "dec"),
        ("des", "des_noao", "des_dr2.main", "ra", "dec"),
        ("smash", "smash_noao", "smash_dr2.object", "ra", "dec"),
        ("unwise", "unwise_noao", "unwise_dr1.object", "ra", "dec"),
        ("allwise_dl", "allwise_noao", "allwise.source", "ra", "dec"),
    ]:
        src = cm.resolve_source(alias, {})
        assert not src.is_local, f"{alias} should be remote"
        assert src.access_method == "tap", f"{alias} should use TAP"
        assert src.name == expected_name, f"{alias} → {expected_name}"
        assert src.access_identifier == expected_table
        assert src.archive == "noao_datalab"
        assert src.tap_url == "https://datalab.noirlab.edu/tap"
        assert src.ra_column == expect_ra, f"{alias}: RA={src.ra_column}, expected {expect_ra}"
        assert src.dec_column == expect_dec, f"{alias}: Dec={src.dec_column}, expected {expect_dec}"


def test_resolve_source_datalab_aliases_point_to_tractor_not_object(cm):
    """decals and ls_dr10 resolve to the tractor table (with photometry)."""
    for alias in ("decals", "decals_dr10", "desils", "ls_dr10"):
        src = cm.resolve_source(alias, {})
        assert src.access_identifier == "ls_dr10.tractor", (
            f"{alias} should resolve to tractor table, got {src.access_identifier}"
        )

    # The object table is accessible via decals_objects / ls_dr10_objects
    for alias in ("decals_objects", "ls_dr10_objects"):
        src = cm.resolve_source(alias, {})
        assert src.access_identifier == "ls_dr10.object", (
            f"{alias} should resolve to object table, got {src.access_identifier}"
        )


def test_resolve_source_unknown_raises(cm):
    with pytest.raises(InputError):
        cm.resolve_source("totally_unknown_thing", {})


def test_resolve_source_needs_coords_when_undetectable(cm):
    df = pl.DataFrame({"object_id": [1, 2, 3], "mag_g": [1.0, 2.0, 3.0]})
    src = cm.resolve_source(df, {})
    assert src.is_local
    assert src.ra_column is None
    assert src.dec_column is None
    src2 = cm.resolve_source(df, {"ra_column": "object_id", "dec_column": "mag_g"})
    assert src2.ra_column == "object_id"


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


def test_id_join_without_ra_dec_columns(cm):
    """id_join must succeed on tables whose columns are NOT RA/Dec-named."""
    a = pl.DataFrame({"object_id": [1, 2, 3], "mag_g": [1.0, 2.0, 3.0]})
    b = pl.DataFrame({"object_id": [2, 3, 4], "mag_r": [9.0, 8.0, 7.0]})
    out = cm.crossmatch(a, b, id_join=True, id_column_1="object_id", id_column_2="object_id")
    assert out.height == 2
    assert {"mag_g", "mag_r"}.issubset(out.columns)


def test_sky_match_still_errors_when_ra_dec_missing(cm):
    """A sky match on a table without RA/Dec-named columns must fail loudly."""
    from xmatch.exceptions import CrossMatchError as _Cme

    a = pl.DataFrame({"object_id": [1], "mag_g": [1.0]})
    b = pl.DataFrame({"object_id": [1], "mag_g": [1.0]})
    with pytest.raises((_Cme,)):
        cm.crossmatch(a, b)


# ----------------------------------------------------------------- did-you-mean
def test_suggest_returns_close_match(cm):
    """cm.suggest returns similar catalogue/alias names for a typo."""
    matches = cm.suggest("gaiaesa")
    assert "gaia_esa" in matches


def test_suggest_empty_for_unrelated_input(cm):
    """cm.suggest is empty when nothing is close enough (cutoff not met)."""
    assert cm.suggest("zzzqxqxqxqx") == []


def test_suggest_lowercases_argument(cm):
    """Catalogues are stored lowercase; the helper should too."""
    assert "gaia_esa" in cm.suggest("GAIAESA")
    assert "gaia_esa" in cm.suggest("GaiaEsa")


def test_input_error_carries_source(cm):
    """InputError raised from resolve_source carries the user input as `.source`."""
    from xmatch.exceptions import InputError

    with pytest.raises(InputError) as info:
        cm.resolve_source("gaia_typo_xyz", {})
    assert info.value.source == "gaia_typo_xyz"


# ------------------------------------------------------- multi-catalogue crossmatch
@pytest.fixture
def three_frames():
    """Three overlapping catalogues around the same region."""
    a = pl.DataFrame(
        {"id": [1, 2, 3, 4], "ra": [10.0, 20.0, 30.0, 40.0], "dec": [5.0, 6.0, 7.0, 8.0]}
    )
    b = pl.DataFrame(
        {
            "id": [10, 20, 30, 40],
            "ra": [10.00005, 20.00005, 30.00005, 40.00005],
            "dec": [5.0, 6.0, 7.0, 8.0],
        }
    )
    c = pl.DataFrame(
        {
            "id": [100, 200, 300],
            "ra": [10.0001, 20.0001, 30.0001],
            "dec": [5.0, 6.0, 7.0],
        }
    )
    return a, b, c


def test_crossmatch_multi_three_way_intersection(cm, three_frames):
    """3-way sequential crossmatch: A×B×C intersection should return 3 rows."""
    a, b, c = three_frames
    out = cm.crossmatch_multi([a, b, c], radius_arcsec=1.0, join_type="1and2")
    assert isinstance(out, pl.DataFrame)
    # All 3 of A match B, and those 3 also match C (at small radius)
    assert out.height == 3
    # Column naming: A cols unchanged; B cols get _2; C cols get _3.
    assert "ra" in out.columns       # from cat-1
    assert "ra_2" in out.columns     # from cat-2 (renamed)
    assert "ra_3" in out.columns     # from cat-3 (renamed)
    assert "sep_arcsec" in out.columns
    # Verify sep_arcsec appears exactly once (not duplicated from chain).
    assert out.columns.count("sep_arcsec") == 1


def test_crossmatch_multi_lazy(cm, three_frames):
    """multi-way with lazy=True returns a LazyFrame."""
    a, b, c = three_frames
    out = cm.crossmatch_multi([a, b, c], radius_arcsec=1.0, lazy=True)
    assert isinstance(out, pl.LazyFrame)
    assert out.collect().height == 3


def test_crossmatch_multi_writes_output(cm, three_frames, tmp_path):
    """multi-way with output_file returns None and writes to disk."""
    a, b, c = three_frames
    out_path = tmp_path / "multi.parquet"
    res = cm.crossmatch_multi([a, b, c], radius_arcsec=1.0, output_file=out_path)
    assert res is None
    assert pl.read_parquet(out_path).height == 3


def test_crossmatch_multi_requires_two_catalogues(cm):
    """crossmatch_multi with fewer than 2 catalogues raises CrossMatchError."""
    with pytest.raises(CrossMatchError, match="At least two"):
        cm.crossmatch_multi([pl.DataFrame({"ra": [1.0], "dec": [2.0]})])


def test_crossmatch_multi_four_way(cm):
    """4-way match with diminishing radius should filter progressively."""
    a = pl.DataFrame({"ra": [10.0, 20.0], "dec": [5.0, 6.0]})
    b = pl.DataFrame({"ra": [10.00005, 20.00005], "dec": [5.0, 6.0]})
    c = pl.DataFrame({"ra": [10.0001, 20.0001], "dec": [5.0, 6.0]})
    d = pl.DataFrame({"ra": [10.00015, 50.0], "dec": [5.0, 50.0]})
    out = cm.crossmatch_multi([a, b, c, d], radius_arcsec=1.0)
    # Only the first source star matches through all 4 catalogues
    assert out.height == 1


def test_crossmatch_multi_preserves_first_ra_dec(cm, three_frames):
    """The first catalogue's RA/Dec column names survive un-renamed through the chain."""
    a, b, c = three_frames
    out = cm.crossmatch_multi([a, b, c], radius_arcsec=1.0)
    # First catalogue's RA/Dec are preserved
    assert "ra" in out.columns
    assert "dec" in out.columns


# ----------------------------------------------------------- remote-first multi
@pytest.mark.slow
def test_crossmatch_multi_remote_first_catalogue(cm, tmp_path):
    """Multi-way crossmatch where catalogue 1 is a remote TAP catalogue
    (``gaia_esa``) and catalogues 2, 3 are local files.

    Exercises the ``first_src`` download guard in ``crossmatch_multi()`` and
    verifies that the remote source is materialised before the chain proceeds.
    """
    import urllib.error
    import urllib.request

    # --- network gate -------------------------------------------------------
    try:
        urllib.request.urlopen("https://gea.esac.esa.int/tap-server/tap/sync", timeout=5).read(1)
    except (urllib.error.URLError, TimeoutError, OSError):
        pytest.skip("ESA Gaia TAP endpoint is not reachable from this host.")

    # --- Vega region (bright, dense — guaranteed Gaia coverage) -------------
    ra_vega = 279.23473479
    dec_vega = 38.78368896
    cone_deg = 0.005  # ~18 arcsec — small enough for a fast TAP round-trip

    # --- local catalogue 2: a single star at Vega's position -----------------
    local2 = pl.DataFrame(
        {"id": [1], "ra": [ra_vega], "dec": [dec_vega]}
    )
    local2_path = tmp_path / "local2.csv"
    local2.write_csv(local2_path)

    # --- local catalogue 3: a star offset ~0.5" from Vega -------------------
    local3 = pl.DataFrame(
        {"id": [101], "ra": [ra_vega + 0.00014], "dec": [dec_vega]}
    )
    local3_path = tmp_path / "local3.csv"
    local3.write_csv(local3_path)

    # --- 3-way: gaia_esa × local2 × local3 ----------------------------------
    out = cm.crossmatch_multi(
        ["gaia_esa", str(local2_path), str(local3_path)],
        ra=ra_vega,
        dec=dec_vega,
        radius_deg=cone_deg,
        radius_arcsec=1.0,
        join_type="1and2",
    )

    assert isinstance(out, pl.DataFrame)
    assert out.height >= 1, "expected at least 1 three-way match at Vega"
    # Gaia (cat-1) columns: ra, dec, source_id, phot_g_mean_mag, …
    assert "ra" in out.columns
    assert "dec" in out.columns
    assert "source_id" in out.columns
    # Local cat-2 "id" — no collision with Gaia columns (ra, dec, source_id, …)
    assert "id" in out.columns
    # Local cat-3 "id" — collides with cat-2's "id" → _3 suffix
    assert "id_3" in out.columns
    # sep_arcsec appears exactly once
    assert out.columns.count("sep_arcsec") == 1
