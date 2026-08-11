from pathlib import Path

import polars as pl
import pytest

from xmatch import CrossMatch
from xmatch.exceptions import ConfigError, CrossMatchError, InputError


@pytest.fixture
def cm():
    return CrossMatch()  # uses the bundled xmatch.yaml


# ------------------------------------------------------------------ config
def test_bundled_config_loads_and_validates(cm):
    assert "gaia_cds" in cm.catalogues_config
    assert "gaia_esa" in cm.catalogues_config
    assert cm.resolve_name("gaia") == "gaia_cds"


def test_get_catalogue_config_merges_service(cm):
    cfg = cm.get_catalogue_config("gaia")
    assert cfg["_catalogue_name"] == "gaia_cds"
    assert cfg["access_method"] == "tap"  # inherited from the archive service
    assert cfg["ra_column"] == "RA_ICRS"
    esa = cm.get_catalogue_config("gaia_esa")
    assert esa["ra_column"] == "ra"


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
    assert src.name == "gaia_cds"


def test_gaia_source_exposes_complete_space_motion_metadata(cm):
    # `gaia` defaults to CDS VizieR (`I/355/gaiadr3`)
    src = cm.resolve_source("gaia", {})
    assert src.parallax_column == "Plx"
    assert src.radial_velocity_column == "RV"
    assert "Plx" in src.default_columns
    assert "RV" in src.default_columns
    assert src.astrometric_covariance_columns is None
    # the ESA mirror keeps the archive schema + covariance block
    esa = cm.resolve_source("gaia_esa", {})
    assert esa.parallax_column == "parallax"
    assert esa.astrometric_covariance_columns is not None
    assert esa.astrometric_covariance_columns["pmra_pmdec_corr"] == "pmra_pmdec_corr"


def test_local_source_preserves_space_motion_overrides(cm):
    frame = pl.DataFrame(
        {
            "ra": [1.0],
            "dec": [2.0],
            "pmra": [3.0],
            "pmdec": [4.0],
            "epoch": [2016.0],
            "parallax": [5.0],
            "rv": [6.0],
        }
    )
    src = cm.resolve_source(
        frame,
        {
            "pm_ra_column": "pmra",
            "pm_dec_column": "pmdec",
            "epoch_column": "epoch",
            "parallax_column": "parallax",
            "radial_velocity_column": "rv",
        },
    )
    assert src.pm_ra_column == "pmra"
    assert src.pm_dec_column == "pmdec"
    assert src.epoch_column == "epoch"
    assert src.parallax_column == "parallax"
    assert src.radial_velocity_column == "rv"


def test_resolve_source_datalab_alias(cm):
    """Data Lab catalogue aliases resolve to TAP-backed CatalogueSource."""
    for alias, expected_name, expected_table, expect_ra, expect_dec in [
        ("nsc", "nsc", "nsc_dr2.object", "ra", "dec"),
        ("des", "des", "des_dr2.main", "ra", "dec"),
        ("smash", "smash", "smash_dr2.object", "ra", "dec"),
        ("unwise", "unwise", "unwise_dr1.object", "ra", "dec"),
        ("allwise_dl", "allwise", "allwise.source", "ra", "dec"),
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
    assert "gaia" in matches


def test_suggest_empty_for_unrelated_input(cm):
    """cm.suggest is empty when nothing is close enough (cutoff not met)."""
    assert cm.suggest("zzzqxqxqxqx") == []


def test_suggest_lowercases_argument(cm):
    """Catalogues are stored lowercase; the helper should too."""
    assert "gaia" in cm.suggest("GAIAESA") or "gaia_esa" in cm.suggest("GAIAESA")
    assert "gaia" in cm.suggest("GaiaEsa") or "gaia_esa" in cm.suggest("GaiaEsa")


def test_suggest_preserves_original_case_with_mixed_case_pool():
    """Regression for palette/case-insensitive-fuzzy-matching-*.
    suggest() must tolerate mixed-case pool entries (e.g. ``Gaia_DR3``) and
    return suggestions in their **original** casing rather than the
    lowercased form picked up for matching.
    """
    from unittest import mock

    cm = CrossMatch()
    # Simulate a YAML edit that used mixed-case catalogue names; pool now
    # contains both the bundled lowercase names and a few mixed-case extras.
    mixed_pool = {
        "Gaia_DR3": "fake_config",
        "twoMASS_psc": "fake_config",
        "allWISE_src": "fake_config",
    }
    with (
        mock.patch.object(cm, "catalogues_config", mixed_pool),
        mock.patch.object(cm, "aliases_config", {}),
    ):
        # Mixed-case input: should still match and return ORIGINAL casing.
        out_upper = cm.suggest("GAIA_DR3")
        assert "Gaia_DR3" in out_upper
        # And should NOT echo the lowercase form.
        assert "gaia_dr3" not in out_upper

        out_mixed = cm.suggest("Gaia_dr3")
        assert "Gaia_DR3" in out_mixed

        out_close = cm.suggest("twomass_psc")
        assert "twoMASS_psc" in out_close


def test_suggest_empty_pool_returns_empty():
    """suggest() against an empty pool must return [], not raise."""
    from unittest import mock

    cm = CrossMatch()
    with (
        mock.patch.object(cm, "catalogues_config", {}),
        mock.patch.object(cm, "aliases_config", {}),
    ):
        assert cm.suggest("anything") == []
        assert cm.suggest("") == []


def test_suggest_case_insensitive_aliases(cm):
    """Aliases pool entries are matched case-insensitively too."""
    from unittest import mock

    aliases_mixed = {"DECaLS_DR10": "decals", "ls_DR10_Extra": "decals"}
    with (
        mock.patch.object(cm, "aliases_config", aliases_mixed),
        mock.patch.dict(cm.catalogues_config, {}, clear=True),
    ):
        out = cm.suggest("decals_dr10_extra")
        assert "ls_DR10_Extra" in out


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
    assert "ra" in out.columns  # from cat-1
    assert "ra_2" in out.columns  # from cat-2 (renamed)
    assert "ra_3" in out.columns  # from cat-3 (renamed)
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

    # --- Gaia DR3 source region (guaranteed Gaia coverage) -------------
    ra_vega = 279.24289761
    dec_vega = 38.78910887
    cone_deg = 0.005  # ~18 arcsec — small enough for a fast TAP round-trip

    # --- local catalogue 2: a single star at Vega's position -----------------
    local2 = pl.DataFrame({"id": [1], "ra": [ra_vega], "dec": [dec_vega]})
    local2_path = tmp_path / "local2.csv"
    local2.write_csv(local2_path)

    # --- local catalogue 3: a star offset ~0.5" from Vega -------------------
    local3 = pl.DataFrame({"id": [101], "ra": [ra_vega + 0.00014], "dec": [dec_vega]})
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


# ----------------------------------------------------------- union match
@pytest.fixture
def union_frames():
    """Catalogues with overlapping and non-overlapping sources."""
    a = pl.DataFrame({"id": [1, 2, 3], "ra": [10.0, 20.0, 30.0], "dec": [5.0, 6.0, 7.0]})
    b = pl.DataFrame(
        {
            "id": [10, 20],
            "ra": [10.00005, 20.00005],  # matches a[0] and a[1]
            "dec": [5.0, 6.0],
        }
    )
    c = pl.DataFrame(
        {
            "id": [100],
            "ra": [10.0001],  # matches a[0] and b[0]
            "dec": [5.0],
        }
    )
    return a, b, c


def test_union_match_two_catalogue(union_frames):
    """2-catalogue union should include all sources from both sides."""
    a, b, _ = union_frames
    cm = CrossMatch()
    out = cm.union_match([a, b], radius_arcsec=1.0)
    assert isinstance(out, pl.DataFrame)
    assert "_src_cats" in out.columns
    # a[0]↔b[0] match, a[1]↔b[1] match, a[2] unmatched. ALL b rows matched.
    # Full outer: 2 matched pairs + 1 a-only = 3 rows, _src_cats: {"1+2", "1"}
    src_cats = set(out["_src_cats"].to_list())
    assert "1+2" in src_cats, f"Expected matched rows, got {src_cats}"
    assert "1" in src_cats, f"Expected unmatched left row, got {src_cats}"
    assert out.height == 3


def test_union_match_two_catalogue_with_unmatched_right():
    """2-catalogue union where some right rows are unmatched → "2" in _src_cats."""
    a = pl.DataFrame({"ra": [10.0, 20.0], "dec": [5.0, 6.0]})
    b = pl.DataFrame({"ra": [10.00005, 50.0], "dec": [5.0, 5.0]})
    cm = CrossMatch()
    out = cm.union_match([a, b], radius_arcsec=1.0)
    assert isinstance(out, pl.DataFrame)
    assert "_src_cats" in out.columns
    # a[0]↔b[0] match → "1+2"; a[1] unmatched → "1"; b[1] unmatched → "2"
    src_cats = set(out["_src_cats"].to_list())
    assert src_cats == {"1+2", "1", "2"}, f"Expected all three categories, got {src_cats}"
    assert out.height == 3


def test_union_match_three_catalogue(union_frames):
    """3-catalogue union should include all sources with correct _src_cats."""
    a, b, c = union_frames
    cm = CrossMatch()
    out = cm.union_match([a, b, c], radius_arcsec=1.0)
    assert isinstance(out, pl.DataFrame)
    assert "_src_cats" in out.columns
    # Verify _src_cats values exist
    src_cats = out["_src_cats"].to_list()
    assert len(src_cats) == out.height
    # All values should be non-empty strings
    assert all(isinstance(v, str) and len(v) > 0 for v in src_cats)
    # Should include 3-catalogue matched rows (a[0] ↔ b[0] ↔ c[0])
    assert "1+2+3" in src_cats, f"Expected '1+2+3' in _src_cats, got {set(src_cats)}"


def test_union_match_output_file(union_frames, tmp_path):
    """union_match should stream to output file and return None."""
    a, b, _ = union_frames
    cm = CrossMatch()
    out_path = tmp_path / "union.parquet"
    res = cm.union_match([a, b], radius_arcsec=1.0, output_file=out_path)
    assert res is None
    result = pl.read_parquet(out_path)
    assert "_src_cats" in result.columns
    assert result.height >= 2


def test_union_match_lazy(union_frames):
    """union_match with lazy=True returns a LazyFrame."""
    a, b, _ = union_frames
    cm = CrossMatch()
    out = cm.union_match([a, b], radius_arcsec=1.0, lazy=True)
    assert isinstance(out, pl.LazyFrame)
    df = out.collect()
    assert "_src_cats" in df.columns


def test_union_match_all_unmatched_islands():
    """Catalogues with no spatial overlap should all appear as separate islands."""
    a = pl.DataFrame({"ra": [10.0], "dec": [5.0]})
    b = pl.DataFrame({"ra": [50.0], "dec": [5.0]})  # ~40 deg away → no match
    c = pl.DataFrame({"ra": [100.0], "dec": [5.0]})  # ~90 deg away → no match
    cm = CrossMatch()
    out = cm.union_match([a, b, c], radius_arcsec=1.0)
    assert out.height == 3  # one row from each catalogue
    src_cats = set(out["_src_cats"].to_list())
    assert src_cats == {"1", "2", "3"}, f"Expected isolated catalogues, got {src_cats}"


# ------------------------------------------------------------------- fof_match
@pytest.fixture
def fof_frames():
    """Three catalogues around the same sky region with overlapping sources.

    cat1 (primary): 4 sources — A, B, C, D at different positions.
    cat2: 3 sources — A', B' matching A and B; E is new.
    cat3: 2 sources — B'' matching B; F is new.

    FoF bundles expected:
        A  ↔ A'                   = bundle {A,  cat2-A'}
        B  ↔ B'  ↔ B''            = bundle {B,  cat2-B', cat3-B''}  (3-cat chain)
        C                         = bundle {C}  (isolated in primary)
        D  ↔ ?                    = bundle {D}  (no match in other catalogues)
        E  (cat2 only)            = NOT in output (no primary match)
        F  (cat3 only)            = NOT in output (no primary match)
    """
    a = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "ra": [10.0, 20.0, 30.0, 40.0],
            "dec": [5.0, 6.0, 7.0, 8.0],
        }
    )
    b = pl.DataFrame(
        {
            "id": [101, 102, 103],
            "ra": [10.00005, 20.00005, 50.0],  # matches A, B, and E is new
            "dec": [5.0, 6.0, 5.0],
        }
    )
    c = pl.DataFrame(
        {
            "id": [201, 202],
            "ra": [20.00005, 60.0],  # matches B, and F is new
            "dec": [6.0, 5.0],
        }
    )
    return a, b, c


def test_fof_match_two_catalogue_basic_transitive_closure():
    """Two catalogues with overlapping sources → bundles."""
    a = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "ra": [10.0, 20.0, 30.0],
            "dec": [5.0, 6.0, 7.0],
        }
    )
    b = pl.DataFrame(
        {
            "id": [101, 102, 103],
            "ra": [10.00005, 20.00005, 50.0],
            "dec": [5.0, 6.0, 5.0],
        }
    )
    cm = CrossMatch()
    out = cm.fof_match([a, b], radius_arcsec=1.0)
    assert isinstance(out, pl.DataFrame)
    assert "bundle_id" in out.columns
    assert "n_cats" in out.columns
    assert "_src_cats" in out.columns
    # A↔101 (bundle), B↔102 (bundle), C isolated, D[...] wait there are 3 primary
    # Primary sources: A(1)↔101, B(2)↔102, C(3) isolated → 3 bundles
    assert out.height == 3
    # Check _src_cats values
    src_cats = dict(zip(out["bundle_id"].to_list(), out["_src_cats"].to_list(), strict=False))
    assert "1+2" in src_cats.values(), f"Expected '1+2' bundles, got {src_cats}"
    assert "1" in src_cats.values(), f"Expected '1' (isolated) bundles, got {src_cats}"


def test_fof_match_three_catalogue_transitive_chain(fof_frames):
    """Three catalogues with transitive chain: A↔A' and B↔B'↔B'' → bundles."""
    a, b, c = fof_frames
    cm = CrossMatch()
    out = cm.fof_match([a, b, c], radius_arcsec=1.0)
    assert isinstance(out, pl.DataFrame)
    # Primary has 4 sources: A(1)↔101, B(2)↔102+201, C(3) isolated, D(4) isolated
    # So 4 bundles total.
    assert out.height == 4
    # Find the 3-catalogue bundle (B↔102↔201)
    src_cats_list = out["_src_cats"].to_list()
    assert "1+2+3" in src_cats_list, (
        f"Expected transitive 3-cat bundle '1+2+3', got {set(src_cats_list)}"
    )
    # Verify n_cats column
    n_cats_vals = out["n_cats"].to_list()
    assert max(n_cats_vals) >= 3, (
        f"Expected at least one bundle with n_cats=3, got max {max(n_cats_vals)}"
    )


def test_fof_match_output_file(fof_frames, tmp_path):
    """FoF with output_file writes to disk and returns None."""
    a, b, c = fof_frames
    cm = CrossMatch()
    out_path = tmp_path / "fof.parquet"
    res = cm.fof_match([a, b, c], radius_arcsec=1.0, output_file=out_path)
    assert res is None
    result = pl.read_parquet(out_path)
    assert "bundle_id" in result.columns
    assert "_src_cats" in result.columns
    assert result.height == 4


def test_fof_match_no_matches_returns_isolated_bundles():
    """When no catalogue overlaps, each primary source becomes an isolated
    single-catalogue bundle (one row per primary source)."""
    a = pl.DataFrame({"ra": [10.0], "dec": [5.0]})
    b = pl.DataFrame({"ra": [50.0], "dec": [5.0]})  # far away
    cm = CrossMatch()
    out = cm.fof_match([a, b], radius_arcsec=1.0)
    assert isinstance(out, pl.DataFrame)
    # One bundle per primary source, each isolated.
    assert out.height == 1
    assert out["bundle_id"][0] == 0
    assert out["n_cats"][0] == 1
    assert out["_src_cats"][0] == "1"


def test_fof_match_requires_two_catalogues():
    """fof_match with <2 catalogues raises CrossMatchError."""
    cm = CrossMatch()
    with pytest.raises(CrossMatchError, match="at least 2"):
        cm.fof_match([pl.DataFrame({"ra": [1.0], "dec": [2.0]})])


def test_fof_match_all_primary_isolated():
    """All primary sources are isolated → one bundle per primary source."""
    a = pl.DataFrame(
        {
            "ra": [10.0, 20.0, 30.0],
            "dec": [5.0, 6.0, 7.0],
        }
    )
    b = pl.DataFrame(
        {
            "ra": [50.0, 60.0, 70.0],
            "dec": [5.0, 6.0, 7.0],
        }
    )
    c = pl.DataFrame(
        {
            "ra": [100.0, 110.0],
            "dec": [5.0, 6.0],
        }
    )
    cm = CrossMatch()
    out = cm.fof_match([a, b, c], radius_arcsec=1.0)
    assert out.height == 3  # one per primary source
    # All bundles should be single-catalogue
    assert all(v == 1 for v in out["n_cats"].to_list())
    assert all(v == "1" for v in out["_src_cats"].to_list())


def test_fof_match_preserves_primary_columns(fof_frames):
    """Primary catalogue's column names should survive un-renamed."""
    a, b, c = fof_frames
    cm = CrossMatch()
    out = cm.fof_match([a, b, c], radius_arcsec=1.0)
    # Primary (cat-1) columns should be present without suffix
    assert "id" in out.columns
    assert "ra" in out.columns
    assert "dec" in out.columns
    # Cat-2/3 columns should have suffixes when colliding
    assert "id_2" in out.columns or "id" in out.columns


def test_fof_match_column_averaging_and_strings():
    """When multiple cat-2 sources match the same primary source, numeric
    columns are averaged and string columns take the first value."""
    a = pl.DataFrame(
        {
            "ra": [10.0],
            "dec": [5.0],
            "name": ["primary_star"],
        }
    )
    b = pl.DataFrame(
        {
            "ra": [10.00005, 10.0001],  # both match the primary
            "dec": [5.0, 5.0],
            "mag": [15.0, 15.2],
            "label": ["detection_A", "detection_B"],
        }
    )
    cm = CrossMatch()
    out = cm.fof_match([a, b], radius_arcsec=1.0)
    assert out.height == 1
    # mag should be averaged
    assert "mag" in out.columns
    mag_val = out["mag"][0]
    assert 14.9 < mag_val < 15.3, f"Expected mag ~15.1, got {mag_val}"
    # label should be first value (string)
    assert "label" in out.columns
    assert out["label"][0] == "detection_A"


# ---------------------------------------------------------- union auto-route
def _fake_remote(name: str = "remote-survey"):
    from xmatch.sources import CatalogueSource

    return CatalogueSource(
        name=name,
        is_local=False,
        access_method="hats",
        access_identifier="vos:cadc:test",
    )


def test_union_match_auto_routes_all_remote_full_sky(monkeypatch, tmp_path):
    """No region + every input remote -> union routes to engine='ray-union'."""
    cm = CrossMatch()
    monkeypatch.setattr(cm, "resolve_source", lambda value, overrides: _fake_remote())
    seen = {}

    def spy_ray(*args, **kwargs):
        seen["n"] = len(args[0])
        seen["engine"] = kwargs.get("engine")
        seen["out"] = args[1] if len(args) > 1 else kwargs.get("output_file")
        seen["join_type"] = kwargs.get("join_type")

    monkeypatch.setattr(cm, "_ray_union_multi", spy_ray)
    out = tmp_path / "fullsky.hats"
    cm.union_match(["gaia", "desils", "allwise"], output_file=out, radius_arcsec=1.5)
    assert seen.get("n") == 3
    assert seen.get("engine") == "ray-union"
    assert seen.get("join_type") in ("1or2", "all")
    assert seen.get("out") == out


def test_union_match_all_remote_without_output_gives_actionable_error(monkeypatch):
    cm = CrossMatch()
    monkeypatch.setattr(cm, "resolve_source", lambda value, overrides: _fake_remote())
    with pytest.raises(CrossMatchError, match=r"-o/--output"):
        cm.union_match(["gaia", "desils"], radius_arcsec=1.5)


def test_union_match_with_region_does_not_auto_route(monkeypatch):
    """An explicit region keeps the sequential path even for remote inputs."""
    cm = CrossMatch()
    monkeypatch.setattr(cm, "resolve_source", lambda value, overrides: _fake_remote())
    seen = {}

    def spy_multi(*args, **kwargs):
        seen["engine"] = kwargs.get("engine")
        return pl.DataFrame()

    monkeypatch.setattr(cm, "_multi_match_impl", spy_multi)
    cm.union_match(["gaia", "desils"], ra=180, dec=-30, radius_deg=0.01, radius_arcsec=1.5)
    assert seen.get("engine") in (None, "auto")


def test_union_match_local_frames_stay_in_process(monkeypatch):
    """Local inputs are never auto-routed; the sequential union still runs."""
    cm = CrossMatch()
    monkeypatch.setattr(
        cm, "_ray_union_multi", lambda *args, **kwargs: pytest.fail("ray-union must not run")
    )
    left = pl.DataFrame({"ra": [1.0], "dec": [2.0], "m": [3.0]})
    right = pl.DataFrame({"ra": [1.00001], "dec": [2.00001], "m2": [4.0]})
    out = cm.union_match([left, right], radius_arcsec=5.0)
    assert out.height == 1
    assert "_src_cats" in out.columns
    assert "sep_arcsec" in out.columns
    assert out["m"][0] == 3.0 and out["m2"][0] == 4.0  # both column sets survive


# ---------------------------------------------------- driver-level resilience
def _tiny_hats(tmp_path, name: str, ra: float, dec: float) -> Path:
    """One-partition local HATS catalogue (mirror-free offline input)."""
    d = tmp_path / name
    (d / "dataset" / "Norder=0" / "Dir=0").mkdir(parents=True)
    pl.DataFrame({"ra": [ra], "dec": [dec], "m": [1.0]}).write_parquet(
        d / "dataset" / "Norder=0" / "Dir=0" / "Npix=0.parquet"
    )
    (d / "properties").write_text(
        "dataproduct_type=object\nobs_collection=xmatch-test\n"
        "hats_col_ra=ra\nhats_col_dec=dec\nhats_ordering=NESTED\nhats_nrows=1\n"
    )
    return d


def test_ray_union_driver_retries_transient_failures(monkeypatch, tmp_path):
    """--retries N: a driver death resumes; run.jsonl records every attempt."""
    import json as _json

    import xmatch.ray_union as ru

    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient driver death")
        ru._LAST_PLAN = None

    monkeypatch.setattr(ru, "ray_union_match", flaky)
    cm = CrossMatch()
    out = tmp_path / "fullsky.hats"
    cm.union_match(
        [str(_tiny_hats(tmp_path, "a", 0.0, 0.0)), str(_tiny_hats(tmp_path, "b", 0.01, 0.01))],
        output_file=out,
        engine="ray-union",
        no_sync=True,
        retries=2,
        radius_arcsec=1.5,
    )
    assert calls["n"] == 3  # two retries after the first failure
    events = [_json.loads(line) for line in (out / "run.jsonl").read_text().splitlines()]
    kinds = [e["event"] for e in events]
    assert kinds.count("attempt_start") == 3
    assert kinds.count("attempt_failed") == 2
    assert "done" in kinds


def test_ray_union_driver_retries_exhausted_reraise(monkeypatch, tmp_path):
    """After --retries N failures the original error still propagates."""
    import xmatch.ray_union as ru

    def always_boom(*args, **kwargs):
        raise RuntimeError("driver death")

    monkeypatch.setattr(ru, "ray_union_match", always_boom)
    cm = CrossMatch()
    with pytest.raises(RuntimeError, match="driver death"):
        cm.union_match(
            [str(_tiny_hats(tmp_path, "a", 0.0, 0.0)), str(_tiny_hats(tmp_path, "b", 0.01, 0.01))],
            output_file=tmp_path / "fullsky.hats",
            engine="ray-union",
            no_sync=True,
            retries=1,
            radius_arcsec=1.5,
        )


def test_ray_union_no_sync_reads_surviving_replica(tmp_path):
    """--no-sync + remote catalogue: the union reads the first surviving
    copy across cache roots (primary -> replicas), not just the primary.

    The gaia mirror exists ONLY under a replica root; pre-fix the union
    resolved it against the primary root and crashed in build_union_plan
    ("no HATS partitions") even though a copy was present.
    """
    pytest.importorskip("ray")
    from xmatch.mirror import _safe_name, _version_dir

    cm = CrossMatch()
    primary = tmp_path / "primary"
    replica = tmp_path / "replica"
    src = cm.resolve_source("gaia_cds", {})
    rel = f"{_safe_name(src.name)}/{_version_dir(src)}"
    part = replica / rel / "dataset" / "Norder=0" / "Dir=2" / "Npix=0"
    part.mkdir(parents=True)
    pl.DataFrame({"RA_ICRS": [10.0], "DE_ICRS": [5.0], "Source": [1]}).write_parquet(
        part / "Npix=0.parquet"
    )
    (replica / rel / "properties").write_text(
        "hats_col_ra=RA_ICRS\nhats_col_dec=DE_ICRS\nhats_ordering=NESTED\n"
    )
    cm.config["cache"] = {"roots": [str(replica)]}
    out = tmp_path / "union.hats"
    cm.union_match(
        ["gaia_cds", str(_tiny_hats(tmp_path, "b", 10.001, 5.001))],
        output_file=out,
        engine="ray-union",
        no_sync=True,
        cache_root=str(primary),
        radius_arcsec=2.0,  # 5.1" apart: no match, both islands must survive
    )
    n = sum(pl.read_parquet(p).height for p in (out / "dataset").rglob("Npix=*.parquet"))
    assert n == 2  # gaia mirror row + local b row, each its own island


def test_union_vos_output_stages_local_and_uploads(monkeypatch, tmp_path):
    """-o vos:... output: the engine writes a local staging dir (an existing
    remote tree is pulled back first, so resume continues) and the completed
    tree is uploaded on success."""
    import xmatch.ray_union as ru
    from xmatch.storage import LocalStorage

    seen: dict = {}

    def fake_union(*args, **kwargs):
        out = Path(kwargs["output_file"])
        seen["engine_output"] = str(out)
        seen["staged_files"] = sorted(
            p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()
        )
        (out / "dataset" / "Norder=0" / "Dir=0").mkdir(parents=True)
        pl.DataFrame({"ra": [0.0], "dec": [0.0], "_src_cats": ["1+2"]}).write_parquet(
            out / "dataset" / "Norder=0" / "Dir=0" / "Npix=0.parquet"
        )
        (out / "properties").write_text("dataproduct_type=object\nhats_nrows=1\n")
        (out / "resume.state").write_text("ok\n")
        ru._LAST_PLAN = None

    monkeypatch.setattr(ru, "ray_union_match", fake_union)
    vos_root = tmp_path / "vosroot"
    (vos_root / "full.hats").mkdir(parents=True)
    (vos_root / "full.hats" / "old_marker").write_text("pre-existing\n")
    monkeypatch.setattr("xmatch.storage.open_storage", lambda root: LocalStorage(vos_root))

    cm = CrossMatch()
    cm.union_match(
        [str(_tiny_hats(tmp_path, "a", 0.0, 0.0)), str(_tiny_hats(tmp_path, "b", 0.01, 0.01))],
        output_file="vos:hats/xmatch/full.hats",
        engine="ray-union",
        no_sync=True,
        radius_arcsec=1.5,
    )
    # the engine worked on a local staging dir, with the old tree pulled back
    assert seen["engine_output"] != "vos:hats/xmatch/full.hats"
    assert seen["engine_output"].endswith("full.hats")
    assert "old_marker" in seen["staged_files"]
    # and the completed tree was uploaded to the vos: root
    assert (vos_root / "full.hats" / "properties").is_file()
    assert (vos_root / "full.hats" / "resume.state").is_file()


def test_io_vos_output_staged_upload(monkeypatch, tmp_path):
    """write_frame with a vos: output stages locally and uploads through the
    storage layer (VOSpace has no POSIX path)."""
    from xmatch import io_utils
    from xmatch.storage import LocalStorage

    fake = LocalStorage(tmp_path / "vosroot")
    monkeypatch.setattr("xmatch.storage.open_storage", lambda root: fake)

    io_utils.write_frame(
        pl.DataFrame({"ra": [1.0], "dec": [2.0], "m": [3.0]}),
        "vos:hats/xmatch/out.parquet",
    )
    assert fake.exists("out.parquet")
    assert pl.read_parquet(tmp_path / "vosroot" / "out.parquet").height == 1

    io_utils.write_frame(
        pl.DataFrame({"ra": [1.0], "dec": [2.0]}),
        "vos:hats/xmatch/out.csv",
    )
    assert fake.exists("out.csv")
    assert "ra,dec" in (tmp_path / "vosroot" / "out.csv").read_text()
