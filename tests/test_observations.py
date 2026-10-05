import json

import polars as pl
import pytest

from xmatch import CrossMatch


def _source(data, **metadata):
    return CrossMatch().resolve_source(
        pl.DataFrame(data),
        {"id_column": "id", "release_namespace": "pilot:dr1", **metadata},
    )


def test_inventory_preserves_native_values_and_release_identity():
    from xmatch.observations import source_inventory

    source = _source({"id": [17, 18], "ra": [359.9, 0.1], "dec": [2.0, -2.0]})
    rows = source_inventory(source).collect()
    assert rows["native_id"].to_list() == ["17", "18"]
    assert rows["ra_deg"].to_list() == [359.9, 0.1]
    assert rows["epoch_jyear"].null_count() == 2
    assert rows["native"].to_list()[0] == {"id": 17, "ra": 359.9, "dec": 2.0}
    reordered = source_inventory(source.with_frame(source.lazy().reverse())).collect()
    assert reordered["source_id"].to_list() == list(reversed(rows["source_id"].to_list()))
    other = _source({"id": [17]}, release_namespace="pilot:dr2")
    assert source_inventory(other).collect()["source_id"][0] != rows["source_id"][0]


def test_namespaced_ids_distinguish_integer_string_and_delimiters():
    from xmatch.observations import namespaced_source_id

    assert namespaced_source_id("a:b", "c") != namespaced_source_id("a", "b:c")
    assert namespaced_source_id("a", 17) != namespaced_source_id("a", "17")
    for identifier in (None, "", True, 1.0):
        with pytest.raises(ValueError):
            namespaced_source_id("a", identifier)


@pytest.mark.parametrize(
    "unit,value,expected",
    [
        ("Jy", -2.0, -2.0),
        ("mJy", -2.0, -0.002),
        ("uJy", -2.0, -0.000002),
        ("nanomaggy", 1.0, 0.000003631),
    ],
)
def test_flux_normalization_preserves_negative_flux(unit, value, expected):
    from xmatch.observations import normalize_photometry

    source = _source(
        {"id": [17], "f": [value], "e": [0.5]},
        photometry=[
            {
                "passband": "pilot:g:v1",
                "value_column": "f",
                "error_column": "e",
                "unit": unit,
                "method": "psf",
            }
        ],
    )
    row = normalize_photometry(source).collect().row(0, named=True)
    assert row["native_value"] == value
    assert row["flux_jy"] == pytest.approx(expected)
    assert row["measurement_method"] == "psf"
    assert row["passband"] == "pilot:g:v1"


def test_magnitudes_have_asymmetric_flux_errors_and_vega_requires_zeropoint():
    from xmatch.observations import normalize_photometry

    mapping = {"passband": "pilot:g:v1", "value_column": "m", "error_column": "e", "unit": "ABmag"}
    source = _source({"id": [1], "m": [20.0], "e": [0.5]}, photometry=[mapping])
    row = normalize_photometry(source).collect().row(0, named=True)
    assert row["flux_jy"] == pytest.approx(0.00003631)
    assert row["flux_error_jy"] is None
    assert row["flux_error_upper_jy"] > row["flux_error_lower_jy"] > 0
    source.photometry = [{**mapping, "unit": "Vegamag"}]
    row = normalize_photometry(source).collect().row(0, named=True)
    assert row["native_value"] == 20.0 and row["flux_jy"] is None
    source.photometry = [{**mapping, "unit": "Vegamag", "zeropoint_jy": 1000.0}]
    assert normalize_photometry(source).collect()["flux_jy"][0] == pytest.approx(1e-5)


def test_photometry_keeps_missing_upper_limit_and_observation_epoch_distinct():
    from xmatch.observations import normalize_photometry

    source = _source(
        {
            "id": [1, 2, 3],
            "flux": [1.0, None, 4.0],
            "status": ["D", "M", "L"],
            "mjd": [50000.0, None, 50002.0],
        },
        epoch=2016.0,
        photometry=[
            {
                "passband": "pilot:g:v1",
                "value_column": "flux",
                "unit": "Jy",
                "status_column": "status",
                "status_values": {"D": "measured", "M": "missing", "L": "upper_limit"},
                "observation_time_column": "mjd",
                "time_format": "mjd",
                "time_scale": "utc",
            }
        ],
    )
    rows = normalize_photometry(source).collect()
    assert rows["status"].to_list() == ["measured", "missing", "upper_limit"]
    assert rows["observation_time"].to_list() == [50000.0, None, 50002.0]
    assert rows["time_format"].to_list() == ["mjd"] * 3
    assert rows["flux_jy"].to_list() == [1.0, None, None]
    assert rows["upper_limit_jy"].to_list() == [None, None, 4.0]


def test_raw_property_evidence_keeps_conflicts_and_native_types():
    from xmatch.observations import normalize_property_evidence

    source = _source(
        {"id": [1], "z1": [0.1], "z2": [0.2], "type": ["QSO"], "rv": [52.0]},
        property_evidence=[
            {
                "property": "redshift.spec",
                "value_column": "z1",
                "unit": "dimensionless",
                "category": "inferred_spectroscopy",
                "method": "pipeline-a",
            },
            {
                "property": "redshift.spec",
                "value_column": "z2",
                "unit": "dimensionless",
                "category": "inferred_spectroscopy",
                "method": "pipeline-b",
            },
            {
                "property": "type.native",
                "value_column": "type",
                "category": "literature_assertion",
                "method": "native-classifier",
            },
            {
                "property": "velocity.radial",
                "value_column": "rv",
                "unit": "km/s",
                "category": "measured",
                "method": "spectrum",
                "reference_frame": "unknown",
                "doppler_convention": "unknown",
            },
        ],
    )
    rows = normalize_property_evidence(source).collect()
    assert rows["native_value"].to_list() == ["0.1", "0.2", "QSO", "52.0"]
    assert rows["property"].to_list().count("redshift.spec") == 2
    assert rows["evidence_id"].n_unique() == 4
    assert rows.filter(pl.col("property") == "velocity.radial")["reference_frame"][0] == "unknown"


def test_resolution_preserves_observation_metadata_for_config_and_local_sources():
    mapping = {"passband": "pilot:g", "value_column": "f", "unit": "Jy"}
    cm = CrossMatch()
    cm.catalogues_config["pilot"] = {
        "archive": "cds",
        "service_id": "tap_service",
        "access_identifier": "pilot.main",
        "release_namespace": "pilot:dr1",
        "photometry": [mapping],
        "release_metadata": {"citation": "explicit upstream reference"},
    }
    remote = cm.resolve_source("pilot", {})
    local = _source({"id": [1], "f": [1.0]}, photometry=[mapping])
    assert remote.release_namespace == local.release_namespace == "pilot:dr1"
    assert remote.photometry == local.photometry == [mapping]
    assert remote.release_metadata["citation"] == "explicit upstream reference"


def test_source_resolution_retains_explicit_astrometric_frame():
    source = _source(
        {"id": [1], "l": [10.0], "b": [20.0]}, ra_column="l", dec_column="b", frame="galactic"
    )
    assert source.frame == "galactic"


def test_immutable_release_accounts_for_sources_without_photometry(tmp_path):
    from xmatch.observations import write_observation_release

    destination = tmp_path / "release"
    source = _source({"id": [1, 2]})
    manifest = write_observation_release(
        destination,
        [source],
        release_id="pilot:r1",
        provenance={"software_version": "test", "input_checksum": "explicit"},
    )
    assert manifest["sources"][0]["source_count"] == 2
    assert manifest["sources"][0]["photometry_count"] == 0
    assert json.loads((destination / "manifest.json").read_text()) == manifest
    assert pl.read_parquet(destination / manifest["sources"][0]["inventory_path"]).height == 2
    with pytest.raises(FileExistsError):
        write_observation_release(destination, [source], release_id="pilot:r1", provenance={})


@pytest.mark.parametrize("ids", [[1, 1], [1, None]])
def test_invalid_native_identity_never_publishes_release(tmp_path, ids):
    from xmatch.observations import write_observation_release

    destination = tmp_path / "release"
    with pytest.raises(ValueError, match="[Ii][Dd]|identity|identifier"):
        write_observation_release(
            destination, [_source({"id": ids})], release_id="pilot:r1", provenance={}
        )
    assert not destination.exists()


def test_normalization_rejects_undeclared_or_invalid_metadata():
    from xmatch.observations import (
        normalize_photometry,
        normalize_property_evidence,
        source_inventory,
    )

    source = _source({"id": [1], "f": [1.0]})
    source.release_namespace = None
    with pytest.raises(ValueError, match="release_namespace"):
        source_inventory(source)
    source.release_namespace = "pilot:dr1"
    source.photometry = [{"passband": "g", "value_column": "f", "unit": "counts"}]
    assert normalize_photometry(source).collect()["flux_jy"][0] is None
    source.photometry = [{"passband": "g", "value_column": "absent", "unit": "Jy"}]
    with pytest.raises(ValueError, match="absent"):
        normalize_photometry(source)
    source.property_evidence = [
        {"property": "redshift", "value_column": "f", "category": "prediction", "method": "model"}
    ]
    with pytest.raises(ValueError, match="model_id"):
        normalize_property_evidence(source)


def test_bad_uncertainties_and_overflow_do_not_become_finite_measurements():
    from xmatch.observations import normalize_photometry

    source = _source(
        {"id": [1, 2, 3], "f": [1.0, float("nan"), 2.0], "e": [-1.0, 0.2, float("inf")]},
        photometry=[{"passband": "g", "value_column": "f", "error_column": "e", "unit": "Jy"}],
    )
    rows = normalize_photometry(source).collect()
    assert rows["status"].to_list() == ["measured", "invalid", "measured"]
    assert rows["flux_error_jy"].null_count() == 3
    assert rows["flags"].to_list()[0] == ["invalid_uncertainty"]
    source = _source(
        {"id": [1], "m": [-10000.0]},
        photometry=[{"passband": "g", "value_column": "m", "unit": "ABmag"}],
    )
    row = normalize_photometry(source).collect().row(0, named=True)
    assert row["flux_jy"] is None and row["flags"] == ["conversion_nonfinite"]
    assert row["native_value"] == -10000.0


def test_duplicate_measurement_mapping_is_rejected():
    from xmatch.observations import normalize_photometry

    mapping = {"passband": "g", "value_column": "f", "unit": "Jy"}
    with pytest.raises(ValueError, match="duplicate"):
        normalize_photometry(_source({"id": [1], "f": [1.0]}, photometry=[mapping, mapping]))


def test_evidence_preserves_posterior_array_and_upstream_observation_id():
    from xmatch.observations import normalize_property_evidence

    source = _source(
        {"id": [1], "posterior": [[0.2, 0.8]], "obs_id": ["spec-42"]},
        property_evidence=[
            {
                "property": "redshift.posterior",
                "value_column": "posterior",
                "category": "prediction",
                "model_id": "model:v1",
                "method": "test",
                "upstream_evidence_id_column": "obs_id",
                "input_features": ["flux_g", "flux_r"],
            }
        ],
    )
    row = normalize_property_evidence(source).collect().row(0, named=True)
    assert json.loads(row["native_value"]) == [0.2, 0.8]
    assert row["upstream_evidence_id"] == "spec-42"
    assert json.loads(row["input_features_json"]) == ["flux_g", "flux_r"]


def test_release_verification_rejects_tampering_and_escaped_paths(tmp_path):
    from xmatch.observations import verify_observation_release, write_observation_release

    path = tmp_path / "release"
    manifest = write_observation_release(
        path, [_source({"id": [1]})], release_id="pilot:r1", provenance={}
    )
    assert verify_observation_release(path) == manifest
    shard = path / manifest["sources"][0]["inventory_path"]
    with shard.open("ab") as stream:
        stream.write(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        verify_observation_release(path)
    manifest["sources"][0]["inventory_path"] = "../external.parquet"
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="escapes"):
        verify_observation_release(path)


def test_photometry_keeps_passband_curve_and_limit_confidence_metadata():
    from xmatch.observations import normalize_photometry

    source = _source(
        {"id": [1], "f": [5.0], "state": ["L"]},
        photometry=[
            {
                "passband": "survey:g",
                "passband_version": "curve:v2",
                "passband_uri": "https://example.org/g.dat",
                "value_column": "f",
                "unit": "uJy",
                "status_column": "state",
                "status_values": {"L": "upper_limit"},
                "limit_sigma": 5.0,
                "limit_convention": "provider detection threshold",
            }
        ],
    )
    row = normalize_photometry(source).collect().row(0, named=True)
    assert row["passband_version"] == "curve:v2"
    assert row["limit_sigma"] == 5.0
    assert row["limit_confidence"] is None
    assert row["upper_limit_jy"] == pytest.approx(5e-6)
    source.photometry[0]["limit_confidence"] = 1.5
    with pytest.raises(ValueError, match="limit_confidence"):
        normalize_photometry(source)


def test_required_observation_columns_include_evidence_and_astrometric_fields():
    from xmatch.observations import required_observation_columns

    source = _source(
        {"id": [1]},
        ra_column="ra",
        dec_column="dec",
        epoch_column="epoch",
        pm_ra_column="pmra",
        photometry=[
            {
                "passband": "g",
                "value_column": "flux",
                "error_column": "error",
                "unit": "Jy",
                "status_column": "state",
                "status_values": {"D": "measured"},
            }
        ],
        property_evidence=[
            {
                "property": "redshift.spec",
                "value_column": "z",
                "upstream_evidence_id_column": "spec_id",
                "category": "measured",
                "method": "native",
            }
        ],
    )
    assert required_observation_columns(source) == [
        "id",
        "ra",
        "dec",
        "epoch",
        "pmra",
        "flux",
        "error",
        "state",
        "z",
        "spec_id",
    ]


def test_magnitude_uncertainty_conversion_avoids_intermediate_overflow_and_flags_true_overflow():
    from xmatch.observations import normalize_photometry

    source = _source(
        {"id": [1, 2], "m": [1000.0, 20.0], "e": [900.0, 900.0]},
        photometry=[{"passband": "g", "value_column": "m", "error_column": "e", "unit": "ABmag"}],
    )
    rows = normalize_photometry(source).collect()
    assert rows["flux_error_upper_jy"][0] == pytest.approx(3.631e-37, abs=1e-45)
    assert rows["flux_error_upper_jy"][1] is None
    assert rows["flags"].to_list()[1] == ["conversion_uncertainty_nonfinite"]


def test_vega_flux_conversion_combines_zeropoint_before_exponentiation():
    from xmatch.observations import normalize_photometry

    source = _source(
        {"id": [1], "m": [-800.0]},
        photometry=[
            {"passband": "g", "value_column": "m", "unit": "Vegamag", "zeropoint_jy": 1e-20}
        ],
    )
    row = normalize_photometry(source).collect().row(0, named=True)
    assert row["flux_jy"] == pytest.approx(1e300)
    assert row["flags"] == []


def test_invalid_upper_limit_is_retained_and_flagged():
    from xmatch.observations import normalize_photometry

    source = _source(
        {"id": [1], "f": [-1.0], "status": ["L"]},
        photometry=[
            {
                "passband": "g",
                "value_column": "f",
                "unit": "Jy",
                "status_column": "status",
                "status_values": {"L": "upper_limit"},
            }
        ],
    )
    row = normalize_photometry(source).collect().row(0, named=True)
    assert row["native_value"] == -1.0 and row["upper_limit_jy"] is None
    assert row["status"] == "invalid" and row["flags"] == ["invalid_upper_limit"]


def test_dispatch_rejects_mixed_frames_before_matching_and_keeps_id_join_available():
    from xmatch.exceptions import CrossMatchError
    from xmatch.request import MatchRequest

    source = _source({"id": [1], "ra": [10.0], "dec": [20.0]})
    other = _source({"id": [1], "ra": [10.0], "dec": [20.0]}, frame="galactic")
    cm = CrossMatch()
    request = MatchRequest(source.lazy(), other.lazy(), engine="fast")
    with pytest.raises(CrossMatchError, match="frame"):
        cm._dispatch(source, other, request)
    request.id_join = True
    assert cm._dispatch(source, other, request).collect().height == 1


def test_local_target_epoch_requires_icrs_even_when_frames_agree():
    from xmatch.exceptions import CrossMatchError
    from xmatch.matchers import MatchSpec
    from xmatch.request import MatchRequest

    source = _source({"id": [1], "ra": [10.0], "dec": [20.0]}, frame="galactic", epoch=2020.0)
    request = MatchRequest(
        source.lazy(), source.lazy(), engine="fast", spec=MatchSpec(target_epoch=2020.0)
    )
    with pytest.raises(CrossMatchError, match="ICRS"):
        CrossMatch()._local_match(source, source, source.lazy(), source.lazy(), request)


def test_nway_rejects_mixed_frames_before_remote_download():
    from xmatch.exceptions import CrossMatchError

    cm = CrossMatch()
    for name, frame in [("equatorial", "icrs"), ("galactic", "galactic")]:
        cm.catalogues_config[name] = {
            "archive": "cds",
            "service_id": "tap_service",
            "access_method": "unsupported",
            "access_identifier": "pilot.main",
            "frame": frame,
            "ra_column": "ra",
            "dec_column": "dec",
        }
    with pytest.raises(CrossMatchError, match="frame"):
        cm.nway_match(["equatorial", "galactic"])


def test_same_frame_spherical_geometry_does_not_require_icrs():
    from xmatch.request import MatchRequest

    source = _source({"id": [1], "ra": [10.0], "dec": [20.0]}, frame="galactic")
    request = MatchRequest(source.lazy(), source.lazy(), engine="fast")
    assert CrossMatch()._dispatch(source, source, request).collect().height == 1
