"""Tests for user-config merge, adopt helpers, and ad-hoc table-id resolution."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from xmatch import CrossMatch
from xmatch.discovery import (
    catalogue_entry_from_schema,
    guess_endpoint,
    looks_like_table_id,
    split_table_id,
)
from xmatch.exceptions import ConfigError, InputError
from xmatch.user_config import (
    append_catalogue_to_user_config,
    deep_merge,
    format_catalogue_yaml,
    load_merged_config,
)


def test_box_cone_predicate_for_noirlab():
    from xmatch.remote_tap import _cone_predicate, _uses_box_cone

    assert _uses_box_cone("https://datalab.noirlab.edu/tap")
    assert not _uses_box_cone("http://tapvizier.u-strasbg.fr/TAPVizieR/tap")
    pred = _cone_predicate('"ra"', '"dec"', 150.1, 2.18, 0.05, box=True)
    assert "BETWEEN" in pred
    assert "CONTAINS" not in pred
    pred2 = _cone_predicate('"ra"', '"dec"', 150.1, 2.18, 0.05, box=False)
    assert "CONTAINS" in pred2
    wrap = _cone_predicate('"ra"', '"dec"', 0.1, 0.0, 0.5, box=True)
    assert " OR " in wrap
    assert "BETWEEN" in wrap


def test_looks_like_table_id_and_guess():
    assert looks_like_table_id("II/349/ps1")
    assert looks_like_table_id("J/A+A/588/A103/cat2rxs")
    assert looks_like_table_id("ls_dr10.tractor")
    assert not looks_like_table_id("gaia")
    assert not looks_like_table_id("my_sources.csv")
    assert not looks_like_table_id("data/my_catalog")
    assert guess_endpoint("II/349/ps1") == "vizier"
    assert guess_endpoint("ls_dr10.tractor") == "noirlab"
    assert guess_endpoint("gaiadr3.gaia_source") == "gaia"
    assert split_table_id("ls_dr10.tractor") == ("ls_dr10", "tractor")
    assert split_table_id("II/349/ps1") == (None, "II/349/ps1")


def test_get_catalogue_config_resolves_aliases():
    cm = CrossMatch()
    cfg = cm.get_catalogue_config("gaia")
    assert cfg["_catalogue_name"] == "gaia_cds"
    cfg2 = cm.get_catalogue_config("gaia_esa")
    assert cfg2["_catalogue_name"] == "gaia_esa"


def test_catalogue_entry_from_schema_builds_yaml_ready_dict():
    schema = {
        "access_identifier": "II/349/ps1",
        "ra_column": "RAJ2000",
        "dec_column": "DEJ2000",
        "columns_list": ["objID", "RAJ2000", "DEJ2000", "gmag", "rmag"],
    }
    name, entry = catalogue_entry_from_schema("II/349/ps1", schema, archive="cds", name="ps1_test")
    assert name == "ps1_test"
    assert entry["archive"] == "cds"
    assert entry["ra_column"] == "RAJ2000"
    assert entry["id_column"] == "objID"
    assert "gmag" in entry["default_columns"]
    block = format_catalogue_yaml(name, entry)
    assert "ps1_test:" in block
    assert "access_identifier: II/349/ps1" in block or 'access_identifier: "II/349/ps1"' in block


def test_deep_merge_and_user_overlay(tmp_path, monkeypatch):
    bundled = {
        "archives": {
            "cds": {"tap_service": {"access_url": "http://example", "access_method": "tap"}}
        },
        "catalogues": {
            "gaia_esa": {
                "archive": "cds",
                "service_id": "tap_service",
                "access_identifier": "I/355/gaiadr3",
                "ra_column": "RA_ICRS",
                "dec_column": "DE_ICRS",
            }
        },
        "catalogue_aliases": {"gaia": "gaia_esa"},
    }
    overlay = {
        "catalogues": {
            "ps1": {
                "archive": "cds",
                "service_id": "tap_service",
                "access_identifier": "II/349/ps1",
                "ra_column": "RAJ2000",
                "dec_column": "DEJ2000",
            }
        },
        "catalogue_aliases": {"ps1_cds": "ps1"},
    }
    merged = deep_merge(bundled, overlay)
    assert "gaia_esa" in merged["catalogues"]
    assert "ps1" in merged["catalogues"]
    assert merged["catalogue_aliases"]["ps1_cds"] == "ps1"

    user = tmp_path / "user.yaml"
    user.write_text(yaml.safe_dump(overlay))
    monkeypatch.setattr("xmatch.user_config.find_user_config_path", lambda: user)
    monkeypatch.setattr(
        "xmatch.user_config.bundled_config_path",
        lambda: Path(__file__).resolve().parents[1] / "src" / "xmatch" / "xmatch.yaml",
    )
    cfg, primary = load_merged_config(include_user_overlay=True)
    assert primary == user
    assert "ps1" in cfg["catalogues"]
    assert "gaia_cds" in cfg["catalogues"]
    assert "gaia_esa" in cfg["catalogues"]  # from real bundled baseline


def test_append_catalogue_merges_archives_on_second_adopt(tmp_path):
    dest = tmp_path / "xmatch.yaml"
    cds = {
        "cds": {
            "tap_service": {
                "access_url": "http://tapvizier.u-strasbg.fr/TAPVizieR/tap",
                "access_method": "tap",
            }
        }
    }
    noao = {
        "noao_datalab": {
            "tap_service": {
                "access_url": "https://datalab.noirlab.edu/tap",
                "access_method": "tap",
            }
        }
    }
    entry1 = {
        "description": "a",
        "archive": "cds",
        "service_id": "tap_service",
        "access_identifier": "II/349/ps1",
        "ra_column": "RAJ2000",
        "dec_column": "DEJ2000",
    }
    entry2 = {
        "description": "b",
        "archive": "noao_datalab",
        "service_id": "tap_service",
        "access_identifier": "catwise2020.main",
        "ra_column": "ra",
        "dec_column": "dec",
    }
    append_catalogue_to_user_config("ps1_extra", entry1, path=dest, archives=cds)
    append_catalogue_to_user_config("catwise_extra", entry2, path=dest, archives=noao)
    data = yaml.safe_load(dest.read_text())
    assert "cds" in data["archives"]
    assert "noao_datalab" in data["archives"]
    assert "ps1_extra" in data["catalogues"]
    assert "catwise_extra" in data["catalogues"]


def test_append_catalogue_to_user_config(tmp_path):
    dest = tmp_path / "xmatch.yaml"
    archives = {
        "cds": {
            "description": "CDS",
            "tap_service": {
                "access_url": "http://tapvizier.u-strasbg.fr/TAPVizieR/tap",
                "access_method": "tap",
            },
        }
    }
    entry = {
        "description": "test",
        "archive": "cds",
        "service_id": "tap_service",
        "access_identifier": "II/349/ps1",
        "ra_column": "RAJ2000",
        "dec_column": "DEJ2000",
    }
    path = append_catalogue_to_user_config(
        "ps1_local", entry, path=dest, alias="ps1x", archives=archives
    )
    assert path == dest
    data = yaml.safe_load(dest.read_text())
    assert "ps1_local" in data["catalogues"]
    assert data["catalogue_aliases"]["ps1x"] == "ps1_local"
    with pytest.raises(ConfigError, match="already exists"):
        append_catalogue_to_user_config("ps1_local", entry, path=dest, archives=archives)


def test_bundled_new_survey_aliases():
    cm = CrossMatch()
    assert cm.resolve_name("ps1") == "ps1"
    assert cm.resolve_name("twomass") == "twomass"
    assert cm.resolve_name("catwise") == "catwise"
    assert cm.resolve_name("sdss") == "sdss"
    assert cm.resolve_name("nvss") == "nvss"
    assert cm.resolve_name("delve3") == "delve3"
    assert cm.resolve_name("desi") == "desi"
    src = cm.resolve_source("ps1", {})
    assert src.access_identifier == "II/349/ps1"
    assert src.ra_column == "RAJ2000"


def test_resolve_access_identifier_uses_bundled_catalogue():
    """Pasting ACCESS from `xmatch list` must resolve to the short-name entry."""
    cm = CrossMatch()
    src = cm.resolve_source("II/349/ps1", {})
    assert src.name == "ps1"
    assert src.default_columns is not None
    assert "gmag" in src.default_columns
    # Data Lab tractor id → desils, not a raw TAP_SCHEMA probe.
    src2 = cm.resolve_source("ls_dr10.tractor", {})
    assert src2.name == "desils"
    assert src2.access_identifier == "ls_dr10.tractor"


def test_source_from_table_id_uses_mock_schema(monkeypatch):
    cm = CrossMatch()

    def fake_schema(tap_url, table_id, auth_session=None):
        return {
            "columns": None,
            "columns_count": 5,
            "ra_column": "RAJ2000",
            "dec_column": "DEJ2000",
            "columns_list": ["objID", "RAJ2000", "DEJ2000", "gmag", "rmag"],
            "access_identifier": table_id.strip('"'),
        }

    monkeypatch.setattr("xmatch.crossmatch.get_table_schema", fake_schema)
    # Unknown VizieR id (not in bundled config) → ad-hoc TAP path.
    src = cm.resolve_source("II/999/not_a_real_cat", {})
    assert not src.is_local
    assert src.access_method == "tap"
    assert src.access_identifier == "II/999/not_a_real_cat"
    assert src.ra_column == "RAJ2000"

    with pytest.raises(InputError):
        cm.resolve_source("not_a_catalogue_or_table", {})
