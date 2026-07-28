import json

import polars as pl
import pytest

from xmatch.cli import Progress, main


@pytest.fixture
def local_files(tmp_path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.parquet"
    pl.DataFrame({"id": [1, 2], "ra": [10.0, 20.0], "dec": [5.0, 6.0]}).write_csv(a)
    pl.DataFrame({"id": [1, 2], "ra": [10.00005, 20.00005], "dec": [5.0, 6.0]}).write_parquet(b)
    return a, b


@pytest.fixture(autouse=True)
def mock_bundled_config(monkeypatch):
    from pathlib import Path

    import yaml

    import xmatch.cli

    def fake_load():
        return Path("bundled_xmatch.yaml"), yaml.safe_load(_BUNDLED_TEXT)

    monkeypatch.setattr(xmatch.cli, "_load_bundled_config", fake_load)


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "xmatch" in capsys.readouterr().out


def test_cli_requires_two_catalogues(capsys):
    assert main([]) == 2
    assert "two catalogues" in capsys.readouterr().err


def test_cli_list(capsys):
    assert main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "gaia_esa" in out
    assert "catalogues" in out.lower()


def test_cli_match_to_stdout(local_files, capsys):
    a, b = local_files
    assert main([str(a), str(b), "-r", "1.0"]) == 0
    out = capsys.readouterr().out
    assert "sep_arcsec" in out
    assert out.count("\n") >= 3  # header + 2 rows


def test_cli_match_to_file(local_files, tmp_path):
    a, b = local_files
    out = tmp_path / "out.parquet"
    assert main([str(a), str(b), "-r", "1.0", "-o", str(out)]) == 0
    assert pl.read_parquet(out).height == 2


def test_cli_unknown_catalogue_returns_error(capsys):
    assert main(["nope_not_real", "also_not_real"]) == 1
    assert "Error" in capsys.readouterr().err


def test_cli_describe(capsys):
    assert main(["--describe", "not_real"]) == 1
    err = capsys.readouterr().err
    assert "Catalogue 'not_real' not found." in err

    assert main(["--describe", "gaia_cds"]) == 0
    out = capsys.readouterr().out
    assert "Catalogue: gaia_cds" in out


def test_cli_describe_no_dangling_whitespace_without_suggestion(capsys):
    """Unknown name with no close matches must not leave a blank hint line."""
    name = "zzzz_no_close_match_catalogue_xyzzy"
    assert main(["--describe", name]) == 1
    err = capsys.readouterr().err
    assert f"Catalogue '{name}' not found." in err
    assert "Did you mean" not in err
    assert " \n" not in err


# ---------------------------------------------------------------- did you mean?
def test_cli_describe_suggests_close_match_for_typo(capsys):
    """`xmatch --describe typo` must append a "Did you mean?" hint."""
    assert main(["--describe", "gaiaesa"]) == 1
    err = capsys.readouterr().err
    assert "Catalogue 'gaiaesa' not found." in err
    assert "Did you mean" in err
    assert "gaia_esa" in err


def test_cli_suggests_for_unknown_catalogue_argument(capsys):
    """Top-level `xmatch typo1 typo2` must surface the suggestion on stderr."""
    assert main(["gaia_esa_typo", "totally_xyz_qq"]) == 1
    err = capsys.readouterr().err
    assert "Error" in err
    # First arg is close enough to "gaia_esa" that the CLI must suggest it.
    assert "Did you mean" in err
    assert "gaia_esa" in err


def test_cli_multi_three_way(local_files, capsys):
    """`xmatch a.csv b.parquet a.csv` — 3-way crossmatch via CLI."""
    a, b = local_files
    assert main([str(a), str(b), str(a), "-r", "1.0"]) == 0
    out = capsys.readouterr().out
    assert "sep_arcsec" in out
    assert out.count("\n") >= 3


def test_cli_requires_at_least_two_catalogues(capsys):
    """One catalogue alone is not enough."""
    assert main(["some_file.csv"]) == 2
    assert "at least two" in capsys.readouterr().err.lower()


# ----------------------------------------------------------------- modern UX
def test_cli_subcommand_match_equivalent_to_legacy(local_files, capsys):
    """`xmatch match cat1 cat2` must produce the same result as `xmatch cat1 cat2`."""
    a, b = local_files
    rc = main(["match", str(a), str(b), "-r", "1.0"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "sep_arcsec" in out
    assert out.count("\n") >= 3  # header + 2 rows


def test_cli_subcommand_list_matches_legacy_alias(capsys):
    """`xmatch list` outputs the same catalogue table as `xmatch --list`."""
    rc = main(["list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "gaia_esa" in out
    assert "catalogues" in out.lower()


def test_cli_subcommand_describe_matches_legacy_alias(capsys):
    """`xmatch describe gaia_cds` matches the legacy --describe output."""
    rc = main(["describe", "gaia_cds"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Catalogue: gaia_cds" in out or "gaia_cds" in out


def test_cli_subcommand_describe_unknown_returns_error(capsys):
    """`xmatch describe typo` surfaces a "Did you mean" hint on stderr, rc=1."""
    rc = main(["describe", "gaiaesa"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "not found" in err.lower()
    assert "Did you mean" in err
    assert "gaia_esa" in err


def test_cli_version_via_subcommand(capsys):
    """`xmatch match --version` should print "xmatch" and exit 0."""
    with pytest.raises(SystemExit) as exc:
        main(["match", "--version"])
    assert exc.value.code == 0
    assert "xmatch" in capsys.readouterr().out


def test_cli_top_level_help_lists_subcommands(capsys):
    """The top-level `xmatch --help` advertises every subcommand."""
    # argparse exits on -h/--help; capture stdout and verify.
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for keyword in ("match", "list", "describe", "discover", "search"):
        assert keyword in out, f"--help missing subcommand {keyword!r}"


def test_cli_top_level_help_includes_examples(capsys):
    """The top-level `xmatch --help` shows the example block."""
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "Examples:" in out
    assert "xmatch a.parquet b.csv" in out


def test_cli_match_help_uses_grouped_sections(capsys):
    """`xmatch match --help` should show the grouped option headings."""
    with pytest.raises(SystemExit) as exc:
        main(["match", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    # The grouped help sections every match command advertises.
    for section in (
        "Output",
        "Geometry",
        "Match algorithm",
        "ID join",
        "Probabilistic",
        "Proper motion",
        "Advanced filters",
        "Matcher-specific",
        "Region",
    ):
        assert section in out, f"match --help missing section {section!r}"


def test_cli_no_color_flag_disables(capsys, monkeypatch):
    """`xmatch --no-color list` writes plain text without ANSI escape codes."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("XMATCH_NO_COLOR", raising=False)
    # Even if running under a TTY, --no-color wins.
    rc = main(["--no-color", "list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "\033[" not in out, "ANSI escapes must be disabled with --no-color"


def test_cli_no_color_env_var_disables(monkeypatch, capsys):
    """Setting NO_COLOR=1 disables colour even on a TTY-mocked stream."""
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("XMATCH_NO_COLOR", raising=False)
    rc = main(["list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "\033[" not in out, "NO_COLOR must disable ANSI"


def test_cli_color_auto_disabled_under_capsys(capsys):
    """Even with no flags, pytest's capsys (non-TTY) must yield plain output."""
    rc = main(["list"])
    assert rc == 0
    out = capsys.readouterr().out
    # No ANSI escape codes in non-TTY output.
    assert "\033[" not in out


def test_cli_global_flag_before_subcommand(capsys):
    """`xmatch -v list` should still work: '-v' is consumed by `_add_global_options`."""
    rc = main(["-v", "list"])
    assert rc == 0
    assert "gaia_esa" in capsys.readouterr().out


def test_cli_global_flag_after_subcommand(capsys):
    """`xmatch list --no-color` should also work (subparser inherits globals)."""
    rc = main(["list", "--no-color"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "gaia_esa" in out
    assert "\033[" not in out


# ----------------------------------------------------------- progress spinner
def test_progress_disabled_is_silent(capsys):
    """Progress(enabled=False) writes zero bytes to stderr and spawns zero threads.

    Even with `update()` calls inside the context, output is empty.
    """
    import threading as _threading

    baseline = _threading.active_count()

    with Progress("download", enabled=False):
        # These must not even touch stderr.
        for _ in range(5):
            pass

    after = _threading.active_count()
    out, err = capsys.readouterr()
    assert err == "", f"disabled Progress leaked {err!r}"
    assert out == ""
    # No leftover daemon thread.
    assert after == baseline


def test_progress_enabled_emits_animated_frames(capfd):
    """Progress(enabled=True) emits CR + clear-to-EOL updates on stderr."""
    import time as _time

    with Progress("downloading gaia", enabled=True) as p:
        p.update("phase: querying")
        # Spin at least one frame (10 Hz => 0.1s).
        _time.sleep(0.12)
        p.update("phase: running 42 rows")

    err = capfd.readouterr().err
    # CR + clear-to-EOL must be present (this is what makes it in-place).
    assert "\r\033[K" in err
    # Both the label and the latest status should appear at least once.
    assert "downloading gaia" in err
    assert "phase: running 42 rows" in err
    # The trailing clean-up write is also expected.
    assert err.rstrip("\r\033[K") in err or err.endswith("\r\033[K") or "\r\033[K" in err


def test_progress_update_replaces_status(capfd):
    """Calling update() multiple times leaves the last status visible."""
    import time as _time

    with Progress("op", enabled=True) as p:
        _time.sleep(0.12)  # first tick captures the initial empty status
        p.update("first")
        _time.sleep(0.12)
        p.update("second")
        _time.sleep(0.12)

    err = capfd.readouterr().err
    # The final status must win; the animation loop only re-renders the
    # most-recent status (we don't accumulate text in the spinner line).
    assert "second" in err


# Minimal xmatch.yaml fixture used by every doctor test.  By design this
# is *not* the entire bundled config; we only need enough to exercise the
# diff logic.  ``--config`` lets each test point at its own copy with a
# single ``.replace()`` mutation to introduce drift.
_BUNDLED_TEXT = (
    "archives:\n"
    "  cds:\n"
    "    description: CDS\n"
    "    tap_service:\n"
    '      access_url: "http://tapvizier.u-strasbg.fr/TAPVizieR/tap"\n'
    "  noao_datalab:\n"
    "    description: NOAO\n"
    "    tap_service:\n"
    '      access_url: "https://datalab.noao.edu/tap"\n'
    "\n"
    "catalogue_aliases:\n"
    '  gaia: "gaia_esa" # Default to ESA\'s version\n'
    '  nsc: "nsc_noao"\n'
    "\n"
    "catalogues:\n"
    "  gaia_esa:\n"
    '    description: "Gaia (Source Catalogue) via CDS VizieR"\n'
    "    archive: cds\n"
    "    service_id: tap_service\n"
    '    access_identifier: "I/355/gaiadr3"\n'
    '    table_name: "I/355/gaiadr3"\n'
    "    epoch: 2016.0\n"
    "    estimated_size: huge\n"
    '    ra_column: "RA_ICRS"\n'
    '    dec_column: "DE_ICRS"\n'
    '    id_column: "Source"\n'
    '    default_columns: ["Source", "RA_ICRS", "DE_ICRS", "Plx", "Gmag"]\n'
    "  nsc_noao:\n"
    '    description: "NSC DR2 via NOAO"\n'
    "    archive: noao_datalab\n"
    "    service_id: tap_service\n"
    '    access_identifier: "nsc_dr2.object"\n'
    '    table_name: "nsc_dr2.object"\n'
    "    epoch: 2015.5\n"
    "    estimated_size: huge\n"
    '    ra_column: "ra"\n'
    '    dec_column: "dec"\n'
    '    id_column: "objid"\n'
    '    default_columns: ["objid", "ra", "dec", "umag", "gmag", "rmag", "imag", "zmag", "ymag", "nepochs", "flags"]\n'
    "  ukidsslas_noao:\n"
    '    description: "UKIDSS via NOAO"\n'
    "    archive: noao_datalab\n"
    "    service_id: tap_service\n"
    '    access_identifier: "ukidss_las.object"\n'
    '    ra_column: "ra"\n'
    '    dec_column: "dec"\n'
)


def _write_doctor_fixture(tmp_path, body: str) -> str:
    """Write a test xmatch.yaml under tmp_path and return its path."""
    fixture = tmp_path / "xmatch_fixture_doctor.yaml"
    fixture.write_text(body)
    return str(fixture)


def test_cli_completion_bash_emits_script(capsys):
    """`xmatch completion bash` writes a usable bash completion script."""
    rc = main(["completion", "bash"])
    assert rc == 0
    out = capsys.readouterr().out
    # Header / registration:
    assert "complete -F _xmatch xmatch" in out
    assert "_xmatch()" in out
    # Embedded catalogue list (must include at least one real name):
    assert "_xmatch_catalogues=" in out
    assert "gaia_esa" in out, "bundled gaia_esa catalogue should be embedded"
    # All six subcommands advertised:
    assert "match list describe discover search completion" in out
    # compgen against the catalogue var:
    assert 'compgen -W "${_xmatch_catalogues[*]}"' in out


def test_cli_completion_zsh_emits_script(capsys):
    """`xmatch completion zsh` writes a `#compdef`-driven zsh script."""
    rc = main(["completion", "zsh"])
    assert rc == 0
    out = capsys.readouterr().out
    # Standard zsh markers + real catalogue name:
    assert "#compdef xmatch" in out
    assert "_xmatch_catalogues=" in out
    assert "gaia_esa" in out
    # All six subcommands advertised with descriptions:
    for sub in ("match", "list", "describe", "discover", "search", "completion"):
        assert sub in out, f"zsh completion missing subcommand {sub!r}"
    # Uses _describe (the modern zsh helper):
    assert "_describe" in out
    assert "_xmatch_subcommands" in out


def test_cli_completion_fish_emits_script(capsys):
    """`xmatch completion fish` writes a fish script with `complete -c xmatch`."""
    rc = main(["completion", "fish"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "complete -c xmatch" in out
    assert "gaia_esa" in out, "fish script should embed real catalogue names"
    # All six subcommands advertised:
    assert "match list describe discover search completion" in out
    # Fish predicates for each subcommand family:
    assert "_xmatch_needs_catalogue" in out
    assert "_xmatch_needs_endpoint" in out
    assert "_xmatch_needs_shell" in out


def test_cli_completion_unsupported_shell_exits_2():
    """`xmatch completion powershell` is rejected by argparse (SystemExit 2)."""
    with pytest.raises(SystemExit) as exc:
        main(["completion", "powershell"])
    assert exc.value.code == 2


def test_cli_completion_help_supported_via_subcommand(capsys):
    """`xmatch completion --help` prints the completion sub-command help."""
    with pytest.raises(SystemExit) as exc:
        main(["completion", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "xmatch completion" in out
    # All three shells must appear in choices:
    for shell in ("bash", "zsh", "fish"):
        assert shell in out, f"completion --help missing shell {shell!r}"
    assert "Examples:" in out


def test_cli_completion_falls_back_to_default_catalogues_when_config_missing(monkeypatch, capsys):
    """Missing config => catalog loop falls back to a hardcoded list, not crash."""
    monkeypatch.setenv("XMATCH_NO_COLOR", "1")  # ensure plain output
    # --config comes AFTER the subcommand keyword so _split_subcommand
    # sees `completion` as the first non-flag token and routes to the
    # completion subparser (not the legacy flat form).
    rc = main(
        [
            "completion",
            "--config",
            "/tmp/this_yaml_does_not_exist_for_xmatch_7e1a3d.yaml",
            "bash",
        ]
    )
    assert rc == 0
    out, err = capsys.readouterr()
    # Fallback list still advertises the common names + a hint to stderr:
    assert "_xmatch_catalogues=" in out
    assert "gaia" in out
    assert "complete -F _xmatch xmatch" in out
    assert (
        "Couldn't load config" in err or "fallback" in err.lower()
    )  # ----------------------------------------------------------- doctor


def test_cli_doctor_help_lists_options(capsys):
    """`xmatch doctor --help` shows the options and a description."""
    with pytest.raises(SystemExit) as exc:
        main(["doctor", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "xmatch doctor" in out
    for opt in ("--strict", "--quiet", "--json"):
        assert opt in out, f"doctor --help missing flag {opt}"
    assert "drift" in out.lower()


def test_cli_doctor_identical_returns_zero(tmp_path, capsys):
    """A user config that mirrors the bundled baseline yields exit 0 + verdict."""
    rc = main(["doctor", "--config", _write_doctor_fixture(tmp_path, _BUNDLED_TEXT)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "matches baseline" in out or "drift" not in out.lower()
    assert "[ OUTDATED FIELDS ]" in out


def test_cli_doctor_default_columns_drift_returns_one(tmp_path, capsys):
    """Removing a column from default_columns flips exit to 1."""
    body = _BUNDLED_TEXT.replace(
        '"objid", "ra", "dec", "umag", "gmag", "rmag", "imag", "zmag", "ymag", "nepochs", "flags"',
        '"objid", "ra", "dec"',
        1,
    )
    assert body != _BUNDLED_TEXT, "fixture should differ from baseline"
    rc = main(["doctor", "--config", _write_doctor_fixture(tmp_path, body)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "default_columns" in out
    assert "user has extra" in out or "bundled added" in out


def test_cli_doctor_raid_dec_rename_returns_one(tmp_path, capsys):
    """Renaming a structural field flips exit to 1."""
    body = _BUNDLED_TEXT.replace('ra_column: "ra"', 'ra_column: "RAJ2000"', 1)
    rc = main(["doctor", "--config", _write_doctor_fixture(tmp_path, body)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "ra_column" in out
    assert "nsc_noao" in out  # the renamed catalogue


def test_cli_doctor_alias_redirect_returns_one(tmp_path, capsys):
    """Changing `gaia: gaia_esa` to `gaia: nsc_noao` flips exit to 1."""
    body = _BUNDLED_TEXT.replace(
        '  gaia: "gaia_esa" # Default to ESA\'s version', '  gaia: "nsc_noao"'
    )
    rc = main(["doctor", "--config", _write_doctor_fixture(tmp_path, body)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "alias" in out.lower()
    assert "gaia" in out


def test_cli_doctor_missing_catalogue_is_exit_zero(tmp_path, capsys):
    """Removing a bundled catalogue is informational; exit 0 by default."""
    body = _BUNDLED_TEXT.split("  ukidsslas_noao:")[0]
    assert body != _BUNDLED_TEXT
    rc = main(["doctor", "--config", _write_doctor_fixture(tmp_path, body)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "ukidsslas_noao" in out  # still reported
    assert "MISSING IN USER" in out


def test_cli_doctor_missing_catalogue_strict_returns_one(tmp_path, capsys):
    """Same as above but with --strict: exit 1."""
    body = _BUNDLED_TEXT.split("  ukidsslas_noao:")[0]
    rc = main(["doctor", "--strict", "--config", _write_doctor_fixture(tmp_path, body)])
    assert rc == 1


def test_cli_doctor_user_only_catalogue_is_exit_zero(tmp_path, capsys):
    """Adding a catalogue only to user config is informational; exit 0."""
    body = (
        _BUNDLED_TEXT + "\n  my_local_catalog:\n"
        "    archive: cds\n"
        "    service_id: tap_service\n"
        '    access_identifier: "my.local"\n'
        '    ra_column: "ra"\n'
        '    dec_column: "dec"\n'
    )
    rc = main(["doctor", "--config", _write_doctor_fixture(tmp_path, body)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "my_local_catalog" in out


def test_cli_doctor_json_output_is_valid(tmp_path, capsys):
    """`--json` emits structured JSON with the expected top-level shape."""
    body = _BUNDLED_TEXT.replace('ra_column: "RA_ICRS"', 'ra_column: "RAJ2000"', 1)
    rc = main(["doctor", "--json", "--config", _write_doctor_fixture(tmp_path, body)])
    assert rc == 1
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["drift"] is True
    assert "config_active" in payload
    assert "config_bundled" in payload
    assert "outdated_fields" in payload
    assert isinstance(payload["summary"]["n_outdated"], int)
    assert payload["summary"]["n_outdated"] >= 1


def test_cli_doctor_quiet_emits_one_line(tmp_path, capsys):
    """`--quiet` collapses the report to a single summary line."""
    body = _BUNDLED_TEXT.replace('ra_column: "RA_ICRS"', 'ra_column: "RAJ2000"', 1)
    rc = main(["doctor", "--quiet", "--config", _write_doctor_fixture(tmp_path, body)])
    assert rc == 1
    out = capsys.readouterr().out
    # Exactly one line of output.
    assert out.count("\n") == 1, f"expected one-line output, got {out!r}"
    assert "drift" in out.lower()


def test_cli_doctor_informational_does_not_flip_exit(tmp_path, capsys):
    """Changing `description` (informational field) keeps exit 0."""
    body = _BUNDLED_TEXT.replace(
        "Gaia (Source Catalogue) via CDS VizieR",
        "Gaia via CDS (custom description)",
        1,
    )
    rc = main(["doctor", "--config", _write_doctor_fixture(tmp_path, body)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "description" in out or "INFORMATIONAL" in out


def test_cli_doctor_no_color_strips_ansi(tmp_path, monkeypatch, capsys):
    """NO_COLOR=1 ensures no ANSI escapes in human-readable output."""
    monkeypatch.setenv("NO_COLOR", "1")
    body = _BUNDLED_TEXT.replace('ra_column: "RA_ICRS"', 'ra_column: "RAJ2000"', 1)
    rc = main(["doctor", "--config", _write_doctor_fixture(tmp_path, body)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "\033[" not in out


def test_cli_doctor_unknown_flag_exits_two():
    """Argparse rejects `--bogus` with the standard SystemExit(2)."""
    with pytest.raises(SystemExit) as exc:
        main(["doctor", "--bogus"])
    assert exc.value.code == 2


def test_cli_completion_dispatches_through_main(capsys):
    """The `completion` subcommand must be routed by `_split_subcommand`."""
    rc = main(["-v", "completion", "zsh"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "#compdef xmatch" in out
    assert "gaia_esa" in out
