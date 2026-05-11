import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Adjust the import based on your project structure
from xmatch.stilts import (
    StiltsError,
    _build_correlation_expression,
    _build_error_value_expression,
    _build_stilts_command,
    _get_error_config,
    _run_stilts,
    crossmatch_id,
)

# --- Tests for _build_stilts_command ---


def test_build_stilts_command_basic():
    """Test basic command construction without java opts/tmpdir."""
    task = "tmatch2"
    params = {"in1": "cat1.fits", "in2": "cat2.fits", "out": "match.fits"}
    # Assume stilts.jar is default or in STILTS_JAR
    expected_start = [
        "java",
        "-jar",
        os.getenv("STILTS_JAR", "stilts.jar"),
        "-verbose",
        "-disk",
        task,
    ]
    expected_params = ["in1=cat1.fits", "in2=cat2.fits", "out=match.fits"]
    cmd = _build_stilts_command(task, params)
    assert cmd[: len(expected_start)] == expected_start
    assert sorted(cmd[len(expected_start) :]) == sorted(expected_params)


def test_build_stilts_command_with_opts():
    """Test command construction with java opts and tmpdir."""
    task = "tpipe"
    params = {"cmd": 'addcol ra_err "0.1"'}
    java_opts = "-Xmx2g -Dsome.flag=true"
    tmpdir = "/path/to/tmp"
    expected_start = [
        "java",
        "-Xmx2g",
        "-Dsome.flag=true",
        f"-Djava.io.tmpdir={tmpdir}",
        "-jar",
        os.getenv("STILTS_JAR", "stilts.jar"),
        "-verbose",
        "-disk",
        task,
    ]
    expected_params = ['cmd=addcol ra_err "0.1"']
    cmd = _build_stilts_command(task, params, java_opts=java_opts, tmpdir=tmpdir)
    assert cmd[: len(expected_start)] == expected_start
    assert cmd[len(expected_start) :] == expected_params


def test_build_stilts_command_with_base_cmd():
    """Test command construction using stilts_cmd_base."""
    task = "plot2d"
    params = {"x": "ra", "y": "dec"}
    base_cmd = "/opt/stilts/bin/stilts -otheropt"
    expected = ["/opt/stilts/bin/stilts", "-otheropt", task, "x=ra", "y=dec"]
    cmd = _build_stilts_command(task, params, stilts_cmd_base=base_cmd)
    assert cmd == expected


def test_build_stilts_command_list_param():
    """Test command construction with a list parameter."""
    task = "votcopy"
    params = {"in": "in.vot", "out": "out.fits", "select": ["col1", "col2"]}
    base_cmd = "stilts"
    expected = ["stilts", task, "in=in.vot", "out=out.fits", "select=col1 col2"]
    cmd = _build_stilts_command(task, params, stilts_cmd_base=base_cmd)
    # Order of params might vary, sort the param part
    assert cmd[:2] == expected[:2]
    assert sorted(cmd[2:]) == sorted(expected[2:])


# --- Tests for _get_error_config ---


@pytest.mark.parametrize(
    "config, axis, expected",
    [
        (
            {"ra_err_column": "eRA", "pos_err_units": "mas", "default_pos_error_arcsec": 0.01},
            "ra",
            ("eRA", None, 0.01, "mas"),
        ),
        (
            {
                "dec_ivar_column": "ivarDEC",
                "pos_err_units": "arcsec",
                "default_pos_error_arcsec": 0.1,
            },
            "dec",
            (None, "ivarDEC", 0.1, "arcsec"),
        ),
        (
            {"ra_error": "errRA", "units": "deg"},
            "ra",
            ("errRA", None, None, "deg"),
        ),  # Fallback units
        ({}, "ra", (None, None, None, "arcsec")),  # Defaults
        ({"default_pos_error_arcsec": 0.05}, "dec", (None, None, 0.05, "arcsec")),
    ],
)
def test_get_error_config(
    config: Dict[str, Any],
    axis: str,
    expected: Tuple[Optional[str], Optional[str], Optional[float], str],
):
    """Test extraction of error config for RA/Dec."""
    assert _get_error_config(config, axis) == expected


# --- Tests for _build_error_value_expression ---


@pytest.mark.parametrize(
    "err_col, ivar_col, floor, units, expected_expr",
    [
        ("eRA", None, None, "mas", "eRA * 0.001"),  # Convert mas to arcsec
        ("eRA", None, 0.01, "mas", "max(eRA * 0.001, 0.01)"),  # Floor applied
        (None, "ivarRA", None, "arcsec", "sqrt(1.0 / ivarRA)"),  # ivar to sigma
        (None, "ivarRA", 0.02, "arcsec", "max(sqrt(1.0 / ivarRA), 0.02)"),  # Floor with ivar
        ("errRA", None, None, "arcsec", "errRA"),  # Already arcsec
        (None, None, 0.03, "arcsec", "0.03"),  # Only floor
        (None, None, None, "arcsec", "null"),  # No error info
        ("eRA_deg", None, None, "deg", "eRA_deg * 3600.0"),  # Convert deg to arcsec
        ("eRA_deg", None, 0.05, "deg", "max(eRA_deg * 3600.0, 0.05)"),  # Floor with deg
        # Test case insensitivity of units
        ("eRA", None, None, "MAS", "eRA * 0.001"),
        ("eRA_DEG", None, None, "DEG", "eRA_DEG * 3600.0"),
    ],
)
def test_build_error_value_expression(
    err_col: Optional[str],
    ivar_col: Optional[str],
    floor: Optional[float],
    units: str,
    expected_expr: str,
):
    """Test building STILTS expression for error values."""
    assert _build_error_value_expression(err_col, ivar_col, floor, units) == expected_expr


# --- Tests for _build_correlation_expression ---


@pytest.mark.parametrize(
    "col_name, expected_expr",
    [
        ("ra_dec_corr", "ra_dec_corr"),  # Direct column name
        (None, "0"),  # No correlation column, defaults to 0
        ("", "0"),  # Empty column name, defaults to 0
    ],
)
def test_build_correlation_expression(col_name: Optional[str], expected_expr: str):
    """Test building STILTS expression for correlation term."""
    assert _build_correlation_expression(col_name) == expected_expr


# --- Tests for _run_stilts ---


# Mock CompletedProcess for successful subprocess.run
@pytest.fixture
def mock_subprocess_run_success():
    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.stdout = "STILTS task completed successfully."
    mock_process.stderr = ""
    return mock_process


# Mock CalledProcessError for failed subprocess.run
@pytest.fixture
def mock_subprocess_run_failure():
    return subprocess.CalledProcessError(
        cmd=["java", "-jar", "stilts.jar", "badtask"],
        returncode=1,
        stderr="Error: Unknown task badtask",
    )


@patch("subprocess.run")
def test_run_stilts_success(mock_run, mock_subprocess_run_success):
    """Test _run_stilts successful execution."""
    mock_run.return_value = mock_subprocess_run_success

    task = "tpipe"
    params = {"in": "in.fits", "cmd": 'select "MAG < 20"'}

    # Call the function - should not raise an error
    _run_stilts(task, params, stilts_cmd_base="stilts")

    # Check that subprocess.run was called with the expected arguments
    expected_args = ["stilts", task, "in=in.fits", 'cmd=select "MAG < 20"']
    mock_run.assert_called_once()
    call_args = mock_run.call_args[0][0]
    # Compare the core arguments, ignoring kwargs like capture_output etc.
    assert call_args == expected_args


@patch("subprocess.run")
def test_run_stilts_file_not_found(mock_run):
    """Test _run_stilts handling FileNotFoundError."""
    mock_run.side_effect = FileNotFoundError("Command 'java' not found")

    task = "tpipe"
    params = {"in": "in.fits"}

    with pytest.raises(StiltsError, match=r"command failed.*java.*not found"):
        _run_stilts(task, params)  # Use default command build which needs java


@patch("subprocess.run")
def test_run_stilts_called_process_error(mock_run, mock_subprocess_run_failure):
    """Test _run_stilts handling CalledProcessError."""
    mock_run.side_effect = mock_subprocess_run_failure

    task = "badtask"
    params = {"in": "in.fits"}

    with pytest.raises(StiltsError, match=r"failed with exit code 1"):
        _run_stilts(task, params, stilts_cmd_base="stilts")


# --- Fixtures ---


@pytest.fixture
def sample_df():
    """Sample DataFrame for input."""
    return pd.DataFrame({"ra": [10.0], "dec": [20.0], "id": [1]})  # Simplified


@pytest.fixture
def sample_config_sky():
    """Sample catalogue config for crossmatch_sky tests."""
    return {
        "_input_path": "cat1.fits",
        "_catalogue_name": "cat1",
        "ra_column": "ra",
        "dec_column": "dec",
        "id_column": "id",  # Example ID column
        "default_pos_error_arcsec": 0.1,
        "pos_err_units": "arcsec",
        "is_local": True,  # Assume local file for simplicity here
    }


@pytest.fixture
def sample_config_sky_errors():
    """Sample catalogue config with error columns."""
    return {
        "_input_path": "cat2.fits",
        "_catalogue_name": "cat2",
        "ra_column": "ra",
        "dec_column": "dec",
        "ra_err_column": "ra_err",
        "dec_err_column": "dec_err",
        "corr_column": "ra_dec_corr",
        "pos_err_units": "mas",
        "default_pos_error_arcsec": 0.01,
        "is_local": True,
    }


# --- Tests for crossmatch_sky ---


@patch("xmatch.stilts._run_stilts")
@patch("xmatch.stilts._prepare_input_table", side_effect=lambda df, td, fn: str(Path(td) / fn))
@patch("shutil.copy2")
@patch("tempfile.NamedTemporaryFile")
# Mock the error config helpers to simplify testing crossmatch_sky logic
@patch("xmatch.stilts._get_error_config", return_value=(None, None, 0.1, "arcsec"))
@patch("xmatch.stilts._build_error_value_expression", return_value="0.1")
@patch("xmatch.stilts._build_correlation_expression", return_value="0")
def test_crossmatch_sky_basic(
    m_corr_expr, m_err_expr, m_err_conf, m_tmpf, m_copy, m_prep, m_run, sample_df, sample_config_sky
):
    """Test basic crossmatch_sky call (sky matcher)."""
    # Mock NamedTemporaryFile to return a predictable name
    mock_tmp_file = MagicMock()
    mock_tmp_file.name = "/tmp/final_output_sky.parquet"
    m_tmpf.return_value = mock_tmp_file

    method_config = {"params": {"matcher": "sky", "params": "2.0"}}  # 2 arcsec radius
    output_suffix = "_sky_match"

    result_path = crossmatch_id(
        catalogue_1_df=sample_df,
        catalogue_2_df=sample_df,  # Use same for simplicity
        config_1=sample_config_sky,
        config_2=sample_config_sky,
        method_config=method_config,
        output_suffix=output_suffix,
        columns_1=None,
        columns_2=None,
    )

    assert result_path == "/tmp/final_output_sky.parquet"

    # Check _prepare_input_table was called twice
    assert m_prep.call_count == 2
    prep_calls = m_prep.call_args_list
    assert prep_calls[0][0][2] == "catalogue_1_sky_match.fits"
    assert prep_calls[1][0][2] == "catalogue_2_sky_match.fits"

    # Check _run_stilts was called once
    m_run.assert_called_once()
    run_args, run_kwargs = m_run.call_args
    assert run_args[0] == "tmatch2"  # Task name
    params = run_args[1]
    assert params["matcher"] == "sky"
    assert params["params"] == "2.0"
    assert params["in1"].endswith("catalogue_1_sky_match.fits")
    assert params["in2"].endswith("catalogue_2_sky_match.fits")
    assert params["out"].endswith("output_sky_match.parquet")
    assert params["ofmt"] == "parquet-snappy"
    assert params["values1"] == "ra dec"
    assert params["values2"] == "ra dec"

    # Check shutil.copy2 was called
    m_copy.assert_called_once()
    assert m_copy.call_args[0][0].endswith("output_sky_match.parquet")
    assert m_copy.call_args[0][1] == "/tmp/final_output_sky.parquet"


@patch("xmatch.stilts._run_stilts")
@patch("xmatch.stilts._prepare_input_table", side_effect=lambda df, td, fn: str(Path(td) / fn))
@patch("shutil.copy2")
@patch("tempfile.NamedTemporaryFile")
# Mock error config helpers to return error column names etc.
@patch("xmatch.stilts._get_error_config")
@patch("xmatch.stilts._build_error_value_expression")
@patch("xmatch.stilts._build_correlation_expression")
def test_crossmatch_sky_skyellipse(
    m_corr_expr,
    m_err_expr,
    m_err_conf,
    m_tmpf,
    m_copy,
    m_prep,
    m_run,
    sample_df,
    sample_config_sky_errors,
):
    """Test crossmatch_sky call with skyellipse matcher and error columns."""

    # Configure mocks for error helpers
    def err_conf_side_effect(cfg, axis):
        if axis == "ra":
            return (
                cfg.get("ra_err_column"),
                None,
                cfg.get("default_pos_error_arcsec"),
                cfg.get("pos_err_units"),
            )
        if axis == "dec":
            return (
                cfg.get("dec_err_column"),
                None,
                cfg.get("default_pos_error_arcsec"),
                cfg.get("pos_err_units"),
            )
        return (None, None, cfg.get("default_pos_error_arcsec"), cfg.get("pos_err_units"))

    m_err_conf.side_effect = err_conf_side_effect
    m_err_expr.side_effect = lambda err, ivar, floor, unit: (
        f"{err}*{0.001 if unit == 'mas' else 1.0}"
    )
    m_corr_expr.side_effect = lambda corr_col: corr_col if corr_col else "0"

    mock_tmp_file = MagicMock()
    mock_tmp_file.name = "/tmp/final_output_ellipse.parquet"
    m_tmpf.return_value = mock_tmp_file

    method_config = {"params": {"matcher": "skyellipse", "params": "5.0"}}  # 5 sigma
    output_suffix = "_ellipse"
    columns_1 = ["id", "ra", "dec"]
    columns_2 = ["ra_err", "dec_err"]  # Keep only specific cols from cat2

    result_path = crossmatch_id(
        catalogue_1_df=sample_df,
        catalogue_2_df=sample_df,  # Use same df, but different config
        config_1=sample_config_sky_errors,  # Config with errors
        config_2=sample_config_sky_errors,
        method_config=method_config,
        output_suffix=output_suffix,
        columns_1=columns_1,
        columns_2=columns_2,
    )

    assert result_path == "/tmp/final_output_ellipse.parquet"
    assert m_prep.call_count == 2
    m_run.assert_called_once()
    run_args, run_kwargs = m_run.call_args
    assert run_args[0] == "tmatch2"
    params = run_args[1]
    assert params["matcher"] == "skyellipse"
    assert params["params"] == "5.0"
    # Check value expressions based on mocked helpers
    assert params["values1"] == "ra dec ra_err*0.001 dec_err*0.001 ra_dec_corr"
    assert params["values2"] == "ra dec ra_err*0.001 dec_err*0.001 ra_dec_corr"
    # Check output columns command
    # Should keep all from cat1 (*) plus specified from cat2 (_2) plus _N variants if duplicates
    # Exact ocmd construction is complex, check essential parts
    assert "ocmd" in params
    assert 'keepcols "* ' in params["ocmd"]  # Keep all from cat1
    assert "ra_err_2 dec_err_2" in params["ocmd"]  # Keep specified from cat2 (renamed)

    m_copy.assert_called_once()
    assert m_copy.call_args[0][0].endswith("output_ellipse.parquet")
    assert m_copy.call_args[0][1] == "/tmp/final_output_ellipse.parquet"


# Remove TODO
# TODO: Add tests for wrapper functions like stilts_cdsskymatch, crossmatch_sky, etc.
# These will involve mocking _run_stilts and potentially file operations.
