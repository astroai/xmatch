#!/usr/bin/env python3
"""Tests for CLI version handling."""

import os
import re
import subprocess
import sys
from io import StringIO
from unittest.mock import patch

import pytest

import xmatch
from xmatch import __version__

# Get the directory where the script is located
testdir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(testdir, "inputs")
output_dir = os.path.join(testdir, "outputs")
os.makedirs(output_dir, exist_ok=True)


def test_cli_uses_version_from_init():
    """Test that the CLI uses the __version__ from __init__.py."""
    # Create a temporary stdout to capture version output
    stdout = StringIO()

    # Patch sys.stdout and sys.argv to run with --version
    with (
        patch("sys.stdout", stdout),
        patch("sys.argv", ["xmatch", "--version"]),
        patch("sys.exit") as mock_exit,
    ):  # Prevent actual exit
        # Import and run main
        from xmatch.cli import main

        main()

        # Check exit was called (version action calls exit)
        mock_exit.assert_called_once()

        # Get the output
        output = stdout.getvalue()

        # Expected format: "xmatch X.Y.Z" where X.Y.Z is the version
        expected = f"xmatch {__version__}\n"
        assert output == expected, f"Expected version output '{expected}', got '{output}'"


def test_cli_version_matches_package_version():
    """Test that the CLI version matches the package version."""
    # Run actual CLI process with --version
    try:
        output = subprocess.check_output(
            [sys.executable, "-m", "xmatch", "--version"],
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )

        # Extract version from CLI output
        match = re.search(r"xmatch\s+(\d+\.\d+\.\d+(?:\w+)?)", output)
        assert match is not None, f"Couldn't extract version from output: {output}"

        cli_version = match.group(1)
        package_version = __version__

        # Compare with package __version__
        assert cli_version == package_version, (
            f"CLI version '{cli_version}' doesn't match package version '{package_version}'"
        )

    except subprocess.CalledProcessError as e:
        pytest.fail(f"CLI command failed with error: {e.output}")


def test_version_not_hardcoded_in_parser():
    """Test that the version is not hardcoded in the argument parser."""
    # Read the cli.py file content
    import inspect

    # Get the file path of the cli module
    cli_file = inspect.getfile(xmatch.cli)

    with open(cli_file, "r") as f:
        content = f.read()

    # Look for hardcoded version pattern in parser definition
    hardcoded_version_pattern = r'version="[^"]+"'
    if re.search(hardcoded_version_pattern, content):
        # If found, check it's using the __version__ variable
        correct_pattern = r'version=f"[^"]+ *\{__version__\}"'
        assert re.search(correct_pattern, content), (
            "Version appears to be hardcoded in parser, not using __version__ variable"
        )


# When loading test files:
def test_something():
    os.path.join(input_dir, "some_input.csv")
    os.path.join(output_dir, "some_output.parquet")
    # ...existing code...
