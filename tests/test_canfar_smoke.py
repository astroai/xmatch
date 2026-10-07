"""Argument and syntax checks for the CANFAR smoke script."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "canfar-smoke.sh"


def test_canfar_smoke_rejects_a_single_catalogue_before_creating_scratch(tmp_path: Path) -> None:
    scratch = tmp_path / "must-not-be-created"
    env = dict(os.environ, TMP_SCRATCH_DIR=str(scratch))
    result = subprocess.run(
        ["bash", str(SCRIPT), "gaia"],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    assert result.returncode == 2
    assert "at least two configured catalogue names" in result.stderr
    assert not scratch.exists()


def test_canfar_smoke_has_valid_bash_syntax() -> None:
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
