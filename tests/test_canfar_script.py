"""CANFAR cluster script tests: POSIX-sh syntax + the dry-run contract.

The script never touches sbatch in these tests (``--dry-run`` exits before
any submission), so they run on any machine with ``bash``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "canfar-cluster.sh"


def test_canfar_script_syntax() -> None:
    """The cluster script parses under POSIX sh (``sh -n``)."""
    for shell in ("sh", "bash"):
        res = subprocess.run(
            [shell, "-n", str(SCRIPT)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert res.returncode == 0, (shell, res.stderr)


def test_canfar_script_dry_run_output() -> None:
    """``--dry-run`` prints the exact plan, exits 0, never submits."""
    res = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert res.returncode == 0, res.stderr
    out = res.stdout
    assert "ray start --head" in out
    assert "--num-cpus=" in out
    assert "RAY_ADDRESS=" in out
    assert "ray stop" in out
    assert "Submitted batch job" not in out


def test_canfar_script_unknown_option_fails() -> None:
    """Unknown options exit non-zero with usage (no submission)."""
    res = subprocess.run(
        ["bash", str(SCRIPT), "--bogus"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert res.returncode != 0
    assert "unknown option" in res.stderr
    assert "Submitted batch job" not in res.stdout
