"""CANFAR ray-job script tests: POSIX-sh syntax + the dry-run contract.

The script never touches the platform in these tests (``--dry-run`` exits
before any ``canfar`` / ``astroai-workload`` call), so they run on any
machine with ``bash``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "canfar-ray-job.sh"


def test_canfar_ray_job_syntax() -> None:
    """The ray-job script parses under POSIX sh (``sh -n``)."""
    for shell in ("sh", "bash"):
        res = subprocess.run(
            [shell, "-n", str(SCRIPT)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert res.returncode == 0, (shell, res.stderr)


def test_canfar_ray_job_dry_run_output() -> None:
    """``--dry-run`` prints the plan, exits 0, never touches the platform."""
    res = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert res.returncode == 0, res.stderr
    out = res.stdout
    assert "canfar ps" in out
    assert "astroai-workload cluster ensure --workers 4 --cores 1 --ram 4" in out
    assert "astroai-workload submit --cmd" in out
    assert "--wait" in out
    assert "--address" in out  # the submit line names its target truthfully
    assert "RAY_ADDRESS" not in out  # no Slurm-era leftovers
    assert "Submitted batch job" not in out


def test_canfar_ray_job_unknown_option_fails() -> None:
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
