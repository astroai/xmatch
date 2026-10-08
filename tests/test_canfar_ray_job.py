"""CANFAR ray-job script tests: POSIX-sh syntax + the dry-run contract.

The script never touches the platform in these tests (``--dry-run`` exits
before any ``canfar`` / ``astroai`` call), so they run on any
machine with ``bash``.
"""

from __future__ import annotations

import os
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
        ["bash", str(SCRIPT), "--dry-run", "--command", "pixi run xmatcher --help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert res.returncode == 0, res.stderr
    out = res.stdout
    assert "canfar ps" in out
    assert "astroai cluster start --min-workers 4 --max-workers 4 --cores 1 --ram 4" in out
    assert "astroai jobs submit --cmd" in out
    assert "--wait" in out
    assert "--address" in out  # the submit line names its target truthfully
    assert "ASTROAI_RAY_JOBS_ADDRESS" not in out
    assert "Submitted batch job" not in out


def test_canfar_ray_job_requires_bounded_command() -> None:
    res = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert res.returncode == 2
    assert "--command is required" in res.stderr


def test_canfar_ray_job_manager_uses_supported_resource_flags() -> None:
    res = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--dry-run",
            "--command",
            "pixi run xmatcher --help",
            "--create-manager",
            "images.canfar.net/astroai/ray-manager:latest",
            "--manager-name",
            "xmatcher-test",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert res.returncode == 0, res.stderr
    assert "canfar create --cpu 2 --memory 8 --name xmatcher-test contributed" in res.stdout
    assert "--cores 2" not in res.stdout
    assert "--ram 8" not in res.stdout


def test_canfar_ray_job_uses_current_address_environment_variable() -> None:
    env = dict(os.environ, CANFAR_RAY_JOBS_ADDRESS="https://ray.example/jobs")
    res = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run", "--command", "pixi run xmatcher --help"],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    assert res.returncode == 0, res.stderr
    assert "skipped — CANFAR_RAY_JOBS_ADDRESS already set" in res.stdout
    assert "https://ray.example/jobs" in res.stdout


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
