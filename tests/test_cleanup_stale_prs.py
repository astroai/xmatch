"""Smoke tests for ``scripts/cleanup-stale-prs.py`` main() ordering.

Validates the four flag combinations behave as documented in
``CONTRIBUTING.md``'s quarterly section:

* No flags         — dry-run text rendering, no destructive ops.
* ``--json`` alone — dry-run JSON rendering, no destructive ops.
* ``--apply`` alone — destructive ops + human-readable rollback info.
* ``--apply --json`` — destructive ops fire FIRST, then the JSON renderer
                       emits the post-apply state so audit logs include
                       per-row rollback pointers (regression for the
                       short-circuit bug where ``--json`` returned early
                       before ``--apply`` was honoured).

The tests mock ``fetch_open_prs`` / ``filter_stale`` /
``close_and_debranch`` / ``render_json`` so no real ``gh`` / ``git`` is
invoked.
"""

import importlib.util
import json
import pathlib
import sys
from unittest import mock

# The script lives at ``scripts/cleanup-stale-prs.py`` (hyphenated filename)
# but Python's normal sys.path-based import mechanism looks for an
# underscore-named file.  Load the module explicitly by file path,
# registering it under the conventional underscore name so the rest of the
# test file can ``from csp import StalePR`` etc. via the csp binding below.
_SCRIPT_PATH = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "cleanup-stale-prs.py"
_spec = importlib.util.spec_from_file_location("cleanup_stale_prs", str(_SCRIPT_PATH))
assert _spec is not None and _spec.loader is not None, f"could not load spec for {_SCRIPT_PATH}"
csp = importlib.util.module_from_spec(_spec)
sys.modules["cleanup_stale_prs"] = csp  # cache so any later lookup resolves here
_spec.loader.exec_module(csp)


def _fake_row(number=113, head="bolt/optimize-pandas-decode-abc", sha="deadbeef"):
    """Build a StalePR in the pre-apply state."""
    return csp.StalePR(
        number=number,
        head=head,
        title="vectorize pandas byte column type check",
        author="sfabbro",
        additions=14,
        deletions=3,
        updated="2025-01-01",
        group="bolt",
        # Pre-apply state: rollback pointers are None.
        pr_rollback_file=None,
        branch_rollback_sha=None,
    )


def test_apply_then_json_runs_apply_first():
    """Regression for the --json short-circuit before --apply bug.

    When both flags are set, ``close_and_debranch`` must run before
    ``render_json`` so the JSON dump captures the per-row rollback
    pointers populated by the destructive phase.
    """
    fake_rows = [_fake_row()]
    call_order: list[str] = []

    def fake_close(rows):
        # Simulate the real close_and_debranch row mutation in-place.
        rows[0].pr_rollback_file = "/tmp/cleanup-prs-populated.txt"
        rows[0].branch_rollback_sha = "abcdef0123456789"
        call_order.append("close")

    def fake_render(rows):
        call_order.append("render_json")
        return json.dumps([r.__dict__ for r in rows])

    with (
        mock.patch.object(csp, "fetch_open_prs", return_value=[]),
        mock.patch.object(csp, "filter_stale", return_value=fake_rows),
        mock.patch.object(csp, "close_and_debranch", side_effect=fake_close),
        mock.patch.object(csp, "render_json", side_effect=fake_render),
    ):
        rc = csp.main(["--apply", "--json"])

    assert rc == 0
    assert call_order == ["close", "render_json"], (
        f"--apply must execute BEFORE --json render; actual order: {call_order}"
    )


def test_json_alone_does_not_call_close(capsys):
    """``--json`` alone is dry-run; close_and_debranch must not fire."""
    fake_rows = [_fake_row()]
    with (
        mock.patch.object(csp, "fetch_open_prs", return_value=[]),
        mock.patch.object(csp, "filter_stale", return_value=fake_rows),
        mock.patch.object(csp, "close_and_debranch") as mock_close,
    ):
        rc = csp.main(["--json"])

    assert rc == 0
    mock_close.assert_not_called()
    # JSON is still rendered; emit an empty rollback block in the parseable form.
    _ = capsys.readouterr().out


def test_apply_alone_runs_close_once_emit_text(capsys):
    """``--apply`` alone: close fires once, JSON is not rendered."""
    fake_rows = [_fake_row()]
    with (
        mock.patch.object(csp, "fetch_open_prs", return_value=[]),
        mock.patch.object(csp, "filter_stale", return_value=fake_rows),
        mock.patch.object(
            csp, "close_and_debranch", return_value=("/tmp/prs.txt", "/tmp/br.txt")
        ) as mock_close,
    ):
        rc = csp.main(["--apply"])

    assert rc == 0
    mock_close.assert_called_once_with(fake_rows)
    out = capsys.readouterr().out
    # The text-mode rollback section is still rendered.
    assert "Done. Rollback lists" in out
    assert "/tmp/prs.txt" in out


def test_apply_json_with_empty_rows_skips_apply(capsys):
    """``--apply --json`` with zero matches: skip --apply, dump empty array."""
    with (
        mock.patch.object(csp, "fetch_open_prs", return_value=[]),
        mock.patch.object(csp, "filter_stale", return_value=[]),
        mock.patch.object(csp, "close_and_debranch") as mock_close,
        mock.patch.object(csp, "render_json", return_value="[]"),
    ):
        rc = csp.main(["--apply", "--json"])

    assert rc == 0
    mock_close.assert_not_called()


def test_apply_json_emits_post_apply_state_in_payload(capsys):
    """``--apply --json`` post-apply JSON includes the populated rollback pointers."""
    fake_rows = [_fake_row()]

    def fake_close(rows):
        rows[0].pr_rollback_file = "/tmp/post-apply-prs.txt"
        rows[0].branch_rollback_sha = "sha-after-close"

    def fake_render(rows):
        return json.dumps([r.__dict__ for r in rows])

    with (
        mock.patch.object(csp, "fetch_open_prs", return_value=[]),
        mock.patch.object(csp, "filter_stale", return_value=fake_rows),
        mock.patch.object(csp, "close_and_debranch", side_effect=fake_close),
        mock.patch.object(csp, "render_json", side_effect=fake_render),
    ):
        rc = csp.main(["--apply", "--json"])

    assert rc == 0
    out = capsys.readouterr().out
    # Locate the JSON payload line (starts with `[` and ends with `]`).
    payload_lines = [ln.strip() for ln in out.splitlines() if ln.strip().startswith("[")]
    assert payload_lines, f"No JSON payload found in stdout:\n{out}"
    parsed = json.loads(payload_lines[-1])
    assert parsed[0]["number"] == 113
    assert parsed[0]["pr_rollback_file"] == "/tmp/post-apply-prs.txt"
    assert parsed[0]["branch_rollback_sha"] == "sha-after-close"


def test_apply_json_status_lines_go_to_stderr_not_stdout(capsys):
    """End-to-end regression: when ``--apply --json`` is set, stdout
    contains ONLY the JSON dump.  Exercises the REAL ``close_and_debranch``
    with ``subprocess.run`` mocked (so ``gh`` / ``git`` are not actually
    invoked) -- this catches stdout leaks that originate from the
    destructive path itself, not just from ``main``.
    """
    import subprocess as _subprocess

    fake_rows = [_fake_row()]

    def fake_subprocess_run(cmd, *args, **kwargs):
        """Return a synthetic CompletedProcess for any ``gh`` / ``git`` call."""
        if cmd and cmd[0] == "git":
            # close_and_debranch's git ls-remote --heads query — provide a
            # realistic SHA so the row's branch_rollback_sha gets populated.
            return _subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout="abc12345deadbeef\trefs/heads/test-head\n",
            )
        # gh pr close / any other gh call: success.
        return _subprocess.CompletedProcess(args=cmd, returncode=0)

    with (
        mock.patch.object(csp, "fetch_open_prs", return_value=[]),
        mock.patch.object(csp, "filter_stale", return_value=fake_rows),
        mock.patch.object(csp.subprocess, "run", side_effect=fake_subprocess_run),
        mock.patch.object(
            csp, "render_json", side_effect=lambda r: json.dumps([x.__dict__ for x in r])
        ),
    ):
        rc = csp.main(["--apply", "--json"])

    assert rc == 0
    captured = capsys.readouterr()
    stdout = captured.out
    stderr = captured.err

    # stdout contains ONLY the JSON dump.
    payload_lines = [ln.strip() for ln in stdout.splitlines() if ln.strip().startswith("[")]
    assert payload_lines, f"No JSON payload found in stdout: {stdout!r}"

    # stdout must NOT contain any descriptive or progress markers.
    # (This is the critical regression test: if close_and_debranch's
    # 'closing #N' line leaks to stdout, jq can't parse the output.)
    assert "==>" not in stdout, f"stdout must NOT contain '==>': {stdout!r}"
    assert "closing" not in stdout, f"stdout must NOT contain 'closing': {stdout!r}"
    assert "Fetch" not in stdout, f"stdout must NOT contain 'Fetch': {stdout!r}"

    # Stderr carries all the descriptive messages.
    assert "==>" in stderr, f"stderr expected to carry '==>': {stderr!r}"
    assert "Fetch" in stderr, f"stderr expected to carry Fetch: {stderr!r}"
    assert "closing" in stderr, f"stderr expected to carry 'closing': {stderr!r}"
    assert "--apply" in stderr, f"stderr expected to carry --apply progress: {stderr!r}"

    # JSON parses cleanly from stdout.
    parsed = json.loads(payload_lines[-1])
    assert parsed[0]["number"] == 113
    # The destructive path populated the rollback pointers.
    assert parsed[0]["pr_rollback_file"] is not None
    assert parsed[0]["branch_rollback_sha"] == "abc12345deadbeef"


def test_json_alone_routes_fetching_status_to_stderr(capsys):
    """``--json`` alone (dry-run, no --apply) also routes status to stderr."""
    fake_rows = [_fake_row()]
    with (
        mock.patch.object(csp, "fetch_open_prs", return_value=[]),
        mock.patch.object(csp, "filter_stale", return_value=fake_rows),
        mock.patch.object(csp, "close_and_debranch") as mock_close,
        mock.patch.object(
            csp, "render_json", side_effect=lambda r: json.dumps([x.__dict__ for x in r])
        ),
    ):
        rc = csp.main(["--json"])

    assert rc == 0
    mock_close.assert_not_called()
    captured = capsys.readouterr()
    # Fetching and status lines on stderr.
    assert "==>" in captured.err
    assert "Fetching" in captured.err
    # stdout contains only JSON.
    payload_lines = [ln.strip() for ln in captured.out.splitlines() if ln.strip().startswith("[")]
    assert payload_lines
    # No contamination.
    assert "==>" not in captured.out
