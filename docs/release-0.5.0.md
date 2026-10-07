# xmatch 0.5.0

Released 2026-10-07 through the private GitHub repository at tag `v0.5.0`.
Repository access is required. This project is not published on PyPI because
another project owns the `xmatch` name.

This release adds source-preserving observation and candidate releases,
versioned association snapshots and deltas, HATS mirroring and distributed
full-sky union, epoch-aware uncertainty propagation, and additional
probabilistic matchers. See the
[CHANGELOG.md](https://github.com/astroai/xmatch/blob/v0.5.0/CHANGELOG.md) for details.

The GitHub release assets are the CI-tested wheel and source distribution,
with `SHA256SUMS`. Download the wheel from the
[v0.5.0 release](https://github.com/astroai/xmatch/releases/tag/v0.5.0)
(repository access required), then install it:

```bash
python -m pip install ./xmatch-0.5.0-py3-none-any.whl
```

The final local `ci-local` run passed 652 tests, with 9 skipped and 33
deselected, in 456.02 s; it reported 13 ERFA warnings. Ruff, formatting, Mypy
over 69 files and compilation also passed. The latest recorded Ubuntu run, at
`f86f5a6`, passed 645 tests, with 12 skipped and 33 deselected, plus lint,
typing, compilation, archive-byte and metadata checks, and installed-wheel CLI
verification. The revised live ESA remote-first test passed separately (1
passed in 225.40 s); PyVO deleted its UWS job in 80.67 s.
On CANFAR, eight independent worker sessions, each with one CPU, matched
1,000,000 rows per input in a 60.004 s median, compared with 283.778 s on one
worker (4.73×). See the
[CANFAR scaling report](https://github.com/astroai/xmatch/blob/v0.5.0/docs/canfar-scaling-2026-10-07.md) for run details.
