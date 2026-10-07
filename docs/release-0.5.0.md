# xmatch 0.5.0

Released 2026-10-07 at the public GitHub release for tag `v0.5.0`. The release
was first published while the repository was private; the repository and its
release assets are now public. No PyPI package is published because another
project owns the `xmatch` name.

This release adds source-preserving observation and candidate releases,
versioned association snapshots and deltas, HATS mirroring and distributed
full-sky union, epoch-aware uncertainty propagation, and additional
probabilistic matchers. See the
[CHANGELOG.md](https://github.com/astroai/xmatch/blob/v0.5.0/CHANGELOG.md) for details.

The GitHub release assets are the CI-tested wheel and source distribution,
with `SHA256SUMS`. Download the wheel from the
[v0.5.0 release](https://github.com/astroai/xmatch/releases/tag/v0.5.0), then
install it:

```bash
python -m pip install ./xmatch-0.5.0-py3-none-any.whl
```

On the unchanged v0.5.0 source tree, the final local full suite passed 684
tests, with 10 skipped and 13 warnings, in 489.60 s. Separate optional
integration checks passed: 17 Torchsky tests, 23 Torchfits I/O tests, one
Healpy oracle test, and two XGBoost/Ray parity tests.
The unchanged live ESA remote-first test also passed separately (1 passed in
225.40 s); PyVO deleted its UWS job in 80.67 s. The GitHub Actions Linux
release gate for tag `v0.5.0` passed with 649 passed, 12 skipped, 33
deselected, and 13 ERFA warnings in 178.02 s; archive, metadata, and
installed-wheel checks also passed. See
[PR #204](https://github.com/astroai/xmatch/pull/204) and the
[CI run](https://github.com/astroai/xmatch/actions/runs/37691455480).
On CANFAR, eight independent worker sessions, each with one CPU, matched
1,000,000 rows per input in a 60.004 s median, compared with 283.778 s on one
worker (4.73×). See the
[CANFAR scaling report](https://github.com/astroai/xmatch/blob/v0.5.0/docs/canfar-scaling-2026-10-07.md) for run details.
