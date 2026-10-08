# xmatcher v0.5.1 (unreleased)

## Breaking rename

- Python imports and the command-line executable change from `xmatch` to
  `xmatcher`.
- The bundled and user configuration file changes from `xmatch.yaml` to
  `xmatcher.yaml`.
- Configuration environment variables change from the `XMATCH_` prefix to
  `XMATCHER_`.
- The default cache path changes from `~/.cache/xmatch` to
  `~/.cache/xmatcher`. The documented CANFAR working/output root is
  `/arc/projects/hats/xmatcher`.

The `xmatch.*` association schema identifiers remain stable. Existing v0.5.0
GitHub release assets and historical provenance such as
`software_version: xmatch:0.5.0` are unchanged.

## Migration

To reuse an existing user configuration, copy
`~/.config/xmatch/xmatch.yaml` to `~/.config/xmatcher/xmatcher.yaml`.
The `adopt` command writes catalogue entries to this new user configuration.
To reuse the previous local cache, set
`XMATCHER_CACHE_ROOT=~/.cache/xmatch`. The shared CANFAR cache base
`/arc/projects/hats` is unchanged; the xmatcher working/output directory is
`/arc/projects/hats/xmatcher`.

## Installation

PyPI publication is pending. Install from the current source checkout:

```bash
git clone https://github.com/astroai/xmatcher.git
cd xmatcher
python -m pip install ".[cds,hats-ray,torchfits,ml]"
```

## Verification

The final source passed the full Pixi suite: 686 passed, 9 skipped, and 13
warnings in 1000.24 seconds. The live ESA TAP test passed in 771.77 seconds;
the suite included 32 benchmark checks. `pixi run ci-local` passed with 653
passed, 9 skipped, 33 deselected, and 13 warnings in 179.63 seconds.

Optional engine checks passed: 11 passed with no skips for Torchsky, HATS,
Healpy, Torchfits, and related fallback coverage; the XGBoost/Ray parity gate
passed 2 checks with 14 other cases deselected (Ray 2.59.0, XGBoost 3.4.2).
These local checks do not include a new CANFAR compute run.

GitHub release CI and published-asset checksums remain pending until the
v0.5.1 tag is created.
