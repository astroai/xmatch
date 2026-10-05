# Contributing to xmatch

Thank you for contributing to `xmatch`!

## Development Setup

We recommend [Pixi](https://pixi.sh) for a reproducible environment with all compiled dependencies (`polars`, `scipy`, `astropy`, `cdshealpix`, `ray`):

```bash
git clone https://github.com/astroai/xmatch.git
cd xmatch
pixi install
```

On shared or NFS-mounted systems (such as CANFAR), isolate user site-packages before running Python or tests:

```bash
export PYTHONNOUSERSITE=1
unset PYTHONPATH
```

## Linting, Formatting, and Tests

Run the fast preflight check (lint, format check, and bytecode compilation) before committing:

```bash
pixi run preflight-push
```

Auto-format and lint with Ruff:

```bash
pixi run format
pixi run lint
```

Run targeted unit tests or the full non-slow test suite:

```bash
# Targeted module tests
pixi run test tests/test_matchers.py -q

# Full unit test suite (excluding network-dependent slow and benchmark tests)
pixi run test -m "not slow and not bench"
```

## Pull Requests

- Follow [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `perf:`, `refactor:`, `test:`).
- Ensure `pixi run preflight-push` and relevant unit tests pass before opening a pull request against `astroai/xmatch`.
