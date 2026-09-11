# AGENTS.md — xmatch

## Remotes (AstroAI fork workflow)

| Remote | Points at | Use |
|--------|-----------|-----|
| `origin` | `sfabbro/xmatch` | Push `wip/*` only |
| `upstream` | `astroai/xmatch` | Sync `main`; PR target |

`main` tracks `upstream/main`. Never force-push `astroai` `main`.

```bash
git fetch upstream && git rebase upstream/main
git checkout -b wip/<topic>
git push -u origin HEAD
gh pr create -R astroai/xmatch --head sfabbro:$(git branch --show-current)
```

Workspace layout, CANFAR, and `/arc` install rules: parent workspace `AGENTS.md` (`~/src/AGENTS.md`).

## Environment

Pixi only. Do not use system Python or bare pip when the Pixi env is available.

```bash
pixi install
```

## Verification

- Fast: `pixi run preflight-push`
- Before push: `pixi run ci-local`
- Opt-in: `pixi run test -m "not slow and not bench"`

Read `.cursor/harness/config.json` and the README for task-specific checks.
