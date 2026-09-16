# AGENTS.md — xmatch

Operational guidance, architectural standards, and workflow instructions for AI coding agents in `xmatch`.

---

## 1. Package Mission & Goals

`xmatch` is designed as the **definitive, high-performance, format-agnostic Swiss-Army-knife for astronomical catalogue crossmatching**:
1. **Universal I/O**: Local files (`.parquet`, `.fits`, `.csv`, `.tsv`), remote TAP/ADQL databases, cloud storage (`vos:` CANFAR VOSpace, S3, HTTP), and HATS spatial directory trees.
2. **Infinite Scale**: In-process small tables up to multi-billion-row full-sky surveys via Polars streaming pipelines, bounded-memory out-of-core partitions, HEALPix spatial zoning, and distributed Ray clusters.
3. **SOTA Crossmatch Science**: Full spectrum of positional and probabilistic matchers adapting to varying depths, astrometric uncertainties, proper-motion kinematics, and multi-band photometry (`sky`, `skyerr`, `skyellipse`, `target_epoch`, `pm_prior`, `lr`, `ml`, `xgb`, `auf`, `macauff`, `p_match`, `nway_match`, `fof_match`).
4. **Relational Joins & Master Unions**: Inner (`1and2`), left (`all1`), right (`all2`), full outer (`1or2`, `all`), anti-joins (`1not2`), relational ID joins (`id_join`), and sequential chains (`crossmatch_multi`).
5. **Local Mirroring & Offline Acceleration**: Remote queries and full surveys mirrored into a local HATS cache (`xmatch sync`) with rate-limiting, progress checkpoints, and fast local Arrow/Polars retrieval.
6. **Persistent Incremental Master Unions**: Assembling multi-wavelength photometric surveys into a unified, versioned HATS master dataset (`engine="ray-union"`) that serves as an evolving foundation dataset to train multimodal sky models.

---

## 2. Codebase Architecture Map

All source code lives in `src/xmatch/`:

| Module | Core Responsibility |
|---|---|
| [`crossmatch.py`](file:///scratch/src/xmatch/src/xmatch/crossmatch.py) | `CrossMatch` orchestrator, pairwise dispatch, `crossmatch_multi`, `union_match`, `nway_match`, `fof_match`. |
| [`matchers.py`](file:///scratch/src/xmatch/src/xmatch/matchers.py) | All spatial and probabilistic matching algorithms (`sky`, `skyerr`, `skyellipse`, `lr`, `ml`, `xgb`, `auf`, `macauff`). Dispatches to `fast` (SciPy `cKDTree`), `zone` (cdshealpix), `torchsky`, `astropy`, `stilts`. |
| [`ray_union.py`](file:///scratch/src/xmatch/src/xmatch/ray_union.py) | Distributed $N$-way full-outer-join union producing full-sky partitioned HATS trees (`Norder/Dir/Npix.parquet`) with resumable state. |
| [`ray_engine.py`](file:///scratch/src/xmatch/src/xmatch/ray_engine.py) | HEALPix pixel-level Ray task distribution for cone crossmatching. |
| [`mirror.py`](file:///scratch/src/xmatch/src/xmatch/mirror.py) | `xmatch sync`: Keyset TAP paging, remote HATS size/manifest sync, token-bucket rate limiter, pause/resume. |
| [`io_utils.py`](file:///scratch/src/xmatch/src/xmatch/io_utils.py) | Polars lazy scanning (`scan_frame`) and streaming sinks (`write_frame`, `write_hats`) for Parquet, CSV, TSV, FITS (Torchfits/Astropy), and HATS. |
| [`storage.py`](file:///scratch/src/xmatch/src/xmatch/storage.py) | Storage abstraction: `LocalStorage`, `VOSpaceStorage` (`vos:`), `S3Storage`, `HttpStorage`. |
| [`association.py`](file:///scratch/src/xmatch/src/xmatch/association.py) | Audit-grade `xmatch.association.v1` contract, deterministic SHA-256 IDs, release directories, component sidecars, equivalence tracking. |
| [`bayes.py`](file:///scratch/src/xmatch/src/xmatch/bayes.py) | Budavári & Szalay (2008) Bayes factor, KDE priors, stable log-sum-exp posterior. |
| [`cli.py`](file:///scratch/src/xmatch/src/xmatch/cli.py) | CLI subcommands: `match`, `sync`, `list`, `describe`, `search`, `discover`, `adopt`. |

Documentation resides exclusively in `docs/`:
- [`docs/api.md`](file:///scratch/src/xmatch/docs/api.md): Complete public Python API reference.
- [`docs/usage.md`](file:///scratch/src/xmatch/docs/usage.md): Practical user guide and cookbooks.
- [`docs/algorithms.md`](file:///scratch/src/xmatch/docs/algorithms.md): Algorithms survey, literature references, and Astropy vs SciPy cKDTree deep dive.
- [`docs/roadmap.md`](file:///scratch/src/xmatch/docs/roadmap.md): Strategic roadmap to v1.0, deep Polars integration, and release checklist.
- [`docs/association-v1.md`](file:///scratch/src/xmatch/docs/association-v1.md): Versioned association contract specification.

---

## 3. Environment & Verification Hygiene

### Mandatory Environment Settings

On CANFAR and networked environments, Python checks `~/.local` over NFS by default, which can cause significant import latency. Always enforce:

```bash
export PYTHONNOUSERSITE=1
unset PYTHONPATH
```

### Verification Tiers

To keep agent workflows responsive and avoid launching long-running tasks:

1. **Tier 1 — Fast Preflight (< 5s)**:
   Always run before committing or after formatting:
   ```bash
   pixi run preflight-push
   ```
2. **Tier 2 — Fast Targeted Unit Tests (~1–3s)**:
   Target only the modified module:
   ```bash
   export PYTHONNOUSERSITE=1
   pixi run test tests/test_io_utils.py -v
   pixi run test tests/test_astro_utils.py -v
   ```
3. **Tier 3 — CI Smoke Gate (~10–15s)**:
   Verify the documentation-to-test smoke guards:
   ```bash
   export PYTHONNOUSERSITE=1
   pixi run test tests/test_ci_smoke.py -v
   ```
4. **Tier 4 — Full Test Suite (Opt-in Before Major Releases)**:
   *Do NOT run the full test suite casually* (running ~160 tests with Ray and Astropy can take several minutes):
   ```bash
   export PYTHONNOUSERSITE=1
   pixi run test -m "not slow and not bench"
   ```

---

## 4. Remotes & Git Workflow (AstroAI Fork)

| Remote | Points at | Purpose |
|---|---|---|
| `origin` | `<user>/xmatch` | Main development fork (push to `main` or topic branches) |
| `upstream` | `astroai/xmatch` | Canonical repo; sync `main`; target for Pull Requests |

```bash
# Sync with canonical upstream
git fetch upstream && git rebase upstream/main

# Push to personal fork
git push -u origin HEAD

# Create Pull Request
gh pr create -R astroai/xmatch --head sfabbro:main
```

- Commit messages must follow Conventional Commits (`feat:`, `fix:`, `docs:`, `perf:`, `refactor:`, `test:`).
- Always ensure `pixi run preflight-push` passes before opening or updating a PR.
