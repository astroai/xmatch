# Stellar Catalogue Pipeline

> **Planning note, not a verified inventory or execution recipe.** Survey names,
> VizieR identifiers, local/HATS paths, sizes, Hugging Face data, and CANFAR
> image tags below are candidates from an earlier design pass. Verify current
> availability, schemas, permissions, storage costs, and image tags before
> scheduling work. The pipeline described here is not implemented in this
> repository. Pairwise native HATS matching materializes globally for adaptive,
> RING, mixed-order, or epoch-aligned cases; its partition-neighbor path is
> limited to fixed-radius `sky` matching over uniform-order NESTED inputs.
> Distributed `ray-union` handles mixed-order/RING layouts and uncertainty or
> epoch halos with compressed interval planning, which can require extra scans.
> These engine-specific paths are not a general bounded-memory guarantee.

## Goal

Build a HATS-based crossmatch pipeline on CANFAR that:
1. Caches survey catalogs locally (`/arc/projects/hats/`, `/scratch/` for ephemeral)
2. Crossmatches them against Gaia DR3 (full catalog, not just XP stars)
3. Applies proper-motion corrections via xmatch
4. Produces a general-purpose master HATS catalog (not MSA-specific)
5. Enables efficient streaming for future MSA pretraining

## Candidate Catalog Inventory (unverified)

### Tier 0 — Core (must have)

| Survey | VizieR | HATS Path | Coord Cols | Key Columns | Size |
|--------|--------|-----------|------------|-------------|------|
| Gaia DR3 (photometry) | I/355/gaiadr3 | `gaia_dr3/I/355/gaiadr3/` | `RA_ICRS`, `DE_ICRS` | Gmag, BPmag, Rmag, Plx, pmRA, pmDE, RUWE, Teff, logg, [Fe/H] | ~10GB |
| Gaia DR3 XP | I/355/xpcont | Separate product | `RA_ICRS`, `DE_ICRS` | bp_coefficients[55], rp_coefficients[55] | ~25GB |
| CatWISE2020 | II/365/catwise | `catwise2020/II/365/catwise/` | `RA_ICRS`, `DE_ICRS` | mW1, mW2, pmRA, pmDE | ~20GB |
| Pan-STARRS DR2 | II/389/ps1_dr2 | `ps1_dr2/II/389/ps1_dr2/` | `RAJ2000`, `DEJ2000` | gmag, rmag, imag, zmag, ymag | ~30GB |
| SkyMapper DR4 | II/379/smssdr4 | `skymapper_dr4/II/379/smssdr4/` | `RAICRS`, `DEICRS` | uPSF, vPSF, gPSF, rPSF, iPSF, zPSF | ~15GB |
| 2MASS | II/246/out | `2mass/II/246/out/` | `RAJ2000`, `DEJ2000` | Jmag, Hmag, Kmag | ~5GB |

### Tier 1 — High value (include if possible)

| Survey | VizieR | HATS Path | Coord Cols | Key Columns |
|--------|--------|-----------|------------|-------------|
| GALEX GR6+7 | II/335/galex_ais | `galex_gr67/II/335/galex_ais/` | `RAJ2000`, `DEJ2000` | FUVmag, NUVmag |
| S-PLUS DR4 | II/380/splusdr4 | `splus_dr4/II/380/splusdr4/` | `RAJ2000`, `DEJ2000` | 12 bands (u, J0378-J0861, g, r, i, z) |
| VHS DR5 | II/367/vhs_dr5 | `vhs_dr5/II/367/vhs_dr5/` | `RAJ2000`, `DEJ2000` | Ypmag, Jpmag, Hpmag, Kspmag |

### Tier 2 — Future (user provides)

| Survey | Status |
|--------|--------|
| Pristine DR2 | User will provide HATS catalog |
| UNIONS | User will provide HATS catalog |

### Not included

- **SDSS**: DR19 has no new photometry. DR16 on VizieR is old. Skip for now — PS1 + SMSS + S-PLUS cover similar wavelengths.
- **UKIRT/UHS**: UHS not on VizieR. UKIDSS available but VHS covers similar range.

Approximate sizes and endpoint/product paths in the table are not checked
against current archive metadata. Treat them as rough planning placeholders.

## Gaia XP Strategy (options to verify)

Three options (in order of preference):

1. **Check whether `UniverseTBD/mmu_gaia_gaia` HATS is available and licensed** — the reported row count, schema, and URL are unverified. If suitable, test opening it and converting a small sample to local HATS before planning a full transfer.

2. **Check the proposed bulk XP CSV.gz files** at `/arc/projects/k-pop/spectra/gaia/dr3/` — the path, file count, schema, and availability depend on the CANFAR account/project. Confirm access and inspect representative files before using them.

3. **Investigate VizieR I/355/xpcont TAP** — confirm that the current service exposes the required coefficients and supports a practical query size before relying on chunked downloads.

## Proposed Architecture (not implemented)

```
hats-catalog-pipeline/
├── pyproject.toml              # pixi config, minimal deps
├── configs/
│   └── catalogs.yaml           # catalog registry: paths, coords, columns, PM info
├── scripts/
│   ├── fetch.py                # download VizieR HATS → /arc/projects/hats/
│   ├── fetch_gaia_xp.py        # fetch XP coefficients (HATS or bulk CSV)
│   ├── crossmatch.py           # xmatch with PM propagation via torchsky engine
│   └── build_master.py         # assemble final HATS catalog
└── src/hats_catalog/
    ├── fetch.py                # catalog fetching logic
    ├── crossmatch.py           # crossmatch orchestration
    ├── pm.py                   # PM propagation (via xmatch --target-epoch)
    └── schema.py               # column definitions, catalog metadata
```

## Data Flow

```
Step 1: fetch.py
  VizieR HATS URLs ──→ /arc/projects/hats/{catalog_name}/
  (persistent cache, run once per catalog)

Step 2: fetch_gaia_xp.py
  UniverseTBD/mmu_gaia_gaia ──→ /arc/projects/hats/gaia_xp/
  OR /arc/projects/k-pop/spectra/gaia/dr3/*.csv.gz ──→ /arc/projects/hats/gaia_xp/
  (XP coefficients with source_id for joining)

Step 3: crossmatch.py
  Gaia DR3 reference epoch J2016.0
    × CatWISE2020 propagated to J2016.0 only where valid epoch and PM exist
    × other surveys with measured motion propagated to J2016.0
    × surveys without PM either matched at their declared epoch or handled
      with an explicitly accepted missing-motion prior and positional errors
  ──→ /arc/projects/hats/gaia_xmatch/

Step 4: build_master.py
  Gaia XP (via source_id join) + crossmatch results
  ──→ /arc/projects/hats/master/
```

## Storage Layout

```
/arc/projects/hats/                    # persistent
  gaia_dr3/                            # unverified size estimate; measure current product
  gaia_xp/                             # unverified size estimate; measure current product
  catwise2020/                         # unverified size estimate; measure current product
  ps1_dr2/                             # unverified size estimate; measure current product
  skymapper_dr4/                       # unverified size estimate; measure current product
  twomass/                             # unverified size estimate; measure current product
  galex/                               # unverified size estimate; measure current product
  splus_dr4/                           # unverified size estimate; measure current product
  vhs_dr5/                             # unverified size estimate; measure current product
  gaia_xmatch/                         # intermediate crossmatch
  master/                              # final catalog

/scratch/                              # ephemeral, fast
  hats-work/                           # temporary during crossmatch
```

## CANFAR Execution

### Images to verify before use

- `images.canfar.net/astroai/base:26.07` — proposed image only; verify it exists and contains the required tools.
- `images.canfar.net/astroai/ray-manager:26.07` + `ray-worker:26.07` — proposed tags only; verify availability, compatibility, and cluster configuration.

### Job Sequence

1. **Session 1**: `fetch.py` — download all VizieR HATS catalogs (~hours)
2. **Session 2**: `fetch_gaia_xp.py` — fetch XP coefficients (~hours)
3. **Session 3**: `crossmatch.py` — sequential left joins (~days, I/O bound)
4. **Session 4**: `build_master.py` — merge XP + crossmatch (~hours)

### Dependencies

```
hats-import    # FITS → HATS conversion
lsdb           # HATS catalog I/O
xmatch         # crossmatch orchestration
torchsky       # HEALPix-accelerated matching
torchfits      # FITS I/O
pyarrow        # Parquet
astropy        # FITS/VOTable
healpy         # HEALPix
polars         # data manipulation
```

## PM Propagation

- A target epoch is only valid when each moved source has a known reference
  epoch and finite proper motion, or when the user explicitly enables and
  accepts `pm_prior` for missing-motion rows. Gaia DR3 positions are already
  referred to J2016.0; they do not need propagation when that is the target.
- CatWISE and every other catalogue need verified epoch and motion metadata;
  column names alone are not evidence of units or conventions.
- Catalogues without proper motion should not silently be treated as
  stationary across epoch differences. Either compare at a scientifically
  justified native epoch, supply a validated motion model, or use an explicitly
  accepted missing-motion prior with suitable positional uncertainty.
- `xmatch` validates frames, epochs, and motion metadata; it does not infer
  missing scientific metadata. Consult `docs/usage.md` before planning the full
  survey scale because memory behavior depends on input ordering and matcher.

## Open Questions

1. **Gaia XP source**: Option 1 (HuggingFace HATS) vs Option 2 (bulk CSV.gz on CANFAR)? Option 1 is simpler; Option 2 uses existing local data.

2. **SDSS**: Skip entirely, or import DR16 photometry from VizieR for completeness? PS1 + SMSS + S-PLUS already cover u/g/r/i/z wavelengths.

3. **Crossmatch order**: Sequential left joins (Gaia × CatWISE, then result × PS1, etc.) — simple, robust. Or N-way Bayesian (xmatch.nway_match) — more complex but handles degeneracies.

4. **Pixel threshold**: How many rows per HEALPix partition? Current MSA uses 2M rows/key. Suggest 1M for HATS (balances memory vs parallelism).

## Implementation Priority

1. **Phase 1**: Repo setup + `fetch.py` + `configs/catalogs.yaml`
2. **Phase 2**: `fetch_gaia_xp.py` (XP coefficients)
3. **Phase 3**: `crossmatch.py` (the core)
4. **Phase 4**: `build_master.py` (final assembly)
5. **Phase 5**: Streaming dataset for MSA (separate work)
