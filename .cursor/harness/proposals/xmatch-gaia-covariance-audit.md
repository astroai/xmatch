# Xmatch Gaia epoch-covariance audit

status: proposal only — no production code changed

## What exists

- `CatalogueSource` already has `ra_err_column`, `dec_err_column`, and
  `corr_column`.  `skyellipse` already forms a two-dimensional local tangent
  covariance and applies a Mahalanobis acceptance test in all supported local
  engines (`fast`, `zone`, and `astropy`).
- The three Gaia source entries (`gaia_esa`, `gaia_noao`, `gaia_cds`) declare
  the positional uncertainty/correlation fields, but their `default_columns`
  omit them.  A remote `skyellipse` request therefore silently uses the
  configured `default_pos_error_arcsec` unless a user happens to request the
  fields manually.
- `_download_remote` force-adds the seven mean kinematic columns for
  `target_epoch`, but no uncertainty or correlation columns.  The TAP
  projection code otherwise faithfully selects every requested field.
- `MatchRequest`/`SideOverrides` can override RA/Dec, errors, and mean
  kinematics, but not `corr_column`; the CLI exposes no uncertainty/correlation
  or local kinematic mapping flags.  Consequently, custom local Gaia-like
  input cannot request `skyellipse` metadata through the public CLI.
- `_apply_proper_motion` overwrites the catalogue coordinate columns with
  propagated means.  It has no side channel for propagated covariances, so
  `_pos_covariance` still interprets the *reference-epoch* error ellipse at
  the target epoch.  `pm_prior` adds an isotropic heuristic drift term; it is
  not a substitute for the correlated Gaia covariance propagation.
- Torchsky `propagate_space_motion` has an autograd-finite 6x6 Jacobian test,
  but no public batched covariance/Jacobian API.  Xmatch calls it only for
  propagated means.  The current `astro_utils` adapter returns only RA/Dec.

## Gaia data model facts that govern the scope

Gaia DR3 publishes standard errors and correlations for the five astrometric
quantities `(alpha*, delta, parallax, pmra, pmdec)`, where `alpha*` is the
cosine-weighted RA tangent coordinate.  It does **not** publish a joint
astrometry--spectroscopic-RV covariance in `gaia_source`.  In particular,
using `radial_velocity_error` as a sixth diagonal element would imply
independence rather than establish it, and the required cross terms are absent.

The primary Gaia documentation explicitly defines the five-dimensional error
vector and the ten pairwise correlations:

- [Gaia EDR3 transformations and error propagation](https://gea.esac.esa.int/archive/documentation/GEDR3/Data_processing/chap_cu3ast/sec_cu3ast_intro/ssec_cu3ast_intro_tansforms.html)
- [Gaia DR3 `gaia_source` data model](https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_main_source_catalogue/ssec_dm_gaia_source.html)

Therefore the first supported contract should be: **propagate the Gaia 5x5
astrometric covariance conditional on the measured RV**.  The RV affects the
mean 6D trajectory, but its measurement uncertainty is deliberately excluded
from the formal propagated covariance.  This is scientifically honest and is
strictly better than treating old-epoch errors as target-epoch errors.

## Minimal implementation path

### 1. Torchsky: one narrow public primitive first

Add a tensor-native function beside `propagate_space_motion`, rather than
putting finite differences in Xmatch:

```python
propagate_space_motion_position_jacobian(
    lon_deg, lat_deg, pm_lon_coslat_mas_yr, pm_lat_mas_yr,
    parallax_mas, radial_velocity_km_s, *,
    source_epoch_jyear, target_epoch_jyear,
) -> Tensor  # (..., 2, 5)
```

Its derivative must map the input Gaia basis
`(delta_alpha_star_mas, delta_delta_mas, delta_parallax_mas,
delta_pmra_mas_yr, delta_pmdec_mas_yr)` to the **target-epoch local tangent
plane** `(delta_alpha_star_arcsec, delta_delta_arcsec)`.  It should use the
existing differentiable 6D propagation core and autograd/vmap internally; it
must not expose a RA-in-degrees Jacobian because longitude wrapping and polar
coordinates make that basis unsuitable for covariance transport.  RV remains
an input to the mean trajectory but has no Jacobian column in this first API.

Xmatch then computes, row-wise, `C_target = J @ C_gaia5 @ J.T`.  Symmetrise
numerically with `(C + C.T)/2`; reject nonfinite/non-positive-semidefinite
results for the covariance path and retain the existing position-only ellipse
or configured floor.  Do not manufacture zero PM/parallax errors for missing
five-parameter Gaia rows.

An analogous 4-parameter angular-motion Jacobian is a later extension.  It
is not needed for the first Gaia 5D path and should not delay it.  The
existing `pm_prior` remains the fallback for sources with no usable covariance.

### 2. Xmatch metadata: make the complete Gaia covariance declarative

Extend `CatalogueSource` with a single optional ordered mapping, e.g.
`astrometric_covariance_columns: dict[str, str]`, containing these fifteen
Gaia fields:

```text
ra_error, dec_error, parallax_error, pmra_error, pmdec_error,
ra_dec_corr, ra_parallax_corr, ra_pmra_corr, ra_pmdec_corr,
dec_parallax_corr, dec_pmra_corr, dec_pmdec_corr,
parallax_pmra_corr, parallax_pmdec_corr, pmra_pmdec_corr
```

Use a fixed internal canonical order; do not add fifteen individual dataclass
attributes.  Keep the existing three position-field attributes for backwards
compatibility and ordinary `skyellipse`.

Populate the ESA/NOIRLab Gaia entries with the lowercase Gaia names.  CDS
should be deferred until its exact VizieR DR3 column names and units are
verified; `RADECor` alone does not establish names for the other nine
correlations.  The functionality should simply fall back for that provider,
rather than guessing or mis-projecting a covariance.

`SideOverrides` needs one matching optional mapping for Python callers.  For
the CLI, add only `--astrometric-covariance-columns-1/2` as a compact
comma-separated canonical-name mapping (or defer CLI mapping until the local
kinematic mapping work).  Fifteen standalone flags would be needless surface
area.  `--describe` and `doctor` should display/check the mapping as one
metadata field.

### 3. Projection and local matching

When either `matcher == "skyellipse"` or `target_epoch is not None`, union the
currently requested/default remote columns with positional-error/correlation
metadata.  When `target_epoch` and complete covariance metadata are both
present, also union all fifteen covariance fields.  This must happen in
`CrossMatch._download_remote`, before `download_from_tap`; it covers explicit
restrictive `columns_1/2` requests exactly as the present kinematic forcing
does.

Store propagated covariance as two private per-row columns (three scalars:
variance alpha*, variance delta, covariance alpha*/delta) during
`_apply_proper_motion`.  Update `_pos_covariance` to prefer those columns,
then add the configured floor and any PM-prior term in the same local tangent
basis.  This avoids changing the public result schema because `_build_result`
already strips private PM-drift fields; extend that stripping to the covariance
columns.  The existing `corr_column` path remains the correct reference-epoch
fallback.

Remote epoch matching already bypasses server-side TAP/CDS matching.  Preserve
that rule: target-epoch covariance propagation must run locally with the
propagated coordinates and the corresponding target-epoch covariance.

### 4. Tests that prove the scientific contract

1. Unit-test construction of a Gaia 5x5 covariance from errors/correlations,
   including symmetry, units (mas vs mas/yr), a PSD bad-row fallback, and no
   zero-imputation.
2. Torchsky test: `J @ C @ J.T` agrees with a Monte-Carlo propagation of a
   well-conditioned Gaia-like state away from a pole and RA wrap.  Test batched
   CPU/CUDA consistency where CUDA is available.
3. Xmatch target-epoch test: a high-PM synthetic source has a materially
   larger/differently oriented target ellipse; `skyellipse` accepts/rejects a
   candidate according to the transported covariance, not its reference
   ellipse.  Include RA=359.999° and high declination cases.
4. Remote mock: an explicit minimal `columns_2` still receives every required
   covariance field for `skyellipse + target_epoch`; a source without declared
   complete covariance does not request guessed fields.
5. Regression: non-target `skyellipse`, ordinary `skyerr`, incomplete Gaia
   rows, HATS rejection, and output schema retain today’s behavior.

## Explicit non-goals for this increment

- No fake 6x6 Gaia/RV covariance and no `radial_velocity_error` contribution
  without published cross-covariances.
- No covariance propagation for HATS/out-of-core target-epoch matching, which
  is already deliberately unsupported.
- No replacement of all generic catalogue error metadata with Gaia-specific
  fields; the mapping is optional and only activates when fully populated.
- No change to the candidate cone policy beyond the existing conservative
  maximum-eigenvalue prefilter; target covariance simply feeds that same policy.

## Files that would change

- Torchsky: `src/torchsky/wcs/frames.py`, `src/torchsky/wcs/__init__.py`,
  `tests/test_wcs_frames.py`.
- Xmatch: `src/xmatch/sources.py`, `request.py`, `crossmatch.py`,
  `matchers.py`, `xmatch.yaml`, `cli.py`, plus focused remote/matcher/CLI tests
  and user documentation.

## Recommended sequence

1. Land and validate the Torchsky tangent-plane Jacobian primitive.
2. Add Xmatch’s declarative Gaia mapping + forced remote projection and a
   reference-epoch `skyellipse` regression; this alone fixes silent remote
   error/floor degradation.
3. Wire Jacobian covariance transport into target-epoch local matching.
4. Add compact CLI local mappings only after the Python/config contract is
   stable.
