"""Generate the frozen 1M-row DESI × Gaia DR3 cone benchmark snapshots.

Produces two deterministic parquet files inside ``tests/data/benchmark/``:

* ``desi_l.parquet`` — 10 000 rows, DESI-like targets (≈ 0.5 MB)
* ``gaia_r.parquet`` — 1 000 000 rows, Gaia DR3-like background (≈ 30 MB)

Both snapshots are bit-identical across regenerations (seed = 42, fully
deterministic ``Generator``), so the benchmark corpus is truly frozen —
a CONTRIBUTING-time edit to xmatch's matchers kernel that regresses
performance will produce the SAME bit-identical parquet on a fresh run,
which keeps subsequent benchmark runs comparable.

The snapshot columns match the real-catalogue subset that xmatch needs to
run a positional crossmatch: ``source_id`` (int64, distinct ranges per
side), ``ra``, ``dec``, ``rae``, ``dee`` (positional errors in arcsec),
and ``mag_g`` (a single-band flux used by the probabilistic engine).

Usage::

    python scripts/gen_benchmark_data.py

COMMIT POLICY
-------------

The two parquet files (~30 MB total) are meant to be committed to the repo,
not regenerated per-machine.  ``.gitignore`` does not exclude
``tests/data/`` so the snapshots land in version control the first time the
generator runs.  The snapshot is durable across numpy and polars minor
versions because the test reads semantic content via ``pl.read_parquet`` —
never bytes — so the snappy encoding may shift without invalidating results.

Schema contract
~~~~~~~~~~~~~~~

The two parquets share an implicit schema:

* ``source_id``   — ``int64`` (DESI side starts at 1,000,000; Gaia side starts at 10,000,000)
* ``ra``, ``dec`` — ``float64`` degrees
* ``rae``, ``dee`` — ``float64`` arcsec
* ``mag_g``      — ``float64`` uniform in [12, 21] for Gaia, [18, 22.5] for DESI

Renaming any column, changing its dtype, or shifting its units (e.g.
``rae`` → milliarcsec) invalidates the snapshot even if regeneration still
runs cleanly.  On any such change, bump ``SEED`` above and regenerate
instead of relying on the existing files.

Shape invariants
~~~~~~~~~~~~~~~~

Bench budgets in ``tests/test_benchmarks_io_1m.py`` are calibrated to BOTH
the schema AND the row-count + cone geometry:

* ``N_DESI = 10_000`` — left catalogue rows (matches-budget is computed
  from this N on the small side)
* ``N_GAIA = 1_000_000`` — right catalogue rows (matches-budget is
  computed from this N on the large side; the 4 GB RSS ceiling is also
  sized for 1 M right-side rows in memory)
* ``CENTRE_RA_DEG = 180.0``, ``CENTRE_DEC_DEG = -30.0``,
  ``CONE_RADIUS_DEG = 0.5`` — the cone geometry whose projected solid-angle
  drives expected matches (~12,300 at ``MATCH_RADIUS_ARCSEC = 2.0``)

Silently changing ``N_GAIA`` from 1 M to 100 K would invalidate the RSS
budget (which would never trigger) and the expected-match sanity floor
(which is calibrated to the full cone density).  Silently widening the cone
would inflate match counts and shift the parity baseline.  Bump ``SEED``
and regenerate alongside any such change.

* **Sample distribution** — uniform-in-cone via the flat-patch approximation
  (``_cone_sample_flat``, valid for cone ≲ 3°).  Realistic stellar-density
  gradients are intentionally NOT modelled; switching to non-uniform
  sampling on either side (e.g. Gaia star-density mocks or DESI-side
  realistic quasar density) shifts match counts AND invalidates the parity
  baseline derived from the first successful engine.  Any density change
  that materially shifts per-row counts (beyond the 5%/±500 parity
  tolerance) must be paired with bumping ``SEED`` and regenerating the bench
  fixtures; boundary-density shifts that stay within tolerance are fine.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

# --------------------------------------------------------------------------- #
# Snapshot specification (DO NOT edit lightly — bench budgets in
# tests/test_benchmarks_io_1m.py are tuned to these values).
# --------------------------------------------------------------------------- #
SEED: int = 42
N_DESI: int = 10_000  # left catalogue rows (DESI-like targets)
N_GAIA: int = 1_000_000  # right catalogue rows (Gaia DR3-like background)
CENTRE_RA_DEG: float = 180.0
CENTRE_DEC_DEG: float = -30.0
CONE_RADIUS_DEG: float = 0.5
DESI_RA_ERR_ARCSEC: float = 10.0  # DESI achieves ~10 mas per axis
DESI_DEC_ERR_ARCSEC: float = 10.0
GAIA_RA_ERR_ARCSEC: float = 0.5  # Gaia end-of-mission typical positional error
GAIA_DEC_ERR_ARCSEC: float = 0.5
MATCH_RADIUS_ARCSEC: float = 2.0  # typical DESI × Gaia crossmatch radius

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "tests" / "data" / "benchmark"


def _cone_sample_flat(
    *, n: int, ra0: float, dec0: float, cone_deg: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Sample (RA, Dec) uniformly in a small-angle patch around ``(ra0, dec0)``.

    For ``cone_deg`` < ~3° the spherical cap approximates a flat patch on the
    celestial sphere — and the flat sampling is O(n) faster than the
    rejection-based spherical-cap method (which discards ~half of the samples
    for thin cones anyway).
    """
    cos_dec = np.cos(np.radians(dec0))
    ra_off = rng.uniform(-cone_deg, cone_deg, size=n) / cos_dec
    dec_off = rng.uniform(-cone_deg, cone_deg, size=n)
    return ra0 + ra_off, dec0 + dec_off


def _assemble(
    *,
    ra: np.ndarray,
    dec: np.ndarray,
    rng: np.random.Generator,
    base_id: int,
    ra_err_arcsec: float,
    dec_err_arcsec: float,
    mag_range: tuple[float, float],
) -> pl.DataFrame:
    """Wrap ``(ra, dec)`` arrays in a polars DataFrame with realistic xmatch columns.

    Per-row errors are jittered by ±20% around the nominal value to mimic
    how real catalogues have a spread of measurement uncertainties.  Mag
    distribution is uniform in the given range — close enough for a memory +
    throughput benchmark; a full colour-colour joint distribution matters
    only for photometric-prior benchmarks (nway), and those run on a separate
    weight schema documented separately.
    """
    n = ra.size
    # Distinct int64 id ranges so DESI/Gaia ids never collide under joins.
    source_id = np.arange(n, dtype=np.int64) + base_id
    rae = np.full(n, ra_err_arcsec, dtype=np.float64) * rng.uniform(0.8, 1.2, size=n)
    dee = np.full(n, dec_err_arcsec, dtype=np.float64) * rng.uniform(0.8, 1.2, size=n)
    mag = rng.uniform(*mag_range, size=n).round(2)
    return pl.DataFrame(
        {
            "source_id": source_id,
            "ra": ra.astype(np.float64),
            "dec": dec.astype(np.float64),
            "rae": rae,
            "dee": dee,
            "mag_g": mag,
        }
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # DESI-like catalogue (small).
    desi_ra, desi_dec = _cone_sample_flat(
        n=N_DESI,
        ra0=CENTRE_RA_DEG,
        dec0=CENTRE_DEC_DEG,
        cone_deg=CONE_RADIUS_DEG,
        rng=rng,
    )
    desi_df = _assemble(
        ra=desi_ra,
        dec=desi_dec,
        rng=rng,
        base_id=1_000_000,
        ra_err_arcsec=DESI_RA_ERR_ARCSEC,
        dec_err_arcsec=DESI_DEC_ERR_ARCSEC,
        mag_range=(18.0, 22.5),
    )
    desi_path = OUTPUT_DIR / "desi_l.parquet"
    # Same row-group sizing as the Gaia file for consistency in any future
    # benchmark that wants to scan both side-by-side.  10K-row groups are
    # negligible overhead on a 10 KB file but keep the alignment cheap.
    desi_df.write_parquet(desi_path, compression="snappy", row_group_size=10_000)
    print(f"  {desi_path}  ({desi_path.stat().st_size / 1e6:.1f} MB, {len(desi_df):,} rows)")

    # Gaia-DR3-like catalogue (1M rows).  Use a separate offset of the same
    # rng so the two catalogues' rows are statistically independent but the
    # whole pair is still bit-identical given seed=42.
    gaia_rng = np.random.default_rng(SEED + 1)
    gaia_ra, gaia_dec = _cone_sample_flat(
        n=N_GAIA,
        ra0=CENTRE_RA_DEG,
        dec0=CENTRE_DEC_DEG,
        cone_deg=CONE_RADIUS_DEG,
        rng=gaia_rng,
    )
    gaia_df = _assemble(
        ra=gaia_ra,
        dec=gaia_dec,
        rng=gaia_rng,
        base_id=10_000_000,
        ra_err_arcsec=GAIA_RA_ERR_ARCSEC,
        dec_err_arcsec=GAIA_DEC_ERR_ARCSEC,
        mag_range=(12.0, 21.0),
    )
    gaia_path = OUTPUT_DIR / "gaia_r.parquet"
    gaia_df.write_parquet(gaia_path, compression="snappy", row_group_size=100_000)
    print(f"  {gaia_path}  ({gaia_path.stat().st_size / 1e6:.1f} MB, {len(gaia_df):,} rows)")

    # Expected match count at the documented radius (printed for sanity only).
    cone_area_arcsec2 = np.pi * (CONE_RADIUS_DEG * 3600) ** 2
    pi_r2 = np.pi * MATCH_RADIUS_ARCSEC**2
    expected = N_DESI * N_GAIA * pi_r2 / cone_area_arcsec2
    print(
        f'\nExpected matches @ r={MATCH_RADIUS_ARCSEC}" in '
        f"{CONE_RADIUS_DEG}° cone: ~{expected:,.0f}\n"
        "  (rule-of-thumb; actual count will be ≈ this ± sampling noise)"
    )


if __name__ == "__main__":
    main()
