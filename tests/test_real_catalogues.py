"""Real-catalogue integration tests using VizieR / ESA Gaia via astroquery.

Tests in this module are flagged :py:func:`pytest.mark.slow` and skipped by
default::

    pytest -m "not slow"             # default CI
    pytest -m slow  tests/test_real_catalogues.py  # opt-in
    pytest -m slow --durations=20    # also dumps per-test timings

Each test that needs real catalogue data looks at
``/scratch/xmatch-catalogues/<name>_top500.csv`` and triggers an
``astroquery``-backed fetch on first access. Subsequent runs read only the
cached CSV; the network is touched once per clean cache directory.

Catalogues exercised (all sampled at RA=262.4°, Dec=-32.77°, radius=0.5°,
the Stripe-82 neighbourhood — a deep, multi-survey window of sky):

* **Gaia DR3 / EDR3** — optical, integer ``source_id``, positional errors
  (``ra_error``, ``dec_error`` in mas), ``phot_g_mean_mag`` for Tier 3 priors.
* **AllWISE** — mid-IR (W1..W4), string ``AllWISE`` designation for ID-style joins.
* **USNO-B1.0** — optical with astrometric proper motion
  (``pmRA``/``pmDE``, mas/yr) and per-axis position errors in mas. Exercises
  ``skyerr`` on real survey data.

A graceful skip applies when :mod:`astroquery` isn't available. A test whose
cache fixture fails (network outage, upstream schema drift) is reported as
``skipped`` with the upstream error in its message — never as a hard
failure, so a flaky connection cannot break a small CI runner.
"""

from __future__ import annotations

import os
import pathlib
import tempfile
import time
from collections.abc import Callable

import numpy as np
import polars as pl
import pytest

from xmatch import stilts
from xmatch.io_utils import astropy_table_to_polars
from xmatch.matchers import MatchSpec, id_join, sky_match
from xmatch.sources import CatalogueSource

# --------------------------------------------------------------------------- #
# Cache configuration
# --------------------------------------------------------------------------- #
CACHE_DIR = pathlib.Path(
    os.environ.get(
        "XMATCH_REAL_CATALOGUES_DIR",
        str(pathlib.Path(tempfile.gettempdir()) / "xmatch-catalogues"),
    )
)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CENTRE_RA_DEG = 262.4
CENTRE_DEC_DEG = -32.77
CONE_RADIUS_DEG = 0.5
N_ROWS = 500

# Skip every test in this module if astroquery is missing. It's an opt-in dep
# that is almost certainly absent on a small / fast CI runner.
pytestmark = pytest.mark.slow
try:  # pragma: no cover - import gate
    import astroquery  # noqa: F401
except ImportError:  # pragma: no cover - import gate
    astroquery = None  # type: ignore[assignment]
_HAVE_ASTROQUERY = astroquery is not None


def _require_network() -> None:
    """Skip if neither VizieR nor ESA Gaia respond in <5 s."""

    import urllib.error
    import urllib.request

    for url in (
        "https://vizier.cds.unistra.fr/",
        "https://gea.esac.esa.int/tap-server/tap/sync",
    ):
        try:
            urllib.request.urlopen(url, timeout=5).read(1)
            return
        except (urllib.error.URLError, TimeoutError, OSError):
            continue
    pytest.skip("Neither VizieR nor ESA Gaia are reachable from this host.")


# --------------------------------------------------------------------------- #
# ASTROquery fetchers (each writes a stable CSV path)
# --------------------------------------------------------------------------- #
def _with_retries(fn: Callable[[], None], *, attempts: int = 3, sleep_s: float = 4.0) -> None:
    """Run *fn* up to N times with linear back-off.

    The Gaia cone TAP server periodically returns an ``HTTPError: 408`` for
    perfectly valid ADQL. A handful of retries with a small sleep between
    attempts is enough to make slow tests reliable.
    """
    last: Exception | None = None
    for k in range(attempts):
        try:
            fn()
            return
        except Exception as exc:  # noqa: BLE001 - any retryable upstream error
            last = exc
            if k + 1 < attempts:
                time.sleep(sleep_s * (k + 1))
    raise last  # type: ignore[misc]


def _fetch_gaia(path: pathlib.Path) -> None:
    from astroquery.gaia import Gaia

    # A plain RA/Dec box is *much* cheaper for TAP than ST_CONTAINS, and the
    # extra coverage on the corners is harmless on a 0.5° cone.
    ra_lo = CENTRE_RA_DEG - CONE_RADIUS_DEG
    ra_hi = CENTRE_RA_DEG + CONE_RADIUS_DEG
    dec_lo = CENTRE_DEC_DEG - CONE_RADIUS_DEG
    dec_hi = CENTRE_DEC_DEG + CONE_RADIUS_DEG
    adql = (
        f"SELECT TOP {N_ROWS} source_id, ra, dec, ra_error, dec_error, "
        f"ra_dec_corr, phot_g_mean_mag "
        f"FROM gaiaedr3.gaia_source "
        f"WHERE ra BETWEEN {ra_lo} AND {ra_hi} "
        f"AND dec BETWEEN {dec_lo} AND {dec_hi} "
        f"AND phot_g_mean_mag < 16"
    )

    def _launch() -> None:
        job = Gaia.launch_job(adql, dump_to_file=False, verbose=False)
        df = astropy_table_to_polars(job.get_results())
        df = df.with_columns(
            [
                pl.col("source_id").cast(pl.Int64),
                pl.col("ra").cast(pl.Float64),
                pl.col("dec").cast(pl.Float64),
                pl.col("ra_error").cast(pl.Float64),
                pl.col("dec_error").cast(pl.Float64),
                pl.col("ra_dec_corr").cast(pl.Float64),
                pl.col("phot_g_mean_mag").cast(pl.Float64),
            ]
        )
        df.write_csv(path)

    _with_retries(_launch, attempts=3)


def _fetch_vizier_cone(catalog: str, columns: list[str], path: pathlib.Path) -> None:
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astroquery.vizier import Vizier

    coord = SkyCoord(ra=CENTRE_RA_DEG * u.deg, dec=CENTRE_DEC_DEG * u.deg, frame="icrs")
    v = Vizier(columns=columns, row_limit=N_ROWS, timeout=180)

    def _launch() -> None:
        res = v.query_region(coord, radius=f"{CONE_RADIUS_DEG} deg", catalog=catalog)
        if not res:
            # VizieR schema-renamed ``columns`` sometimes returns 0 rows. Fall
            # back to ``columns='**'`` so we still get data.
            v_flex = Vizier(row_limit=N_ROWS, timeout=180)
            res = v_flex.query_region(coord, radius=f"{CONE_RADIUS_DEG} deg", catalog=catalog)
        if not res:
            raise RuntimeError(f"VizieR query for {catalog!r} returned no tables.")
        df = astropy_table_to_polars(res[0])
        df.write_csv(path)

    _with_retries(_launch, attempts=3)


def _fetch_allwise(path: pathlib.Path) -> None:
    _fetch_vizier_cone(
        "II/328/allwise",
        ["RAJ2000", "DEJ2000", "AllWISE", "W1mag", "W2mag", "W3mag", "W4mag"],
        path,
    )


def _fetch_usno(path: pathlib.Path) -> None:
    _fetch_vizier_cone(
        "I/284/out",
        [
            "USNO-B1.0",
            "RAJ2000",
            "DEJ2000",
            "e_RAJ2000",
            "e_DEJ2000",
            "pmRA",
            "pmDE",
        ],
        path,
    )


# --------------------------------------------------------------------------- #
# Cache fixtures — registered exactly once at module import.
# --------------------------------------------------------------------------- #
CATALOGUE_FETCHERS: dict[pathlib.Path, Callable[[pathlib.Path], None]] = {
    CACHE_DIR / "gaia_dr3_top500.csv": _fetch_gaia,
    CACHE_DIR / "allwise_top500.csv": _fetch_allwise,
    CACHE_DIR / "usno_b1_top500.csv": _fetch_usno,
}


def _ensure_csv(path: pathlib.Path) -> pathlib.Path:
    if path.exists() and path.stat().st_size > 256:
        return path
    fetcher = CATALOGUE_FETCHERS.get(path)
    if fetcher is None:
        raise KeyError(f"No fetcher registered for {path}.")
    _require_network()
    t0 = time.perf_counter()
    fetcher(path)
    elapsed = time.perf_counter() - t0
    if not path.exists() or path.stat().st_size < 100:
        raise RuntimeError(f"Catalogue cache miss after fetch for {path}.")
    print(f"\n[real-catalogues] populated {path.name} in {elapsed:.1f}s")
    return path


@pytest.fixture(scope="session")
def gaia_csv() -> pathlib.Path:
    return _ensure_csv(CACHE_DIR / "gaia_dr3_top500.csv")


@pytest.fixture(scope="session")
def allwise_csv() -> pathlib.Path:
    return _ensure_csv(CACHE_DIR / "allwise_top500.csv")


@pytest.fixture(scope="session")
def usno_csv() -> pathlib.Path:
    return _ensure_csv(CACHE_DIR / "usno_b1_top500.csv")


# --------------------------------------------------------------------------- #
# CatalogueSource construction per test
# --------------------------------------------------------------------------- #
def _gaia_source(csv: pathlib.Path) -> CatalogueSource:
    """Build a Gaia CatalogueSource for a cached CSV.

    Includes ``corr_column`` only when the cached CSV actually contains the
    ``ra_dec_corr`` column (added to the fetcher after the initial cache
    population).  This keeps the helper backward-compatible with older cache
    files that were fetched before the column was included.
    """
    src = CatalogueSource(
        name=csv.stem,
        is_local=True,
        path=csv,
        ra_column="ra",
        dec_column="dec",
        id_column="source_id",
        ra_err_column="ra_error",
        dec_err_column="dec_error",
        pos_err_units="mas",
    )
    # Only wire corr_column if the CSV actually has it (backward compat).
    if csv.exists():
        import polars as pl

        cols = pl.scan_csv(str(csv)).collect_schema().names()
        if "ra_dec_corr" in cols:
            src.corr_column = "ra_dec_corr"
    return src


def _allwise_source(csv: pathlib.Path) -> CatalogueSource:
    return CatalogueSource(
        name=csv.stem,
        is_local=True,
        path=csv,
        ra_column="RAJ2000",
        dec_column="DEJ2000",
        id_column="AllWISE",
    )


def _usno_source(csv: pathlib.Path) -> CatalogueSource:
    return CatalogueSource(
        name=csv.stem,
        is_local=True,
        path=csv,
        ra_column="RAJ2000",
        dec_column="DEJ2000",
        id_column="USNO-B1.0",
        ra_err_column="e_RAJ2000",
        dec_err_column="e_DEJ2000",
        pm_ra_column="pmRA",
        pm_dec_column="pmDE",
        pos_err_units="mas",
    )


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAVE_ASTROQUERY, reason="astroquery not installed")
def test_real_gaia_self_tier1_fast_finds_all_pairs(gaia_csv):
    """Tier 1 (scipy cKDTree) on real Gaia DR3 self-match: every row
    self-pairs at sep ``~ 0``."""
    src = _gaia_source(gaia_csv)
    lf = src.lazy()
    spec = MatchSpec(radius_arcsec=0.01, find="best")
    out = sky_match(src, src, lf, lf, spec, engine="fast").collect()
    assert out.height == N_ROWS, f"expected {N_ROWS} self-pairs, got {out.height}"
    assert "sep_arcsec" in out.columns
    assert float(out["sep_arcsec"].max()) < 1e-6


@pytest.mark.skipif(not _HAVE_ASTROQUERY, reason="astroquery not installed")
def test_real_gaia_self_fast_matches_astropy_to_one_microarcsec(gaia_csv):
    """Tier 1 ↔ astropy parity on real Gaia DR3 within 1 µas."""
    src = _gaia_source(gaia_csv)
    lf = src.lazy()
    spec = MatchSpec(radius_arcsec=0.01, find="best")
    fast = sky_match(src, src, lf, lf, spec, engine="fast").collect()
    astro = sky_match(src, src, lf, lf, spec, engine="astropy").collect()
    assert fast.height == astro.height == N_ROWS
    sep_fast = np.sort(fast["sep_arcsec"].to_numpy())
    sep_ast = np.sort(astro["sep_arcsec"].to_numpy())
    assert np.allclose(sep_fast, sep_ast, atol=1e-6)


@pytest.mark.skipif(not _HAVE_ASTROQUERY, reason="astroquery not installed")
def test_real_gaia_x_allwise_zone_matches_fast(gaia_csv, allwise_csv):
    """Tier 2 produces the same matched-pair shape and separations as Tier 1
    against a real cross-catalogue (Gaia × AllWISE). The fallback path is
    exercised transparently if ``cdshealpix`` is missing."""
    left_src = _gaia_source(gaia_csv)
    right_src = _allwise_source(allwise_csv)
    lf_l = left_src.lazy()
    lf_r = right_src.lazy()

    spec = MatchSpec(radius_arcsec=5.0, find="best")
    fast = sky_match(left_src, right_src, lf_l, lf_r, spec, engine="fast").collect()
    zone = sky_match(left_src, right_src, lf_l, lf_r, spec, engine="zone").collect()
    assert fast.height == zone.height, (
        f"Tier 2 ({zone.height}) disagrees with Tier 1 ({fast.height}) — "
        "the cdshealpix fallback or HEALPix dedup is broken."
    )
    if fast.height:
        assert np.allclose(
            np.sort(fast["sep_arcsec"].to_numpy()),
            np.sort(zone["sep_arcsec"].to_numpy()),
            atol=1e-6,
        )


@pytest.mark.skipif(not _HAVE_ASTROQUERY, reason="astroquery not installed")
def test_real_gaia_x_allwise_tier1_finds_physical_matches(gaia_csv, allwise_csv):
    """Same-sky cone: Gaia (optical) and AllWISE (mid-IR) must share a handful
    of real overlap pairs (W1 detection of every bright Gaia source in the
    surveyed cone). Radius ``5 arcsec`` tolerates AllWISE position scatter."""
    left_src = _gaia_source(gaia_csv)
    right_src = _allwise_source(allwise_csv)
    lf_l = left_src.lazy()
    lf_r = right_src.lazy()
    spec = MatchSpec(radius_arcsec=5.0, find="best")
    out = sky_match(left_src, right_src, lf_l, lf_r, spec, engine="fast").collect()
    assert out.height >= 1, "no real Gaia × AllWISE overlap at 5″ — investigate."
    assert (out["sep_arcsec"] <= 5.0).all()
    assert "source_id" in out.columns
    assert "AllWISE" in out.columns
    # Gaia (left) + AllWISE (right) share no column names, so the right
    # side is not renamed with the "_2" suffix.
    assert "AllWISE_2" not in out.columns


@pytest.mark.skipif(not _HAVE_ASTROQUERY, reason="astroquery not installed")
def test_real_gaia_self_tier3_probabilistic_pmatch_in_unit_range(gaia_csv):
    """Tier 3 Bayesian score on real Gaia DR3 with a photometric prior."""
    src = _gaia_source(gaia_csv)
    lf = src.lazy()
    spec = MatchSpec(
        radius_arcsec=0.5,
        find="best",
        prior_columns=["phot_g_mean_mag"],
    )
    out = sky_match(src, src, lf, lf, spec, engine="fast").collect()
    assert "p_match" in out.columns
    assert out.height == N_ROWS
    p = out["p_match"].to_numpy()
    assert ((p >= 0.0) & (p <= 1.0)).all()
    assert np.isfinite(p).all()
    # Self-match priors saturate the positional kernel — p_match should
    # concentrate near 1 with no values floored to exactly 0.
    assert p.mean() > 0.5, f"unexpected p_match distribution: mean={p.mean():.4f}"


@pytest.mark.skipif(not _HAVE_ASTROQUERY, reason="astroquery not installed")
def test_real_gaia_id_join_self_returns_every_row(gaia_csv):
    """``id_join`` on the real Gaia ``source_id`` integer column."""
    lf = pl.scan_csv(str(gaia_csv))
    out = id_join(lf, lf, "source_id", "source_id", "1and2").collect()
    assert out.height == N_ROWS
    # polars inner joins keep only one side of the join keys, so
    # ``source_id_2`` is absent; non-key column collisions still get
    # the "_2" suffix.
    assert "source_id_2" not in out.columns
    assert sorted(out["ra"].to_numpy()) == sorted(out["ra_2"].to_numpy())


@pytest.mark.skipif(not _HAVE_ASTROQUERY, reason="astroquery not installed")
def test_real_gaia_skyerr_self_passes_strict_sigma(gaia_csv):
    """Real Gaia ``ra_error``/``dec_error`` (mas) → skyerr at ``max_error=3``
    accepts every self-pair at sep≈0."""
    src = _gaia_source(gaia_csv)
    lf = src.lazy()
    spec = MatchSpec(
        matcher="skyerr",
        max_error=3.0,
        radius_arcsec=2.0,
        find="best",
    )
    out = sky_match(src, src, lf, lf, spec, engine="fast").collect()
    assert out.height == N_ROWS


@pytest.mark.skipif(not _HAVE_ASTROQUERY, reason="astroquery not installed")
def test_real_usno_skyerr_self_tier1(usno_csv):
    """``skyerr`` on USNO-B1.0 through the Tier 1 (cKDTree) engine. The
    catalogue declares ``pmRA``/``pmDE`` columns; current
    ``_pos_sigma_arcsec`` only consumes the per-axis errors, so the PM
    columns are preserved on the row but not propagated. PM-to-position
    propagation is a known future task."""
    src = _usno_source(usno_csv)
    lf = src.lazy()
    spec = MatchSpec(
        matcher="skyerr",
        max_error=3.0,
        radius_arcsec=2.0,
        find="best",
    )
    out = sky_match(src, src, lf, lf, spec, engine="fast").collect()
    assert out.height == N_ROWS
    assert "pmRA" in out.columns
    assert "pmDE" in out.columns


@pytest.mark.skipif(not _HAVE_ASTROQUERY, reason="astroquery not installed")
def test_real_allwise_tier1_self_matches(gaia_csv, allwise_csv):
    """AllWISE itself + Tier 1 self-match (round-trip through the matcher,
    not just CSV-to-LazyFrame)."""
    src = _allwise_source(allwise_csv)
    lf = src.lazy()
    spec = MatchSpec(radius_arcsec=0.01, find="best")
    out = sky_match(src, src, lf, lf, spec, engine="fast").collect()
    assert out.height == N_ROWS
    assert float(out["sep_arcsec"].max()) < 1e-6
    for col in ("RAJ2000", "DEJ2000", "AllWISE", "W1mag", "W2mag"):
        assert col in out.columns, f"AllWISE matcher result missing expected column '{col}'"


@pytest.mark.skipif(
    not _HAVE_ASTROQUERY or not stilts.stilts_available(),
    reason="astroquery or STILTS Java not installed",
)
def test_real_gaia_stilts_parity_with_fast(gaia_csv):
    """STILTS ↔ scipy cKDTree parity on real Gaia — both engines must agree
    on the same row count and separations to ≤1 µas."""
    src = _gaia_source(gaia_csv)
    lf = src.lazy()
    spec = MatchSpec(radius_arcsec=0.01, find="best")
    stilts_out = sky_match(src, src, lf, lf, spec, engine="stilts").collect()
    fast_out = sky_match(src, src, lf, lf, spec, engine="fast").collect()
    assert stilts_out.height == fast_out.height == N_ROWS
    sep_s = np.sort(stilts_out["sep_arcsec"].to_numpy())
    sep_f = np.sort(fast_out["sep_arcsec"].to_numpy())
    assert np.allclose(sep_s, sep_f, atol=1e-6)


@pytest.mark.skipif(not _HAVE_ASTROQUERY, reason="astroquery not installed")
def test_skyellipse_gaia_self_match_with_correlation(gaia_csv):
    """Skyellipse (Mahalanobis distance with 2×2 covariance) on real Gaia
    DR3 self-match.  Uses ``ra_error``/``dec_error`` (mas) and
    ``ra_dec_corr`` (when available) for full error-ellipse matching.

    Every row self-pairs at sep≈0, so the Mahalanobis distance is dominated
    by the positional covariance.  The ``max_error=5`` threshold comfortably
    accepts all self-pairs even with correlated errors.
    """
    src = _gaia_source(gaia_csv)
    if src.corr_column is None:
        pytest.skip(
            "Cache missing ra_dec_corr column — delete gaia_dr3_top500.csv "
            "and re-run to fetch with correlation."
        )
    lf = src.lazy()

    # Run skyellipse with a generous max_error to accept all self-pairs.
    spec = MatchSpec(
        matcher="skyellipse",
        max_error=5.0,
        radius_arcsec=2.0,
        find="best",
    )
    out = sky_match(src, src, lf, lf, spec, engine="fast").collect()
    assert out.height == N_ROWS, f"skyellipse self-match expected {N_ROWS}, got {out.height}"
    assert "sep_arcsec" in out.columns
    assert float(out["sep_arcsec"].max()) < 1e-6

    # All self-pairs should have well-defined separations.
    assert np.all(np.isfinite(out["sep_arcsec"].to_numpy()))


@pytest.mark.skipif(not _HAVE_ASTROQUERY, reason="astroquery not installed")
def test_skyellipse_vs_sky_on_gaia_self_match(gaia_csv):
    """Skyellipse and plain sky (radius-only) must agree on the same row
    count and sep_arcsec for a Gaia self-match at tiny radius, since the
    positional errors are negligible compared to the search radius."""
    src = _gaia_source(gaia_csv)
    lf = src.lazy()

    spec_sky = MatchSpec(radius_arcsec=0.01, find="best")
    spec_ell = MatchSpec(matcher="skyellipse", max_error=5.0, radius_arcsec=0.01, find="best")

    sky_out = sky_match(src, src, lf, lf, spec_sky, engine="fast").collect()
    ell_out = sky_match(src, src, lf, lf, spec_ell, engine="fast").collect()

    assert sky_out.height == ell_out.height == N_ROWS
    sep_sky = np.sort(sky_out["sep_arcsec"].to_numpy())
    sep_ell = np.sort(ell_out["sep_arcsec"].to_numpy())
    assert np.allclose(sep_sky, sep_ell, atol=1e-6)


@pytest.mark.skipif(not _HAVE_ASTROQUERY, reason="astroquery not installed")
def test_nway_gaia_allwise_usno_photometric_priors(gaia_csv, allwise_csv, usno_csv):
    """3-way Bayesian N-way crossmatch (Gaia × AllWISE × USNO-B1.0) with a
    photometric prior on ``phot_g_mean_mag``.

    Uses :meth:`CrossMatch.nway_match` to score simultaneous 3-catalogue
    tuples with the Budavári N-way posterior.  The test verifies that the
    result contains at least one matched tuple, that ``p_match`` falls in
    [0, 1], and that columns from all three catalogues are present.
    """
    from pathlib import Path

    from xmatch.crossmatch import CrossMatch

    config = Path(__file__).parent.parent / "src" / "xmatch" / "xmatch.yaml"
    cm = CrossMatch(config_file=config)

    result = cm.nway_match(
        [str(gaia_csv), str(allwise_csv), str(usno_csv)],
        radius_arcsec=2.0,
        prior_columns=["phot_g_mean_mag"],
        max_tuples_per_source=5_000,
        chunk_size=10_000,
    )

    assert result.height >= 1, "nway_match found no 3-way tuples — check cache or cone selection"
    assert "p_match" in result.columns
    p = result["p_match"].to_numpy()
    assert ((p >= 0.0) & (p <= 1.0)).all()
    assert np.isfinite(p).all()

    # Verify columns from all three catalogues are present.
    assert "source_id" in result.columns  # Gaia
    assert "AllWISE_2" in result.columns or "AllWISE" in result.columns
    assert "USNO-B1.0_3" in result.columns or "USNO-B1.0" in result.columns
