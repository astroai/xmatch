"""Ray-union driver tests: independent brute-force oracle + resume + rest tiling.

The oracle re-implements the documented union semantics from scratch
(pairwise separations + star-shaped row-set enumeration), so a mismatch
means :mod:`xmatch.ray_union` drifted from its spec rather than from
itself.  All inputs are tiny deterministic frames; ``ray.init`` runs
once per test through :func:`ray_union_match`.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from xmatch import hats_native, matchers, ray_union
from xmatch.sources import CatalogueSource

HAVE_RAY = importlib.util.find_spec("ray") is not None

pytestmark = pytest.mark.skipif(not HAVE_RAY, reason="ray is not installed")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _catalogue(path: Path, name: str, df: pl.DataFrame) -> CatalogueSource:
    path.mkdir(parents=True, exist_ok=True)
    (path / "dataset" / "Norder=0" / "Dir=0").mkdir(parents=True, exist_ok=True)
    df.write_parquet(path / "dataset" / "Norder=0" / "Dir=0" / "Npix=0.parquet")
    (path / "properties").write_text(
        "dataproduct_type=object\nobs_collection=xmatch-test\n"
        "hats_col_ra=ra\nhats_col_dec=dec\nhats_ordering=NESTED\nhats_nrows=%d\n" % df.height
    )
    return CatalogueSource(name=name, is_local=True, path=path, ra_column="ra", dec_column="dec")


def _read_output(out: Path) -> pl.DataFrame:
    pixels = hats_native.list_hats_pixels(out)
    frames = [pl.read_parquet(p) for _, _, p in pixels]
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def _sep_arcsec(r1, r2) -> float:
    d1, d2 = math.radians(r1["dec"]), math.radians(r2["dec"])
    dra = math.radians(r1["ra"] - r2["ra"])
    chord = math.sin(d1) * math.sin(d2) + math.cos(d1) * math.cos(d2) * math.cos(dra)
    return math.degrees(math.acos(max(-1.0, min(1.0, chord)))) * 3600.0


def _oracle_union3(
    a: pl.DataFrame, b: pl.DataFrame, c: pl.DataFrame, sep_arcsec: float
) -> list[tuple[str, tuple[str, ...], float]]:
    """Sequential 3-way full-outer union: row sets radiate from the
    lowest-indexed catalogue present.

    * A rows star outward: every combination of their B/C matches within
      ``sep_arcsec`` (B-C pairs need not match each other: the set's edges
      are the hub edges, ``sep`` = min over *used* edges).
    * B rows that have NO A match star their own sets (``2`` / ``2+3``) —
      a B row with an A match is folded into A's sets and never appears
      again, exactly the engine's chunk-ownership rule.
    * C rows with no A or B match are ``3`` singles.
    """
    a_rows = a.to_dicts()
    b_rows = b.to_dicts()
    c_rows = c.to_dicts()

    def mates(row, others: list[dict], sep: float) -> list[tuple[dict, float]]:
        out = []
        for o in others:
            s = _sep_arcsec(row, o)
            if s <= sep:
                out.append((o, s))
        return out

    nan = float("nan")
    results: list[tuple[str, tuple[str, ...], float]] = []
    for ar in a_rows:
        bm = mates(ar, b_rows, sep_arcsec)
        cm = mates(ar, c_rows, sep_arcsec)
        if not bm and not cm:
            results.append(("1", (ar["id"],), nan))
        for bo, bs in bm:
            results.append(("1+2", (ar["id"], bo["id"]), bs))
        for co, cs in cm:
            results.append(("1+3", (ar["id"], co["id"]), cs))
        for bo, bs in bm:
            for co, cs in cm:
                results.append(("1+2+3", (ar["id"], bo["id"], co["id"]), min(bs, cs)))
    for br in b_rows:
        if any(_sep_arcsec(ar, br) <= sep_arcsec for ar in a_rows):
            continue  # folded into A's sets
        cm = mates(br, c_rows, sep_arcsec)
        if not cm:
            results.append(("2", (br["id"],), nan))
        for co, cs in cm:
            results.append(("2+3", (br["id"], co["id"]), cs))
    for cr in c_rows:
        if any(_sep_arcsec(ar, cr) <= sep_arcsec for ar in a_rows) or any(
            _sep_arcsec(br, cr) <= sep_arcsec for br in b_rows
        ):
            continue
        results.append(("3", (cr["id"],), nan))
    return sorted(results)


def _ray_rows3(df: pl.DataFrame) -> list[tuple[str, tuple[str, ...], float]]:
    def canonical(row) -> tuple[str, tuple[str, ...], float]:
        ids = tuple(v for v in (row.get("id"), row.get("id_2"), row.get("id_3")) if v is not None)
        return (row["_src_cats"], ids, float(row["sep_arcsec"]))

    return sorted(canonical(r) for r in df.to_dicts())


def _assert_matches_oracle(got: list[tuple[str, tuple[str, ...], float]], want: list) -> None:
    assert len(got) == len(want), (len(got), len(want))
    for g, w in zip(got, want, strict=True):
        assert g[0] == w[0], (g, w)
        assert g[1] == w[1], (g, w)
        assert (
            (g[2] == w[2]) or (math.isnan(g[2]) and math.isnan(w[2])) or (abs(g[2] - w[2]) < 1e-2)
        ), (g, w)


def test_ray_union_3catalogue_vs_oracle(tmp_path: Path) -> None:
    """3-catalogue union: multi-edge seps, A-star sets, B-rest {2,3} combos
    vs a sequential 3-way oracle.  Regression: non-hub centre chunks with a
    lower-indexed mate emitted duplicate {j, k} row sets (a single oracle row
    '1+2+3' instead of four duplicates).

    Every fixture group holds >= 16 rows so the tiler (threshold 15) splits
    it all the way to fine pixels — a sparse pixel would balloon the cone
    radius and kill the rest geometry.
    """
    a = pl.DataFrame(
        {
            # cluster α: 40 rows (20 positions x twins) around (0,0), all mated
            # to B-near and c0; cluster α1: 32 rows at (5,5) — '1' singles.
            # Every cluster keeps >= 32 rows so the tiler never stops coarse on
            # a boundary-split child (<= threshold rows).
            "id": [f"a{i}" for i in range(40)] + [f"a1_{i}" for i in range(32)],
            "ra": [0.001 * (i % 20) for i in range(40)] + [5.0 + 0.001 * i for i in range(32)],
            "dec": [0.001 * (i % 20) for i in range(40)] + [5.0 + 0.001 * i for i in range(32)],
        }
    )
    b = pl.DataFrame(
        {
            # near: 32 rows (16 positions x twins) at (0,0)+0.001i — mates of
            # cluster α; far: 32 rows at (15,5)+0.001i — rest of B, mated to
            # C-far; b2: 32 rows at (30,30) — '2' singles
            "id": [f"b{i}" for i in range(32)]
            + [f"bf{i}" for i in range(32)]
            + [f"b2_{i}" for i in range(32)],
            "ra": [0.001 * (i % 16) for i in range(32)]
            + [15.0 + 0.001 * i for i in range(32)]
            + [30.0 + 0.001 * i for i in range(32)],
            "dec": [0.001 * (i % 16) for i in range(32)]
            + [5.0 + 0.001 * i for i in range(32)]
            + [30.0 + 0.001 * i for i in range(32)],
        }
    )
    c = pl.DataFrame(
        {
            # c0: 32 rows near (0,0) — mates of cluster α; cf: 32 rows near
            # B-far (covered by B's cone, NOT rest); c1: 32 rows at (45,45)
            "id": [f"c0_{i}" for i in range(32)]
            + [f"cf{i}" for i in range(32)]
            + [f"c1_{i}" for i in range(32)],
            "ra": [0.0007 + 0.001 * i for i in range(32)]
            + [15.0002 + 0.001 * i for i in range(32)]
            + [45.0 + 0.001 * i for i in range(32)],
            "dec": [0.0007 + 0.001 * i for i in range(32)]
            + [5.0002 + 0.001 * i for i in range(32)]
            + [45.0 + 0.001 * i for i in range(32)],
        }
    )
    src_a = _tiled_catalogue(tmp_path / "a", "a", a, threshold=15)
    src_b = _tiled_catalogue(tmp_path / "b", "b", b, threshold=15)
    src_c = _tiled_catalogue(tmp_path / "c", "c", c, threshold=15)
    out = tmp_path / "out"
    ray_union.ray_union_match(
        [src_a, src_b, src_c], sep_arcsec=5.0, output_file=str(out), hats_threshold=15
    )

    plan = ray_union.last_plan()
    assert plan is not None
    assert plan.rest, "expected uncovered rest partitions"
    # B's far cluster and (30,30) singles, plus C's (45,45) singles, must be
    # rest runs; the C-far cluster sits inside B's far cone, so no rest
    # partition may be centered near it (its {2,3} sets star from B's chunk)
    b_rest = [r for r in plan.rest if r.cat == 1]
    c_rest = [r for r in plan.rest if r.cat == 2]
    assert b_rest and c_rest, plan.rest
    near_bfar = False
    for r in b_rest:
        ra_c, dec_c = ray_union._pixel_center_deg(
            plan.catalogues[r.cat].partitions[r.part_idx].order,
            plan.catalogues[r.cat].partitions[r.part_idx].pix,
        )
        if abs(ra_c - 15.0) < 1.0 and abs(dec_c - 5.0) < 1.0:
            near_bfar = True
    assert near_bfar, "B's far cluster must be rest"
    # C's only uncovered pixels are the (45,45) singles — the far cluster is
    # covered by B's cone (its {2,3} sets star from B's chunk)
    for r in c_rest:
        ra_c, dec_c = ray_union._pixel_center_deg(
            plan.catalogues[r.cat].partitions[r.part_idx].order,
            plan.catalogues[r.cat].partitions[r.part_idx].pix,
        )
        assert abs(ra_c - 45.0) < 1.5 and abs(dec_c - 45.0) < 1.5, (r, ra_c, dec_c)

    df = _read_output(out)
    _assert_matches_oracle(_ray_rows3(df), _oracle_union3(a, b, c, 5.0))

    import hats  # noqa: PLC0415

    hats.read_hats(out)  # raises on overlapping trees
    props = (out / "properties").read_text()
    assert "obs_collection=" in props
    got_nrows = next(
        int(line.split("=", 1)[1]) for line in props.splitlines() if line.startswith("hats_nrows=")
    )
    assert got_nrows == len(_ray_rows3(df)), (got_nrows, len(_ray_rows3(df)))


def _oracle_union(
    a: pl.DataFrame, b: pl.DataFrame, sep_arcsec: float
) -> list[tuple[str, tuple[str, ...], float]]:
    """Every output row as ``(_src_cats, (ids,), sep_arcsec)``.

    Semantics mirrored from the sequential-union oracle (and the module
    docstring): rows radiate from the lowest-indexed catalogue present; each
    output row chooses AT MOST ONE partner per catalogue (null or a mate
    within radius — the "captain" of that catalogue); a singleton is kept
    ONLY for rows without any mate (the sequential full-outer join never
    emits a lone row for a matched one).  ``sep`` = minimum separation over
    the chosen edges.
    """
    a_rows = a.to_dicts()
    b_rows = b.to_dicts()
    results: list[tuple[str, tuple[str, ...], float | None]] = []
    covered_b: set[int] = set()

    def emit(ids: tuple[str | None, str | None], sep: float | None) -> None:
        src = "+".join(str(k + 1) for k, v in enumerate(ids) if v is not None)
        results.append(
            (src, tuple(v for v in ids if v is not None), sep if sep is not None else float("nan"))
        )

    for ar in a_rows:
        mates = [
            (j, _sep_arcsec(ar, br))
            for j, br in enumerate(b_rows)
            if _sep_arcsec(ar, br) <= sep_arcsec
        ]
        for j, _ in mates:
            covered_b.add(j)
        if not mates:
            emit((ar["id"], None), None)  # singleton only when unmated
        for j, d in mates:  # at most one mate per partner catalogue
            emit((ar["id"], b_rows[j]["id"]), d)

    for j, br in enumerate(b_rows):
        if j not in covered_b:
            emit((None, br["id"]), None)

    return sorted((src, ids, sep if sep is not None else float("nan")) for src, ids, sep in results)


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def test_ray_union_matches_brute_force_oracle(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    a = pl.DataFrame(
        {
            "ra": (rng.normal(0.0, 0.02, 40) % 360.0).round(6),
            "dec": (rng.normal(0.0, 0.02, 40) % 90.0).round(6),
            "id": [f"A{i}" for i in range(40)],
        }
    )
    b = pl.DataFrame(
        {
            "ra": (rng.normal(0.0, 0.02, 25) % 360.0).round(6),
            "dec": (rng.normal(0.0, 0.02, 25) % 90.0).round(6),
            "id": [f"B{i}" for i in range(25)],
        }
    )
    base = tmp_path / "cats"
    src_a = _catalogue(base / "cata", "cata", a)
    src_b = _catalogue(base / "catb", "catb", b)

    out = tmp_path / "out"
    ray_union.ray_union_match(
        [src_a, src_b],
        sep_arcsec=30.0,
        output_file=str(out),
        hats_threshold=1_000_000,
        max_tuples=100_000,
    )

    df = _read_output(out)
    assert sorted(df["_src_cats"].unique().to_list()) == ["1", "1+2", "2"]
    oracle = _oracle_union(a, b, 30.0)

    def canonical(row) -> tuple[str, tuple[str, ...], float]:
        ids = tuple(v for v in (row.get("id"), row.get("id_2")) if v is not None)
        return (row["_src_cats"], ids, float(row["sep_arcsec"]))

    ray_rows = sorted(canonical(r) for r in df.to_dicts())
    assert len(ray_rows) == len(oracle)
    for got, want in zip(ray_rows, oracle, strict=True):
        assert got[0] == want[0], (got, want)
        assert got[1] == want[1], (got, want)
        assert (
            (got[2] == want[2])
            or (math.isnan(got[2]) and math.isnan(want[2]))
            or (abs(got[2] - want[2]) < 1e-4)
        ), (got, want)


def test_resume_skips_done_chunks(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    a = pl.DataFrame(
        {
            "ra": (rng.normal(0.0, 0.02, 30) % 360.0).round(6),
            "dec": (rng.normal(0.0, 0.02, 30) % 90.0).round(6),
            "id": [f"A{i}" for i in range(30)],
        }
    )
    b = pl.DataFrame(
        {
            "ra": (rng.normal(0.0, 0.02, 20) % 360.0).round(6),
            "dec": (rng.normal(0.0, 0.02, 20) % 90.0).round(6),
            "id": [f"B{i}" for i in range(20)],
        }
    )
    base = tmp_path / "cats"
    src_a = _catalogue(base / "a", "a", a)
    src_b = _catalogue(base / "b", "b", b)
    out = tmp_path / "out"
    ray_union.ray_union_match([src_a, src_b], sep_arcsec=20.0, output_file=str(out))

    chunks = list((out / "chunks").glob("*.parquet"))
    assert chunks
    mtimes = {p.name: p.stat().st_mtime_ns for p in chunks}
    state = (out / "resume.state").read_text()
    assert '"status": "done"' in state
    first_rows = sum(pl.read_parquet(p).height for _, _, p in hats_native.list_hats_pixels(out))

    # second run: chunks untouched (skipped), final dataset identical
    ray_union.ray_union_match([src_a, src_b], sep_arcsec=20.0, output_file=str(out))
    for p in (out / "chunks").glob("*.parquet"):
        assert p.stat().st_mtime_ns == mtimes[p.name]
    assert (
        sum(pl.read_parquet(p).height for _, _, p in hats_native.list_hats_pixels(out))
        == first_rows
    )
    assert '"status": "done"' in (out / "resume.state").read_text()


def test_far_partner_rows_survive_as_singles(tmp_path: Path) -> None:
    """Partner rows far outside the hub footprint (no mate within the radius)
    must not be lost: they come out as ``2`` singles inside the hub tile, in a
    valid (hats-readable) output tree."""
    a = pl.DataFrame(
        {
            "id": [f"A{i}" for i in range(50)],
            "ra": [10.0 + i * 0.001 for i in range(50)],
            "dec": [-5.0 + i * 0.001 for i in range(50)],
        }
    )
    b = pl.DataFrame(
        {
            "id": [f"B{i}" for i in range(12)],
            "ra": [80.0 + i * 0.001 for i in range(12)],
            "dec": [80.0 + i * 0.001 for i in range(12)],
        }
    )
    base = tmp_path / "src"
    src_a = _catalogue(base / "a", "a", a)
    src_b = _catalogue(base / "b", "b", b)
    out = tmp_path / "out"
    ray_union.ray_union_match(
        [src_a, src_b], sep_arcsec=5.0, output_file=str(out), hats_threshold=1_000_000
    )

    pixels = hats_native.list_hats_pixels(out)
    total = sum(pl.read_parquet(p).height for _, _, p in pixels)
    assert total == 62

    import hats  # noqa: PLC0415

    cat = hats.read_hats(out)  # raises on overlapping trees
    assert len(cat.get_healpix_pixels()) == len(pixels)
    df = _read_output(out)
    assert sorted(df["_src_cats"].unique().to_list()) == ["1", "2"]
    assert (df["_src_cats"] == "2").sum() == 12


def test_cone_pixels_expands_parent_pixels_to_all_children() -> None:
    """A cone_search parent (fully-inside subtree) must yield EVERY child at
    the target depth, not just the first one (regression: sibling cells were
    dropped, hiding partner partitions inside the cone)."""
    order, pix, radius, depth = 11, 7, 3.0, 6
    got = sorted(ray_union._cone_pixels(order, pix, radius, depth))

    import cdshealpix  # noqa: PLC0415
    from astropy import units as u  # noqa: PLC0415
    from astropy.coordinates import Latitude, Longitude  # noqa: PLC0415

    lon, lat = ray_union._pixel_center_deg(order, pix)
    ipix, depths, _ = cdshealpix.cone_search(
        Longitude(lon, unit="deg"), Latitude(lat, unit="deg"), radius * u.deg, depth
    )
    want = []
    for ipx, d in zip(np.asarray(ipix).tolist(), np.asarray(depths).tolist(), strict=True):
        dif = depth - int(d)
        if dif <= 0:
            want.append((depth, int(ipx)))
        else:
            base = int(ipx) << (2 * dif)
            want.extend((depth, i) for i in range(base, base + (1 << (2 * dif))))
    assert got == sorted(want)

    # any sibling child of every parent must be present — the old bug kept
    # only the first child of each fully-inside subtree
    parents = {
        int(ipx)
        for ipx, d in zip(np.asarray(ipix).tolist(), np.asarray(depths).tolist(), strict=True)
        if int(d) < depth
    }
    assert parents, "cone_search must return a fully-inside parent at this geometry"
    for p in parents:
        assert (depth, (p << 2) + 1) in got, (p, got)


def _tiled_catalogue(root: Path, name: str, df: pl.DataFrame, threshold: int) -> CatalogueSource:
    """Local HATS catalogue with real (threshold-tiled) partitions."""
    from xmatch.mirror import _write_hats_native

    _write_hats_native(df, root, ra_column="ra", dec_column="dec", threshold=threshold)
    return CatalogueSource(name=name, is_local=True, path=root, ra_column="ra", dec_column="dec")


def test_rest_partition_rows_emitted_once(tmp_path: Path) -> None:
    """Regression: rows of a partition no earlier cone covers were emitted
    TWICE (own centre chunk + rest task).  Fine tiling (hats_threshold=15)
    with a far-flung B cluster: the far partition is genuinely uncovered, and
    the covered near rows are identical twins of A rows (matched at 0")."""
    a = pl.DataFrame(
        {
            "id": [f"A{i}" for i in range(40)],
            "ra": [10.0 + 0.001 * (i % 20) for i in range(40)],
            "dec": [-5.0 + 0.001 * (i % 20) for i in range(40)],
        }
    )
    b = pl.DataFrame(
        {
            "id": [f"Bnear{i}" for i in range(16)] + [f"Bfar{i}" for i in range(20)],
            "ra": [10.0 + 0.001 * i for i in range(16)] + [80.0 + 0.001 * i for i in range(20)],
            "dec": [-5.0 + 0.001 * i for i in range(16)] + [80.0 + 0.001 * i for i in range(20)],
        }
    )
    src_a = _tiled_catalogue(tmp_path / "a", "a", a, threshold=15)
    src_b = _tiled_catalogue(tmp_path / "b", "b", b, threshold=15)
    out = tmp_path / "out"
    ray_union.ray_union_match(
        [src_a, src_b], sep_arcsec=5.0, output_file=str(out), hats_threshold=15
    )

    plan = ray_union.last_plan()
    assert plan is not None
    assert plan.rest, "expected at least one uncovered (rest) partition"
    for r in plan.rest:  # every rest partition must be the far cluster
        rest_part = plan.catalogues[r.cat].partitions[r.part_idx]
        ra_c, dec_c = ray_union._pixel_center_deg(rest_part.order, rest_part.pix)
        assert abs(ra_c - 80.0) < 1.0 and abs(dec_c - 80.0) < 1.0, rest_part

    df = _read_output(out)
    oracle = _oracle_union(a, b, 5.0)

    def canonical(row) -> tuple[str, tuple[str, ...], float]:
        ids = tuple(v for v in (row.get("id"), row.get("id_2")) if v is not None)
        return (row["_src_cats"], ids, float(row["sep_arcsec"]))

    ray_rows = sorted(canonical(r) for r in df.to_dicts())
    assert len(ray_rows) == len(oracle), (len(ray_rows), len(oracle))
    for got, want in zip(ray_rows, oracle, strict=True):
        assert got[0] == want[0], (got, want)
        assert got[1] == want[1], (got, want)
        # engine sep has a ~0.003 arcsec float-noise floor for near-coincident
        # rows (chord from cos/sin roundtrip); ids above are the real contract
        assert (
            (got[2] == want[2])
            or (math.isnan(got[2]) and math.isnan(want[2]))
            or (abs(got[2] - want[2]) < 1e-2)
        ), (got, want)

    import hats  # noqa: PLC0415

    hats.read_hats(out)  # raises on overlapping trees
    total = sum(pl.read_parquet(p).height for _, _, p in hats_native.list_hats_pixels(out))
    assert total == 60  # 8 '1' (A twins at positions 16..19) + 32 '1+2' (A
    # twins at positions 0..15, each mated to one of the 16 Bnear rows)
    # + 20 '2' (Bfar)


def test_block_combos_gates_rows_with_lower_indexed_mate() -> None:
    """A centre row of catalogue j that ALSO has a lower-indexed mate is
    owned by that lower centre: its {j, k} sets must not star from HERE too
    (regression: the canonical filter only dropped combos *selecting* a
    lower partner, so a mated B row still emitted its {B, C} sets and
    participated twice; the sequential oracle emits one '1+2+3' row)."""
    chord = matchers._arcsec_to_chord(5.0)
    centre_ra = np.array([0.0])
    centre_dec = np.array([0.0005])  # 1.8" from the lower-indexed mate
    cat_sel, seps, srcs, _ = ray_union._block_combos(
        centre_ra,
        centre_dec,
        np.array([7], dtype=np.int64),
        [
            (np.array([0.0]), np.array([0.0])),  # catalogue 1 (LOWER)
            (np.array([0.0010]), np.array([0.0])),  # catalogue 3 (higher)
        ],
        chord,
        max_tuples=10_000,
        centre_label=2,
        cat_labels=[1, 3],
        centre_cat=1,
    )
    assert len(seps) == 0, f"mated row must be skipped entirely, got {srcs}"
    # sanity: same row with NO lower mate keeps its {2,3} set + single handling
    cat_sel2, seps2, srcs2, _ = ray_union._block_combos(
        centre_ra,
        centre_dec,
        np.array([7], dtype=np.int64),
        [
            (np.array([], dtype=float), np.array([], dtype=float)),  # no A row
            (np.array([0.0010]), np.array([0.0])),
        ],
        chord,
        max_tuples=10_000,
        centre_label=2,
        cat_labels=[1, 3],
        centre_cat=1,
    )
    assert len(seps2) == 1 and srcs2 == ["2+3"], (seps2, srcs2)


def test_max_tuples_cap_full_product_under_cap_and_sep_cut() -> None:
    """max_tuples truncation: (1) products below the cap must emit EVERY
    combo (regression: a per-partner pow-root cap dropped 28 of 51 valid
    combos at max_tuples=10_000); (2) above the cap, the cut is by
    separation, not pool index."""
    n = 40
    ra = (0.00002 * np.arange(n)).round(9)  # 0.072" spacing, 2.9" span
    dec = np.zeros(n)
    chord = matchers._arcsec_to_chord(5.0)
    centre = (np.zeros(1), np.zeros(1))

    # product = 41 <= cap: every combo, in pool-index order, no singles
    cat_sel, seps, srcs, _ = ray_union._block_combos(
        centre[0],
        centre[1],
        np.array([0], dtype=np.int64),
        [(ra, dec), (np.array([], dtype=float), np.array([], dtype=float))],
        chord,
        max_tuples=10_000,
        centre_label=1,
        cat_labels=[2, 3],
    )
    assert len(seps) == n, (len(seps), n)
    assert all(s == "1+2" for s in srcs)
    assert np.allclose(seps, np.sort(seps))  # index order == sep order here

    # product = 41 > cap 10: exactly the ten CLOSEST combos survive
    cat_sel, seps, srcs, _ = ray_union._block_combos(
        centre[0],
        centre[1],
        np.array([0], dtype=np.int64),
        [(ra, dec), (np.array([], dtype=float), np.array([], dtype=float))],
        chord,
        max_tuples=10,
        centre_label=1,
        cat_labels=[2, 3],
    )
    assert len(seps) == 10, seps
    expected = 0.00002 * np.arange(1, 11) * 3600.0  # 0.072" .. 0.72"
    assert np.allclose(np.sort(seps), expected, atol=0.01), (np.sort(seps), expected)


# --------------------------------------------------------------------------- #
# driver-level resilience: progress, resume.state, run.jsonl on a real run
# --------------------------------------------------------------------------- #
def test_driver_progress_state_and_run_log(tmp_path: Path) -> None:
    """A real (tiny) ray-union leaves a progress trail + audit log."""
    import json as _json

    from xmatch import CrossMatch

    a = _catalogue(
        tmp_path / "a",
        "cat_a",
        pl.DataFrame(
            {
                "ra": [0.0, 0.5, 10.0],
                "dec": [0.0, 0.5, 10.0],
                "m": [1.0, 2.0, 3.0],
            }
        ),
    )
    b = _catalogue(
        tmp_path / "b",
        "cat_b",
        pl.DataFrame(
            {
                "ra": [0.00001, 0.50001, 80.0],
                "dec": [0.00001, 0.50001, -80.0],
                "n": [4.0, 5.0, 6.0],
            }
        ),
    )
    out = tmp_path / "driver.hats"
    msgs: list[str] = []
    cm = CrossMatch()
    cm.union_match(
        [a.path, b.path],
        output_file=out,
        engine="ray-union",
        radius_arcsec=1.5,
        progress_cb=msgs.append,
    )
    assert (out / "run.jsonl").exists()
    assert (out / "resume.state").exists()

    state = _json.loads((out / "resume.state").read_text())
    assert state["status"] == "done"
    assert state["rows"] >= 3  # 2 matched pairs + 2 singles (1+2 == 3 rows)

    events = [_json.loads(line) for line in (out / "run.jsonl").read_text().splitlines()]
    kinds = [e["event"] for e in events]
    assert kinds[0] == "attempt_start"  # crossmatch driver wrapper
    assert "start" in kinds  # ray-union pipeline start
    assert kinds[-1] == "done"
    assert "chunk_done" in kinds
    assert any("ray-union:" in m and "%" in m for m in msgs), msgs

    joined = _read_output(out)
    assert {c for c in ("ra", "dec", "m", "n", "_src_cats", "sep_arcsec")} <= set(joined.columns)


# --------------------------------------------------------------------------- #
# mixed HATS depths + RING ordering + RAY_ADDRESS-less init fallback
# --------------------------------------------------------------------------- #
def _pixeled_catalogue(path: Path, name: str, df: pl.DataFrame, order: int) -> CatalogueSource:
    """Local HATS catalogue tiled at ``order``: one parquet per occupied
    NESTED pixel (the mirror-layout shape the union reads)."""
    import cdshealpix
    from astropy.coordinates import Latitude, Longitude

    path.mkdir(parents=True, exist_ok=True)
    pix = np.asarray(
        cdshealpix.lonlat_to_healpix(
            Longitude(df["ra"].to_numpy(), unit="deg"),
            Latitude(df["dec"].to_numpy(), unit="deg"),
            np.full(len(df), order, dtype=np.uint64),
        )
    )
    for p in np.unique(pix):
        sub = df.filter(pl.Series("_p", pix) == p)
        dirtree = (int(p) // 10000) * 10000
        rel = path / "dataset" / f"Norder={order}" / f"Dir={dirtree}" / f"Npix={int(p)}.parquet"
        rel.parent.mkdir(parents=True, exist_ok=True)
        sub.write_parquet(rel)
    (path / "properties").write_text(
        "dataproduct_type=object\nobs_collection=xmatch-test\n"
        "hats_col_ra=ra\nhats_col_dec=dec\nhats_ordering=NESTED\n"
        "hats_nrows=%d\nhats_max_depth=%d\n" % (df.height, order)
    )
    return CatalogueSource(name=name, is_local=True, path=path, ra_column="ra", dec_column="dec")


def _ring_catalogue(path: Path, name: str, df: pl.DataFrame, order: int) -> CatalogueSource:
    """Local HATS catalogue with RING pixel ids on disk + ``hats_ordering=RING``
    (a mirror that preserved the source's own ordering)."""
    import cdshealpix
    from astropy.coordinates import Latitude, Longitude

    path.mkdir(parents=True, exist_ok=True)
    nested = np.asarray(
        cdshealpix.lonlat_to_healpix(
            Longitude(df["ra"].to_numpy(), unit="deg"),
            Latitude(df["dec"].to_numpy(), unit="deg"),
            np.full(len(df), order, dtype=np.uint64),
        )
    )
    ring = np.asarray(cdshealpix.to_ring(nested, order))
    for p in np.unique(ring):
        sub = df.filter(pl.Series("_p", ring) == p)
        dirtree = (int(p) // 10000) * 10000
        rel = path / "dataset" / f"Norder={order}" / f"Dir={dirtree}" / f"Npix={int(p)}.parquet"
        rel.parent.mkdir(parents=True, exist_ok=True)
        sub.write_parquet(rel)
    (path / "properties").write_text(
        "dataproduct_type=object\nobs_collection=xmatch-test\n"
        "hats_col_ra=ra\nhats_col_dec=dec\nhats_ordering=RING\n"
        "hats_nrows=%d\nhats_max_depth=%d\n" % (df.height, order)
    )
    return CatalogueSource(name=name, is_local=True, path=path, ra_column="ra", dec_column="dec")


def test_union_mixed_orders_oracle(tmp_path: Path) -> None:
    """Inputs at Norder 0 / 1 / 2: matches found across depth boundaries.

    The cone radius is sized by the coarsest partition diagonal, so the
    deeper catalogues' pixels are searched with the full radius and every
    in-radius pair is found whatever the input tilings.
    """
    from xmatch.storage import open_storage

    a = pl.DataFrame(
        {
            "ra": [5.0, -5.0, 5.0, -5.0],
            "dec": [5.0, 5.0, -5.0, -5.0],
            "id": ["a0", "a1", "a2", "a3"],
        }
    )
    b = pl.DataFrame({"ra": [5.0005, 120.0], "dec": [5.0005, 30.0], "id": ["b0", "b1"]})
    c = pl.DataFrame(
        {
            "ra": [4.9995, 120.0005, 240.0, -120.0],
            "dec": [4.9995, 30.0005, -20.0, 60.0],
            "id": ["c0", "c1", "c2", "c3"],
        }
    )
    base = tmp_path / "cats"
    src_a = _pixeled_catalogue(base / "cata", "cata", a, 0)
    src_b = _pixeled_catalogue(base / "catb", "catb", b, 1)
    src_c = _pixeled_catalogue(base / "catc", "catc", c, 2)

    # the depth exercise is real: 1 / >=2 / >=4 partitions at orders 0/1/2
    def n_parts(src: CatalogueSource) -> int:
        return len(ray_union._list_partitions("", open_storage(str(src.path))))

    assert n_parts(src_a) == 1
    assert n_parts(src_b) >= 2
    assert n_parts(src_c) >= 4

    out = tmp_path / "out"
    ray_union.ray_union_match(
        [src_a, src_b, src_c],
        sep_arcsec=60.0,
        output_file=str(out),
        hats_threshold=1_000_000,
        max_tuples=100_000,
    )
    df = _read_output(out)
    _assert_matches_oracle(_ray_rows3(df), _oracle_union3(a, b, c, 60.0))

    # the specific cross-depth sets are present
    srcs = sorted(df["_src_cats"].unique().to_list())
    assert "1+2+3" in srcs  # a0-b0-c0 cluster across all three depths
    assert "2+3" in srcs  # b1-c1 at Norder 1 x Norder 2
    assert "3" in srcs  # c2, c3 singles
    # plan geometry keeps each catalogue's own depth
    plan = ray_union.last_plan()
    assert sorted({p.order for p in plan.catalogues[0].partitions}) == [0]
    assert sorted({p.order for p in plan.catalogues[1].partitions}) == [1]
    assert sorted({p.order for p in plan.catalogues[2].partitions}) == [2]


def test_union_ring_input_converted(tmp_path: Path) -> None:
    """A RING-ordered input is converted to NESTED in the plan: the union
    matches the oracle and the plan's pixels are the NESTED equivalents of
    the on-disk RING ids."""
    import cdshealpix
    from astropy.coordinates import Latitude, Longitude

    a = pl.DataFrame({"ra": [0.0, 20.0], "dec": [0.0, 30.0], "id": ["a0", "a1"]})
    b = pl.DataFrame({"ra": [0.001, 20.0], "dec": [0.001, 30.0], "id": ["b0", "b1"]})
    base = tmp_path / "cats"
    src_a = _pixeled_catalogue(base / "cata", "cata", a, 1)
    src_b = _ring_catalogue(base / "catb", "catb", b, 1)

    def _pix(ra: float, dec: float, order: int) -> int:
        return int(
            np.asarray(
                cdshealpix.lonlat_to_healpix(
                    Longitude(np.array([ra]), unit="deg"),
                    Latitude(np.array([dec]), unit="deg"),
                    np.array([order], dtype=np.uint64),
                )
            )[0]
        )

    on_disk_ring = [
        int(np.asarray(cdshealpix.to_ring(np.asarray([_pix(ra, dec, 1)], dtype=np.uint64), 1))[0])
        for ra, dec in zip(b["ra"], b["dec"], strict=True)
    ]
    expected_nested = sorted(
        _pix(float(ra), float(dec), 1) for ra, dec in zip(b["ra"], b["dec"], strict=True)
    )
    # sanity: the fixture really is RING-tiled (ring ids != nested ids)
    assert on_disk_ring != expected_nested

    out = tmp_path / "out"
    ray_union.ray_union_match(
        [src_a, src_b],
        sep_arcsec=60.0,
        output_file=str(out),
        hats_threshold=1_000_000,
        max_tuples=100_000,
    )

    df = _read_output(out)
    oracle = _oracle_union(a, b, 60.0)

    def canonical(row) -> tuple[str, tuple[str, ...], float]:
        ids = tuple(v for v in (row.get("id"), row.get("id_2")) if v is not None)
        return (row["_src_cats"], ids, float(row["sep_arcsec"]))

    ray_rows = sorted(canonical(r) for r in df.to_dicts())
    assert len(ray_rows) == len(oracle)
    for got, want in zip(ray_rows, oracle, strict=True):
        assert got[0] == want[0], (got, want)
        assert got[1] == want[1], (got, want)

    plan = ray_union.last_plan()
    b_pix = sorted(p.pix for p in plan.catalogues[1].partitions)
    assert b_pix == expected_nested, (b_pix, expected_nested)


def test_ray_init_falls_back_local_on_connection_error(tmp_path: Path, monkeypatch) -> None:
    """RAY_ADDRESS unset + 'auto' unreachable -> a fresh local cluster."""
    import ray

    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    calls: list[dict] = []
    real_init = ray.init

    def fake_init(**kw) -> None:
        calls.append(kw)
        if kw.get("address"):
            raise ConnectionError("no running cluster")
        return real_init(**kw)

    monkeypatch.setattr(ray, "init", fake_init)
    try:
        a = _catalogue(
            tmp_path / "cata", "cata", pl.DataFrame({"ra": [0.0], "dec": [0.0], "id": ["a0"]})
        )
        b = _catalogue(
            tmp_path / "catb", "catb", pl.DataFrame({"ra": [0.001], "dec": [0.0], "id": ["b0"]})
        )
        out = tmp_path / "out"
        ray_union.ray_union_match(
            [a, b],
            sep_arcsec=60.0,
            output_file=str(out),
            hats_threshold=1_000_000,
            max_tuples=100_000,
        )
        assert calls[0]["address"] == "auto"
        assert calls[1]["address"] is None
        assert _read_output(out).height == 1  # the pair matched
    finally:
        ray.shutdown()


# --------------------------------------------------------------------------- #
# HATS Dir= convention (regression: Dir was `pix // 10000`, so any output
# pixel >= 10000 landed in a directory no HATS-conforming reader looks in)
# --------------------------------------------------------------------------- #
def test_hats_dir_follows_hats_convention() -> None:
    """`Dir=` must be ``(pix // 10000) * 10000`` — the value ``hats`` derives."""
    assert ray_union._hats_dir(0) == 0
    assert ray_union._hats_dir(9_999) == 0
    assert ray_union._hats_dir(10_000) == 10_000
    assert ray_union._hats_dir(12_345) == 10_000
    assert ray_union._hats_dir(1_234_567) == 1_230_000

    pytest.importorskip("hats")
    from hats.pixel_math import HealpixPixel

    for pix in (0, 9_999, 10_000, 12_345, 1_234_567):
        assert ray_union._hats_dir(pix) == HealpixPixel(0, pix).dir, pix


def _deep_pixel_catalogue(root: Path, name: str, df: pl.DataFrame, order: int) -> CatalogueSource:
    """HATS catalogue tiled at a chosen order (mirrors `_pixeled_catalogue`)."""
    import cdshealpix
    from astropy.coordinates import Latitude, Longitude

    root.mkdir(parents=True, exist_ok=True)
    ra = df["ra"].to_numpy()
    dec = df["dec"].to_numpy()
    pix = np.asarray(
        cdshealpix.lonlat_to_healpix(
            Longitude(ra, unit="deg"), Latitude(dec, unit="deg"), np.full(len(df), order)
        )
    ).astype(np.int64)
    for p in np.unique(pix):
        sub = df.filter(pl.Series("_p", pix) == p)
        rel = (
            root
            / "dataset"
            / f"Norder={order}"
            / f"Dir={(int(p) // 10000) * 10000}"
            / f"Npix={int(p)}.parquet"
        )
        rel.parent.mkdir(parents=True, exist_ok=True)
        sub.write_parquet(rel)
    (root / "properties").write_text(
        "dataproduct_type=object\nobs_collection=xmatch-test\n"
        "hats_col_ra=ra\nhats_col_dec=dec\nhats_ordering=NESTED\n"
        "hats_nrows=%d\nhats_max_depth=%d\n" % (df.height, order)
    )
    return CatalogueSource(name=name, is_local=True, path=root, ra_column="ra", dec_column="dec")


def test_union_output_dir_convention_for_deep_pixels(tmp_path: Path) -> None:
    """Output partitions above pixel 10000 must use the ``*10000`` Dir tree.

    The bug was invisible below Norder 7 (nside <= 64 has no pixel >= 10000),
    so a small deep-order input pins it: every emitted file, and every
    ``partition_info`` row, must agree with the ``hats`` library's
    ``HealpixPixel.dir``.
    """
    import cdshealpix
    from astropy.coordinates import Latitude, Longitude
    from hats.pixel_math import HealpixPixel

    order = 6
    # Pick a coordinate whose order-``order`` pixel is >= 10000.
    ra0, dec0 = None, -40.0
    for candidate in np.arange(0.0, 360.0, 5.0):
        pix = int(
            np.asarray(
                cdshealpix.lonlat_to_healpix(
                    Longitude(np.asarray([candidate]), unit="deg"),
                    Latitude(np.asarray([dec0]), unit="deg"),
                    order,
                )
            )[0]
        )
        if pix > 10_000:
            ra0 = float(candidate)
            break
    assert ra0 is not None, "no order-6 pixel > 10000 found"

    a = pl.DataFrame({"ra": [ra0], "dec": [dec0], "id": ["a"]})
    b = pl.DataFrame({"ra": [ra0 + 0.00005], "dec": [dec0], "id": ["b"]})
    src_a = _deep_pixel_catalogue(tmp_path / "a", "a", a, order)
    src_b = _deep_pixel_catalogue(tmp_path / "b", "b", b, order)

    out = tmp_path / "out"
    ray_union.ray_union_match(
        [src_a, src_b],
        sep_arcsec=5.0,
        output_file=str(out),
        hats_threshold=1_000_000,
        max_tuples=1_000,
    )

    files = sorted(out.rglob("Npix=*.parquet"))
    assert files, "expected output partitions"
    for f in files:
        order_name = f.parent.parent.name
        dir_name = f.parent.name
        pix = int(f.name.removeprefix("Npix=").removesuffix(".parquet"))
        assert pix > 10_000, f"test needs a deep pixel, got {pix}"
        expected = HealpixPixel(int(order_name.removeprefix("Norder=")), pix).dir
        assert dir_name == f"Dir={expected}", f"{f}: {dir_name} != Dir={expected}"

    info = pl.read_csv(out / "partition_info.csv")
    for row in info.iter_rows(named=True):
        assert row["Dir"] == HealpixPixel(row["Norder"], row["Npix"]).dir

    import hats  # noqa: PLC0415

    assert hats.read_hats(out).get_healpix_pixels(), "output must be readable"
