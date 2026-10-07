"""Adversarial checks for distributed matching, independent of a Ray cluster."""

import dataclasses

import numpy as np
import polars as pl
import pytest

from tests.test_ray_union import _pixeled_catalogue
from xmatch import hats_native, matchers, ray_union


def _combos(pools, *, max_tuples=10000, matcher="sky", **kwargs):
    return ray_union._block_combos(
        np.array([0.0]),
        np.array([0.0]),
        np.array([0]),
        pools,
        matchers._arcsec_to_chord(10.0),
        max_tuples,
        centre_label=1,
        cat_labels=list(range(2, len(pools) + 2)),
        matcher=matcher,
        **kwargs,
    )


def test_union_cap_uses_actual_separations_with_sparse_pool_ids():
    ra = np.array([10.0, 2 / 3600, 20.0, 1 / 3600, 30.0, 3 / 3600])
    sels, seps, _, _ = _combos([(ra, np.zeros(6))], max_tuples=2)
    assert set(sels[0]) == {1, 3}
    np.testing.assert_allclose(np.sort(seps), [1.0, 2.0], atol=1e-7)


def test_union_cap_is_exact_against_complete_product():
    pools = [
        (np.array([3.0, 1.0, 2.0]) / 3600, np.zeros(3)),
        (np.array([6.0, 4.0, 5.0]) / 3600, np.zeros(3)),
    ]
    _, complete, _, _ = _combos(pools)
    sels, capped, _, _ = _combos(pools, max_tuples=5)
    assert len(set(zip(sels[0], sels[1], strict=True))) == 5
    np.testing.assert_allclose(np.sort(capped), np.sort(complete)[:5], atol=1e-7)


def test_union_separation_preserves_sub_milliarcsecond_precision():
    _, seps, _, _ = _combos([(np.array([1e-7]), np.array([0.0]))])
    np.testing.assert_allclose(seps, [0.00036], atol=1e-10)


def test_union_ellipse_handles_ra_wrap():
    sels, seps, _, _ = _combos(
        [(np.array([359.9999]), np.array([0.0]))],
        matcher="skyellipse",
        centre_err=(np.ones(1), np.ones(1), np.zeros(1)),
        pool_errs=[(np.ones(1), np.ones(1), np.zeros(1))],
    )
    assert sels[0].tolist() == [0]
    np.testing.assert_allclose(seps, [0.36], atol=1e-8)


def _plan(sources, out, **kwargs):
    return ray_union.build_union_plan(
        sources,
        sep_arcsec=1.0,
        hats_threshold=1000,
        task_rows=10000,
        chunk_memory_gb=1.0,
        max_tuples=1000,
        out_dir=str(out),
        **kwargs,
    )


def test_union_memory_guard_uses_binary_gibibytes(tmp_path, monkeypatch):
    from xmatch.exceptions import CrossMatchError

    sources = [
        _pixeled_catalogue(
            tmp_path / str(i),
            str(i),
            pl.DataFrame({"ra": [10.0], "dec": [0.0]}),
            6,
        )
        for i in range(2)
    ]
    # Two matching one-row partitions form a 2-row pool. 1e9 bytes is below
    # 0.95 GiB, while the exact 1 GiB estimate must fit a 1 GiB budget.
    for bytes_per_row, budget_gib in ((500_000_000, 0.95), (512 * 1024**2, 1.0)):
        monkeypatch.setattr(ray_union, "_BYTES_PER_ROW", bytes_per_row)
        plan = ray_union.build_union_plan(
            sources,
            sep_arcsec=1.0,
            hats_threshold=1000,
            task_rows=10000,
            chunk_memory_gb=budget_gib,
            max_tuples=1000,
            out_dir=str(tmp_path / "out"),
        )
        assert plan.chunks[0].est_rows == 2

    # A row count below task_rows must still respect the independent memory budget.
    with pytest.raises(CrossMatchError, match="chunk-memory-gb"):
        ray_union.build_union_plan(
            sources,
            sep_arcsec=1.0,
            hats_threshold=1000,
            task_rows=10000,
            chunk_memory_gb=0.99,
            max_tuples=1000,
            out_dir=str(tmp_path / "out"),
        )


def test_assemble_block_does_not_allocate_unused_null_frame(monkeypatch):
    cats = [
        ray_union.CataloguePlan(
            "left",
            "",
            "",
            "ra",
            "dec",
            final_cols=["ra", "dec", "id"],
            dtypes={"ra": pl.Float64, "dec": pl.Float64, "id": pl.Int64},
        ),
        ray_union.CataloguePlan(
            "right",
            "",
            "",
            "ra_2",
            "dec_2",
            final_cols=["ra_2", "dec_2", "id_2"],
            dtypes={"ra_2": pl.Float64, "dec_2": pl.Float64, "id_2": pl.Int64},
        ),
    ]
    plan = ray_union.UnionPlan(cats, 1.0, 0.0, 10, [], "")

    def unexpected_null_frame(*args, **kwargs):
        raise AssertionError("a nonempty partner frame must not allocate null columns")

    monkeypatch.setattr(ray_union, "_null_cat_frame", unexpected_null_frame)
    result = ray_union._assemble_block(
        plan,
        0,
        pl.DataFrame({"ra": [10.0], "dec": [0.0], "id": [1]}),
        {1: pl.DataFrame({"ra_2": [10.0], "dec_2": [0.0], "id_2": [2]})},
        {1: np.array([0])},
        np.array([0.0]),
        ["1+2"],
        np.array([0]),
    )
    assert result.select("id", "id_2").rows() == [(1, 2)]


def test_union_plan_covers_actual_uncertainties(tmp_path):
    sources = []
    for i, ra in enumerate([0.0, 0.15]):
        src = _pixeled_catalogue(
            tmp_path / str(i), str(i), pl.DataFrame({"ra": [ra], "dec": [0.0], "err": [400.0]}), 12
        )
        sources.append(dataclasses.replace(src, ra_err_column="err", dec_err_column="err"))
    plan = _plan(sources, tmp_path / "out", matcher="skyerr")
    assert plan.chunks[0].cand_idx.get(1) == [0]
    result = ray_union._run_chunk(plan, plan.chunks[0])
    assert result["rows"] == 1


def test_union_plan_covers_actual_motion(tmp_path):
    moving = _pixeled_catalogue(
        tmp_path / "a",
        "a",
        pl.DataFrame({"ra": [0.0], "dec": [0.0], "pmra": [360000.0], "pmdec": [0.0]}),
        12,
    )
    moving = dataclasses.replace(moving, epoch=2000.0, pm_ra_column="pmra", pm_dec_column="pmdec")
    fixed = _pixeled_catalogue(
        tmp_path / "b", "b", pl.DataFrame({"ra": [0.9998984794], "dec": [0.0]}), 12
    )
    plan = _plan([moving, fixed], tmp_path / "out", target_epoch=2010.0)
    assert plan.chunks[0].cand_idx.get(1) == [0]
    assert ray_union._run_chunk(plan, plan.chunks[0])["rows"] == 1


def test_native_hats_handles_different_partition_orders(tmp_path):
    sources = []
    for i, order in enumerate([6, 9]):
        src = _pixeled_catalogue(
            tmp_path / str(i), str(i), pl.DataFrame({"ra": [42.0], "dec": [5.0], "id": [i]}), order
        )
        sources.append(
            dataclasses.replace(src, access_method="hats", access_identifier=str(src.path))
        )
    result = hats_native.hats_native_crossmatch(*sources, matchers.MatchSpec(radius_arcsec=1.0))
    assert result.height == 1
    assert result["id_2"].to_list() == [1]


def test_fof_connects_via_secondary_catalogue():
    from xmatch import CrossMatch

    frames = [
        pl.DataFrame({"ra": [x / 3600], "dec": [0.0], "id": [i]})
        for i, x in enumerate([0.0, 0.75, 1.5])
    ]
    result = CrossMatch().fof_match(frames, radius_arcsec=1.0)
    assert result["_src_cats"].to_list() == ["1+2+3"]


def test_ray_union_honors_error_and_motion_side_overrides():
    from xmatch.crossmatch import _side_overrides

    supplied = {
        "ra_err_column_1": "err",
        "pos_err_units_1": "mas",
        "epoch_1": 2000.0,
        "pm_ra_column_1": "pmra",
        "pm_dec_column_1": "pmdec",
    }
    assert _side_overrides(supplied, 1) == {k[:-2]: v for k, v in supplied.items()}


def test_ray_union_rejects_ignored_semantic_options(tmp_path):
    from xmatch import CrossMatch
    from xmatch.exceptions import CrossMatchError

    with pytest.raises(CrossMatchError, match="filter_expr"):
        CrossMatch().union_match(
            ["missing-a", "missing-b"],
            engine="ray-union",
            output_file=tmp_path / "out",
            filter_expr="ra > 0",
        )


def test_ray_union_rejects_output_overlapping_input(tmp_path):
    from xmatch.exceptions import CrossMatchError

    src = _pixeled_catalogue(tmp_path / "a", "a", pl.DataFrame({"ra": [0.0], "dec": [0.0]}), 2)
    with pytest.raises(CrossMatchError, match="overlap"):
        ray_union.ray_union_match([src, src], sep_arcsec=1.0, output_file=str(src.path))


@pytest.mark.parametrize("radius", [float("nan"), float("inf"), -1.0, 0.0])
def test_union_plan_rejects_nonfinite_or_nonpositive_radius(tmp_path, radius):
    from xmatch.exceptions import CrossMatchError

    src = _pixeled_catalogue(tmp_path / "a", "a", pl.DataFrame({"ra": [0.0], "dec": [0.0]}), 2)
    with pytest.raises(CrossMatchError, match="radius_arcsec"):
        ray_union.build_union_plan(
            [src],
            sep_arcsec=radius,
            hats_threshold=1000,
            task_rows=10000,
            chunk_memory_gb=1.0,
            max_tuples=1000,
            out_dir=str(tmp_path / "out"),
        )


def test_cli_global_config_can_precede_subcommand():
    from xmatch.cli import _split_subcommand

    assert _split_subcommand(["--config", "custom.yaml", "list"]) == (
        "list",
        ["--config", "custom.yaml"],
    )
    assert _split_subcommand(["--config", "list", "describe", "gaia"]) == (
        "describe",
        ["--config", "list", "gaia"],
    )


def test_cli_invalid_ranking_weights_fail_loudly():
    from xmatch.cli import _parse_extra_distance_cols
    from xmatch.exceptions import CrossMatchError

    with pytest.raises(CrossMatchError, match="weight"):
        _parse_extra_distance_cols("mag:oops")


def test_nway_uses_per_axis_uncertainty(monkeypatch):
    import xmatch.crossmatch as module
    from xmatch import CrossMatch

    captured = []
    original = module.compute_nway_p_match

    def capture(ras, decs, sigmas, *args, **kwargs):
        captured.extend(sigmas)
        return original(ras, decs, sigmas, *args, **kwargs)

    monkeypatch.setattr(module, "compute_nway_p_match", capture)
    frame = pl.DataFrame({"ra": [42.0], "dec": [5.0], "err": [1.0]})
    CrossMatch().nway_match(
        [frame, frame],
        radius_arcsec=1.0,
        ra_err_column_1="err",
        dec_err_column_1="err",
        ra_err_column_2="err",
        dec_err_column_2="err",
    )
    np.testing.assert_allclose(captured, np.ones((2, 1)))


def test_mirrored_cone_keeps_exact_centres_at_tiny_radii():
    from xmatch.crossmatch import _cone_filter_frame

    for dec in np.linspace(-89.0, 89.0, 50):
        frame = pl.DataFrame({"ra": [42.0], "dec": [dec]})
        assert _cone_filter_frame(frame, "ra", "dec", 42.0, dec, 1e-9).height == 1


def test_union_output_has_valid_coordinates_and_disjoint_tiling(tmp_path):
    import cdshealpix
    from astropy.coordinates import Latitude, Longitude

    a = _pixeled_catalogue(tmp_path / "a", "a", pl.DataFrame({"ra": [42.0], "dec": [5.0]}), 3)
    # A secondary-only source lies inside the primary's coarse pixel but is
    # outside the matching radius. Its fine input tile overlaps the hub tile.
    b = _pixeled_catalogue(tmp_path / "b", "b", pl.DataFrame({"ra": [42.1], "dec": [5.0]}), 7)
    out = tmp_path / "out"
    plan = _plan([a, b], out)
    for chunk in plan.chunks:
        ray_union._run_chunk(plan, chunk)
    for rest in plan.rest:
        ray_union._run_rest(plan, rest)
    ray_union._assemble(plan)
    props = hats_native._read_properties(out)
    pixels = hats_native.list_hats_pixels(out)
    assert len(pixels) == 1
    for order, pix, file in pixels:
        frame = pl.read_parquet(file)
        ra, dec = props["hats_col_ra"], props["hats_col_dec"]
        assert frame[ra].null_count() == frame[dec].null_count() == 0
        actual = cdshealpix.lonlat_to_healpix(
            Longitude(frame[ra].to_numpy(), unit="deg"),
            Latitude(frame[dec].to_numpy(), unit="deg"),
            order,
        )
        assert np.all(actual == pix)


def test_native_hats_empty_result_retains_measurement_schema(tmp_path):
    sources = []
    for i, ra in enumerate([42.0, 42.01]):
        src = _pixeled_catalogue(
            tmp_path / str(i), str(i), pl.DataFrame({"ra": [ra], "dec": [5.0], "id": [i]}), 6
        )
        sources.append(
            dataclasses.replace(src, access_method="hats", access_identifier=str(src.path))
        )
    result = hats_native.hats_native_crossmatch(*sources, matchers.MatchSpec(radius_arcsec=1.0))
    assert result.is_empty()
    assert set(result.columns) == {"ra", "dec", "id", "ra_2", "dec_2", "id_2", "sep_arcsec"}


def test_hats_loader_retains_zero_row_partition_schema(tmp_path):
    from tests.test_ray_union import _catalogue

    empty = pl.DataFrame(schema={"ra": pl.Float64, "dec": pl.Float64, "id": pl.String})
    src = _catalogue(tmp_path / "empty", "empty", empty)
    src = dataclasses.replace(src, access_method="hats", access_identifier=str(src.path))
    assert hats_native.load_hats_all(src).schema == empty.schema


def test_sparse_deep_cone_does_not_expand_full_sky(monkeypatch):
    # A full-sky cone at order 29 covers >10^18 cells, but only three exist.
    # The covering must stay compressed and intersect actual partitions.
    class Cover:
        @staticmethod
        def cone_search(*args):
            return np.arange(12), np.zeros(12, dtype=int), np.ones(12, dtype=bool)

    monkeypatch.setattr(ray_union, "_cdshealpix", lambda: Cover())
    monkeypatch.setattr(ray_union, "_pixel_center_deg", lambda *args: (0.0, 0.0))
    cat = ray_union.CataloguePlan(
        "sparse",
        "",
        "",
        "ra",
        "dec",
        partitions=[ray_union.PartitionPlan(29, pix, "") for pix in [0, 100, 2**58]],
    )
    index = ray_union._partition_index(cat, 29)
    got = ray_union._cone_candidate_idx(cat.partitions[0], [cat], [29], 180.0, pix_index=[index])
    assert got == {0: [0, 1, 2]}


def test_union_rejects_overlapping_input_tiles():
    cat = ray_union.CataloguePlan(
        "overlap",
        "",
        "",
        "ra",
        "dec",
        partitions=[ray_union.PartitionPlan(1, 0, ""), ray_union.PartitionPlan(2, 1, "")],
    )
    from xmatch.exceptions import CrossMatchError

    with pytest.raises(CrossMatchError, match="overlap"):
        ray_union._partition_index(cat, 2)


def test_union_null_columns_preserve_nested_and_temporal_types(tmp_path):
    frames = [
        pl.DataFrame({"ra": [42.0], "dec": [5.0], "samples": [[1.0, 2.0]]}),
        pl.DataFrame({"ra": [43.0], "dec": [5.0], "samples": [[3.0]]}).with_columns(
            pl.Series("observed", [1], dtype=pl.Datetime("ns", "UTC"))
        ),
    ]
    sources = [_pixeled_catalogue(tmp_path / str(i), str(i), f, 6) for i, f in enumerate(frames)]
    plan = _plan(sources, tmp_path / "out")
    null = ray_union._null_cat_frame(plan.catalogues[1], 1)
    assert null.schema["samples_2"] == pl.List(pl.Float64)
    assert null.schema["observed"] == pl.Datetime("ns", "UTC")
    for chunk in plan.chunks:
        ray_union._run_chunk(plan, chunk)
    for rest in plan.rest:
        ray_union._run_rest(plan, rest)
    assert ray_union._assemble(plan)["rows"] == 2


def test_empty_union_is_readable_hats_with_schema(tmp_path):
    from tests.test_ray_union import _catalogue

    empty = pl.DataFrame(schema={"ra": pl.Float64, "dec": pl.Float64, "id": pl.String})
    source = _catalogue(tmp_path / "empty", "empty", empty)
    out = tmp_path / "out"
    plan = _plan([source], out)
    for chunk in plan.chunks:
        ray_union._run_chunk(plan, chunk)
    assert ray_union._assemble(plan)["rows"] == 0
    assert hats_native._read_properties(out)["hats_nrows"] == "0"
    result = hats_native.load_hats_all(
        dataclasses.replace(source, path=out, access_identifier=str(out))
    )
    assert result.schema["id"] == pl.String
    assert result.height == 0


def test_native_hats_margin_honours_radii_larger_than_adjacent_pixels(monkeypatch):
    import sys
    import types

    import cdshealpix
    from astropy.coordinates import Latitude, Longitude

    monkeypatch.setitem(sys.modules, "torchsky.catalogs", None)
    # Previously the optional healpy path returned only immediate neighbours,
    # regardless of the requested match radius.
    monkeypatch.setitem(
        sys.modules, "healpy", types.SimpleNamespace(get_all_neighbours=lambda *a, **k: [])
    )
    order = 6
    centre = int(
        cdshealpix.lonlat_to_healpix(Longitude(42.0, unit="deg"), Latitude(5.0, unit="deg"), order)[
            0
        ]
    )
    distant = int(
        cdshealpix.lonlat_to_healpix(Longitude(45.0, unit="deg"), Latitude(5.0, unit="deg"), order)[
            0
        ]
    )
    assert distant in hats_native.margin_pixels(order, centre, 4 * 3600, [distant])


def test_nway_photometric_score_is_independent_of_chunk_size():
    from xmatch import CrossMatch

    left = pl.DataFrame(
        {"ra": [0.0, 1.0, 2.0], "dec": [0.0] * 3, "mag": [0.0, 5.0, 10.0], "err": [10000.0] * 3}
    )
    right = left.with_columns(pl.Series("mag", [0.1, 5.2, 10.3]))
    params = {f"{coord}_err_column_{side}": "err" for side in (1, 2) for coord in ("ra", "dec")}
    scores = []
    for chunk_size in (1, 3):
        out = CrossMatch().nway_match(
            [left, right], prior_columns=["mag"], chunk_size=chunk_size, **params
        )
        scores.append(out.sort("ra")["p_match"].to_numpy())
    np.testing.assert_allclose(scores[0], scores[1], rtol=0.0, atol=1e-14)


def test_nway_empty_result_retains_input_schema():
    from xmatch import CrossMatch

    left = pl.DataFrame({"ra": [0.0], "dec": [0.0], "id": ["a"]})
    right = pl.DataFrame({"ra": [1.0], "dec": [0.0], "id": ["b"]})
    out = CrossMatch().nway_match([left, right])
    assert out.schema["id"] == out.schema["id_2"] == pl.String
    assert out.is_empty()


def test_nway_scores_use_propagated_covariance(monkeypatch):
    from xmatch import CrossMatch, crossmatch

    captured = []

    def propagate(left, right, *args, propagate_covariance=False, **kwargs):
        assert propagate_covariance
        return left.with_columns(
            pl.lit(4.0).alias(matchers._PROPAGATED_COV_EE),
            pl.lit(16.0).alias(matchers._PROPAGATED_COV_NN),
            pl.lit(0.0).alias(matchers._PROPAGATED_COV_EN),
        ), right

    def score(ras, decs, sigmas, *args, **kwargs):
        captured.extend(sigmas)
        return np.ones(len(ras[0]))

    monkeypatch.setattr(matchers, "_apply_proper_motion", propagate)
    monkeypatch.setattr(crossmatch, "compute_nway_p_match", score)
    frame = pl.DataFrame({"ra": [42.0], "dec": [5.0], "err": [1.0]})
    params = {f"{coord}_err_column_{side}": "err" for side in (1, 2) for coord in ("ra", "dec")}
    CrossMatch().nway_match([frame, frame], target_epoch=2020.0, **params)
    np.testing.assert_allclose(captured, np.full((2, 1), np.sqrt(10.0)))


def test_mirrored_cone_does_not_inflate_tiny_requested_radius():
    from xmatch.crossmatch import _cone_filter_frame

    frame = pl.DataFrame({"ra": [42.0, 42.0 + 1e-8], "dec": [0.0, 0.0]})
    selected = _cone_filter_frame(frame, "ra", "dec", 42.0, 0.0, 1e-9)
    assert selected.height == 1


def test_union_tuple_cap_handles_products_larger_than_int64():
    pools = [(np.array([1 / 3600]), np.array([0.0])) for _ in range(70)]
    sels, seps, membership, _ = _combos(pools, max_tuples=1)
    assert len(seps) == 1
    assert all(indices.tolist() == [0] for indices in sels.values())
    assert membership[0].count("+") == 70


def test_remote_projection_preserves_coordinates_and_scoring_columns(monkeypatch):
    from xmatch import CrossMatch, MatchRequest, crossmatch, remote_tap
    from xmatch.sources import CatalogueSource

    captured = []
    source = CatalogueSource(
        "remote",
        is_local=False,
        access_method="tap",
        access_identifier="t",
        ra_column="ra",
        dec_column="dec",
        ra_err_column="err",
        dec_err_column="err",
        default_columns=["id"],
    )
    req = MatchRequest.from_params(
        "a",
        "b",
        ra=42.0,
        dec=5.0,
        radius_deg=0.1,
        prior_columns=["mag"],
        extra_distance_cols={"colour": 1.0},
    )
    monkeypatch.setattr(crossmatch, "_try_mirrored_cone", lambda *a: None)
    monkeypatch.setattr(
        remote_tap,
        "download_from_tap",
        lambda *a, columns=None, **k: captured.append(columns) or pl.DataFrame(),
    )
    CrossMatch()._download_remote(source, req, prefix="1")
    assert {"id", "ra", "dec", "err", "mag", "colour"}.issubset(captured[0])


def test_shared_endpoint_is_applied_to_multi_source_resolution():
    from xmatch.crossmatch import _side_overrides

    assert (
        _side_overrides({"endpoint": "vizier", "endpoint_2": "noirlab"}, 1)["endpoint"] == "vizier"
    )
    assert (
        _side_overrides({"endpoint": "vizier", "endpoint_2": "noirlab"}, 2)["endpoint"] == "noirlab"
    )


def test_nway_requires_declared_scoring_uncertainties():
    from xmatch import CrossMatch
    from xmatch.exceptions import CrossMatchError

    frame = pl.DataFrame({"ra": [42.0], "dec": [5.0]})
    with pytest.raises(CrossMatchError, match="positional.*error"):
        CrossMatch().nway_match([frame, frame])


def test_nonlocal_partition_fingerprint_detects_same_size_edits(tmp_path, monkeypatch):
    import shutil

    from xmatch.storage import Storage

    file = tmp_path / "part"
    file.write_bytes(b"first")

    class Remote(Storage):
        def stage_in(self, rel, local):
            shutil.copyfile(file, local)

    cat = ray_union.CataloguePlan(
        "remote", "vos:test", "", "ra", "dec", partitions=[ray_union.PartitionPlan(0, 0, "part")]
    )
    plan = ray_union.UnionPlan([cat], 1.0, 0.0, 10, [], str(tmp_path / "out"))
    monkeypatch.setattr(ray_union, "open_storage", lambda root: Remote())
    first = ray_union._input_fingerprint(plan)
    file.write_bytes(b"other")
    assert ray_union._input_fingerprint(plan) != first


def test_tap_same_service_best_uses_canonical_matcher(monkeypatch):
    from xmatch import CrossMatch, MatchRequest, remote_tap
    from xmatch.sources import CatalogueSource

    left = CatalogueSource(
        "a",
        is_local=False,
        access_method="tap",
        access_identifier="a",
        tap_url="https://example.invalid/tap",
        ra_column="ra",
        dec_column="dec",
    )
    right = dataclasses.replace(left, name="b", access_identifier="b")
    cm = CrossMatch()
    frames = [
        pl.DataFrame({"ra": [42.0], "dec": [5.0]}),
        pl.DataFrame({"ra": [42.0 + 0.2 / 3600, 42.0 + 0.8 / 3600], "dec": [5.0, 5.0]}),
    ]
    monkeypatch.setattr(
        cm, "_download_remote", lambda src, *a, prefix=None, **k: frames[int(prefix) - 1]
    )
    monkeypatch.setattr(
        remote_tap, "tap_self_join", lambda *a, **k: pytest.fail("server join cannot honour best")
    )
    req = MatchRequest.from_params(
        "a", "b", engine="fast", find="best", ra=42.0, dec=5.0, radius_deg=0.01
    )
    result = cm._remote_vs_remote(left, right, req).collect()
    assert result.height == 1
    assert result["sep_arcsec"][0] < 0.3


def test_cds_uncertainty_request_uses_canonical_matcher(monkeypatch):
    from xmatch import CrossMatch, MatchRequest, remote_cds
    from xmatch.sources import CatalogueSource

    left = CatalogueSource(
        "a", is_local=True, ra_column="ra", dec_column="dec", default_pos_error_arcsec=0.01
    )
    right = dataclasses.replace(
        left, name="b", is_local=False, access_method="cds_xmatch", access_identifier="cat"
    )
    local = pl.DataFrame({"ra": [42.0], "dec": [5.0]})
    left = left.with_frame(local.lazy())
    remote = pl.DataFrame({"ra": [42.0 + 0.5 / 3600], "dec": [5.0]})
    cm = CrossMatch()
    monkeypatch.setattr(cm, "_download_remote", lambda *a, **k: remote)
    monkeypatch.setattr(
        remote_cds,
        "cds_xmatch_local_remote",
        lambda *a, **k: pytest.fail("server cannot honour skyerr"),
    )
    req = MatchRequest.from_params("a", "b", engine="fast", matcher="skyerr")
    assert cm._local_vs_remote(left, right, req).collect().is_empty()


def test_nway_accepts_explicit_uncertainties_for_every_catalogue(monkeypatch):
    from xmatch import CrossMatch, crossmatch

    captured = []

    def score(ras, decs, sigmas, *args, **kwargs):
        captured.extend(sigmas)
        return np.ones(len(ras[0]))

    monkeypatch.setattr(crossmatch, "compute_nway_p_match", score)

    frame = pl.DataFrame({"ra": [42.0], "dec": [5.0]})
    result = CrossMatch().nway_match(
        [frame, frame, frame],
        **{f"default_pos_error_arcsec_{i}": i * 0.1 for i in (1, 2, 3)},
    )
    np.testing.assert_allclose(captured, [[0.1], [0.2], [0.3]])
    assert result.height == 1
    assert result["p_match"].is_finite().all()


def test_mirrored_cone_honours_ring_partition_ordering(tmp_path, monkeypatch):
    import cdshealpix

    from xmatch import mirror
    from xmatch.crossmatch import _try_mirrored_cone

    source = _pixeled_catalogue(
        tmp_path / "mirror", "mirror", pl.DataFrame({"ra": [42.0], "dec": [5.0]}), 6
    )
    order, nested, file = hats_native.list_hats_pixels(source.path)[0]
    ring = int(cdshealpix.to_ring(np.array([nested], dtype=np.uint64), order)[0])
    ring_file = (
        source.path
        / "dataset"
        / f"Norder={order}"
        / f"Dir={ray_union._hats_dir(ring)}"
        / f"Npix={ring}.parquet"
    )
    ring_file.parent.mkdir(parents=True, exist_ok=True)
    file.rename(ring_file)
    properties = source.path / "properties"
    properties.write_text(properties.read_text().replace("NESTED", "RING"))
    source = dataclasses.replace(source, is_local=False, access_method="tap")
    monkeypatch.setattr(mirror, "locate_mirrored", lambda *a, **k: (str(source.path), ""))
    result = _try_mirrored_cone(source, None, 42.0, 5.0, 0.01)
    assert result is not None
    assert result.height == 1


@pytest.mark.parametrize(
    "ra,dec,radius", [(np.nan, 5.0, 0.1), (42.0, 95.0, 0.1), (42.0, 5.0, -1.0)]
)
def test_cached_remote_download_validates_region_before_cache(monkeypatch, ra, dec, radius):
    from xmatch import CrossMatch, MatchRequest, crossmatch
    from xmatch.exceptions import CrossMatchError

    cm = CrossMatch()
    source = cm.resolve_source("gaia_esa", {})
    monkeypatch.setattr(
        crossmatch, "_try_mirrored_cone", lambda *a: pl.DataFrame({"ra": [42.0], "dec": [5.0]})
    )
    req = MatchRequest("a", "b", ra=ra, dec=dec, radius_deg=radius)
    with pytest.raises(CrossMatchError, match="region"):
        cm._download_remote(source, req, prefix="1")


@pytest.mark.parametrize(
    "spec", [matchers.MatchSpec(matcher="skyerr"), matchers.MatchSpec(target_epoch=2020.0)]
)
def test_remote_variable_halos_require_explicit_region(monkeypatch, spec):
    from xmatch import CrossMatch, MatchRequest, crossmatch
    from xmatch.exceptions import CrossMatchError

    cm = CrossMatch()
    frame = pl.DataFrame({"ra": [42.0], "dec": [5.0]})
    source = cm.resolve_source("gaia_esa", {})
    local = cm.resolve_source(frame, {})
    monkeypatch.setattr(crossmatch, "_try_mirrored_cone", lambda *a: frame)
    req = MatchRequest("a", "b", spec=spec)
    with pytest.raises(CrossMatchError, match="explicit region"):
        cm._download_remote(source, req, prefix="2", region_from=frame.lazy(), local=local)


@pytest.mark.parametrize("find", ["best", "all"])
def test_ray_pixel_tree_respects_one_cpu_allocation(monkeypatch, find):
    from types import SimpleNamespace

    import ray
    import scipy.spatial

    from xmatch import matchers, ray_engine

    real_tree = scipy.spatial.cKDTree
    calls = []

    def tree(points):
        actual = real_tree(points)

        def query(*args, **kwargs):
            calls.append(kwargs["workers"])
            return actual.query(*args, **kwargs)

        def query_ball_point(*args, **kwargs):
            calls.append(kwargs["workers"])
            return actual.query_ball_point(*args, **kwargs)

        return SimpleNamespace(n=actual.n, query=query, query_ball_point=query_ball_point)

    monkeypatch.setattr(scipy.spatial, "cKDTree", tree)
    monkeypatch.setattr(ray, "get", lambda value: value)
    monkeypatch.setattr(ray_engine, "_RAY_PIXEL_BATCH", None)
    function = ray_engine._get_ray_pixel_batch()._function
    xyz = np.array([[1.0, 0.0, 0.0]])
    left, right, _ = function(
        0,
        np.array([0]),
        xyz,
        {0: xyz},
        {0: np.array([0])},
        matchers.MatchSpec(find=find),
        0.01,
    )
    assert left.tolist() == right.tolist() == [0]
    assert calls == [1]


def test_native_hats_ray_worker_limits_tree_threads(monkeypatch):
    from xmatch import hats_native, matchers
    from xmatch.sources import CatalogueSource

    actual_match = matchers._scipy_match
    calls = []

    def match(*args, **kwargs):
        calls.append(kwargs.get("workers"))
        return actual_match(*args, **kwargs)

    monkeypatch.setattr(matchers, "_scipy_match", match)
    frame = pl.DataFrame({"id": [1], "ra": [10.0], "dec": [0.0]})
    source = CatalogueSource(name="test", is_local=True, ra_column="ra", dec_column="dec")
    result = hats_native._match_frames(
        source, source, frame, frame, matchers.MatchSpec(), "ray", "_2", worker_task=True
    )
    assert result.height == 1
    assert calls == [1]


@pytest.mark.parametrize("backend", ["pair", "union"])
def test_strict_ray_preserves_failed_explicit_address(monkeypatch, tmp_path, backend):
    import os

    import ray

    from xmatch import ray_engine
    from xmatch.exceptions import CrossMatchError
    from xmatch.sources import CatalogueSource

    address = "unreachable-test-address:6379"
    calls = []

    def init(**kwargs):
        calls.append(kwargs["address"])
        raise ConnectionError("unreachable")

    monkeypatch.setenv("RAY_ADDRESS", address)
    monkeypatch.setattr(ray, "is_initialized", lambda: False)
    monkeypatch.setattr(ray, "init", init)
    frame = pl.DataFrame({"id": [1], "ra": [10.0], "dec": [0.0]})
    with pytest.raises(CrossMatchError, match="fallback_policy='error'"):
        if backend == "pair":
            source = CatalogueSource(name="test", is_local=True, ra_column="ra", dec_column="dec")
            ray_engine.ray_zone_match(
                frame, frame, source, source, matchers.MatchSpec(fallback_policy="error")
            )
        else:
            source = _pixeled_catalogue(tmp_path / "input", "test", frame, order=1)
            ray_union.ray_union_match(
                [source, source],
                sep_arcsec=1,
                output_file=str(tmp_path / "output"),
                fallback_policy="error",
            )
    assert calls == [address]
    assert os.environ["RAY_ADDRESS"] == address


def test_ray_address_warning_fallback_preserves_environment(monkeypatch, caplog):
    import os

    import ray

    from xmatch import ray_engine
    from xmatch.sources import CatalogueSource

    def init(**kwargs):
        raise ConnectionError("unreachable")

    monkeypatch.setenv("RAY_ADDRESS", "unreachable-test-address:6379")
    monkeypatch.setattr(ray, "is_initialized", lambda: False)
    monkeypatch.setattr(ray, "init", init)
    frame = pl.DataFrame({"ra": [10.0], "dec": [0.0]})
    source = CatalogueSource(name="test", is_local=True, ra_column="ra", dec_column="dec")
    left, right, seps = ray_engine.ray_zone_match(
        frame, frame, source, source, matchers.MatchSpec(fallback_policy="warn")
    )
    assert left.tolist() == right.tolist() == [0]
    np.testing.assert_allclose(seps, [0.0], atol=1e-9)
    assert os.environ["RAY_ADDRESS"] == "unreachable-test-address:6379"
    assert "using single-machine zone" in caplog.text
