"""Bounded-memory local CSV/Parquet matching contract."""

from __future__ import annotations

import hashlib
from pathlib import Path

import polars as pl
import pytest

from xmatcher import CrossMatch, MatchRequest, MatchSpec, SideOverrides
from xmatcher.cli import _build_match_subparser
from xmatcher.exceptions import CrossMatchError


def _catalogues(tmp_path: Path) -> tuple[Path, Path]:
    left = pl.DataFrame(
        {
            "id": range(24),
            "ra": [15.0 + i * 0.01 for i in range(24)],
            # Straddle an order=2 declination-zone boundary at zero.
            "dec": [-0.00002 if i % 2 else 0.00002 for i in range(24)],
            "payload": ["left-" + "x" * 100 for _ in range(24)],
        }
    )
    right = pl.DataFrame(
        {
            "id": range(24),
            "ra": [15.0 + i * 0.01 + 0.00001 for i in range(24)],
            "dec": [0.00002 if i % 2 else -0.00002 for i in range(24)],
            "payload": ["right-" + "y" * 100 for _ in range(24)],
        }
    )
    p1, p2 = tmp_path / "left.csv", tmp_path / "right.parquet"
    left.write_csv(p1)
    right.write_parquet(p2)
    return p1, p2


def test_forced_spill_matches_in_memory_and_cleans_scratch(tmp_path):
    p1, p2 = _catalogues(tmp_path)
    cm = CrossMatch()
    expected = cm.crossmatch(p1, p2, radius_arcsec=1.0, engine="fast")
    output = tmp_path / "matches.parquet"
    scratch = tmp_path / "scratch"

    result = cm.crossmatch(
        p1,
        p2,
        output_file=output,
        radius_arcsec=1.0,
        engine="fast",
        memory_budget_bytes=4096,
        scratch_dir=scratch,
        partition_order=2,
    )

    assert result is None
    assert pl.read_parquet(output).equals(expected)
    assert list(scratch.iterdir()) == []
    assert cm.last_spill_stats is not None
    stats = cm.last_spill_stats
    assert stats["batch_pairs"] > 1
    assert stats["peak_proxy_bytes"] <= max(
        stats["memory_budget_bytes"], stats["minimum_batch_proxy_bytes"]
    )


def test_forced_spill_outer_matches_eager_with_unmatched_islands(tmp_path):
    padding = "o" * 600
    left = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "ra": [10.0, 20.0, 30.0],
            "dec": [0.00002, 1.0, -0.00002],
            "left_value": ["matched", "left-only", "boundary"],
            "pad": [padding] * 3,
        }
    )
    right = pl.DataFrame(
        {
            "id": [10, 11, 12],
            "ra": [10.00001, 40.0, 30.00001],
            "dec": [-0.00002, 2.0, 0.00002],
            "right_value": ["matched", "right-only", "boundary"],
            "pad": [padding] * 3,
        }
    )
    p1, p2 = tmp_path / "outer-left.parquet", tmp_path / "outer-right.csv"
    left.write_parquet(p1)
    right.write_csv(p2)
    cm = CrossMatch()
    expected = cm.crossmatch(p1, p2, radius_arcsec=1.0, join_type="1or2", engine="fast")
    output = tmp_path / "outer.parquet"
    scratch = tmp_path / "scratch"

    cm.crossmatch(
        p1,
        p2,
        output_file=output,
        radius_arcsec=1.0,
        join_type="1or2",
        engine="fast",
        memory_budget_bytes=1024,
        scratch_dir=scratch,
        partition_order=2,
    )

    actual = pl.read_parquet(output)
    assert actual.equals(expected)
    assert actual["left_value"].to_list() == ["matched", "boundary", "left-only", None]
    assert actual["right_value"].to_list() == ["matched", "boundary", None, "right-only"]
    assert list(scratch.iterdir()) == []
    assert cm.last_spill_stats is not None
    assert cm.last_spill_stats["peak_proxy_bytes"] <= max(
        cm.last_spill_stats["memory_budget_bytes"],
        cm.last_spill_stats["minimum_batch_proxy_bytes"],
    )
    second = tmp_path / "outer-order4.parquet"
    cm.crossmatch(
        p1,
        p2,
        output_file=second,
        radius_arcsec=1.0,
        join_type="1or2",
        engine="fast",
        memory_budget_bytes=1024,
        partition_order=4,
    )
    assert (
        hashlib.sha256(output.read_bytes()).digest() == hashlib.sha256(second.read_bytes()).digest()
    )


def test_forced_spill_two_catalogue_union_matches_eager_tags(tmp_path):
    p1, p2 = _catalogues(tmp_path)
    right = pl.read_parquet(p2).with_columns(
        pl.when(pl.col("id") == 23).then(999.0).otherwise(pl.col("ra")).alias("ra")
    )
    right.write_parquet(p2)
    cm = CrossMatch()
    expected = cm.union_match([p1, p2], radius_arcsec=1.0, engine="fast")
    output = tmp_path / "union.parquet"

    result = cm.union_match(
        [p1, p2],
        output_file=output,
        radius_arcsec=1.0,
        engine="fast",
        memory_budget_bytes=4096,
        partition_order=2,
    )

    assert result is None
    assert pl.read_parquet(output).equals(expected)
    assert set(expected["_src_cats"]) == {"1+2", "1", "2"}


@pytest.mark.parametrize("empty_side", ["left", "right"])
def test_forced_spill_outer_preserves_nonempty_side(tmp_path, empty_side):
    padding = "e" * 1200
    empty = pl.DataFrame(
        schema={"id": pl.Int64, "ra": pl.Float64, "dec": pl.Float64, "pad": pl.String}
    )
    populated = pl.DataFrame(
        {"id": [1, 2], "ra": [10.0, 20.0], "dec": [0.0, 1.0], "pad": [padding] * 2}
    )
    left, right = (empty, populated) if empty_side == "left" else (populated, empty)
    p1, p2 = tmp_path / "left.parquet", tmp_path / "right.parquet"
    left.write_parquet(p1)
    right.write_parquet(p2)
    expected = CrossMatch().crossmatch(p1, p2, radius_arcsec=1.0, join_type="1or2", engine="fast")
    output = tmp_path / "empty-side.parquet"

    CrossMatch().crossmatch(
        p1,
        p2,
        output_file=output,
        radius_arcsec=1.0,
        join_type="1or2",
        engine="fast",
        memory_budget_bytes=1024,
        partition_order=2,
    )

    assert pl.read_parquet(output).equals(expected)


def test_forced_spill_outer_find_all_matches_eager(tmp_path):
    padding = "a" * 600
    left = pl.DataFrame({"id": [1, 2], "ra": [10.0, 20.0], "dec": [0.0, 0.0], "pad": [padding] * 2})
    right = pl.DataFrame(
        {
            "id": [10, 11, 12],
            "ra": [10.00001, 9.99999, 30.0],
            "dec": [0.0, 0.0, 0.0],
            "pad": [padding] * 3,
        }
    )
    p1, p2 = tmp_path / "all-left.csv", tmp_path / "all-right.csv"
    left.write_csv(p1)
    right.write_csv(p2)
    expected = CrossMatch().crossmatch(
        p1, p2, radius_arcsec=1.0, find="all", join_type="1or2", engine="fast"
    )
    output = tmp_path / "all-outer.parquet"

    CrossMatch().crossmatch(
        p1,
        p2,
        output_file=output,
        radius_arcsec=1.0,
        find="all",
        join_type="1or2",
        engine="fast",
        memory_budget_bytes=1024,
        partition_order=2,
    )

    assert pl.read_parquet(output).equals(expected)


@pytest.mark.parametrize("find", ["best", "all"])
def test_spill_is_deterministic_with_equal_distance_ties(tmp_path, find):
    padding = "z" * 600
    left = pl.DataFrame({"id": [1], "ra": [10.0], "dec": [0.0], "pad": [padding]})
    right = pl.DataFrame(
        {
            "id": [10, 11],
            "ra": [10.0 - 0.25 / 3600.0, 10.0 + 0.25 / 3600.0],
            "dec": [0.0, 0.0],
            "pad": [padding, padding],
        }
    )
    p1, p2 = tmp_path / "left.csv", tmp_path / "right.csv"
    left.write_csv(p1)
    right.write_csv(p2)
    digests = []
    for order in (1, 4):
        output = tmp_path / f"matches-{order}.csv"
        CrossMatch().crossmatch(
            p1,
            p2,
            output_file=output,
            radius_arcsec=1.0,
            find=find,
            engine="fast",
            memory_budget_bytes=1024,
            partition_order=order,
        )
        digests.append(hashlib.sha256(output.read_bytes()).hexdigest())
        matched = pl.read_csv(output)
        assert matched["id_2"].to_list() == ([10] if find == "best" else [10, 11])
    assert digests[0] == digests[1]


def test_uncertainty_halo_matches_exact_astropy_path(tmp_path):
    padding = "p" * 200
    left = pl.DataFrame(
        {
            "id": range(8),
            "ra": [20.0 + i * 0.02 for i in range(8)],
            "dec": [-0.00002] * 8,
            "ra_err": [0.1] * 8,
            "dec_err": [0.1] * 8,
            "pad": [padding] * 8,
        }
    )
    rows = []
    for i in range(8):
        rows.extend(
            [
                (2 * i, 20.0 + i * 0.02 + 0.20 / 3600.0, 0.00002, 0.01, 0.01, padding),
                (2 * i + 1, 20.0 + i * 0.02 + 0.50 / 3600.0, 0.00002, 0.2, 0.2, padding),
            ]
        )
    right = pl.DataFrame(rows, schema=left.schema, orient="row")
    p1, p2 = tmp_path / "left.parquet", tmp_path / "right.parquet"
    left.write_parquet(p1)
    right.write_parquet(p2)
    side = SideOverrides(ra_err_column="ra_err", dec_err_column="dec_err")
    spec = MatchSpec(matcher="skyerr", max_error=3.0, find="best")
    cm = CrossMatch()
    expected = cm.crossmatch_request(
        MatchRequest(p1, p2, spec=spec, side1=side, side2=side, engine="astropy")
    ).sort("id")
    output = tmp_path / "uncertainty.parquet"

    cm.crossmatch_request(
        MatchRequest(
            p1,
            p2,
            spec=spec,
            side1=side,
            side2=side,
            engine="astropy",
            output_file=output,
            memory_budget_bytes=1024,
            partition_order=3,
        )
    )

    assert pl.read_parquet(output).sort("id").equals(expected)


@pytest.mark.parametrize("join_type", ["1and2", "1or2"])
def test_spill_failure_is_atomic_and_cleans_scratch(tmp_path, monkeypatch, join_type):
    p1, p2 = _catalogues(tmp_path)
    output = tmp_path / "matches.parquet"
    output.write_bytes(b"original")
    scratch = tmp_path / "scratch"

    def fail(*args, **kwargs):
        raise RuntimeError("injected batch failure")

    monkeypatch.setattr("xmatcher.out_of_core.sky_match", fail)
    with pytest.raises(RuntimeError, match="injected batch failure"):
        CrossMatch().crossmatch(
            p1,
            p2,
            output_file=output,
            radius_arcsec=1.0,
            join_type=join_type,
            engine="fast",
            memory_budget_bytes=4096,
            scratch_dir=scratch,
        )
    assert output.read_bytes() == b"original"
    assert list(scratch.iterdir()) == []


def test_large_fits_is_rejected_before_the_eager_reader(tmp_path, monkeypatch):
    left, right = tmp_path / "left.fits", tmp_path / "right.parquet"
    left.write_bytes(b"not-a-fits-table" * 200)
    pl.DataFrame({"ra": [1.0], "dec": [2.0]}).write_parquet(right)

    def must_not_read(path):
        raise AssertionError("large FITS should be rejected before eager reading")

    monkeypatch.setattr("xmatcher.io_utils._read_fits", must_not_read)
    with pytest.raises(CrossMatchError, match="CSV/Parquet"):
        CrossMatch().crossmatch(
            left,
            right,
            output_file=tmp_path / "out.parquet",
            memory_budget_bytes=1024,
        )


def test_spill_requires_a_streaming_output(tmp_path):
    p1, p2 = _catalogues(tmp_path)
    with pytest.raises(CrossMatchError, match="requires output_file"):
        CrossMatch().crossmatch(
            p1,
            p2,
            radius_arcsec=1.0,
            engine="fast",
            memory_budget_bytes=4096,
        )


def test_cli_exposes_bounded_memory_controls():
    args = _build_match_subparser().parse_args(
        [
            "left.csv",
            "right.parquet",
            "--memory-budget-bytes",
            "8192",
            "--scratch-dir",
            "/tmp/xmatcher",
            "--partition-order",
            "4",
        ]
    )
    assert args.memory_budget_bytes == 8192
    assert args.scratch_dir == "/tmp/xmatcher"
    assert args.partition_order == "4"
