"""Sparse candidate releases conserve sources and explicit alternatives."""

import math

import polars as pl
import pytest

from xmatcher.candidates import (
    normalize_candidate_hypotheses,
    verify_candidate_release,
    write_candidate_release,
)
from xmatcher.sources import CatalogueSource


def _source(name, ra, dec, *, ids=None, **metadata):
    return CatalogueSource(
        name=name,
        is_local=True,
        id_column="id",
        ra_column="ra",
        dec_column="dec",
        release_namespace=f"pilot/{name}/v1",
        _frame=pl.DataFrame(
            {"id": ids if ids is not None else list(range(len(ra))), "ra": ra, "dec": dec}
        ).lazy(),
        **metadata,
    )


def test_candidate_release_preserves_islands_and_all_competitors(tmp_path):
    # Two plausible counterparts, one source in each survey with no counterpart.
    sources = [
        _source("optical", [359.99999, 120.0], [0.0, 0.0], ids=[2**60, 2**60 + 1]),
        _source("infrared", [0.00001, 359.99998, 240.0], [0.0, 0.0, 0.0]),
        _source("uv", [35.0], [90.0]),
    ]
    manifest = write_candidate_release(
        tmp_path / "candidates",
        sources,
        radius_arcsec=1.0,
        memory_budget_bytes=1024,
        partition_order=3,
    )
    assert manifest["source_count"] == 6
    assert manifest["candidate_count"] == 2
    directory = tmp_path / "candidates"
    inventory = pl.read_parquet(directory / "sources.parquet")
    pairs = pl.read_parquet(directory / "candidates.parquet")
    assert inventory["source_id"].n_unique() == 6
    assert sorted([pairs["source_id"].n_unique(), pairs["candidate_id"].n_unique()]) == [1, 2]
    assert pairs["sep_arcsec"].sort().to_list() == pytest.approx([0.036, 0.072], abs=1e-7)
    assert inventory.filter(pl.col("candidate_count") == 0).height == 3
    assert verify_candidate_release(directory) == manifest
    with pytest.raises(FileExistsError):
        write_candidate_release(directory, sources, radius_arcsec=1.0)
    # Content checksums detect damaged publications.
    with (directory / "candidates.parquet").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        verify_candidate_release(directory)


def test_candidate_set_does_not_depend_on_partition_layout(tmp_path):
    sources = [
        _source("a", [0.0, 80.0, 190.0], [0.0, 89.99999, -89.99999]),
        _source("b", [359.99999, 200.0, 330.0], [0.0, 89.99999, -89.99999]),
    ]
    results = []
    for order in (1, 5):
        directory = tmp_path / str(order)
        write_candidate_release(
            directory, sources, radius_arcsec=1.0, memory_budget_bytes=2048, partition_order=order
        )
        results.append(pl.read_parquet(directory / "candidates.parquet"))
    assert results[0].equals(results[1])
    assert results[0].height == 3


def test_hypotheses_include_no_match_for_every_source_and_normalize_stably():
    sources = pl.DataFrame({"source_id": ["a", "b", "c"]})
    candidates = pl.DataFrame(
        {
            "source_id": ["a", "a", "b"],
            "candidate_id": ["x", "y", "z"],
            "log_weight": [math.log(2), 0.0, 1000.0],
        }
    )
    result = normalize_candidate_hypotheses(
        sources, candidates, candidate_namespace="targets/v1"
    ).collect()
    a = result.filter(pl.col("source_id") == "a")
    assert a.filter(pl.col("candidate_id") == "x")["probability"].item() == pytest.approx(0.5)
    assert a.filter(pl.col("candidate_id") == "y")["probability"].item() == pytest.approx(0.25)
    assert a.filter(pl.col("hypothesis_kind") == "no_match")["probability"].item() == pytest.approx(
        0.25
    )
    assert result.filter(pl.col("source_id") == "c")["probability"].item() == 1.0
    assert result.group_by("source_id").agg(pl.col("probability").sum())[
        "probability"
    ].to_list() == pytest.approx([1.0] * 3)
    assert result["score_semantics"].unique().to_list() == ["assumed_prior_posterior"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_hypotheses_fail_on_invalid_or_unregistered_evidence(bad):
    sources = pl.DataFrame({"source_id": ["a"]})
    with pytest.raises(ValueError, match="log weight"):
        normalize_candidate_hypotheses(
            sources,
            pl.DataFrame(
                {
                    "source_id": ["a"],
                    "candidate_id": ["b"],
                    "log_weight": [bad],
                }
            ),
            candidate_namespace="targets/v1",
        )
    with pytest.raises(ValueError, match="unknown source"):
        normalize_candidate_hypotheses(
            sources,
            pl.DataFrame(
                {
                    "source_id": ["unknown"],
                    "candidate_id": ["b"],
                    "log_weight": [0.0],
                }
            ),
            candidate_namespace="targets/v1",
        )


def test_candidate_failure_never_publishes_partial_release(tmp_path):
    invalid = _source("bad", [0.0], [91.0])
    with pytest.raises(ValueError, match="coordinate"):
        write_candidate_release(tmp_path / "release", [invalid, _source("b", [0.0], [0.0])])
    assert not (tmp_path / "release").exists()


def test_candidate_release_rejects_duplicate_native_identities(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        write_candidate_release(
            tmp_path / "release", [_source("duplicate", [0.0, 1.0], [0.0, 0.0], ids=[1, 1])]
        )
    assert not (tmp_path / "release").exists()


def test_candidate_release_rejects_unconverted_coordinate_frames(tmp_path):
    with pytest.raises(ValueError, match="ICRS"):
        write_candidate_release(
            tmp_path / "release", [_source("galactic", [0.0], [0.0], frame="galactic")]
        )
    assert not (tmp_path / "release").exists()


@pytest.mark.parametrize("parameter", ["radius_arcsec", "target_epoch"])
def test_candidate_release_rejects_boolean_scientific_parameters(tmp_path, parameter):
    kwargs = {parameter: True}
    with pytest.raises(ValueError, match="finite|radius"):
        write_candidate_release(
            tmp_path / "release",
            [_source("a", [0.0], [0.0]), _source("b", [1.0], [0.0])],
            **kwargs,
        )
    assert not (tmp_path / "release").exists()


def test_empty_hypotheses_preserve_explicit_empty_schema():
    sources = pl.DataFrame(schema={"source_id": pl.String})
    pairs = pl.DataFrame(
        schema={"source_id": pl.String, "candidate_id": pl.String, "log_weight": pl.Float64}
    )
    assert (
        normalize_candidate_hypotheses(sources, pairs, candidate_namespace="targets/v1")
        .collect()
        .height
        == 0
    )


def test_hypotheses_reject_boolean_no_match_weight():
    pairs = pl.DataFrame(
        schema={"source_id": pl.String, "candidate_id": pl.String, "log_weight": pl.Float64}
    )
    with pytest.raises(ValueError, match="no-match log weight"):
        normalize_candidate_hypotheses(
            pl.DataFrame({"source_id": ["a"]}),
            pairs,
            candidate_namespace="targets/v1",
            no_match_log_weight=True,
        )


def test_hypotheses_orient_pairs_and_do_not_compete_between_surveys():
    inventory = pl.DataFrame(
        {
            "source_id": ["opt", "ir", "uv"],
            "release_namespace": ["optical/v1", "infrared/v1", "uv/v1"],
        }
    )
    pairs = pl.DataFrame(
        {
            "source_id": ["ir", "opt", "ir"],
            "candidate_id": ["opt", "uv", "uv"],
            "source_namespace": ["infrared/v1", "optical/v1", "infrared/v1"],
            "candidate_namespace": ["optical/v1", "uv/v1", "uv/v1"],
            "log_weight": [0.0, 0.0, 0.0],
        }
    )
    result = normalize_candidate_hypotheses(
        inventory, pairs, candidate_namespace="infrared/v1"
    ).collect()
    assert result.height == 4
    assert set(result["source_id"]) == {"opt", "uv"}
    assert result.filter(pl.col("hypothesis_kind") == "counterpart")["candidate_id"].to_list() == [
        "ir",
        "ir",
    ]
    assert result["probability"].to_list() == [0.5] * 4
    assert result["candidate_namespace"].unique().to_list() == ["infrared/v1"]


def test_hypotheses_reject_pair_endpoint_in_wrong_inventory_namespace():
    inventory = pl.DataFrame(
        {
            "source_id": ["s", "c", "d"],
            "release_namespace": ["source/v1", "infrared/v1", "optical/v1"],
        }
    )
    pairs = pl.DataFrame(
        {
            "source_id": ["s"],
            "candidate_id": ["d"],
            "source_namespace": ["source/v1"],
            "candidate_namespace": ["infrared/v1"],
            "log_weight": [0.0],
        }
    )

    with pytest.raises(ValueError):
        normalize_candidate_hypotheses(inventory, pairs, candidate_namespace="infrared/v1")


def test_hypotheses_reject_unregistered_candidate_namespace():
    inventory = pl.DataFrame(
        {"source_id": ["s", "c"], "release_namespace": ["source/v1", "target/v1"]}
    )
    pairs = pl.DataFrame(
        {
            "source_id": ["s"],
            "candidate_id": ["c"],
            "source_namespace": ["source/v1"],
            "candidate_namespace": ["target/v1"],
            "log_weight": [0.0],
        }
    )

    with pytest.raises(ValueError):
        normalize_candidate_hypotheses(inventory, pairs, candidate_namespace="typo/v1")


def test_hypotheses_allow_unobserved_empty_target_namespace():
    inventory = pl.DataFrame({"source_id": ["s"], "release_namespace": ["source/v1"]})
    pairs = pl.DataFrame(
        schema={
            "source_id": pl.String,
            "candidate_id": pl.String,
            "source_namespace": pl.String,
            "candidate_namespace": pl.String,
            "log_weight": pl.Float64,
        }
    )

    result = normalize_candidate_hypotheses(
        inventory, pairs, candidate_namespace="empty-target/v1"
    ).collect()

    assert result.select("source_id", "candidate_id", "probability", "hypothesis_kind").rows() == [
        ("s", None, 1.0, "no_match")
    ]
    assert result["candidate_namespace"].unique().to_list() == ["empty-target/v1"]


def test_candidate_release_cleans_lock_if_temporary_directory_setup_fails(tmp_path, monkeypatch):
    output = tmp_path / "release"
    sources = [_source("one", [0.0], [0.0]), _source("two", [1.0], [1.0])]

    def fail_mkdtemp(*args, **kwargs):
        raise OSError("injected temporary-directory setup failure")

    with monkeypatch.context() as patcher:
        patcher.setattr("xmatcher.candidates.tempfile.mkdtemp", fail_mkdtemp)
        with pytest.raises(OSError, match="injected temporary-directory setup failure"):
            write_candidate_release(output, sources)

    assert not (tmp_path / ".release.lock").exists()
    assert write_candidate_release(output, sources)["source_count"] == 2


def test_frame_overrides_reach_typed_and_params_matching():
    from xmatcher import CrossMatch, MatchRequest, SideOverrides
    from xmatcher.exceptions import CrossMatchError

    frame = pl.DataFrame({"id": [1], "ra": [10.0], "dec": [0.0]})
    with pytest.raises(CrossMatchError, match="frame"):
        CrossMatch().crossmatch_request(
            MatchRequest(
                frame,
                frame,
                side1=SideOverrides(frame="galactic"),
                side2=SideOverrides(frame="icrs"),
                engine="fast",
            )
        )
    with pytest.raises(CrossMatchError, match="frame"):
        CrossMatch().crossmatch(frame, frame, frame_1="galactic", frame_2="icrs", engine="fast")


@pytest.mark.parametrize("motion", [{"target_epoch": 2016.0}, {"pm_prior": True}])
def test_distributed_union_rejects_unimplemented_motion(tmp_path, motion):
    from xmatcher import CrossMatch
    from xmatcher.exceptions import CrossMatchError

    frame = pl.DataFrame({"id": [1], "ra": [10.0], "dec": [0.0]})
    with pytest.raises(CrossMatchError, match="ray-union.*motion"):
        CrossMatch().union_match(
            [frame, frame],
            output_file=tmp_path / "union.hats",
            engine="ray-union",
            **motion,
        )
    assert not (tmp_path / "union.hats").exists()


def test_distributed_entry_point_rejects_mixed_frames(tmp_path):
    from xmatcher.exceptions import CrossMatchError
    from xmatcher.ray_union import ray_union_match

    with pytest.raises(CrossMatchError, match="frame"):
        ray_union_match(
            [
                _source("one", [0.0], [0.0]),
                _source("two", [0.0], [0.0], frame="galactic"),
            ],
            sep_arcsec=1.0,
            output_file=str(tmp_path / "union"),
        )
    assert not (tmp_path / "union").exists()


def test_candidate_validation_failure_does_not_publish(tmp_path, monkeypatch):
    def reject(_):
        raise ValueError("invalid proposed release")

    monkeypatch.setattr("xmatcher.candidates.verify_candidate_release", reject)
    with pytest.raises(ValueError, match="invalid proposed release"):
        write_candidate_release(tmp_path / "release", [_source("one", [0.0], [0.0])])
    assert not (tmp_path / "release").exists()
    assert not list(tmp_path.glob(".release*"))


def test_forced_spill_aligns_epochs_before_partitioning(tmp_path):
    from xmatcher import CrossMatch, MatchRequest, MatchSpec, SideOverrides

    # A high-motion star crosses an RA partition boundary and the seam.
    left = pl.DataFrame(
        {
            "id": [1],
            "ra": [359.99],
            "dec": [0.0],
            "pmra": [3600.0],
            "pmdec": [0.0],
            "pad": ["x" * 2000],
        }
    )
    right = pl.DataFrame({"id": [2], "ra": [0.01], "dec": [0.0], "pad": ["y" * 2000]})
    left.write_parquet(tmp_path / "left.parquet")
    right.write_parquet(tmp_path / "right.parquet")
    request = MatchRequest(
        tmp_path / "left.parquet",
        tmp_path / "right.parquet",
        spec=MatchSpec(radius_arcsec=0.01, target_epoch=2020.0, fallback_policy="error"),
        side1=SideOverrides(epoch=2000.0, pm_ra_column="pmra", pm_dec_column="pmdec"),
        side2=SideOverrides(epoch=2020.0),
        engine="fast",
        memory_budget_bytes=1024,
        partition_order=4,
        output_file=tmp_path / "matches.parquet",
    )
    CrossMatch().crossmatch_request(request)
    assert pl.read_parquet(request.output_file)["id_2"].to_list() == [2]


def test_target_epoch_spill_rejects_missing_motion_instead_of_assuming_stationary(tmp_path):
    from xmatcher import CrossMatch, MatchRequest, MatchSpec, SideOverrides
    from xmatcher.exceptions import CrossMatchError

    path = tmp_path / "input.parquet"
    pl.DataFrame({"id": [1], "ra": [10.0], "dec": [0.0], "pad": ["x" * 2000]}).write_parquet(path)
    with pytest.raises(CrossMatchError, match="proper motion"):
        CrossMatch().crossmatch_request(
            MatchRequest(
                path,
                path,
                spec=MatchSpec(target_epoch=2020.0, fallback_policy="error"),
                side1=SideOverrides(epoch=2000.0),
                side2=SideOverrides(epoch=2020.0),
                memory_budget_bytes=1024,
                output_file=tmp_path / "out.parquet",
                engine="fast",
            )
        )
