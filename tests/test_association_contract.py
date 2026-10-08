from __future__ import annotations

import copy
import json
from importlib.resources import files

import pytest

from xmatcher.association import ASSOCIATION_SCHEMA_VERSION, AssociationRecord


def _fixture_records() -> list[dict]:
    payload = json.loads(
        files("xmatcher").joinpath("fixtures/association_v1.json").read_text(encoding="utf-8")
    )
    assert payload["schema_version"] == ASSOCIATION_SCHEMA_VERSION
    return payload["records"]


def test_packaged_association_v1_fixture_round_trips() -> None:
    records = [AssociationRecord.from_mapping(item) for item in _fixture_records()]

    assert [record.to_dict() for record in records] == _fixture_records()
    assert len({record.association_id for record in records}) == len(records)


def test_ambiguous_candidates_remain_distinct_records() -> None:
    records = [AssociationRecord.from_mapping(item) for item in _fixture_records()]
    ambiguous = [record for record in records if "ambiguous" in record.flags]

    assert len(ambiguous) == 2
    assert len({record.candidate_id for record in ambiguous}) == 2
    assert {record.decision.value for record in ambiguous} == {"candidate"}


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda item: item["score"].update(value=1.01), "must be in"),
        (
            lambda item: item["score"].update(semantics="calibrated_probability"),
            "calibration_id",
        ),
        (lambda item: item.update(separation_arcsec=float("nan")), "finite"),
        (lambda item: item.update(association_id="tampered"), "does not match"),
        (lambda item: item.update(evaluation_epoch_jyear=None), "epoch_unknown"),
        (lambda item: item.update(evaluation_epoch_jyear=2016.0), "does not match"),
    ],
)
def test_association_v1_rejects_adversarial_records(mutate, message: str) -> None:
    item = copy.deepcopy(_fixture_records()[0])
    mutate(item)

    with pytest.raises(ValueError, match=message):
        AssociationRecord.from_mapping(item)


def test_association_v1_preserves_extension_flags() -> None:
    item = copy.deepcopy(_fixture_records()[0])
    item["flags"].append("survey:blended")

    record = AssociationRecord.from_mapping(item)

    assert "survey:blended" in record.to_dict()["flags"]
