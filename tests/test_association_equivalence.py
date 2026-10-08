from __future__ import annotations

import json
from dataclasses import replace
from importlib.resources import files

import pytest

from xmatcher.association import (
    ASSOCIATION_MEMBER_EQUIVALENCE_RELEASE_SCHEMA_VERSION,
    AssociationComponentMember,
    AssociationDecision,
    AssociationMemberEquivalence,
    AssociationProvenance,
    AssociationRecord,
    classify_association_component_deltas,
    construct_association_components,
    iter_association_member_equivalence_release,
    load_association_member_equivalence_release_manifest,
    verify_association_component_delta_release,
    verify_association_member_equivalence_release,
    write_association_component_delta_release,
    write_association_component_release,
    write_association_member_equivalence_release,
    write_association_release,
)


def _records() -> list[AssociationRecord]:
    payload = json.loads(
        files("xmatcher").joinpath("fixtures/association_v1.json").read_text(encoding="utf-8")
    )
    return sorted(
        (AssociationRecord.from_mapping(item) for item in payload["records"]),
        key=lambda record: record.association_id,
    )


def _renamed_endpoints(tmp_path):
    parent_records = _records()
    parent_directory = tmp_path / "parent-association"
    parent = write_association_release(
        parent_records,
        parent_directory,
        provenance=parent_records[0].provenance,
    )
    parent_source_release, parent_candidate_release = parent.provenance.input_release_ids
    current_source_release = f"{parent_source_release}:renamed"
    current_candidate_release = f"{parent_candidate_release}:renamed"
    current_provenance = AssociationProvenance(
        (current_source_release, current_candidate_release),
        parent.provenance.software_version,
        parent.provenance.parameters_sha256,
    )
    record_pairs = [
        (
            record,
            replace(
                record,
                source_id=f"renamed:{record.source_id}",
                candidate_id=f"renamed:{record.candidate_id}",
                provenance=current_provenance,
            ),
        )
        for record in parent_records
    ]
    current_records = sorted(
        (current_record for _, current_record in record_pairs),
        key=lambda record: record.association_id,
    )
    current_directory = tmp_path / "current-association"
    current = write_association_release(
        current_records,
        current_directory,
        provenance=current_provenance,
        parent_release_id=parent.release_id,
    )
    pairs = {
        (
            AssociationComponentMember(parent_source_release, parent_record.source_id),
            AssociationComponentMember(current_source_release, current_record.source_id),
        )
        for parent_record, current_record in record_pairs
    }
    pairs.update(
        {
            (
                AssociationComponentMember(parent_candidate_release, parent_record.candidate_id),
                AssociationComponentMember(current_candidate_release, current_record.candidate_id),
            )
            for parent_record, current_record in record_pairs
        }
    )
    equivalences = sorted(
        (
            AssociationMemberEquivalence(
                parent.release_id,
                current.release_id,
                parent_member,
                current_member,
            )
            for parent_member, current_member in pairs
        ),
        key=lambda item: item.equivalence_id,
    )
    return (
        parent_records,
        current_records,
        parent_directory,
        current_directory,
        parent,
        current,
        equivalences,
    )


def test_member_equivalence_release_is_deterministic_bijective_and_endpoint_bound(
    tmp_path,
) -> None:
    *_, parent_directory, current_directory, parent, current, equivalences = _renamed_endpoints(
        tmp_path
    )
    first = write_association_member_equivalence_release(
        equivalences,
        parent_directory,
        current_directory,
        tmp_path / "equivalences-first",
    )
    duplicate = write_association_member_equivalence_release(
        equivalences,
        parent_directory,
        current_directory,
        tmp_path / "equivalences-duplicate",
    )

    assert first.equivalence_release_id.startswith(
        f"{ASSOCIATION_MEMBER_EQUIVALENCE_RELEASE_SCHEMA_VERSION}:"
    )
    assert first.equivalence_release_id == duplicate.equivalence_release_id
    assert first.equivalences_sha256 == duplicate.equivalences_sha256
    assert first.equivalence_count == len(equivalences)
    assert first.parent_association_release_id == parent.release_id
    assert first.current_association_release_id == current.release_id
    assert (
        load_association_member_equivalence_release_manifest(tmp_path / "equivalences-first")
        == first
    )
    assert list(iter_association_member_equivalence_release(tmp_path / "equivalences-first")) == (
        equivalences
    )
    assert (
        verify_association_member_equivalence_release(
            tmp_path / "equivalences-first",
            parent_association_release_directory=parent_directory,
            current_association_release_directory=current_directory,
        )
        == first
    )


def test_namespace_equivalence_preserves_component_continuity(tmp_path) -> None:
    (
        _,
        _,
        parent_association_directory,
        current_association_directory,
        parent_association,
        current_association,
        equivalences,
    ) = _renamed_endpoints(tmp_path)
    parent_source, parent_candidate = parent_association.provenance.input_release_ids
    current_source, current_candidate = current_association.provenance.input_release_ids
    decisions = tuple(AssociationDecision)
    parent_components = construct_association_components(
        parent_association_directory,
        source_input_release_id=parent_source,
        candidate_input_release_id=parent_candidate,
        included_decisions=decisions,
    )
    parent_component_directory = tmp_path / "parent-components"
    write_association_component_release(
        parent_components,
        parent_association_directory,
        parent_component_directory,
    )
    equivalence_directory = tmp_path / "equivalences"
    write_association_member_equivalence_release(
        equivalences,
        parent_association_directory,
        current_association_directory,
        equivalence_directory,
    )
    current_components = construct_association_components(
        current_association_directory,
        source_input_release_id=current_source,
        candidate_input_release_id=current_candidate,
        included_decisions=decisions,
        parent_component_directory=parent_component_directory,
        member_equivalence_directory=equivalence_directory,
    )
    assert len(current_components) == len(parent_components)
    assert {len(component.parent_component_ids) for component in current_components} == {1}

    current_component_directory = tmp_path / "current-components"
    write_association_component_release(
        current_components,
        current_association_directory,
        current_component_directory,
        parent_component_directory=parent_component_directory,
    )
    with pytest.raises(ValueError, match="exact parent membership overlap"):
        classify_association_component_deltas(
            parent_component_directory,
            current_component_directory,
        )
    deltas = classify_association_component_deltas(
        parent_component_directory,
        current_component_directory,
        member_equivalence_directory=equivalence_directory,
    )
    assert len(deltas) == len(parent_components)
    assert {delta.kind.value for delta in deltas} == {"continued"}

    delta_directory = tmp_path / "deltas"
    manifest = write_association_component_delta_release(
        parent_component_directory,
        current_component_directory,
        delta_directory,
        member_equivalence_directory=equivalence_directory,
    )
    assert (
        verify_association_component_delta_release(
            delta_directory,
            parent_component_directory=parent_component_directory,
            current_component_directory=current_component_directory,
            member_equivalence_directory=equivalence_directory,
        )
        == manifest
    )


def test_member_equivalence_rejects_guesses_and_non_bijective_crosswalks(tmp_path) -> None:
    *_, parent_directory, current_directory, parent, current, equivalences = _renamed_endpoints(
        tmp_path
    )
    with pytest.raises(ValueError, match="namespace change"):
        AssociationMemberEquivalence(
            parent.release_id,
            current.release_id,
            equivalences[0].parent_member,
            AssociationComponentMember(
                equivalences[0].parent_member.input_release_id,
                equivalences[0].current_member.member_id,
            ),
        )

    duplicate_parent = AssociationMemberEquivalence(
        parent.release_id,
        current.release_id,
        equivalences[0].parent_member,
        AssociationComponentMember(
            equivalences[0].current_member.input_release_id,
            f"{equivalences[0].current_member.member_id}:other",
        ),
    )
    invalid = sorted([equivalences[0], duplicate_parent], key=lambda item: item.equivalence_id)
    target = tmp_path / "non-bijective"
    with pytest.raises(ValueError, match="parent member cannot have multiple"):
        write_association_member_equivalence_release(
            invalid,
            parent_directory,
            current_directory,
            target,
        )
    assert not target.exists()
    assert list(tmp_path.glob(".non-bijective.tmp-*")) == []


def test_member_equivalence_requires_direct_association_lineage(tmp_path) -> None:
    (
        _,
        current_records,
        parent_directory,
        _,
        _,
        _,
        equivalences,
    ) = _renamed_endpoints(tmp_path)
    unrelated_directory = tmp_path / "unrelated-association"
    unrelated = write_association_release(
        current_records,
        unrelated_directory,
        provenance=current_records[0].provenance,
    )

    assert unrelated.release_id != equivalences[0].current_association_release_id
    with pytest.raises(ValueError, match="direct parent/current lineage"):
        write_association_member_equivalence_release(
            equivalences,
            parent_directory,
            unrelated_directory,
            tmp_path / "invalid-lineage",
        )


def test_member_equivalence_verifier_rejects_tampering(tmp_path) -> None:
    *_, parent_directory, current_directory, _, _, equivalences = _renamed_endpoints(tmp_path)
    target = tmp_path / "equivalences"
    write_association_member_equivalence_release(
        equivalences,
        parent_directory,
        current_directory,
        target,
    )
    path = target / "equivalences.jsonl"
    rows = path.read_text(encoding="utf-8").splitlines()
    payload = json.loads(rows[0])
    payload["current_member"]["member_id"] += ":tampered"
    rows[0] = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="equivalence_id does not match"):
        verify_association_member_equivalence_release(target)
