from __future__ import annotations

import json
from importlib.resources import files

import pytest

from xmatch.association import (
    ASSOCIATION_COMPONENT_RELEASE_SCHEMA_VERSION,
    ASSOCIATION_COMPONENT_SCHEMA_VERSION,
    ASSOCIATION_RELEASE_SCHEMA_VERSION,
    AssociationComponent,
    AssociationComponentMember,
    AssociationRecord,
    iter_association_component_release,
    load_association_component_release_manifest,
    verify_association_component_release,
    write_association_component_release,
    write_association_release,
)


def test_component_construction_includes_sources_without_selected_edges(tmp_path) -> None:
    from xmatch.association import AssociationDecision, construct_association_components

    directory, manifest = _association_release(tmp_path, "association")
    left_release, right_release = _records()[0].provenance.input_release_ids
    island = AssociationComponentMember(right_release, "specz:source:isolated")
    components = construct_association_components(
        directory,
        source_input_release_id=left_release,
        candidate_input_release_id=right_release,
        included_decisions=[AssociationDecision.SELECTED],
        additional_members=[island],
    )
    assert any(component.members == (island,) for component in components)
    published = write_association_component_release(components, directory, tmp_path / "components")
    assert published.membership_count == sum(len(component.members) for component in components)
    with pytest.raises(ValueError, match="input release"):
        construct_association_components(
            directory,
            source_input_release_id=left_release,
            candidate_input_release_id=right_release,
            included_decisions=[],
            additional_members=[AssociationComponentMember("unregistered", "island")],
        )


def _records() -> list[AssociationRecord]:
    payload = json.loads(
        files("xmatch").joinpath("fixtures/association_v1.json").read_text(encoding="utf-8")
    )
    return sorted(
        (AssociationRecord.from_mapping(item) for item in payload["records"]),
        key=lambda record: record.association_id,
    )


def _association_release(tmp_path, name: str, *, parent_release_id: str | None = None):
    records = _records()
    directory = tmp_path / name
    manifest = write_association_release(
        records,
        directory,
        provenance=records[0].provenance,
        parent_release_id=parent_release_id,
    )
    return directory, manifest


def _components(association_release_id: str) -> list[AssociationComponent]:
    left_release, right_release = _records()[0].provenance.input_release_ids
    components = [
        AssociationComponent(
            association_release_id,
            (
                AssociationComponentMember(left_release, "cosmos2020:source:101"),
                AssociationComponentMember(right_release, "specz:source:9001"),
            ),
        ),
        AssociationComponent(
            association_release_id,
            (
                AssociationComponentMember(right_release, "specz:source:9003"),
                AssociationComponentMember(left_release, "cosmos2020:source:202"),
                AssociationComponentMember(right_release, "specz:source:9002"),
            ),
        ),
    ]
    return sorted(components, key=lambda component: component.component_id)


def test_component_release_round_trip_has_stable_content_identity(tmp_path) -> None:
    association_directory, association_manifest = _association_release(tmp_path, "association")
    components = _components(association_manifest.release_id)

    first = write_association_component_release(
        components,
        association_directory,
        tmp_path / "components-first",
    )
    duplicate = write_association_component_release(
        components,
        association_directory,
        tmp_path / "components-duplicate",
    )

    assert first.component_release_id.startswith(f"{ASSOCIATION_COMPONENT_RELEASE_SCHEMA_VERSION}:")
    assert first.component_release_id == duplicate.component_release_id
    assert first.memberships_sha256 == duplicate.memberships_sha256
    assert first.association_release_id == association_manifest.release_id
    assert first.provenance == association_manifest.provenance
    assert first.component_count == 2
    assert first.membership_count == 5
    assert load_association_component_release_manifest(tmp_path / "components-first") == first
    memberships = list(iter_association_component_release(tmp_path / "components-first"))
    assert [(row.component_id, row.input_release_id, row.member_id) for row in memberships] == [
        (component.component_id, member.input_release_id, member.member_id)
        for component in components
        for member in component.members
    ]
    assert (
        verify_association_component_release(
            tmp_path / "components-first",
            association_release_directory=association_directory,
        )
        == first
    )


def test_component_ids_are_membership_exact_and_association_release_scoped(tmp_path) -> None:
    parent_directory, parent_manifest = _association_release(tmp_path, "parent-association")
    child_directory, child_manifest = _association_release(
        tmp_path,
        "child-association",
        parent_release_id=parent_manifest.release_id,
    )

    parent_components = _components(parent_manifest.release_id)
    child_components = _components(child_manifest.release_id)

    assert parent_directory != child_directory
    assert parent_components[0].component_id.startswith(f"{ASSOCIATION_COMPONENT_SCHEMA_VERSION}:")
    assert {component.component_id for component in parent_components}.isdisjoint(
        component.component_id for component in child_components
    )
    changed_membership = AssociationComponent(
        parent_manifest.release_id,
        parent_components[0].members + (parent_components[1].members[0],),
    )
    assert changed_membership.component_id != parent_components[0].component_id


def test_component_release_records_merge_lineage_to_verified_parent(tmp_path) -> None:
    parent_association_directory, parent_association = _association_release(
        tmp_path, "parent-association"
    )
    parent_components = _components(parent_association.release_id)
    parent_component_directory = tmp_path / "parent-components"
    parent_component_manifest = write_association_component_release(
        parent_components,
        parent_association_directory,
        parent_component_directory,
    )
    child_association_directory, child_association = _association_release(
        tmp_path,
        "child-association",
        parent_release_id=parent_association.release_id,
    )
    merged = AssociationComponent(
        child_association.release_id,
        tuple(member for component in parent_components for member in component.members),
        tuple(component.component_id for component in parent_components),
    )
    child_component_directory = tmp_path / "child-components"

    child_manifest = write_association_component_release(
        [merged],
        child_association_directory,
        child_component_directory,
        parent_component_directory=parent_component_directory,
    )

    assert child_manifest.parent_component_release_id == (
        parent_component_manifest.component_release_id
    )
    memberships = list(iter_association_component_release(child_component_directory))
    assert {row.parent_component_ids for row in memberships} == {
        tuple(component.component_id for component in parent_components)
    }
    assert (
        verify_association_component_release(
            child_component_directory,
            association_release_directory=child_association_directory,
            parent_component_directory=parent_component_directory,
        )
        == child_manifest
    )


def test_component_release_rejects_dangling_or_mismatched_lineage(tmp_path) -> None:
    parent_association_directory, parent_association = _association_release(
        tmp_path, "parent-association"
    )
    parent_components = _components(parent_association.release_id)
    parent_component_directory = tmp_path / "parent-components"
    write_association_component_release(
        parent_components,
        parent_association_directory,
        parent_component_directory,
    )
    child_association_directory, child_association = _association_release(
        tmp_path,
        "child-association",
        parent_release_id=parent_association.release_id,
    )
    fake_component_id = f"{ASSOCIATION_COMPONENT_SCHEMA_VERSION}:{'f' * 64}"
    dangling = AssociationComponent(
        child_association.release_id,
        parent_components[0].members,
        (fake_component_id,),
    )

    with pytest.raises(ValueError, match="unknown parent"):
        write_association_component_release(
            [dangling],
            child_association_directory,
            tmp_path / "dangling",
            parent_component_directory=parent_component_directory,
        )

    wrong_parent_id = f"{ASSOCIATION_RELEASE_SCHEMA_VERSION}:{'e' * 64}"
    wrong_child_directory, wrong_child = _association_release(
        tmp_path,
        "wrong-child-association",
        parent_release_id=wrong_parent_id,
    )
    with pytest.raises(ValueError, match="association release lineage"):
        write_association_component_release(
            _components(wrong_child.release_id),
            wrong_child_directory,
            tmp_path / "mismatched",
            parent_component_directory=parent_component_directory,
        )

    assert not (tmp_path / "dangling").exists()
    assert not (tmp_path / "mismatched").exists()


def test_component_release_rejects_invalid_partition_without_partial_publish(tmp_path) -> None:
    association_directory, association_manifest = _association_release(tmp_path, "association")
    components = _components(association_manifest.release_id)
    left_release = components[0].members[0].input_release_id
    shared_member = components[0].members[0]
    overlapping = AssociationComponent(
        association_manifest.release_id,
        (shared_member, AssociationComponentMember(left_release, "cosmos2020:source:other")),
    )
    target = tmp_path / "overlapping"

    with pytest.raises(ValueError, match="multiple components"):
        write_association_component_release(
            sorted([components[0], overlapping], key=lambda component: component.component_id),
            association_directory,
            target,
        )

    assert not target.exists()
    assert list(tmp_path.glob(".overlapping.tmp-*")) == []

    unknown_release = AssociationComponent(
        association_manifest.release_id,
        (AssociationComponentMember("catalog:unknown:v1", "source:1"),),
    )
    with pytest.raises(ValueError, match="absent from provenance"):
        write_association_component_release(
            [unknown_release],
            association_directory,
            tmp_path / "unknown-release",
        )

    with pytest.raises(ValueError, match="strictly ordered"):
        write_association_component_release(
            list(reversed(components)),
            association_directory,
            tmp_path / "unsorted",
        )

    with pytest.raises(ValueError, match="parent component release is required"):
        lineaged = AssociationComponent(
            association_manifest.release_id,
            components[0].members,
            (components[0].component_id,),
        )
        write_association_component_release(
            [lineaged],
            association_directory,
            tmp_path / "missing-parent-release",
        )


def test_component_verifier_rejects_membership_and_manifest_tampering(tmp_path) -> None:
    association_directory, association_manifest = _association_release(tmp_path, "association")
    target = tmp_path / "components"
    write_association_component_release(
        _components(association_manifest.release_id),
        association_directory,
        target,
    )
    memberships_path = target / "memberships.jsonl"
    lines = memberships_path.read_text(encoding="utf-8").splitlines()
    payload = json.loads(lines[0])
    payload["member_id"] += ":tampered"
    lines[0] = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    memberships_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="component_id does not match"):
        verify_association_component_release(target)

    clean = tmp_path / "clean-components"
    write_association_component_release(
        _components(association_manifest.release_id),
        association_directory,
        clean,
    )
    manifest_path = clean / "manifest.json"
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_payload["membership_count"] += 1
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="component_release_id does not match"):
        verify_association_component_release(clean)


def test_empty_component_release_is_valid(tmp_path) -> None:
    association_directory, association_manifest = _association_release(tmp_path, "association")
    target = tmp_path / "empty-components"

    manifest = write_association_component_release([], association_directory, target)

    assert manifest.association_release_id == association_manifest.release_id
    assert manifest.component_count == 0
    assert manifest.membership_count == 0
    assert list(iter_association_component_release(target)) == []
    assert verify_association_component_release(target) == manifest
