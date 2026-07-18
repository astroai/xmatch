from __future__ import annotations

import json
from importlib.resources import files

import pytest

from xmatch.association import (
    ASSOCIATION_COMPONENT_DELTA_RELEASE_SCHEMA_VERSION,
    AssociationComponent,
    AssociationComponentDeltaKind,
    AssociationComponentMember,
    AssociationDecision,
    AssociationRecord,
    classify_association_component_deltas,
    construct_association_components,
    iter_association_component_delta_release,
    load_association_component_delta_release_manifest,
    verify_association_component_delta_release,
    write_association_component_delta_release,
    write_association_component_release,
    write_association_release,
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


def _component_endpoints(tmp_path):
    parent_association_directory, parent_association = _association_release(
        tmp_path, "parent-association"
    )
    current_association_directory, current_association = _association_release(
        tmp_path,
        "current-association",
        parent_release_id=parent_association.release_id,
    )
    left_release, right_release = parent_association.provenance.input_release_ids

    def member(release: str, value: str) -> AssociationComponentMember:
        return AssociationComponentMember(release, value)

    parent_components = {
        "continued": AssociationComponent(
            parent_association.release_id,
            (member(left_release, "a"), member(right_release, "b")),
        ),
        "merge-left": AssociationComponent(
            parent_association.release_id,
            (member(left_release, "c"),),
        ),
        "merge-right": AssociationComponent(
            parent_association.release_id,
            (member(right_release, "d"),),
        ),
        "split": AssociationComponent(
            parent_association.release_id,
            (member(left_release, "e"), member(right_release, "f")),
        ),
        "retired": AssociationComponent(
            parent_association.release_id,
            (member(left_release, "g"),),
        ),
    }
    parent_component_directory = tmp_path / "parent-components"
    parent_component_manifest = write_association_component_release(
        sorted(parent_components.values(), key=lambda component: component.component_id),
        parent_association_directory,
        parent_component_directory,
    )
    current_components = {
        "continued": AssociationComponent(
            current_association.release_id,
            parent_components["continued"].members,
            (parent_components["continued"].component_id,),
        ),
        "merged": AssociationComponent(
            current_association.release_id,
            parent_components["merge-left"].members + parent_components["merge-right"].members,
            (
                parent_components["merge-left"].component_id,
                parent_components["merge-right"].component_id,
            ),
        ),
        "split-left": AssociationComponent(
            current_association.release_id,
            (parent_components["split"].members[0],),
            (parent_components["split"].component_id,),
        ),
        "split-right": AssociationComponent(
            current_association.release_id,
            (parent_components["split"].members[1],),
            (parent_components["split"].component_id,),
        ),
        "created": AssociationComponent(
            current_association.release_id,
            (member(right_release, "h"),),
        ),
    }
    current_component_directory = tmp_path / "current-components"
    current_component_manifest = write_association_component_release(
        sorted(current_components.values(), key=lambda component: component.component_id),
        current_association_directory,
        current_component_directory,
        parent_component_directory=parent_component_directory,
    )
    return (
        parent_components,
        current_components,
        parent_component_directory,
        current_component_directory,
        parent_component_manifest,
        current_component_manifest,
    )


def test_component_construction_requires_explicit_namespaces_and_policy(tmp_path) -> None:
    parent_directory, parent_manifest = _association_release(tmp_path, "parent")
    current_directory, current_manifest = _association_release(
        tmp_path,
        "current",
        parent_release_id=parent_manifest.release_id,
    )
    source_release, candidate_release = parent_manifest.provenance.input_release_ids
    decisions = (AssociationDecision.SELECTED, AssociationDecision.CANDIDATE)
    parent_components = construct_association_components(
        parent_directory,
        source_input_release_id=source_release,
        candidate_input_release_id=candidate_release,
        included_decisions=decisions,
    )
    parent_component_directory = tmp_path / "constructed-parent-components"
    write_association_component_release(
        parent_components,
        parent_directory,
        parent_component_directory,
    )

    current_components = construct_association_components(
        current_directory,
        source_input_release_id=source_release,
        candidate_input_release_id=candidate_release,
        included_decisions=reversed(decisions),
        parent_component_directory=parent_component_directory,
    )

    assert len(parent_components) == len(current_components) == 3
    assert sorted(len(component.members) for component in parent_components) == [2, 2, 3]
    assert {
        member.input_release_id for component in current_components for member in component.members
    } == {
        source_release,
        candidate_release,
    }
    assert {len(component.parent_component_ids) for component in current_components} == {1}
    assert current_components == sorted(current_components, key=lambda item: item.component_id)
    with pytest.raises(ValueError, match="absent from association provenance"):
        construct_association_components(
            current_directory,
            source_input_release_id="catalog:unknown:v1",
            candidate_input_release_id=candidate_release,
            included_decisions=decisions,
        )
    with pytest.raises(ValueError, match="AssociationDecision"):
        construct_association_components(
            current_directory,
            source_input_release_id=source_release,
            candidate_input_release_id=candidate_release,
            included_decisions=("selected",),
        )


def test_component_delta_classification_covers_all_topology_changes(tmp_path) -> None:
    (
        parent_components,
        current_components,
        parent_directory,
        current_directory,
        parent_manifest,
        current_manifest,
    ) = _component_endpoints(tmp_path)

    deltas = classify_association_component_deltas(parent_directory, current_directory)
    by_kind = {delta.kind: delta for delta in deltas}

    assert set(by_kind) == set(AssociationComponentDeltaKind)
    assert by_kind[AssociationComponentDeltaKind.CREATED].current_component_ids == (
        current_components["created"].component_id,
    )
    assert by_kind[AssociationComponentDeltaKind.CONTINUED].parent_component_ids == (
        parent_components["continued"].component_id,
    )
    assert set(by_kind[AssociationComponentDeltaKind.MERGED].parent_component_ids) == {
        parent_components["merge-left"].component_id,
        parent_components["merge-right"].component_id,
    }
    assert set(by_kind[AssociationComponentDeltaKind.SPLIT].current_component_ids) == {
        current_components["split-left"].component_id,
        current_components["split-right"].component_id,
    }
    assert by_kind[AssociationComponentDeltaKind.RETIRED].parent_component_ids == (
        parent_components["retired"].component_id,
    )
    assert {delta.parent_component_release_id for delta in deltas} == {
        parent_manifest.component_release_id
    }
    assert {delta.current_component_release_id for delta in deltas} == {
        current_manifest.component_release_id
    }


def test_component_delta_artifact_is_deterministic_and_endpoint_verified(tmp_path) -> None:
    *_, parent_directory, current_directory, parent_manifest, current_manifest = (
        _component_endpoints(tmp_path)
    )
    first = write_association_component_delta_release(
        parent_directory,
        current_directory,
        tmp_path / "deltas-first",
    )
    duplicate = write_association_component_delta_release(
        parent_directory,
        current_directory,
        tmp_path / "deltas-duplicate",
    )

    assert first.delta_release_id.startswith(
        f"{ASSOCIATION_COMPONENT_DELTA_RELEASE_SCHEMA_VERSION}:"
    )
    assert first.delta_release_id == duplicate.delta_release_id
    assert first.deltas_sha256 == duplicate.deltas_sha256
    assert first.delta_count == 5
    assert first.parent_association_release_id == parent_manifest.association_release_id
    assert first.current_association_release_id == current_manifest.association_release_id
    assert load_association_component_delta_release_manifest(tmp_path / "deltas-first") == first
    assert list(iter_association_component_delta_release(tmp_path / "deltas-first")) == (
        classify_association_component_deltas(parent_directory, current_directory)
    )
    assert (
        verify_association_component_delta_release(
            tmp_path / "deltas-first",
            parent_component_directory=parent_directory,
            current_component_directory=current_directory,
        )
        == first
    )


def test_component_delta_rejects_false_declared_lineage(tmp_path) -> None:
    (
        parent_components,
        _,
        parent_directory,
        _,
        _,
        _,
    ) = _component_endpoints(tmp_path)
    current_association_directory = tmp_path / "current-association"
    current_association_id = json.loads(
        (current_association_directory / "manifest.json").read_text(encoding="utf-8")
    )["release_id"]
    false_component = AssociationComponent(
        current_association_id,
        parent_components["continued"].members,
        (parent_components["merge-left"].component_id,),
    )
    false_directory = tmp_path / "false-current-components"
    write_association_component_release(
        [false_component],
        current_association_directory,
        false_directory,
        parent_component_directory=parent_directory,
    )

    target = tmp_path / "false-deltas"
    with pytest.raises(ValueError, match="exact parent membership overlap"):
        write_association_component_delta_release(parent_directory, false_directory, target)
    assert not target.exists()


def test_component_delta_verifier_rejects_row_and_manifest_tampering(tmp_path) -> None:
    *_, parent_directory, current_directory, _, _ = _component_endpoints(tmp_path)
    target = tmp_path / "deltas"
    write_association_component_delta_release(parent_directory, current_directory, target)
    deltas_path = target / "deltas.jsonl"
    lines = deltas_path.read_text(encoding="utf-8").splitlines()
    payload = json.loads(lines[0])
    payload["delta_id"] = payload["delta_id"][:-1] + (
        "0" if payload["delta_id"][-1] != "0" else "1"
    )
    lines[0] = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    deltas_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="delta_id does not match"):
        verify_association_component_delta_release(target)

    byte_tampered = tmp_path / "byte-tampered-deltas"
    write_association_component_delta_release(
        parent_directory,
        current_directory,
        byte_tampered,
    )
    byte_path = byte_tampered / "deltas.jsonl"
    byte_lines = byte_path.read_text(encoding="utf-8").splitlines()
    byte_lines[0] = json.dumps(json.loads(byte_lines[0]))
    byte_path.write_text("\n".join(byte_lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="deltas_sha256"):
        verify_association_component_delta_release(byte_tampered)

    clean = tmp_path / "clean-deltas"
    write_association_component_delta_release(parent_directory, current_directory, clean)
    manifest_path = clean / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["delta_count"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="delta_release_id does not match"):
        verify_association_component_delta_release(clean)
