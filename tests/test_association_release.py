from __future__ import annotations

import json
from importlib.resources import files

import pytest

from xmatch.association import (
    ASSOCIATION_RELEASE_SCHEMA_VERSION,
    AssociationProvenance,
    AssociationRecord,
    iter_association_release,
    load_association_release_manifest,
    verify_association_release,
    write_association_release,
)


def _records() -> list[AssociationRecord]:
    payload = json.loads(
        files("xmatch").joinpath("fixtures/association_v1.json").read_text(encoding="utf-8")
    )
    records = [AssociationRecord.from_mapping(item) for item in payload["records"]]
    return sorted(records, key=lambda record: record.association_id)


def test_release_streams_round_trip_with_content_identity(tmp_path) -> None:
    records = _records()
    consumed: list[str] = []

    def stream():
        for record in records:
            consumed.append(record.association_id)
            yield record.to_dict()

    target = tmp_path / "release"
    manifest = write_association_release(
        stream(),
        target,
        provenance=records[0].provenance,
    )

    assert consumed == [record.association_id for record in records]
    assert manifest.release_id.startswith(f"{ASSOCIATION_RELEASE_SCHEMA_VERSION}:")
    assert manifest.record_count == len(records)
    assert load_association_release_manifest(target) == manifest
    assert list(iter_association_release(target)) == records
    assert verify_association_release(target) == manifest


def test_release_identity_is_stable_and_parent_is_part_of_identity(tmp_path) -> None:
    records = _records()
    provenance = records[0].provenance
    first = write_association_release(records, tmp_path / "first", provenance=provenance)
    duplicate = write_association_release(records, tmp_path / "duplicate", provenance=provenance)
    child = write_association_release(
        records,
        tmp_path / "child",
        provenance=provenance,
        parent_release_id=first.release_id,
    )

    assert duplicate.release_id == first.release_id
    assert duplicate.records_sha256 == first.records_sha256
    assert child.release_id != first.release_id
    assert child.parent_release_id == first.release_id


def test_release_rejects_malformed_parent_before_writing(tmp_path) -> None:
    records = _records()
    target = tmp_path / "release"

    with pytest.raises(ValueError, match="must identify"):
        write_association_release(
            records,
            target,
            provenance=records[0].provenance,
            parent_release_id=f"{ASSOCIATION_RELEASE_SCHEMA_VERSION}:not-a-digest",
        )

    assert not target.exists()


def test_release_rejects_unsorted_or_duplicate_records_without_partial_publish(tmp_path) -> None:
    records = _records()
    target = tmp_path / "bad"

    with pytest.raises(ValueError, match="strictly ordered"):
        write_association_release(
            [records[1], records[0]],
            target,
            provenance=records[0].provenance,
        )

    assert not target.exists()
    assert list(tmp_path.glob(".bad.tmp-*")) == []

    with pytest.raises(ValueError, match="strictly ordered"):
        write_association_release(
            [records[0], records[0]],
            target,
            provenance=records[0].provenance,
        )


def test_release_rejects_mixed_provenance(tmp_path) -> None:
    records = _records()
    other = AssociationProvenance(
        records[0].provenance.input_release_ids,
        "xmatch:other",
        records[0].provenance.parameters_sha256,
    )

    with pytest.raises(ValueError, match="provenance differs"):
        write_association_release(records, tmp_path / "release", provenance=other)


def test_release_verifier_rejects_byte_level_record_tampering(tmp_path) -> None:
    records = _records()
    target = tmp_path / "release"
    write_association_release(records, target, provenance=records[0].provenance)
    records_path = target / "associations.jsonl"
    lines = records_path.read_text(encoding="utf-8").splitlines()
    lines[0] = json.dumps(json.loads(lines[0]))
    records_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="records_sha256"):
        verify_association_release(target)


def test_release_verifier_rejects_manifest_tampering(tmp_path) -> None:
    records = _records()
    target = tmp_path / "release"
    write_association_release(records, target, provenance=records[0].provenance)
    manifest_path = target / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["record_count"] += 1
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="release_id does not match"):
        verify_association_release(target)


def test_release_never_overwrites_existing_path(tmp_path) -> None:
    records = _records()
    target = tmp_path / "release"
    target.mkdir()
    marker = target / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        write_association_release(records, target, provenance=records[0].provenance)

    assert marker.read_text(encoding="utf-8") == "keep"


def test_empty_release_is_valid_with_explicit_provenance(tmp_path) -> None:
    provenance = _records()[0].provenance
    target = tmp_path / "empty"

    manifest = write_association_release([], target, provenance=provenance)

    assert manifest.record_count == 0
    assert list(iter_association_release(target)) == []
    assert verify_association_release(target) == manifest
