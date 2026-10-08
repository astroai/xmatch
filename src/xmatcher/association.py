"""Versioned candidate-association records and release artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

# Persisted v1 wire names stay fixed across the package rename to preserve IDs.
ASSOCIATION_SCHEMA_VERSION = "xmatch.association.v1"
ASSOCIATION_RELEASE_SCHEMA_VERSION = "xmatch.association.release.v1"
ASSOCIATION_COMPONENT_SCHEMA_VERSION = "xmatch.association.component.v1"
ASSOCIATION_COMPONENT_MEMBERSHIP_SCHEMA_VERSION = "xmatch.association.component.membership.v1"
ASSOCIATION_COMPONENT_RELEASE_SCHEMA_VERSION = "xmatch.association.component.release.v1"
ASSOCIATION_COMPONENT_DELTA_SCHEMA_VERSION = "xmatch.association.component.delta.v1"
ASSOCIATION_COMPONENT_DELTA_RELEASE_SCHEMA_VERSION = "xmatch.association.component.delta.release.v1"
ASSOCIATION_MEMBER_EQUIVALENCE_SCHEMA_VERSION = "xmatch.association.member.equivalence.v1"
ASSOCIATION_MEMBER_EQUIVALENCE_RELEASE_SCHEMA_VERSION = (
    "xmatch.association.member.equivalence.release.v1"
)

_RELEASE_MANIFEST = "manifest.json"
_RELEASE_RECORDS = "associations.jsonl"
_COMPONENT_MEMBERSHIPS = "memberships.jsonl"
_COMPONENT_DELTAS = "deltas.jsonl"
_MEMBER_EQUIVALENCES = "equivalences.jsonl"


class ScoreSemantics(StrEnum):
    """Scientific interpretation of an association score."""

    RANKING_SCORE = "ranking_score"
    ASSUMED_PRIOR_POSTERIOR = "assumed_prior_posterior"
    CALIBRATED_PROBABILITY = "calibrated_probability"


class AssociationDecision(StrEnum):
    """Disposition of a candidate without implying physical truth."""

    CANDIDATE = "candidate"
    SELECTED = "selected"
    REJECTED = "rejected"


class AssociationComponentDeltaKind(StrEnum):
    """Topology change between two component releases."""

    CREATED = "created"
    CONTINUED = "continued"
    MERGED = "merged"
    SPLIT = "split"
    RETIRED = "retired"


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty string without surrounding whitespace")
    return value


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _exact_keys(data: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(data)
    if actual != expected:
        raise ValueError(f"{name} fields must be exactly {sorted(expected)}; got {sorted(actual)}")


def _sha256(value: str, name: str) -> str:
    digest = _text(value, name)
    if not digest.startswith("sha256:") or len(digest) != 71:
        raise ValueError(f"{name} must be 'sha256:' followed by 64 hexadecimal digits")
    try:
        int(digest[7:], 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be 'sha256:' followed by 64 hexadecimal digits") from exc
    return digest


def _content_id(value: str, schema_version: str, name: str) -> str:
    identifier = _text(value, name)
    prefix = f"{schema_version}:"
    digest = identifier.removeprefix(prefix)
    if not identifier.startswith(prefix) or len(digest) != 64:
        raise ValueError(f"{name} must identify {schema_version}")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must identify {schema_version}") from exc
    return identifier


def _canonical_json(data: Mapping[str, Any]) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True)
class AssociationProvenance:
    """Inputs and executable configuration that produced a candidate record."""

    input_release_ids: tuple[str, ...]
    software_version: str
    parameters_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.input_release_ids, (tuple, list)):
            raise ValueError("input_release_ids must be a sequence")
        releases = tuple(
            sorted(_text(value, "input_release_ids item") for value in self.input_release_ids)
        )
        if len(releases) < 2 or len(set(releases)) != len(releases):
            raise ValueError("input_release_ids must contain at least two distinct releases")
        object.__setattr__(self, "input_release_ids", releases)
        _text(self.software_version, "software_version")
        _sha256(self.parameters_sha256, "parameters_sha256")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> AssociationProvenance:
        _exact_keys(
            data,
            {"input_release_ids", "software_version", "parameters_sha256"},
            "provenance",
        )
        releases = data["input_release_ids"]
        if not isinstance(releases, list):
            raise ValueError("input_release_ids must be a list")
        return cls(tuple(releases), data["software_version"], data["parameters_sha256"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_release_ids": list(self.input_release_ids),
            "software_version": self.software_version,
            "parameters_sha256": self.parameters_sha256,
        }


@dataclass(frozen=True)
class AssociationRecord:
    """One evaluated source/candidate pair under ``xmatch.association.v1``."""

    evidence_id: str
    source_id: str
    candidate_id: str
    method: str
    method_version: str
    separation_arcsec: float
    evaluation_epoch_jyear: float | None
    decision: AssociationDecision
    score_name: str
    score_value: float
    score_semantics: ScoreSemantics
    calibration_id: str | None
    flags: tuple[str, ...]
    provenance: AssociationProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.decision, AssociationDecision):
            raise ValueError("decision is not defined by association.v1")
        if not isinstance(self.score_semantics, ScoreSemantics):
            raise ValueError("score_semantics is not defined by association.v1")
        if not isinstance(self.provenance, AssociationProvenance):
            raise ValueError("provenance must be AssociationProvenance")
        if not isinstance(self.flags, (tuple, list)):
            raise ValueError("flags must be a sequence")
        for name in (
            "evidence_id",
            "source_id",
            "candidate_id",
            "method",
            "method_version",
            "score_name",
        ):
            _text(getattr(self, name), name)
        separation = _finite_number(self.separation_arcsec, "separation_arcsec")
        if separation < 0:
            raise ValueError("separation_arcsec must be non-negative")
        object.__setattr__(self, "separation_arcsec", separation)
        if self.evaluation_epoch_jyear is not None:
            epoch = _finite_number(self.evaluation_epoch_jyear, "evaluation_epoch_jyear")
            if epoch <= 0:
                raise ValueError("evaluation_epoch_jyear must be positive")
            object.__setattr__(self, "evaluation_epoch_jyear", epoch)
        score = _finite_number(self.score_value, "score_value")
        if self.score_semantics is not ScoreSemantics.RANKING_SCORE and not 0 <= score <= 1:
            raise ValueError(f"{self.score_semantics.value} score_value must be in [0, 1]")
        object.__setattr__(self, "score_value", score)
        if self.score_semantics is ScoreSemantics.CALIBRATED_PROBABILITY:
            _text(self.calibration_id, "calibration_id")
        elif self.calibration_id is not None:
            raise ValueError("calibration_id is only valid for calibrated_probability")
        flags = tuple(sorted(_text(value, "flags item") for value in self.flags))
        if len(set(flags)) != len(flags):
            raise ValueError("flags must not contain duplicates")
        if self.evaluation_epoch_jyear is None and "epoch_unknown" not in flags:
            raise ValueError("evaluation_epoch_jyear=None requires the epoch_unknown flag")
        object.__setattr__(self, "flags", flags)

    @property
    def association_id(self) -> str:
        identity = {
            "schema_version": ASSOCIATION_SCHEMA_VERSION,
            "evidence_id": self.evidence_id,
            "source_id": self.source_id,
            "candidate_id": self.candidate_id,
            "method": self.method,
            "method_version": self.method_version,
            "evaluation_epoch_jyear": self.evaluation_epoch_jyear,
            "input_release_ids": list(self.provenance.input_release_ids),
            "software_version": self.provenance.software_version,
            "parameters_sha256": self.provenance.parameters_sha256,
        }
        payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        return f"{ASSOCIATION_SCHEMA_VERSION}:{hashlib.sha256(payload).hexdigest()}"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> AssociationRecord:
        _exact_keys(
            data,
            {
                "schema_version",
                "association_id",
                "evidence_id",
                "source_id",
                "candidate_id",
                "method",
                "method_version",
                "separation_arcsec",
                "evaluation_epoch_jyear",
                "decision",
                "score",
                "flags",
                "provenance",
            },
            "association",
        )
        if data["schema_version"] != ASSOCIATION_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {ASSOCIATION_SCHEMA_VERSION!r}")
        score = data["score"]
        if not isinstance(score, Mapping):
            raise ValueError("score must be an object")
        _exact_keys(score, {"name", "value", "semantics", "calibration_id"}, "score")
        provenance = data["provenance"]
        if not isinstance(provenance, Mapping):
            raise ValueError("provenance must be an object")
        flags = data["flags"]
        if not isinstance(flags, list):
            raise ValueError("flags must be a list")
        try:
            decision = AssociationDecision(data["decision"])
            semantics = ScoreSemantics(score["semantics"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "decision or score semantics is not defined by association.v1"
            ) from exc
        record = cls(
            evidence_id=data["evidence_id"],
            source_id=data["source_id"],
            candidate_id=data["candidate_id"],
            method=data["method"],
            method_version=data["method_version"],
            separation_arcsec=data["separation_arcsec"],
            evaluation_epoch_jyear=data["evaluation_epoch_jyear"],
            decision=decision,
            score_name=score["name"],
            score_value=score["value"],
            score_semantics=semantics,
            calibration_id=score["calibration_id"],
            flags=tuple(flags),
            provenance=AssociationProvenance.from_mapping(provenance),
        )
        if data["association_id"] != record.association_id:
            raise ValueError("association_id does not match the record identity")
        return record

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ASSOCIATION_SCHEMA_VERSION,
            "association_id": self.association_id,
            "evidence_id": self.evidence_id,
            "source_id": self.source_id,
            "candidate_id": self.candidate_id,
            "method": self.method,
            "method_version": self.method_version,
            "separation_arcsec": self.separation_arcsec,
            "evaluation_epoch_jyear": self.evaluation_epoch_jyear,
            "decision": self.decision.value,
            "score": {
                "name": self.score_name,
                "value": self.score_value,
                "semantics": self.score_semantics.value,
                "calibration_id": self.calibration_id,
            },
            "flags": list(self.flags),
            "provenance": self.provenance.to_dict(),
        }


def _release_identity(
    *,
    record_count: int,
    records_sha256: str,
    parent_release_id: str | None,
    provenance: AssociationProvenance,
) -> dict[str, Any]:
    return {
        "schema_version": ASSOCIATION_RELEASE_SCHEMA_VERSION,
        "association_schema_version": ASSOCIATION_SCHEMA_VERSION,
        "record_count": record_count,
        "records_sha256": records_sha256,
        "parent_release_id": parent_release_id,
        "provenance": provenance.to_dict(),
    }


def _release_id(**identity: Any) -> str:
    digest = hashlib.sha256(_canonical_json(_release_identity(**identity))).hexdigest()
    return f"{ASSOCIATION_RELEASE_SCHEMA_VERSION}:{digest}"


@dataclass(frozen=True)
class AssociationReleaseManifest:
    """Content identity and shared provenance for an association release."""

    release_id: str
    record_count: int
    records_sha256: str
    parent_release_id: str | None
    provenance: AssociationProvenance
    association_schema_version: str = ASSOCIATION_SCHEMA_VERSION
    records_path: str = _RELEASE_RECORDS

    def __post_init__(self) -> None:
        if self.association_schema_version != ASSOCIATION_SCHEMA_VERSION:
            raise ValueError(f"association_schema_version must be {ASSOCIATION_SCHEMA_VERSION!r}")
        if self.records_path != _RELEASE_RECORDS:
            raise ValueError(f"records_path must be {_RELEASE_RECORDS!r}")
        if isinstance(self.record_count, bool) or not isinstance(self.record_count, int):
            raise ValueError("record_count must be a non-negative integer")
        if self.record_count < 0:
            raise ValueError("record_count must be a non-negative integer")
        _sha256(self.records_sha256, "records_sha256")
        if not isinstance(self.provenance, AssociationProvenance):
            raise ValueError("provenance must be AssociationProvenance")
        if self.parent_release_id is not None:
            _content_id(
                self.parent_release_id,
                ASSOCIATION_RELEASE_SCHEMA_VERSION,
                "parent_release_id",
            )
        expected = _release_id(
            record_count=self.record_count,
            records_sha256=self.records_sha256,
            parent_release_id=self.parent_release_id,
            provenance=self.provenance,
        )
        if self.release_id != expected:
            raise ValueError("release_id does not match the release identity")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> AssociationReleaseManifest:
        _exact_keys(
            data,
            {
                "schema_version",
                "release_id",
                "association_schema_version",
                "records_path",
                "record_count",
                "records_sha256",
                "parent_release_id",
                "provenance",
            },
            "association release manifest",
        )
        if data["schema_version"] != ASSOCIATION_RELEASE_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {ASSOCIATION_RELEASE_SCHEMA_VERSION!r}")
        provenance = data["provenance"]
        if not isinstance(provenance, Mapping):
            raise ValueError("provenance must be an object")
        return cls(
            release_id=data["release_id"],
            record_count=data["record_count"],
            records_sha256=data["records_sha256"],
            parent_release_id=data["parent_release_id"],
            provenance=AssociationProvenance.from_mapping(provenance),
            association_schema_version=data["association_schema_version"],
            records_path=data["records_path"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ASSOCIATION_RELEASE_SCHEMA_VERSION,
            "release_id": self.release_id,
            "association_schema_version": self.association_schema_version,
            "records_path": self.records_path,
            "record_count": self.record_count,
            "records_sha256": self.records_sha256,
            "parent_release_id": self.parent_release_id,
            "provenance": self.provenance.to_dict(),
        }


def load_association_release_manifest(
    directory: str | os.PathLike[str],
) -> AssociationReleaseManifest:
    """Load and validate a release manifest without reading its records."""
    path = Path(directory) / _RELEASE_MANIFEST
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read association release manifest: {path}") from exc
    if not isinstance(data, Mapping):
        raise ValueError("association release manifest must be an object")
    return AssociationReleaseManifest.from_mapping(data)


def _validated_release_records(
    directory: Path,
    manifest: AssociationReleaseManifest,
) -> Iterator[AssociationRecord]:
    path = directory / manifest.records_path
    digest = hashlib.sha256()
    previous_id: str | None = None
    count = 0

    try:
        handle = path.open("rb")
    except OSError as exc:
        raise ValueError(f"cannot read association release records: {path}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            digest.update(line)
            if not line.endswith(b"\n"):
                raise ValueError(f"association record line {line_number} is not newline-terminated")
            try:
                data = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"association record line {line_number} is not valid JSON"
                ) from exc
            if not isinstance(data, Mapping):
                raise ValueError(f"association record line {line_number} must be an object")
            record = AssociationRecord.from_mapping(data)
            if record.provenance != manifest.provenance:
                raise ValueError(
                    f"association record line {line_number} provenance differs from the manifest"
                )
            if previous_id is not None and record.association_id <= previous_id:
                raise ValueError("association records must be strictly ordered by association_id")
            previous_id = record.association_id
            count += 1
            yield record

    if count != manifest.record_count:
        raise ValueError(
            f"record_count mismatch: manifest has {manifest.record_count}, records have {count}"
        )
    actual_digest = f"sha256:{digest.hexdigest()}"
    if actual_digest != manifest.records_sha256:
        raise ValueError("records_sha256 does not match associations.jsonl")


def iter_association_release(
    directory: str | os.PathLike[str],
) -> Iterator[AssociationRecord]:
    """Stream validated records; checksum/count validation completes at EOF."""
    path = Path(directory)
    manifest = load_association_release_manifest(path)
    yield from _validated_release_records(path, manifest)


def verify_association_release(
    directory: str | os.PathLike[str],
) -> AssociationReleaseManifest:
    """Stream through every record and return the validated manifest."""
    path = Path(directory)
    manifest = load_association_release_manifest(path)
    for _ in _validated_release_records(path, manifest):
        pass
    return manifest


def write_association_release(
    records: Iterable[AssociationRecord | Mapping[str, Any]],
    directory: str | os.PathLike[str],
    *,
    provenance: AssociationProvenance,
    parent_release_id: str | None = None,
) -> AssociationReleaseManifest:
    """Atomically write a deterministic, content-addressed association release.

    ``records`` is consumed once and must be strictly ordered by
    :attr:`AssociationRecord.association_id`. This keeps publication bounded to
    one record plus the writer buffers; callers can use their table engine's
    external sort before publication.
    """
    if not isinstance(provenance, AssociationProvenance):
        raise ValueError("provenance must be AssociationProvenance")
    if parent_release_id is not None:
        _content_id(
            parent_release_id,
            ASSOCIATION_RELEASE_SCHEMA_VERSION,
            "parent_release_id",
        )

    target = Path(directory)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"association release path already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))

    try:
        digest = hashlib.sha256()
        count = 0
        previous_id: str | None = None
        records_path = temporary / _RELEASE_RECORDS
        with records_path.open("xb") as handle:
            for item in records:
                record = (
                    item
                    if isinstance(item, AssociationRecord)
                    else AssociationRecord.from_mapping(item)
                )
                if record.provenance != provenance:
                    raise ValueError(
                        "association record provenance differs from release provenance"
                    )
                if previous_id is not None and record.association_id <= previous_id:
                    raise ValueError(
                        "association records must be strictly ordered by association_id"
                    )
                previous_id = record.association_id
                line = _canonical_json(record.to_dict()) + b"\n"
                handle.write(line)
                digest.update(line)
                count += 1
            handle.flush()
            os.fsync(handle.fileno())

        records_sha256 = f"sha256:{digest.hexdigest()}"
        identity = {
            "record_count": count,
            "records_sha256": records_sha256,
            "parent_release_id": parent_release_id,
            "provenance": provenance,
        }
        manifest = AssociationReleaseManifest(
            release_id=_release_id(**identity),
            **identity,
        )
        manifest_path = temporary / _RELEASE_MANIFEST
        with manifest_path.open("xb") as handle:
            handle.write(_canonical_json(manifest.to_dict()) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())

        if target.exists() or target.is_symlink():
            raise FileExistsError(f"association release path already exists: {target}")
        # one local publisher owns a target path; concurrent object-store
        # publication should use the provider's conditional-create primitive.
        temporary.rename(target)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


@dataclass(frozen=True)
class AssociationComponentMember:
    """One catalogue object in a release-scoped association component."""

    input_release_id: str
    member_id: str

    def __post_init__(self) -> None:
        _text(self.input_release_id, "input_release_id")
        _text(self.member_id, "member_id")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> AssociationComponentMember:
        _exact_keys(data, {"input_release_id", "member_id"}, "association component member")
        return cls(data["input_release_id"], data["member_id"])

    def to_dict(self) -> dict[str, str]:
        return {
            "input_release_id": self.input_release_id,
            "member_id": self.member_id,
        }


def _member_equivalence_id(
    *,
    parent_association_release_id: str,
    current_association_release_id: str,
    parent_member: AssociationComponentMember,
    current_member: AssociationComponentMember,
) -> str:
    identity = {
        "schema_version": ASSOCIATION_MEMBER_EQUIVALENCE_SCHEMA_VERSION,
        "parent_association_release_id": parent_association_release_id,
        "current_association_release_id": current_association_release_id,
        "parent_member": parent_member.to_dict(),
        "current_member": current_member.to_dict(),
    }
    return (
        f"{ASSOCIATION_MEMBER_EQUIVALENCE_SCHEMA_VERSION}:"
        f"{hashlib.sha256(_canonical_json(identity)).hexdigest()}"
    )


@dataclass(frozen=True)
class AssociationMemberEquivalence:
    """A reviewed one-to-one member identity across input-release namespaces."""

    parent_association_release_id: str
    current_association_release_id: str
    parent_member: AssociationComponentMember
    current_member: AssociationComponentMember

    def __post_init__(self) -> None:
        _content_id(
            self.parent_association_release_id,
            ASSOCIATION_RELEASE_SCHEMA_VERSION,
            "parent_association_release_id",
        )
        _content_id(
            self.current_association_release_id,
            ASSOCIATION_RELEASE_SCHEMA_VERSION,
            "current_association_release_id",
        )
        if not isinstance(self.parent_member, AssociationComponentMember):
            raise ValueError("parent_member must be AssociationComponentMember")
        if not isinstance(self.current_member, AssociationComponentMember):
            raise ValueError("current_member must be AssociationComponentMember")
        if self.parent_member.input_release_id == self.current_member.input_release_id:
            raise ValueError("member equivalence requires an input-release namespace change")

    @property
    def equivalence_id(self) -> str:
        return _member_equivalence_id(
            parent_association_release_id=self.parent_association_release_id,
            current_association_release_id=self.current_association_release_id,
            parent_member=self.parent_member,
            current_member=self.current_member,
        )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> AssociationMemberEquivalence:
        _exact_keys(
            data,
            {
                "schema_version",
                "equivalence_id",
                "parent_association_release_id",
                "current_association_release_id",
                "parent_member",
                "current_member",
            },
            "association member equivalence",
        )
        if data["schema_version"] != ASSOCIATION_MEMBER_EQUIVALENCE_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {ASSOCIATION_MEMBER_EQUIVALENCE_SCHEMA_VERSION!r}"
            )
        if not isinstance(data["parent_member"], Mapping) or not isinstance(
            data["current_member"], Mapping
        ):
            raise ValueError("parent_member and current_member must be objects")
        equivalence = cls(
            parent_association_release_id=data["parent_association_release_id"],
            current_association_release_id=data["current_association_release_id"],
            parent_member=AssociationComponentMember.from_mapping(data["parent_member"]),
            current_member=AssociationComponentMember.from_mapping(data["current_member"]),
        )
        if data["equivalence_id"] != equivalence.equivalence_id:
            raise ValueError("equivalence_id does not match the member equivalence identity")
        return equivalence

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ASSOCIATION_MEMBER_EQUIVALENCE_SCHEMA_VERSION,
            "equivalence_id": self.equivalence_id,
            "parent_association_release_id": self.parent_association_release_id,
            "current_association_release_id": self.current_association_release_id,
            "parent_member": self.parent_member.to_dict(),
            "current_member": self.current_member.to_dict(),
        }


def _member_equivalence_release_identity(
    *,
    parent_association_release_id: str,
    current_association_release_id: str,
    equivalence_count: int,
    equivalences_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": ASSOCIATION_MEMBER_EQUIVALENCE_RELEASE_SCHEMA_VERSION,
        "equivalence_schema_version": ASSOCIATION_MEMBER_EQUIVALENCE_SCHEMA_VERSION,
        "parent_association_release_id": parent_association_release_id,
        "current_association_release_id": current_association_release_id,
        "equivalence_count": equivalence_count,
        "equivalences_sha256": equivalences_sha256,
    }


def _member_equivalence_release_id(**identity: Any) -> str:
    digest = hashlib.sha256(
        _canonical_json(_member_equivalence_release_identity(**identity))
    ).hexdigest()
    return f"{ASSOCIATION_MEMBER_EQUIVALENCE_RELEASE_SCHEMA_VERSION}:{digest}"


@dataclass(frozen=True)
class AssociationMemberEquivalenceReleaseManifest:
    """Content identity for a reviewed, bijective member crosswalk."""

    equivalence_release_id: str
    parent_association_release_id: str
    current_association_release_id: str
    equivalence_count: int
    equivalences_sha256: str
    equivalence_schema_version: str = ASSOCIATION_MEMBER_EQUIVALENCE_SCHEMA_VERSION
    equivalences_path: str = _MEMBER_EQUIVALENCES

    def __post_init__(self) -> None:
        if self.equivalence_schema_version != ASSOCIATION_MEMBER_EQUIVALENCE_SCHEMA_VERSION:
            raise ValueError(
                "equivalence_schema_version must be "
                f"{ASSOCIATION_MEMBER_EQUIVALENCE_SCHEMA_VERSION!r}"
            )
        if self.equivalences_path != _MEMBER_EQUIVALENCES:
            raise ValueError(f"equivalences_path must be {_MEMBER_EQUIVALENCES!r}")
        _content_id(
            self.parent_association_release_id,
            ASSOCIATION_RELEASE_SCHEMA_VERSION,
            "parent_association_release_id",
        )
        _content_id(
            self.current_association_release_id,
            ASSOCIATION_RELEASE_SCHEMA_VERSION,
            "current_association_release_id",
        )
        if (
            isinstance(self.equivalence_count, bool)
            or not isinstance(self.equivalence_count, int)
            or self.equivalence_count < 0
        ):
            raise ValueError("equivalence_count must be a non-negative integer")
        _sha256(self.equivalences_sha256, "equivalences_sha256")
        expected = _member_equivalence_release_id(
            parent_association_release_id=self.parent_association_release_id,
            current_association_release_id=self.current_association_release_id,
            equivalence_count=self.equivalence_count,
            equivalences_sha256=self.equivalences_sha256,
        )
        if self.equivalence_release_id != expected:
            raise ValueError(
                "equivalence_release_id does not match the member equivalence release identity"
            )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> AssociationMemberEquivalenceReleaseManifest:
        _exact_keys(
            data,
            {
                "schema_version",
                "equivalence_release_id",
                "equivalence_schema_version",
                "equivalences_path",
                "parent_association_release_id",
                "current_association_release_id",
                "equivalence_count",
                "equivalences_sha256",
            },
            "association member equivalence release manifest",
        )
        if data["schema_version"] != ASSOCIATION_MEMBER_EQUIVALENCE_RELEASE_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {ASSOCIATION_MEMBER_EQUIVALENCE_RELEASE_SCHEMA_VERSION!r}"
            )
        return cls(
            equivalence_release_id=data["equivalence_release_id"],
            parent_association_release_id=data["parent_association_release_id"],
            current_association_release_id=data["current_association_release_id"],
            equivalence_count=data["equivalence_count"],
            equivalences_sha256=data["equivalences_sha256"],
            equivalence_schema_version=data["equivalence_schema_version"],
            equivalences_path=data["equivalences_path"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ASSOCIATION_MEMBER_EQUIVALENCE_RELEASE_SCHEMA_VERSION,
            "equivalence_release_id": self.equivalence_release_id,
            "equivalence_schema_version": self.equivalence_schema_version,
            "equivalences_path": self.equivalences_path,
            "parent_association_release_id": self.parent_association_release_id,
            "current_association_release_id": self.current_association_release_id,
            "equivalence_count": self.equivalence_count,
            "equivalences_sha256": self.equivalences_sha256,
        }


def load_association_member_equivalence_release_manifest(
    directory: str | os.PathLike[str],
) -> AssociationMemberEquivalenceReleaseManifest:
    """Load an equivalence manifest without reading its crosswalk rows."""
    path = Path(directory) / _RELEASE_MANIFEST
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read association member equivalence manifest: {path}") from exc
    if not isinstance(data, Mapping):
        raise ValueError("association member equivalence release manifest must be an object")
    return AssociationMemberEquivalenceReleaseManifest.from_mapping(data)


def _validated_member_equivalences(
    directory: Path,
    manifest: AssociationMemberEquivalenceReleaseManifest,
) -> Iterator[AssociationMemberEquivalence]:
    path = directory / manifest.equivalences_path
    digest = hashlib.sha256()
    previous_id: str | None = None
    count = 0
    # bijection validation is O(rows) memory; use an external unique
    # index when crosswalk releases outgrow one host.
    parent_members: set[AssociationComponentMember] = set()
    current_members: set[AssociationComponentMember] = set()
    try:
        handle = path.open("rb")
    except OSError as exc:
        raise ValueError(f"cannot read association member equivalences: {path}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            digest.update(line)
            if not line.endswith(b"\n"):
                raise ValueError(
                    f"association member equivalence line {line_number} is not newline-terminated"
                )
            try:
                data = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"association member equivalence line {line_number} is not valid JSON"
                ) from exc
            if not isinstance(data, Mapping):
                raise ValueError(
                    f"association member equivalence line {line_number} must be an object"
                )
            equivalence = AssociationMemberEquivalence.from_mapping(data)
            if (
                equivalence.parent_association_release_id != manifest.parent_association_release_id
                or equivalence.current_association_release_id
                != manifest.current_association_release_id
            ):
                raise ValueError(
                    f"association member equivalence line {line_number} endpoint differs "
                    "from the manifest"
                )
            if previous_id is not None and equivalence.equivalence_id <= previous_id:
                raise ValueError(
                    "association member equivalences must be strictly ordered by equivalence_id"
                )
            if equivalence.parent_member in parent_members:
                raise ValueError("a parent member cannot have multiple current equivalents")
            if equivalence.current_member in current_members:
                raise ValueError("a current member cannot have multiple parent equivalents")
            previous_id = equivalence.equivalence_id
            parent_members.add(equivalence.parent_member)
            current_members.add(equivalence.current_member)
            count += 1
            yield equivalence
    if count != manifest.equivalence_count:
        raise ValueError(
            "equivalence_count mismatch: manifest has "
            f"{manifest.equivalence_count}, equivalences have {count}"
        )
    if f"sha256:{digest.hexdigest()}" != manifest.equivalences_sha256:
        raise ValueError("equivalences_sha256 does not match equivalences.jsonl")


def iter_association_member_equivalence_release(
    directory: str | os.PathLike[str],
) -> Iterator[AssociationMemberEquivalence]:
    """Stream equivalences; checksum, ordering, and bijection validate at EOF."""
    path = Path(directory)
    manifest = load_association_member_equivalence_release_manifest(path)
    yield from _validated_member_equivalences(path, manifest)


def verify_association_member_equivalence_release(
    directory: str | os.PathLike[str],
    *,
    parent_association_release_directory: str | os.PathLike[str] | None = None,
    current_association_release_directory: str | os.PathLike[str] | None = None,
) -> AssociationMemberEquivalenceReleaseManifest:
    """Stream-verify a member crosswalk and optional association endpoints.

    Bijection validation retains the two endpoint-member sets; endpoint checks
    otherwise consume the crosswalk one row at a time.
    """
    path = Path(directory)
    manifest = load_association_member_equivalence_release_manifest(path)
    parent_manifest: AssociationReleaseManifest | None = None
    current_manifest: AssociationReleaseManifest | None = None
    if parent_association_release_directory is not None:
        parent_manifest = verify_association_release(parent_association_release_directory)
        if manifest.parent_association_release_id != parent_manifest.release_id:
            raise ValueError("parent association release does not match the equivalence manifest")
    if current_association_release_directory is not None:
        current_manifest = verify_association_release(current_association_release_directory)
        if manifest.current_association_release_id != current_manifest.release_id:
            raise ValueError("current association release does not match the equivalence manifest")
    if (
        parent_manifest is not None
        and current_manifest is not None
        and current_manifest.parent_release_id != parent_manifest.release_id
    ):
        raise ValueError("association releases are not direct parent/current lineage")
    for equivalence in _validated_member_equivalences(path, manifest):
        if (
            parent_manifest is not None
            and equivalence.parent_member.input_release_id
            not in parent_manifest.provenance.input_release_ids
        ):
            raise ValueError("parent member input release is absent from parent provenance")
        if (
            current_manifest is not None
            and equivalence.current_member.input_release_id
            not in current_manifest.provenance.input_release_ids
        ):
            raise ValueError("current member input release is absent from current provenance")
    return manifest


def write_association_member_equivalence_release(
    equivalences: Iterable[AssociationMemberEquivalence],
    parent_association_release_directory: str | os.PathLike[str],
    current_association_release_directory: str | os.PathLike[str],
    directory: str | os.PathLike[str],
) -> AssociationMemberEquivalenceReleaseManifest:
    """Atomically publish a deterministic, bijective member crosswalk.

    Rows must be strictly ordered by ``equivalence_id``. Equivalence is an
    explicit reviewed assertion; this writer never infers it from bare IDs.
    """
    parent_manifest = verify_association_release(parent_association_release_directory)
    current_manifest = verify_association_release(current_association_release_directory)
    if current_manifest.parent_release_id != parent_manifest.release_id:
        raise ValueError("association releases are not direct parent/current lineage")

    target = Path(directory)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"association member equivalence path already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    try:
        digest = hashlib.sha256()
        count = 0
        previous_id: str | None = None
        parent_members: set[AssociationComponentMember] = set()
        current_members: set[AssociationComponentMember] = set()
        path = temporary / _MEMBER_EQUIVALENCES
        with path.open("xb") as handle:
            for equivalence in equivalences:
                if not isinstance(equivalence, AssociationMemberEquivalence):
                    raise ValueError(
                        "equivalences must contain AssociationMemberEquivalence values"
                    )
                if (
                    equivalence.parent_association_release_id != parent_manifest.release_id
                    or equivalence.current_association_release_id != current_manifest.release_id
                ):
                    raise ValueError("member equivalence differs from association endpoints")
                if (
                    equivalence.parent_member.input_release_id
                    not in parent_manifest.provenance.input_release_ids
                ):
                    raise ValueError("parent member input release is absent from parent provenance")
                if (
                    equivalence.current_member.input_release_id
                    not in current_manifest.provenance.input_release_ids
                ):
                    raise ValueError(
                        "current member input release is absent from current provenance"
                    )
                if previous_id is not None and equivalence.equivalence_id <= previous_id:
                    raise ValueError(
                        "association member equivalences must be strictly ordered by equivalence_id"
                    )
                if equivalence.parent_member in parent_members:
                    raise ValueError("a parent member cannot have multiple current equivalents")
                if equivalence.current_member in current_members:
                    raise ValueError("a current member cannot have multiple parent equivalents")
                previous_id = equivalence.equivalence_id
                parent_members.add(equivalence.parent_member)
                current_members.add(equivalence.current_member)
                line = _canonical_json(equivalence.to_dict()) + b"\n"
                handle.write(line)
                digest.update(line)
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        identity = {
            "parent_association_release_id": parent_manifest.release_id,
            "current_association_release_id": current_manifest.release_id,
            "equivalence_count": count,
            "equivalences_sha256": f"sha256:{digest.hexdigest()}",
        }
        manifest = AssociationMemberEquivalenceReleaseManifest(
            equivalence_release_id=_member_equivalence_release_id(**identity),
            **identity,
        )
        manifest_path = temporary / _RELEASE_MANIFEST
        with manifest_path.open("xb") as handle:
            handle.write(_canonical_json(manifest.to_dict()) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"association member equivalence path already exists: {target}")
        # one local publisher owns a target path; object stores need
        # their native conditional-create primitive for concurrent publication.
        temporary.rename(target)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _current_to_parent_equivalences(
    directory: str | os.PathLike[str],
    *,
    parent_association_release_id: str,
    current_association_release_id: str,
    parent_input_release_ids: tuple[str, ...],
    current_input_release_ids: tuple[str, ...],
) -> dict[AssociationComponentMember, AssociationComponentMember]:
    manifest = verify_association_member_equivalence_release(directory)
    if (
        manifest.parent_association_release_id != parent_association_release_id
        or manifest.current_association_release_id != current_association_release_id
    ):
        raise ValueError("member equivalence release does not match component endpoints")
    result: dict[AssociationComponentMember, AssociationComponentMember] = {}
    for equivalence in iter_association_member_equivalence_release(directory):
        if equivalence.parent_member.input_release_id not in parent_input_release_ids:
            raise ValueError("parent member input release is absent from parent provenance")
        if equivalence.current_member.input_release_id not in current_input_release_ids:
            raise ValueError("current member input release is absent from current provenance")
        result[equivalence.current_member] = equivalence.parent_member
    return result


def _component_id(
    association_release_id: str,
    *,
    member_count: int,
    members_sha256: str,
) -> str:
    identity = {
        "schema_version": ASSOCIATION_COMPONENT_SCHEMA_VERSION,
        "association_release_id": association_release_id,
        "member_count": member_count,
        "members_sha256": members_sha256,
    }
    digest = hashlib.sha256(_canonical_json(identity)).hexdigest()
    return f"{ASSOCIATION_COMPONENT_SCHEMA_VERSION}:{digest}"


def _component_member_identity(
    members: Iterable[AssociationComponentMember],
) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    for member in members:
        digest.update(_canonical_json(member.to_dict()) + b"\n")
        count += 1
    return count, f"sha256:{digest.hexdigest()}"


@dataclass(frozen=True)
class AssociationComponent:
    """Exact membership and optional ancestry for one release-scoped component."""

    association_release_id: str
    members: tuple[AssociationComponentMember, ...]
    parent_component_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _content_id(
            self.association_release_id,
            ASSOCIATION_RELEASE_SCHEMA_VERSION,
            "association_release_id",
        )
        if not isinstance(self.members, (tuple, list)):
            raise ValueError("members must be a sequence")
        if not all(isinstance(member, AssociationComponentMember) for member in self.members):
            raise ValueError("members must contain AssociationComponentMember values")
        members = tuple(
            sorted(self.members, key=lambda item: (item.input_release_id, item.member_id))
        )
        if not members:
            raise ValueError("an association component must contain at least one member")
        if len(set(members)) != len(members):
            raise ValueError("association component members must not contain duplicates")
        object.__setattr__(self, "members", members)
        if not isinstance(self.parent_component_ids, (tuple, list)):
            raise ValueError("parent_component_ids must be a sequence")
        parents = tuple(
            sorted(
                _content_id(value, ASSOCIATION_COMPONENT_SCHEMA_VERSION, "parent_component_id")
                for value in self.parent_component_ids
            )
        )
        if len(set(parents)) != len(parents):
            raise ValueError("parent_component_ids must not contain duplicates")
        object.__setattr__(self, "parent_component_ids", parents)

    @property
    def component_id(self) -> str:
        member_count, members_sha256 = _component_member_identity(self.members)
        return _component_id(
            self.association_release_id,
            member_count=member_count,
            members_sha256=members_sha256,
        )


@dataclass(frozen=True)
class AssociationComponentMembership:
    """One canonical row in a component membership release."""

    association_release_id: str
    component_id: str
    input_release_id: str
    member_id: str
    parent_component_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _content_id(
            self.association_release_id,
            ASSOCIATION_RELEASE_SCHEMA_VERSION,
            "association_release_id",
        )
        _content_id(self.component_id, ASSOCIATION_COMPONENT_SCHEMA_VERSION, "component_id")
        _text(self.input_release_id, "input_release_id")
        _text(self.member_id, "member_id")
        if not isinstance(self.parent_component_ids, (tuple, list)):
            raise ValueError("parent_component_ids must be a sequence")
        parents = tuple(
            sorted(
                _content_id(value, ASSOCIATION_COMPONENT_SCHEMA_VERSION, "parent_component_id")
                for value in self.parent_component_ids
            )
        )
        if len(set(parents)) != len(parents):
            raise ValueError("parent_component_ids must not contain duplicates")
        object.__setattr__(self, "parent_component_ids", parents)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> AssociationComponentMembership:
        _exact_keys(
            data,
            {
                "schema_version",
                "association_release_id",
                "component_id",
                "input_release_id",
                "member_id",
                "parent_component_ids",
            },
            "association component membership",
        )
        if data["schema_version"] != ASSOCIATION_COMPONENT_MEMBERSHIP_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {ASSOCIATION_COMPONENT_MEMBERSHIP_SCHEMA_VERSION!r}"
            )
        parents = data["parent_component_ids"]
        if not isinstance(parents, list):
            raise ValueError("parent_component_ids must be a list")
        return cls(
            association_release_id=data["association_release_id"],
            component_id=data["component_id"],
            input_release_id=data["input_release_id"],
            member_id=data["member_id"],
            parent_component_ids=tuple(parents),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ASSOCIATION_COMPONENT_MEMBERSHIP_SCHEMA_VERSION,
            "association_release_id": self.association_release_id,
            "component_id": self.component_id,
            "input_release_id": self.input_release_id,
            "member_id": self.member_id,
            "parent_component_ids": list(self.parent_component_ids),
        }


def _component_release_identity(
    *,
    association_release_id: str,
    component_count: int,
    membership_count: int,
    memberships_sha256: str,
    parent_component_release_id: str | None,
    provenance: AssociationProvenance,
) -> dict[str, Any]:
    return {
        "schema_version": ASSOCIATION_COMPONENT_RELEASE_SCHEMA_VERSION,
        "membership_schema_version": ASSOCIATION_COMPONENT_MEMBERSHIP_SCHEMA_VERSION,
        "association_release_id": association_release_id,
        "component_count": component_count,
        "membership_count": membership_count,
        "memberships_sha256": memberships_sha256,
        "parent_component_release_id": parent_component_release_id,
        "provenance": provenance.to_dict(),
    }


def _component_release_id(**identity: Any) -> str:
    digest = hashlib.sha256(_canonical_json(_component_release_identity(**identity))).hexdigest()
    return f"{ASSOCIATION_COMPONENT_RELEASE_SCHEMA_VERSION}:{digest}"


@dataclass(frozen=True)
class AssociationComponentReleaseManifest:
    """Content identity and provenance for component membership and lineage."""

    component_release_id: str
    association_release_id: str
    component_count: int
    membership_count: int
    memberships_sha256: str
    parent_component_release_id: str | None
    provenance: AssociationProvenance
    membership_schema_version: str = ASSOCIATION_COMPONENT_MEMBERSHIP_SCHEMA_VERSION
    memberships_path: str = _COMPONENT_MEMBERSHIPS

    def __post_init__(self) -> None:
        _content_id(
            self.association_release_id,
            ASSOCIATION_RELEASE_SCHEMA_VERSION,
            "association_release_id",
        )
        if self.membership_schema_version != ASSOCIATION_COMPONENT_MEMBERSHIP_SCHEMA_VERSION:
            raise ValueError(
                "membership_schema_version must be "
                f"{ASSOCIATION_COMPONENT_MEMBERSHIP_SCHEMA_VERSION!r}"
            )
        if self.memberships_path != _COMPONENT_MEMBERSHIPS:
            raise ValueError(f"memberships_path must be {_COMPONENT_MEMBERSHIPS!r}")
        for value, name in (
            (self.component_count, "component_count"),
            (self.membership_count, "membership_count"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.component_count > self.membership_count:
            raise ValueError("component_count must not exceed membership_count")
        _sha256(self.memberships_sha256, "memberships_sha256")
        if self.parent_component_release_id is not None:
            _content_id(
                self.parent_component_release_id,
                ASSOCIATION_COMPONENT_RELEASE_SCHEMA_VERSION,
                "parent_component_release_id",
            )
        if not isinstance(self.provenance, AssociationProvenance):
            raise ValueError("provenance must be AssociationProvenance")
        expected = _component_release_id(
            association_release_id=self.association_release_id,
            component_count=self.component_count,
            membership_count=self.membership_count,
            memberships_sha256=self.memberships_sha256,
            parent_component_release_id=self.parent_component_release_id,
            provenance=self.provenance,
        )
        if self.component_release_id != expected:
            raise ValueError("component_release_id does not match the component release identity")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> AssociationComponentReleaseManifest:
        _exact_keys(
            data,
            {
                "schema_version",
                "component_release_id",
                "association_release_id",
                "membership_schema_version",
                "memberships_path",
                "component_count",
                "membership_count",
                "memberships_sha256",
                "parent_component_release_id",
                "provenance",
            },
            "association component release manifest",
        )
        if data["schema_version"] != ASSOCIATION_COMPONENT_RELEASE_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {ASSOCIATION_COMPONENT_RELEASE_SCHEMA_VERSION!r}"
            )
        provenance = data["provenance"]
        if not isinstance(provenance, Mapping):
            raise ValueError("provenance must be an object")
        return cls(
            component_release_id=data["component_release_id"],
            association_release_id=data["association_release_id"],
            component_count=data["component_count"],
            membership_count=data["membership_count"],
            memberships_sha256=data["memberships_sha256"],
            parent_component_release_id=data["parent_component_release_id"],
            provenance=AssociationProvenance.from_mapping(provenance),
            membership_schema_version=data["membership_schema_version"],
            memberships_path=data["memberships_path"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ASSOCIATION_COMPONENT_RELEASE_SCHEMA_VERSION,
            "component_release_id": self.component_release_id,
            "association_release_id": self.association_release_id,
            "membership_schema_version": self.membership_schema_version,
            "memberships_path": self.memberships_path,
            "component_count": self.component_count,
            "membership_count": self.membership_count,
            "memberships_sha256": self.memberships_sha256,
            "parent_component_release_id": self.parent_component_release_id,
            "provenance": self.provenance.to_dict(),
        }


def load_association_component_release_manifest(
    directory: str | os.PathLike[str],
) -> AssociationComponentReleaseManifest:
    """Load a component release manifest without reading its membership rows."""
    path = Path(directory) / _RELEASE_MANIFEST
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read association component release manifest: {path}") from exc
    if not isinstance(data, Mapping):
        raise ValueError("association component release manifest must be an object")
    return AssociationComponentReleaseManifest.from_mapping(data)


def _finish_component(
    component_id: str,
    association_release_id: str,
    member_count: int,
    member_digest: Any,
) -> None:
    expected = _component_id(
        association_release_id,
        member_count=member_count,
        members_sha256=f"sha256:{member_digest.hexdigest()}",
    )
    if component_id != expected:
        raise ValueError("component_id does not match its exact release-scoped membership")


def _validated_component_memberships(
    directory: Path,
    manifest: AssociationComponentReleaseManifest,
    *,
    parent_component_ids: set[str] | None = None,
) -> Iterator[AssociationComponentMembership]:
    path = directory / manifest.memberships_path
    digest = hashlib.sha256()
    previous_key: tuple[str, str, str] | None = None
    current_component_id: str | None = None
    current_parent_ids: tuple[str, ...] = ()
    current_member_digest = hashlib.sha256()
    current_member_count = 0
    membership_count = 0
    component_count = 0
    # exact partition validation uses O(members) memory; replace this
    # set with an external unique index when one release outgrows local memory.
    seen_members: set[tuple[str, str]] = set()

    try:
        handle = path.open("rb")
    except OSError as exc:
        raise ValueError(f"cannot read association component memberships: {path}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            digest.update(line)
            if not line.endswith(b"\n"):
                raise ValueError(
                    f"association component membership line {line_number} is not newline-terminated"
                )
            try:
                data = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"association component membership line {line_number} is not valid JSON"
                ) from exc
            if not isinstance(data, Mapping):
                raise ValueError(
                    f"association component membership line {line_number} must be an object"
                )
            membership = AssociationComponentMembership.from_mapping(data)
            if membership.association_release_id != manifest.association_release_id:
                raise ValueError(
                    f"association component membership line {line_number} release differs "
                    "from the manifest"
                )
            if membership.input_release_id not in manifest.provenance.input_release_ids:
                raise ValueError(
                    f"association component membership line {line_number} input release "
                    "is absent from provenance"
                )
            key = (
                membership.component_id,
                membership.input_release_id,
                membership.member_id,
            )
            if previous_key is not None and key <= previous_key:
                raise ValueError(
                    "association component memberships must be strictly ordered by "
                    "component_id, input_release_id, member_id"
                )
            previous_key = key
            member_key = (membership.input_release_id, membership.member_id)
            if member_key in seen_members:
                raise ValueError("an association member cannot belong to multiple components")
            seen_members.add(member_key)

            if current_component_id != membership.component_id:
                if current_component_id is not None:
                    _finish_component(
                        current_component_id,
                        manifest.association_release_id,
                        current_member_count,
                        current_member_digest,
                    )
                current_component_id = membership.component_id
                current_parent_ids = membership.parent_component_ids
                current_member_digest = hashlib.sha256()
                current_member_count = 0
                component_count += 1
                missing = (
                    set(current_parent_ids).difference(parent_component_ids)
                    if parent_component_ids is not None
                    else set()
                )
                if missing:
                    raise ValueError("parent_component_ids contains an unknown parent component")
            elif membership.parent_component_ids != current_parent_ids:
                raise ValueError(
                    "parent_component_ids must be identical for every component member"
                )

            member = AssociationComponentMember(
                membership.input_release_id,
                membership.member_id,
            )
            current_member_digest.update(_canonical_json(member.to_dict()) + b"\n")
            current_member_count += 1
            membership_count += 1
            yield membership

    if current_component_id is not None:
        _finish_component(
            current_component_id,
            manifest.association_release_id,
            current_member_count,
            current_member_digest,
        )
    if component_count != manifest.component_count:
        raise ValueError(
            f"component_count mismatch: manifest has {manifest.component_count}, "
            f"memberships have {component_count}"
        )
    if membership_count != manifest.membership_count:
        raise ValueError(
            f"membership_count mismatch: manifest has {manifest.membership_count}, "
            f"memberships have {membership_count}"
        )
    actual_digest = f"sha256:{digest.hexdigest()}"
    if actual_digest != manifest.memberships_sha256:
        raise ValueError("memberships_sha256 does not match memberships.jsonl")


def _verified_parent_component_ids(
    directory: str | os.PathLike[str],
) -> tuple[AssociationComponentReleaseManifest, set[str]]:
    manifest = verify_association_component_release(directory)
    component_ids = {
        membership.component_id for membership in iter_association_component_release(directory)
    }
    return manifest, component_ids


def iter_association_component_release(
    directory: str | os.PathLike[str],
    *,
    parent_component_directory: str | os.PathLike[str] | None = None,
) -> Iterator[AssociationComponentMembership]:
    """Stream membership rows; checksum, counts, and component IDs validate at EOF."""
    path = Path(directory)
    manifest = load_association_component_release_manifest(path)
    parent_component_ids: set[str] | None = None
    if parent_component_directory is not None:
        parent_manifest, parent_component_ids = _verified_parent_component_ids(
            parent_component_directory
        )
        if manifest.parent_component_release_id != parent_manifest.component_release_id:
            raise ValueError("parent component release does not match the manifest")
    yield from _validated_component_memberships(
        path,
        manifest,
        parent_component_ids=parent_component_ids,
    )


def verify_association_component_release(
    directory: str | os.PathLike[str],
    *,
    association_release_directory: str | os.PathLike[str] | None = None,
    parent_component_directory: str | os.PathLike[str] | None = None,
) -> AssociationComponentReleaseManifest:
    """Verify a component release and its optional association/parent inputs."""
    path = Path(directory)
    manifest = load_association_component_release_manifest(path)
    association_manifest: AssociationReleaseManifest | None = None
    if association_release_directory is not None:
        association_manifest = verify_association_release(association_release_directory)
        if manifest.association_release_id != association_manifest.release_id:
            raise ValueError("association release does not match the component manifest")
        if manifest.provenance != association_manifest.provenance:
            raise ValueError("component provenance differs from the association release")

    parent_component_ids: set[str] | None = None
    if parent_component_directory is not None:
        parent_manifest, parent_component_ids = _verified_parent_component_ids(
            parent_component_directory
        )
        if manifest.parent_component_release_id != parent_manifest.component_release_id:
            raise ValueError("parent component release does not match the manifest")
        if (
            association_manifest is not None
            and association_manifest.parent_release_id != parent_manifest.association_release_id
        ):
            raise ValueError("parent component release does not match association release lineage")

    for _ in _validated_component_memberships(
        path,
        manifest,
        parent_component_ids=parent_component_ids,
    ):
        pass
    return manifest


def write_association_component_release(
    components: Iterable[AssociationComponent],
    association_release_directory: str | os.PathLike[str],
    directory: str | os.PathLike[str],
    *,
    parent_component_directory: str | os.PathLike[str] | None = None,
) -> AssociationComponentReleaseManifest:
    """Atomically publish deterministic component membership and lineage.

    Components must be strictly ordered by ``component_id``. Member order is
    canonicalized within each component. A parent component directory is
    required when any component declares lineage.
    """
    association_manifest = verify_association_release(association_release_directory)
    parent_component_release_id: str | None = None
    parent_component_ids: set[str] | None = None
    if parent_component_directory is not None:
        parent_manifest, parent_component_ids = _verified_parent_component_ids(
            parent_component_directory
        )
        if association_manifest.parent_release_id != parent_manifest.association_release_id:
            raise ValueError("parent component release does not match association release lineage")
        parent_component_release_id = parent_manifest.component_release_id

    target = Path(directory)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"association component release path already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))

    try:
        digest = hashlib.sha256()
        component_count = 0
        membership_count = 0
        previous_component_id: str | None = None
        # exact partition validation uses O(members) memory; replace
        # this set with an external unique index for survey-scale releases.
        seen_members: set[tuple[str, str]] = set()
        memberships_path = temporary / _COMPONENT_MEMBERSHIPS
        with memberships_path.open("xb") as handle:
            for component in components:
                if not isinstance(component, AssociationComponent):
                    raise ValueError("components must contain AssociationComponent values")
                if component.association_release_id != association_manifest.release_id:
                    raise ValueError("component association release differs from the input release")
                component_id = component.component_id
                if previous_component_id is not None and component_id <= previous_component_id:
                    raise ValueError(
                        "association components must be strictly ordered by component_id"
                    )
                previous_component_id = component_id
                missing_parents = set(component.parent_component_ids).difference(
                    parent_component_ids or ()
                )
                if component.parent_component_ids and parent_component_ids is None:
                    raise ValueError(
                        "parent component release is required when component lineage is present"
                    )
                if missing_parents:
                    raise ValueError("parent_component_ids contains an unknown parent component")

                for member in component.members:
                    if (
                        member.input_release_id
                        not in association_manifest.provenance.input_release_ids
                    ):
                        raise ValueError("component member input release is absent from provenance")
                    member_key = (member.input_release_id, member.member_id)
                    if member_key in seen_members:
                        raise ValueError(
                            "an association member cannot belong to multiple components"
                        )
                    seen_members.add(member_key)
                    membership = AssociationComponentMembership(
                        association_release_id=association_manifest.release_id,
                        component_id=component_id,
                        input_release_id=member.input_release_id,
                        member_id=member.member_id,
                        # repeating lineage keeps one canonical stream;
                        # a future schema can split it out if components become huge.
                        parent_component_ids=component.parent_component_ids,
                    )
                    line = _canonical_json(membership.to_dict()) + b"\n"
                    handle.write(line)
                    digest.update(line)
                    membership_count += 1
                component_count += 1
            handle.flush()
            os.fsync(handle.fileno())

        memberships_sha256 = f"sha256:{digest.hexdigest()}"
        identity = {
            "association_release_id": association_manifest.release_id,
            "component_count": component_count,
            "membership_count": membership_count,
            "memberships_sha256": memberships_sha256,
            "parent_component_release_id": parent_component_release_id,
            "provenance": association_manifest.provenance,
        }
        manifest = AssociationComponentReleaseManifest(
            component_release_id=_component_release_id(**identity),
            **identity,
        )
        manifest_path = temporary / _RELEASE_MANIFEST
        with manifest_path.open("xb") as handle:
            handle.write(_canonical_json(manifest.to_dict()) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())

        if target.exists() or target.is_symlink():
            raise FileExistsError(f"association component release path already exists: {target}")
        # one local publisher owns a target path; concurrent object-store
        # publication should use the provider's conditional-create primitive.
        temporary.rename(target)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _component_partition(
    directory: str | os.PathLike[str],
    *,
    parent_component_directory: str | os.PathLike[str] | None = None,
) -> tuple[
    AssociationComponentReleaseManifest,
    dict[str, tuple[AssociationComponentMember, ...]],
    dict[str, tuple[str, ...]],
]:
    manifest = verify_association_component_release(
        directory,
        parent_component_directory=parent_component_directory,
    )
    members: dict[str, list[AssociationComponentMember]] = {}
    parents: dict[str, tuple[str, ...]] = {}
    for membership in iter_association_component_release(directory):
        members.setdefault(membership.component_id, []).append(
            AssociationComponentMember(membership.input_release_id, membership.member_id)
        )
        parents[membership.component_id] = membership.parent_component_ids
    return manifest, {key: tuple(value) for key, value in members.items()}, parents


def construct_association_components(
    association_release_directory: str | os.PathLike[str],
    *,
    source_input_release_id: str,
    candidate_input_release_id: str,
    included_decisions: Iterable[AssociationDecision],
    additional_members: Iterable[AssociationComponentMember] = (),
    parent_component_directory: str | os.PathLike[str] | None = None,
    member_equivalence_directory: str | os.PathLike[str] | None = None,
) -> list[AssociationComponent]:
    """Build deterministic connected components from an association release.

    Endpoint release IDs and included decisions are required so construction
    never guesses catalogue namespaces or scientific acceptance policy. Supply
    inventory members to retain isolated sources and sources whose edges were
    excluded by that policy; connectivity alone does not assert object identity.
    """
    manifest = verify_association_release(association_release_directory)
    endpoint_releases = (
        _text(source_input_release_id, "source_input_release_id"),
        _text(candidate_input_release_id, "candidate_input_release_id"),
    )
    if not set(endpoint_releases).issubset(manifest.provenance.input_release_ids):
        raise ValueError("endpoint input release is absent from association provenance")
    try:
        decisions = set(included_decisions)
    except TypeError as exc:
        raise ValueError("included_decisions must be an iterable") from exc
    if not all(isinstance(decision, AssociationDecision) for decision in decisions):
        raise ValueError("included_decisions must contain AssociationDecision values")

    parent_by_member: dict[AssociationComponentMember, str] = {}
    current_to_parent: dict[AssociationComponentMember, AssociationComponentMember] = {}
    if member_equivalence_directory is not None and parent_component_directory is None:
        raise ValueError("member equivalence requires a parent component release")
    if parent_component_directory is not None:
        parent_manifest, parent_members, _ = _component_partition(parent_component_directory)
        if manifest.parent_release_id != parent_manifest.association_release_id:
            raise ValueError("parent component release does not match association release lineage")
        for component_id, members in parent_members.items():
            for member in members:
                parent_by_member[member] = component_id
        if member_equivalence_directory is not None:
            current_to_parent = _current_to_parent_equivalences(
                member_equivalence_directory,
                parent_association_release_id=parent_manifest.association_release_id,
                current_association_release_id=manifest.release_id,
                parent_input_release_ids=parent_manifest.provenance.input_release_ids,
                current_input_release_ids=manifest.provenance.input_release_ids,
            )

    # union-find retains O(nodes) state; survey-scale construction
    # should move this exact contract behind an external graph engine.
    roots: dict[AssociationComponentMember, AssociationComponentMember] = {}

    def find(member: AssociationComponentMember) -> AssociationComponentMember:
        root = roots.setdefault(member, member)
        while root != roots[root]:
            root = roots[root]
        while member != root:
            member, roots[member] = roots[member], root
        return root

    def union(left: AssociationComponentMember, right: AssociationComponentMember) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            left_key = (left_root.input_release_id, left_root.member_id)
            right_key = (right_root.input_release_id, right_root.member_id)
            if left_key < right_key:
                roots[right_root] = left_root
            else:
                roots[left_root] = right_root

    for member in additional_members:
        if not isinstance(member, AssociationComponentMember):
            raise ValueError("additional_members must contain AssociationComponentMember values")
        if member.input_release_id not in manifest.provenance.input_release_ids:
            raise ValueError(
                "additional member input release is absent from association provenance"
            )
        find(member)

    for record in iter_association_release(association_release_directory):
        if record.decision not in decisions:
            continue
        union(
            AssociationComponentMember(endpoint_releases[0], record.source_id),
            AssociationComponentMember(endpoint_releases[1], record.candidate_id),
        )

    grouped: dict[AssociationComponentMember, list[AssociationComponentMember]] = {}
    for member in roots:
        grouped.setdefault(find(member), []).append(member)

    components = [
        AssociationComponent(
            manifest.release_id,
            tuple(members),
            tuple(
                sorted(
                    {
                        parent_by_member[parent_member]
                        for member in members
                        if (parent_member := current_to_parent.get(member, member))
                        in parent_by_member
                    }
                )
            ),
        )
        for members in grouped.values()
    ]
    return sorted(components, key=lambda component: component.component_id)


def _component_delta_id(
    *,
    parent_association_release_id: str,
    parent_component_release_id: str,
    current_association_release_id: str,
    current_component_release_id: str,
    kind: AssociationComponentDeltaKind,
    parent_component_ids: tuple[str, ...],
    current_component_ids: tuple[str, ...],
) -> str:
    identity = {
        "schema_version": ASSOCIATION_COMPONENT_DELTA_SCHEMA_VERSION,
        "parent_association_release_id": parent_association_release_id,
        "parent_component_release_id": parent_component_release_id,
        "current_association_release_id": current_association_release_id,
        "current_component_release_id": current_component_release_id,
        "kind": kind.value,
        "parent_component_ids": list(parent_component_ids),
        "current_component_ids": list(current_component_ids),
    }
    return (
        f"{ASSOCIATION_COMPONENT_DELTA_SCHEMA_VERSION}:"
        f"{hashlib.sha256(_canonical_json(identity)).hexdigest()}"
    )


@dataclass(frozen=True)
class AssociationComponentDelta:
    """One deterministic topology change between release-scoped components."""

    parent_association_release_id: str
    parent_component_release_id: str
    current_association_release_id: str
    current_component_release_id: str
    kind: AssociationComponentDeltaKind
    parent_component_ids: tuple[str, ...]
    current_component_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for value, schema, name in (
            (
                self.parent_association_release_id,
                ASSOCIATION_RELEASE_SCHEMA_VERSION,
                "parent_association_release_id",
            ),
            (
                self.parent_component_release_id,
                ASSOCIATION_COMPONENT_RELEASE_SCHEMA_VERSION,
                "parent_component_release_id",
            ),
            (
                self.current_association_release_id,
                ASSOCIATION_RELEASE_SCHEMA_VERSION,
                "current_association_release_id",
            ),
            (
                self.current_component_release_id,
                ASSOCIATION_COMPONENT_RELEASE_SCHEMA_VERSION,
                "current_component_release_id",
            ),
        ):
            _content_id(value, schema, name)
        if not isinstance(self.kind, AssociationComponentDeltaKind):
            raise ValueError("kind is not defined by association component delta.v1")
        for name in ("parent_component_ids", "current_component_ids"):
            values = getattr(self, name)
            if not isinstance(values, (tuple, list)):
                raise ValueError(f"{name} must be a sequence")
            normalized = tuple(
                sorted(
                    _content_id(value, ASSOCIATION_COMPONENT_SCHEMA_VERSION, name)
                    for value in values
                )
            )
            if len(set(normalized)) != len(normalized):
                raise ValueError(f"{name} must not contain duplicates")
            object.__setattr__(self, name, normalized)
        counts = (len(self.parent_component_ids), len(self.current_component_ids))
        expected = {
            AssociationComponentDeltaKind.CREATED: (0, 1),
            AssociationComponentDeltaKind.CONTINUED: (1, 1),
            AssociationComponentDeltaKind.RETIRED: (1, 0),
        }
        if self.kind in expected and counts != expected[self.kind]:
            raise ValueError(f"{self.kind.value} component delta has invalid endpoint counts")
        if self.kind is AssociationComponentDeltaKind.MERGED and not (
            counts[0] >= 2 and counts[1] == 1
        ):
            raise ValueError("merged component delta has invalid endpoint counts")
        if self.kind is AssociationComponentDeltaKind.SPLIT and not (
            counts[0] == 1 and counts[1] >= 2
        ):
            raise ValueError("split component delta has invalid endpoint counts")

    @property
    def delta_id(self) -> str:
        return _component_delta_id(
            parent_association_release_id=self.parent_association_release_id,
            parent_component_release_id=self.parent_component_release_id,
            current_association_release_id=self.current_association_release_id,
            current_component_release_id=self.current_component_release_id,
            kind=self.kind,
            parent_component_ids=self.parent_component_ids,
            current_component_ids=self.current_component_ids,
        )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> AssociationComponentDelta:
        _exact_keys(
            data,
            {
                "schema_version",
                "delta_id",
                "parent_association_release_id",
                "parent_component_release_id",
                "current_association_release_id",
                "current_component_release_id",
                "kind",
                "parent_component_ids",
                "current_component_ids",
            },
            "association component delta",
        )
        if data["schema_version"] != ASSOCIATION_COMPONENT_DELTA_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {ASSOCIATION_COMPONENT_DELTA_SCHEMA_VERSION!r}"
            )
        for name in ("parent_component_ids", "current_component_ids"):
            if not isinstance(data[name], list):
                raise ValueError(f"{name} must be a list")
        try:
            kind = AssociationComponentDeltaKind(data["kind"])
        except (TypeError, ValueError) as exc:
            raise ValueError("kind is not defined by association component delta.v1") from exc
        delta = cls(
            parent_association_release_id=data["parent_association_release_id"],
            parent_component_release_id=data["parent_component_release_id"],
            current_association_release_id=data["current_association_release_id"],
            current_component_release_id=data["current_component_release_id"],
            kind=kind,
            parent_component_ids=tuple(data["parent_component_ids"]),
            current_component_ids=tuple(data["current_component_ids"]),
        )
        if data["delta_id"] != delta.delta_id:
            raise ValueError("delta_id does not match the component delta identity")
        return delta

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ASSOCIATION_COMPONENT_DELTA_SCHEMA_VERSION,
            "delta_id": self.delta_id,
            "parent_association_release_id": self.parent_association_release_id,
            "parent_component_release_id": self.parent_component_release_id,
            "current_association_release_id": self.current_association_release_id,
            "current_component_release_id": self.current_component_release_id,
            "kind": self.kind.value,
            "parent_component_ids": list(self.parent_component_ids),
            "current_component_ids": list(self.current_component_ids),
        }


def _classified_component_deltas(
    parent_component_directory: str | os.PathLike[str],
    current_component_directory: str | os.PathLike[str],
    *,
    member_equivalence_directory: str | os.PathLike[str] | None = None,
) -> tuple[
    AssociationComponentReleaseManifest,
    AssociationComponentReleaseManifest,
    list[AssociationComponentDelta],
]:
    parent_manifest, parent_components, _ = _component_partition(parent_component_directory)
    current_manifest, current_components, declared_parents = _component_partition(
        current_component_directory,
        parent_component_directory=parent_component_directory,
    )
    parent_by_member = {
        member: component_id
        for component_id, members in parent_components.items()
        for member in members
    }
    current_to_parent: dict[AssociationComponentMember, AssociationComponentMember] = {}
    if member_equivalence_directory is not None:
        current_to_parent = _current_to_parent_equivalences(
            member_equivalence_directory,
            parent_association_release_id=parent_manifest.association_release_id,
            current_association_release_id=current_manifest.association_release_id,
            parent_input_release_ids=parent_manifest.provenance.input_release_ids,
            current_input_release_ids=current_manifest.provenance.input_release_ids,
        )
    current_parents: dict[str, set[str]] = {}
    parent_children: dict[str, set[str]] = {
        component_id: set() for component_id in parent_components
    }
    for current_id, members in current_components.items():
        overlaps = {
            parent_by_member[parent_member]
            for member in members
            if (parent_member := current_to_parent.get(member, member)) in parent_by_member
        }
        if overlaps != set(declared_parents[current_id]):
            raise ValueError(
                "current component lineage does not match exact parent membership overlap"
            )
        current_parents[current_id] = overlaps
        for parent_id in overlaps:
            parent_children[parent_id].add(current_id)

    def delta(
        kind: AssociationComponentDeltaKind,
        parent_ids: tuple[str, ...],
        current_ids: tuple[str, ...],
    ) -> AssociationComponentDelta:
        return AssociationComponentDelta(
            parent_association_release_id=parent_manifest.association_release_id,
            parent_component_release_id=parent_manifest.component_release_id,
            current_association_release_id=current_manifest.association_release_id,
            current_component_release_id=current_manifest.component_release_id,
            kind=kind,
            parent_component_ids=parent_ids,
            current_component_ids=current_ids,
        )

    deltas: list[AssociationComponentDelta] = []
    for current_id, parent_ids in current_parents.items():
        if not parent_ids:
            deltas.append(delta(AssociationComponentDeltaKind.CREATED, (), (current_id,)))
        elif len(parent_ids) > 1:
            deltas.append(
                delta(
                    AssociationComponentDeltaKind.MERGED,
                    tuple(sorted(parent_ids)),
                    (current_id,),
                )
            )
        else:
            parent_id = next(iter(parent_ids))
            if len(parent_children[parent_id]) == 1:
                deltas.append(
                    delta(
                        AssociationComponentDeltaKind.CONTINUED,
                        (parent_id,),
                        (current_id,),
                    )
                )
    for parent_id, current_ids in parent_children.items():
        if not current_ids:
            deltas.append(delta(AssociationComponentDeltaKind.RETIRED, (parent_id,), ()))
        elif len(current_ids) > 1:
            deltas.append(
                delta(
                    AssociationComponentDeltaKind.SPLIT,
                    (parent_id,),
                    tuple(sorted(current_ids)),
                )
            )
    return parent_manifest, current_manifest, sorted(deltas, key=lambda item: item.delta_id)


def classify_association_component_deltas(
    parent_component_directory: str | os.PathLike[str],
    current_component_directory: str | os.PathLike[str],
    *,
    member_equivalence_directory: str | os.PathLike[str] | None = None,
) -> list[AssociationComponentDelta]:
    """Classify exact or explicitly equivalent member overlap."""
    return _classified_component_deltas(
        parent_component_directory,
        current_component_directory,
        member_equivalence_directory=member_equivalence_directory,
    )[2]


def _component_delta_release_identity(
    *,
    parent_association_release_id: str,
    parent_component_release_id: str,
    current_association_release_id: str,
    current_component_release_id: str,
    delta_count: int,
    deltas_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": ASSOCIATION_COMPONENT_DELTA_RELEASE_SCHEMA_VERSION,
        "delta_schema_version": ASSOCIATION_COMPONENT_DELTA_SCHEMA_VERSION,
        "parent_association_release_id": parent_association_release_id,
        "parent_component_release_id": parent_component_release_id,
        "current_association_release_id": current_association_release_id,
        "current_component_release_id": current_component_release_id,
        "delta_count": delta_count,
        "deltas_sha256": deltas_sha256,
    }


def _component_delta_release_id(**identity: Any) -> str:
    digest = hashlib.sha256(
        _canonical_json(_component_delta_release_identity(**identity))
    ).hexdigest()
    return f"{ASSOCIATION_COMPONENT_DELTA_RELEASE_SCHEMA_VERSION}:{digest}"


@dataclass(frozen=True)
class AssociationComponentDeltaReleaseManifest:
    """Content identity for a deterministic component delta artifact."""

    delta_release_id: str
    parent_association_release_id: str
    parent_component_release_id: str
    current_association_release_id: str
    current_component_release_id: str
    delta_count: int
    deltas_sha256: str
    delta_schema_version: str = ASSOCIATION_COMPONENT_DELTA_SCHEMA_VERSION
    deltas_path: str = _COMPONENT_DELTAS

    def __post_init__(self) -> None:
        if self.delta_schema_version != ASSOCIATION_COMPONENT_DELTA_SCHEMA_VERSION:
            raise ValueError(
                f"delta_schema_version must be {ASSOCIATION_COMPONENT_DELTA_SCHEMA_VERSION!r}"
            )
        if self.deltas_path != _COMPONENT_DELTAS:
            raise ValueError(f"deltas_path must be {_COMPONENT_DELTAS!r}")
        for value, schema, name in (
            (
                self.parent_association_release_id,
                ASSOCIATION_RELEASE_SCHEMA_VERSION,
                "parent_association_release_id",
            ),
            (
                self.parent_component_release_id,
                ASSOCIATION_COMPONENT_RELEASE_SCHEMA_VERSION,
                "parent_component_release_id",
            ),
            (
                self.current_association_release_id,
                ASSOCIATION_RELEASE_SCHEMA_VERSION,
                "current_association_release_id",
            ),
            (
                self.current_component_release_id,
                ASSOCIATION_COMPONENT_RELEASE_SCHEMA_VERSION,
                "current_component_release_id",
            ),
        ):
            _content_id(value, schema, name)
        if isinstance(self.delta_count, bool) or not isinstance(self.delta_count, int):
            raise ValueError("delta_count must be a non-negative integer")
        if self.delta_count < 0:
            raise ValueError("delta_count must be a non-negative integer")
        _sha256(self.deltas_sha256, "deltas_sha256")
        expected = _component_delta_release_id(
            parent_association_release_id=self.parent_association_release_id,
            parent_component_release_id=self.parent_component_release_id,
            current_association_release_id=self.current_association_release_id,
            current_component_release_id=self.current_component_release_id,
            delta_count=self.delta_count,
            deltas_sha256=self.deltas_sha256,
        )
        if self.delta_release_id != expected:
            raise ValueError("delta_release_id does not match the component delta release identity")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> AssociationComponentDeltaReleaseManifest:
        _exact_keys(
            data,
            {
                "schema_version",
                "delta_release_id",
                "delta_schema_version",
                "deltas_path",
                "parent_association_release_id",
                "parent_component_release_id",
                "current_association_release_id",
                "current_component_release_id",
                "delta_count",
                "deltas_sha256",
            },
            "association component delta release manifest",
        )
        if data["schema_version"] != ASSOCIATION_COMPONENT_DELTA_RELEASE_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {ASSOCIATION_COMPONENT_DELTA_RELEASE_SCHEMA_VERSION!r}"
            )
        return cls(
            delta_release_id=data["delta_release_id"],
            parent_association_release_id=data["parent_association_release_id"],
            parent_component_release_id=data["parent_component_release_id"],
            current_association_release_id=data["current_association_release_id"],
            current_component_release_id=data["current_component_release_id"],
            delta_count=data["delta_count"],
            deltas_sha256=data["deltas_sha256"],
            delta_schema_version=data["delta_schema_version"],
            deltas_path=data["deltas_path"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ASSOCIATION_COMPONENT_DELTA_RELEASE_SCHEMA_VERSION,
            "delta_release_id": self.delta_release_id,
            "delta_schema_version": self.delta_schema_version,
            "deltas_path": self.deltas_path,
            "parent_association_release_id": self.parent_association_release_id,
            "parent_component_release_id": self.parent_component_release_id,
            "current_association_release_id": self.current_association_release_id,
            "current_component_release_id": self.current_component_release_id,
            "delta_count": self.delta_count,
            "deltas_sha256": self.deltas_sha256,
        }


def load_association_component_delta_release_manifest(
    directory: str | os.PathLike[str],
) -> AssociationComponentDeltaReleaseManifest:
    """Load a component delta manifest without reading its delta rows."""
    path = Path(directory) / _RELEASE_MANIFEST
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read association component delta manifest: {path}") from exc
    if not isinstance(data, Mapping):
        raise ValueError("association component delta release manifest must be an object")
    return AssociationComponentDeltaReleaseManifest.from_mapping(data)


def _validated_component_deltas(
    directory: Path,
    manifest: AssociationComponentDeltaReleaseManifest,
) -> Iterator[AssociationComponentDelta]:
    path = directory / manifest.deltas_path
    digest = hashlib.sha256()
    previous_id: str | None = None
    count = 0
    try:
        handle = path.open("rb")
    except OSError as exc:
        raise ValueError(f"cannot read association component deltas: {path}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            digest.update(line)
            if not line.endswith(b"\n"):
                raise ValueError(
                    f"association component delta line {line_number} is not newline-terminated"
                )
            try:
                data = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"association component delta line {line_number} is not valid JSON"
                ) from exc
            if not isinstance(data, Mapping):
                raise ValueError(
                    f"association component delta line {line_number} must be an object"
                )
            delta = AssociationComponentDelta.from_mapping(data)
            for field in (
                "parent_association_release_id",
                "parent_component_release_id",
                "current_association_release_id",
                "current_component_release_id",
            ):
                if getattr(delta, field) != getattr(manifest, field):
                    raise ValueError(
                        f"association component delta line {line_number} endpoint differs "
                        "from the manifest"
                    )
            if previous_id is not None and delta.delta_id <= previous_id:
                raise ValueError(
                    "association component deltas must be strictly ordered by delta_id"
                )
            previous_id = delta.delta_id
            count += 1
            yield delta
    if count != manifest.delta_count:
        raise ValueError(
            f"delta_count mismatch: manifest has {manifest.delta_count}, deltas have {count}"
        )
    actual_digest = f"sha256:{digest.hexdigest()}"
    if actual_digest != manifest.deltas_sha256:
        raise ValueError("deltas_sha256 does not match deltas.jsonl")


def iter_association_component_delta_release(
    directory: str | os.PathLike[str],
) -> Iterator[AssociationComponentDelta]:
    """Stream delta rows; checksum and count validation complete at EOF."""
    path = Path(directory)
    manifest = load_association_component_delta_release_manifest(path)
    yield from _validated_component_deltas(path, manifest)


def verify_association_component_delta_release(
    directory: str | os.PathLike[str],
    *,
    parent_component_directory: str | os.PathLike[str] | None = None,
    current_component_directory: str | os.PathLike[str] | None = None,
    member_equivalence_directory: str | os.PathLike[str] | None = None,
) -> AssociationComponentDeltaReleaseManifest:
    """Verify a delta artifact and, when supplied, its component endpoints."""
    path = Path(directory)
    manifest = load_association_component_delta_release_manifest(path)
    actual = list(_validated_component_deltas(path, manifest))
    if parent_component_directory is not None:
        parent_manifest = verify_association_component_release(parent_component_directory)
        if (
            manifest.parent_association_release_id != parent_manifest.association_release_id
            or manifest.parent_component_release_id != parent_manifest.component_release_id
        ):
            raise ValueError("parent component release does not match the delta manifest")
    if current_component_directory is not None:
        current_manifest = verify_association_component_release(current_component_directory)
        if (
            manifest.current_association_release_id != current_manifest.association_release_id
            or manifest.current_component_release_id != current_manifest.component_release_id
        ):
            raise ValueError("current component release does not match the delta manifest")
    if parent_component_directory is not None and current_component_directory is not None:
        expected = classify_association_component_deltas(
            parent_component_directory,
            current_component_directory,
            member_equivalence_directory=member_equivalence_directory,
        )
        if actual != expected:
            raise ValueError("component deltas do not match exact endpoint membership overlap")
    return manifest


def write_association_component_delta_release(
    parent_component_directory: str | os.PathLike[str],
    current_component_directory: str | os.PathLike[str],
    directory: str | os.PathLike[str],
    *,
    member_equivalence_directory: str | os.PathLike[str] | None = None,
) -> AssociationComponentDeltaReleaseManifest:
    """Atomically publish deterministic deltas for two verified endpoints."""
    parent_manifest, current_manifest, deltas = _classified_component_deltas(
        parent_component_directory,
        current_component_directory,
        member_equivalence_directory=member_equivalence_directory,
    )
    target = Path(directory)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"association component delta path already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    try:
        digest = hashlib.sha256()
        deltas_path = temporary / _COMPONENT_DELTAS
        with deltas_path.open("xb") as handle:
            for delta in deltas:
                line = _canonical_json(delta.to_dict()) + b"\n"
                handle.write(line)
                digest.update(line)
            handle.flush()
            os.fsync(handle.fileno())
        identity = {
            "parent_association_release_id": parent_manifest.association_release_id,
            "parent_component_release_id": parent_manifest.component_release_id,
            "current_association_release_id": current_manifest.association_release_id,
            "current_component_release_id": current_manifest.component_release_id,
            "delta_count": len(deltas),
            "deltas_sha256": f"sha256:{digest.hexdigest()}",
        }
        manifest = AssociationComponentDeltaReleaseManifest(
            delta_release_id=_component_delta_release_id(**identity),
            **identity,
        )
        manifest_path = temporary / _RELEASE_MANIFEST
        with manifest_path.open("xb") as handle:
            handle.write(_canonical_json(manifest.to_dict()) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"association component delta path already exists: {target}")
        # one local publisher owns a target path; object stores need
        # their native conditional-create operation for concurrent publication.
        temporary.rename(target)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
