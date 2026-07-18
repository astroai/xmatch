"""Versioned candidate-association records and release artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

ASSOCIATION_SCHEMA_VERSION = "xmatch.association.v1"
ASSOCIATION_RELEASE_SCHEMA_VERSION = "xmatch.association.release.v1"

_RELEASE_MANIFEST = "manifest.json"
_RELEASE_RECORDS = "associations.jsonl"


class ScoreSemantics(str, Enum):
    """Scientific interpretation of an association score."""

    RANKING_SCORE = "ranking_score"
    ASSUMED_PRIOR_POSTERIOR = "assumed_prior_posterior"
    CALIBRATED_PROBABILITY = "calibrated_probability"


class AssociationDecision(str, Enum):
    """Disposition of a candidate without implying physical truth."""

    CANDIDATE = "candidate"
    SELECTED = "selected"
    REJECTED = "rejected"


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
    def from_mapping(cls, data: Mapping[str, Any]) -> "AssociationProvenance":
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
    def from_mapping(cls, data: Mapping[str, Any]) -> "AssociationRecord":
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
    def from_mapping(cls, data: Mapping[str, Any]) -> "AssociationReleaseManifest":
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
    if target.exists():
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

        if target.exists():
            raise FileExistsError(f"association release path already exists: {target}")
        # ponytail: one local publisher owns a target path; concurrent object-store
        # publication should use the provider's conditional-create primitive.
        temporary.rename(target)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
