"""Versioned candidate-association records."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

ASSOCIATION_SCHEMA_VERSION = "xmatch.association.v1"


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
        digest = _text(self.parameters_sha256, "parameters_sha256")
        if not digest.startswith("sha256:") or len(digest) != 71:
            raise ValueError(
                "parameters_sha256 must be 'sha256:' followed by 64 hexadecimal digits"
            )
        try:
            int(digest[7:], 16)
        except ValueError as exc:
            raise ValueError(
                "parameters_sha256 must be 'sha256:' followed by 64 hexadecimal digits"
            ) from exc

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
