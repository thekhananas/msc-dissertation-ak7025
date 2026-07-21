"""Typed allocation contract for the benchmark before case authoring."""

from collections import Counter, defaultdict
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from socratic_tutor.benchmark.command_io import load_command_model
from socratic_tutor.benchmark.common import RelativePath, Sha256
from socratic_tutor.benchmark.evaluator.models import EvidencePattern
from socratic_tutor.benchmark.hashing import file_sha256
from socratic_tutor.contracts import ContractModel

_ID_PATTERN = r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$"


class DesignStatus(StrEnum):
    """Whether authoring may still change the allocation."""

    DRAFT = "draft"
    FROZEN = "frozen"


class AuxiliarySplit(StrEnum):
    """Non-held-out splits reserved before primary case authoring."""

    DEVELOPMENT = "development"
    CALIBRATION = "calibration"


class TransferRole(StrEnum):
    """How a case transfers the misconception into a new context."""

    DIRECT = "direct"
    BOUNDARY = "boundary"
    COMPOSED = "composed"


class DifficultyBand(StrEnum):
    """Semantic difficulty, not code length or algorithmic difficulty."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ConceptDesign(ContractModel):
    """One approved concept and exactly two target misconceptions."""

    concept_id: str = Field(pattern=_ID_PATTERN)
    name: str = Field(min_length=1)
    misconception_ids: tuple[str, str]

    @model_validator(mode="after")
    def validate_misconceptions(self) -> "ConceptDesign":
        if len(set(self.misconception_ids)) != 2:
            raise ValueError("A concept must contain two distinct misconceptions")
        if any(not _is_valid_id(value) for value in self.misconception_ids):
            raise ValueError("Misconception IDs must be normalized kebab-case IDs")
        return self


class AuxiliaryFamilyReservation(ContractModel):
    """A task family unavailable to the held-out split."""

    split: AuxiliarySplit
    task_family: str = Field(pattern=_ID_PATTERN)
    concept_id: str | None = Field(default=None, pattern=_ID_PATTERN)
    provenance: Literal["existing", "planned"]


class HeldOutCaseAllocation(ContractModel):
    """One evaluator-private slot to be filled by authored case material."""

    case_id: str = Field(pattern=_ID_PATTERN)
    concept_id: str = Field(pattern=_ID_PATTERN)
    misconception_id: str = Field(pattern=_ID_PATTERN)
    task_family: str = Field(pattern=_ID_PATTERN)
    transfer_role: TransferRole
    difficulty: DifficultyBand
    evidence_pattern: EvidencePattern
    evidence_context: str = Field(min_length=1)
    criterion_context: str = Field(min_length=1)
    design_intent: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_distinct_contexts(self) -> "HeldOutCaseAllocation":
        if self.evidence_context.casefold() == self.criterion_context.casefold():
            raise ValueError("Evidence and criterion contexts must differ")
        return self


class BenchmarkDesignPlan(ContractModel):
    """Complete 24-case allocation and split reservation for benchmark v1."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.design_plan.v1"] = "benchmark.design_plan.v1"
    benchmark_version: str = Field(min_length=1)
    status: DesignStatus
    prepared_at_utc: datetime
    frozen_at_utc: datetime | None = None
    concept_blueprint_ref: RelativePath
    concept_blueprint_sha256: Sha256
    concepts: tuple[ConceptDesign, ...] = Field(min_length=4, max_length=4)
    auxiliary_families: tuple[AuxiliaryFamilyReservation, ...] = Field(min_length=1)
    held_out_cases: tuple[HeldOutCaseAllocation, ...] = Field(min_length=24, max_length=24)

    @model_validator(mode="after")
    def validate_design(self) -> "BenchmarkDesignPlan":
        _require_utc(self.prepared_at_utc, "Design preparation time")
        if self.status is DesignStatus.DRAFT and self.frozen_at_utc is not None:
            raise ValueError("A draft design cannot have a freeze time")
        if self.status is DesignStatus.FROZEN:
            if self.frozen_at_utc is None:
                raise ValueError("A frozen design requires a freeze time")
            _require_utc(self.frozen_at_utc, "Design freeze time")

        concept_ids = tuple(concept.concept_id for concept in self.concepts)
        if len(set(concept_ids)) != 4:
            raise ValueError("The design must contain four distinct concepts")
        concept_lookup = {concept.concept_id: concept for concept in self.concepts}
        misconception_owners: dict[str, str] = {}
        for concept in self.concepts:
            for misconception_id in concept.misconception_ids:
                if misconception_id in misconception_owners:
                    raise ValueError("Misconception IDs must be globally unique")
                misconception_owners[misconception_id] = concept.concept_id

        case_ids = tuple(case.case_id for case in self.held_out_cases)
        if len(set(case_ids)) != 24:
            raise ValueError("The design must contain 24 unique held-out case IDs")
        held_out_families = tuple(case.task_family for case in self.held_out_cases)
        if len(set(held_out_families)) != 24:
            raise ValueError("Every held-out case requires a distinct task family")

        grouped: dict[tuple[str, str], list[HeldOutCaseAllocation]] = defaultdict(list)
        patterns_by_concept: dict[str, set[EvidencePattern]] = defaultdict(set)
        for case in self.held_out_cases:
            concept = concept_lookup.get(case.concept_id)
            if concept is None:
                raise ValueError(f"Unknown concept in case {case.case_id}")
            if case.misconception_id not in concept.misconception_ids:
                raise ValueError(f"Misconception does not belong to concept in case {case.case_id}")
            grouped[(case.concept_id, case.misconception_id)].append(case)
            patterns_by_concept[case.concept_id].add(case.evidence_pattern)

        expected_roles = set(TransferRole)
        expected_difficulties = set(DifficultyBand)
        for misconception_id, concept_id in misconception_owners.items():
            cases = grouped[(concept_id, misconception_id)]
            if len(cases) != 3:
                raise ValueError(f"Misconception {misconception_id} requires exactly three cases")
            if {case.transfer_role for case in cases} != expected_roles:
                raise ValueError(f"Misconception {misconception_id} must cover all transfer roles")
            if {case.difficulty for case in cases} != expected_difficulties:
                raise ValueError(
                    f"Misconception {misconception_id} must cover all difficulty bands"
                )

        expected_patterns = set(EvidencePattern)
        for concept_id in concept_ids:
            if patterns_by_concept[concept_id] != expected_patterns:
                raise ValueError(f"Concept {concept_id} must cover every evidence pattern")
        pattern_counts = Counter(case.evidence_pattern for case in self.held_out_cases)
        if sorted(pattern_counts.values()) != [4, 5, 5, 5, 5]:
            raise ValueError("Evidence patterns must be balanced across the 24 held-out cases")

        auxiliary_families = tuple(item.task_family for item in self.auxiliary_families)
        if len(set(auxiliary_families)) != len(auxiliary_families):
            raise ValueError("Auxiliary task families must be unique")
        if set(auxiliary_families) & set(held_out_families):
            raise ValueError("Auxiliary and held-out task families must be disjoint")
        for reservation in self.auxiliary_families:
            if reservation.concept_id is not None and reservation.concept_id not in concept_lookup:
                raise ValueError("Auxiliary family references an unknown concept")
        for concept_id in concept_ids:
            reserved_splits = {
                item.split for item in self.auxiliary_families if item.concept_id == concept_id
            }
            if reserved_splits != set(AuxiliarySplit):
                raise ValueError(
                    f"Concept {concept_id} requires development and calibration families"
                )

        evidence_contexts = [case.evidence_context.casefold() for case in self.held_out_cases]
        criterion_contexts = [case.criterion_context.casefold() for case in self.held_out_cases]
        if len(set(evidence_contexts)) != len(evidence_contexts):
            raise ValueError("Evidence contexts must be unique")
        if len(set(criterion_contexts)) != len(criterion_contexts):
            raise ValueError("Criterion contexts must be unique")
        return self


class BenchmarkDesignReadError(RuntimeError):
    """The allocation or its approved blueprint cannot be verified."""


def load_and_verify_design(path: Path) -> BenchmarkDesignPlan:
    """Load a design and verify the exact concept blueprint bytes it references."""

    plan = load_command_model(path, BenchmarkDesignPlan)
    blueprint_path = path.parent / plan.concept_blueprint_ref
    try:
        blueprint_content = blueprint_path.read_bytes()
    except OSError as error:
        raise BenchmarkDesignReadError(
            f"Could not read concept blueprint: {blueprint_path}"
        ) from error
    if file_sha256(blueprint_content) != plan.concept_blueprint_sha256:
        raise BenchmarkDesignReadError("Concept blueprint hash does not match the design plan")
    return plan


def _is_valid_id(value: str) -> bool:
    import re

    return re.fullmatch(_ID_PATTERN, value) is not None


def _require_utc(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must use UTC")
