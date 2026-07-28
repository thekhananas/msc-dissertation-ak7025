"""Evaluator-private authored definitions and outcome specifications."""

from datetime import date, datetime
from enum import StrEnum
from typing import Any, Literal, cast

from pydantic import Field, model_validator

from socratic_tutor.benchmark.common import RelativePath, Sha256
from socratic_tutor.benchmark.hashing import model_content_hash
from socratic_tutor.benchmark.public.models import (
    EXPECTED_CONDITIONS,
    BenchmarkCaseView,
    BenchmarkCondition,
    TaskFamilySplitView,
)
from socratic_tutor.contracts.models import ContractModel


class ManifestStatus(StrEnum):
    """A held-out run may consume only a separately validated frozen manifest."""

    DRAFT = "draft"
    FROZEN = "frozen"


class ArtifactClass(StrEnum):
    """All authored file classes, including evaluator-only material."""

    PUBLIC_FIXTURE = "public_fixture"
    EVIDENCE_PROBE = "evidence_probe"
    EVIDENCE_TEST = "evidence_test"
    INITIAL_TRACKER = "initial_tracker"
    CONTROL = "control"
    SPLIT = "split"
    LIMITATIONS = "limitations"
    PROMPT = "prompt"
    CRITERION_PROBE = "criterion_probe"
    CRITERION_TEST = "criterion_test"
    CRITERION_RUBRIC = "criterion_rubric"
    LABEL_RATIONALE = "label_rationale"
    REVIEW = "review"
    STRUCTURAL_DIFFERENCE = "structural_difference"


class EvidencePattern(StrEnum):
    """Evaluator-only stratum describing the relationship between evidence and outcome."""

    SUPPORTED_MASTERY = "supported_mastery"
    FALSE_PUBLIC_MASTERY = "false_public_mastery"
    HONEST_NON_MASTERY = "honest_non_mastery"
    UNDERCONFIDENCE = "underconfidence"
    AMBIGUOUS = "ambiguous"


class FileInventoryEntry(ContractModel):
    """Expected digest and size for one authored benchmark file."""

    path: RelativePath
    artifact_class: ArtifactClass
    schema_id: str = Field(min_length=1)
    sha256: Sha256
    byte_size: int = Field(ge=0)


class CriterionSpec(ContractModel):
    """Held-out task specification unavailable to decision-time packages."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.criterion_spec.v1"] = "benchmark.criterion_spec.v1"
    benchmark_version: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    misconception_id: str = Field(min_length=1)
    transfer_case_id: str = Field(min_length=1)
    evidence_pattern: EvidencePattern
    criterion_probe_id: str = Field(min_length=1)
    criterion_prompt_ref: RelativePath
    test_bundle_ref: RelativePath
    test_bundle_sha256: Sha256
    rubric_ref: RelativePath
    rubric_sha256: Sha256
    label_rationale_ref: RelativePath
    review_record_ref: RelativePath
    structural_difference_record_ref: RelativePath


class AuthoredCase(ContractModel):
    """Authoring-time join between one public case and its private outcome task."""

    public: BenchmarkCaseView
    criterion: CriterionSpec

    @model_validator(mode="after")
    def require_matching_identity(self) -> "AuthoredCase":
        if self.public.case_id != self.criterion.case_id:
            raise ValueError("Public and criterion case IDs must match")
        if self.public.benchmark_version != self.criterion.benchmark_version:
            raise ValueError("Public and criterion benchmark versions must match")
        return self


class ReviewDecision(StrEnum):
    """Independent review outcome for exact content hashes."""

    PENDING = "pending"
    APPROVED = "approved"
    CHANGES_REQUESTED = "changes_requested"


class ReviewedFile(ContractModel):
    """Exact authored file covered by a review decision."""

    path: RelativePath
    sha256: Sha256


class ReviewRecord(ContractModel):
    """Review provenance kept separate from model-generated outputs."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.review.v1"] = "benchmark.review.v1"
    review_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    author: str = Field(min_length=1)
    reviewer: str = Field(min_length=1)
    reviewed_on: date | None = None
    decision: ReviewDecision
    concerns: tuple[str, ...] = ()
    adjudication: str | None = None
    reviewed_files: tuple[ReviewedFile, ...] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_completion_timestamp(cls, value: Any) -> Any:
        """Read prior timestamp-based manifests without preserving a time value."""

        if not isinstance(value, dict) or "reviewed_at_utc" not in value:
            return cast(Any, value)
        normalized: dict[str, Any] = cast(dict[str, Any], value).copy()
        legacy_value: Any = normalized.pop("reviewed_at_utc")
        if "reviewed_on" not in normalized:
            normalized["reviewed_on"] = (
                legacy_value[:10] if isinstance(legacy_value, str) else legacy_value
            )
        return cast(Any, normalized)

    @model_validator(mode="after")
    def validate_review_completion(self) -> "ReviewRecord":
        if self.decision is ReviewDecision.PENDING and self.reviewed_on is not None:
            raise ValueError("Pending reviews cannot have a completion date")
        return self


class PrimaryMetricSpec(ContractModel):
    """Prespecified primary endpoint metadata."""

    name: Literal["paired_brier_difference"] = "paired_brier_difference"
    favourable_direction: Literal["positive"] = "positive"
    aggregation_unit: Literal["case"] = "case"
    interval_method: str = Field(min_length=1)
    missingness_rule: str = Field(min_length=1)


class GenerationSpec(ContractModel):
    """Pinned generation condition; model access remains optional for draft fixtures."""

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    system_prompt_version: str = Field(min_length=1)
    system_prompt_ref: RelativePath
    system_prompt_sha256: Sha256
    repeats_per_case: int = Field(ge=1)
    temperature: float = Field(ge=0.0, le=2.0)
    max_output_tokens: int = Field(ge=1)


class AuthoredBenchmarkManifest(ContractModel):
    """Root authoring model containing both public and evaluator material."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.manifest.v1"] = "benchmark.manifest.v1"
    benchmark_version: str = Field(min_length=1)
    status: ManifestStatus
    frozen_at_utc: datetime | None = None
    parent_version: str | None = None
    manifest_hash: Sha256 | None = None
    conditions: tuple[BenchmarkCondition, ...]
    cases: tuple[AuthoredCase, ...] = Field(min_length=1)
    splits: tuple[TaskFamilySplitView, ...] = Field(min_length=1)
    files: tuple[FileInventoryEntry, ...] = Field(min_length=1)
    reviews: tuple[ReviewRecord, ...] = Field(min_length=1)
    primary_metric: PrimaryMetricSpec
    generation: GenerationSpec
    limitations_ref: RelativePath

    @model_validator(mode="after")
    def validate_manifest_invariants(self) -> "AuthoredBenchmarkManifest":
        if self.conditions != EXPECTED_CONDITIONS:
            raise ValueError("Benchmark conditions must match the prespecified ordered set")
        if self.status is ManifestStatus.FROZEN and self.frozen_at_utc is None:
            raise ValueError("Frozen manifests require a freeze timestamp")
        if self.status is ManifestStatus.DRAFT and self.frozen_at_utc is not None:
            raise ValueError("Draft manifests cannot have a freeze timestamp")

        case_ids = tuple(item.public.case_id for item in self.cases)
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("Case IDs must be unique")
        review_ids = tuple(review.review_id for review in self.reviews)
        if len(set(review_ids)) != len(review_ids):
            raise ValueError("Review IDs must be unique")
        reviewed_case_ids = tuple(review.case_id for review in self.reviews)
        if len(set(reviewed_case_ids)) != len(reviewed_case_ids):
            raise ValueError("Each case may have only one review record")
        paths = tuple(item.path for item in self.files)
        if len(set(paths)) != len(paths):
            raise ValueError("Inventory paths must be unique")
        split_roles = tuple(item.split for item in self.splits)
        if len(set(split_roles)) != len(split_roles):
            raise ValueError("Each split role may appear only once")
        return self


def authored_manifest_content_hash(manifest: AuthoredBenchmarkManifest) -> Sha256:
    """Hash the manifest while excluding its self-referential digest field."""

    return model_content_hash(manifest, exclude={"manifest_hash"})


class EvaluatorBenchmarkManifest(ContractModel):
    """Private projection used only after the decision phase is sealed."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.evaluator_manifest.v1"] = "benchmark.evaluator_manifest.v1"
    benchmark_version: str
    source_manifest_hash: Sha256
    public_projection_hash: Sha256
    projection_hash: Sha256
    projected_at_utc: datetime
    criteria: tuple[CriterionSpec, ...]
    reviews: tuple[ReviewRecord, ...]
    files: tuple[FileInventoryEntry, ...]
