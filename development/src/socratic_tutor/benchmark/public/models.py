"""Strict decision-time benchmark models with no criterion representation."""

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from socratic_tutor.benchmark.common import RelativePath, Sha256
from socratic_tutor.contracts.models import ContractModel


class BenchmarkCondition(StrEnum):
    """Prespecified conditions applied to the same public case state."""

    DIALOGUE_ONLY = "dialogue_only"
    PROBE_INFORMED = "probe_informed"
    UNRELATED_PROBE = "unrelated_probe"
    CORRUPTED_PROBE = "corrupted_probe"


EXPECTED_CONDITIONS = tuple(BenchmarkCondition)


class BenchmarkSplit(StrEnum):
    """Task-family-level roles; rows are never split independently."""

    DEVELOPMENT = "development"
    CALIBRATION = "calibration"
    POLICY_SELECTION = "policy_selection"
    HELD_OUT = "held_out"


class PublicArtifactClass(StrEnum):
    """File classes permitted in a public manifest projection."""

    PUBLIC_FIXTURE = "public_fixture"
    EVIDENCE_PROBE = "evidence_probe"
    EVIDENCE_TEST = "evidence_test"
    INITIAL_TRACKER = "initial_tracker"
    CONTROL = "control"
    SPLIT = "split"
    LIMITATIONS = "limitations"
    PROMPT = "prompt"


class PublicFileEntry(ContractModel):
    """Content-addressed file safe for decision-time readers."""

    path: RelativePath
    artifact_class: PublicArtifactClass
    schema_id: str = Field(min_length=1)
    sha256: Sha256
    byte_size: int = Field(ge=0)


class TaskFamilySplitView(ContractModel):
    """Public task-family assignment for one data role."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.split.v1"] = "benchmark.split.v1"
    split: BenchmarkSplit
    task_families: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_unique_families(self) -> "TaskFamilySplitView":
        if len(set(self.task_families)) != len(self.task_families):
            raise ValueError("Task families must be unique within a split")
        return self


class BenchmarkCaseView(ContractModel):
    """Complete public case projection available before outcome reveal."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.case_public.v1"] = "benchmark.case_public.v1"
    benchmark_version: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    task_family: str = Field(min_length=1)
    split: BenchmarkSplit
    target_concept: str = Field(min_length=1)
    public_fixture_ref: RelativePath
    evidence_probe_ref: RelativePath
    evidence_test_ref: RelativePath
    initial_tracker_ref: RelativePath
    initial_tracker_hash: Sha256
    unrelated_control_ref: RelativePath
    corruption_spec_ref: RelativePath
    case_content_hash: Sha256


class PublicBenchmarkManifest(ContractModel):
    """Content-addressed manifest consumed by decision-time commands."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.public_manifest.v1"] = "benchmark.public_manifest.v1"
    benchmark_version: str = Field(min_length=1)
    source_manifest_hash: Sha256
    projection_hash: Sha256
    projected_at_utc: datetime
    conditions: tuple[BenchmarkCondition, ...]
    cases: tuple[BenchmarkCaseView, ...] = Field(min_length=1)
    splits: tuple[TaskFamilySplitView, ...] = Field(min_length=1)
    files: tuple[PublicFileEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_identity_and_conditions(self) -> "PublicBenchmarkManifest":
        if self.conditions != EXPECTED_CONDITIONS:
            raise ValueError("Benchmark conditions must match the prespecified ordered set")
        case_ids = tuple(case.case_id for case in self.cases)
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("Case IDs must be unique")
        paths = tuple(entry.path for entry in self.files)
        if len(set(paths)) != len(paths):
            raise ValueError("Public file paths must be unique")
        return self
