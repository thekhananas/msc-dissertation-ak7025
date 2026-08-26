"""Pre-run sealing and guarded execution for the canonical acquisition study."""

from __future__ import annotations

import resource
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath
from time import perf_counter_ns
from typing import Literal, Self

import yaml
from pydantic import Field, TypeAdapter, ValidationError, model_validator

from socratic_tutor.acquisition_study.calibration import estimate_probe_reliability
from socratic_tutor.acquisition_study.canonical_stream import (
    CanonicalStreamReport,
    canonical_stream_schema_hashes,
    write_verified_canonical_streams,
)
from socratic_tutor.acquisition_study.contracts import PolicyId
from socratic_tutor.acquisition_study.plan import (
    AcquisitionAnalysisSpecification,
    AcquisitionEnvironmentSpecification,
    EvaluationEnvironmentId,
    load_acquisition_study_plan,
)
from socratic_tutor.benchmark.artifacts import immutable_json_bytes, write_immutable_json
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import canonical_sha256, file_sha256, model_content_hash
from socratic_tutor.contracts import ContractModel
from socratic_tutor.repository_state import current_clean_revision

_POLICY_ORDER = (
    PolicyId.RELIABILITY_AWARE_BOUNDED,
    PolicyId.SEEDED_RANDOM_BOUNDED,
    PolicyId.UNCERTAINTY_ONLY_BOUNDED,
    PolicyId.PLUG_IN_EVSI_BOUNDED,
    PolicyId.NEVER_PROBE,
    PolicyId.ALWAYS_PROBE_BOUNDED,
    PolicyId.ORACLE_TRUE_RELIABILITY_BOUNDED,
)
_ENVIRONMENT_ORDER = tuple(EvaluationEnvironmentId)
_PRE_RUN_MANIFEST_FILE = "canonical_pre_run_manifest.json"
_PRE_RUN_GATE_FILE = "canonical_pre_run_gate_report.json"
_STREAM_REPORT_FILE = "canonical_stream_report.json"
_RUN_MANIFEST_FILE = "canonical_run_manifest.json"
_JSON_OBJECT = TypeAdapter(dict[str, object])


class CanonicalExecutionError(ValueError):
    """The canonical execution package is incomplete, changed, or already used."""


class DevelopmentEvidenceRequirement(ContractModel):
    """One completed development artifact required before canonical execution."""

    evidence_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{2,79}$")
    path: str = Field(min_length=1)
    schema_id: str = Field(min_length=1)
    content_hash_field: Literal["manifest_hash", "report_hash"]
    expected_content_hash: Sha256
    required_result_status: str | None = None
    required_true_fields: tuple[str, ...] = ()
    required_false_fields: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_requirement(self) -> Self:
        _validate_relative_path(self.path)
        if set(self.required_true_fields) & set(self.required_false_fields):
            raise ValueError("An evidence field cannot be required both true and false")
        return self


class ReviewedFileRequirement(ContractModel):
    """One exact independent-review file required by the pre-run seal."""

    file_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{2,79}$")
    path: str = Field(min_length=1)
    expected_file_sha256: Sha256

    @model_validator(mode="after")
    def validate_requirement(self) -> Self:
        _validate_relative_path(self.path)
        return self


class CanonicalExecutionPlan(ContractModel):
    """Frozen instructions and prerequisites for the only canonical run."""

    schema_version: Literal[1] = 1
    schema_id: Literal["acquisition_study.canonical_execution_plan.v1"] = (
        "acquisition_study.canonical_execution_plan.v1"
    )
    study_id: Literal["reliability-aware-probing-v1"] = "reliability-aware-probing-v1"
    plan_status: Literal["frozen_before_canonical_evaluation"] = (
        "frozen_before_canonical_evaluation"
    )
    required_git_branch: Literal["main"] = "main"
    environment_specification_path: Literal["configs/acquisition-study/v1-environments.yaml"] = (
        "configs/acquisition-study/v1-environments.yaml"
    )
    environment_specification_hash: Sha256
    analysis_specification_path: Literal["configs/acquisition-study/v1-analysis.yaml"] = (
        "configs/acquisition-study/v1-analysis.yaml"
    )
    analysis_specification_hash: Sha256
    pixi_lock_path: Literal["pixi.lock"] = "pixi.lock"
    pixi_lock_sha256: Sha256
    public_schema_hash: Sha256
    restricted_schema_hash: Sha256
    pre_run_output_root: Literal["artifacts/acquisition-study/canonical-v1-pre-run"] = (
        "artifacts/acquisition-study/canonical-v1-pre-run"
    )
    canonical_output_root: Literal["artifacts/acquisition-study/canonical-v1"] = (
        "artifacts/acquisition-study/canonical-v1"
    )
    run_id: Literal["reliability-aware-probing-v1-canonical"] = (
        "reliability-aware-probing-v1-canonical"
    )
    split: Literal["evaluation"] = "evaluation"
    environment_ids: tuple[EvaluationEnvironmentId, ...] = Field(min_length=7, max_length=7)
    policy_ids: tuple[PolicyId, ...] = Field(min_length=7, max_length=7)
    development_episodes_per_environment: Literal[50] = 50
    evaluation_episodes_per_environment: Literal[2000] = 2000
    evaluation_episode_index_start: Literal[50] = 50
    evaluation_episode_index_stop_exclusive: Literal[2050] = 2050
    candidates_per_episode: Literal[40] = 40
    matched_probe_budget: Literal[20] = 20
    expected_environment_count: Literal[7] = 7
    expected_policy_count: Literal[7] = 7
    expected_episode_count: Literal[14000] = 14_000
    expected_public_row_count: Literal[98000] = 98_000
    expected_restricted_row_count: Literal[14000] = 14_000
    expected_case_prediction_count: Literal[3920000] = 3_920_000
    expected_selected_probe_count: Literal[1960000] = 1_960_000
    canonical_evaluation_run_limit: Literal[1] = 1
    exact_seeded_retry_required: Literal[True] = True
    dynamic_policy_loading_allowed: Literal[False] = False
    parameter_selection_after_evaluation_allowed: Literal[False] = False
    evaluation_outcomes_update_reliability_allowed: Literal[False] = False
    external_model_calls_expected: Literal[0] = 0
    sandbox_calls_expected: Literal[0] = 0
    human_records_expected: Literal[0] = 0
    human_learning_claim_allowed: Literal[False] = False
    tutoring_efficacy_claim_allowed: Literal[False] = False
    cognitive_offloading_claim_allowed: Literal[False] = False
    deployed_cost_saving_claim_allowed: Literal[False] = False
    known_blocking_defect_ids: tuple[str, ...] = Field(max_length=0)
    development_evidence: tuple[DevelopmentEvidenceRequirement, ...] = Field(
        min_length=8,
        max_length=8,
    )
    reviewed_files: tuple[ReviewedFileRequirement, ...] = Field(min_length=4, max_length=4)
    plan_hash: Sha256

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        if self.environment_ids != _ENVIRONMENT_ORDER:
            raise ValueError("Canonical plan differs from the frozen environment order")
        if self.policy_ids != _POLICY_ORDER:
            raise ValueError("Canonical plan differs from the frozen policy order")
        if (
            self.evaluation_episode_index_stop_exclusive - self.evaluation_episode_index_start
            != self.evaluation_episodes_per_environment
        ):
            raise ValueError("Canonical evaluation episode range does not reconcile")
        if (
            self.expected_environment_count * self.evaluation_episodes_per_environment
            != self.expected_episode_count
        ):
            raise ValueError("Canonical episode count does not reconcile")
        if self.expected_episode_count * self.expected_policy_count != (
            self.expected_public_row_count
        ):
            raise ValueError("Canonical public row count does not reconcile")
        if self.expected_restricted_row_count != self.expected_episode_count:
            raise ValueError("Canonical restricted row count does not reconcile")
        if self.expected_public_row_count * self.candidates_per_episode != (
            self.expected_case_prediction_count
        ):
            raise ValueError("Canonical prediction count does not reconcile")
        expected_selected = self.expected_episode_count * (
            5 * self.matched_probe_budget + self.candidates_per_episode
        )
        if self.expected_selected_probe_count != expected_selected:
            raise ValueError("Canonical selected-probe count does not reconcile")
        evidence_ids = tuple(item.evidence_id for item in self.development_evidence)
        review_ids = tuple(item.file_id for item in self.reviewed_files)
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("Canonical evidence identifiers must be unique")
        if len(review_ids) != len(set(review_ids)):
            raise ValueError("Canonical review identifiers must be unique")
        if self.plan_hash != model_content_hash(self, exclude={"plan_hash"}):
            raise ValueError("Canonical execution plan hash does not match its content")
        return self


class SealedDevelopmentEvidence(ContractModel):
    """Verified identity of one development artifact."""

    evidence_id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    schema_id: str = Field(min_length=1)
    file_sha256: Sha256
    content_hash: Sha256
    result_status: str | None = None


class SealedReviewedFile(ContractModel):
    """Verified identity of one independent-review file."""

    file_id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    file_sha256: Sha256


class CanonicalPreRunManifest(ContractModel):
    """Immutable record of the exact code, inputs, contracts, and workload."""

    schema_version: Literal[1] = 1
    schema_id: Literal["acquisition_study.canonical_pre_run_manifest.v1"] = (
        "acquisition_study.canonical_pre_run_manifest.v1"
    )
    study_id: Literal["reliability-aware-probing-v1"] = "reliability-aware-probing-v1"
    run_id: Literal["reliability-aware-probing-v1-canonical"] = (
        "reliability-aware-probing-v1-canonical"
    )
    sealed_on: date
    code_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    git_branch: Literal["main"] = "main"
    repository_clean: Literal[True] = True
    execution_plan_hash: Sha256
    execution_plan_file_sha256: Sha256
    environment_specification_hash: Sha256
    analysis_specification_hash: Sha256
    calibration_hash: Sha256
    pixi_lock_sha256: Sha256
    public_schema_hash: Sha256
    restricted_schema_hash: Sha256
    environment_ids: tuple[EvaluationEnvironmentId, ...] = Field(min_length=7, max_length=7)
    policy_ids: tuple[PolicyId, ...] = Field(min_length=7, max_length=7)
    evaluation_episode_index_start: Literal[50] = 50
    evaluation_episode_index_stop_exclusive: Literal[2050] = 2050
    episodes_per_environment: Literal[2000] = 2000
    candidates_per_episode: Literal[40] = 40
    matched_probe_budget: Literal[20] = 20
    expected_episode_count: Literal[14000] = 14_000
    expected_public_row_count: Literal[98000] = 98_000
    expected_restricted_row_count: Literal[14000] = 14_000
    expected_case_prediction_count: Literal[3920000] = 3_920_000
    expected_selected_probe_count: Literal[1960000] = 1_960_000
    canonical_output_root: Literal["artifacts/acquisition-study/canonical-v1"] = (
        "artifacts/acquisition-study/canonical-v1"
    )
    canonical_output_absent_at_seal: Literal[True] = True
    canonical_evaluation_run_limit: Literal[1] = 1
    exact_seeded_retry_required: Literal[True] = True
    dynamic_policy_loading_allowed: Literal[False] = False
    parameter_selection_after_evaluation_allowed: Literal[False] = False
    development_evidence: tuple[SealedDevelopmentEvidence, ...] = Field(
        min_length=8,
        max_length=8,
    )
    reviewed_files: tuple[SealedReviewedFile, ...] = Field(min_length=4, max_length=4)
    known_blocking_defect_ids: tuple[str, ...] = Field(max_length=0)
    external_model_calls_expected: Literal[0] = 0
    sandbox_calls_expected: Literal[0] = 0
    human_records_expected: Literal[0] = 0
    human_learning_claim_allowed: Literal[False] = False
    tutoring_efficacy_claim_allowed: Literal[False] = False
    cognitive_offloading_claim_allowed: Literal[False] = False
    deployed_cost_saving_claim_allowed: Literal[False] = False
    manifest_hash: Sha256

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        if self.environment_ids != _ENVIRONMENT_ORDER or self.policy_ids != _POLICY_ORDER:
            raise ValueError("Pre-run manifest differs from the frozen study order")
        if self.manifest_hash != model_content_hash(self, exclude={"manifest_hash"}):
            raise ValueError("Pre-run manifest hash does not match its content")
        return self


class CanonicalSealCheck(ContractModel):
    """One successful pre-run gate check."""

    check_id: str = Field(pattern=r"^[a-z][a-z0-9_]{2,79}$")
    status: Literal["passed"] = "passed"
    detail: str = Field(min_length=1)


class CanonicalPreRunGateReport(ContractModel):
    """Machine-readable proof that every pre-run gate passed."""

    schema_version: Literal[1] = 1
    schema_id: Literal["acquisition_study.canonical_pre_run_gate_report.v1"] = (
        "acquisition_study.canonical_pre_run_gate_report.v1"
    )
    study_id: Literal["reliability-aware-probing-v1"] = "reliability-aware-probing-v1"
    run_id: Literal["reliability-aware-probing-v1-canonical"] = (
        "reliability-aware-probing-v1-canonical"
    )
    pre_run_manifest_hash: Sha256
    pre_run_manifest_file_sha256: Sha256
    check_count: Literal[8] = 8
    checks: tuple[CanonicalSealCheck, ...] = Field(min_length=8, max_length=8)
    gate_passed: Literal[True] = True
    report_hash: Sha256

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        check_ids = tuple(item.check_id for item in self.checks)
        if len(check_ids) != len(set(check_ids)):
            raise ValueError("Canonical pre-run check identifiers must be unique")
        if self.report_hash != model_content_hash(self, exclude={"report_hash"}):
            raise ValueError("Canonical pre-run gate hash does not match its content")
        return self


class CanonicalRunManifest(ContractModel):
    """Completion record for the raw canonical matrix, before analysis."""

    schema_version: Literal[1] = 1
    schema_id: Literal["acquisition_study.canonical_run_manifest.v1"] = (
        "acquisition_study.canonical_run_manifest.v1"
    )
    study_id: Literal["reliability-aware-probing-v1"] = "reliability-aware-probing-v1"
    run_id: Literal["reliability-aware-probing-v1-canonical"] = (
        "reliability-aware-probing-v1-canonical"
    )
    code_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    execution_plan_hash: Sha256
    pre_run_manifest_hash: Sha256
    pre_run_gate_report_hash: Sha256
    stream_report_file: Literal["canonical_stream_report.json"] = _STREAM_REPORT_FILE
    stream_report_file_sha256: Sha256
    stream_report_hash: Sha256
    public_file: Literal["canonical_public_policy_metrics.jsonl"]
    public_file_sha256: Sha256
    restricted_file: Literal["canonical_restricted_episodes.jsonl"]
    restricted_file_sha256: Sha256
    elapsed_nanoseconds: int = Field(ge=1)
    peak_rss_bytes: int = Field(ge=1)
    peak_memory_source: Literal["resource.getrusage_rusage_self"] = "resource.getrusage_rusage_self"
    episode_count: Literal[14000] = 14_000
    policy_result_count: Literal[98000] = 98_000
    case_prediction_count: Literal[3920000] = 3_920_000
    selected_probe_count: Literal[1960000] = 1_960_000
    exclusion_count: Literal[0] = 0
    canonical_estimate_count: Literal[1] = 1
    exact_seeded_retry_verified: Literal[True] = True
    external_model_call_count: Literal[0] = 0
    sandbox_call_count: Literal[0] = 0
    human_record_count: Literal[0] = 0
    completion_status: Literal["canonical_raw_evaluation_complete_unanalysed"] = (
        "canonical_raw_evaluation_complete_unanalysed"
    )
    scientific_interpretation_allowed: Literal[False] = False
    human_learning_claim_supported: Literal[False] = False
    tutoring_efficacy_claim_supported: Literal[False] = False
    cognitive_offloading_claim_supported: Literal[False] = False
    deployed_cost_saving_claim_supported: Literal[False] = False
    manifest_hash: Sha256

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        if self.manifest_hash != model_content_hash(self, exclude={"manifest_hash"}):
            raise ValueError("Canonical run manifest hash does not match its content")
        return self


@dataclass(frozen=True, slots=True)
class _StaticInputs:
    specification: AcquisitionEnvironmentSpecification
    analysis: AcquisitionAnalysisSpecification
    pixi_lock_hash: Sha256
    public_schema_hash: Sha256
    restricted_schema_hash: Sha256
    evidence: tuple[SealedDevelopmentEvidence, ...]
    reviewed_files: tuple[SealedReviewedFile, ...]


def load_canonical_execution_plan(path: Path) -> CanonicalExecutionPlan:
    """Load and self-verify the frozen canonical execution plan."""

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise CanonicalExecutionError("Canonical execution plan must be a mapping")
        return CanonicalExecutionPlan.model_validate(raw)
    except CanonicalExecutionError:
        raise
    except (OSError, ValidationError, yaml.YAMLError) as error:
        raise CanonicalExecutionError(f"Could not load canonical execution plan: {path}") from error


def seal_canonical_execution(
    execution_plan_path: Path,
    *,
    sealed_on: date,
) -> tuple[CanonicalPreRunManifest, CanonicalPreRunGateReport]:
    """Verify all prerequisites and publish an immutable pre-run seal."""

    plan = load_canonical_execution_plan(execution_plan_path)
    project_root = execution_plan_path.resolve().parents[2]
    code_revision, branch = _clean_repository_state(project_root, plan.required_git_branch)
    static = _load_static_inputs(project_root, plan)
    canonical_output_root = _project_path(project_root, plan.canonical_output_root)
    if canonical_output_root.exists():
        raise CanonicalExecutionError(
            f"Canonical output root already exists: {plan.canonical_output_root}"
        )
    try:
        plan_file_hash = file_sha256(execution_plan_path.read_bytes())
    except OSError as error:
        raise CanonicalExecutionError("Could not hash the canonical execution plan") from error

    manifest_content = {
        "schema_version": 1,
        "schema_id": "acquisition_study.canonical_pre_run_manifest.v1",
        "study_id": plan.study_id,
        "run_id": plan.run_id,
        "sealed_on": sealed_on,
        "code_revision": code_revision,
        "git_branch": branch,
        "repository_clean": True,
        "execution_plan_hash": plan.plan_hash,
        "execution_plan_file_sha256": plan_file_hash,
        "environment_specification_hash": static.specification.specification_hash,
        "analysis_specification_hash": static.analysis.plan_hash,
        "calibration_hash": estimate_probe_reliability(static.specification).calibration_hash,
        "pixi_lock_sha256": static.pixi_lock_hash,
        "public_schema_hash": static.public_schema_hash,
        "restricted_schema_hash": static.restricted_schema_hash,
        "environment_ids": plan.environment_ids,
        "policy_ids": plan.policy_ids,
        "evaluation_episode_index_start": plan.evaluation_episode_index_start,
        "evaluation_episode_index_stop_exclusive": (plan.evaluation_episode_index_stop_exclusive),
        "episodes_per_environment": plan.evaluation_episodes_per_environment,
        "candidates_per_episode": plan.candidates_per_episode,
        "matched_probe_budget": plan.matched_probe_budget,
        "expected_episode_count": plan.expected_episode_count,
        "expected_public_row_count": plan.expected_public_row_count,
        "expected_restricted_row_count": plan.expected_restricted_row_count,
        "expected_case_prediction_count": plan.expected_case_prediction_count,
        "expected_selected_probe_count": plan.expected_selected_probe_count,
        "canonical_output_root": plan.canonical_output_root,
        "canonical_output_absent_at_seal": True,
        "canonical_evaluation_run_limit": plan.canonical_evaluation_run_limit,
        "exact_seeded_retry_required": plan.exact_seeded_retry_required,
        "dynamic_policy_loading_allowed": plan.dynamic_policy_loading_allowed,
        "parameter_selection_after_evaluation_allowed": (
            plan.parameter_selection_after_evaluation_allowed
        ),
        "development_evidence": static.evidence,
        "reviewed_files": static.reviewed_files,
        "known_blocking_defect_ids": plan.known_blocking_defect_ids,
        "external_model_calls_expected": plan.external_model_calls_expected,
        "sandbox_calls_expected": plan.sandbox_calls_expected,
        "human_records_expected": plan.human_records_expected,
        "human_learning_claim_allowed": plan.human_learning_claim_allowed,
        "tutoring_efficacy_claim_allowed": plan.tutoring_efficacy_claim_allowed,
        "cognitive_offloading_claim_allowed": plan.cognitive_offloading_claim_allowed,
        "deployed_cost_saving_claim_allowed": plan.deployed_cost_saving_claim_allowed,
    }
    manifest = CanonicalPreRunManifest.model_validate(
        {**manifest_content, "manifest_hash": canonical_sha256(manifest_content)}
    )
    checks = (
        CanonicalSealCheck(
            check_id="clean_committed_revision",
            detail=f"Clean {branch} revision {code_revision} is fixed.",
        ),
        CanonicalSealCheck(
            check_id="frozen_plan_and_lock",
            detail="Execution plan, environment plan, analysis plan, and Pixi lock match.",
        ),
        CanonicalSealCheck(
            check_id="policy_and_environment_order",
            detail="Seven fixed policies and seven fixed environments use the declared order.",
        ),
        CanonicalSealCheck(
            check_id="disjoint_episode_ranges",
            detail="Development indices 0-49 and evaluation indices 50-2049 are disjoint.",
        ),
        CanonicalSealCheck(
            check_id="workload_and_budget",
            detail="The 14,000-episode workload and all matched probe budgets reconcile.",
        ),
        CanonicalSealCheck(
            check_id="public_restricted_boundary",
            detail="Public and restricted record schemas match their frozen hashes.",
        ),
        CanonicalSealCheck(
            check_id="development_and_review_evidence",
            detail="Eight development artifacts and four review files match their hashes.",
        ),
        CanonicalSealCheck(
            check_id="unused_output_and_claim_limits",
            detail="The output root is unused; one-run and claim limits remain active.",
        ),
    )
    manifest_file_hash = file_sha256(immutable_json_bytes(manifest))
    gate_content = {
        "schema_version": 1,
        "schema_id": "acquisition_study.canonical_pre_run_gate_report.v1",
        "study_id": plan.study_id,
        "run_id": plan.run_id,
        "pre_run_manifest_hash": manifest.manifest_hash,
        "pre_run_manifest_file_sha256": manifest_file_hash,
        "check_count": len(checks),
        "checks": checks,
        "gate_passed": True,
    }
    gate = CanonicalPreRunGateReport.model_validate(
        {**gate_content, "report_hash": canonical_sha256(gate_content)}
    )
    seal_root = _project_path(project_root, plan.pre_run_output_root)
    write_immutable_json(seal_root / _PRE_RUN_MANIFEST_FILE, manifest)
    write_immutable_json(seal_root / _PRE_RUN_GATE_FILE, gate)
    return manifest, gate


def run_canonical_execution(
    execution_plan_path: Path,
) -> CanonicalRunManifest:
    """Execute the sealed matrix once without exposing an alternative configuration."""

    plan = load_canonical_execution_plan(execution_plan_path)
    project_root = execution_plan_path.resolve().parents[2]
    seal_root = _project_path(project_root, plan.pre_run_output_root)
    pre_run_manifest_path = seal_root / _PRE_RUN_MANIFEST_FILE
    gate_report_path = seal_root / _PRE_RUN_GATE_FILE
    pre_run = _load_model(pre_run_manifest_path, CanonicalPreRunManifest)
    gate = _load_model(gate_report_path, CanonicalPreRunGateReport)
    code_revision, branch = _clean_repository_state(project_root, plan.required_git_branch)
    if branch != pre_run.git_branch or code_revision != pre_run.code_revision:
        raise CanonicalExecutionError("Current Git revision differs from the pre-run seal")
    if pre_run.execution_plan_hash != plan.plan_hash:
        raise CanonicalExecutionError("Pre-run seal belongs to another execution plan")
    if gate.pre_run_manifest_hash != pre_run.manifest_hash or not gate.gate_passed:
        raise CanonicalExecutionError("Pre-run gate does not authorise this manifest")
    try:
        actual_manifest_file_hash = file_sha256(pre_run_manifest_path.read_bytes())
    except OSError as error:
        raise CanonicalExecutionError("Could not hash the pre-run manifest") from error
    if gate.pre_run_manifest_file_sha256 != actual_manifest_file_hash:
        raise CanonicalExecutionError("Pre-run manifest bytes differ from the gate report")

    static = _load_static_inputs(project_root, plan)
    _compare_static_inputs(pre_run, static)
    output_root = _project_path(project_root, plan.canonical_output_root)
    _verify_recoverable_output_root(output_root)
    started = perf_counter_ns()
    stream_report = write_verified_canonical_streams(
        static.specification,
        static.analysis,
        output_root=output_root,
    )
    elapsed = perf_counter_ns() - started
    _verify_stream_report(plan, pre_run, stream_report)
    stream_report_path = output_root / _STREAM_REPORT_FILE
    write_immutable_json(stream_report_path, stream_report)
    stream_report_file_hash = file_sha256(immutable_json_bytes(stream_report))
    run_content = {
        "schema_version": 1,
        "schema_id": "acquisition_study.canonical_run_manifest.v1",
        "study_id": plan.study_id,
        "run_id": plan.run_id,
        "code_revision": code_revision,
        "execution_plan_hash": plan.plan_hash,
        "pre_run_manifest_hash": pre_run.manifest_hash,
        "pre_run_gate_report_hash": gate.report_hash,
        "stream_report_file": _STREAM_REPORT_FILE,
        "stream_report_file_sha256": stream_report_file_hash,
        "stream_report_hash": stream_report.report_hash,
        "public_file": stream_report.public_file,
        "public_file_sha256": stream_report.public_file_sha256,
        "restricted_file": stream_report.restricted_file,
        "restricted_file_sha256": stream_report.restricted_file_sha256,
        "elapsed_nanoseconds": elapsed,
        "peak_rss_bytes": _peak_rss_bytes(),
        "peak_memory_source": "resource.getrusage_rusage_self",
        "episode_count": stream_report.episode_count,
        "policy_result_count": stream_report.public_row_count,
        "case_prediction_count": stream_report.case_prediction_count,
        "selected_probe_count": stream_report.selected_probe_count,
        "exclusion_count": 0,
        "canonical_estimate_count": stream_report.canonical_estimate_count,
        "exact_seeded_retry_verified": stream_report.deterministic_replay_verified,
        "external_model_call_count": stream_report.external_model_call_count,
        "sandbox_call_count": stream_report.sandbox_call_count,
        "human_record_count": stream_report.human_record_count,
        "completion_status": "canonical_raw_evaluation_complete_unanalysed",
        "scientific_interpretation_allowed": False,
        "human_learning_claim_supported": False,
        "tutoring_efficacy_claim_supported": False,
        "cognitive_offloading_claim_supported": False,
        "deployed_cost_saving_claim_supported": False,
    }
    run_manifest = CanonicalRunManifest.model_validate(
        {**run_content, "manifest_hash": canonical_sha256(run_content)}
    )
    write_immutable_json(output_root / _RUN_MANIFEST_FILE, run_manifest)
    return run_manifest


def _load_static_inputs(
    project_root: Path,
    plan: CanonicalExecutionPlan,
) -> _StaticInputs:
    specification, analysis = load_acquisition_study_plan(
        _project_path(project_root, plan.environment_specification_path),
        _project_path(project_root, plan.analysis_specification_path),
    )
    if specification.specification_hash != plan.environment_specification_hash:
        raise CanonicalExecutionError("Environment specification differs from the execution plan")
    if analysis.plan_hash != plan.analysis_specification_hash:
        raise CanonicalExecutionError("Analysis specification differs from the execution plan")
    if (
        specification.episodes.development_episodes_per_environment
        != plan.development_episodes_per_environment
        or specification.episodes.evaluation_episodes_per_environment
        != plan.evaluation_episodes_per_environment
        or specification.episodes.candidates_per_episode != plan.candidates_per_episode
        or analysis.primary.exact_selected_probes_per_episode != plan.matched_probe_budget
    ):
        raise CanonicalExecutionError("Frozen episode shape differs from the execution plan")
    pixi_lock_path = _project_path(project_root, plan.pixi_lock_path)
    try:
        pixi_lock_hash = file_sha256(pixi_lock_path.read_bytes())
    except OSError as error:
        raise CanonicalExecutionError("Could not hash the Pixi lock") from error
    if pixi_lock_hash != plan.pixi_lock_sha256:
        raise CanonicalExecutionError("Pixi lock differs from the execution plan")
    public_schema_hash, restricted_schema_hash = canonical_stream_schema_hashes()
    if (
        public_schema_hash != plan.public_schema_hash
        or restricted_schema_hash != plan.restricted_schema_hash
    ):
        raise CanonicalExecutionError("Canonical record schema differs from the execution plan")
    evidence = tuple(
        verify_development_evidence(project_root, item) for item in plan.development_evidence
    )
    reviewed_files = tuple(
        _verify_reviewed_file(project_root, item) for item in plan.reviewed_files
    )
    return _StaticInputs(
        specification=specification,
        analysis=analysis,
        pixi_lock_hash=pixi_lock_hash,
        public_schema_hash=public_schema_hash,
        restricted_schema_hash=restricted_schema_hash,
        evidence=evidence,
        reviewed_files=reviewed_files,
    )


def verify_development_evidence(
    project_root: Path,
    requirement: DevelopmentEvidenceRequirement,
) -> SealedDevelopmentEvidence:
    path = _project_path(project_root, requirement.path)
    try:
        raw_bytes = path.read_bytes()
        payload = _JSON_OBJECT.validate_json(raw_bytes)
    except (OSError, ValidationError) as error:
        raise CanonicalExecutionError(
            f"Could not read development evidence: {requirement.evidence_id}"
        ) from error
    if payload.get("schema_id") != requirement.schema_id:
        raise CanonicalExecutionError(
            f"Development evidence schema changed: {requirement.evidence_id}"
        )
    claimed_hash = payload.get(requirement.content_hash_field)
    if claimed_hash != requirement.expected_content_hash:
        raise CanonicalExecutionError(
            f"Development evidence hash changed: {requirement.evidence_id}"
        )
    content = dict(payload)
    content.pop(requirement.content_hash_field, None)
    if canonical_sha256(content) != requirement.expected_content_hash:
        raise CanonicalExecutionError(
            f"Development evidence self-hash failed: {requirement.evidence_id}"
        )
    result_status = payload.get("result_status")
    if requirement.required_result_status is not None and (
        result_status != requirement.required_result_status
    ):
        raise CanonicalExecutionError(
            f"Development evidence is incomplete: {requirement.evidence_id}"
        )
    for field_name in requirement.required_true_fields:
        if payload.get(field_name) is not True:
            raise CanonicalExecutionError(
                f"Development evidence did not pass {field_name}: {requirement.evidence_id}"
            )
    for field_name in requirement.required_false_fields:
        if payload.get(field_name) is not False:
            raise CanonicalExecutionError(
                f"Development evidence violated {field_name}: {requirement.evidence_id}"
            )
    return SealedDevelopmentEvidence(
        evidence_id=requirement.evidence_id,
        path=requirement.path,
        schema_id=requirement.schema_id,
        file_sha256=file_sha256(raw_bytes),
        content_hash=requirement.expected_content_hash,
        result_status=result_status if isinstance(result_status, str) else None,
    )


def _verify_reviewed_file(
    project_root: Path,
    requirement: ReviewedFileRequirement,
) -> SealedReviewedFile:
    path = _project_path(project_root, requirement.path)
    try:
        actual_hash = file_sha256(path.read_bytes())
    except OSError as error:
        raise CanonicalExecutionError(
            f"Could not read reviewed file: {requirement.file_id}"
        ) from error
    if actual_hash != requirement.expected_file_sha256:
        raise CanonicalExecutionError(f"Reviewed file changed: {requirement.file_id}")
    return SealedReviewedFile(
        file_id=requirement.file_id,
        path=requirement.path,
        file_sha256=actual_hash,
    )


def _compare_static_inputs(
    pre_run: CanonicalPreRunManifest,
    static: _StaticInputs,
) -> None:
    if (
        pre_run.environment_specification_hash != static.specification.specification_hash
        or pre_run.analysis_specification_hash != static.analysis.plan_hash
        or pre_run.pixi_lock_sha256 != static.pixi_lock_hash
        or pre_run.public_schema_hash != static.public_schema_hash
        or pre_run.restricted_schema_hash != static.restricted_schema_hash
        or pre_run.development_evidence != static.evidence
        or pre_run.reviewed_files != static.reviewed_files
    ):
        raise CanonicalExecutionError("A sealed input changed after the pre-run gate")
    calibration_hash = estimate_probe_reliability(static.specification).calibration_hash
    if pre_run.calibration_hash != calibration_hash:
        raise CanonicalExecutionError("Calibration output changed after the pre-run gate")


def _verify_stream_report(
    plan: CanonicalExecutionPlan,
    pre_run: CanonicalPreRunManifest,
    report: CanonicalStreamReport,
) -> None:
    if (
        report.environment_specification_hash != pre_run.environment_specification_hash
        or report.analysis_plan_hash != pre_run.analysis_specification_hash
        or report.calibration_hash != pre_run.calibration_hash
        or report.public_schema_hash != pre_run.public_schema_hash
        or report.restricted_schema_hash != pre_run.restricted_schema_hash
        or report.episode_count != plan.expected_episode_count
        or report.public_row_count != plan.expected_public_row_count
        or report.restricted_row_count != plan.expected_restricted_row_count
        or report.case_prediction_count != plan.expected_case_prediction_count
        or report.selected_probe_count != plan.expected_selected_probe_count
    ):
        raise CanonicalExecutionError("Canonical stream does not match the pre-run seal")


def _verify_recoverable_output_root(output_root: Path) -> None:
    if not output_root.exists():
        return
    files = {path.name for path in output_root.iterdir() if path.is_file()}
    if _RUN_MANIFEST_FILE in files:
        raise CanonicalExecutionError("Canonical evaluation is already complete")
    allowed = {
        "canonical_public_policy_metrics.jsonl",
        "canonical_restricted_episodes.jsonl",
        _STREAM_REPORT_FILE,
    }
    unexpected = files - allowed
    if unexpected or any(path.is_dir() for path in output_root.iterdir()):
        raise CanonicalExecutionError("Canonical output root contains unexpected content")


def _clean_repository_state(project_root: Path, required_branch: str) -> tuple[str, str]:
    try:
        revision = current_clean_revision(
            project_root,
            dirty_message="Commit or remove all visible changes before sealing",
            untracked_files="all",
            required_branch=required_branch,
            operation_name="Canonical execution",
            invalid_revision_message="Git did not return a full lowercase commit SHA",
            error_factory=CanonicalExecutionError,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise CanonicalExecutionError("Could not verify the Git repository") from error
    return revision, required_branch


def _load_model[ModelT: ContractModel](path: Path, model: type[ModelT]) -> ModelT:
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        raise CanonicalExecutionError(f"Could not load sealed artifact: {path.name}") from error


def _project_path(project_root: Path, relative_path: str) -> Path:
    _validate_relative_path(relative_path)
    resolved = (project_root / relative_path).resolve()
    if not resolved.is_relative_to(project_root.resolve()):
        raise CanonicalExecutionError(f"Path leaves the project root: {relative_path}")
    return resolved


def _validate_relative_path(value: str) -> None:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value != path.as_posix():
        raise ValueError(f"Path must be a normal relative POSIX path: {value}")


def _peak_rss_bytes() -> int:
    raw_value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw_value if sys.platform == "darwin" else raw_value * 1024
