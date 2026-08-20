"""Hash-verified primary analysis of the sealed canonical acquisition stream."""

from __future__ import annotations

import csv
import hashlib
import io
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from statistics import fmean
from typing import Literal, Self

import numpy as np
import yaml
from pydantic import Field, ValidationError, model_validator

from socratic_tutor.acquisition_study.canonical_execution import (
    CanonicalPreRunGateReport,
    CanonicalPreRunManifest,
    CanonicalRunManifest,
    load_canonical_execution_plan,
)
from socratic_tutor.acquisition_study.canonical_stream import (
    CanonicalPublicPolicyMetric,
    CanonicalStreamReport,
)
from socratic_tutor.acquisition_study.contracts import PolicyId
from socratic_tutor.acquisition_study.plan import (
    AcquisitionAnalysisSpecification,
    EnvironmentRole,
    EvaluationEnvironmentId,
    load_acquisition_study_plan,
)
from socratic_tutor.acquisition_study.primary_analysis import (
    EnvironmentPairedEffect,
    stratified_paired_bootstrap_distributions,
)
from socratic_tutor.benchmark.artifacts import (
    immutable_json_bytes,
    write_immutable_bytes,
    write_immutable_json,
)
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import canonical_sha256, file_sha256, model_content_hash
from socratic_tutor.contracts import ContractModel

_CANDIDATE = PolicyId.RELIABILITY_AWARE_BOUNDED
_COMPARATORS = (
    PolicyId.SEEDED_RANDOM_BOUNDED,
    PolicyId.UNCERTAINTY_ONLY_BOUNDED,
    PolicyId.PLUG_IN_EVSI_BOUNDED,
)
_POLICY_ORDER = (
    _CANDIDATE,
    *_COMPARATORS,
    PolicyId.NEVER_PROBE,
    PolicyId.ALWAYS_PROBE_BOUNDED,
    PolicyId.ORACLE_TRUE_RELIABILITY_BOUNDED,
)
_ENVIRONMENT_ORDER = tuple(EvaluationEnvironmentId)
_EXECUTION_PLAN_FILE = "canonical_primary_analysis_execution_plan.json"
_REPORT_FILE = "canonical_primary_analysis_report.json"
_ENVIRONMENT_TABLE_FILE = "canonical_primary_environment_effects.csv"
_COMPARISON_TABLE_FILE = "canonical_primary_comparisons.csv"
_CLAIM_TABLE_FILE = "canonical_primary_claim_audit.csv"
_MANIFEST_FILE = "canonical_primary_analysis_manifest.json"


class CanonicalPrimaryAnalysisError(ValueError):
    """The sealed source or primary analysis violates the frozen design."""


class CanonicalPrimarySourcePlan(ContractModel):
    """Source identities and rules frozen before canonical outcomes are read."""

    schema_version: Literal[1] = 1
    schema_id: Literal["acquisition_study.canonical_primary_source_plan.v1"] = (
        "acquisition_study.canonical_primary_source_plan.v1"
    )
    study_id: Literal["reliability-aware-probing-v1"] = "reliability-aware-probing-v1"
    plan_status: Literal["frozen_before_canonical_outcome_inspection"]
    frozen_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    raw_policy_outcomes_inspected_before_freeze: Literal[False] = False
    canonical_execution_plan_path: str
    canonical_execution_plan_hash: Sha256
    analysis_specification_path: str
    analysis_specification_hash: Sha256
    pixi_lock_path: str
    pixi_lock_sha256: Sha256
    pre_run_manifest_path: str
    pre_run_manifest_hash: Sha256
    pre_run_manifest_file_sha256: Sha256
    pre_run_gate_path: str
    pre_run_gate_hash: Sha256
    pre_run_gate_file_sha256: Sha256
    canonical_run_manifest_path: str
    canonical_run_manifest_hash: Sha256
    canonical_run_manifest_file_sha256: Sha256
    canonical_stream_report_path: str
    canonical_stream_report_hash: Sha256
    canonical_stream_report_file_sha256: Sha256
    public_stream_path: str
    public_stream_sha256: Sha256
    restricted_stream_sha256: Sha256
    canonical_code_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    candidate_policy_id: Literal[PolicyId.RELIABILITY_AWARE_BOUNDED] = _CANDIDATE
    comparator_policy_ids: tuple[PolicyId, ...] = Field(min_length=3, max_length=3)
    environment_ids: tuple[EvaluationEnvironmentId, ...] = Field(min_length=7, max_length=7)
    policy_ids: tuple[PolicyId, ...] = Field(min_length=7, max_length=7)
    evaluation_episode_index_start: Literal[50] = 50
    evaluation_episode_index_stop_exclusive: Literal[2050] = 2050
    episodes_per_environment: Literal[2000] = 2000
    candidates_per_episode: Literal[40] = 40
    matched_probe_budget: Literal[20] = 20
    expected_public_row_count: Literal[98000] = 98_000
    interval_method: Literal["stratified_paired_episode_percentile_bootstrap"]
    bootstrap_repetitions: Literal[20000] = 20_000
    bootstrap_seed: Literal[9052505] = 9_052_505
    simultaneous_confidence_level: float = Field(gt=0.95, lt=1.0)
    precision_target_half_width: float = Field(gt=0.0, le=1.0)
    overall_advantage_rule: Literal["comparison_claim_holds_for_all_three_comparators"]
    robustness_rule: Literal[
        "overall_advantage_and_no_held_out_point_regression_against_plug_in_evsi"
    ]
    output_root: Literal["artifacts/acquisition-study/canonical-v1/primary-analysis-v1"]
    restricted_stream_access_allowed: Literal[False] = False
    external_calls_allowed: Literal[False] = False
    human_learning_claim_allowed: Literal[False] = False
    tutoring_efficacy_claim_allowed: Literal[False] = False
    cognitive_offloading_claim_allowed: Literal[False] = False
    deployed_cost_saving_claim_allowed: Literal[False] = False
    state_of_the_art_claim_allowed: Literal[False] = False
    plan_hash: Sha256

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        for value in (
            self.canonical_execution_plan_path,
            self.analysis_specification_path,
            self.pixi_lock_path,
            self.pre_run_manifest_path,
            self.pre_run_gate_path,
            self.canonical_run_manifest_path,
            self.canonical_stream_report_path,
            self.public_stream_path,
            self.output_root,
        ):
            _validate_relative_path(value)
        if self.comparator_policy_ids != _COMPARATORS:
            raise ValueError("Primary comparators differ from the frozen order")
        if self.environment_ids != _ENVIRONMENT_ORDER or self.policy_ids != _POLICY_ORDER:
            raise ValueError("Canonical source plan differs from the frozen study order")
        if (
            self.evaluation_episode_index_stop_exclusive - self.evaluation_episode_index_start
            != self.episodes_per_environment
        ):
            raise ValueError("Canonical source episode range does not reconcile")
        expected_rows = (
            len(self.environment_ids) * self.episodes_per_environment * len(self.policy_ids)
        )
        if self.expected_public_row_count != expected_rows:
            raise ValueError("Canonical source row count does not reconcile")
        if not math.isclose(self.precision_target_half_width, 0.02, abs_tol=1e-12):
            raise ValueError("Canonical precision target differs from the frozen plan")
        if self.plan_hash != model_content_hash(self, exclude={"plan_hash"}):
            raise ValueError("Canonical primary source-plan hash does not match its content")
        return self


class CanonicalPrimaryExecutionPlan(ContractModel):
    """Committed analysis identity joined to the pre-outcome source plan."""

    schema_version: Literal[1] = 1
    schema_id: Literal["acquisition_study.canonical_primary_execution_plan.v1"] = (
        "acquisition_study.canonical_primary_execution_plan.v1"
    )
    study_id: Literal["reliability-aware-probing-v1"] = "reliability-aware-probing-v1"
    source_plan_hash: Sha256
    source_plan_file_sha256: Sha256
    canonical_code_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    analysis_code_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    canonical_run_manifest_hash: Sha256
    canonical_stream_report_hash: Sha256
    public_stream_sha256: Sha256
    frozen_analysis_specification_hash: Sha256
    pixi_lock_sha256: Sha256
    candidate_policy_id: Literal[PolicyId.RELIABILITY_AWARE_BOUNDED] = _CANDIDATE
    comparator_policy_ids: tuple[PolicyId, ...] = Field(min_length=3, max_length=3)
    held_out_environment_ids: tuple[EvaluationEnvironmentId, ...] = Field(
        min_length=6,
        max_length=6,
    )
    episodes_per_environment: Literal[2000] = 2000
    interval_method: Literal["stratified_paired_episode_percentile_bootstrap"]
    bootstrap_repetitions: Literal[20000] = 20_000
    bootstrap_seed: Literal[9052505] = 9_052_505
    simultaneous_confidence_level: float = Field(gt=0.95, lt=1.0)
    precision_target_half_width: float = Field(gt=0.0, le=1.0)
    restricted_stream_accessed: Literal[False] = False
    external_call_count: Literal[0] = 0
    plan_hash: Sha256

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        if self.comparator_policy_ids != _COMPARATORS:
            raise ValueError("Canonical execution comparators differ from the frozen order")
        if self.held_out_environment_ids != _ENVIRONMENT_ORDER[1:]:
            raise ValueError("Canonical execution held-out environments differ")
        if not math.isclose(self.precision_target_half_width, 0.02, abs_tol=1e-12):
            raise ValueError("Canonical execution precision target changed")
        if self.plan_hash != model_content_hash(self, exclude={"plan_hash"}):
            raise ValueError("Canonical primary execution-plan hash does not match its content")
        return self


class CanonicalComparatorInterval(ContractModel):
    """Multiplicity-adjusted candidate comparison over held-out environments."""

    candidate_policy_id: Literal[PolicyId.RELIABILITY_AWARE_BOUNDED] = _CANDIDATE
    comparator_policy_id: PolicyId
    episodes_per_environment: int = Field(ge=1)
    environment_effects: tuple[EnvironmentPairedEffect, ...] = Field(
        min_length=6,
        max_length=6,
    )
    macro_mean_paired_effect: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    confidence_level: float = Field(gt=0.95, lt=1.0)
    interval_lower: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    interval_upper: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    interval_half_width: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    precision_target_half_width: float = Field(gt=0.0, le=1.0)
    precision_target_met: bool
    bootstrap_repetitions: int = Field(ge=10_000)
    bootstrap_seed: int = Field(ge=0)
    paired_effects_hash: Sha256
    bootstrap_distribution_hash: Sha256
    simultaneous_interval_rule_met: bool
    worst_environment_id: EvaluationEnvironmentId
    worst_environment_mean_effect: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        if self.comparator_policy_id not in _COMPARATORS:
            raise ValueError("Canonical interval uses a non-primary comparator")
        if (
            tuple(row.environment_id for row in self.environment_effects)
            != (_ENVIRONMENT_ORDER[1:])
        ):
            raise ValueError("Canonical interval has incomplete held-out coverage")
        if any(
            row.environment_role is not EnvironmentRole.HELD_OUT_MISMATCH
            or row.comparator_policy_id is not self.comparator_policy_id
            or row.episode_count != self.episodes_per_environment
            for row in self.environment_effects
        ):
            raise ValueError("Canonical interval contains inconsistent summaries")
        expected_macro = fmean(row.mean_paired_effect for row in self.environment_effects)
        if not math.isclose(self.macro_mean_paired_effect, expected_macro, abs_tol=1e-12):
            raise ValueError("Canonical interval does not equally weight environments")
        if self.interval_lower > self.interval_upper:
            raise ValueError("Canonical interval bounds are reversed")
        expected_half_width = (self.interval_upper - self.interval_lower) / 2.0
        if not math.isclose(self.interval_half_width, expected_half_width, abs_tol=1e-12):
            raise ValueError("Canonical interval half-width is inconsistent")
        if self.precision_target_met is not (
            self.interval_half_width <= self.precision_target_half_width
        ):
            raise ValueError("Canonical precision flag is inconsistent")
        if self.simultaneous_interval_rule_met is not (self.interval_upper < 0.0):
            raise ValueError("Canonical interval decision is inconsistent")
        worst = max(self.environment_effects, key=lambda row: row.mean_paired_effect)
        if self.worst_environment_id is not worst.environment_id or not math.isclose(
            self.worst_environment_mean_effect,
            worst.mean_paired_effect,
            abs_tol=1e-12,
        ):
            raise ValueError("Canonical worst-environment result is inconsistent")
        return self


ClaimStatus = Literal[
    "supported_within_declared_simulator",
    "not_supported_by_primary_rule",
    "prohibited_by_design",
]


class CanonicalClaimAssessment(ContractModel):
    """One result statement with its permitted interpretation."""

    claim_id: str = Field(pattern=r"^[a-z][a-z0-9_]{2,79}$")
    statement: str = Field(min_length=1)
    status: ClaimStatus
    basis: str = Field(min_length=1)


class CanonicalPrimaryAnalysisReport(ContractModel):
    """Prespecified canonical primary result, limited to the declared simulator."""

    schema_version: Literal[1] = 1
    schema_id: Literal["acquisition_study.canonical_primary_analysis_report.v1"] = (
        "acquisition_study.canonical_primary_analysis_report.v1"
    )
    study_id: Literal["reliability-aware-probing-v1"] = "reliability-aware-probing-v1"
    split: Literal["evaluation"] = "evaluation"
    result_status: Literal["canonical_primary_complete_secondary_pending"] = (
        "canonical_primary_complete_secondary_pending"
    )
    execution_plan_hash: Sha256
    source_plan_hash: Sha256
    canonical_run_manifest_hash: Sha256
    canonical_stream_report_hash: Sha256
    public_stream_sha256: Sha256
    source_public_row_count: int = Field(ge=1)
    candidate_policy_id: Literal[PolicyId.RELIABILITY_AWARE_BOUNDED] = _CANDIDATE
    comparator_policy_ids: tuple[PolicyId, ...] = Field(min_length=3, max_length=3)
    episodes_per_environment: int = Field(ge=1)
    matched_environment_effects: tuple[EnvironmentPairedEffect, ...] = Field(
        min_length=3,
        max_length=3,
    )
    held_out_primary_intervals: tuple[CanonicalComparatorInterval, ...] = Field(
        min_length=3,
        max_length=3,
    )
    all_simultaneous_interval_rules_met: bool
    all_precision_targets_met: bool
    no_held_out_point_regression_against_plug_in_evsi: bool
    overall_advantage_rule_met: bool
    robustness_rule_met: bool
    claim_assessments: tuple[CanonicalClaimAssessment, ...] = Field(
        min_length=7,
        max_length=7,
    )
    inference_scope: Literal["independent_episodes_from_declared_simulator_only"] = (
        "independent_episodes_from_declared_simulator_only"
    )
    restricted_stream_accessed: Literal[False] = False
    external_model_call_count: Literal[0] = 0
    sandbox_call_count: Literal[0] = 0
    human_record_count: Literal[0] = 0
    exact_analysis_retry_verified: Literal[True] = True
    report_hash: Sha256

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        if self.comparator_policy_ids != _COMPARATORS:
            raise ValueError("Canonical report comparator order changed")
        if tuple(row.comparator_policy_id for row in self.matched_environment_effects) != (
            _COMPARATORS
        ) or any(
            row.environment_id is not EvaluationEnvironmentId.MATCHED
            or row.environment_role is not EnvironmentRole.MATCHED_SANITY_CHECK
            for row in self.matched_environment_effects
        ):
            raise ValueError("Canonical report matched summaries changed")
        if tuple(row.comparator_policy_id for row in self.held_out_primary_intervals) != (
            _COMPARATORS
        ):
            raise ValueError("Canonical report interval order changed")
        expected_rows = len(_ENVIRONMENT_ORDER) * len(_POLICY_ORDER) * self.episodes_per_environment
        if self.source_public_row_count != expected_rows:
            raise ValueError("Canonical report source row count is inconsistent")
        expected_all = all(
            row.simultaneous_interval_rule_met for row in self.held_out_primary_intervals
        )
        expected_precision = all(
            row.precision_target_met for row in self.held_out_primary_intervals
        )
        plug_in = self.held_out_primary_intervals[-1]
        expected_no_regression = all(
            row.mean_paired_effect <= 0.0 for row in plug_in.environment_effects
        )
        if self.all_simultaneous_interval_rules_met is not expected_all:
            raise ValueError("Canonical all-comparison flag is inconsistent")
        if self.all_precision_targets_met is not expected_precision:
            raise ValueError("Canonical precision flag is inconsistent")
        if self.no_held_out_point_regression_against_plug_in_evsi is not expected_no_regression:
            raise ValueError("Canonical plug-in guardrail is inconsistent")
        if self.overall_advantage_rule_met is not expected_all:
            raise ValueError("Canonical overall-advantage rule is inconsistent")
        if self.robustness_rule_met is not (expected_all and expected_no_regression):
            raise ValueError("Canonical robustness rule is inconsistent")
        expected_claim_ids = (
            "all_comparator_advantage",
            "declared_mismatch_robustness",
            "human_learning",
            "tutoring_efficacy",
            "cognitive_offloading",
            "deployed_cost_saving",
            "state_of_the_art",
        )
        if tuple(row.claim_id for row in self.claim_assessments) != expected_claim_ids:
            raise ValueError("Canonical claim audit changed")
        if self.report_hash != model_content_hash(self, exclude={"report_hash"}):
            raise ValueError("Canonical primary report hash does not match its content")
        return self


class CanonicalPrimaryAnalysisManifest(ContractModel):
    """Exact file inventory for the canonical primary analysis."""

    schema_version: Literal[1] = 1
    schema_id: Literal["acquisition_study.canonical_primary_analysis_manifest.v1"] = (
        "acquisition_study.canonical_primary_analysis_manifest.v1"
    )
    study_id: Literal["reliability-aware-probing-v1"] = "reliability-aware-probing-v1"
    source_plan_hash: Sha256
    execution_plan_file: Literal["canonical_primary_analysis_execution_plan.json"] = (
        _EXECUTION_PLAN_FILE
    )
    execution_plan_file_sha256: Sha256
    execution_plan_hash: Sha256
    report_file: Literal["canonical_primary_analysis_report.json"] = _REPORT_FILE
    report_file_sha256: Sha256
    report_hash: Sha256
    environment_table_file: Literal["canonical_primary_environment_effects.csv"] = (
        _ENVIRONMENT_TABLE_FILE
    )
    environment_table_sha256: Sha256
    comparison_table_file: Literal["canonical_primary_comparisons.csv"] = _COMPARISON_TABLE_FILE
    comparison_table_sha256: Sha256
    claim_table_file: Literal["canonical_primary_claim_audit.csv"] = _CLAIM_TABLE_FILE
    claim_table_sha256: Sha256
    public_stream_sha256: Sha256
    source_public_row_count: Literal[98000] = 98_000
    restricted_stream_accessed: Literal[False] = False
    exact_analysis_retry_verified: Literal[True] = True
    completion_status: Literal["canonical_primary_complete_secondary_pending"] = (
        "canonical_primary_complete_secondary_pending"
    )
    manifest_hash: Sha256

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        if self.manifest_hash != model_content_hash(self, exclude={"manifest_hash"}):
            raise ValueError("Canonical primary manifest hash does not match its content")
        return self


@dataclass(frozen=True, slots=True)
class PublicStreamExpectation:
    """Expected public row layout, also usable by small contract tests."""

    environment_ids: tuple[EvaluationEnvironmentId, ...]
    policy_ids: tuple[PolicyId, ...]
    episode_index_start: int
    episodes_per_environment: int
    candidates_per_episode: int
    matched_probe_budget: int
    environment_specification_hash: Sha256
    analysis_plan_hash: Sha256
    calibration_hash: Sha256
    public_stream_sha256: Sha256

    @property
    def expected_row_count(self) -> int:
        return len(self.environment_ids) * self.episodes_per_environment * len(self.policy_ids)


@dataclass(frozen=True, slots=True)
class LoadedPublicErrors:
    """Small in-memory projection of a much larger verified public stream."""

    errors: dict[tuple[EvaluationEnvironmentId, PolicyId], tuple[float, ...]]
    row_count: int
    byte_count: int
    file_sha256: Sha256


def load_canonical_primary_source_plan(path: Path) -> CanonicalPrimarySourcePlan:
    """Load and self-verify the source plan frozen before outcome inspection."""

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise CanonicalPrimaryAnalysisError("Canonical primary source plan must be a mapping")
        return CanonicalPrimarySourcePlan.model_validate(raw)
    except CanonicalPrimaryAnalysisError:
        raise
    except (OSError, ValidationError, yaml.YAMLError) as error:
        raise CanonicalPrimaryAnalysisError(
            f"Could not load canonical primary source plan: {path}"
        ) from error


def load_public_classification_errors(
    path: Path,
    expectation: PublicStreamExpectation,
) -> LoadedPublicErrors:
    """Verify every public row while retaining only paired classification errors."""

    grouped: dict[tuple[EvaluationEnvironmentId, PolicyId], list[float]] = {
        (environment_id, policy_id): []
        for environment_id in expectation.environment_ids
        for policy_id in expectation.policy_ids
    }
    digest = hashlib.sha256()
    byte_count = 0
    row_count = 0
    current_candidate_hash: Sha256 | None = None
    try:
        with path.open("rb") as handle:
            for row_index, raw_line in enumerate(handle):
                digest.update(raw_line)
                byte_count += len(raw_line)
                if not raw_line.endswith(b"\n") or raw_line == b"\n":
                    raise CanonicalPrimaryAnalysisError(
                        f"Public stream row {row_index + 1} is not canonical JSONL"
                    )
                if row_index >= expectation.expected_row_count:
                    raise CanonicalPrimaryAnalysisError("Public stream contains extra rows")
                row = CanonicalPublicPolicyMetric.model_validate_json(raw_line)
                environment_index, remainder = divmod(
                    row_index,
                    expectation.episodes_per_environment * len(expectation.policy_ids),
                )
                episode_offset, policy_index = divmod(remainder, len(expectation.policy_ids))
                expected_environment = expectation.environment_ids[environment_index]
                expected_episode = expectation.episode_index_start + episode_offset
                expected_policy = expectation.policy_ids[policy_index]
                metric = row.metric
                if (
                    metric.environment_id is not expected_environment
                    or metric.episode_index != expected_episode
                    or metric.policy_id is not expected_policy
                ):
                    raise CanonicalPrimaryAnalysisError(
                        f"Public stream row {row_index + 1} violates the sealed order"
                    )
                if (
                    row.environment_specification_hash != expectation.environment_specification_hash
                    or row.analysis_plan_hash != expectation.analysis_plan_hash
                    or row.calibration_hash != expectation.calibration_hash
                    or metric.candidate_count != expectation.candidates_per_episode
                ):
                    raise CanonicalPrimaryAnalysisError(
                        f"Public stream row {row_index + 1} uses another sealed input"
                    )
                expected_selected = _expected_selected_count(
                    expected_policy,
                    expectation.matched_probe_budget,
                    expectation.candidates_per_episode,
                )
                if metric.selected_probe_count != expected_selected:
                    raise CanonicalPrimaryAnalysisError(
                        f"Public stream row {row_index + 1} violates its probe budget"
                    )
                if policy_index == 0:
                    current_candidate_hash = row.shared_candidate_hash
                elif row.shared_candidate_hash != current_candidate_hash:
                    raise CanonicalPrimaryAnalysisError(
                        f"Public stream row {row_index + 1} breaks episode pairing"
                    )
                grouped[(expected_environment, expected_policy)].append(
                    metric.final_classification_error
                )
                row_count += 1
    except CanonicalPrimaryAnalysisError:
        raise
    except (OSError, ValidationError) as error:
        raise CanonicalPrimaryAnalysisError(f"Could not verify public stream: {path}") from error
    actual_hash = digest.hexdigest()
    if row_count != expectation.expected_row_count:
        raise CanonicalPrimaryAnalysisError("Public stream row count is incomplete")
    if actual_hash != expectation.public_stream_sha256:
        raise CanonicalPrimaryAnalysisError("Public stream bytes differ from the sealed hash")
    if any(len(values) != expectation.episodes_per_environment for values in grouped.values()):
        raise CanonicalPrimaryAnalysisError("Public stream paired coverage is incomplete")
    return LoadedPublicErrors(
        errors={key: tuple(values) for key, values in grouped.items()},
        row_count=row_count,
        byte_count=byte_count,
        file_sha256=actual_hash,
    )


def analyse_canonical_primary_errors(
    errors: dict[tuple[EvaluationEnvironmentId, PolicyId], tuple[float, ...]],
    *,
    analysis: AcquisitionAnalysisSpecification,
    execution_plan_hash: Sha256,
    source_plan_hash: Sha256,
    canonical_run_manifest_hash: Sha256,
    canonical_stream_report_hash: Sha256,
    public_stream_sha256: Sha256,
    source_public_row_count: int,
) -> CanonicalPrimaryAnalysisReport:
    """Apply the frozen paired primary analysis to a verified error projection."""

    if analysis.policies.candidate is not _CANDIDATE:
        raise CanonicalPrimaryAnalysisError("Candidate policy differs from the frozen plan")
    if analysis.policies.matched_budget_comparators != _COMPARATORS:
        raise CanonicalPrimaryAnalysisError("Comparators differ from the frozen plan")
    expected_keys = {
        (environment_id, policy_id)
        for environment_id in _ENVIRONMENT_ORDER
        for policy_id in _POLICY_ORDER
    }
    if set(errors) != expected_keys:
        raise CanonicalPrimaryAnalysisError("Public error projection has incomplete coverage")
    episode_counts = {len(values) for values in errors.values()}
    if len(episode_counts) != 1:
        raise CanonicalPrimaryAnalysisError("Public error projection mixes episode counts")
    episodes_per_environment = episode_counts.pop()
    if episodes_per_environment < 1:
        raise CanonicalPrimaryAnalysisError("Public error projection contains no episodes")
    if source_public_row_count != (
        len(_ENVIRONMENT_ORDER) * len(_POLICY_ORDER) * episodes_per_environment
    ):
        raise CanonicalPrimaryAnalysisError("Public error projection row count does not reconcile")

    effects = {
        (environment_id, comparator): tuple(
            candidate - reference
            for candidate, reference in zip(
                errors[(environment_id, _CANDIDATE)],
                errors[(environment_id, comparator)],
                strict=True,
            )
        )
        for environment_id in _ENVIRONMENT_ORDER
        for comparator in _COMPARATORS
    }
    summaries = {
        (environment_id, comparator): _summarise_environment(
            errors,
            effects[(environment_id, comparator)],
            environment_id=environment_id,
            comparator=comparator,
        )
        for environment_id in _ENVIRONMENT_ORDER
        for comparator in _COMPARATORS
    }
    distributions = stratified_paired_bootstrap_distributions(effects, analysis=analysis)
    alpha = 1.0 - analysis.primary.simultaneous_confidence_level
    intervals: list[CanonicalComparatorInterval] = []
    for comparator in _COMPARATORS:
        environment_rows = tuple(
            summaries[(environment_id, comparator)]
            for environment_id in analysis.primary.held_out_environments
        )
        bootstrap = distributions[comparator]
        lower, upper = np.quantile(
            bootstrap,
            (alpha / 2.0, 1.0 - alpha / 2.0),
            method="linear",
        )
        lower_float = float(lower)
        upper_float = float(upper)
        half_width = (upper_float - lower_float) / 2.0
        worst = max(environment_rows, key=lambda row: row.mean_paired_effect)
        intervals.append(
            CanonicalComparatorInterval(
                comparator_policy_id=comparator,
                episodes_per_environment=episodes_per_environment,
                environment_effects=environment_rows,
                macro_mean_paired_effect=fmean(row.mean_paired_effect for row in environment_rows),
                confidence_level=analysis.primary.simultaneous_confidence_level,
                interval_lower=lower_float,
                interval_upper=upper_float,
                interval_half_width=half_width,
                precision_target_half_width=analysis.primary.precision_target_half_width,
                precision_target_met=(half_width <= analysis.primary.precision_target_half_width),
                bootstrap_repetitions=analysis.primary.bootstrap_repetitions,
                bootstrap_seed=analysis.primary.bootstrap_seed,
                paired_effects_hash=canonical_sha256(
                    tuple(
                        (row.environment_id, effects[(row.environment_id, comparator)])
                        for row in environment_rows
                    )
                ),
                bootstrap_distribution_hash=canonical_sha256(bootstrap.tolist()),
                simultaneous_interval_rule_met=upper_float < 0.0,
                worst_environment_id=worst.environment_id,
                worst_environment_mean_effect=worst.mean_paired_effect,
            )
        )
    frozen_intervals = tuple(intervals)
    all_rules = all(row.simultaneous_interval_rule_met for row in frozen_intervals)
    all_precision = all(row.precision_target_met for row in frozen_intervals)
    plug_in = frozen_intervals[-1]
    no_plugin_regression = all(row.mean_paired_effect <= 0.0 for row in plug_in.environment_effects)
    robust = all_rules and no_plugin_regression
    claims = _claim_assessments(all_rules=all_rules, robust=robust)
    content = {
        "schema_version": 1,
        "schema_id": "acquisition_study.canonical_primary_analysis_report.v1",
        "study_id": "reliability-aware-probing-v1",
        "split": "evaluation",
        "result_status": "canonical_primary_complete_secondary_pending",
        "execution_plan_hash": execution_plan_hash,
        "source_plan_hash": source_plan_hash,
        "canonical_run_manifest_hash": canonical_run_manifest_hash,
        "canonical_stream_report_hash": canonical_stream_report_hash,
        "public_stream_sha256": public_stream_sha256,
        "source_public_row_count": source_public_row_count,
        "candidate_policy_id": _CANDIDATE,
        "comparator_policy_ids": _COMPARATORS,
        "episodes_per_environment": episodes_per_environment,
        "matched_environment_effects": tuple(
            summaries[(EvaluationEnvironmentId.MATCHED, comparator)] for comparator in _COMPARATORS
        ),
        "held_out_primary_intervals": frozen_intervals,
        "all_simultaneous_interval_rules_met": all_rules,
        "all_precision_targets_met": all_precision,
        "no_held_out_point_regression_against_plug_in_evsi": no_plugin_regression,
        "overall_advantage_rule_met": all_rules,
        "robustness_rule_met": robust,
        "claim_assessments": claims,
        "inference_scope": "independent_episodes_from_declared_simulator_only",
        "restricted_stream_accessed": False,
        "external_model_call_count": 0,
        "sandbox_call_count": 0,
        "human_record_count": 0,
        "exact_analysis_retry_verified": True,
    }
    return CanonicalPrimaryAnalysisReport.model_validate(
        {**content, "report_hash": canonical_sha256(content)}
    )


def run_canonical_primary_analysis(
    source_plan_path: Path,
    *,
    analysis_code_revision: str,
) -> CanonicalPrimaryAnalysisManifest:
    """Verify sources, run the primary analysis twice, and publish exact artifacts."""

    source_plan = load_canonical_primary_source_plan(source_plan_path)
    project_root = source_plan_path.resolve().parents[2]
    pre_run, run_manifest, stream_report, analysis = _load_sources(
        project_root,
        source_plan,
    )
    source_plan_file_hash = _hash_path(source_plan_path, "canonical primary source plan")
    execution_content = {
        "schema_version": 1,
        "schema_id": "acquisition_study.canonical_primary_execution_plan.v1",
        "study_id": source_plan.study_id,
        "source_plan_hash": source_plan.plan_hash,
        "source_plan_file_sha256": source_plan_file_hash,
        "canonical_code_revision": run_manifest.code_revision,
        "analysis_code_revision": analysis_code_revision,
        "canonical_run_manifest_hash": run_manifest.manifest_hash,
        "canonical_stream_report_hash": stream_report.report_hash,
        "public_stream_sha256": stream_report.public_file_sha256,
        "frozen_analysis_specification_hash": analysis.plan_hash,
        "pixi_lock_sha256": source_plan.pixi_lock_sha256,
        "candidate_policy_id": analysis.policies.candidate,
        "comparator_policy_ids": analysis.policies.matched_budget_comparators,
        "held_out_environment_ids": analysis.primary.held_out_environments,
        "episodes_per_environment": stream_report.episodes_per_environment,
        "interval_method": analysis.primary.interval_method,
        "bootstrap_repetitions": analysis.primary.bootstrap_repetitions,
        "bootstrap_seed": analysis.primary.bootstrap_seed,
        "simultaneous_confidence_level": analysis.primary.simultaneous_confidence_level,
        "precision_target_half_width": analysis.primary.precision_target_half_width,
        "restricted_stream_accessed": False,
        "external_call_count": 0,
    }
    analysis_execution = CanonicalPrimaryExecutionPlan.model_validate(
        {**execution_content, "plan_hash": canonical_sha256(execution_content)}
    )
    expectation = PublicStreamExpectation(
        environment_ids=source_plan.environment_ids,
        policy_ids=source_plan.policy_ids,
        episode_index_start=source_plan.evaluation_episode_index_start,
        episodes_per_environment=source_plan.episodes_per_environment,
        candidates_per_episode=source_plan.candidates_per_episode,
        matched_probe_budget=source_plan.matched_probe_budget,
        environment_specification_hash=pre_run.environment_specification_hash,
        analysis_plan_hash=pre_run.analysis_specification_hash,
        calibration_hash=pre_run.calibration_hash,
        public_stream_sha256=source_plan.public_stream_sha256,
    )
    loaded = load_public_classification_errors(
        _project_path(project_root, source_plan.public_stream_path),
        expectation,
    )
    report = analyse_canonical_primary_errors(
        loaded.errors,
        analysis=analysis,
        execution_plan_hash=analysis_execution.plan_hash,
        source_plan_hash=source_plan.plan_hash,
        canonical_run_manifest_hash=run_manifest.manifest_hash,
        canonical_stream_report_hash=stream_report.report_hash,
        public_stream_sha256=loaded.file_sha256,
        source_public_row_count=loaded.row_count,
    )
    retry = analyse_canonical_primary_errors(
        loaded.errors,
        analysis=analysis,
        execution_plan_hash=analysis_execution.plan_hash,
        source_plan_hash=source_plan.plan_hash,
        canonical_run_manifest_hash=run_manifest.manifest_hash,
        canonical_stream_report_hash=stream_report.report_hash,
        public_stream_sha256=loaded.file_sha256,
        source_public_row_count=loaded.row_count,
    )
    if report != retry:
        raise CanonicalPrimaryAnalysisError("Exact primary-analysis retry changed")

    environment_csv = _environment_table_bytes(report)
    comparison_csv = _comparison_table_bytes(report)
    claim_csv = _claim_table_bytes(report)
    output_root = _project_path(project_root, source_plan.output_root)
    manifest_content = {
        "schema_version": 1,
        "schema_id": "acquisition_study.canonical_primary_analysis_manifest.v1",
        "study_id": source_plan.study_id,
        "source_plan_hash": source_plan.plan_hash,
        "execution_plan_file": _EXECUTION_PLAN_FILE,
        "execution_plan_file_sha256": file_sha256(immutable_json_bytes(analysis_execution)),
        "execution_plan_hash": analysis_execution.plan_hash,
        "report_file": _REPORT_FILE,
        "report_file_sha256": file_sha256(immutable_json_bytes(report)),
        "report_hash": report.report_hash,
        "environment_table_file": _ENVIRONMENT_TABLE_FILE,
        "environment_table_sha256": file_sha256(environment_csv),
        "comparison_table_file": _COMPARISON_TABLE_FILE,
        "comparison_table_sha256": file_sha256(comparison_csv),
        "claim_table_file": _CLAIM_TABLE_FILE,
        "claim_table_sha256": file_sha256(claim_csv),
        "public_stream_sha256": loaded.file_sha256,
        "source_public_row_count": loaded.row_count,
        "restricted_stream_accessed": False,
        "exact_analysis_retry_verified": True,
        "completion_status": "canonical_primary_complete_secondary_pending",
    }
    manifest = CanonicalPrimaryAnalysisManifest.model_validate(
        {**manifest_content, "manifest_hash": canonical_sha256(manifest_content)}
    )
    write_immutable_json(output_root / _EXECUTION_PLAN_FILE, analysis_execution)
    write_immutable_json(output_root / _REPORT_FILE, report)
    write_immutable_bytes(output_root / _ENVIRONMENT_TABLE_FILE, environment_csv)
    write_immutable_bytes(output_root / _COMPARISON_TABLE_FILE, comparison_csv)
    write_immutable_bytes(output_root / _CLAIM_TABLE_FILE, claim_csv)
    write_immutable_json(output_root / _MANIFEST_FILE, manifest)
    return manifest


def _load_sources(
    project_root: Path,
    source_plan: CanonicalPrimarySourcePlan,
) -> tuple[
    CanonicalPreRunManifest,
    CanonicalRunManifest,
    CanonicalStreamReport,
    AcquisitionAnalysisSpecification,
]:
    execution_path = _project_path(project_root, source_plan.canonical_execution_plan_path)
    execution = load_canonical_execution_plan(execution_path)
    pre_run = _load_model(
        _project_path(project_root, source_plan.pre_run_manifest_path),
        CanonicalPreRunManifest,
    )
    gate = _load_model(
        _project_path(project_root, source_plan.pre_run_gate_path),
        CanonicalPreRunGateReport,
    )
    run_manifest_path = _project_path(project_root, source_plan.canonical_run_manifest_path)
    run_manifest = _load_model(run_manifest_path, CanonicalRunManifest)
    stream_report_path = _project_path(project_root, source_plan.canonical_stream_report_path)
    stream_report = _load_model(stream_report_path, CanonicalStreamReport)
    _, analysis = load_acquisition_study_plan(
        _project_path(project_root, execution.environment_specification_path),
        _project_path(project_root, source_plan.analysis_specification_path),
    )
    actual = {
        "canonical_execution_plan_hash": execution.plan_hash,
        "analysis_specification_hash": analysis.plan_hash,
        "pixi_lock_sha256": _hash_path(
            _project_path(project_root, source_plan.pixi_lock_path),
            "Pixi lock",
        ),
        "pre_run_manifest_hash": pre_run.manifest_hash,
        "pre_run_manifest_file_sha256": _hash_path(
            _project_path(project_root, source_plan.pre_run_manifest_path),
            "pre-run manifest",
        ),
        "pre_run_gate_hash": gate.report_hash,
        "pre_run_gate_file_sha256": _hash_path(
            _project_path(project_root, source_plan.pre_run_gate_path),
            "pre-run gate report",
        ),
        "canonical_run_manifest_hash": run_manifest.manifest_hash,
        "canonical_run_manifest_file_sha256": _hash_path(
            run_manifest_path,
            "canonical run manifest",
        ),
        "canonical_stream_report_hash": stream_report.report_hash,
        "canonical_stream_report_file_sha256": _hash_path(
            stream_report_path,
            "canonical stream report",
        ),
        "public_stream_sha256": stream_report.public_file_sha256,
        "restricted_stream_sha256": stream_report.restricted_file_sha256,
        "canonical_code_revision": run_manifest.code_revision,
    }
    expected = {key: getattr(source_plan, key) for key in actual}
    if actual != expected:
        changed = ", ".join(key for key in actual if actual[key] != expected[key])
        raise CanonicalPrimaryAnalysisError(f"Canonical source identity changed: {changed}")
    if (
        gate.pre_run_manifest_hash != pre_run.manifest_hash
        or run_manifest.pre_run_manifest_hash != pre_run.manifest_hash
        or run_manifest.pre_run_gate_report_hash != gate.report_hash
        or run_manifest.stream_report_hash != stream_report.report_hash
        or run_manifest.stream_report_file_sha256 != source_plan.canonical_stream_report_file_sha256
        or run_manifest.completion_status != "canonical_raw_evaluation_complete_unanalysed"
        or stream_report.public_row_count != source_plan.expected_public_row_count
        or stream_report.episodes_per_environment != source_plan.episodes_per_environment
        or not stream_report.deterministic_replay_verified
    ):
        raise CanonicalPrimaryAnalysisError("Canonical source reports do not reconcile")
    return pre_run, run_manifest, stream_report, analysis


def _summarise_environment(
    errors: dict[tuple[EvaluationEnvironmentId, PolicyId], tuple[float, ...]],
    effects: tuple[float, ...],
    *,
    environment_id: EvaluationEnvironmentId,
    comparator: PolicyId,
) -> EnvironmentPairedEffect:
    candidate_errors = errors[(environment_id, _CANDIDATE)]
    comparator_errors = errors[(environment_id, comparator)]
    return EnvironmentPairedEffect(
        environment_id=environment_id,
        environment_role=(
            EnvironmentRole.MATCHED_SANITY_CHECK
            if environment_id is EvaluationEnvironmentId.MATCHED
            else EnvironmentRole.HELD_OUT_MISMATCH
        ),
        comparator_policy_id=comparator,
        episode_count=len(effects),
        candidate_mean_classification_error=fmean(candidate_errors),
        comparator_mean_classification_error=fmean(comparator_errors),
        mean_paired_effect=fmean(effects),
        candidate_better_episode_count=sum(effect < 0.0 for effect in effects),
        tied_episode_count=sum(effect == 0.0 for effect in effects),
        comparator_better_episode_count=sum(effect > 0.0 for effect in effects),
        paired_effects_hash=canonical_sha256(effects),
    )


def _claim_assessments(
    *,
    all_rules: bool,
    robust: bool,
) -> tuple[CanonicalClaimAssessment, ...]:
    return (
        CanonicalClaimAssessment(
            claim_id="all_comparator_advantage",
            statement=(
                "The candidate has lower held-out prediction error than all three "
                "matched-budget comparators."
            ),
            status=(
                "supported_within_declared_simulator"
                if all_rules
                else "not_supported_by_primary_rule"
            ),
            basis="All three multiplicity-adjusted interval upper bounds must be below zero.",
        ),
        CanonicalClaimAssessment(
            claim_id="declared_mismatch_robustness",
            statement="The candidate remains useful across the six declared mismatch environments.",
            status=(
                "supported_within_declared_simulator" if robust else "not_supported_by_primary_rule"
            ),
            basis=(
                "The all-comparator rule must pass, with no held-out point regression "
                "against plug-in EVSI."
            ),
        ),
        CanonicalClaimAssessment(
            claim_id="human_learning",
            statement="The method improves human learning.",
            status="prohibited_by_design",
            basis="The study contains no human learner records.",
        ),
        CanonicalClaimAssessment(
            claim_id="tutoring_efficacy",
            statement="The method improves live tutoring.",
            status="prohibited_by_design",
            basis="The study evaluates simulated prediction, not tutoring outcomes.",
        ),
        CanonicalClaimAssessment(
            claim_id="cognitive_offloading",
            statement="The method reduces cognitive offloading.",
            status="prohibited_by_design",
            basis="The study does not measure learner cognition or behaviour.",
        ),
        CanonicalClaimAssessment(
            claim_id="deployed_cost_saving",
            statement="The method saves money in a deployed tutoring service.",
            status="prohibited_by_design",
            basis="The study uses a fixed simulated probe budget and makes no deployment calls.",
        ),
        CanonicalClaimAssessment(
            claim_id="state_of_the_art",
            statement="The method is state of the art.",
            status="prohibited_by_design",
            basis=(
                "The experiment does not compare against the full published field on a "
                "shared benchmark."
            ),
        ),
    )


def _environment_table_bytes(report: CanonicalPrimaryAnalysisReport) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        (
            "environment_id",
            "environment_role",
            "candidate_policy_id",
            "comparator_policy_id",
            "episode_count",
            "candidate_mean_classification_error",
            "comparator_mean_classification_error",
            "candidate_minus_comparator_error",
            "candidate_better_episode_count",
            "tied_episode_count",
            "comparator_better_episode_count",
        )
    )
    rows = (
        *report.matched_environment_effects,
        *(
            effect
            for interval in report.held_out_primary_intervals
            for effect in interval.environment_effects
        ),
    )
    rows = tuple(
        sorted(
            rows,
            key=lambda row: (
                _ENVIRONMENT_ORDER.index(row.environment_id),
                _COMPARATORS.index(row.comparator_policy_id),
            ),
        )
    )
    for row in rows:
        writer.writerow(
            (
                row.environment_id.value,
                row.environment_role.value,
                row.candidate_policy_id.value,
                row.comparator_policy_id.value,
                row.episode_count,
                _float_text(row.candidate_mean_classification_error),
                _float_text(row.comparator_mean_classification_error),
                _float_text(row.mean_paired_effect),
                row.candidate_better_episode_count,
                row.tied_episode_count,
                row.comparator_better_episode_count,
            )
        )
    return output.getvalue().encode("utf-8")


def _comparison_table_bytes(report: CanonicalPrimaryAnalysisReport) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        (
            "comparator_policy_id",
            "held_out_macro_candidate_minus_comparator_error",
            "simultaneous_interval_lower",
            "simultaneous_interval_upper",
            "confidence_level",
            "interval_half_width",
            "precision_target_half_width",
            "precision_target_met",
            "comparison_rule_met",
            "worst_environment_id",
            "worst_environment_candidate_minus_comparator_error",
        )
    )
    for row in report.held_out_primary_intervals:
        writer.writerow(
            (
                row.comparator_policy_id.value,
                _float_text(row.macro_mean_paired_effect),
                _float_text(row.interval_lower),
                _float_text(row.interval_upper),
                _float_text(row.confidence_level),
                _float_text(row.interval_half_width),
                _float_text(row.precision_target_half_width),
                str(row.precision_target_met).lower(),
                str(row.simultaneous_interval_rule_met).lower(),
                row.worst_environment_id.value,
                _float_text(row.worst_environment_mean_effect),
            )
        )
    return output.getvalue().encode("utf-8")


def _claim_table_bytes(report: CanonicalPrimaryAnalysisReport) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(("claim_id", "statement", "status", "basis"))
    for row in report.claim_assessments:
        writer.writerow((row.claim_id, row.statement, row.status, row.basis))
    return output.getvalue().encode("utf-8")


def _expected_selected_count(policy_id: PolicyId, matched_budget: int, candidates: int) -> int:
    if policy_id is PolicyId.NEVER_PROBE:
        return 0
    if policy_id is PolicyId.ALWAYS_PROBE_BOUNDED:
        return candidates
    return matched_budget


def _float_text(value: float) -> str:
    return format(value, ".17g")


def _load_model[ModelT: ContractModel](path: Path, model: type[ModelT]) -> ModelT:
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        raise CanonicalPrimaryAnalysisError(f"Could not load canonical source: {path}") from error


def _hash_path(path: Path, label: str) -> Sha256:
    try:
        return file_sha256(path.read_bytes())
    except OSError as error:
        raise CanonicalPrimaryAnalysisError(f"Could not hash {label}: {path}") from error


def _project_path(project_root: Path, relative_path: str) -> Path:
    _validate_relative_path(relative_path)
    resolved = (project_root / relative_path).resolve()
    if not resolved.is_relative_to(project_root.resolve()):
        raise CanonicalPrimaryAnalysisError(f"Path leaves the project root: {relative_path}")
    return resolved


def _validate_relative_path(value: str) -> None:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value != path.as_posix():
        raise ValueError(f"Path must be a normal relative POSIX path: {value}")
