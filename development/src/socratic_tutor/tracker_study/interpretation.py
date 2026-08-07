"""Claim-bounded interpretation of the canonical tracker-study result."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError, model_validator

from socratic_tutor.benchmark.artifacts import write_immutable_json
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import file_sha256, model_content_hash
from socratic_tutor.contracts import ContractModel
from socratic_tutor.tracker_study.analysis import (
    TrackerCanonicalAnalysisPlan,
    TrackerCanonicalAnalysisReport,
)
from socratic_tutor.tracker_study.canonical import (
    CanonicalExecutionPlan,
    load_canonical_execution_plan,
)
from socratic_tutor.tracker_study.runtime import (
    TrackerDevelopmentRuntimePlan,
    TrackerDevelopmentRuntimeReport,
)
from socratic_tutor.tracker_study.sensitivity import (
    TrackerDevelopmentSensitivityPlan,
    TrackerDevelopmentSensitivityReport,
)
from socratic_tutor.tracker_study.simulation import CanonicalStressMatrixManifest


class TrackerInterpretationError(ValueError):
    """Canonical tracker sources do not support one coherent interpretation."""


class TrackerCanonicalInterpretationPlan(ContractModel):
    schema_version: Literal[1] = 1
    schema_id: Literal["tracker_study.canonical_interpretation_plan.v1"] = (
        "tracker_study.canonical_interpretation_plan.v1"
    )
    study_id: str
    run_id: str
    canonical_execution_plan_hash: Sha256
    canonical_simulation_manifest_hash: Sha256
    canonical_analysis_plan_hash: Sha256
    canonical_analysis_report_hash: Sha256
    development_sensitivity_plan_hash: Sha256
    development_sensitivity_report_hash: Sha256
    development_runtime_plan_hash: Sha256
    development_runtime_report_hash: Sha256
    interpretation_code_revision: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    pixi_lock_sha256: Sha256
    created_at_utc: datetime
    network_call_count: Literal[0] = 0
    sandbox_call_count: Literal[0] = 0
    human_record_count: Literal[0] = 0
    plan_hash: Sha256

    @model_validator(mode="after")
    def validate_plan(self) -> TrackerCanonicalInterpretationPlan:
        _require_utc(self.created_at_utc)
        if self.plan_hash != model_content_hash(self, exclude={"plan_hash"}):
            raise ValueError("Tracker interpretation plan hash does not match its content")
        return self


class TrackerCanonicalInterpretationReport(ContractModel):
    schema_version: Literal[1] = 1
    schema_id: Literal["tracker_study.canonical_interpretation_report.v1"] = (
        "tracker_study.canonical_interpretation_report.v1"
    )
    study_id: str
    run_id: str
    interpretation_plan_hash: Sha256
    canonical_episode_count_per_condition: int = Field(ge=1)
    candidate_tracker: Literal["bounded_channel_aware"] = "bounded_channel_aware"
    reference_tracker: Literal["channel_aware_virtual_evidence"] = "channel_aware_virtual_evidence"
    adverse_brier_mean_difference: float = Field(ge=-1.0, le=1.0)
    adverse_brier_interval_lower: float = Field(ge=-1.0, le=1.0)
    adverse_brier_interval_upper: float = Field(ge=-1.0, le=1.0)
    secondary_sign_swap_p_value: float = Field(ge=0.0, le=1.0)
    clean_brier_mean_degradation: float = Field(ge=-1.0, le=1.0)
    clean_brier_interval_lower: float = Field(ge=-1.0, le=1.0)
    clean_brier_interval_upper: float = Field(ge=-1.0, le=1.0)
    clean_brier_maximum_allowed_degradation: float = Field(ge=0.0, le=1.0)
    primary_decision_status: Literal[
        "robust_under_declared_simulator",
        "not_robust_primary_interval",
        "not_robust_clean_guardrail",
        "not_robust_primary_and_clean_guardrail",
    ]
    plain_result: str = Field(min_length=1)
    trade_off: str = Field(min_length=1)
    mathematical_status: Literal[
        "project_specific_bounded_evidence_design_over_established_bayesian_odds"
    ] = "project_specific_bounded_evidence_design_over_established_bayesian_odds"
    sensitivity_role: Literal["development_only_no_parameter_selection"] = (
        "development_only_no_parameter_selection"
    )
    learned_trust_extension_status: Literal[
        "omitted_to_protect_time_box_and_avoid_post_test_model_selection"
    ] = "omitted_to_protect_time_box_and_avoid_post_test_model_selection"
    inference_scope: Literal["declared_glass_box_simulator_only"] = (
        "declared_glass_box_simulator_only"
    )
    limitations: tuple[str, ...] = Field(min_length=6, max_length=6)
    simulator_robustness_claim_allowed: Literal[True] = True
    human_learning_claim_allowed: Literal[False] = False
    tutoring_efficacy_claim_allowed: Literal[False] = False
    reduced_cognitive_offloading_claim_allowed: Literal[False] = False
    cognitive_bandwidth_validity_claim_allowed: Literal[False] = False
    completed_at_utc: datetime
    report_hash: Sha256

    @model_validator(mode="after")
    def validate_report(self) -> TrackerCanonicalInterpretationReport:
        _require_utc(self.completed_at_utc)
        if self.adverse_brier_interval_lower > self.adverse_brier_interval_upper:
            raise ValueError("Adverse Brier interval bounds are reversed")
        if self.clean_brier_interval_lower > self.clean_brier_interval_upper:
            raise ValueError("Clean Brier interval bounds are reversed")
        expected = _decision_status(
            primary_passed=self.adverse_brier_interval_upper < 0.0,
            clean_passed=(
                self.clean_brier_interval_upper <= self.clean_brier_maximum_allowed_degradation
            ),
        )
        if self.primary_decision_status != expected:
            raise ValueError("Interpretation differs from the prespecified decision rule")
        if self.report_hash != model_content_hash(self, exclude={"report_hash"}):
            raise ValueError("Tracker interpretation report hash does not match its content")
        return self


def run_canonical_interpretation(
    *,
    execution_plan_path: Path,
    canonical_root: Path,
    sensitivity_root: Path,
    runtime_root: Path,
    pixi_lock_path: Path,
    output_root: Path,
    interpretation_code_revision: str,
    created_at_utc: datetime | None = None,
) -> TrackerCanonicalInterpretationReport:
    """Reconcile source records and publish the narrow result wording they support."""

    execution = load_canonical_execution_plan(execution_plan_path)
    simulation = _load(
        canonical_root / "canonical_stress_matrix_manifest.json",
        CanonicalStressMatrixManifest,
    )
    analysis_plan = _load(
        canonical_root / "canonical_analysis_plan.json",
        TrackerCanonicalAnalysisPlan,
    )
    analysis = _load(
        canonical_root / "canonical_analysis_report.json",
        TrackerCanonicalAnalysisReport,
    )
    sensitivity_plan = _load(
        sensitivity_root / "development_sensitivity_plan.json",
        TrackerDevelopmentSensitivityPlan,
    )
    sensitivity = _load(
        sensitivity_root / "development_sensitivity_report.json",
        TrackerDevelopmentSensitivityReport,
    )
    runtime_plan = _load(
        runtime_root / "development_runtime_plan.json",
        TrackerDevelopmentRuntimePlan,
    )
    runtime = _load(
        runtime_root / "development_runtime_report.json",
        TrackerDevelopmentRuntimeReport,
    )
    _validate_lineage(
        execution=execution,
        simulation=simulation,
        analysis_plan=analysis_plan,
        analysis=analysis,
        sensitivity_plan=sensitivity_plan,
        sensitivity=sensitivity,
        runtime_plan=runtime_plan,
        runtime=runtime,
    )
    try:
        lock_hash = file_sha256(pixi_lock_path.read_bytes())
    except OSError as error:
        raise TrackerInterpretationError("Could not hash the interpretation environment") from error
    completed_at = created_at_utc or datetime.now(UTC)
    _require_utc(completed_at)
    plan_content = {
        "schema_version": 1,
        "schema_id": "tracker_study.canonical_interpretation_plan.v1",
        "study_id": analysis.study_id,
        "run_id": analysis.run_id,
        "canonical_execution_plan_hash": execution.plan_hash,
        "canonical_simulation_manifest_hash": simulation.manifest_hash,
        "canonical_analysis_plan_hash": analysis_plan.plan_hash,
        "canonical_analysis_report_hash": analysis.report_hash,
        "development_sensitivity_plan_hash": sensitivity_plan.plan_hash,
        "development_sensitivity_report_hash": sensitivity.report_hash,
        "development_runtime_plan_hash": runtime_plan.plan_hash,
        "development_runtime_report_hash": runtime.report_hash,
        "interpretation_code_revision": interpretation_code_revision,
        "pixi_lock_sha256": lock_hash,
        "created_at_utc": completed_at,
        "network_call_count": 0,
        "sandbox_call_count": 0,
        "human_record_count": 0,
    }
    plan_draft = TrackerCanonicalInterpretationPlan.model_construct(
        _fields_set=set(plan_content),
        **plan_content,
        plan_hash="0" * 64,
    )
    plan = TrackerCanonicalInterpretationPlan.model_validate(
        {**plan_content, "plan_hash": model_content_hash(plan_draft, exclude={"plan_hash"})}
    )
    existing_path = output_root / "canonical_interpretation_report.json"
    if existing_path.exists():
        existing = _load(existing_path, TrackerCanonicalInterpretationReport)
        if existing.interpretation_plan_hash != plan.plan_hash:
            existing_plan = _load(
                output_root / "canonical_interpretation_plan.json",
                TrackerCanonicalInterpretationPlan,
            )
            if _plan_sources(existing_plan) != _plan_sources(plan):
                raise TrackerInterpretationError(
                    "Existing interpretation belongs to different source reports"
                )
        return existing
    write_immutable_json(output_root / "canonical_interpretation_plan.json", plan)

    primary = analysis.primary_adverse_brier
    clean = analysis.clean_brier_guardrail
    report_content = {
        "schema_version": 1,
        "schema_id": "tracker_study.canonical_interpretation_report.v1",
        "study_id": analysis.study_id,
        "run_id": analysis.run_id,
        "interpretation_plan_hash": plan.plan_hash,
        "canonical_episode_count_per_condition": primary.episode_count,
        "candidate_tracker": "bounded_channel_aware",
        "reference_tracker": "channel_aware_virtual_evidence",
        "adverse_brier_mean_difference": primary.mean_effect,
        "adverse_brier_interval_lower": primary.interval_lower,
        "adverse_brier_interval_upper": primary.interval_upper,
        "secondary_sign_swap_p_value": primary.two_sided_p_value,
        "clean_brier_mean_degradation": clean.mean_effect,
        "clean_brier_interval_lower": clean.interval_lower,
        "clean_brier_interval_upper": clean.interval_upper,
        "clean_brier_maximum_allowed_degradation": analysis.maximum_clean_brier_degradation,
        "primary_decision_status": analysis.primary_decision_status,
        "plain_result": _plain_result(analysis),
        "trade_off": _trade_off(analysis),
        "mathematical_status": (
            "project_specific_bounded_evidence_design_over_established_bayesian_odds"
        ),
        "sensitivity_role": "development_only_no_parameter_selection",
        "learned_trust_extension_status": (
            "omitted_to_protect_time_box_and_avoid_post_test_model_selection"
        ),
        "inference_scope": "declared_glass_box_simulator_only",
        "limitations": (
            "The observation probabilities were authored, not estimated from learners.",
            "Mastery is represented by one binary simulated state.",
            "The trackers assume the declared channel likelihoods and update order.",
            "The episodes describe repeated draws from this simulator, not a student population.",
            "The runtime comparison comes from one local machine and excludes tutor generation.",
            "No post-test parameter fitting or learned trust-weight extension was performed.",
        ),
        "simulator_robustness_claim_allowed": True,
        "human_learning_claim_allowed": False,
        "tutoring_efficacy_claim_allowed": False,
        "reduced_cognitive_offloading_claim_allowed": False,
        "cognitive_bandwidth_validity_claim_allowed": False,
        "completed_at_utc": completed_at,
    }
    report_draft = TrackerCanonicalInterpretationReport.model_construct(
        _fields_set=set(report_content),
        **report_content,
        report_hash="0" * 64,
    )
    report = TrackerCanonicalInterpretationReport.model_validate(
        {
            **report_content,
            "report_hash": model_content_hash(report_draft, exclude={"report_hash"}),
        }
    )
    write_immutable_json(existing_path, report)
    return report


def _validate_lineage(
    *,
    execution: CanonicalExecutionPlan,
    simulation: CanonicalStressMatrixManifest,
    analysis_plan: TrackerCanonicalAnalysisPlan,
    analysis: TrackerCanonicalAnalysisReport,
    sensitivity_plan: TrackerDevelopmentSensitivityPlan,
    sensitivity: TrackerDevelopmentSensitivityReport,
    runtime_plan: TrackerDevelopmentRuntimePlan,
    runtime: TrackerDevelopmentRuntimeReport,
) -> None:
    if simulation.canonical_execution_plan_hash != execution.plan_hash:
        raise TrackerInterpretationError("Simulation belongs to another execution plan")
    if analysis_plan.canonical_execution_plan_hash != execution.plan_hash:
        raise TrackerInterpretationError("Analysis belongs to another execution plan")
    if analysis_plan.simulation_manifest_hash != simulation.manifest_hash:
        raise TrackerInterpretationError("Analysis belongs to another simulation")
    if analysis.analysis_plan_hash != analysis_plan.plan_hash:
        raise TrackerInterpretationError("Canonical report belongs to another analysis plan")
    episode_counts = {
        execution.episodes_per_condition,
        simulation.episodes_per_condition,
        analysis_plan.episode_count_per_condition,
        analysis.primary_adverse_brier.episode_count,
        analysis.clean_brier_guardrail.episode_count,
    }
    if len(episode_counts) != 1:
        raise TrackerInterpretationError("Canonical sources use different episode counts")
    if sensitivity.sensitivity_plan_hash != sensitivity_plan.plan_hash:
        raise TrackerInterpretationError("Sensitivity report belongs to another plan")
    if runtime.runtime_plan_hash != runtime_plan.plan_hash:
        raise TrackerInterpretationError("Runtime report belongs to another plan")
    hashes = {
        simulation.configuration_hash,
        analysis_plan.configuration_hash,
        sensitivity_plan.configuration_hash,
        runtime_plan.configuration_hash,
        execution.configuration_hash,
    }
    if len(hashes) != 1:
        raise TrackerInterpretationError("Tracker sources use different configurations")
    analysis_hashes = {
        analysis_plan.analysis_specification_hash,
        sensitivity_plan.analysis_specification_hash,
        runtime_plan.analysis_specification_hash,
        execution.analysis_specification_hash,
    }
    if len(analysis_hashes) != 1:
        raise TrackerInterpretationError("Tracker sources use different analysis specifications")


def _plain_result(analysis: TrackerCanonicalAnalysisReport) -> str:
    primary = analysis.primary_adverse_brier
    if analysis.primary_decision_status == "robust_under_declared_simulator":
        return (
            "Across the five difficult evidence conditions, the bounded tracker had lower "
            f"mean Brier error than ordinary channel-aware Bayes by {-primary.mean_effect:.4f}; "
            f"the 95% paired interval was [{primary.interval_lower:.4f}, "
            f"{primary.interval_upper:.4f}]. It passed the prespecified simulator rule."
        )
    return (
        "The bounded tracker did not pass both parts of the prespecified simulator rule; "
        f"its adverse-condition Brier difference was {primary.mean_effect:.4f}, with a 95% "
        f"paired interval of [{primary.interval_lower:.4f}, {primary.interval_upper:.4f}]."
    )


def _trade_off(analysis: TrackerCanonicalAnalysisReport) -> str:
    clean = analysis.clean_brier_guardrail
    return (
        "Bounding evidence reduced the damage caused by unreliable evidence, but changed "
        f"clean-evidence Brier error by {clean.mean_effect:.4f}; the upper 95% bound was "
        f"{clean.interval_upper:.4f}, against the fixed "
        f"{analysis.maximum_clean_brier_degradation:.4f} "
        "maximum allowed degradation."
    )


def _decision_status(*, primary_passed: bool, clean_passed: bool) -> str:
    if primary_passed and clean_passed:
        return "robust_under_declared_simulator"
    if not primary_passed and not clean_passed:
        return "not_robust_primary_and_clean_guardrail"
    if not primary_passed:
        return "not_robust_primary_interval"
    return "not_robust_clean_guardrail"


def _plan_sources(plan: TrackerCanonicalInterpretationPlan) -> tuple[Sha256, ...]:
    return (
        plan.canonical_execution_plan_hash,
        plan.canonical_simulation_manifest_hash,
        plan.canonical_analysis_plan_hash,
        plan.canonical_analysis_report_hash,
        plan.development_sensitivity_plan_hash,
        plan.development_sensitivity_report_hash,
        plan.development_runtime_plan_hash,
        plan.development_runtime_report_hash,
        plan.pixi_lock_sha256,
        plan.interpretation_code_revision,
    )


def _load[ModelT: ContractModel](path: Path, model: type[ModelT]) -> ModelT:
    try:
        return model.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValidationError) as error:
        raise TrackerInterpretationError(f"Could not verify tracker source: {path}") from error


def _require_utc(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("Tracker interpretation times must be timezone-aware UTC")
