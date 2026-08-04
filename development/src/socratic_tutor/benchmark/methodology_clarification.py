"""Pre-criterion statement of the external benchmark's scope and claims."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from socratic_tutor.benchmark.analysis_spec import (
    AnalysisSpecification,
    analysis_specification_hash,
)
from socratic_tutor.benchmark.artifacts import write_immutable_json
from socratic_tutor.benchmark.calibration import UncalibratedDecisionReport
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.evaluator.scoring import PrimaryMetric
from socratic_tutor.benchmark.external_protocol import ExternalModelExecutionProtocol
from socratic_tutor.benchmark.hashing import canonical_sha256, model_content_hash
from socratic_tutor.benchmark.public_rating import PublicAnswerRatingReport
from socratic_tutor.contracts import ContractModel, EvidenceCategory


class TrackerCategoryDelta(ContractModel):
    """One fixed category update used by the authored simple tracker."""

    category: EvidenceCategory
    delta: float = Field(ge=-1.0, le=1.0)


class MethodologyClarificationPlan(ContractModel):
    """Reviewed identities and wording frozen before criterion access."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.methodology_clarification_plan.v1"] = (
        "benchmark.methodology_clarification_plan.v1"
    )
    clarification_version: Literal["pre-criterion-v1"]
    decision_owner: str = Field(min_length=1)
    frozen_on: date
    external_protocol_hash: Sha256
    analysis_specification_hash: Sha256
    calibration_decision_hash: Sha256
    public_rating_report_hash: Sha256
    source_manifest_hash: Sha256


class MethodologyClarification(ContractModel):
    """Content-addressed boundary on design, interpretation, and claims."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.methodology_clarification.v1"] = (
        "benchmark.methodology_clarification.v1"
    )
    clarification_version: Literal["pre-criterion-v1"]
    decision_owner: str = Field(min_length=1)
    frozen_on: date
    external_protocol_hash: Sha256
    analysis_specification_hash: Sha256
    calibration_decision_hash: Sha256
    public_rating_report_hash: Sha256
    source_manifest_hash: Sha256
    population: Literal["one_pinned_external_evaluation_model_route"]
    provider_id: str = Field(min_length=1)
    requested_model_id: str = Field(min_length=1)
    authored_case_count: Literal[24]
    model_runs_per_case: Literal[1]
    predictor_inputs: tuple[
        Literal["blinded_public_answer_category"],
        Literal["isolated_executable_probe_outcome"],
        Literal["isolated_contrastive_control_outcome"],
    ]
    criterion: Literal["binary_pass_or_fail_on_a_later_authored_executable_task"]
    primary_estimand: Literal["mean_paired_classification_error_difference_by_case"]
    interpretation: Literal["associational_prediction_on_this_authored_corpus_not_causal"]
    isolated_definition: Literal[
        "fresh_model_context_without_history_from_any_other_benchmark_channel"
    ]
    withheld_criterion_definition: Literal[
        "hidden_from_tracker_and_decision_process_not_claimed_absent_from_model_pretraining"
    ]
    tracker_version: Literal["simple-v1"]
    tracker_kind: Literal["authored_additive_heuristic"]
    tracker_prior: float = Field(ge=0.0, le=1.0)
    tracker_category_deltas: tuple[TrackerCategoryDelta, ...] = Field(min_length=6, max_length=6)
    evidence_confidence_consumed: Literal[False] = False
    policy_threshold: float = Field(ge=0.0, le=1.0)
    calibrated_probability_claim_allowed: Literal[False] = False
    prohibited_claims: tuple[
        Literal["latent_learner_state_measurement"],
        Literal["human_learning_improvement"],
        Literal["human_student_generalisation"],
        Literal["calibrated_mastery_probability"],
        Literal["tutoring_efficacy"],
        Literal["reduced_cognitive_offloading"],
    ]
    required_before_criterion_access: Literal[True] = True
    sufficient_to_unlock_criterion_access: Literal[False] = False
    clarification_hash: Sha256

    @model_validator(mode="after")
    def validate_clarification(self) -> MethodologyClarification:
        if self.tracker_prior != 0.5:
            raise ValueError("Clarification v1 requires the frozen 0.5 tracker prior")
        if self.policy_threshold != 0.6:
            raise ValueError("Clarification v1 requires the frozen 0.60 policy threshold")
        categories = tuple(item.category for item in self.tracker_category_deltas)
        if len(set(categories)) != len(EvidenceCategory) or set(categories) != set(
            EvidenceCategory
        ):
            raise ValueError("Clarification must declare one delta per evidence category")
        if self.clarification_hash != model_content_hash(self, exclude={"clarification_hash"}):
            raise ValueError("Methodology clarification hash does not match its content")
        return self


def load_methodology_clarification_plan(path: Path) -> MethodologyClarificationPlan:
    """Load the reviewed clarification identities from YAML."""

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"Could not read methodology clarification plan: {path}") from error
    return MethodologyClarificationPlan.model_validate(raw)


def load_public_answer_rating_report(path: Path) -> PublicAnswerRatingReport:
    """Load and validate the rating report committed before criterion access."""

    return _load_json_model(path, PublicAnswerRatingReport)


def freeze_methodology_clarification(
    *,
    plan: MethodologyClarificationPlan,
    protocol: ExternalModelExecutionProtocol,
    analysis_specification: AnalysisSpecification,
    calibration_report: UncalibratedDecisionReport,
    public_rating_report: PublicAnswerRatingReport,
    output_path: Path,
) -> MethodologyClarification:
    """Validate frozen inputs and publish the exact pre-criterion claim boundary."""

    specification_hash = analysis_specification_hash(analysis_specification)
    _validate_bound_inputs(
        plan=plan,
        protocol=protocol,
        specification_hash=specification_hash,
        calibration_report=calibration_report,
        public_rating_report=public_rating_report,
    )
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.methodology_clarification.v1",
        "clarification_version": plan.clarification_version,
        "decision_owner": plan.decision_owner,
        "frozen_on": plan.frozen_on,
        "external_protocol_hash": protocol.protocol_hash,
        "analysis_specification_hash": specification_hash,
        "calibration_decision_hash": calibration_report.decision.decision_hash,
        "public_rating_report_hash": public_rating_report.report_hash,
        "source_manifest_hash": protocol.source_manifest_hash,
        "population": "one_pinned_external_evaluation_model_route",
        "provider_id": protocol.provider_id,
        "requested_model_id": protocol.requested_model_id,
        "authored_case_count": public_rating_report.item_count,
        "model_runs_per_case": protocol.repeats_per_case,
        "predictor_inputs": (
            "blinded_public_answer_category",
            "isolated_executable_probe_outcome",
            "isolated_contrastive_control_outcome",
        ),
        "criterion": "binary_pass_or_fail_on_a_later_authored_executable_task",
        "primary_estimand": "mean_paired_classification_error_difference_by_case",
        "interpretation": "associational_prediction_on_this_authored_corpus_not_causal",
        "isolated_definition": (
            "fresh_model_context_without_history_from_any_other_benchmark_channel"
        ),
        "withheld_criterion_definition": (
            "hidden_from_tracker_and_decision_process_not_claimed_absent_from_model_pretraining"
        ),
        "tracker_version": calibration_report.decision.tracker_version,
        "tracker_kind": "authored_additive_heuristic",
        "tracker_prior": 0.5,
        "tracker_category_deltas": (
            {"category": "correct", "delta": 0.2},
            {"category": "incorrect", "delta": -0.2},
            {"category": "misconception", "delta": -0.2},
            {"category": "conflicting", "delta": -0.1},
            {"category": "uncertain", "delta": 0.0},
            {"category": "empty", "delta": 0.0},
        ),
        "evidence_confidence_consumed": False,
        "policy_threshold": calibration_report.decision.policy_threshold,
        "calibrated_probability_claim_allowed": False,
        "prohibited_claims": (
            "latent_learner_state_measurement",
            "human_learning_improvement",
            "human_student_generalisation",
            "calibrated_mastery_probability",
            "tutoring_efficacy",
            "reduced_cognitive_offloading",
        ),
        "required_before_criterion_access": True,
        "sufficient_to_unlock_criterion_access": False,
    }
    clarification = MethodologyClarification.model_validate(
        {**content, "clarification_hash": canonical_sha256(content)}
    )
    write_immutable_json(output_path, clarification)
    return clarification


def _validate_bound_inputs(
    *,
    plan: MethodologyClarificationPlan,
    protocol: ExternalModelExecutionProtocol,
    specification_hash: Sha256,
    calibration_report: UncalibratedDecisionReport,
    public_rating_report: PublicAnswerRatingReport,
) -> None:
    expected = {
        "external protocol": (plan.external_protocol_hash, protocol.protocol_hash),
        "analysis specification": (plan.analysis_specification_hash, specification_hash),
        "calibration decision": (
            plan.calibration_decision_hash,
            calibration_report.decision.decision_hash,
        ),
        "public rating report": (
            plan.public_rating_report_hash,
            public_rating_report.report_hash,
        ),
        "source manifest": (plan.source_manifest_hash, protocol.source_manifest_hash),
    }
    for name, (declared, actual) in expected.items():
        if declared != actual:
            raise ValueError(f"Clarification plan belongs to a different {name}")
    if protocol.analysis_specification_hash != specification_hash:
        raise ValueError("External protocol belongs to a different analysis specification")
    if calibration_report.analysis_specification_hash != specification_hash:
        raise ValueError("Calibration report belongs to a different analysis specification")
    if calibration_report.decision.primary_metric is not (
        PrimaryMetric.PAIRED_CLASSIFICATION_ERROR_DIFFERENCE
    ):
        raise ValueError("Clarification requires the frozen binary classification-error metric")
    if protocol.repeats_per_case != 1:
        raise ValueError("Clarification v1 requires one model run per case")
    if public_rating_report.item_count != 24:
        raise ValueError("Clarification v1 requires the complete 24-case public rating report")
    if calibration_report.decision.tracker_version != "simple-v1":
        raise ValueError("Clarification v1 requires tracker simple-v1")
    if calibration_report.decision.policy_threshold != 0.6:
        raise ValueError("Clarification v1 requires the frozen 0.60 policy threshold")


def _load_json_model[ModelT: ContractModel](path: Path, model: type[ModelT]) -> ModelT:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read JSON artifact: {path}") from error
    return model.model_validate(raw)
