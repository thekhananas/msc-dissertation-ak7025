"""Post-hoc sensitivity analysis for verified executable-probe scoring defects."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, ValidationError, model_validator

from socratic_tutor.benchmark.artifacts import write_immutable_json
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.external_seal import ExternalEvidenceExecutionArtifact
from socratic_tutor.benchmark.failure_review import (
    FailureReviewLabel,
    FailureReviewRecordReport,
)
from socratic_tutor.benchmark.hashing import file_sha256, model_content_hash
from socratic_tutor.benchmark.primary_analysis import PrimaryAnalysisReport
from socratic_tutor.benchmark.statistics import (
    McNemarResult,
    PairedCaseInference,
    exact_mcnemar,
    paired_case_inference,
)
from socratic_tutor.contracts import ContractModel


class ProbeCorrectionSensitivityError(ValueError):
    """Correction inputs do not support the declared post-hoc analysis."""


class ProbeCorrectionDirective(ContractModel):
    """One reviewed correction to an observed probe decision."""

    case_id: str = Field(min_length=1)
    observed_probe_decision: bool
    corrected_probe_decision: bool
    correction_basis: Literal["correct_tuple_rejected_as_yaml_list"]
    failure_review_record_hash: Sha256
    evidence_response_hash: Sha256
    evidence_execution_artifact_hash: Sha256
    evidence_test_sha256: Sha256

    @model_validator(mode="after")
    def validate_directive(self) -> ProbeCorrectionDirective:
        if self.observed_probe_decision is self.corrected_probe_decision:
            raise ValueError("Probe correction must change the observed decision")
        return self


class ProbeCorrectionSpecification(ContractModel):
    """Post-reveal correction claims and immutable source identities."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.probe_correction_specification.v1"] = (
        "benchmark.probe_correction_specification.v1"
    )
    benchmark_version: Literal["v1"] = "v1"
    status: Literal["post_reveal_error_correction"] = "post_reveal_error_correction"
    source_primary_report_hash: Sha256
    source_failure_review_report_hash: Sha256
    corrections: tuple[ProbeCorrectionDirective, ...] = Field(min_length=1)
    interpretation_role: Literal["post_hoc_sensitivity_cannot_replace_sealed_primary"] = (
        "post_hoc_sensitivity_cannot_replace_sealed_primary"
    )

    @model_validator(mode="after")
    def validate_specification(self) -> ProbeCorrectionSpecification:
        case_ids = [correction.case_id for correction in self.corrections]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("Probe corrections contain duplicate case identities")
        return self


class ProbeCorrectionAnalysisPlan(ContractModel):
    """Bound inputs and software identity for one correction analysis."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.probe_correction_analysis_plan.v1"] = (
        "benchmark.probe_correction_analysis_plan.v1"
    )
    benchmark_version: Literal["v1"] = "v1"
    run_id: str = Field(min_length=1)
    correction_specification_hash: Sha256
    primary_report_hash: Sha256
    failure_review_report_hash: Sha256
    method: Literal["paired_binary_probe_decision_correction_v1"] = (
        "paired_binary_probe_decision_correction_v1"
    )
    analysis_code_revision: str = Field(min_length=1)
    analysis_pixi_lock_hash: Sha256
    created_at_utc: datetime
    plan_hash: Sha256

    @model_validator(mode="after")
    def validate_plan(self) -> ProbeCorrectionAnalysisPlan:
        _require_utc(self.created_at_utc)
        if self.plan_hash != model_content_hash(self, exclude={"plan_hash"}):
            raise ValueError("Probe-correction plan hash does not match its content")
        return self


class ProbeCorrectionCaseResult(ContractModel):
    """Observed and corrected values for one affected case."""

    case_id: str = Field(min_length=1)
    criterion_outcome: bool
    dialogue_correct: bool
    observed_probe_decision: bool
    corrected_probe_decision: bool
    observed_probe_correct: bool
    corrected_probe_correct: bool
    observed_case_effect: float = Field(ge=-1.0, le=1.0)
    corrected_case_effect: float = Field(ge=-1.0, le=1.0)
    case_effect_delta: float = Field(ge=-2.0, le=2.0)
    failure_review_record_hash: Sha256
    evidence_response_hash: Sha256
    evidence_execution_artifact_hash: Sha256
    evidence_test_sha256: Sha256

    @model_validator(mode="after")
    def validate_result(self) -> ProbeCorrectionCaseResult:
        if self.observed_case_effect not in {-1.0, 0.0, 1.0}:
            raise ValueError("Observed correction effect must be -1, 0, or 1")
        if self.corrected_case_effect not in {-1.0, 0.0, 1.0}:
            raise ValueError("Corrected correction effect must be -1, 0, or 1")
        if self.case_effect_delta not in {-2.0, -1.0, 1.0, 2.0}:
            raise ValueError("Correction effect delta must be non-zero and integral")
        if self.observed_probe_correct is not (
            self.observed_probe_decision is self.criterion_outcome
        ):
            raise ValueError("Observed probe correctness does not reconcile")
        if self.corrected_probe_correct is not (
            self.corrected_probe_decision is self.criterion_outcome
        ):
            raise ValueError("Corrected probe correctness does not reconcile")
        if self.case_effect_delta != self.corrected_case_effect - self.observed_case_effect:
            raise ValueError("Correction effect delta does not reconcile")
        return self


class ProbeCorrectionSummary(ContractModel):
    """One complete paired result before or after correction."""

    case_count: int = Field(ge=1)
    inference: PairedCaseInference
    mcnemar: McNemarResult
    valid_evidence_improvement_count: int = Field(ge=0)
    valid_evidence_regression_count: int = Field(ge=0)
    net_improvement_count: int
    paired_effect_from_counts: float = Field(ge=-1.0, le=1.0)
    interval_threshold_status: Literal[
        "interval_above_minimum",
        "interval_crosses_minimum",
        "interval_below_minimum",
    ]

    @model_validator(mode="after")
    def validate_summary(self) -> ProbeCorrectionSummary:
        if (
            self.case_count != self.inference.case_count
            or self.case_count != self.mcnemar.pair_count
        ):
            raise ValueError("Correction-summary case counts do not reconcile")
        if self.valid_evidence_improvement_count != self.mcnemar.right_only_correct_count:
            raise ValueError("Correction-summary improvements differ from McNemar")
        if self.valid_evidence_regression_count != self.mcnemar.left_only_correct_count:
            raise ValueError("Correction-summary regressions differ from McNemar")
        expected_net = self.valid_evidence_improvement_count - self.valid_evidence_regression_count
        if self.net_improvement_count != expected_net:
            raise ValueError("Correction-summary net count does not reconcile")
        expected_effect = expected_net / self.case_count
        if abs(self.paired_effect_from_counts - expected_effect) > 1e-12:
            raise ValueError("Correction-summary count-derived effect does not reconcile")
        if abs(self.paired_effect_from_counts - self.inference.mean_case_effect) > 1e-12:
            raise ValueError("Correction-summary effects do not reconcile")
        if self.interval_threshold_status != _interval_threshold_status(self.inference):
            raise ValueError("Correction-summary interval status does not reconcile")
        return self


class ProbeCorrectionSensitivityReport(ContractModel):
    """Sealed primary result beside a post-hoc corrected sensitivity result."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.probe_correction_sensitivity_report.v1"] = (
        "benchmark.probe_correction_sensitivity_report.v1"
    )
    benchmark_version: Literal["v1"] = "v1"
    run_id: str = Field(min_length=1)
    analysis_plan_hash: Sha256
    correction_count: int = Field(ge=1)
    corrections: tuple[ProbeCorrectionCaseResult, ...] = Field(min_length=1)
    original: ProbeCorrectionSummary
    corrected: ProbeCorrectionSummary
    paired_effect_change: float = Field(ge=-2.0, le=2.0)
    sealed_primary_remains_unchanged: Literal[True] = True
    post_hoc_result_cannot_replace_primary: Literal[True] = True
    analysis_scope: Literal["fixed_authored_corpus_and_declared_corrections_only"] = (
        "fixed_authored_corpus_and_declared_corrections_only"
    )
    network_calls_made: Literal[0] = 0
    sandbox_calls_made: Literal[0] = 0
    completed_at_utc: datetime
    report_hash: Sha256

    @model_validator(mode="after")
    def validate_report(self) -> ProbeCorrectionSensitivityReport:
        _require_utc(self.completed_at_utc)
        if self.correction_count != len(self.corrections):
            raise ValueError("Correction count does not reconcile")
        if len({record.case_id for record in self.corrections}) != self.correction_count:
            raise ValueError("Correction report contains duplicate cases")
        expected_change = (
            self.corrected.paired_effect_from_counts - self.original.paired_effect_from_counts
        )
        if abs(self.paired_effect_change - expected_change) > 1e-12:
            raise ValueError("Correction report effect change does not reconcile")
        if self.report_hash != model_content_hash(self, exclude={"report_hash"}):
            raise ValueError("Probe-correction report hash does not match its content")
        return self


def load_probe_correction_specification(path: Path) -> ProbeCorrectionSpecification:
    """Load one correction specification from YAML."""

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        return ProbeCorrectionSpecification.model_validate(payload)
    except (OSError, yaml.YAMLError, ValidationError) as error:
        raise ProbeCorrectionSensitivityError(
            f"Could not verify probe-correction specification: {path}"
        ) from error


def run_probe_correction_sensitivity(
    *,
    specification_path: Path,
    primary_report_path: Path,
    failure_review_report_path: Path,
    evidence_execution_root: Path,
    pixi_lock_path: Path,
    output_root: Path,
    analysis_code_revision: str,
    created_at_utc: datetime,
) -> ProbeCorrectionSensitivityReport:
    """Recompute paired results after declared, reviewed probe-decision corrections."""

    _require_utc(created_at_utc)
    specification = load_probe_correction_specification(specification_path)
    primary = _load_json(primary_report_path, PrimaryAnalysisReport)
    review = _load_json(failure_review_report_path, FailureReviewRecordReport)
    if primary.report_hash != specification.source_primary_report_hash:
        raise ProbeCorrectionSensitivityError("Primary report differs from the correction source")
    if review.report_hash != specification.source_failure_review_report_hash:
        raise ProbeCorrectionSensitivityError("Failure review differs from the correction source")
    if created_at_utc <= max(primary.completed_at_utc, review.completed_at_utc):
        raise ProbeCorrectionSensitivityError("Correction analysis must follow its source reports")
    try:
        pixi_lock_hash = file_sha256(pixi_lock_path.read_bytes())
    except OSError as error:
        raise ProbeCorrectionSensitivityError(
            f"Could not read Pixi lock: {pixi_lock_path}"
        ) from error

    plan = _create_plan(
        specification=specification,
        primary=primary,
        review=review,
        pixi_lock_hash=pixi_lock_hash,
        analysis_code_revision=analysis_code_revision,
        created_at_utc=created_at_utc,
    )
    root = output_root.resolve()
    report_path = root / "probe_correction_sensitivity_report.json"
    if report_path.exists():
        report = _load_json(report_path, ProbeCorrectionSensitivityReport)
        if report.analysis_plan_hash != plan.plan_hash:
            raise ProbeCorrectionSensitivityError(
                "Existing correction report belongs to another plan"
            )
        return report
    write_immutable_json(root / "probe_correction_analysis_plan.json", plan)

    corrections = _verify_corrections(
        specification=specification,
        primary=primary,
        review=review,
        evidence_execution_root=evidence_execution_root,
    )
    original_pairs = tuple(
        (record.dialogue_correct, record.valid_evidence_correct) for record in primary.case_results
    )
    corrected_decisions = {
        correction.case_id: correction.corrected_probe_decision for correction in corrections
    }
    corrected_pairs = tuple(
        (
            record.dialogue_correct,
            corrected_decisions.get(record.case_id, record.valid_evidence_binary_decision)
            is record.criterion_outcome,
        )
        for record in primary.case_results
    )
    original = _summarize(primary, original_pairs)
    if original.inference != primary.inference or original.mcnemar != primary.mcnemar:
        raise ProbeCorrectionSensitivityError("Reconstructed original result differs from primary")
    corrected = _summarize(primary, corrected_pairs)
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.probe_correction_sensitivity_report.v1",
        "benchmark_version": primary.benchmark_version,
        "run_id": primary.run_id,
        "analysis_plan_hash": plan.plan_hash,
        "correction_count": len(corrections),
        "corrections": corrections,
        "original": original,
        "corrected": corrected,
        "paired_effect_change": (
            corrected.paired_effect_from_counts - original.paired_effect_from_counts
        ),
        "sealed_primary_remains_unchanged": True,
        "post_hoc_result_cannot_replace_primary": True,
        "analysis_scope": "fixed_authored_corpus_and_declared_corrections_only",
        "network_calls_made": 0,
        "sandbox_calls_made": 0,
        "completed_at_utc": created_at_utc,
    }
    draft = ProbeCorrectionSensitivityReport.model_construct(
        _fields_set=set(content), **content, report_hash="0" * 64
    )
    report = ProbeCorrectionSensitivityReport.model_validate(
        {**content, "report_hash": model_content_hash(draft, exclude={"report_hash"})}
    )
    write_immutable_json(report_path, report)
    return report


def _verify_corrections(
    *,
    specification: ProbeCorrectionSpecification,
    primary: PrimaryAnalysisReport,
    review: FailureReviewRecordReport,
    evidence_execution_root: Path,
) -> tuple[ProbeCorrectionCaseResult, ...]:
    primary_by_case = {record.case_id: record for record in primary.case_results}
    review_by_case = {record.case_id: record for record in review.records}
    results: list[ProbeCorrectionCaseResult] = []
    for directive in specification.corrections:
        case = primary_by_case.get(directive.case_id)
        reviewed = review_by_case.get(directive.case_id)
        if case is None or reviewed is None:
            raise ProbeCorrectionSensitivityError(
                f"Correction case is absent from a source report: {directive.case_id}"
            )
        if case.valid_evidence_binary_decision is not directive.observed_probe_decision:
            raise ProbeCorrectionSensitivityError(
                f"Observed probe decision differs for {directive.case_id}"
            )
        if (
            reviewed.selection_role != "primary_failure"
            or reviewed.category is not FailureReviewLabel.ITEM_AMBIGUITY_OR_TEST_DEFECT
            or reviewed.record_hash != directive.failure_review_record_hash
            or directive.evidence_response_hash not in reviewed.evidence_references
        ):
            raise ProbeCorrectionSensitivityError(
                f"Failure review does not support correction for {directive.case_id}"
            )
        execution_path = evidence_execution_root / f"{directive.evidence_response_hash}.json"
        execution = _load_json(execution_path, ExternalEvidenceExecutionArtifact)
        if (
            execution.case_id != directive.case_id
            or execution.response_hash != directive.evidence_response_hash
            or execution.artifact_hash != directive.evidence_execution_artifact_hash
            or execution.evidence_test_sha256 != directive.evidence_test_sha256
            or execution.outcome_status != "completed"
            or execution.passed != 0
            or execution.failed is None
            or execution.failed < 1
        ):
            raise ProbeCorrectionSensitivityError(
                f"Evidence execution differs from declared defect for {directive.case_id}"
            )
        observed_correct = case.valid_evidence_correct
        corrected_correct = directive.corrected_probe_decision is case.criterion_outcome
        observed_effect = case.case_effect
        corrected_effect = float(not case.dialogue_correct) - float(not corrected_correct)
        results.append(
            ProbeCorrectionCaseResult(
                case_id=directive.case_id,
                criterion_outcome=case.criterion_outcome,
                dialogue_correct=case.dialogue_correct,
                observed_probe_decision=directive.observed_probe_decision,
                corrected_probe_decision=directive.corrected_probe_decision,
                observed_probe_correct=observed_correct,
                corrected_probe_correct=corrected_correct,
                observed_case_effect=observed_effect,
                corrected_case_effect=corrected_effect,
                case_effect_delta=corrected_effect - observed_effect,
                failure_review_record_hash=reviewed.record_hash,
                evidence_response_hash=execution.response_hash,
                evidence_execution_artifact_hash=execution.artifact_hash,
                evidence_test_sha256=execution.evidence_test_sha256,
            )
        )
    return tuple(results)


def _summarize(
    primary: PrimaryAnalysisReport,
    pairs: tuple[tuple[bool, bool], ...],
) -> ProbeCorrectionSummary:
    effects = tuple(float(not left) - float(not right) for left, right in pairs)
    source = primary.inference
    inference = paired_case_inference(
        effects,
        bootstrap_resamples=source.bootstrap_resamples,
        permutation_resamples=source.permutation_draw_count,
        minimum_interpretable_effect=source.minimum_interpretable_effect,
        random_seed=source.random_seed,
    )
    mcnemar = exact_mcnemar(pairs)
    improvements = mcnemar.right_only_correct_count
    regressions = mcnemar.left_only_correct_count
    net = improvements - regressions
    return ProbeCorrectionSummary(
        case_count=len(pairs),
        inference=inference,
        mcnemar=mcnemar,
        valid_evidence_improvement_count=improvements,
        valid_evidence_regression_count=regressions,
        net_improvement_count=net,
        paired_effect_from_counts=net / len(pairs),
        interval_threshold_status=_interval_threshold_status(inference),
    )


def _create_plan(
    *,
    specification: ProbeCorrectionSpecification,
    primary: PrimaryAnalysisReport,
    review: FailureReviewRecordReport,
    pixi_lock_hash: Sha256,
    analysis_code_revision: str,
    created_at_utc: datetime,
) -> ProbeCorrectionAnalysisPlan:
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.probe_correction_analysis_plan.v1",
        "benchmark_version": primary.benchmark_version,
        "run_id": primary.run_id,
        "correction_specification_hash": model_content_hash(specification),
        "primary_report_hash": primary.report_hash,
        "failure_review_report_hash": review.report_hash,
        "method": "paired_binary_probe_decision_correction_v1",
        "analysis_code_revision": analysis_code_revision,
        "analysis_pixi_lock_hash": pixi_lock_hash,
        "created_at_utc": created_at_utc,
    }
    draft = ProbeCorrectionAnalysisPlan.model_construct(
        _fields_set=set(content), **content, plan_hash="0" * 64
    )
    return ProbeCorrectionAnalysisPlan.model_validate(
        {**content, "plan_hash": model_content_hash(draft, exclude={"plan_hash"})}
    )


def _interval_threshold_status(
    inference: PairedCaseInference,
) -> Literal["interval_above_minimum", "interval_crosses_minimum", "interval_below_minimum"]:
    if inference.interval_lower >= inference.minimum_interpretable_effect:
        return "interval_above_minimum"
    if inference.interval_upper < inference.minimum_interpretable_effect:
        return "interval_below_minimum"
    return "interval_crosses_minimum"


def _load_json[ModelT: ContractModel](path: Path, model: type[ModelT]) -> ModelT:
    try:
        return model.model_validate_json(path.read_bytes())
    except (OSError, ValidationError) as error:
        raise ProbeCorrectionSensitivityError(
            f"Could not verify probe-correction source: {path}"
        ) from error


def _require_utc(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("Probe-correction timestamp must be UTC")
