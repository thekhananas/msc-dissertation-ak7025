"""Content-addressed interpretation of the sealed benchmark result."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError, model_validator

from socratic_tutor.benchmark.artifacts import write_immutable_json
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import file_sha256, model_content_hash
from socratic_tutor.benchmark.negative_result_audit import (
    NegativeResultAuditPlan,
    NegativeResultAuditReport,
)
from socratic_tutor.benchmark.primary_analysis import PrimaryAnalysisReport
from socratic_tutor.benchmark.probe_correction_sensitivity import (
    ProbeCorrectionAnalysisPlan,
    ProbeCorrectionSensitivityReport,
)
from socratic_tutor.benchmark.secondary_analysis import SecondaryAnalysisReport
from socratic_tutor.contracts import ContractModel

PrimaryConclusion = Literal["positive", "negative", "inconclusive"]
SpecificityConclusion = Literal["specific", "non_specific"]
CombinedConclusion = Literal[
    "positive_and_specific",
    "positive_but_non_specific",
    "negative",
    "inconclusive",
]


class ResultInterpretationError(ValueError):
    """Source reports do not support one internally consistent result note."""


class ResultInterpretationPlan(ContractModel):
    """Immutable report identities and software version for one interpretation."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.result_interpretation_plan.v1"] = (
        "benchmark.result_interpretation_plan.v1"
    )
    benchmark_version: Literal["v1"] = "v1"
    run_id: str = Field(min_length=1)
    primary_report_hash: Sha256
    secondary_report_hash: Sha256
    probe_correction_plan_hash: Sha256
    probe_correction_report_hash: Sha256
    negative_result_audit_plan_hash: Sha256
    negative_result_audit_report_hash: Sha256
    analysis_code_revision: str = Field(min_length=1)
    analysis_pixi_lock_hash: Sha256
    record_owner: Literal["dissertation_author"]
    created_at_utc: datetime
    plan_hash: Sha256

    @model_validator(mode="after")
    def validate_plan(self) -> ResultInterpretationPlan:
        _require_utc(self.created_at_utc)
        if self.plan_hash != model_content_hash(self, exclude={"plan_hash"}):
            raise ValueError("Result-interpretation plan hash does not match its content")
        return self


class ResultInterpretationReport(ContractModel):
    """Plain result wording bound to the reports from which it was derived."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.result_interpretation_report.v1"] = (
        "benchmark.result_interpretation_report.v1"
    )
    benchmark_version: Literal["v1"] = "v1"
    run_id: str = Field(min_length=1)
    interpretation_plan_hash: Sha256
    total_case_count: Literal[24]
    eligible_case_count: int = Field(ge=1, le=24)
    missing_case_count: int = Field(ge=0, le=24)
    primary_mean_effect: float = Field(ge=-1.0, le=1.0)
    primary_interval_lower: float = Field(ge=-1.0, le=1.0)
    primary_interval_upper: float = Field(ge=-1.0, le=1.0)
    minimum_interpretable_effect: float = Field(gt=0.0, le=1.0)
    exact_mcnemar_p_value: float = Field(ge=0.0, le=1.0)
    primary_conclusion: PrimaryConclusion
    specificity_mean_effect: float = Field(ge=-1.0, le=1.0)
    specificity_interval_lower: float = Field(ge=-1.0, le=1.0)
    specificity_interval_upper: float = Field(ge=-1.0, le=1.0)
    specificity_conclusion: SpecificityConclusion
    combined_conclusion: CombinedConclusion
    primary_wording: str = Field(min_length=1)
    specificity_wording: str = Field(min_length=1)
    combined_wording: str = Field(min_length=1)
    interpretation_rule_status: Literal["post_result_reporting_rule_not_confirmatory_gate"] = (
        "post_result_reporting_rule_not_confirmatory_gate"
    )
    evidence_specificity_role: Literal["secondary_cannot_rescue_primary"] = (
        "secondary_cannot_rescue_primary"
    )
    post_hoc_correction_role: Literal["sensitivity_cannot_replace_sealed_primary"] = (
        "sensitivity_cannot_replace_sealed_primary"
    )
    result_scope: Literal["one_pinned_model_run_on_one_fixed_authored_corpus"] = (
        "one_pinned_model_run_on_one_fixed_authored_corpus"
    )
    human_learning_claim_allowed: Literal[False] = False
    tutoring_efficacy_claim_allowed: Literal[False] = False
    reduced_cognitive_offloading_claim_allowed: Literal[False] = False
    absence_of_significance_means_no_effect: Literal[False] = False
    network_calls_made: Literal[0] = 0
    sandbox_calls_made: Literal[0] = 0
    completed_at_utc: datetime
    report_hash: Sha256

    @model_validator(mode="after")
    def validate_report(self) -> ResultInterpretationReport:
        _require_utc(self.completed_at_utc)
        if self.total_case_count != self.eligible_case_count + self.missing_case_count:
            raise ValueError("Interpretation case counts do not reconcile")
        if self.primary_interval_lower > self.primary_interval_upper:
            raise ValueError("Primary interval bounds are reversed")
        if self.specificity_interval_lower > self.specificity_interval_upper:
            raise ValueError("Specificity interval bounds are reversed")
        expected_primary = classify_primary(
            interval_lower=self.primary_interval_lower,
            interval_upper=self.primary_interval_upper,
            minimum_interpretable_effect=self.minimum_interpretable_effect,
        )
        if self.primary_conclusion != expected_primary:
            raise ValueError("Primary conclusion differs from the reporting rule")
        expected_specificity: SpecificityConclusion = (
            "specific" if self.specificity_mean_effect > 0.0 else "non_specific"
        )
        if self.specificity_conclusion != expected_specificity:
            raise ValueError("Specificity conclusion differs from its mean effect")
        if self.combined_conclusion != combine_conclusions(expected_primary, expected_specificity):
            raise ValueError("Combined conclusion differs from its source conclusions")
        if self.report_hash != model_content_hash(self, exclude={"report_hash"}):
            raise ValueError("Result-interpretation report hash does not match its content")
        return self


def classify_primary(
    *,
    interval_lower: float,
    interval_upper: float,
    minimum_interpretable_effect: float,
) -> PrimaryConclusion:
    """Map an interval to cautious report language, not a new hypothesis test."""

    if interval_lower >= minimum_interpretable_effect:
        return "positive"
    if interval_upper < 0.0:
        return "negative"
    return "inconclusive"


def combine_conclusions(
    primary: PrimaryConclusion, specificity: SpecificityConclusion
) -> CombinedConclusion:
    if primary == "negative":
        return "negative"
    if primary == "inconclusive":
        return "inconclusive"
    return "positive_and_specific" if specificity == "specific" else "positive_but_non_specific"


def run_result_interpretation(
    *,
    primary_report_path: Path,
    secondary_report_path: Path,
    probe_correction_plan_path: Path,
    probe_correction_report_path: Path,
    negative_result_audit_plan_path: Path,
    negative_result_audit_report_path: Path,
    pixi_lock_path: Path,
    output_root: Path,
    analysis_code_revision: str,
    created_at_utc: datetime,
) -> ResultInterpretationReport:
    """Validate source lineage and write one immutable interpretation record."""

    _require_utc(created_at_utc)
    primary = _load_json(primary_report_path, PrimaryAnalysisReport)
    secondary = _load_json(secondary_report_path, SecondaryAnalysisReport)
    correction_plan = _load_json(probe_correction_plan_path, ProbeCorrectionAnalysisPlan)
    correction = _load_json(probe_correction_report_path, ProbeCorrectionSensitivityReport)
    audit_plan = _load_json(negative_result_audit_plan_path, NegativeResultAuditPlan)
    audit = _load_json(negative_result_audit_report_path, NegativeResultAuditReport)
    _validate_lineage(primary, secondary, correction_plan, correction, audit_plan, audit)
    if created_at_utc <= max(
        primary.completed_at_utc,
        secondary.completed_at_utc,
        correction.completed_at_utc,
        audit.completed_at_utc,
    ):
        raise ResultInterpretationError("Interpretation must follow all source reports")
    try:
        lock_hash = file_sha256(pixi_lock_path.read_bytes())
    except OSError as error:
        raise ResultInterpretationError("Could not hash the interpretation environment") from error

    plan = _create_plan(
        primary=primary,
        secondary=secondary,
        correction_plan=correction_plan,
        correction=correction,
        audit_plan=audit_plan,
        audit=audit,
        analysis_code_revision=analysis_code_revision,
        lock_hash=lock_hash,
        created_at_utc=created_at_utc,
    )
    root = output_root.resolve()
    report_path = root / "result_interpretation_report.json"
    if report_path.exists():
        report = _load_json(report_path, ResultInterpretationReport)
        if report.interpretation_plan_hash != plan.plan_hash:
            raise ResultInterpretationError("Existing interpretation belongs to another plan")
        return report
    write_immutable_json(root / "result_interpretation_plan.json", plan)

    inference = primary.inference
    specificity = secondary.evidence_specificity.combined_specificity_inference
    primary_conclusion = classify_primary(
        interval_lower=inference.interval_lower,
        interval_upper=inference.interval_upper,
        minimum_interpretable_effect=inference.minimum_interpretable_effect,
    )
    specificity_conclusion: SpecificityConclusion = (
        "specific" if specificity.mean_case_effect > 0.0 else "non_specific"
    )
    combined = combine_conclusions(primary_conclusion, specificity_conclusion)
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.result_interpretation_report.v1",
        "benchmark_version": "v1",
        "run_id": primary.run_id,
        "interpretation_plan_hash": plan.plan_hash,
        "total_case_count": primary.total_case_count,
        "eligible_case_count": primary.eligible_case_count,
        "missing_case_count": primary.missing_case_count,
        "primary_mean_effect": inference.mean_case_effect,
        "primary_interval_lower": inference.interval_lower,
        "primary_interval_upper": inference.interval_upper,
        "minimum_interpretable_effect": inference.minimum_interpretable_effect,
        "exact_mcnemar_p_value": primary.mcnemar.two_sided_exact_p_value,
        "primary_conclusion": primary_conclusion,
        "specificity_mean_effect": specificity.mean_case_effect,
        "specificity_interval_lower": specificity.interval_lower,
        "specificity_interval_upper": specificity.interval_upper,
        "specificity_conclusion": specificity_conclusion,
        "combined_conclusion": combined,
        "primary_wording": _primary_wording(primary, primary_conclusion),
        "specificity_wording": _specificity_wording(specificity_conclusion),
        "combined_wording": _combined_wording(combined),
        "interpretation_rule_status": "post_result_reporting_rule_not_confirmatory_gate",
        "evidence_specificity_role": "secondary_cannot_rescue_primary",
        "post_hoc_correction_role": "sensitivity_cannot_replace_sealed_primary",
        "result_scope": "one_pinned_model_run_on_one_fixed_authored_corpus",
        "human_learning_claim_allowed": False,
        "tutoring_efficacy_claim_allowed": False,
        "reduced_cognitive_offloading_claim_allowed": False,
        "absence_of_significance_means_no_effect": False,
        "network_calls_made": 0,
        "sandbox_calls_made": 0,
        "completed_at_utc": created_at_utc,
    }
    draft = ResultInterpretationReport.model_construct(
        _fields_set=set(content), **content, report_hash="0" * 64
    )
    report = ResultInterpretationReport.model_validate(
        {**content, "report_hash": model_content_hash(draft, exclude={"report_hash"})}
    )
    write_immutable_json(report_path, report)
    return report


def _validate_lineage(
    primary: PrimaryAnalysisReport,
    secondary: SecondaryAnalysisReport,
    correction_plan: ProbeCorrectionAnalysisPlan,
    correction: ProbeCorrectionSensitivityReport,
    audit_plan: NegativeResultAuditPlan,
    audit: NegativeResultAuditReport,
) -> None:
    if secondary.primary_analysis_report_hash != primary.report_hash:
        raise ResultInterpretationError("Secondary report belongs to another primary result")
    if correction_plan.primary_report_hash != primary.report_hash:
        raise ResultInterpretationError("Correction plan belongs to another primary result")
    if correction.analysis_plan_hash != correction_plan.plan_hash:
        raise ResultInterpretationError("Correction report belongs to another correction plan")
    if correction.run_id != primary.run_id:
        raise ResultInterpretationError("Correction and primary run identities differ")
    if audit.analysis_plan_hash != audit_plan.plan_hash:
        raise ResultInterpretationError("Negative-result audit belongs to another plan")
    if audit.benchmark_version != primary.benchmark_version:
        raise ResultInterpretationError("Negative-result audit uses another benchmark")


def _create_plan(
    *,
    primary: PrimaryAnalysisReport,
    secondary: SecondaryAnalysisReport,
    correction_plan: ProbeCorrectionAnalysisPlan,
    correction: ProbeCorrectionSensitivityReport,
    audit_plan: NegativeResultAuditPlan,
    audit: NegativeResultAuditReport,
    analysis_code_revision: str,
    lock_hash: Sha256,
    created_at_utc: datetime,
) -> ResultInterpretationPlan:
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.result_interpretation_plan.v1",
        "benchmark_version": "v1",
        "run_id": primary.run_id,
        "primary_report_hash": primary.report_hash,
        "secondary_report_hash": secondary.report_hash,
        "probe_correction_plan_hash": correction_plan.plan_hash,
        "probe_correction_report_hash": correction.report_hash,
        "negative_result_audit_plan_hash": audit_plan.plan_hash,
        "negative_result_audit_report_hash": audit.report_hash,
        "analysis_code_revision": analysis_code_revision,
        "analysis_pixi_lock_hash": lock_hash,
        "record_owner": "dissertation_author",
        "created_at_utc": created_at_utc,
    }
    draft = ResultInterpretationPlan.model_construct(
        _fields_set=set(content), **content, plan_hash="0" * 64
    )
    return ResultInterpretationPlan.model_validate(
        {**content, "plan_hash": model_content_hash(draft, exclude={"plan_hash"})}
    )


def _primary_wording(primary: PrimaryAnalysisReport, conclusion: PrimaryConclusion) -> str:
    inference = primary.inference
    if inference.mean_case_effect > 0.0:
        effect_wording = f"valid evidence had lower error by {inference.mean_case_effect:.4f}"
    elif inference.mean_case_effect < 0.0:
        effect_wording = f"valid evidence had higher error by {-inference.mean_case_effect:.4f}"
    else:
        effect_wording = "the two conditions had the same mean error"
    result = (
        f"Across {primary.eligible_case_count} eligible authored cases, {effect_wording}; "
        f"the 95% paired interval was "
        f"[{inference.interval_lower:.4f}, {inference.interval_upper:.4f}], and exact McNemar "
        f"p was {primary.mcnemar.two_sided_exact_p_value:.4f}."
    )
    if conclusion == "positive":
        return result + " The whole interval cleared the frozen practical threshold."
    if conclusion == "negative":
        return result + " The whole interval was below zero."
    return (
        result + " The primary result is inconclusive; this does not mean that the effect is zero."
    )


def _specificity_wording(conclusion: SpecificityConclusion) -> str:
    if conclusion == "specific":
        return (
            "Valid evidence performed better than the average contrastive control in the "
            "secondary analysis."
        )
    return (
        "Valid evidence did not perform better than the average contrastive control in the "
        "secondary analysis."
    )


def _combined_wording(conclusion: CombinedConclusion) -> str:
    if conclusion == "positive_and_specific":
        return (
            "The result is positive on this corpus, and the secondary comparison supports "
            "evidence specificity."
        )
    if conclusion == "positive_but_non_specific":
        return (
            "The primary result is positive, but the secondary comparison does not support "
            "evidence specificity."
        )
    if conclusion == "negative":
        return (
            "The result is negative on this corpus; secondary analyses do not alter that "
            "conclusion."
        )
    return (
        "The primary result remains inconclusive; secondary specificity and post-hoc "
        "corrections cannot rescue it."
    )


def _load_json[ModelT: ContractModel](path: Path, model: type[ModelT]) -> ModelT:
    try:
        return model.model_validate_json(path.read_bytes())
    except (OSError, ValidationError) as error:
        raise ResultInterpretationError(
            f"Could not verify interpretation source: {path}"
        ) from error


def _require_utc(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("Result-interpretation timestamp must be UTC")
