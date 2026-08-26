"""Frozen secondary analysis for whether valid evidence beats contrastive controls."""

from __future__ import annotations

from collections import defaultdict
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
from socratic_tutor.benchmark.evaluator.aggregation import CaseAggregate, CaseComparison
from socratic_tutor.benchmark.evaluator.scoring import PrimaryMetric
from socratic_tutor.benchmark.external_protocol import ExternalModelExecutionProtocol
from socratic_tutor.benchmark.hashing import canonical_sha256, model_content_hash
from socratic_tutor.benchmark.json_io import load_json_model
from socratic_tutor.benchmark.public_rating import PublicRatingProtocolAmendment
from socratic_tutor.benchmark.statistics import PairedCaseInference, paired_case_inference
from socratic_tutor.contracts import ContractModel


class EvidenceSpecificityAmendmentPlan(ContractModel):
    """Reviewed identities that the secondary-analysis amendment must bind."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.evidence_specificity_amendment_plan.v1"] = (
        "benchmark.evidence_specificity_amendment_plan.v1"
    )
    amendment_version: Literal["evidence-specificity-v1"]
    base_analysis_specification_hash: Sha256
    base_external_protocol_hash: Sha256
    calibration_decision_hash: Sha256


class EvidenceSpecificityAmendment(ContractModel):
    """Content-addressed equations and inherited inference rules."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.evidence_specificity_amendment.v1"] = (
        "benchmark.evidence_specificity_amendment.v1"
    )
    amendment_version: Literal["evidence-specificity-v1"]
    base_analysis_specification_hash: Sha256
    base_external_protocol_hash: Sha256
    calibration_decision_hash: Sha256
    selected_loss: Literal["classification_error"]
    primary_case_effect: Literal["dialogue_loss_minus_valid_probe_loss"]
    valid_vs_unrelated_effect: Literal["unrelated_probe_loss_minus_valid_probe_loss"]
    valid_vs_corrupted_effect: Literal["corrupted_probe_loss_minus_valid_probe_loss"]
    combined_specificity_effect: Literal["mean_unrelated_and_corrupted_loss_minus_valid_probe_loss"]
    aggregation_unit: Literal["case"]
    aggregation_version: str = Field(min_length=1)
    missingness_rule: Literal["exclude_missing_criterion_and_report_denominator"]
    bootstrap_method: Literal["paired_case_percentile"]
    bootstrap_resamples: int = Field(ge=1000)
    permutation_method: Literal["within_case_sign_swap"]
    permutation_resamples: int = Field(ge=1000)
    random_seed: int = Field(ge=0, le=2**32 - 1)
    primary_minimum_interpretable_effect: float = Field(gt=0.0, le=1.0)
    secondary_role: Literal["interpretation_only_cannot_rescue_primary"]
    positive_mean_interpretation: Literal["valid_evidence_is_more_specific_than_controls"]
    nonpositive_mean_interpretation: Literal["primary_effect_is_not_evidence_specific"]
    amendment_hash: Sha256

    @model_validator(mode="after")
    def validate_amendment(self) -> EvidenceSpecificityAmendment:
        if self.bootstrap_resamples != self.permutation_resamples:
            raise ValueError("Specificity inference must preserve the frozen resample count")
        if self.amendment_hash != model_content_hash(self, exclude={"amendment_hash"}):
            raise ValueError("Evidence-specificity amendment hash does not match its content")
        return self


class EvidenceSpecificityCaseEffect(ContractModel):
    """Three hand-checkable secondary effects for one independent case."""

    benchmark_version: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    model_route_id: str = Field(min_length=1)
    valid_vs_unrelated_effect: float = Field(ge=-1.0, le=1.0)
    valid_vs_corrupted_effect: float = Field(ge=-1.0, le=1.0)
    combined_specificity_effect: float = Field(ge=-1.0, le=1.0)
    source_aggregate_hashes: tuple[Sha256, Sha256, Sha256]
    effect_hash: Sha256

    @model_validator(mode="after")
    def validate_effect(self) -> EvidenceSpecificityCaseEffect:
        expected = (self.valid_vs_unrelated_effect + self.valid_vs_corrupted_effect) / 2
        if abs(self.combined_specificity_effect - expected) > 1e-12:
            raise ValueError("Combined specificity must average the two contrastive effects")
        if len(set(self.source_aggregate_hashes)) != 3:
            raise ValueError("Specificity effect requires three distinct aggregate sources")
        if self.effect_hash != model_content_hash(self, exclude={"effect_hash"}):
            raise ValueError("Evidence-specificity case-effect hash does not match its content")
        return self


class EvidenceSpecificityReport(ContractModel):
    """Secondary uncertainty summaries that cannot alter the primary endpoint."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.evidence_specificity_report.v1"] = (
        "benchmark.evidence_specificity_report.v1"
    )
    amendment_hash: Sha256
    total_case_count: int = Field(ge=1)
    eligible_case_count: int = Field(ge=1)
    missing_case_count: int = Field(ge=0)
    case_effects: tuple[EvidenceSpecificityCaseEffect, ...] = Field(min_length=1)
    valid_vs_unrelated_inference: PairedCaseInference
    valid_vs_corrupted_inference: PairedCaseInference
    combined_specificity_inference: PairedCaseInference
    interpretation: Literal[
        "valid_evidence_is_more_specific_than_controls",
        "primary_effect_is_not_evidence_specific",
    ]
    confirmatory_gate_allowed: Literal[False] = False
    report_hash: Sha256

    @model_validator(mode="after")
    def validate_report(self) -> EvidenceSpecificityReport:
        if self.total_case_count != self.eligible_case_count + self.missing_case_count:
            raise ValueError("Specificity case counts do not reconcile")
        if self.eligible_case_count != len(self.case_effects):
            raise ValueError("Eligible specificity count does not match case effects")
        expected = (
            "valid_evidence_is_more_specific_than_controls"
            if self.combined_specificity_inference.mean_case_effect > 0.0
            else "primary_effect_is_not_evidence_specific"
        )
        if self.interpretation != expected:
            raise ValueError("Specificity interpretation does not match the combined mean")
        if self.report_hash != model_content_hash(self, exclude={"report_hash"}):
            raise ValueError("Evidence-specificity report hash does not match its content")
        return self


class ExternalRunPreflight(ContractModel):
    """Proof that both mandatory amendments match the held-out protocol."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.external_run_preflight.v1"] = (
        "benchmark.external_run_preflight.v1"
    )
    external_protocol_hash: Sha256
    analysis_specification_hash: Sha256
    calibration_decision_hash: Sha256
    public_rating_amendment_hash: Sha256
    evidence_specificity_amendment_hash: Sha256
    gate_passed: Literal[True] = True
    preflight_hash: Sha256

    @model_validator(mode="after")
    def validate_preflight(self) -> ExternalRunPreflight:
        if self.preflight_hash != model_content_hash(self, exclude={"preflight_hash"}):
            raise ValueError("External-run preflight hash does not match its content")
        return self


def load_evidence_specificity_amendment_plan(path: Path) -> EvidenceSpecificityAmendmentPlan:
    """Load the reviewed amendment identities from YAML."""

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"Could not read evidence-specificity amendment plan: {path}") from error
    return EvidenceSpecificityAmendmentPlan.model_validate(raw)


def load_evidence_specificity_amendment(path: Path) -> EvidenceSpecificityAmendment:
    return load_json_model(path, EvidenceSpecificityAmendment)


def load_uncalibrated_decision_report(path: Path) -> UncalibratedDecisionReport:
    return load_json_model(path, UncalibratedDecisionReport)


def load_external_run_preflight(path: Path) -> ExternalRunPreflight:
    """Load and validate the frozen gate required before held-out generation."""

    return load_json_model(path, ExternalRunPreflight)


def freeze_evidence_specificity_amendment(
    *,
    plan: EvidenceSpecificityAmendmentPlan,
    protocol: ExternalModelExecutionProtocol,
    analysis_specification: AnalysisSpecification,
    calibration_report: UncalibratedDecisionReport,
    output_path: Path,
) -> EvidenceSpecificityAmendment:
    """Freeze the secondary equations while inheriting the primary analysis rules."""

    specification_hash = analysis_specification_hash(analysis_specification)
    if plan.base_analysis_specification_hash != specification_hash:
        raise ValueError("Specificity plan belongs to a different analysis specification")
    if plan.base_external_protocol_hash != protocol.protocol_hash:
        raise ValueError("Specificity plan belongs to a different external protocol")
    if protocol.analysis_specification_hash != specification_hash:
        raise ValueError("External protocol belongs to a different analysis specification")
    if plan.calibration_decision_hash != calibration_report.decision.decision_hash:
        raise ValueError("Specificity plan belongs to a different calibration decision")
    if calibration_report.analysis_specification_hash != specification_hash:
        raise ValueError("Calibration decision belongs to a different analysis specification")
    if (
        calibration_report.decision.primary_metric
        is not PrimaryMetric.PAIRED_CLASSIFICATION_ERROR_DIFFERENCE
    ):
        raise ValueError("Specificity v1 requires the frozen classification-error fallback")
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.evidence_specificity_amendment.v1",
        "amendment_version": plan.amendment_version,
        "base_analysis_specification_hash": specification_hash,
        "base_external_protocol_hash": protocol.protocol_hash,
        "calibration_decision_hash": calibration_report.decision.decision_hash,
        "selected_loss": "classification_error",
        "primary_case_effect": "dialogue_loss_minus_valid_probe_loss",
        "valid_vs_unrelated_effect": "unrelated_probe_loss_minus_valid_probe_loss",
        "valid_vs_corrupted_effect": "corrupted_probe_loss_minus_valid_probe_loss",
        "combined_specificity_effect": ("mean_unrelated_and_corrupted_loss_minus_valid_probe_loss"),
        "aggregation_unit": analysis_specification.aggregation_unit,
        "aggregation_version": analysis_specification.aggregation_version,
        "missingness_rule": analysis_specification.missingness_rule,
        "bootstrap_method": analysis_specification.bootstrap_method,
        "bootstrap_resamples": analysis_specification.bootstrap_resamples,
        "permutation_method": analysis_specification.permutation_method,
        "permutation_resamples": analysis_specification.permutation_resamples,
        "random_seed": analysis_specification.random_seed,
        "primary_minimum_interpretable_effect": (
            analysis_specification.minimum_interpretable_effect
        ),
        "secondary_role": "interpretation_only_cannot_rescue_primary",
        "positive_mean_interpretation": "valid_evidence_is_more_specific_than_controls",
        "nonpositive_mean_interpretation": "primary_effect_is_not_evidence_specific",
    }
    amendment = EvidenceSpecificityAmendment.model_validate(
        {**content, "amendment_hash": canonical_sha256(content)}
    )
    write_immutable_json(output_path, amendment)
    return amendment


def analyze_evidence_specificity(
    *,
    aggregates: tuple[CaseAggregate, ...],
    amendment: EvidenceSpecificityAmendment,
    output_path: Path,
) -> EvidenceSpecificityReport:
    """Calculate specificity from the existing dialogue-based case aggregates."""

    grouped = _group_required_aggregates(aggregates, amendment)
    effects = tuple(
        effect
        for _, rows in sorted(grouped.items())
        if (effect := _create_case_effect(rows)) is not None
    )
    if not effects:
        raise ValueError("Specificity analysis requires at least one complete case")
    unrelated = _specificity_inference(
        tuple(item.valid_vs_unrelated_effect for item in effects), amendment
    )
    corrupted = _specificity_inference(
        tuple(item.valid_vs_corrupted_effect for item in effects), amendment
    )
    combined = _specificity_inference(
        tuple(item.combined_specificity_effect for item in effects), amendment
    )
    total = len(grouped)
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.evidence_specificity_report.v1",
        "amendment_hash": amendment.amendment_hash,
        "total_case_count": total,
        "eligible_case_count": len(effects),
        "missing_case_count": total - len(effects),
        "case_effects": effects,
        "valid_vs_unrelated_inference": unrelated,
        "valid_vs_corrupted_inference": corrupted,
        "combined_specificity_inference": combined,
        "interpretation": (
            amendment.positive_mean_interpretation
            if combined.mean_case_effect > 0.0
            else amendment.nonpositive_mean_interpretation
        ),
        "confirmatory_gate_allowed": False,
    }
    report = EvidenceSpecificityReport.model_validate(
        {**content, "report_hash": canonical_sha256(content)}
    )
    write_immutable_json(output_path, report)
    return report


def create_external_run_preflight(
    *,
    protocol: ExternalModelExecutionProtocol,
    analysis_specification: AnalysisSpecification,
    calibration_report: UncalibratedDecisionReport,
    public_rating_amendment: PublicRatingProtocolAmendment,
    evidence_specificity_amendment: EvidenceSpecificityAmendment,
    output_path: Path,
) -> ExternalRunPreflight:
    """Fail unless all mandatory pre-execution artifacts share one frozen identity."""

    specification_hash = analysis_specification_hash(analysis_specification)
    if protocol.analysis_specification_hash != specification_hash:
        raise ValueError("External protocol belongs to another analysis specification")
    if calibration_report.analysis_specification_hash != specification_hash:
        raise ValueError("Calibration report belongs to another analysis specification")
    if public_rating_amendment.base_protocol_hash != protocol.protocol_hash:
        raise ValueError("Public-rating amendment belongs to another external protocol")
    if evidence_specificity_amendment.base_external_protocol_hash != protocol.protocol_hash:
        raise ValueError("Specificity amendment belongs to another external protocol")
    if evidence_specificity_amendment.base_analysis_specification_hash != specification_hash:
        raise ValueError("Specificity amendment belongs to another analysis specification")
    decision_hash = calibration_report.decision.decision_hash
    if evidence_specificity_amendment.calibration_decision_hash != decision_hash:
        raise ValueError("Specificity amendment belongs to another calibration decision")
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.external_run_preflight.v1",
        "external_protocol_hash": protocol.protocol_hash,
        "analysis_specification_hash": specification_hash,
        "calibration_decision_hash": decision_hash,
        "public_rating_amendment_hash": public_rating_amendment.amendment_hash,
        "evidence_specificity_amendment_hash": evidence_specificity_amendment.amendment_hash,
        "gate_passed": True,
    }
    preflight = ExternalRunPreflight.model_validate(
        {**content, "preflight_hash": canonical_sha256(content)}
    )
    write_immutable_json(output_path, preflight)
    return preflight


def _group_required_aggregates(
    aggregates: tuple[CaseAggregate, ...],
    amendment: EvidenceSpecificityAmendment,
) -> dict[tuple[str, str, str, str], dict[CaseComparison, CaseAggregate]]:
    required = {
        CaseComparison.PRIMARY_VALID,
        CaseComparison.UNRELATED_CONTROL,
        CaseComparison.CORRUPTED_CONTROL,
    }
    grouped: defaultdict[tuple[str, str, str, str], dict[CaseComparison, CaseAggregate]] = (
        defaultdict(dict)
    )
    for aggregate in aggregates:
        if aggregate.comparison not in required:
            continue
        if aggregate.aggregation_version != amendment.aggregation_version:
            raise ValueError("Case aggregate uses another aggregation version")
        if _normalized_missingness_rule(aggregate.missingness_rule) != amendment.missingness_rule:
            raise ValueError("Case aggregate uses another missingness rule")
        identity = (
            aggregate.benchmark_version,
            aggregate.run_id,
            aggregate.case_id,
            aggregate.model_route_id,
        )
        if aggregate.comparison in grouped[identity]:
            raise ValueError("Duplicate case aggregate for a specificity comparison")
        grouped[identity][aggregate.comparison] = aggregate
    if not grouped:
        raise ValueError("Specificity analysis found no required case aggregates")
    if any(set(rows) != required for rows in grouped.values()):
        raise ValueError("Every specificity case requires valid, unrelated, and corrupted rows")
    return dict(grouped)


def _specificity_inference(
    effects: tuple[float, ...],
    amendment: EvidenceSpecificityAmendment,
) -> PairedCaseInference:
    return paired_case_inference(
        effects,
        bootstrap_resamples=amendment.bootstrap_resamples,
        permutation_resamples=amendment.permutation_resamples,
        minimum_interpretable_effect=amendment.primary_minimum_interpretable_effect,
        random_seed=amendment.random_seed,
    )


def _normalized_missingness_rule(value: str) -> str:
    """Normalize the legacy spaced spelling used by the sealed v1 aggregate rows."""

    return "_".join(value.split())


def _create_case_effect(
    rows: dict[CaseComparison, CaseAggregate],
) -> EvidenceSpecificityCaseEffect | None:
    primary = rows[CaseComparison.PRIMARY_VALID]
    unrelated = rows[CaseComparison.UNRELATED_CONTROL]
    corrupted = rows[CaseComparison.CORRUPTED_CONTROL]
    if any(row.case_effect is None for row in (primary, unrelated, corrupted)):
        return None
    assert primary.case_effect is not None
    assert unrelated.case_effect is not None
    assert corrupted.case_effect is not None
    valid_vs_unrelated = primary.case_effect - unrelated.case_effect
    valid_vs_corrupted = primary.case_effect - corrupted.case_effect
    content = {
        "benchmark_version": primary.benchmark_version,
        "run_id": primary.run_id,
        "case_id": primary.case_id,
        "model_route_id": primary.model_route_id,
        "valid_vs_unrelated_effect": valid_vs_unrelated,
        "valid_vs_corrupted_effect": valid_vs_corrupted,
        "combined_specificity_effect": (valid_vs_unrelated + valid_vs_corrupted) / 2,
        "source_aggregate_hashes": (
            primary.record_hash,
            unrelated.record_hash,
            corrupted.record_hash,
        ),
    }
    return EvidenceSpecificityCaseEffect.model_validate(
        {**content, "effect_hash": canonical_sha256(content)}
    )
