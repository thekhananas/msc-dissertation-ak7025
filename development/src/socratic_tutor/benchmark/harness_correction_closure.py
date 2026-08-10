# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
"""Close benchmark interpretation after the complete post-hoc harness correction."""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from socratic_tutor.benchmark.artifacts import (
    write_immutable_bytes,
    write_immutable_json,
)
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.harness_correction_analysis import (
    HarnessCorrectionAnalysisPlan,
    HarnessCorrectionAnalysisReport,
)
from socratic_tutor.benchmark.harness_correction_replay import HarnessCorrectionReplayReport
from socratic_tutor.benchmark.hashing import file_sha256, model_content_hash
from socratic_tutor.benchmark.primary_analysis_seal import PrimaryAnalysisManifest
from socratic_tutor.benchmark.probe_correction_sensitivity import (
    ProbeCorrectionSensitivityReport,
)
from socratic_tutor.benchmark.resource_reconciliation import ResourceReconciliationReport
from socratic_tutor.benchmark.result_interpretation import ResultInterpretationReport
from socratic_tutor.contracts import ContractModel

ClaimStatus = Literal[
    "not_supported_after_complete_correction",
    "descriptive_post_hoc_only",
    "outside_study_scope",
]
OutputStatus = Literal[
    "historical_context_only",
    "superseded_for_final_interpretation",
]

CLAIM_ORDER = (
    "probe_prediction_advantage",
    "relevance_specific_advantage",
    "corrupted_evidence_contrast",
    "human_learning",
    "tutoring_efficacy",
    "reduced_cognitive_offloading",
)
OUTPUT_ORDER = (
    "sealed_primary_result",
    "three_case_probe_correction",
    "original_result_interpretation",
    "primary_result_figure_v1",
    "specificity_figure_v1",
    "dependence_figure_v1",
    "resource_figure_v1",
)


class HarnessCorrectionClosureError(ValueError):
    """Closure sources do not support one ordered final interpretation."""


class CorrectionClaim(ContractModel):
    """One result claim and the strongest wording supported after correction."""

    claim_id: str = Field(min_length=1)
    status: ClaimStatus
    estimate: float | None = Field(default=None, ge=-1.0, le=1.0)
    plain_language: str = Field(min_length=1)
    evidence_scope: str = Field(min_length=1)


class PriorOutputDisposition(ContractModel):
    """Whether an earlier result output remains suitable for final reporting."""

    output_id: str = Field(min_length=1)
    status: OutputStatus
    reason: str = Field(min_length=1)


class HarnessCorrectionClosurePlan(ContractModel):
    """Source and software identities for one correction closure."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.harness_correction_closure_plan.v1"] = (
        "benchmark.harness_correction_closure_plan.v1"
    )
    benchmark_version: Literal["v1"] = "v1"
    run_id: str = Field(min_length=1)
    original_closure_manifest_hash: Sha256
    correction_replay_report_hash: Sha256
    correction_analysis_plan_hash: Sha256
    correction_analysis_report_hash: Sha256
    partial_correction_report_hash: Sha256
    original_interpretation_report_hash: Sha256
    original_resource_report_hash: Sha256
    method: Literal["ordered_post_hoc_correction_closure_v1"] = (
        "ordered_post_hoc_correction_closure_v1"
    )
    code_revision: str = Field(min_length=1)
    pixi_lock_sha256: Sha256
    created_at_utc: datetime
    plan_hash: Sha256

    @model_validator(mode="after")
    def validate_plan(self) -> HarnessCorrectionClosurePlan:
        _require_utc(self.created_at_utc, "Correction closure plan time")
        if self.plan_hash != model_content_hash(self, exclude={"plan_hash"}):
            raise ValueError("Correction closure plan hash does not match its content")
        return self


class HarnessCorrectionClosureReport(ContractModel):
    """Final claim boundary after retaining both sealed and corrected evidence."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.harness_correction_closure_report.v1"] = (
        "benchmark.harness_correction_closure_report.v1"
    )
    benchmark_version: Literal["v1"] = "v1"
    run_id: str = Field(min_length=1)
    closure_plan_hash: Sha256
    closure_status: Literal["reclosed_after_material_post_hoc_harness_correction"] = (
        "reclosed_after_material_post_hoc_harness_correction"
    )
    total_case_count: Literal[24] = 24
    eligible_case_count: Literal[23] = 23
    missing_case_ids: tuple[Literal["h-c2m2-03"], ...] = ("h-c2m2-03",)
    changed_execution_outcome_count: int = Field(ge=1, le=48)
    original_dialogue_correct_count: int = Field(ge=0, le=23)
    original_probe_correct_count: int = Field(ge=0, le=23)
    corrected_dialogue_correct_count: int = Field(ge=0, le=23)
    corrected_probe_correct_count: int = Field(ge=0, le=23)
    original_primary_effect: float = Field(ge=-1.0, le=1.0)
    corrected_primary_effect: float = Field(ge=-1.0, le=1.0)
    primary_effect_change: float = Field(ge=-2.0, le=2.0)
    corrected_primary_interval: tuple[float, float]
    corrected_exact_mcnemar_p_value: float = Field(ge=0.0, le=1.0)
    original_combined_specificity_effect: float = Field(ge=-1.0, le=1.0)
    corrected_combined_specificity_effect: float = Field(ge=-1.0, le=1.0)
    corrected_valid_vs_unrelated_effect: float = Field(ge=-1.0, le=1.0)
    corrected_valid_vs_corrupted_effect: float = Field(ge=-1.0, le=1.0)
    claims: tuple[CorrectionClaim, ...] = Field(min_length=6, max_length=6)
    prior_outputs: tuple[PriorOutputDisposition, ...] = Field(min_length=7, max_length=7)
    claim_table_sha256: Sha256
    final_benchmark_wording: str = Field(min_length=1)
    original_seal_preserved: Literal[True] = True
    partial_correction_superseded: Literal[True] = True
    corrected_result_is_post_hoc: Literal[True] = True
    human_learning_claim_allowed: Literal[False] = False
    tutoring_efficacy_claim_allowed: Literal[False] = False
    reduced_cognitive_offloading_claim_allowed: Literal[False] = False
    network_calls_made: Literal[0] = 0
    sandbox_calls_made: Literal[0] = 0
    completed_at_utc: datetime
    report_hash: Sha256

    @model_validator(mode="after")
    def validate_report(self) -> HarnessCorrectionClosureReport:
        _require_utc(self.completed_at_utc, "Correction closure completion time")
        if self.primary_effect_change != (
            self.corrected_primary_effect - self.original_primary_effect
        ):
            raise ValueError("Correction closure effect change does not reconcile")
        if self.corrected_primary_interval[0] > self.corrected_primary_interval[1]:
            raise ValueError("Correction closure interval bounds are reversed")
        if tuple(claim.claim_id for claim in self.claims) != CLAIM_ORDER:
            raise ValueError("Correction closure claim order is incomplete")
        if tuple(output.output_id for output in self.prior_outputs) != OUTPUT_ORDER:
            raise ValueError("Correction closure output order is incomplete")
        if self.claim_table_sha256 != file_sha256(_claim_table_bytes(self.claims)):
            raise ValueError("Correction closure claim table hash does not match")
        if self.report_hash != model_content_hash(self, exclude={"report_hash"}):
            raise ValueError("Correction closure report hash does not match its content")
        return self


def close_harness_correction(
    *,
    original_closure_path: Path,
    correction_replay_report_path: Path,
    correction_analysis_plan_path: Path,
    correction_analysis_report_path: Path,
    partial_correction_report_path: Path,
    original_interpretation_report_path: Path,
    original_resource_report_path: Path,
    pixi_lock_path: Path,
    output_root: Path,
    code_revision: str,
    created_at_utc: datetime | None = None,
) -> HarnessCorrectionClosureReport:
    """Verify the complete correction chain and publish its final claim boundary."""

    original_closure = _load_json(original_closure_path, PrimaryAnalysisManifest)
    replay = _load_json(correction_replay_report_path, HarnessCorrectionReplayReport)
    analysis_plan = _load_json(correction_analysis_plan_path, HarnessCorrectionAnalysisPlan)
    analysis = _load_json(correction_analysis_report_path, HarnessCorrectionAnalysisReport)
    partial = _load_json(partial_correction_report_path, ProbeCorrectionSensitivityReport)
    interpretation = _load_json(original_interpretation_report_path, ResultInterpretationReport)
    resource = _load_json(original_resource_report_path, ResourceReconciliationReport)
    root = output_root.resolve()
    plan_path = root / "harness_correction_closure_plan.json"
    existing_plan = (
        _load_json(plan_path, HarnessCorrectionClosurePlan) if plan_path.exists() else None
    )
    effective_created_at = (
        existing_plan.created_at_utc
        if existing_plan is not None
        else created_at_utc or datetime.now(UTC)
    )
    _validate_sources(
        original_closure=original_closure,
        replay=replay,
        analysis_plan=analysis_plan,
        analysis=analysis,
        partial=partial,
        interpretation=interpretation,
        resource=resource,
        created_at_utc=effective_created_at,
    )
    try:
        lock_hash = file_sha256(pixi_lock_path.read_bytes())
    except OSError as error:
        raise HarnessCorrectionClosureError("Could not read the Pixi lock") from error
    plan = _create_plan(
        original_closure=original_closure,
        replay=replay,
        analysis_plan=analysis_plan,
        analysis=analysis,
        partial=partial,
        interpretation=interpretation,
        resource=resource,
        code_revision=code_revision,
        lock_hash=lock_hash,
        created_at_utc=effective_created_at,
    )
    if existing_plan is not None and existing_plan != plan:
        raise HarnessCorrectionClosureError(
            "Existing correction closure uses different sources or code"
        )
    write_immutable_json(plan_path, plan)
    report_path = root / "harness_correction_closure_report.json"
    if report_path.exists():
        report = _load_json(report_path, HarnessCorrectionClosureReport)
        if report.closure_plan_hash != plan.plan_hash:
            raise HarnessCorrectionClosureError("Existing closure belongs to another plan")
        return report

    claims = _build_claims(analysis)
    prior_outputs = _build_prior_outputs()
    claim_table = _claim_table_bytes(claims)
    original_dialogue, original_probe = _original_condition_counts(resource)
    corrected_dialogue = _condition_correct_count(analysis, "dialogue_only")
    corrected_probe = _condition_correct_count(analysis, "probe_informed")
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.harness_correction_closure_report.v1",
        "benchmark_version": "v1",
        "run_id": original_closure.run_id,
        "closure_plan_hash": plan.plan_hash,
        "closure_status": "reclosed_after_material_post_hoc_harness_correction",
        "total_case_count": 24,
        "eligible_case_count": 23,
        "missing_case_ids": ("h-c2m2-03",),
        "changed_execution_outcome_count": replay.changed_outcome_count,
        "original_dialogue_correct_count": original_dialogue,
        "original_probe_correct_count": original_probe,
        "corrected_dialogue_correct_count": corrected_dialogue,
        "corrected_probe_correct_count": corrected_probe,
        "original_primary_effect": analysis.original_sealed_primary_effect,
        "corrected_primary_effect": analysis.corrected_primary_effect,
        "primary_effect_change": (
            analysis.corrected_primary_effect - analysis.original_sealed_primary_effect
        ),
        "corrected_primary_interval": (
            analysis.primary_inference.interval_lower,
            analysis.primary_inference.interval_upper,
        ),
        "corrected_exact_mcnemar_p_value": (analysis.primary_mcnemar.two_sided_exact_p_value),
        "original_combined_specificity_effect": (
            analysis.original_sealed_combined_specificity_effect
        ),
        "corrected_combined_specificity_effect": (analysis.corrected_combined_specificity_effect),
        "corrected_valid_vs_unrelated_effect": (
            analysis.valid_vs_unrelated_inference.mean_case_effect
        ),
        "corrected_valid_vs_corrupted_effect": (
            analysis.valid_vs_corrupted_inference.mean_case_effect
        ),
        "claims": claims,
        "prior_outputs": prior_outputs,
        "claim_table_sha256": file_sha256(claim_table),
        "final_benchmark_wording": (
            "After correcting a systematic sequence-comparison fault, executable "
            "probe evidence improved the tracker decision in 1 of 23 eligible "
            "authored cases and did not outperform unrelated passing evidence. "
            "This post-hoc result does not support a practical or relevance-specific "
            "predictive advantage."
        ),
        "original_seal_preserved": True,
        "partial_correction_superseded": True,
        "corrected_result_is_post_hoc": True,
        "human_learning_claim_allowed": False,
        "tutoring_efficacy_claim_allowed": False,
        "reduced_cognitive_offloading_claim_allowed": False,
        "network_calls_made": 0,
        "sandbox_calls_made": 0,
        "completed_at_utc": effective_created_at,
    }
    draft = HarnessCorrectionClosureReport.model_construct(
        _fields_set=set(content), **content, report_hash="0" * 64
    )
    report = HarnessCorrectionClosureReport.model_validate(
        {**content, "report_hash": model_content_hash(draft, exclude={"report_hash"})}
    )
    write_immutable_bytes(root / "claim_status.csv", claim_table)
    write_immutable_json(report_path, report)
    return report


def _build_claims(
    analysis: HarnessCorrectionAnalysisReport,
) -> tuple[CorrectionClaim, ...]:
    return (
        CorrectionClaim(
            claim_id="probe_prediction_advantage",
            status="not_supported_after_complete_correction",
            estimate=analysis.corrected_primary_effect,
            plain_language=(
                "The corrected probe-informed tracker improved only one of 23 "
                "eligible predictions over dialogue alone."
            ),
            evidence_scope="One pinned model run over 24 authored cases; 23 eligible.",
        ),
        CorrectionClaim(
            claim_id="relevance_specific_advantage",
            status="not_supported_after_complete_correction",
            estimate=analysis.valid_vs_unrelated_inference.mean_case_effect,
            plain_language=(
                "Relevant and unrelated passing probes produced the same tracker "
                "decisions on every eligible case."
            ),
            evidence_scope="Post-hoc reconstruction from the complete corrected replay.",
        ),
        CorrectionClaim(
            claim_id="corrupted_evidence_contrast",
            status="descriptive_post_hoc_only",
            estimate=analysis.valid_vs_corrupted_inference.mean_case_effect,
            plain_language=(
                "Valid passing probes outperformed deliberately inverted probe "
                "outcomes, but this does not show sensitivity to relevance."
            ),
            evidence_scope="Post-hoc contrast using a deterministic corrupted control.",
        ),
        CorrectionClaim(
            claim_id="human_learning",
            status="outside_study_scope",
            estimate=None,
            plain_language="This benchmark did not measure learning by human students.",
            evidence_scope="No human participants or longitudinal learning outcome.",
        ),
        CorrectionClaim(
            claim_id="tutoring_efficacy",
            status="outside_study_scope",
            estimate=None,
            plain_language="This benchmark did not compare tutoring outcomes.",
            evidence_scope="Evaluation-model predictions, not a tutoring trial.",
        ),
        CorrectionClaim(
            claim_id="reduced_cognitive_offloading",
            status="outside_study_scope",
            estimate=None,
            plain_language="This benchmark did not measure dependence on an AI tutor.",
            evidence_scope="No unaided human transfer or delayed-retention measure.",
        ),
    )


def _build_prior_outputs() -> tuple[PriorOutputDisposition, ...]:
    historical = "historical_context_only"
    superseded = "superseded_for_final_interpretation"
    return (
        PriorOutputDisposition(
            output_id="sealed_primary_result",
            status=historical,
            reason="Preserve as the result produced by the frozen original harness.",
        ),
        PriorOutputDisposition(
            output_id="three_case_probe_correction",
            status=superseded,
            reason="It corrected three regressions but did not replay favourable cases.",
        ),
        PriorOutputDisposition(
            output_id="original_result_interpretation",
            status=superseded,
            reason="It predates the complete harness correction.",
        ),
        PriorOutputDisposition(
            output_id="primary_result_figure_v1",
            status=historical,
            reason="It may illustrate the original sealed result only when paired with correction.",
        ),
        PriorOutputDisposition(
            output_id="specificity_figure_v1",
            status=historical,
            reason="It shows the original harness result, not the corrected relevance contrast.",
        ),
        PriorOutputDisposition(
            output_id="dependence_figure_v1",
            status=superseded,
            reason="Its sensitivity estimates derive from the original harness outcomes.",
        ),
        PriorOutputDisposition(
            output_id="resource_figure_v1",
            status=superseded,
            reason="Its 23-of-23 corrected accuracy used the incomplete three-case correction.",
        ),
    )


def _claim_table_bytes(claims: tuple[CorrectionClaim, ...]) -> bytes:
    output = StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(("claim_id", "status", "estimate", "plain_language", "evidence_scope"))
    for claim in claims:
        writer.writerow(
            (
                claim.claim_id,
                claim.status,
                "" if claim.estimate is None else f"{claim.estimate:.6f}",
                claim.plain_language,
                claim.evidence_scope,
            )
        )
    return output.getvalue().encode("utf-8")


def _condition_correct_count(analysis: HarnessCorrectionAnalysisReport, condition: str) -> int:
    return sum(
        score.correct is True
        for case in analysis.case_results
        if case.primary_effect is not None
        for score in case.conditions
        if score.condition.value == condition
    )


def _original_condition_counts(resource: ResourceReconciliationReport) -> tuple[int, int]:
    counts = {item.condition: item.correct_case_count for item in resource.condition_burden}
    return counts["dialogue_only"], counts["probe_informed_observed"]


def _create_plan(
    *,
    original_closure: PrimaryAnalysisManifest,
    replay: HarnessCorrectionReplayReport,
    analysis_plan: HarnessCorrectionAnalysisPlan,
    analysis: HarnessCorrectionAnalysisReport,
    partial: ProbeCorrectionSensitivityReport,
    interpretation: ResultInterpretationReport,
    resource: ResourceReconciliationReport,
    code_revision: str,
    lock_hash: Sha256,
    created_at_utc: datetime,
) -> HarnessCorrectionClosurePlan:
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.harness_correction_closure_plan.v1",
        "benchmark_version": "v1",
        "run_id": original_closure.run_id,
        "original_closure_manifest_hash": original_closure.manifest_hash,
        "correction_replay_report_hash": replay.report_hash,
        "correction_analysis_plan_hash": analysis_plan.plan_hash,
        "correction_analysis_report_hash": analysis.report_hash,
        "partial_correction_report_hash": partial.report_hash,
        "original_interpretation_report_hash": interpretation.report_hash,
        "original_resource_report_hash": resource.report_hash,
        "method": "ordered_post_hoc_correction_closure_v1",
        "code_revision": code_revision,
        "pixi_lock_sha256": lock_hash,
        "created_at_utc": created_at_utc,
    }
    draft = HarnessCorrectionClosurePlan.model_construct(
        _fields_set=set(content), **content, plan_hash="0" * 64
    )
    return HarnessCorrectionClosurePlan.model_validate(
        {**content, "plan_hash": model_content_hash(draft, exclude={"plan_hash"})}
    )


def _validate_sources(
    *,
    original_closure: PrimaryAnalysisManifest,
    replay: HarnessCorrectionReplayReport,
    analysis_plan: HarnessCorrectionAnalysisPlan,
    analysis: HarnessCorrectionAnalysisReport,
    partial: ProbeCorrectionSensitivityReport,
    interpretation: ResultInterpretationReport,
    resource: ResourceReconciliationReport,
    created_at_utc: datetime,
) -> None:
    _require_utc(created_at_utc, "Correction closure time")
    bindings = {item.role: item.content_hash for item in original_closure.artifacts}
    if not replay.gate_passed or replay.provider_call_count != 0:
        raise HarnessCorrectionClosureError("Complete correction replay did not pass")
    if analysis.analysis_plan_hash != analysis_plan.plan_hash:
        raise HarnessCorrectionClosureError("Correction analysis belongs to another plan")
    if analysis_plan.correction_replay_report_hash != replay.report_hash:
        raise HarnessCorrectionClosureError("Correction analysis names another replay")
    if analysis_plan.original_primary_report_hash != bindings.get("primary_report"):
        raise HarnessCorrectionClosureError("Correction analysis names another primary report")
    if analysis_plan.original_secondary_report_hash != bindings.get("secondary_report"):
        raise HarnessCorrectionClosureError("Correction analysis names another secondary report")
    if partial.report_hash != bindings.get("correction_report"):
        raise HarnessCorrectionClosureError("Partial correction is outside the original closure")
    if interpretation.report_hash != bindings.get("interpretation_report"):
        raise HarnessCorrectionClosureError("Interpretation is outside the original closure")
    if resource.report_hash != bindings.get("resource_report"):
        raise HarnessCorrectionClosureError("Resource report is outside the original closure")
    run_ids = {
        original_closure.run_id,
        partial.run_id,
        interpretation.run_id,
        resource.run_id,
    }
    if len(run_ids) != 1:
        raise HarnessCorrectionClosureError("Correction closure sources use different runs")
    if (
        abs(partial.original.paired_effect_from_counts - analysis.original_sealed_primary_effect)
        > 1e-12
    ):
        raise HarnessCorrectionClosureError("Original primary effects do not reconcile")
    if abs(interpretation.primary_mean_effect - analysis.original_sealed_primary_effect) > 1e-12:
        raise HarnessCorrectionClosureError("Original interpretation effect does not reconcile")
    if (
        abs(
            interpretation.specificity_mean_effect
            - analysis.original_sealed_combined_specificity_effect
        )
        > 1e-12
    ):
        raise HarnessCorrectionClosureError("Original specificity effects do not reconcile")
    if partial.corrected.paired_effect_from_counts == analysis.corrected_primary_effect:
        raise HarnessCorrectionClosureError("Partial and complete corrections must remain distinct")
    burdens = {item.condition: item for item in resource.condition_burden}
    if burdens["probe_informed_post_hoc_corrected"].correct_case_count != 23:
        raise HarnessCorrectionClosureError("Expected the documented partial correction result")
    if analysis.eligible_case_count != 23 or analysis.missing_case_ids != ("h-c2m2-03",):
        raise HarnessCorrectionClosureError("Complete correction uses an unexpected denominator")
    if _condition_correct_count(analysis, "dialogue_only") != 18:
        raise HarnessCorrectionClosureError("Corrected dialogue count does not reconcile")
    if _condition_correct_count(analysis, "probe_informed") != 19:
        raise HarnessCorrectionClosureError("Corrected probe count does not reconcile")
    if analysis.valid_vs_unrelated_inference.mean_case_effect != 0.0:
        raise HarnessCorrectionClosureError(
            "Corrected relevance contrast is not the reviewed result"
        )
    latest_source_time = max(
        original_closure.created_at_utc,
        replay.completed_at_utc,
        analysis.completed_at_utc,
        partial.completed_at_utc,
        interpretation.completed_at_utc,
        resource.completed_at_utc,
    )
    if created_at_utc <= latest_source_time:
        raise HarnessCorrectionClosureError("Correction closure must follow every source")


def _load_json[ModelT: ContractModel](path: Path, model: type[ModelT]) -> ModelT:
    try:
        return model.model_validate_json(path.read_bytes())
    except (OSError, ValueError) as error:
        raise HarnessCorrectionClosureError(f"Could not verify closure source: {path}") from error


def _require_utc(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError(f"{name} must use UTC")
