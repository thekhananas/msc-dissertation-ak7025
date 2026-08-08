"""Fail-closed command interface for benchmark validation and rehearsals."""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import BaseModel, Field, model_validator

from socratic_tutor.benchmark.analysis_spec import (
    analysis_specification_hash,
    load_analysis_specification,
)
from socratic_tutor.benchmark.artifacts import write_immutable_json
from socratic_tutor.benchmark.behavior_audit import (
    load_student_behavior_audit_plan,
    run_student_behavior_audit_from_environment,
)
from socratic_tutor.benchmark.binary_sensitivity import (
    load_binary_sensitivity_plan,
    run_binary_sensitivity,
)
from socratic_tutor.benchmark.calibration import (
    load_uncalibrated_decision_plan,
    record_uncalibrated_decision,
)
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.dependence_figure import render_dependence_figure
from socratic_tutor.benchmark.dependence_sensitivity import run_dependence_sensitivity
from socratic_tutor.benchmark.design import load_design
from socratic_tutor.benchmark.evaluator.loader import load_and_verify_manifest
from socratic_tutor.benchmark.evaluator.models import ManifestStatus
from socratic_tutor.benchmark.evaluator.offline import (
    OfflineCriterionPlan,
    run_offline_criterion_phase,
)
from socratic_tutor.benchmark.evaluator.projection import (
    project_evaluator_manifest,
    project_public_manifest,
)
from socratic_tutor.benchmark.evaluator.reporting import evaluate_published_run
from socratic_tutor.benchmark.evidence_scoring import score_evidence_responses
from socratic_tutor.benchmark.evidence_specificity import (
    create_external_run_preflight,
    freeze_evidence_specificity_amendment,
    load_evidence_specificity_amendment,
    load_evidence_specificity_amendment_plan,
    load_external_run_preflight,
    load_uncalibrated_decision_report,
)
from socratic_tutor.benchmark.external_audit import audit_external_decision_seal
from socratic_tutor.benchmark.external_criterion import (
    run_external_criterion_from_environment,
)
from socratic_tutor.benchmark.external_criterion_audit import audit_external_criterion_run
from socratic_tutor.benchmark.external_decision import (
    load_external_decision_generation_plan,
    run_external_decision_generation_from_environment,
)
from socratic_tutor.benchmark.external_diagnostics import run_external_diagnostics
from socratic_tutor.benchmark.external_protocol import (
    freeze_external_model_execution_protocol,
    load_external_model_execution_protocol,
    load_external_model_protocol_freeze_plan,
)
from socratic_tutor.benchmark.external_rehearsal import (
    load_external_route_rehearsal_plan,
    run_external_route_rehearsal_from_environment,
)
from socratic_tutor.benchmark.external_replay import run_external_preanalysis_replay
from socratic_tutor.benchmark.external_scoring import run_external_scoring
from socratic_tutor.benchmark.external_seal import run_external_decision_seal_with_modal
from socratic_tutor.benchmark.failure_figure import render_failure_figure
from socratic_tutor.benchmark.failure_review import prepare_failure_review, record_failure_review
from socratic_tutor.benchmark.failure_review_reliability import (
    run_failure_review_reliability,
)
from socratic_tutor.benchmark.failure_taxonomy import (
    failure_taxonomy_hash,
    load_failure_taxonomy,
)
from socratic_tutor.benchmark.final_replay import run_final_replay
from socratic_tutor.benchmark.freeze import prepare_benchmark_freeze
from socratic_tutor.benchmark.hashing import model_content_hash
from socratic_tutor.benchmark.inferential_hierarchy import (
    freeze_inferential_hierarchy,
    load_binary_sensitivity_report,
    load_inferential_hierarchy_plan,
    load_methodology_clarification,
    load_public_rating_procedure_record,
)
from socratic_tutor.benchmark.methodology_clarification import (
    freeze_methodology_clarification,
    load_methodology_clarification_plan,
    load_public_answer_rating_report,
)
from socratic_tutor.benchmark.missingness_sensitivity import run_missingness_sensitivity
from socratic_tutor.benchmark.negative_result_audit import run_negative_result_audit
from socratic_tutor.benchmark.primary_analysis import run_primary_analysis
from socratic_tutor.benchmark.primary_analysis_seal import seal_primary_analysis
from socratic_tutor.benchmark.primary_figure import render_primary_result_figure
from socratic_tutor.benchmark.probe_correction_sensitivity import (
    run_probe_correction_sensitivity,
)
from socratic_tutor.benchmark.public.offline import (
    OfflineDecisionPlan,
    run_offline_decision_phase,
)
from socratic_tutor.benchmark.public_rating import (
    build_public_answer_rating_packet,
    export_public_rating_workbook,
    finalize_public_answer_ratings,
    freeze_public_rating_boundary,
    load_public_answer_rating_guide,
    load_public_answer_rating_packet,
    load_public_rating_guide_plan,
    load_public_rating_protocol_amendment,
    record_public_answer_ratings,
)
from socratic_tutor.benchmark.public_rating_procedure import (
    freeze_public_rating_procedure,
    load_public_rating_procedure_plan,
)
from socratic_tutor.benchmark.public_rating_procedure import (
    load_public_answer_rating_report as load_procedure_rating_report,
)
from socratic_tutor.benchmark.qualification import (
    load_provider_qualification_plan,
    run_provider_qualification_from_environment,
)
from socratic_tutor.benchmark.rating_reliability import run_rating_reliability
from socratic_tutor.benchmark.readiness import build_benchmark_readiness_report
from socratic_tutor.benchmark.reference_integrity import (
    load_reference_integrity_plan,
    run_reference_integrity_replay,
)
from socratic_tutor.benchmark.research_checks import (
    CaseLinkedShortcutAuditPlan,
    SensitivityPlan,
    ShortcutAuditPlan,
    run_case_linked_shortcut_audit,
    run_sensitivity_analysis,
    run_shortcut_audit,
)
from socratic_tutor.benchmark.resource_figure import render_resource_figure
from socratic_tutor.benchmark.resource_reconciliation import run_resource_reconciliation
from socratic_tutor.benchmark.result_interpretation import run_result_interpretation
from socratic_tutor.benchmark.sandbox_rehearsal import (
    load_sandbox_rehearsal_plan,
    run_recorded_sandbox_rehearsal_with_modal,
)
from socratic_tutor.benchmark.secondary_analysis import run_secondary_analysis
from socratic_tutor.benchmark.specificity_figure import render_specificity_figure
from socratic_tutor.contracts import ContractModel


class SmokeManifest(ContractModel):
    """Final marker for one complete two-phase offline rehearsal."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.smoke_manifest.v1"] = "benchmark.smoke_manifest.v1"
    benchmark_version: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    source_manifest_hash: Sha256
    public_projection_hash: Sha256
    evaluator_projection_hash: Sha256
    decision_summary_hash: Sha256
    criterion_summary_hash: Sha256
    evaluation_report_hash: Sha256
    smoke_hash: Sha256

    @model_validator(mode="after")
    def validate_manifest(self) -> "SmokeManifest":
        if self.smoke_hash != model_content_hash(self, exclude={"smoke_hash"}):
            raise ValueError("Smoke manifest hash does not match its content")
        return self


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="Validate one authored manifest")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--require-frozen", action="store_true")

    smoke = commands.add_parser("smoke", help="Run the two-case offline pipeline")
    smoke.add_argument("--manifest", type=Path, required=True)
    smoke.add_argument("--decision-plan", type=Path, required=True)
    smoke.add_argument("--criterion-plan", type=Path, required=True)
    smoke.add_argument("--output-root", type=Path, required=True)

    shortcuts = commands.add_parser("shortcuts", help="Run the lexical shortcut baseline")
    shortcuts.add_argument("--input", type=Path, required=True)
    shortcuts.add_argument("--output", type=Path, required=True)

    case_shortcuts = commands.add_parser(
        "case-shortcuts", help="Run the case-linked lexical retrieval baseline"
    )
    case_shortcuts.add_argument("--manifest", type=Path, required=True)
    case_shortcuts.add_argument("--input", type=Path, required=True)
    case_shortcuts.add_argument("--output", type=Path, required=True)

    readiness = commands.add_parser(
        "readiness", help="Build the pre-freeze authored-content validation report"
    )
    readiness.add_argument("--manifest", type=Path, required=True)
    readiness.add_argument("--case-shortcut-summary", type=Path, required=True)
    readiness.add_argument("--output", type=Path, required=True)

    freeze_prepare = commands.add_parser(
        "freeze-prepare", help="Validate freeze inputs and calculate a frozen manifest digest"
    )
    freeze_prepare.add_argument("--manifest", type=Path, required=True)
    freeze_prepare.add_argument("--readiness-report", type=Path, required=True)
    freeze_prepare.add_argument("--analysis-specification", type=Path, required=True)
    freeze_prepare.add_argument("--frozen-at-utc", type=_utc_datetime, required=True)
    freeze_prepare.add_argument("--output", type=Path, required=True)

    evidence = commands.add_parser("evidence-score", help="Score recorded evidence responses")
    evidence.add_argument("--manifest", type=Path, required=True)
    evidence.add_argument("--responses", type=Path, required=True)
    evidence.add_argument("--output", type=Path, required=True)

    sensitivity = commands.add_parser("sensitivity", help="Run seeded design sensitivity")
    sensitivity.add_argument("--input", type=Path, required=True)
    sensitivity.add_argument("--output", type=Path, required=True)

    binary_sensitivity = commands.add_parser(
        "binary-sensitivity",
        help="Run the corrected sensitivity study for 24 paired binary cases",
    )
    binary_sensitivity.add_argument("--input", type=Path, required=True)
    binary_sensitivity.add_argument("--output", type=Path, required=True)

    calibration = commands.add_parser(
        "calibration-decision",
        help="Record the allowed scoring metric before held-out criterion access",
    )
    calibration.add_argument("--input", type=Path, required=True)
    calibration.add_argument("--heldout-manifest", type=Path, required=True)
    calibration.add_argument("--benchmark-root", type=Path, required=True)
    calibration.add_argument("--analysis-specification", type=Path, required=True)
    calibration.add_argument("--output", type=Path, required=True)

    taxonomy = commands.add_parser("failure-taxonomy-validate", help="Validate review taxonomy")
    taxonomy.add_argument("--input", type=Path, required=True)

    analysis = commands.add_parser(
        "analysis-spec-validate", help="Validate pre-registered analysis rules"
    )
    analysis.add_argument("--input", type=Path, required=True)

    qualification = commands.add_parser(
        "provider-qualify", help="Run one direct provider against development-only public cases"
    )
    qualification.add_argument("--plan", type=Path, required=True)
    qualification.add_argument("--manifest", type=Path, required=True)
    qualification.add_argument("--output-root", type=Path, required=True)
    qualification.add_argument("--run-id", required=True)

    behavior_audit = commands.add_parser(
        "student-behavior-audit",
        help="Generate development-only learner-state responses for manual review",
    )
    behavior_audit.add_argument("--plan", type=Path, required=True)
    behavior_audit.add_argument("--output-root", type=Path, required=True)
    behavior_audit.add_argument("--run-id", required=True)

    integrity = commands.add_parser(
        "reference-integrity",
        help="Run the network-free, non-empirical v1 integrity replay",
    )
    integrity.add_argument("--plan", type=Path, required=True)
    integrity.add_argument("--manifest", type=Path, required=True)
    integrity.add_argument("--output-root", type=Path, required=True)

    protocol = commands.add_parser(
        "external-protocol-freeze",
        help="Freeze v2 external-model settings before held-out provider calls",
    )
    protocol.add_argument("--plan", type=Path, required=True)
    protocol.add_argument("--manifest", type=Path, required=True)
    protocol.add_argument("--analysis-specification", type=Path, required=True)
    protocol.add_argument("--system-prompt", type=Path, required=True)
    protocol.add_argument("--output", type=Path, required=True)

    rehearsal = commands.add_parser(
        "external-route-rehearse",
        help="Call every isolated channel on two development cases only",
    )
    rehearsal.add_argument("--protocol", type=Path, required=True)
    rehearsal.add_argument("--plan", type=Path, required=True)
    rehearsal.add_argument("--manifest", type=Path, required=True)
    rehearsal.add_argument("--fixture-decision-plan", type=Path, required=True)
    rehearsal.add_argument("--system-prompt", type=Path, required=True)
    rehearsal.add_argument("--output-root", type=Path, required=True)
    rehearsal.add_argument("--run-id", required=True)

    sandbox_rehearsal = commands.add_parser(
        "sandbox-rehearse",
        help="Replay development responses and execute only their code-bearing channels remotely",
    )
    sandbox_rehearsal.add_argument("--plan", type=Path, required=True)
    sandbox_rehearsal.add_argument("--manifest", type=Path, required=True)
    sandbox_rehearsal.add_argument("--recorded-responses", type=Path, required=True)
    sandbox_rehearsal.add_argument("--output-root", type=Path, required=True)
    sandbox_rehearsal.add_argument("--run-id", required=True)

    rating_freeze = commands.add_parser(
        "public-rating-freeze",
        help="Freeze the public-answer rating guide and bind it to the external protocol",
    )
    rating_freeze.add_argument("--plan", type=Path, required=True)
    rating_freeze.add_argument("--protocol", type=Path, required=True)
    rating_freeze.add_argument("--guide-output", type=Path, required=True)
    rating_freeze.add_argument("--amendment-output", type=Path, required=True)

    rating_packet = commands.add_parser(
        "public-rating-packet",
        help="Create a blinded packet from public responses only",
    )
    rating_packet.add_argument("--protocol", type=Path, required=True)
    rating_packet.add_argument("--amendment", type=Path, required=True)
    rating_packet.add_argument("--recorded-responses", type=Path, required=True)
    rating_packet.add_argument("--output", type=Path, required=True)

    rating_workbook = commands.add_parser(
        "public-rating-workbook",
        help="Export a blinded guide and CSV for two independent public-answer raters",
    )
    rating_workbook.add_argument("--guide", type=Path, required=True)
    rating_workbook.add_argument("--packet", type=Path, required=True)
    rating_workbook.add_argument("--output-root", type=Path, required=True)

    rating_record = commands.add_parser(
        "public-rating-record",
        help="Validate one completed rating CSV and create immutable rating JSONL",
    )
    rating_record.add_argument("--guide", type=Path, required=True)
    rating_record.add_argument("--packet", type=Path, required=True)
    rating_record.add_argument("--completed-sheet", type=Path, required=True)
    rating_record.add_argument("--rater-id", required=True)
    rating_record.add_argument("--output", type=Path, required=True)

    rating_finalize = commands.add_parser(
        "public-rating-finalize",
        help="Measure agreement and resolve two independent public-answer rating sets",
    )
    rating_finalize.add_argument("--guide", type=Path, required=True)
    rating_finalize.add_argument("--amendment", type=Path, required=True)
    rating_finalize.add_argument("--packet", type=Path, required=True)
    rating_finalize.add_argument("--ratings-a", type=Path, required=True)
    rating_finalize.add_argument("--ratings-b", type=Path, required=True)
    rating_finalize.add_argument("--adjudications", type=Path)
    rating_finalize.add_argument("--output", type=Path, required=True)

    rating_procedure = commands.add_parser(
        "public-rating-procedure-freeze",
        help="Freeze rater roles, independence, access, tools, order, and adjudication plan",
    )
    rating_procedure.add_argument("--plan", type=Path, required=True)
    rating_procedure.add_argument("--rating-report", type=Path, required=True)
    rating_procedure.add_argument("--output", type=Path, required=True)

    specificity_freeze = commands.add_parser(
        "evidence-specificity-freeze",
        help="Freeze the secondary evidence-specificity equations and inherited inference rules",
    )
    specificity_freeze.add_argument("--plan", type=Path, required=True)
    specificity_freeze.add_argument("--protocol", type=Path, required=True)
    specificity_freeze.add_argument("--analysis-specification", type=Path, required=True)
    specificity_freeze.add_argument("--calibration-report", type=Path, required=True)
    specificity_freeze.add_argument("--output", type=Path, required=True)

    external_preflight = commands.add_parser(
        "external-run-preflight",
        help="Require both frozen amendments before a held-out external run",
    )
    external_preflight.add_argument("--protocol", type=Path, required=True)
    external_preflight.add_argument("--analysis-specification", type=Path, required=True)
    external_preflight.add_argument("--calibration-report", type=Path, required=True)
    external_preflight.add_argument("--public-rating-amendment", type=Path, required=True)
    external_preflight.add_argument("--specificity-amendment", type=Path, required=True)
    external_preflight.add_argument("--output", type=Path, required=True)

    methodology = commands.add_parser(
        "methodology-clarification-freeze",
        help="Freeze the exact design and claim boundary before criterion access",
    )
    methodology.add_argument("--plan", type=Path, required=True)
    methodology.add_argument("--protocol", type=Path, required=True)
    methodology.add_argument("--analysis-specification", type=Path, required=True)
    methodology.add_argument("--calibration-report", type=Path, required=True)
    methodology.add_argument("--public-rating-report", type=Path, required=True)
    methodology.add_argument("--output", type=Path, required=True)

    hierarchy = commands.add_parser(
        "inferential-hierarchy-freeze",
        help="Freeze the primary analysis order and secondary dependence checks",
    )
    hierarchy.add_argument("--plan", type=Path, required=True)
    hierarchy.add_argument("--analysis-specification", type=Path, required=True)
    hierarchy.add_argument("--methodology-clarification", type=Path, required=True)
    hierarchy.add_argument("--binary-sensitivity-report", type=Path, required=True)
    hierarchy.add_argument("--public-rating-procedure", type=Path, required=True)
    hierarchy.add_argument("--specificity-amendment", type=Path, required=True)
    hierarchy.add_argument("--benchmark-design", type=Path, required=True)
    hierarchy.add_argument("--output", type=Path, required=True)

    external_decision = commands.add_parser(
        "external-decision-generate",
        help="Generate resumable held-out public and evidence responses before criterion access",
    )
    external_decision.add_argument("--plan", type=Path, required=True)
    external_decision.add_argument("--protocol", type=Path, required=True)
    external_decision.add_argument("--preflight", type=Path, required=True)
    external_decision.add_argument("--manifest", type=Path, required=True)
    external_decision.add_argument("--analysis-specification", type=Path, required=True)
    external_decision.add_argument("--system-prompt", type=Path, required=True)
    external_decision.add_argument("--output-root", type=Path, required=True)
    external_decision.add_argument("--run-id", required=True)

    external_seal = commands.add_parser(
        "external-decision-seal",
        help="Execute recorded evidence and seal held-out decisions before criterion access",
    )
    external_seal.add_argument("--protocol", type=Path, required=True)
    external_seal.add_argument("--preflight", type=Path, required=True)
    external_seal.add_argument("--generation-report", type=Path, required=True)
    external_seal.add_argument("--recorded-responses", type=Path, required=True)
    external_seal.add_argument("--public-rating-report", type=Path, required=True)
    external_seal.add_argument("--methodology-clarification", type=Path, required=True)
    external_seal.add_argument("--inferential-hierarchy", type=Path, required=True)
    external_seal.add_argument("--manifest", type=Path, required=True)
    external_seal.add_argument("--analysis-specification", type=Path, required=True)
    external_seal.add_argument("--pixi-lock", type=Path, required=True)
    external_seal.add_argument("--output-root", type=Path, required=True)
    external_seal.add_argument("--code-revision", required=True)
    external_seal.add_argument("--dirty-worktree", action="store_true")
    external_seal.add_argument("--created-at-utc", type=_utc_datetime, required=True)

    external_audit = commands.add_parser(
        "external-decision-audit",
        help="Reconcile a held-out decision seal before criterion access",
    )
    external_audit.add_argument("--protocol", type=Path, required=True)
    external_audit.add_argument("--generation-report", type=Path, required=True)
    external_audit.add_argument("--recorded-responses", type=Path, required=True)
    external_audit.add_argument("--public-rating-report", type=Path, required=True)
    external_audit.add_argument("--manifest", type=Path, required=True)
    external_audit.add_argument("--seal-root", type=Path, required=True)
    external_audit.add_argument("--pixi-lock", type=Path, required=True)
    external_audit.add_argument("--output", type=Path, required=True)
    external_audit.add_argument("--audit-code-revision", required=True)
    external_audit.add_argument("--audited-at-utc", type=_utc_datetime, required=True)

    external_criterion = commands.add_parser(
        "external-criterion-run",
        help="Generate and execute criterion responses after an audited decision seal",
    )
    external_criterion.add_argument("--protocol", type=Path, required=True)
    external_criterion.add_argument("--accounting-report", type=Path, required=True)
    external_criterion.add_argument("--manifest", type=Path, required=True)
    external_criterion.add_argument("--system-prompt", type=Path, required=True)
    external_criterion.add_argument("--seal-root", type=Path, required=True)
    external_criterion.add_argument("--pixi-lock", type=Path, required=True)
    external_criterion.add_argument("--code-revision", required=True)
    external_criterion.add_argument("--created-at-utc", type=_utc_datetime, required=True)

    criterion_audit = commands.add_parser(
        "external-criterion-audit",
        help="Verify route, label, missingness, and reveal integrity without scoring",
    )
    criterion_audit.add_argument("--protocol", type=Path, required=True)
    criterion_audit.add_argument("--accounting-report", type=Path, required=True)
    criterion_audit.add_argument("--manifest", type=Path, required=True)
    criterion_audit.add_argument("--system-prompt", type=Path, required=True)
    criterion_audit.add_argument("--seal-root", type=Path, required=True)
    criterion_audit.add_argument("--pixi-lock", type=Path, required=True)
    criterion_audit.add_argument("--output", type=Path, required=True)
    criterion_audit.add_argument("--audit-code-revision", required=True)
    criterion_audit.add_argument("--audited-at-utc", type=_utc_datetime, required=True)

    external_replay = commands.add_parser(
        "external-replay",
        help="Replay all external responses and pre-analysis datasets without network access",
    )
    external_replay.add_argument("--integrity-report", type=Path, required=True)
    external_replay.add_argument("--decision-responses", type=Path, required=True)
    external_replay.add_argument("--seal-root", type=Path, required=True)
    external_replay.add_argument("--pixi-lock", type=Path, required=True)
    external_replay.add_argument("--output-root", type=Path, required=True)
    external_replay.add_argument("--code-revision", required=True)
    external_replay.add_argument("--created-at-utc", type=_utc_datetime, required=True)

    external_score = commands.add_parser(
        "external-score",
        help="Score one audited and replayed external run without external calls",
    )
    external_score.add_argument("--replay-report", type=Path, required=True)
    external_score.add_argument("--analysis-specification", type=Path, required=True)
    external_score.add_argument("--calibration-report", type=Path, required=True)
    external_score.add_argument("--public-manifest", type=Path, required=True)
    external_score.add_argument("--benchmark-root", type=Path, required=True)
    external_score.add_argument("--seal-root", type=Path, required=True)
    external_score.add_argument("--pixi-lock", type=Path, required=True)
    external_score.add_argument("--code-revision", required=True)
    external_score.add_argument("--created-at-utc", type=_utc_datetime, required=True)

    primary_analysis = commands.add_parser(
        "external-primary-analyze",
        help="Compute the frozen primary paired result from scored external datasets",
    )
    primary_analysis.add_argument("--scoring-summary", type=Path, required=True)
    primary_analysis.add_argument("--analysis-specification", type=Path, required=True)
    primary_analysis.add_argument("--inferential-hierarchy", type=Path, required=True)
    primary_analysis.add_argument("--dataset-root", type=Path, required=True)
    primary_analysis.add_argument("--pixi-lock", type=Path, required=True)
    primary_analysis.add_argument("--output-root", type=Path, required=True)
    primary_analysis.add_argument("--code-revision", required=True)
    primary_analysis.add_argument("--created-at-utc", type=_utc_datetime, required=True)

    secondary_analysis = commands.add_parser(
        "external-secondary-analyze",
        help="Compute frozen baseline, control, and evidence-specificity comparisons",
    )
    secondary_analysis.add_argument("--scoring-summary", type=Path, required=True)
    secondary_analysis.add_argument("--primary-plan", type=Path, required=True)
    secondary_analysis.add_argument("--primary-report", type=Path, required=True)
    secondary_analysis.add_argument("--analysis-specification", type=Path, required=True)
    secondary_analysis.add_argument("--inferential-hierarchy", type=Path, required=True)
    secondary_analysis.add_argument("--specificity-amendment", type=Path, required=True)
    secondary_analysis.add_argument("--dataset-root", type=Path, required=True)
    secondary_analysis.add_argument("--pixi-lock", type=Path, required=True)
    secondary_analysis.add_argument("--output-root", type=Path, required=True)
    secondary_analysis.add_argument("--code-revision", required=True)
    secondary_analysis.add_argument("--created-at-utc", type=_utc_datetime, required=True)

    dependence_sensitivity = commands.add_parser(
        "external-dependence-sensitivity",
        help="Run descriptive subgroup, threshold, and tracker-update fragility checks",
    )
    dependence_sensitivity.add_argument("--grid", type=Path, required=True)
    dependence_sensitivity.add_argument("--primary-plan", type=Path, required=True)
    dependence_sensitivity.add_argument("--primary-report", type=Path, required=True)
    dependence_sensitivity.add_argument("--secondary-report", type=Path, required=True)
    dependence_sensitivity.add_argument("--inferential-hierarchy", type=Path, required=True)
    dependence_sensitivity.add_argument("--benchmark-design", type=Path, required=True)
    dependence_sensitivity.add_argument("--dataset-root", type=Path, required=True)
    dependence_sensitivity.add_argument("--pixi-lock", type=Path, required=True)
    dependence_sensitivity.add_argument("--output-root", type=Path, required=True)
    dependence_sensitivity.add_argument("--code-revision", required=True)
    dependence_sensitivity.add_argument("--created-at-utc", type=_utc_datetime, required=True)

    dependence_figure = commands.add_parser(
        "external-dependence-figure",
        help="Render a deterministic report figure from dependence sensitivity results",
    )
    dependence_figure.add_argument("--report", type=Path, required=True)
    dependence_figure.add_argument("--pdf-output", type=Path, required=True)
    dependence_figure.add_argument("--svg-output", type=Path, required=True)
    dependence_figure.add_argument("--manifest", type=Path, required=True)
    dependence_figure.add_argument("--generated-at-utc", type=_utc_datetime)

    primary_figure = commands.add_parser(
        "external-primary-figure",
        help="Render the case-level sealed primary benchmark result",
    )
    primary_figure.add_argument("--primary-report", type=Path, required=True)
    primary_figure.add_argument("--interpretation-plan", type=Path, required=True)
    primary_figure.add_argument("--interpretation-report", type=Path, required=True)
    primary_figure.add_argument("--pixi-lock", type=Path, required=True)
    primary_figure.add_argument("--output-root", type=Path, required=True)
    primary_figure.add_argument("--code-revision", required=True)
    primary_figure.add_argument("--generated-at-utc", type=_utc_datetime)

    specificity_figure = commands.add_parser(
        "external-specificity-figure",
        help="Render the sealed evidence-specificity result and contrastive controls",
    )
    specificity_figure.add_argument("--secondary-report", type=Path, required=True)
    specificity_figure.add_argument("--interpretation-plan", type=Path, required=True)
    specificity_figure.add_argument("--interpretation-report", type=Path, required=True)
    specificity_figure.add_argument("--pixi-lock", type=Path, required=True)
    specificity_figure.add_argument("--output-root", type=Path, required=True)
    specificity_figure.add_argument("--code-revision", required=True)
    specificity_figure.add_argument("--generated-at-utc", type=_utc_datetime)

    missingness_sensitivity = commands.add_parser(
        "external-missingness-bound",
        help="Bound the primary effect over every missing binary criterion outcome",
    )
    missingness_sensitivity.add_argument("--primary-plan", type=Path, required=True)
    missingness_sensitivity.add_argument("--primary-report", type=Path, required=True)
    missingness_sensitivity.add_argument("--dataset-root", type=Path, required=True)
    missingness_sensitivity.add_argument("--pixi-lock", type=Path, required=True)
    missingness_sensitivity.add_argument("--output-root", type=Path, required=True)
    missingness_sensitivity.add_argument("--code-revision", required=True)
    missingness_sensitivity.add_argument("--created-at-utc", type=_utc_datetime, required=True)

    probe_correction = commands.add_parser(
        "probe-correction-sensitivity",
        help="Recompute paired results under reviewed post-hoc probe corrections",
    )
    probe_correction.add_argument("--specification", type=Path, required=True)
    probe_correction.add_argument("--primary-report", type=Path, required=True)
    probe_correction.add_argument("--failure-review-report", type=Path, required=True)
    probe_correction.add_argument("--evidence-execution-root", type=Path, required=True)
    probe_correction.add_argument("--pixi-lock", type=Path, required=True)
    probe_correction.add_argument("--output-root", type=Path, required=True)
    probe_correction.add_argument("--code-revision", required=True)
    probe_correction.add_argument("--created-at-utc", type=_utc_datetime, required=True)

    resource_reconciliation = commands.add_parser(
        "external-resource-reconcile",
        help="Reconcile observed quality, provider workload, and sandbox burden",
    )
    resource_reconciliation.add_argument("--generation-report", type=Path, required=True)
    resource_reconciliation.add_argument("--criterion-report", type=Path, required=True)
    resource_reconciliation.add_argument("--decision-seal-report", type=Path, required=True)
    resource_reconciliation.add_argument("--primary-report", type=Path, required=True)
    resource_reconciliation.add_argument("--correction-report", type=Path, required=True)
    resource_reconciliation.add_argument("--decision-responses", type=Path, required=True)
    resource_reconciliation.add_argument("--criterion-responses", type=Path, required=True)
    resource_reconciliation.add_argument("--evidence-execution-root", type=Path, required=True)
    resource_reconciliation.add_argument("--measured-artifact-root", type=Path, required=True)
    resource_reconciliation.add_argument("--pixi-lock", type=Path, required=True)
    resource_reconciliation.add_argument("--output-root", type=Path, required=True)
    resource_reconciliation.add_argument("--code-revision", required=True)
    resource_reconciliation.add_argument("--created-at-utc", type=_utc_datetime, required=True)

    resource_figure = commands.add_parser(
        "external-resource-figure",
        help="Render observed benchmark quality beside provider and sandbox burden",
    )
    resource_figure.add_argument("--resource-plan", type=Path, required=True)
    resource_figure.add_argument("--resource-report", type=Path, required=True)
    resource_figure.add_argument("--pixi-lock", type=Path, required=True)
    resource_figure.add_argument("--output-root", type=Path, required=True)
    resource_figure.add_argument("--code-revision", required=True)
    resource_figure.add_argument("--generated-at-utc", type=_utc_datetime)

    rating_reliability = commands.add_parser(
        "public-rating-reliability",
        help="Reconstruct category marginals and the complete two-rater confusion matrix",
    )
    rating_reliability.add_argument("--rating-report", type=Path, required=True)
    rating_reliability.add_argument("--procedure-record", type=Path, required=True)
    rating_reliability.add_argument("--ratings-a", type=Path, required=True)
    rating_reliability.add_argument("--ratings-b", type=Path, required=True)
    rating_reliability.add_argument("--pixi-lock", type=Path, required=True)
    rating_reliability.add_argument("--output-root", type=Path, required=True)
    rating_reliability.add_argument("--code-revision", required=True)
    rating_reliability.add_argument("--created-at-utc", type=_utc_datetime, required=True)

    external_diagnostics = commands.add_parser(
        "external-diagnostics",
        help="Summarize safety, missingness, failures, and non-estimable stability",
    )
    external_diagnostics.add_argument("--scoring-summary", type=Path, required=True)
    external_diagnostics.add_argument("--primary-report", type=Path, required=True)
    external_diagnostics.add_argument("--decision-seal-report", type=Path, required=True)
    external_diagnostics.add_argument("--dataset-root", type=Path, required=True)
    external_diagnostics.add_argument("--pixi-lock", type=Path, required=True)
    external_diagnostics.add_argument("--output-root", type=Path, required=True)
    external_diagnostics.add_argument("--code-revision", required=True)
    external_diagnostics.add_argument("--created-at-utc", type=_utc_datetime, required=True)

    failure_review = commands.add_parser(
        "failure-review-prepare",
        help="Prepare observable primary failures and matched successes for review",
    )
    failure_review.add_argument("--taxonomy", type=Path, required=True)
    failure_review.add_argument("--analysis-specification", type=Path, required=True)
    failure_review.add_argument("--manifest", type=Path, required=True)
    failure_review.add_argument("--primary-report", type=Path, required=True)
    failure_review.add_argument("--dataset-root", type=Path, required=True)
    failure_review.add_argument("--generation-responses", type=Path, required=True)
    failure_review.add_argument("--criterion-responses", type=Path, required=True)
    failure_review.add_argument("--pixi-lock", type=Path, required=True)
    failure_review.add_argument("--output-root", type=Path, required=True)
    failure_review.add_argument("--code-revision", required=True)
    failure_review.add_argument("--created-at-utc", type=_utc_datetime, required=True)

    failure_review_record = commands.add_parser(
        "failure-review-record",
        help="Validate and preserve one completed independent failure review",
    )
    failure_review_record.add_argument("--taxonomy", type=Path, required=True)
    failure_review_record.add_argument("--packet", type=Path, required=True)
    failure_review_record.add_argument("--completed-sheet", type=Path, required=True)
    failure_review_record.add_argument("--rater-id", required=True)
    failure_review_record.add_argument("--output", type=Path, required=True)
    failure_review_record.add_argument("--recorded-at-utc", type=_utc_datetime, required=True)

    failure_review_reliability = commands.add_parser(
        "failure-review-reliability",
        help="Compare two complete failure reviews of the same packet",
    )
    failure_review_reliability.add_argument("--primary-review", type=Path, required=True)
    failure_review_reliability.add_argument("--secondary-review", type=Path, required=True)
    failure_review_reliability.add_argument("--pixi-lock", type=Path, required=True)
    failure_review_reliability.add_argument("--output-root", type=Path, required=True)
    failure_review_reliability.add_argument("--code-revision", required=True)
    failure_review_reliability.add_argument("--created-at-utc", type=_utc_datetime, required=True)

    failure_figure = commands.add_parser(
        "external-failure-figure",
        help="Render the observable failure review and reviewer-consistency boundary",
    )
    failure_figure.add_argument("--primary-review", type=Path, required=True)
    failure_figure.add_argument("--secondary-review", type=Path, required=True)
    failure_figure.add_argument("--reliability-plan", type=Path, required=True)
    failure_figure.add_argument("--reliability-report", type=Path, required=True)
    failure_figure.add_argument("--pixi-lock", type=Path, required=True)
    failure_figure.add_argument("--output-root", type=Path, required=True)
    failure_figure.add_argument("--code-revision", required=True)
    failure_figure.add_argument("--generated-at-utc", type=_utc_datetime)

    negative_result_audit = commands.add_parser(
        "negative-result-audit",
        help="Verify that protected benchmark inputs did not change after reveal",
    )
    negative_result_audit.add_argument("--repository-root", type=Path, required=True)
    negative_result_audit.add_argument("--manifest", type=Path, required=True)
    negative_result_audit.add_argument("--analysis-specification", type=Path, required=True)
    negative_result_audit.add_argument("--calibration-report", type=Path, required=True)
    negative_result_audit.add_argument("--decision-seal-plan", type=Path, required=True)
    negative_result_audit.add_argument("--decision-seal-report", type=Path, required=True)
    negative_result_audit.add_argument("--criterion-plan", type=Path, required=True)
    negative_result_audit.add_argument("--criterion-integrity-report", type=Path, required=True)
    negative_result_audit.add_argument("--pixi-lock", type=Path, required=True)
    negative_result_audit.add_argument("--output-root", type=Path, required=True)
    negative_result_audit.add_argument("--code-revision", required=True)
    negative_result_audit.add_argument("--created-at-utc", type=_utc_datetime, required=True)

    result_interpretation = commands.add_parser(
        "result-interpretation",
        help="Record conservative wording for the sealed primary and secondary results",
    )
    result_interpretation.add_argument("--primary-report", type=Path, required=True)
    result_interpretation.add_argument("--secondary-report", type=Path, required=True)
    result_interpretation.add_argument("--probe-correction-plan", type=Path, required=True)
    result_interpretation.add_argument("--probe-correction-report", type=Path, required=True)
    result_interpretation.add_argument("--negative-result-audit-plan", type=Path, required=True)
    result_interpretation.add_argument("--negative-result-audit-report", type=Path, required=True)
    result_interpretation.add_argument("--pixi-lock", type=Path, required=True)
    result_interpretation.add_argument("--output-root", type=Path, required=True)
    result_interpretation.add_argument("--code-revision", required=True)
    result_interpretation.add_argument("--created-at-utc", type=_utc_datetime, required=True)

    final_replay = commands.add_parser(
        "final-replay",
        help="Rebuild scored datasets and analyses without external services",
    )
    final_replay.add_argument("--repository-root", type=Path, required=True)
    final_replay.add_argument("--external-replay-report", type=Path, required=True)
    final_replay.add_argument("--scoring-plan", type=Path, required=True)
    final_replay.add_argument("--scoring-summary", type=Path, required=True)
    final_replay.add_argument("--primary-plan", type=Path, required=True)
    final_replay.add_argument("--primary-report", type=Path, required=True)
    final_replay.add_argument("--secondary-plan", type=Path, required=True)
    final_replay.add_argument("--secondary-report", type=Path, required=True)
    final_replay.add_argument("--analysis-specification", type=Path, required=True)
    final_replay.add_argument("--calibration-report", type=Path, required=True)
    final_replay.add_argument("--public-manifest", type=Path, required=True)
    final_replay.add_argument("--benchmark-root", type=Path, required=True)
    final_replay.add_argument("--seal-root", type=Path, required=True)
    final_replay.add_argument("--inferential-hierarchy", type=Path, required=True)
    final_replay.add_argument("--specificity-amendment", type=Path, required=True)
    final_replay.add_argument("--pixi-lock", type=Path, required=True)
    final_replay.add_argument("--output-root", type=Path, required=True)
    final_replay.add_argument("--code-revision", required=True)
    final_replay.add_argument("--created-at-utc", type=_utc_datetime, required=True)

    analysis_seal = commands.add_parser(
        "primary-analysis-seal",
        help="Bind and close the complete primary-analysis evidence chain",
    )
    analysis_seal.add_argument("--analysis-root", type=Path, required=True)
    analysis_seal.add_argument("--pixi-lock", type=Path, required=True)
    analysis_seal.add_argument("--output-root", type=Path, required=True)
    analysis_seal.add_argument("--code-revision", required=True)
    analysis_seal.add_argument("--created-at-utc", type=_utc_datetime, required=True)

    evaluate = commands.add_parser("evaluate", help="Verify and summarize local datasets")
    evaluate.add_argument("--dataset-root", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result, gate_passed = _dispatch(args)
    except Exception as error:
        _print_json(
            {
                "command": str(args.command),
                "error_type": type(error).__name__,
                "message": str(error),
                "status": "error",
            },
            stream=sys.stderr,
        )
        return 1
    _print_json(
        {
            "command": str(args.command),
            "result": _json_value(result),
            "status": "ok" if gate_passed else "gate_failed",
        },
        stream=sys.stdout,
    )
    return 0 if gate_passed else 2


def _dispatch(args: argparse.Namespace) -> tuple[BaseModel | dict[str, object], bool]:
    command = cast(str, args.command)
    if command == "validate":
        return _validate(args), True
    if command == "smoke":
        return _smoke(args), True
    if command == "shortcuts":
        summary = run_shortcut_audit(
            _load_model(cast(Path, args.input), ShortcutAuditPlan),
            output_path=cast(Path, args.output),
        )
        return summary, summary.gate_passed
    if command == "case-shortcuts":
        summary = run_case_linked_shortcut_audit(
            _load_model(cast(Path, args.input), CaseLinkedShortcutAuditPlan),
            manifest_path=cast(Path, args.manifest),
            output_path=cast(Path, args.output),
        )
        return summary, summary.gate_passed
    if command == "readiness":
        report = build_benchmark_readiness_report(
            manifest_path=cast(Path, args.manifest),
            case_linked_shortcut_summary_path=cast(Path, args.case_shortcut_summary),
            output_path=cast(Path, args.output),
        )
        return report, report.gate_passed
    if command == "freeze-prepare":
        return (
            prepare_benchmark_freeze(
                manifest_path=cast(Path, args.manifest),
                readiness_report_path=cast(Path, args.readiness_report),
                analysis_specification_path=cast(Path, args.analysis_specification),
                frozen_at_utc=cast(datetime, args.frozen_at_utc),
                output_path=cast(Path, args.output),
            ),
            True,
        )
    if command == "evidence-score":
        summary = score_evidence_responses(
            manifest_path=cast(Path, args.manifest),
            responses_path=cast(Path, args.responses),
            output_path=cast(Path, args.output),
        )
        return summary, summary.complete
    if command == "sensitivity":
        summary = run_sensitivity_analysis(
            _load_model(cast(Path, args.input), SensitivityPlan),
            output_path=cast(Path, args.output),
        )
        return summary, summary.gate_passed
    if command == "binary-sensitivity":
        report = run_binary_sensitivity(
            load_binary_sensitivity_plan(cast(Path, args.input)),
            output_path=cast(Path, args.output),
        )
        return report, True
    if command == "calibration-decision":
        report = record_uncalibrated_decision(
            plan=load_uncalibrated_decision_plan(cast(Path, args.input)),
            heldout_manifest_path=cast(Path, args.heldout_manifest),
            benchmark_root=cast(Path, args.benchmark_root),
            analysis_specification_path=cast(Path, args.analysis_specification),
            output_path=cast(Path, args.output),
        )
        return report, True
    if command == "failure-taxonomy-validate":
        taxonomy = load_failure_taxonomy(cast(Path, args.input))
        return {
            "taxonomy_version": taxonomy.taxonomy_version,
            "category_count": len(taxonomy.categories),
            "taxonomy_hash": failure_taxonomy_hash(taxonomy),
        }, True
    if command == "analysis-spec-validate":
        specification = load_analysis_specification(cast(Path, args.input))
        return {
            "benchmark_version": specification.benchmark_version,
            "primary_metric": specification.primary_metric,
            "specification_hash": analysis_specification_hash(specification),
        }, True
    if command == "provider-qualify":
        report = run_provider_qualification_from_environment(
            plan=load_provider_qualification_plan(cast(Path, args.plan)),
            manifest_path=cast(Path, args.manifest),
            output_root=cast(Path, args.output_root),
            run_id=cast(str, args.run_id),
        )
        return report, report.gate_passed
    if command == "student-behavior-audit":
        report = run_student_behavior_audit_from_environment(
            plan=load_student_behavior_audit_plan(cast(Path, args.plan)),
            output_root=cast(Path, args.output_root),
            run_id=cast(str, args.run_id),
        )
        return report, report.generation_complete
    if command == "reference-integrity":
        report = run_reference_integrity_replay(
            plan=load_reference_integrity_plan(cast(Path, args.plan)),
            manifest_path=cast(Path, args.manifest),
            output_root=cast(Path, args.output_root),
        )
        return report, True
    if command == "external-protocol-freeze":
        return (
            freeze_external_model_execution_protocol(
                plan=load_external_model_protocol_freeze_plan(cast(Path, args.plan)),
                manifest_path=cast(Path, args.manifest),
                analysis_specification_path=cast(Path, args.analysis_specification),
                system_prompt_path=cast(Path, args.system_prompt),
                output_path=cast(Path, args.output),
            ),
            True,
        )
    if command == "external-route-rehearse":
        report = run_external_route_rehearsal_from_environment(
            protocol=load_external_model_execution_protocol(cast(Path, args.protocol)),
            plan=load_external_route_rehearsal_plan(cast(Path, args.plan)),
            manifest_path=cast(Path, args.manifest),
            fixture_decision_plan_path=cast(Path, args.fixture_decision_plan),
            system_prompt_path=cast(Path, args.system_prompt),
            output_root=cast(Path, args.output_root),
            run_id=cast(str, args.run_id),
        )
        return report, report.gate_passed
    if command == "sandbox-rehearse":
        report = run_recorded_sandbox_rehearsal_with_modal(
            plan=load_sandbox_rehearsal_plan(cast(Path, args.plan)),
            manifest_path=cast(Path, args.manifest),
            recorded_responses_path=cast(Path, args.recorded_responses),
            output_root=cast(Path, args.output_root),
            run_id=cast(str, args.run_id),
        )
        return report, report.gate_passed
    if command == "public-rating-freeze":
        result = freeze_public_rating_boundary(
            plan=load_public_rating_guide_plan(cast(Path, args.plan)),
            protocol=load_external_model_execution_protocol(cast(Path, args.protocol)),
            guide_output_path=cast(Path, args.guide_output),
            amendment_output_path=cast(Path, args.amendment_output),
        )
        return result, True
    if command == "public-rating-packet":
        packet = build_public_answer_rating_packet(
            protocol=load_external_model_execution_protocol(cast(Path, args.protocol)),
            amendment=load_public_rating_protocol_amendment(cast(Path, args.amendment)),
            recorded_responses_path=cast(Path, args.recorded_responses),
            output_path=cast(Path, args.output),
        )
        return packet, True
    if command == "public-rating-workbook":
        result = export_public_rating_workbook(
            guide=load_public_answer_rating_guide(cast(Path, args.guide)),
            packet=load_public_answer_rating_packet(cast(Path, args.packet)),
            output_root=cast(Path, args.output_root),
        )
        return result, True
    if command == "public-rating-record":
        result = record_public_answer_ratings(
            guide=load_public_answer_rating_guide(cast(Path, args.guide)),
            packet=load_public_answer_rating_packet(cast(Path, args.packet)),
            completed_sheet_path=cast(Path, args.completed_sheet),
            rater_id=cast(str, args.rater_id),
            output_path=cast(Path, args.output),
        )
        return result, True
    if command == "public-rating-finalize":
        report = finalize_public_answer_ratings(
            guide=load_public_answer_rating_guide(cast(Path, args.guide)),
            amendment=load_public_rating_protocol_amendment(cast(Path, args.amendment)),
            packet=load_public_answer_rating_packet(cast(Path, args.packet)),
            ratings_a_path=cast(Path, args.ratings_a),
            ratings_b_path=cast(Path, args.ratings_b),
            adjudications_path=cast(Path | None, args.adjudications),
            output_path=cast(Path, args.output),
        )
        return report, True
    if command == "public-rating-procedure-freeze":
        record = freeze_public_rating_procedure(
            plan=load_public_rating_procedure_plan(cast(Path, args.plan)),
            rating_report=load_procedure_rating_report(cast(Path, args.rating_report)),
            output_path=cast(Path, args.output),
        )
        return record, True
    if command == "evidence-specificity-freeze":
        amendment = freeze_evidence_specificity_amendment(
            plan=load_evidence_specificity_amendment_plan(cast(Path, args.plan)),
            protocol=load_external_model_execution_protocol(cast(Path, args.protocol)),
            analysis_specification=load_analysis_specification(
                cast(Path, args.analysis_specification)
            ),
            calibration_report=load_uncalibrated_decision_report(
                cast(Path, args.calibration_report)
            ),
            output_path=cast(Path, args.output),
        )
        return amendment, True
    if command == "external-run-preflight":
        preflight = create_external_run_preflight(
            protocol=load_external_model_execution_protocol(cast(Path, args.protocol)),
            analysis_specification=load_analysis_specification(
                cast(Path, args.analysis_specification)
            ),
            calibration_report=load_uncalibrated_decision_report(
                cast(Path, args.calibration_report)
            ),
            public_rating_amendment=load_public_rating_protocol_amendment(
                cast(Path, args.public_rating_amendment)
            ),
            evidence_specificity_amendment=load_evidence_specificity_amendment(
                cast(Path, args.specificity_amendment)
            ),
            output_path=cast(Path, args.output),
        )
        return preflight, True
    if command == "methodology-clarification-freeze":
        clarification = freeze_methodology_clarification(
            plan=load_methodology_clarification_plan(cast(Path, args.plan)),
            protocol=load_external_model_execution_protocol(cast(Path, args.protocol)),
            analysis_specification=load_analysis_specification(
                cast(Path, args.analysis_specification)
            ),
            calibration_report=load_uncalibrated_decision_report(
                cast(Path, args.calibration_report)
            ),
            public_rating_report=load_public_answer_rating_report(
                cast(Path, args.public_rating_report)
            ),
            output_path=cast(Path, args.output),
        )
        return clarification, True
    if command == "inferential-hierarchy-freeze":
        hierarchy = freeze_inferential_hierarchy(
            plan=load_inferential_hierarchy_plan(cast(Path, args.plan)),
            analysis_specification=load_analysis_specification(
                cast(Path, args.analysis_specification)
            ),
            methodology=load_methodology_clarification(cast(Path, args.methodology_clarification)),
            sensitivity=load_binary_sensitivity_report(cast(Path, args.binary_sensitivity_report)),
            rating_procedure=load_public_rating_procedure_record(
                cast(Path, args.public_rating_procedure)
            ),
            specificity=load_evidence_specificity_amendment(cast(Path, args.specificity_amendment)),
            design=load_design(cast(Path, args.benchmark_design)),
            output_path=cast(Path, args.output),
        )
        return hierarchy, True
    if command == "external-decision-generate":
        report = run_external_decision_generation_from_environment(
            protocol=load_external_model_execution_protocol(cast(Path, args.protocol)),
            preflight=load_external_run_preflight(cast(Path, args.preflight)),
            plan=load_external_decision_generation_plan(cast(Path, args.plan)),
            manifest_path=cast(Path, args.manifest),
            analysis_specification_path=cast(Path, args.analysis_specification),
            system_prompt_path=cast(Path, args.system_prompt),
            output_root=cast(Path, args.output_root),
            run_id=cast(str, args.run_id),
        )
        return report, report.gate_passed
    if command == "external-decision-seal":
        report = run_external_decision_seal_with_modal(
            protocol_path=cast(Path, args.protocol),
            preflight_path=cast(Path, args.preflight),
            generation_report_path=cast(Path, args.generation_report),
            recorded_responses_path=cast(Path, args.recorded_responses),
            public_rating_report_path=cast(Path, args.public_rating_report),
            methodology_clarification_path=cast(Path, args.methodology_clarification),
            inferential_hierarchy_path=cast(Path, args.inferential_hierarchy),
            manifest_path=cast(Path, args.manifest),
            analysis_specification_path=cast(Path, args.analysis_specification),
            pixi_lock_path=cast(Path, args.pixi_lock),
            output_root=cast(Path, args.output_root),
            code_revision=cast(str, args.code_revision),
            dirty_worktree=cast(bool, args.dirty_worktree),
            created_at_utc=cast(datetime, args.created_at_utc),
        )
        return report, report.gate_passed
    if command == "external-decision-audit":
        report = audit_external_decision_seal(
            protocol_path=cast(Path, args.protocol),
            generation_report_path=cast(Path, args.generation_report),
            recorded_responses_path=cast(Path, args.recorded_responses),
            public_rating_report_path=cast(Path, args.public_rating_report),
            manifest_path=cast(Path, args.manifest),
            seal_root=cast(Path, args.seal_root),
            pixi_lock_path=cast(Path, args.pixi_lock),
            output_path=cast(Path, args.output),
            audit_code_revision=cast(str, args.audit_code_revision),
            audited_at_utc=cast(datetime, args.audited_at_utc),
        )
        return report, report.gate_passed
    if command == "external-criterion-run":
        report = run_external_criterion_from_environment(
            protocol_path=cast(Path, args.protocol),
            accounting_report_path=cast(Path, args.accounting_report),
            manifest_path=cast(Path, args.manifest),
            system_prompt_path=cast(Path, args.system_prompt),
            seal_root=cast(Path, args.seal_root),
            pixi_lock_path=cast(Path, args.pixi_lock),
            code_revision=cast(str, args.code_revision),
            created_at_utc=cast(datetime, args.created_at_utc),
        )
        return report, report.gate_passed
    if command == "external-criterion-audit":
        report = audit_external_criterion_run(
            protocol_path=cast(Path, args.protocol),
            accounting_report_path=cast(Path, args.accounting_report),
            manifest_path=cast(Path, args.manifest),
            system_prompt_path=cast(Path, args.system_prompt),
            seal_root=cast(Path, args.seal_root),
            pixi_lock_path=cast(Path, args.pixi_lock),
            output_path=cast(Path, args.output),
            audit_code_revision=cast(str, args.audit_code_revision),
            audited_at_utc=cast(datetime, args.audited_at_utc),
        )
        return report, report.gate_passed
    if command == "external-replay":
        report = run_external_preanalysis_replay(
            integrity_report_path=cast(Path, args.integrity_report),
            decision_responses_path=cast(Path, args.decision_responses),
            seal_root=cast(Path, args.seal_root),
            pixi_lock_path=cast(Path, args.pixi_lock),
            output_root=cast(Path, args.output_root),
            replay_code_revision=cast(str, args.code_revision),
            created_at_utc=cast(datetime, args.created_at_utc),
        )
        return report, report.gate_passed
    if command == "external-score":
        summary = run_external_scoring(
            replay_report_path=cast(Path, args.replay_report),
            analysis_specification_path=cast(Path, args.analysis_specification),
            calibration_report_path=cast(Path, args.calibration_report),
            public_manifest_path=cast(Path, args.public_manifest),
            benchmark_root=cast(Path, args.benchmark_root),
            seal_root=cast(Path, args.seal_root),
            pixi_lock_path=cast(Path, args.pixi_lock),
            scoring_code_revision=cast(str, args.code_revision),
            created_at_utc=cast(datetime, args.created_at_utc),
        )
        return summary, summary.gate_passed
    if command == "external-primary-analyze":
        report = run_primary_analysis(
            scoring_summary_path=cast(Path, args.scoring_summary),
            analysis_specification_path=cast(Path, args.analysis_specification),
            inferential_hierarchy_path=cast(Path, args.inferential_hierarchy),
            dataset_root=cast(Path, args.dataset_root),
            pixi_lock_path=cast(Path, args.pixi_lock),
            output_root=cast(Path, args.output_root),
            analysis_code_revision=cast(str, args.code_revision),
            created_at_utc=cast(datetime, args.created_at_utc),
        )
        return report, True
    if command == "external-secondary-analyze":
        report = run_secondary_analysis(
            scoring_summary_path=cast(Path, args.scoring_summary),
            primary_analysis_plan_path=cast(Path, args.primary_plan),
            primary_analysis_report_path=cast(Path, args.primary_report),
            analysis_specification_path=cast(Path, args.analysis_specification),
            inferential_hierarchy_path=cast(Path, args.inferential_hierarchy),
            evidence_specificity_amendment_path=cast(Path, args.specificity_amendment),
            dataset_root=cast(Path, args.dataset_root),
            pixi_lock_path=cast(Path, args.pixi_lock),
            output_root=cast(Path, args.output_root),
            analysis_code_revision=cast(str, args.code_revision),
            created_at_utc=cast(datetime, args.created_at_utc),
        )
        return report, True
    if command == "external-dependence-sensitivity":
        report = run_dependence_sensitivity(
            grid_path=cast(Path, args.grid),
            primary_analysis_plan_path=cast(Path, args.primary_plan),
            primary_report_path=cast(Path, args.primary_report),
            secondary_report_path=cast(Path, args.secondary_report),
            inferential_hierarchy_path=cast(Path, args.inferential_hierarchy),
            benchmark_design_path=cast(Path, args.benchmark_design),
            dataset_root=cast(Path, args.dataset_root),
            pixi_lock_path=cast(Path, args.pixi_lock),
            output_root=cast(Path, args.output_root),
            analysis_code_revision=cast(str, args.code_revision),
            created_at_utc=cast(datetime, args.created_at_utc),
        )
        return report, True
    if command == "external-dependence-figure":
        manifest = render_dependence_figure(
            report_path=cast(Path, args.report),
            pdf_output_path=cast(Path, args.pdf_output),
            svg_output_path=cast(Path, args.svg_output),
            manifest_path=cast(Path, args.manifest),
            generated_at_utc=cast(datetime | None, args.generated_at_utc),
        )
        return manifest, True
    if command == "external-primary-figure":
        manifest = render_primary_result_figure(
            primary_report_path=cast(Path, args.primary_report),
            interpretation_plan_path=cast(Path, args.interpretation_plan),
            interpretation_report_path=cast(Path, args.interpretation_report),
            pixi_lock_path=cast(Path, args.pixi_lock),
            output_root=cast(Path, args.output_root),
            publication_code_revision=cast(str, args.code_revision),
            generated_at_utc=cast(datetime | None, args.generated_at_utc),
        )
        return manifest, True
    if command == "external-specificity-figure":
        manifest = render_specificity_figure(
            secondary_report_path=cast(Path, args.secondary_report),
            interpretation_plan_path=cast(Path, args.interpretation_plan),
            interpretation_report_path=cast(Path, args.interpretation_report),
            pixi_lock_path=cast(Path, args.pixi_lock),
            output_root=cast(Path, args.output_root),
            publication_code_revision=cast(str, args.code_revision),
            generated_at_utc=cast(datetime | None, args.generated_at_utc),
        )
        return manifest, True
    if command == "external-missingness-bound":
        report = run_missingness_sensitivity(
            primary_analysis_plan_path=cast(Path, args.primary_plan),
            primary_report_path=cast(Path, args.primary_report),
            dataset_root=cast(Path, args.dataset_root),
            pixi_lock_path=cast(Path, args.pixi_lock),
            output_root=cast(Path, args.output_root),
            analysis_code_revision=cast(str, args.code_revision),
            created_at_utc=cast(datetime, args.created_at_utc),
        )
        return report, True
    if command == "probe-correction-sensitivity":
        report = run_probe_correction_sensitivity(
            specification_path=cast(Path, args.specification),
            primary_report_path=cast(Path, args.primary_report),
            failure_review_report_path=cast(Path, args.failure_review_report),
            evidence_execution_root=cast(Path, args.evidence_execution_root),
            pixi_lock_path=cast(Path, args.pixi_lock),
            output_root=cast(Path, args.output_root),
            analysis_code_revision=cast(str, args.code_revision),
            created_at_utc=cast(datetime, args.created_at_utc),
        )
        return report, True
    if command == "external-resource-figure":
        manifest = render_resource_figure(
            resource_plan_path=cast(Path, args.resource_plan),
            resource_report_path=cast(Path, args.resource_report),
            pixi_lock_path=cast(Path, args.pixi_lock),
            output_root=cast(Path, args.output_root),
            publication_code_revision=cast(str, args.code_revision),
            generated_at_utc=cast(datetime | None, args.generated_at_utc),
        )
        return manifest, True
    if command == "external-resource-reconcile":
        report = run_resource_reconciliation(
            generation_report_path=cast(Path, args.generation_report),
            criterion_report_path=cast(Path, args.criterion_report),
            decision_seal_report_path=cast(Path, args.decision_seal_report),
            primary_report_path=cast(Path, args.primary_report),
            correction_report_path=cast(Path, args.correction_report),
            decision_responses_path=cast(Path, args.decision_responses),
            criterion_responses_path=cast(Path, args.criterion_responses),
            evidence_execution_root=cast(Path, args.evidence_execution_root),
            measured_artifact_root=cast(Path, args.measured_artifact_root),
            pixi_lock_path=cast(Path, args.pixi_lock),
            output_root=cast(Path, args.output_root),
            analysis_code_revision=cast(str, args.code_revision),
            created_at_utc=cast(datetime, args.created_at_utc),
        )
        return report, True
    if command == "public-rating-reliability":
        report = run_rating_reliability(
            rating_report_path=cast(Path, args.rating_report),
            procedure_record_path=cast(Path, args.procedure_record),
            ratings_a_path=cast(Path, args.ratings_a),
            ratings_b_path=cast(Path, args.ratings_b),
            pixi_lock_path=cast(Path, args.pixi_lock),
            output_root=cast(Path, args.output_root),
            analysis_code_revision=cast(str, args.code_revision),
            created_at_utc=cast(datetime, args.created_at_utc),
        )
        return report, True
    if command == "external-diagnostics":
        report = run_external_diagnostics(
            scoring_summary_path=cast(Path, args.scoring_summary),
            primary_report_path=cast(Path, args.primary_report),
            decision_seal_report_path=cast(Path, args.decision_seal_report),
            dataset_root=cast(Path, args.dataset_root),
            pixi_lock_path=cast(Path, args.pixi_lock),
            output_root=cast(Path, args.output_root),
            analysis_code_revision=cast(str, args.code_revision),
            created_at_utc=cast(datetime, args.created_at_utc),
        )
        return report, True
    if command == "failure-review-prepare":
        report = prepare_failure_review(
            taxonomy_path=cast(Path, args.taxonomy),
            analysis_specification_path=cast(Path, args.analysis_specification),
            manifest_path=cast(Path, args.manifest),
            primary_report_path=cast(Path, args.primary_report),
            dataset_root=cast(Path, args.dataset_root),
            generation_responses_path=cast(Path, args.generation_responses),
            criterion_responses_path=cast(Path, args.criterion_responses),
            pixi_lock_path=cast(Path, args.pixi_lock),
            output_root=cast(Path, args.output_root),
            analysis_code_revision=cast(str, args.code_revision),
            created_at_utc=cast(datetime, args.created_at_utc),
        )
        return report, True
    if command == "failure-review-record":
        report = record_failure_review(
            taxonomy_path=cast(Path, args.taxonomy),
            packet_path=cast(Path, args.packet),
            completed_sheet_path=cast(Path, args.completed_sheet),
            rater_id=cast(str, args.rater_id),
            output_path=cast(Path, args.output),
            recorded_at_utc=cast(datetime, args.recorded_at_utc),
        )
        return report, True
    if command == "failure-review-reliability":
        report = run_failure_review_reliability(
            primary_review_path=cast(Path, args.primary_review),
            secondary_review_path=cast(Path, args.secondary_review),
            pixi_lock_path=cast(Path, args.pixi_lock),
            output_root=cast(Path, args.output_root),
            analysis_code_revision=cast(str, args.code_revision),
            created_at_utc=cast(datetime, args.created_at_utc),
        )
        return report, True
    if command == "external-failure-figure":
        manifest = render_failure_figure(
            primary_review_path=cast(Path, args.primary_review),
            secondary_review_path=cast(Path, args.secondary_review),
            reliability_plan_path=cast(Path, args.reliability_plan),
            reliability_report_path=cast(Path, args.reliability_report),
            pixi_lock_path=cast(Path, args.pixi_lock),
            output_root=cast(Path, args.output_root),
            publication_code_revision=cast(str, args.code_revision),
            generated_at_utc=cast(datetime | None, args.generated_at_utc),
        )
        return manifest, True
    if command == "negative-result-audit":
        report = run_negative_result_audit(
            repository_root=cast(Path, args.repository_root),
            manifest_path=cast(Path, args.manifest),
            analysis_specification_path=cast(Path, args.analysis_specification),
            calibration_report_path=cast(Path, args.calibration_report),
            decision_seal_plan_path=cast(Path, args.decision_seal_plan),
            decision_seal_report_path=cast(Path, args.decision_seal_report),
            criterion_plan_path=cast(Path, args.criterion_plan),
            criterion_integrity_report_path=cast(Path, args.criterion_integrity_report),
            pixi_lock_path=cast(Path, args.pixi_lock),
            output_root=cast(Path, args.output_root),
            analysis_code_revision=cast(str, args.code_revision),
            created_at_utc=cast(datetime, args.created_at_utc),
        )
        return report, True
    if command == "result-interpretation":
        report = run_result_interpretation(
            primary_report_path=cast(Path, args.primary_report),
            secondary_report_path=cast(Path, args.secondary_report),
            probe_correction_plan_path=cast(Path, args.probe_correction_plan),
            probe_correction_report_path=cast(Path, args.probe_correction_report),
            negative_result_audit_plan_path=cast(Path, args.negative_result_audit_plan),
            negative_result_audit_report_path=cast(Path, args.negative_result_audit_report),
            pixi_lock_path=cast(Path, args.pixi_lock),
            output_root=cast(Path, args.output_root),
            analysis_code_revision=cast(str, args.code_revision),
            created_at_utc=cast(datetime, args.created_at_utc),
        )
        return report, True
    if command == "final-replay":
        report = run_final_replay(
            repository_root=cast(Path, args.repository_root),
            external_replay_report_path=cast(Path, args.external_replay_report),
            canonical_scoring_plan_path=cast(Path, args.scoring_plan),
            canonical_scoring_summary_path=cast(Path, args.scoring_summary),
            canonical_primary_plan_path=cast(Path, args.primary_plan),
            canonical_primary_report_path=cast(Path, args.primary_report),
            canonical_secondary_plan_path=cast(Path, args.secondary_plan),
            canonical_secondary_report_path=cast(Path, args.secondary_report),
            analysis_specification_path=cast(Path, args.analysis_specification),
            calibration_report_path=cast(Path, args.calibration_report),
            public_manifest_path=cast(Path, args.public_manifest),
            benchmark_root=cast(Path, args.benchmark_root),
            source_seal_root=cast(Path, args.seal_root),
            inferential_hierarchy_path=cast(Path, args.inferential_hierarchy),
            specificity_amendment_path=cast(Path, args.specificity_amendment),
            pixi_lock_path=cast(Path, args.pixi_lock),
            output_root=cast(Path, args.output_root),
            replay_code_revision=cast(str, args.code_revision),
            created_at_utc=cast(datetime, args.created_at_utc),
        )
        return report, report.gate_passed
    if command == "primary-analysis-seal":
        manifest = seal_primary_analysis(
            analysis_root=cast(Path, args.analysis_root),
            pixi_lock_path=cast(Path, args.pixi_lock),
            output_root=cast(Path, args.output_root),
            code_revision=cast(str, args.code_revision),
            created_at_utc=cast(datetime, args.created_at_utc),
        )
        return manifest, True
    if command == "evaluate":
        return (
            evaluate_published_run(
                dataset_root=cast(Path, args.dataset_root),
                output_path=cast(Path, args.output),
            ),
            True,
        )
    raise ValueError(f"Unknown benchmark command: {command}")


def _validate(args: argparse.Namespace) -> dict[str, object]:
    manifest = load_and_verify_manifest(cast(Path, args.manifest))
    if cast(bool, args.require_frozen) and manifest.status is not ManifestStatus.FROZEN:
        raise ValueError("A frozen benchmark manifest is required for this run")
    public = project_public_manifest(manifest)
    evaluator = project_evaluator_manifest(manifest, public)
    return {
        "benchmark_version": manifest.benchmark_version,
        "case_count": len(manifest.cases),
        "evaluator_projection_hash": evaluator.projection_hash,
        "file_count": len(manifest.files),
        "pending_review_count": sum(
            review.decision.value == "pending" for review in manifest.reviews
        ),
        "public_projection_hash": public.projection_hash,
        "source_manifest_hash": public.source_manifest_hash,
        "status": manifest.status.value,
    }


def _smoke(args: argparse.Namespace) -> SmokeManifest:
    manifest_path = cast(Path, args.manifest)
    benchmark_root = manifest_path.resolve().parent
    output_root = cast(Path, args.output_root).resolve()
    decision_plan = _load_model(cast(Path, args.decision_plan), OfflineDecisionPlan)
    authored = load_and_verify_manifest(manifest_path)
    public = project_public_manifest(authored, projected_at_utc=decision_plan.created_at_utc)
    write_immutable_json(output_root / "public_manifest.json", public)
    write_immutable_json(output_root / "offline_decision_plan.json", decision_plan)
    decision_summary = run_offline_decision_phase(
        manifest=public,
        benchmark_root=benchmark_root,
        plan=decision_plan,
        output_root=output_root,
    )
    criterion_plan = _load_model(cast(Path, args.criterion_plan), OfflineCriterionPlan)
    evaluator = project_evaluator_manifest(
        authored,
        public,
        projected_at_utc=decision_plan.created_at_utc,
    )
    write_immutable_json(output_root / "evaluator_manifest.json", evaluator)
    write_immutable_json(output_root / "offline_criterion_plan.json", criterion_plan)
    criterion_summary = run_offline_criterion_phase(
        public_manifest=public,
        evaluator_manifest=evaluator,
        benchmark_root=benchmark_root,
        plan=criterion_plan,
        output_root=output_root,
    )
    evaluation = evaluate_published_run(
        dataset_root=output_root / "datasets",
        output_path=output_root / "evaluation_summary.json",
    )
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.smoke_manifest.v1",
        "benchmark_version": public.benchmark_version,
        "run_id": decision_plan.run_id,
        "source_manifest_hash": public.source_manifest_hash,
        "public_projection_hash": public.projection_hash,
        "evaluator_projection_hash": evaluator.projection_hash,
        "decision_summary_hash": model_content_hash(decision_summary),
        "criterion_summary_hash": model_content_hash(criterion_summary),
        "evaluation_report_hash": evaluation.report_hash,
    }
    draft = SmokeManifest.model_construct(
        _fields_set=set(content),
        **content,
        smoke_hash="0" * 64,
    )
    smoke = SmokeManifest.model_validate(
        {**content, "smoke_hash": model_content_hash(draft, exclude={"smoke_hash"})}
    )
    write_immutable_json(output_root / "smoke_manifest.json", smoke)
    return smoke


def _load_model[ModelT: BaseModel](path: Path, model_type: type[ModelT]) -> ModelT:
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"Could not read command input: {path}") from error
    try:
        if path.suffix.casefold() in {".yaml", ".yml"}:
            raw = yaml.safe_load(content)
        else:
            raw = json.loads(content)
    except (json.JSONDecodeError, yaml.YAMLError) as error:
        raise ValueError(f"Command input is not valid JSON or YAML: {path}") from error
    return model_type.model_validate(raw)


def _utc_datetime(value: str) -> datetime:
    """Parse one explicit ISO 8601 UTC timestamp for a freeze record."""

    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected an ISO 8601 UTC timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise argparse.ArgumentTypeError("Timestamp must use the UTC offset (+00:00 or Z)")
    return parsed


def _json_value(value: BaseModel | dict[str, object]) -> object:
    return value.model_dump(mode="json") if isinstance(value, BaseModel) else value


def _print_json(value: dict[str, object], *, stream: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), file=stream)
