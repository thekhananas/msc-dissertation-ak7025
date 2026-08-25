"""Artifact-backed worked example for the corrected v1 benchmark."""

# pyright: reportArgumentType=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import subprocess
import textwrap
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Literal, cast

import matplotlib as mpl
import yaml
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from pydantic import Field, ValidationError, model_validator

from socratic_tutor.benchmark.artifacts import (
    artifact_locations,
    immutable_json_bytes,
    write_immutable_bytes,
    write_immutable_json,
)
from socratic_tutor.benchmark.command_io import (
    load_command_model,
    print_command_error,
    print_command_result,
)
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.evaluator.datasets import read_criterion_records
from socratic_tutor.benchmark.failure_review import (
    FailureReviewLabel,
    FailureReviewRecord,
    FailureReviewRecordReport,
)
from socratic_tutor.benchmark.failure_review_reliability import (
    FailureReviewReliabilityPlan,
    FailureReviewReliabilityReport,
)
from socratic_tutor.benchmark.generation import GenerationChannel
from socratic_tutor.benchmark.harness_correction_analysis import (
    HarnessCorrectionAnalysisReport,
    HarnessCorrectionCaseResult,
)
from socratic_tutor.benchmark.harness_correction_replay import HarnessCorrectionAttempt
from socratic_tutor.benchmark.hashing import canonical_sha256, file_sha256, model_content_hash
from socratic_tutor.benchmark.parquet import AtomicParquetDatasetStore
from socratic_tutor.benchmark.public.controls import UnrelatedControlSpec
from socratic_tutor.benchmark.public.datasets import read_condition_predictions
from socratic_tutor.benchmark.public.models import EXPECTED_CONDITIONS, BenchmarkCondition
from socratic_tutor.benchmark.replay import RecordedGenerationResponse
from socratic_tutor.contracts import ContractModel
from socratic_tutor.publication.figure_style import (
    BLUE,
    GREEN,
    INK,
    LIGHT_GREY,
    MUTED,
    ORANGE,
    PURPLE,
    FigureProfile,
    style_for,
)
from socratic_tutor.sandbox.python_tests import AuthoredFunctionTestBundle

_CONDITION_LABELS = {
    BenchmarkCondition.DIALOGUE_ONLY: "Dialogue answer",
    BenchmarkCondition.PROBE_INFORMED: "Same-concept check",
    BenchmarkCondition.UNRELATED_PROBE: "Different-concept check",
    BenchmarkCondition.CORRUPTED_PROBE: "Inverted check result",
}
_TITLE = "Why one apparent probing success was misleading"
_SUBTITLE = "Case h-c2m1-02: two different passed checks received the same score."
_FOOTNOTE = (
    "Scope: one post-hoc case from one evaluation-model run. Tracker values are fixed rule scores, "
    "not calibrated probabilities; this is not evidence about human learning."
)
_STYLE: dict[str, object] = {
    "figure.facecolor": "white",
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans", "sans-serif"],
    "pdf.fonttype": 42,
    "savefig.facecolor": "white",
    "svg.fonttype": "none",
    "svg.hashsalt": "socratic-tutor-worked-example-v1",
    "text.color": INK,
}


class WorkedExampleFigureError(ValueError):
    """Worked-example sources are missing, invalid, or inconsistent."""


class WorkedExamplePlan(ContractModel):
    """Frozen case selection and plain-language interpretation."""

    schema_version: Literal[2] = 2
    schema_id: Literal["publication.worked_example_plan.v2"] = "publication.worked_example_plan.v2"
    status: Literal["frozen"] = "frozen"
    figure_id: Literal["FIG-08"] = "FIG-08"
    case_id: Literal["h-c2m1-02"] = "h-c2m1-02"
    sample_id: Literal["001"] = "001"
    decision_response_sample_id: str = Field(min_length=1)
    different_concept_case_id: Literal["h-c3m1-02"] = "h-c3m1-02"
    selection_reason: str = Field(min_length=1)
    public_response_markers: tuple[str, ...] = Field(min_length=2)
    same_concept_response_marker: str = Field(min_length=1)
    different_concept_response_marker: str = Field(min_length=1)
    criterion_response_marker: str = Field(min_length=1)
    different_concept_prompt_ref: str = Field(min_length=1)
    different_concept_prompt_marker: str = Field(min_length=1)
    criterion_prompt_ref: str = Field(min_length=1)
    criterion_tests_ref: str = Field(min_length=1)
    criterion_prompt_marker: str = Field(min_length=1)
    public_summary: str = Field(min_length=1)
    same_concept_summary: str = Field(min_length=1)
    different_concept_summary: str = Field(min_length=1)
    tracker_rule_summary: str = Field(min_length=1)
    criterion_summary: str = Field(min_length=1)
    criterion_test_warning: str = Field(min_length=1)
    interpretation: str = Field(min_length=1)
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_plan(self) -> WorkedExamplePlan:
        if len(set(self.public_response_markers)) != len(self.public_response_markers):
            raise ValueError("Public response markers must be unique")
        return self


class WorkedExamplePrediction(ContractModel):
    """One prediction fixed before the criterion outcome was available."""

    condition: BenchmarkCondition
    label: str = Field(min_length=1)
    tracker_score: float = Field(ge=0.0, le=1.0)
    policy_threshold: float = Field(ge=0.0, le=1.0)
    predicts_success: bool
    matches_authored_outcome: bool
    input_hash: Sha256
    record_hash: Sha256
    committed_at_utc: datetime

    @model_validator(mode="after")
    def validate_prediction(self) -> WorkedExamplePrediction:
        _require_utc(self.committed_at_utc, "Prediction commit time")
        if self.label != _CONDITION_LABELS[self.condition]:
            raise ValueError("Prediction label does not match its condition")
        if self.predicts_success is not (self.tracker_score >= self.policy_threshold):
            raise ValueError("Prediction decision does not match its score")
        return self


class WorkedExampleData(ContractModel):
    """Publication-safe values used by both figure profiles and the later demo."""

    schema_version: Literal[2] = 2
    schema_id: Literal["publication.worked_example_data.v2"] = "publication.worked_example_data.v2"
    figure_id: Literal["FIG-08"] = "FIG-08"
    benchmark_version: Literal["v1"] = "v1"
    run_id: str = Field(min_length=1)
    case_id: Literal["h-c2m1-02"] = "h-c2m1-02"
    sample_id: Literal["001"] = "001"
    evaluation_model: str = Field(min_length=1)
    selection_reason: str = Field(min_length=1)
    public_summary: str = Field(min_length=1)
    public_rating_category: Literal["conflicting"] = "conflicting"
    public_response_hash: Sha256
    same_concept_summary: str = Field(min_length=1)
    same_concept_response_hash: Sha256
    same_concept_passed: int = Field(ge=0)
    same_concept_failed: int = Field(ge=0)
    different_concept_case_id: Literal["h-c3m1-02"] = "h-c3m1-02"
    different_concept_summary: str = Field(min_length=1)
    different_concept_response_hash: Sha256
    different_concept_passed: int = Field(ge=0)
    different_concept_failed: int = Field(ge=0)
    tracker_rule_summary: str = Field(min_length=1)
    predictions: tuple[
        WorkedExamplePrediction,
        WorkedExamplePrediction,
        WorkedExamplePrediction,
        WorkedExamplePrediction,
    ]
    criterion_response_hash: Sha256
    criterion_passed: int = Field(ge=0)
    criterion_failed: int = Field(ge=0)
    criterion_summary: str = Field(min_length=1)
    criterion_test_warning: str = Field(min_length=1)
    criterion_tests_checked_input_mutation: Literal[False] = False
    criterion_revealed_at_utc: datetime
    failure_review_category: Literal[FailureReviewLabel.ITEM_AMBIGUITY_OR_TEST_DEFECT]
    reviewers_agreed_on_category: Literal[True] = True
    primary_review_record_hash: Sha256
    secondary_review_record_hash: Sha256
    failure_review_reliability_report_hash: Sha256
    interpretation: str = Field(min_length=1)
    claim_boundary: str = Field(min_length=1)
    analysis_case_result_hash: Sha256
    evidence_attempt_hash: Sha256
    different_concept_attempt_hash: Sha256
    criterion_attempt_hash: Sha256
    predictions_sealed_before_outcome_reveal: Literal[True] = True
    same_and_different_concept_predictions_match: Literal[True] = True
    tracker_scores_are_calibrated_probabilities: Literal[False] = False
    corrected_result_is_post_hoc: Literal[True] = True
    human_learning_claim_supported: Literal[False] = False
    tutoring_efficacy_claim_supported: Literal[False] = False
    data_hash: Sha256

    @model_validator(mode="after")
    def validate_data(self) -> WorkedExampleData:
        _require_utc(self.criterion_revealed_at_utc, "Criterion reveal time")
        if tuple(row.condition for row in self.predictions) != EXPECTED_CONDITIONS:
            raise ValueError("Worked example requires the four conditions in frozen order")
        if any(row.committed_at_utc >= self.criterion_revealed_at_utc for row in self.predictions):
            raise ValueError("Worked-example predictions must precede criterion reveal")
        by_condition = {row.condition: row for row in self.predictions}
        dialogue = by_condition[BenchmarkCondition.DIALOGUE_ONLY]
        relevant = by_condition[BenchmarkCondition.PROBE_INFORMED]
        different = by_condition[BenchmarkCondition.UNRELATED_PROBE]
        if dialogue.predicts_success or dialogue.matches_authored_outcome:
            raise ValueError("Worked example no longer has the declared dialogue error")
        if not relevant.predicts_success or not relevant.matches_authored_outcome:
            raise ValueError("Worked example no longer has the declared relevant-evidence gain")
        if (
            relevant.predicts_success != different.predicts_success
            or relevant.matches_authored_outcome != different.matches_authored_outcome
        ):
            raise ValueError("Worked example no longer exposes the specificity limitation")
        expected_scores = (0.4, 0.6, 0.6, 0.2)
        if any(
            not math.isclose(row.tracker_score, expected, abs_tol=1e-12)
            for row, expected in zip(self.predictions, expected_scores, strict=True)
        ):
            raise ValueError("Worked example no longer has the frozen tracker scores")
        if any(not math.isclose(row.policy_threshold, 0.6) for row in self.predictions):
            raise ValueError("Worked example no longer uses the frozen 0.60 threshold")
        if self.same_concept_passed != 1 or self.same_concept_failed != 0:
            raise ValueError("Worked example requires the reviewed 1/1 evidence result")
        if self.different_concept_passed != 1 or self.different_concept_failed != 0:
            raise ValueError("Worked example requires the reviewed 1/1 control result")
        if self.criterion_passed != 2 or self.criterion_failed != 0:
            raise ValueError("Worked example requires the reviewed 2/2 criterion result")
        if self.data_hash != model_content_hash(self, exclude={"data_hash"}):
            raise ValueError("Worked-example data hash does not match its content")
        return self


@dataclass(frozen=True, slots=True)
class _WorkedExampleSources:
    data: WorkedExampleData
    plan_sha256: Sha256
    decision_responses_sha256: Sha256
    criterion_responses_sha256: Sha256
    prediction_publication_hash: Sha256
    criterion_publication_hash: Sha256
    analysis_report_hash: Sha256
    analysis_report_sha256: Sha256
    criterion_prompt_sha256: Sha256
    criterion_tests_sha256: Sha256
    different_concept_prompt_sha256: Sha256
    unrelated_control_sha256: Sha256
    primary_review_sha256: Sha256
    secondary_review_sha256: Sha256
    failure_review_reliability_plan_sha256: Sha256
    failure_review_reliability_sha256: Sha256


def render_worked_example_figure(
    *,
    plan_path: Path,
    decision_responses_path: Path,
    criterion_responses_path: Path,
    dataset_root: Path,
    correction_analysis_path: Path,
    correction_attempt_root: Path,
    benchmark_root: Path,
    primary_failure_review_path: Path,
    secondary_failure_review_path: Path,
    failure_review_reliability_plan_path: Path,
    failure_review_reliability_path: Path,
    pixi_lock_path: Path,
    output_root: Path,
    publication_code_revision: str,
    generated_at_utc: datetime | None = None,
) -> dict[str, object]:
    """Publish one verified case as report, presentation, JSON, and CSV artifacts."""

    sources = _load_worked_example(
        plan_path=plan_path,
        decision_responses_path=decision_responses_path,
        criterion_responses_path=criterion_responses_path,
        dataset_root=dataset_root,
        correction_analysis_path=correction_analysis_path,
        correction_attempt_root=correction_attempt_root,
        benchmark_root=benchmark_root,
        primary_failure_review_path=primary_failure_review_path,
        secondary_failure_review_path=secondary_failure_review_path,
        failure_review_reliability_plan_path=failure_review_reliability_plan_path,
        failure_review_reliability_path=failure_review_reliability_path,
    )
    pixi_lock = _read(pixi_lock_path, "Pixi lock")
    generated_at = _resolve_generated_at(
        generated_at_utc,
        output_root / "benchmark_worked_example_figure_manifest.json",
    )
    data_bytes = immutable_json_bytes(sources.data)
    source_table = _source_table_csv(sources.data)
    files: list[dict[str, object]] = []
    vector_files: list[tuple[str, bytes]] = []
    for profile in ("report", "presentation"):
        pdf, svg = _render_vector_files(
            sources.data,
            profile=profile,
            generated_at_utc=generated_at,
        )
        for file_format, content in (("pdf", pdf), ("svg", svg)):
            name = f"benchmark_worked_example_{profile}.{file_format}"
            vector_files.append((name, content))
            files.append(
                {
                    "file": name,
                    "format": file_format,
                    "profile": profile,
                    "sha256": file_sha256(content),
                }
            )

    manifest_content: dict[str, object] = {
        "schema_version": 2,
        "schema_id": "publication.worked_example_figure_manifest.v2",
        "figure_id": "FIG-08",
        "benchmark_version": "v1",
        "run_id": sources.data.run_id,
        "case_id": sources.data.case_id,
        "plan_sha256": sources.plan_sha256,
        "decision_responses_sha256": sources.decision_responses_sha256,
        "criterion_responses_sha256": sources.criterion_responses_sha256,
        "prediction_publication_hash": sources.prediction_publication_hash,
        "criterion_publication_hash": sources.criterion_publication_hash,
        "correction_analysis_report_hash": sources.analysis_report_hash,
        "correction_analysis_report_sha256": sources.analysis_report_sha256,
        "criterion_prompt_sha256": sources.criterion_prompt_sha256,
        "criterion_tests_sha256": sources.criterion_tests_sha256,
        "different_concept_prompt_sha256": sources.different_concept_prompt_sha256,
        "unrelated_control_sha256": sources.unrelated_control_sha256,
        "primary_failure_review_sha256": sources.primary_review_sha256,
        "secondary_failure_review_sha256": sources.secondary_review_sha256,
        "failure_review_reliability_plan_sha256": (sources.failure_review_reliability_plan_sha256),
        "failure_review_reliability_sha256": sources.failure_review_reliability_sha256,
        "analysis_case_result_hash": sources.data.analysis_case_result_hash,
        "evidence_attempt_hash": sources.data.evidence_attempt_hash,
        "criterion_attempt_hash": sources.data.criterion_attempt_hash,
        "different_concept_attempt_hash": sources.data.different_concept_attempt_hash,
        "failure_review_reliability_report_hash": (
            sources.data.failure_review_reliability_report_hash
        ),
        "pixi_lock_sha256": file_sha256(pixi_lock),
        "publication_code_revision": _validate_revision(publication_code_revision),
        "files": files,
        "data_file": "benchmark_worked_example.json",
        "data_sha256": file_sha256(data_bytes),
        "source_table_file": "benchmark_worked_example.csv",
        "source_table_sha256": file_sha256(source_table),
        "generated_at_utc": generated_at.isoformat().replace("+00:00", "Z"),
        "predictions_sealed_before_outcome_reveal": True,
        "corrected_result_is_post_hoc": True,
        "tracker_scores_are_calibrated_probabilities": False,
        "criterion_tests_checked_input_mutation": False,
        "known_item_or_test_defect": True,
        "network_calls_made": 0,
        "sandbox_calls_made": 0,
        "human_learning_claim_supported": False,
        "tutoring_efficacy_claim_supported": False,
    }
    manifest = {**manifest_content, "manifest_hash": canonical_sha256(manifest_content)}
    for name, content in vector_files:
        write_immutable_bytes(output_root / name, content)
    write_immutable_bytes(output_root / "benchmark_worked_example.json", data_bytes)
    write_immutable_bytes(output_root / "benchmark_worked_example.csv", source_table)
    write_immutable_json(output_root / "benchmark_worked_example_figure_manifest.json", manifest)
    return manifest


def _load_worked_example(
    *,
    plan_path: Path,
    decision_responses_path: Path,
    criterion_responses_path: Path,
    dataset_root: Path,
    correction_analysis_path: Path,
    correction_attempt_root: Path,
    benchmark_root: Path,
    primary_failure_review_path: Path,
    secondary_failure_review_path: Path,
    failure_review_reliability_plan_path: Path,
    failure_review_reliability_path: Path,
) -> _WorkedExampleSources:
    plan = load_command_model(plan_path, WorkedExamplePlan)
    decision_bytes = _read(decision_responses_path, "decision response file")
    criterion_bytes = _read(criterion_responses_path, "criterion response file")
    decision_responses = _load_responses(decision_bytes, "decision response file")
    criterion_responses = _load_responses(criterion_bytes, "criterion response file")
    public_response = _select_response(
        decision_responses,
        case_id=plan.case_id,
        sample_id=plan.decision_response_sample_id,
        channel=GenerationChannel.PUBLIC,
    )
    same_concept_response = _select_response(
        decision_responses,
        case_id=plan.case_id,
        sample_id=plan.decision_response_sample_id,
        channel=GenerationChannel.EVIDENCE,
    )
    different_concept_response = _select_case_response(
        decision_responses,
        case_id=plan.different_concept_case_id,
        channel=GenerationChannel.EVIDENCE,
    )
    criterion_response = _select_response(
        criterion_responses,
        case_id=plan.case_id,
        sample_id=plan.sample_id,
        channel=GenerationChannel.CRITERION,
    )
    for marker in plan.public_response_markers:
        if marker not in public_response.final_response:
            raise WorkedExampleFigureError(f"Public response no longer contains marker: {marker}")
    if plan.same_concept_response_marker not in same_concept_response.final_response:
        raise WorkedExampleFigureError("Same-concept response no longer contains its marker")
    if plan.different_concept_response_marker not in different_concept_response.final_response:
        raise WorkedExampleFigureError("Different-concept response no longer contains its marker")
    if plan.criterion_response_marker not in criterion_response.final_response:
        raise WorkedExampleFigureError("Criterion response no longer contains its marker")

    criterion_prompt_bytes = _read_benchmark_source(
        benchmark_root,
        plan.criterion_prompt_ref,
        "criterion prompt",
    )
    if plan.criterion_prompt_marker not in criterion_prompt_bytes.decode("utf-8"):
        raise WorkedExampleFigureError("Criterion prompt no longer states the mutable-input rule")
    criterion_tests_bytes = _read_benchmark_source(
        benchmark_root,
        plan.criterion_tests_ref,
        "criterion tests",
    )
    criterion_tests = _load_test_bundle(criterion_tests_bytes)
    if criterion_tests.function != "discard_tag":
        raise WorkedExampleFigureError("Criterion tests no longer target discard_tag")
    if any("expected_input_after" in check.model_fields_set for check in criterion_tests.checks):
        raise WorkedExampleFigureError(
            "Criterion tests now check input mutation; revise this figure"
        )

    different_prompt_bytes = _read_benchmark_source(
        benchmark_root,
        plan.different_concept_prompt_ref,
        "different-concept prompt",
    )
    if plan.different_concept_prompt_marker not in different_prompt_bytes.decode("utf-8"):
        raise WorkedExampleFigureError("Different-concept prompt no longer uses set mutation")
    unrelated_control_bytes = _read_benchmark_source(
        benchmark_root,
        "controls/unrelated.yaml",
        "different-concept control",
    )
    unrelated_control = _load_unrelated_control(unrelated_control_bytes)
    if unrelated_control.assignments.get(plan.case_id) != plan.different_concept_case_id:
        raise WorkedExampleFigureError("Different-concept control assignment has changed")

    primary_review_bytes, primary_review = _load_failure_review(
        primary_failure_review_path,
        "primary failure review",
    )
    secondary_review_bytes, secondary_review = _load_failure_review(
        secondary_failure_review_path,
        "secondary failure review",
    )
    reliability_plan_bytes, reliability_plan = _load_failure_review_reliability_plan(
        failure_review_reliability_plan_path
    )
    reliability_bytes, reliability = _load_failure_review_reliability(
        failure_review_reliability_path
    )
    primary_review_record, secondary_review_record = _validate_failure_reviews(
        case_id=plan.case_id,
        primary=primary_review,
        secondary=secondary_review,
        primary_file_sha256=file_sha256(primary_review_bytes),
        secondary_file_sha256=file_sha256(secondary_review_bytes),
        reliability_plan=reliability_plan,
        reliability=reliability,
    )

    store = AtomicParquetDatasetStore(dataset_root)
    prediction_dataset = read_condition_predictions(store)
    criterion_dataset = read_criterion_records(store)
    prediction_rows = _select_rows(
        prediction_dataset.table.to_pylist(),
        case_id=plan.case_id,
        sample_id=plan.sample_id,
    )
    criterion_rows = _select_rows(
        criterion_dataset.table.to_pylist(),
        case_id=plan.case_id,
        sample_id=plan.sample_id,
    )
    if len(criterion_rows) != 1:
        raise WorkedExampleFigureError("Worked example requires one criterion row")
    criterion_row = criterion_rows[0]

    analysis_bytes = _read(correction_analysis_path, "correction analysis report")
    try:
        analysis = HarnessCorrectionAnalysisReport.model_validate_json(analysis_bytes)
    except ValidationError as error:
        raise WorkedExampleFigureError("Correction analysis report is invalid") from error
    case_result = _select_case_result(analysis, plan.case_id)
    same_concept_attempt = _load_attempt(
        correction_attempt_root / "evidence" / f"{same_concept_response.response_hash}.json"
    )
    different_concept_attempt = _load_attempt(
        correction_attempt_root / "evidence" / f"{different_concept_response.response_hash}.json"
    )
    criterion_attempt = _load_attempt(
        correction_attempt_root / "criterion" / f"{criterion_response.response_hash}.json"
    )
    _validate_lineage(
        plan=plan,
        public_response=public_response,
        same_concept_response=same_concept_response,
        different_concept_response=different_concept_response,
        criterion_response=criterion_response,
        prediction_rows=prediction_rows,
        criterion_row=criterion_row,
        analysis=analysis,
        case_result=case_result,
        same_concept_attempt=same_concept_attempt,
        different_concept_attempt=different_concept_attempt,
        criterion_attempt=criterion_attempt,
    )
    predictions = _prediction_views(prediction_rows, case_result)
    revealed_at = _datetime_value(criterion_row, "revealed_at_utc")
    route = public_response.request.model_route
    content = {
        "schema_version": 2,
        "schema_id": "publication.worked_example_data.v2",
        "figure_id": "FIG-08",
        "benchmark_version": "v1",
        "run_id": public_response.request.run_id,
        "case_id": plan.case_id,
        "sample_id": plan.sample_id,
        "evaluation_model": f"{route.provider}/{route.model}",
        "selection_reason": plan.selection_reason,
        "public_summary": plan.public_summary,
        "public_rating_category": case_result.public_rating_category,
        "public_response_hash": public_response.response_hash,
        "same_concept_summary": plan.same_concept_summary,
        "same_concept_response_hash": same_concept_response.response_hash,
        "same_concept_passed": case_result.corrected_evidence.passed,
        "same_concept_failed": case_result.corrected_evidence.failed,
        "different_concept_case_id": plan.different_concept_case_id,
        "different_concept_summary": plan.different_concept_summary,
        "different_concept_response_hash": different_concept_response.response_hash,
        "different_concept_passed": different_concept_attempt.corrected.passed,
        "different_concept_failed": different_concept_attempt.corrected.failed,
        "tracker_rule_summary": plan.tracker_rule_summary,
        "predictions": predictions,
        "criterion_response_hash": criterion_response.response_hash,
        "criterion_passed": case_result.corrected_criterion.passed,
        "criterion_failed": case_result.corrected_criterion.failed,
        "criterion_summary": plan.criterion_summary,
        "criterion_test_warning": plan.criterion_test_warning,
        "criterion_tests_checked_input_mutation": False,
        "criterion_revealed_at_utc": revealed_at,
        "failure_review_category": FailureReviewLabel.ITEM_AMBIGUITY_OR_TEST_DEFECT,
        "reviewers_agreed_on_category": True,
        "primary_review_record_hash": primary_review_record.record_hash,
        "secondary_review_record_hash": secondary_review_record.record_hash,
        "failure_review_reliability_report_hash": reliability.report_hash,
        "interpretation": plan.interpretation,
        "claim_boundary": plan.claim_boundary,
        "analysis_case_result_hash": case_result.result_hash,
        "evidence_attempt_hash": same_concept_attempt.attempt_hash,
        "different_concept_attempt_hash": different_concept_attempt.attempt_hash,
        "criterion_attempt_hash": criterion_attempt.attempt_hash,
        "predictions_sealed_before_outcome_reveal": True,
        "same_and_different_concept_predictions_match": True,
        "tracker_scores_are_calibrated_probabilities": False,
        "corrected_result_is_post_hoc": True,
        "human_learning_claim_supported": False,
        "tutoring_efficacy_claim_supported": False,
    }
    draft = WorkedExampleData.model_construct(
        _fields_set=set(content),
        **content,
        data_hash="0" * 64,
    )
    data = WorkedExampleData.model_validate(
        {**content, "data_hash": model_content_hash(draft, exclude={"data_hash"})}
    )
    return _WorkedExampleSources(
        data=data,
        plan_sha256=file_sha256(_read(plan_path, "worked-example plan")),
        decision_responses_sha256=file_sha256(decision_bytes),
        criterion_responses_sha256=file_sha256(criterion_bytes),
        prediction_publication_hash=prediction_dataset.manifest.publication_hash,
        criterion_publication_hash=criterion_dataset.manifest.publication_hash,
        analysis_report_hash=analysis.report_hash,
        analysis_report_sha256=file_sha256(analysis_bytes),
        criterion_prompt_sha256=file_sha256(criterion_prompt_bytes),
        criterion_tests_sha256=file_sha256(criterion_tests_bytes),
        different_concept_prompt_sha256=file_sha256(different_prompt_bytes),
        unrelated_control_sha256=file_sha256(unrelated_control_bytes),
        primary_review_sha256=file_sha256(primary_review_bytes),
        secondary_review_sha256=file_sha256(secondary_review_bytes),
        failure_review_reliability_plan_sha256=file_sha256(reliability_plan_bytes),
        failure_review_reliability_sha256=file_sha256(reliability_bytes),
    )


def _validate_lineage(
    *,
    plan: WorkedExamplePlan,
    public_response: RecordedGenerationResponse,
    same_concept_response: RecordedGenerationResponse,
    different_concept_response: RecordedGenerationResponse,
    criterion_response: RecordedGenerationResponse,
    prediction_rows: tuple[dict[str, object], ...],
    criterion_row: dict[str, object],
    analysis: HarnessCorrectionAnalysisReport,
    case_result: HarnessCorrectionCaseResult,
    same_concept_attempt: HarnessCorrectionAttempt,
    different_concept_attempt: HarnessCorrectionAttempt,
    criterion_attempt: HarnessCorrectionAttempt,
) -> None:
    if len(prediction_rows) != 4:
        raise WorkedExampleFigureError("Worked example requires four condition predictions")
    if analysis.improvement_count != 1 or case_result.primary_effect != 1.0:
        raise WorkedExampleFigureError("Selected case is no longer the sole v1 improvement")
    if case_result.valid_vs_unrelated_effect != 0.0:
        raise WorkedExampleFigureError("Selected case no longer has the specificity limitation")
    response_hashes = {str(row["public_record_hash"]) for row in prediction_rows}
    if response_hashes != {public_response.response_hash}:
        raise WorkedExampleFigureError("Condition predictions reference another public response")
    if str(criterion_row["response_hash"]) != criterion_response.response_hash:
        raise WorkedExampleFigureError("Criterion row references another response")
    run_ids = {str(row["run_id"]) for row in prediction_rows} | {str(criterion_row["run_id"])}
    if run_ids != {public_response.request.run_id}:
        raise WorkedExampleFigureError("Worked-example records belong to different runs")
    expected_attempts = (
        (
            same_concept_attempt,
            GenerationChannel.EVIDENCE,
            same_concept_response.response_hash,
            case_result.evidence_attempt_hash,
            case_result.corrected_evidence,
        ),
        (
            criterion_attempt,
            GenerationChannel.CRITERION,
            criterion_response.response_hash,
            case_result.criterion_attempt_hash,
            case_result.corrected_criterion,
        ),
    )
    for attempt, channel, response_hash, attempt_hash, corrected in expected_attempts:
        if (
            attempt.case_id != plan.case_id
            or attempt.channel is not channel
            or attempt.response_hash != response_hash
            or attempt.attempt_hash != attempt_hash
            or attempt.corrected != corrected
            or attempt.outcome_changed
        ):
            raise WorkedExampleFigureError("Correction attempt does not match the selected case")
    if (
        different_concept_attempt.case_id != plan.different_concept_case_id
        or different_concept_attempt.channel is not GenerationChannel.EVIDENCE
        or different_concept_attempt.response_hash != different_concept_response.response_hash
        or different_concept_attempt.corrected.status != "completed"
        or different_concept_attempt.corrected.passed != 1
        or different_concept_attempt.corrected.failed != 0
        or different_concept_attempt.outcome_changed
    ):
        raise WorkedExampleFigureError("Different-concept execution does not match its source")
    if criterion_row["demonstrated_performance"] is not True:
        raise WorkedExampleFigureError("Selected criterion outcome is no longer successful")


def _prediction_views(
    rows: tuple[dict[str, object], ...],
    case_result: HarnessCorrectionCaseResult,
) -> tuple[
    WorkedExamplePrediction,
    WorkedExamplePrediction,
    WorkedExamplePrediction,
    WorkedExamplePrediction,
]:
    row_by_condition = {BenchmarkCondition(str(row["condition"])): row for row in rows}
    score_by_condition = {score.condition: score for score in case_result.conditions}
    views: list[WorkedExamplePrediction] = []
    for condition in EXPECTED_CONDITIONS:
        row = row_by_condition.get(condition)
        score = score_by_condition.get(condition)
        if row is None or score is None or score.correct is None:
            raise WorkedExampleFigureError("Condition prediction is incomplete")
        tracker_score = float(row["tracker_mastery_probability"])
        if (
            not math.isclose(tracker_score, score.mastery_score, abs_tol=1e-12)
            or str(row["input_hash"]) != score.input_hash
        ):
            raise WorkedExampleFigureError("Sealed and corrected condition scores differ")
        views.append(
            WorkedExamplePrediction(
                condition=condition,
                label=_CONDITION_LABELS[condition],
                tracker_score=tracker_score,
                policy_threshold=score.policy_threshold,
                predicts_success=score.binary_decision,
                matches_authored_outcome=score.correct,
                input_hash=score.input_hash,
                record_hash=str(row["record_hash"]),
                committed_at_utc=_datetime_value(row, "committed_at_utc"),
            )
        )
    return cast(
        tuple[
            WorkedExamplePrediction,
            WorkedExamplePrediction,
            WorkedExamplePrediction,
            WorkedExamplePrediction,
        ],
        tuple(views),
    )


def _render_vector_files(
    data: WorkedExampleData,
    *,
    profile: FigureProfile,
    generated_at_utc: datetime,
) -> tuple[bytes, bytes]:
    with mpl.rc_context(_STYLE):
        figure = _build_figure(data, profile=profile)
        pdf_buffer = BytesIO()
        svg_buffer = BytesIO()
        figure.savefig(
            pdf_buffer,
            format="pdf",
            metadata={
                "Title": _TITLE,
                "Author": "Anas Khan",
                "Subject": "Artifact-backed benchmark worked example",
                "Creator": "Socratic Tutor reproducible figure pipeline",
                "CreationDate": generated_at_utc,
                "ModDate": generated_at_utc,
            },
        )
        figure.savefig(
            svg_buffer,
            format="svg",
            metadata={
                "Title": _TITLE,
                "Description": _FOOTNOTE,
                "Date": generated_at_utc.isoformat(),
            },
        )
        figure.clear()
    return pdf_buffer.getvalue(), svg_buffer.getvalue()


def _build_figure(data: WorkedExampleData, *, profile: FigureProfile) -> Figure:
    style = style_for(profile)
    height = 8.2 if profile == "report" else style.height_in
    figure = Figure(figsize=(style.width_in, height), facecolor="white")
    axis = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    axis.set_axis_off()
    figure.text(
        0.055,
        0.968,
        _TITLE,
        fontsize=style.title_size,
        fontweight="bold",
        color=INK,
        va="top",
    )
    figure.text(
        0.055,
        0.925 if profile == "report" else 0.875,
        _SUBTITLE,
        fontsize=style.font_size,
        color=MUTED,
        va="top",
    )
    if profile == "report":
        _draw_report(axis, data, font_size=style.font_size)
    else:
        _draw_presentation(axis, data, font_size=style.font_size)
    visible_footnote = (
        _FOOTNOTE
        if profile == "report"
        else (
            "Scope: one post-hoc model case. Hand-set scores are not probabilities; "
            "no human-learning claim."
        )
    )
    figure.text(
        0.055,
        0.014,
        textwrap.fill(visible_footnote, width=116 if profile == "report" else 220),
        fontsize=style.font_size - (1.2 if profile == "report" else 1.5),
        color=MUTED,
        va="bottom",
        wrap=True,
    )
    return figure


def _draw_report(axis: Axes, data: WorkedExampleData, *, font_size: float) -> None:
    _content_card(
        axis,
        x=0.055,
        y=0.78,
        width=0.89,
        height=0.11,
        kicker="1  INITIAL ANSWER",
        title="The response contradicted itself",
        body=data.public_summary,
        colour=ORANGE,
        font_size=font_size,
        body_width=88,
    )
    _content_card(
        axis,
        x=0.055,
        y=0.605,
        width=0.425,
        height=0.155,
        kicker="2A  SAME-CONCEPT CHECK",
        title="Permissions aliasing",
        body=data.same_concept_summary,
        colour=BLUE,
        font_size=font_size,
        body_width=46,
    )
    _content_card(
        axis,
        x=0.52,
        y=0.605,
        width=0.425,
        height=0.155,
        kicker="2B  DIFFERENT-CONCEPT CONTROL",
        title="Function and set mutation",
        body=data.different_concept_summary,
        colour=GREEN,
        font_size=font_size,
        body_width=46,
    )
    _score_report_card(axis, data, y=0.385, height=0.185, font_size=font_size)
    _criterion_report_card(axis, data, y=0.205, height=0.15, font_size=font_size)
    _content_card(
        axis,
        x=0.055,
        y=0.045,
        width=0.89,
        height=0.13,
        kicker="5  CONCLUSION",
        title="What this case can support",
        body=data.interpretation,
        colour=ORANGE,
        font_size=font_size,
        body_width=94,
    )


def _score_report_card(
    axis: Axes,
    data: WorkedExampleData,
    *,
    y: float,
    height: float,
    font_size: float,
) -> None:
    _card_background(axis, x=0.055, y=y, width=0.89, height=height, colour=PURPLE)
    _section_heading(
        axis,
        x=0.08,
        y=y + height - 0.022,
        kicker="3  FIXED SCORING RULE",
        title="Both passed checks crossed the same threshold",
        font_size=font_size,
    )
    axis.text(
        0.08,
        y + height - 0.085,
        "0.50 start - 0.10 conflict = 0.40\n"
        "0.40 + 0.20 passed check = 0.60\n"
        "Decision threshold = 0.60",
        transform=axis.transAxes,
        fontsize=font_size - 0.35,
        color=INK,
        va="top",
        linespacing=1.35,
    )
    axis.text(
        0.08,
        y + 0.022,
        "Hand-set tracker scores, not calibrated probabilities",
        transform=axis.transAxes,
        fontsize=font_size - 1.0,
        color=MUTED,
        va="bottom",
    )
    _prediction_table(axis, data, x=0.515, y=y + 0.022, width=0.40, font_size=font_size)


def _criterion_report_card(
    axis: Axes,
    data: WorkedExampleData,
    *,
    y: float,
    height: float,
    font_size: float,
) -> None:
    _card_background(axis, x=0.055, y=y, width=0.89, height=height, colour=ORANGE)
    _section_heading(
        axis,
        x=0.08,
        y=y + height - 0.022,
        kicker="4  LATER TASK",
        title="The authored tests reported a pass",
        font_size=font_size,
    )
    axis.text(
        0.08,
        y + height - 0.087,
        textwrap.fill(data.criterion_summary, width=50),
        transform=axis.transAxes,
        fontsize=font_size - 0.45,
        color=INK,
        va="top",
        linespacing=1.25,
    )
    axis.text(
        0.515,
        y + height - 0.026,
        "TEST LIMIT",
        transform=axis.transAxes,
        fontsize=font_size - 1.0,
        fontweight="bold",
        color=ORANGE,
        va="top",
    )
    axis.text(
        0.515,
        y + height - 0.055,
        textwrap.fill(data.criterion_test_warning, width=54),
        transform=axis.transAxes,
        fontsize=font_size - 0.45,
        color=INK,
        va="top",
        linespacing=1.25,
    )
    axis.text(
        0.515,
        y + 0.018,
        "Later review category: item ambiguity or test defect",
        transform=axis.transAxes,
        fontsize=font_size - 1.0,
        color=MUTED,
        va="bottom",
    )


def _draw_presentation(axis: Axes, data: WorkedExampleData, *, font_size: float) -> None:
    _content_card(
        axis,
        x=0.04,
        y=0.53,
        width=0.26,
        height=0.27,
        kicker="1  INITIAL ANSWER",
        title="The response conflicted",
        body=data.public_summary,
        colour=ORANGE,
        font_size=font_size,
        body_width=34,
    )
    _content_card(
        axis,
        x=0.34,
        y=0.53,
        width=0.28,
        height=0.27,
        kicker="2A  SAME CONCEPT",
        title="Passed 1 of 1 tests",
        body=data.same_concept_summary,
        colour=BLUE,
        font_size=font_size,
        body_width=38,
    )
    _content_card(
        axis,
        x=0.66,
        y=0.53,
        width=0.30,
        height=0.27,
        kicker="2B  CONTROL",
        title="Also passed 1 of 1 tests",
        body=data.different_concept_summary,
        colour=GREEN,
        font_size=font_size,
        body_width=40,
    )
    _score_presentation_card(
        axis, data, x=0.04, y=0.265, width=0.56, height=0.23, font_size=font_size
    )
    _content_card(
        axis,
        x=0.64,
        y=0.265,
        width=0.32,
        height=0.23,
        kicker="4  LATER TASK",
        title="Tests passed; mutation unchecked",
        body=data.criterion_test_warning,
        colour=ORANGE,
        font_size=font_size,
        body_width=43,
    )
    _content_card(
        axis,
        x=0.04,
        y=0.065,
        width=0.92,
        height=0.165,
        kicker="5  CONCLUSION",
        title="This is a measurement warning",
        body=data.interpretation,
        colour=ORANGE,
        font_size=font_size,
        body_width=123,
    )


def _score_presentation_card(
    axis: Axes,
    data: WorkedExampleData,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    font_size: float,
) -> None:
    _card_background(axis, x=x, y=y, width=width, height=height, colour=PURPLE)
    _section_heading(
        axis,
        x=x + 0.022,
        y=y + height - 0.026,
        kicker="3  FIXED RULE, OUTCOME STILL HIDDEN",
        title="Both passed checks: score 0.60, predicts pass",
        font_size=font_size,
    )
    axis.text(
        x + 0.022,
        y + 0.028,
        "Dialogue 0.40 -> predicts fail     |     Inverted result 0.20 -> predicts fail\n"
        "Threshold 0.60; these scores were hand-set, not calibrated probabilities.",
        transform=axis.transAxes,
        fontsize=font_size - 1.5,
        color=INK,
        va="bottom",
        linespacing=1.35,
    )


def _content_card(
    axis: Axes,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    kicker: str,
    title: str,
    body: str,
    colour: str,
    font_size: float,
    body_width: int,
) -> None:
    _card_background(axis, x=x, y=y, width=width, height=height, colour=colour)
    _section_heading(
        axis,
        x=x + 0.025,
        y=y + height - 0.022,
        kicker=kicker,
        title=title,
        font_size=font_size,
    )
    axis.text(
        x + 0.025,
        y + height - (0.071 if font_size < 10 else 0.084),
        textwrap.fill(body, width=body_width),
        transform=axis.transAxes,
        fontsize=font_size - (0.45 if font_size < 10 else 1.5),
        color=INK,
        va="top",
        linespacing=1.28,
    )


def _section_heading(
    axis: Axes,
    *,
    x: float,
    y: float,
    kicker: str,
    title: str,
    font_size: float,
) -> None:
    axis.text(
        x,
        y,
        kicker,
        transform=axis.transAxes,
        fontsize=font_size - (1.4 if font_size < 10 else 2.4),
        fontweight="bold",
        color=MUTED,
        va="top",
    )
    axis.text(
        x,
        y - (0.025 if font_size < 10 else 0.034),
        title,
        transform=axis.transAxes,
        fontsize=font_size + (0.1 if font_size < 10 else 0.2),
        fontweight="bold",
        color=INK,
        va="top",
    )


def _prediction_table(
    axis: Axes,
    data: WorkedExampleData,
    *,
    x: float,
    y: float,
    width: float,
    font_size: float,
) -> None:
    axis.text(
        x,
        y + 0.126,
        "CONDITION",
        transform=axis.transAxes,
        fontsize=font_size - 1.2,
        fontweight="bold",
        color=MUTED,
        va="bottom",
    )
    axis.text(
        x + width,
        y + 0.126,
        "SCORE  DECISION",
        transform=axis.transAxes,
        fontsize=font_size - 1.2,
        fontweight="bold",
        color=MUTED,
        ha="right",
        va="bottom",
    )
    for index, row in enumerate(data.predictions):
        row_y = y + 0.103 - (0.029 * index)
        axis.text(
            x,
            row_y,
            row.label,
            transform=axis.transAxes,
            fontsize=font_size - 0.85,
            color=INK,
            va="center",
        )
        axis.text(
            x + width,
            row_y,
            _short_prediction(row),
            transform=axis.transAxes,
            fontsize=font_size - 0.85,
            color=INK,
            ha="right",
            va="center",
        )


def _card_background(
    axis: Axes,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    colour: str,
) -> None:
    axis.add_patch(
        Rectangle(
            (x, y),
            width,
            height,
            transform=axis.transAxes,
            facecolor="#F7F8F8",
            edgecolor=LIGHT_GREY,
            linewidth=1.0,
        )
    )
    axis.add_patch(
        Rectangle(
            (x, y),
            0.008,
            height,
            transform=axis.transAxes,
            facecolor=colour,
            edgecolor=colour,
            linewidth=0.0,
        )
    )


def _short_prediction(row: WorkedExamplePrediction) -> str:
    answer = "pass" if row.predicts_success else "fail"
    return f"{row.tracker_score:.2f}  ->  {answer}"


def _source_table_csv(data: WorkedExampleData) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(("stage", "record", "value", "source_hash"))
    writer.writerow((1, "public_response", data.public_rating_category, data.public_response_hash))
    writer.writerow(
        (
            2,
            "same_concept_check",
            f"{data.same_concept_passed}_passed_{data.same_concept_failed}_failed",
            data.same_concept_response_hash,
        )
    )
    writer.writerow(
        (
            2,
            "different_concept_check",
            f"{data.different_concept_passed}_passed_{data.different_concept_failed}_failed",
            data.different_concept_response_hash,
        )
    )
    for prediction in data.predictions:
        writer.writerow(
            (
                3,
                prediction.condition.value,
                f"{prediction.tracker_score:.12g}",
                prediction.record_hash,
            )
        )
    writer.writerow(
        (
            4,
            "criterion_outcome",
            f"{data.criterion_passed}_passed_{data.criterion_failed}_failed",
            data.criterion_response_hash,
        )
    )
    writer.writerow(
        (
            4,
            "criterion_test_limit",
            data.criterion_test_warning,
            data.primary_review_record_hash,
        )
    )
    writer.writerow((5, "interpretation", data.interpretation, data.data_hash))
    return output.getvalue().encode("utf-8")


def _load_responses(
    content: bytes,
    label: str,
) -> tuple[RecordedGenerationResponse, ...]:
    try:
        lines = content.decode("utf-8").splitlines()
        records = tuple(
            RecordedGenerationResponse.model_validate_json(line) for line in lines if line.strip()
        )
    except (UnicodeDecodeError, ValidationError, ValueError) as error:
        raise WorkedExampleFigureError(f"{label.capitalize()} is invalid") from error
    if not records:
        raise WorkedExampleFigureError(f"{label.capitalize()} is empty")
    return records


def _select_response(
    records: tuple[RecordedGenerationResponse, ...],
    *,
    case_id: str,
    sample_id: str,
    channel: GenerationChannel,
) -> RecordedGenerationResponse:
    matches = tuple(
        record
        for record in records
        if record.request.case_id == case_id
        and record.request.sample_id == sample_id
        and record.request.channel is channel
    )
    if len(matches) != 1:
        raise WorkedExampleFigureError(
            f"Expected one {channel.value} response for {case_id}, found {len(matches)}"
        )
    return matches[0]


def _select_case_response(
    records: tuple[RecordedGenerationResponse, ...],
    *,
    case_id: str,
    channel: GenerationChannel,
) -> RecordedGenerationResponse:
    matches = tuple(
        record
        for record in records
        if record.request.case_id == case_id and record.request.channel is channel
    )
    if len(matches) != 1:
        raise WorkedExampleFigureError(
            f"Expected one {channel.value} response for {case_id}, found {len(matches)}"
        )
    return matches[0]


def _read_benchmark_source(root: Path, reference: str, label: str) -> bytes:
    resolved_root = root.resolve()
    path = (resolved_root / reference).resolve()
    if not path.is_relative_to(resolved_root):
        raise WorkedExampleFigureError(f"{label.capitalize()} escapes the benchmark root")
    return _read(path, label)


def _load_test_bundle(content: bytes) -> AuthoredFunctionTestBundle:
    try:
        payload = yaml.safe_load(content.decode("utf-8"))
        return AuthoredFunctionTestBundle.model_validate(payload)
    except (UnicodeDecodeError, yaml.YAMLError, ValidationError) as error:
        raise WorkedExampleFigureError("Criterion test bundle is invalid") from error


def _load_unrelated_control(content: bytes) -> UnrelatedControlSpec:
    try:
        payload = yaml.safe_load(content.decode("utf-8"))
        return UnrelatedControlSpec.model_validate(payload)
    except (UnicodeDecodeError, yaml.YAMLError, ValidationError) as error:
        raise WorkedExampleFigureError("Different-concept control is invalid") from error


def _load_failure_review(
    path: Path,
    label: str,
) -> tuple[bytes, FailureReviewRecordReport]:
    content = _read(path, label)
    try:
        return content, FailureReviewRecordReport.model_validate_json(content)
    except ValidationError as error:
        raise WorkedExampleFigureError(f"{label.capitalize()} is invalid") from error


def _load_failure_review_reliability_plan(
    path: Path,
) -> tuple[bytes, FailureReviewReliabilityPlan]:
    content = _read(path, "failure-review reliability plan")
    try:
        return content, FailureReviewReliabilityPlan.model_validate_json(content)
    except ValidationError as error:
        raise WorkedExampleFigureError("Failure-review reliability plan is invalid") from error


def _load_failure_review_reliability(
    path: Path,
) -> tuple[bytes, FailureReviewReliabilityReport]:
    content = _read(path, "failure-review reliability report")
    try:
        return content, FailureReviewReliabilityReport.model_validate_json(content)
    except ValidationError as error:
        raise WorkedExampleFigureError("Failure-review reliability report is invalid") from error


def _validate_failure_reviews(
    *,
    case_id: str,
    primary: FailureReviewRecordReport,
    secondary: FailureReviewRecordReport,
    primary_file_sha256: Sha256,
    secondary_file_sha256: Sha256,
    reliability_plan: FailureReviewReliabilityPlan,
    reliability: FailureReviewReliabilityReport,
) -> tuple[FailureReviewRecord, FailureReviewRecord]:
    if primary.rater_id == secondary.rater_id:
        raise WorkedExampleFigureError("Worked example requires two different reviewers")
    if (
        primary.packet_hash != secondary.packet_hash
        or primary.taxonomy_hash != secondary.taxonomy_hash
    ):
        raise WorkedExampleFigureError("Failure reviews do not share one review packet")
    if (
        reliability_plan.primary_review_report_hash != primary.report_hash
        or reliability_plan.secondary_review_report_hash != secondary.report_hash
        or reliability_plan.primary_review_file_hash != primary_file_sha256
        or reliability_plan.secondary_review_file_hash != secondary_file_sha256
        or reliability.analysis_plan_hash != reliability_plan.plan_hash
        or reliability.rater_ids != (primary.rater_id, secondary.rater_id)
        or reliability.disagreement_count != 0
        or reliability.agreement_count != reliability.item_count
    ):
        raise WorkedExampleFigureError("Failure-review agreement sources do not reconcile")
    primary_record = _select_failure_review_record(primary, case_id)
    secondary_record = _select_failure_review_record(secondary, case_id)
    expected = FailureReviewLabel.ITEM_AMBIGUITY_OR_TEST_DEFECT
    if primary_record.category is not expected or secondary_record.category is not expected:
        raise WorkedExampleFigureError("Reviewers no longer agree on the selected test weakness")
    return primary_record, secondary_record


def _select_failure_review_record(
    report: FailureReviewRecordReport,
    case_id: str,
) -> FailureReviewRecord:
    matches = tuple(record for record in report.records if record.case_id == case_id)
    if len(matches) != 1:
        raise WorkedExampleFigureError("Failure review does not contain the selected case")
    return matches[0]


def _select_rows(
    raw_rows: object,
    *,
    case_id: str,
    sample_id: str,
) -> tuple[dict[str, object], ...]:
    rows = cast(list[dict[str, object]], raw_rows)
    return tuple(
        row
        for row in rows
        if str(row.get("case_id")) == case_id and str(row.get("sample_id")) == sample_id
    )


def _select_case_result(
    report: HarnessCorrectionAnalysisReport,
    case_id: str,
) -> HarnessCorrectionCaseResult:
    matches = tuple(case for case in report.case_results if case.case_id == case_id)
    if len(matches) != 1:
        raise WorkedExampleFigureError("Correction report does not contain the selected case")
    return matches[0]


def _load_attempt(path: Path) -> HarnessCorrectionAttempt:
    try:
        return HarnessCorrectionAttempt.model_validate_json(path.read_bytes())
    except (OSError, ValidationError) as error:
        raise WorkedExampleFigureError(f"Correction attempt is invalid: {path}") from error


def _datetime_value(row: dict[str, object], key: str) -> datetime:
    value = row.get(key)
    if not isinstance(value, datetime):
        raise WorkedExampleFigureError(f"Worked-example row has invalid {key}")
    _require_utc(value, key)
    return value


def _read(path: Path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise WorkedExampleFigureError(f"Could not read {label}: {path}") from error


def _resolve_generated_at(value: datetime | None, manifest_path: Path) -> datetime:
    if value is not None:
        _require_utc(value, "Figure generation time")
        return value
    if manifest_path.exists():
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            existing = datetime.fromisoformat(str(raw["generated_at_utc"]).replace("Z", "+00:00"))
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            raise WorkedExampleFigureError("Existing worked-example manifest is invalid") from error
        _require_utc(existing, "Existing figure generation time")
        return existing
    return datetime.now(UTC)


def _validate_revision(value: str) -> str:
    if re.fullmatch(r"[0-9a-f]{7,40}", value) is None:
        raise WorkedExampleFigureError("Publication revision must be a Git SHA")
    return value


def _require_utc(value: datetime, label: str) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError(f"{label} must use timezone-aware UTC")


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXTERNAL_ROOT = (
    PROJECT_ROOT / "artifacts" / "external-decision" / "cerebras-gpt-oss-120b-20260824-001"
)
DEFAULT_DECISION_SEAL_ROOT = DEFAULT_EXTERNAL_ROOT / "decision-seal"
DEFAULT_CORRECTION_ROOT = (
    PROJECT_ROOT / "artifacts" / "harness-correction" / "cerebras-gpt-oss-120b-20260903-001"
)
DEFAULT_FAILURE_REVIEW_ROOT = (
    DEFAULT_DECISION_SEAL_ROOT / "analysis" / "failure-review-v1" / "review-submissions"
)
DEFAULT_FAILURE_RELIABILITY_ROOT = (
    DEFAULT_DECISION_SEAL_ROOT / "analysis" / "failure-review-reliability-v1"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish one artifact-backed benchmark case for the report and demo"
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=PROJECT_ROOT / "configs" / "publication" / "v1-worked-example.yaml",
    )
    parser.add_argument(
        "--decision-responses",
        type=Path,
        default=DEFAULT_EXTERNAL_ROOT / "recorded_responses.jsonl",
    )
    parser.add_argument(
        "--criterion-responses",
        type=Path,
        default=DEFAULT_DECISION_SEAL_ROOT / "criterion" / "recorded_responses.jsonl",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DECISION_SEAL_ROOT / "datasets",
    )
    parser.add_argument(
        "--correction-analysis",
        type=Path,
        default=DEFAULT_CORRECTION_ROOT / "analysis-v1" / "harness_correction_analysis_report.json",
    )
    parser.add_argument(
        "--correction-attempt-root",
        type=Path,
        default=DEFAULT_CORRECTION_ROOT / "attempts",
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "benchmarks" / "v1",
    )
    parser.add_argument(
        "--primary-failure-review",
        type=Path,
        default=DEFAULT_FAILURE_REVIEW_ROOT / "researcher-primary-review-v2.json",
    )
    parser.add_argument(
        "--secondary-failure-review",
        type=Path,
        default=DEFAULT_FAILURE_REVIEW_ROOT / "independent-secondary-review.json",
    )
    parser.add_argument(
        "--failure-review-reliability-plan",
        type=Path,
        default=DEFAULT_FAILURE_RELIABILITY_ROOT / "failure_review_reliability_plan.json",
    )
    parser.add_argument(
        "--failure-review-reliability",
        type=Path,
        default=DEFAULT_FAILURE_RELIABILITY_ROOT / "failure_review_reliability_report.json",
    )
    parser.add_argument("--pixi-lock", type=Path, default=PROJECT_ROOT / "pixi.lock")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--generated-at-utc", type=_utc_datetime)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        revision = _current_clean_revision()
        output_root = args.output_root or (
            PROJECT_ROOT
            / "artifacts"
            / "publication"
            / f"benchmark-worked-example-v1-{revision[:7]}"
        )
        manifest = render_worked_example_figure(
            plan_path=args.plan,
            decision_responses_path=args.decision_responses,
            criterion_responses_path=args.criterion_responses,
            dataset_root=args.dataset_root,
            correction_analysis_path=args.correction_analysis,
            correction_attempt_root=args.correction_attempt_root,
            benchmark_root=args.benchmark_root,
            primary_failure_review_path=args.primary_failure_review,
            secondary_failure_review_path=args.secondary_failure_review,
            failure_review_reliability_plan_path=args.failure_review_reliability_plan,
            failure_review_reliability_path=args.failure_review_reliability,
            pixi_lock_path=args.pixi_lock,
            output_root=output_root,
            publication_code_revision=revision,
            generated_at_utc=args.generated_at_utc,
        )
    except Exception as error:
        print_command_error(command="benchmark-worked-example-figure", error=error)
        return 1
    print_command_result(
        command="benchmark-worked-example-figure",
        result={
            "artifact_locations": artifact_locations(output_root),
            "manifest": manifest,
        },
    )
    return 0


def _current_clean_revision() -> str:
    branch = _git("branch", "--show-current")
    status = _git("status", "--porcelain", "--untracked-files=all")
    revision = _git("rev-parse", "HEAD")
    if branch != "main":
        raise WorkedExampleFigureError(
            f"Worked-example publication requires branch main, not {branch or 'detached HEAD'}"
        )
    if status:
        raise WorkedExampleFigureError("Worked-example publication requires a clean worktree")
    return _validate_revision(revision)


def _git(*args: str) -> str:
    completed = subprocess.run(
        ("git", *args),
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise WorkedExampleFigureError(completed.stderr.strip() or "Git command failed")
    return completed.stdout.strip()


def _utc_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    _require_utc(parsed, "Figure generation time")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
