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
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, Rectangle
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
from socratic_tutor.benchmark.generation import GenerationChannel
from socratic_tutor.benchmark.harness_correction_analysis import (
    HarnessCorrectionAnalysisReport,
    HarnessCorrectionCaseResult,
)
from socratic_tutor.benchmark.harness_correction_replay import HarnessCorrectionAttempt
from socratic_tutor.benchmark.hashing import canonical_sha256, file_sha256, model_content_hash
from socratic_tutor.benchmark.parquet import AtomicParquetDatasetStore
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

_CONDITION_LABELS = {
    BenchmarkCondition.DIALOGUE_ONLY: "Dialogue only",
    BenchmarkCondition.PROBE_INFORMED: "Relevant evidence",
    BenchmarkCondition.UNRELATED_PROBE: "Unrelated evidence",
    BenchmarkCondition.CORRUPTED_PROBE: "Inverted evidence",
}
_TITLE = "One case shows both the promise and the weakness of executable evidence"
_SUBTITLE = (
    "Case h-c2m1-02; four predictions were fixed before a separate coding task was revealed."
)
_FOOTNOTE = (
    "Scope: one authored case and one Cerebras run. This does not represent a human learner or "
    "show that tutoring improved learning."
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

    schema_version: Literal[1] = 1
    schema_id: Literal["publication.worked_example_plan.v1"] = "publication.worked_example_plan.v1"
    status: Literal["frozen"] = "frozen"
    figure_id: Literal["FIG-08"] = "FIG-08"
    case_id: Literal["h-c2m1-02"] = "h-c2m1-02"
    sample_id: Literal["001"] = "001"
    decision_response_sample_id: str = Field(min_length=1)
    selection_reason: str = Field(min_length=1)
    public_response_markers: tuple[str, ...] = Field(min_length=2)
    evidence_response_marker: str = Field(min_length=1)
    public_summary: str = Field(min_length=1)
    evidence_summary: str = Field(min_length=1)
    interpretation: str = Field(min_length=1)
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_plan(self) -> WorkedExamplePlan:
        if len(set(self.public_response_markers)) != len(self.public_response_markers):
            raise ValueError("Public response markers must be unique")
        if "Unrelated evidence" not in self.interpretation:
            raise ValueError("Worked example must retain the specificity limitation")
        return self


class WorkedExamplePrediction(ContractModel):
    """One prediction fixed before the criterion outcome was available."""

    condition: BenchmarkCondition
    label: str = Field(min_length=1)
    mastery_probability: float = Field(ge=0.0, le=1.0)
    policy_threshold: float = Field(ge=0.0, le=1.0)
    predicts_success: bool
    correct: bool
    input_hash: Sha256
    record_hash: Sha256
    committed_at_utc: datetime

    @model_validator(mode="after")
    def validate_prediction(self) -> WorkedExamplePrediction:
        _require_utc(self.committed_at_utc, "Prediction commit time")
        if self.label != _CONDITION_LABELS[self.condition]:
            raise ValueError("Prediction label does not match its condition")
        if self.predicts_success is not (self.mastery_probability >= self.policy_threshold):
            raise ValueError("Prediction decision does not match its score")
        return self


class WorkedExampleData(ContractModel):
    """Publication-safe values used by both figure profiles and the later demo."""

    schema_version: Literal[1] = 1
    schema_id: Literal["publication.worked_example_data.v1"] = "publication.worked_example_data.v1"
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
    evidence_summary: str = Field(min_length=1)
    evidence_response_hash: Sha256
    evidence_passed: int = Field(ge=0)
    evidence_failed: int = Field(ge=0)
    predictions: tuple[
        WorkedExamplePrediction,
        WorkedExamplePrediction,
        WorkedExamplePrediction,
        WorkedExamplePrediction,
    ]
    criterion_response_hash: Sha256
    criterion_passed: int = Field(ge=0)
    criterion_failed: int = Field(ge=0)
    criterion_demonstrated_performance: Literal[True] = True
    criterion_revealed_at_utc: datetime
    interpretation: str = Field(min_length=1)
    claim_boundary: str = Field(min_length=1)
    analysis_case_result_hash: Sha256
    evidence_attempt_hash: Sha256
    criterion_attempt_hash: Sha256
    predictions_sealed_before_outcome_reveal: Literal[True] = True
    relevant_and_unrelated_predictions_match: Literal[True] = True
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
        unrelated = by_condition[BenchmarkCondition.UNRELATED_PROBE]
        if dialogue.predicts_success or dialogue.correct:
            raise ValueError("Worked example no longer has the declared dialogue error")
        if not relevant.predicts_success or not relevant.correct:
            raise ValueError("Worked example no longer has the declared relevant-evidence gain")
        if (
            relevant.predicts_success != unrelated.predicts_success
            or relevant.correct != unrelated.correct
        ):
            raise ValueError("Worked example no longer exposes the specificity limitation")
        if self.evidence_passed != 1 or self.evidence_failed != 0:
            raise ValueError("Worked example requires the reviewed 1/1 evidence result")
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


def render_worked_example_figure(
    *,
    plan_path: Path,
    decision_responses_path: Path,
    criterion_responses_path: Path,
    dataset_root: Path,
    correction_analysis_path: Path,
    correction_attempt_root: Path,
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
        "schema_version": 1,
        "schema_id": "publication.worked_example_figure_manifest.v1",
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
        "analysis_case_result_hash": sources.data.analysis_case_result_hash,
        "evidence_attempt_hash": sources.data.evidence_attempt_hash,
        "criterion_attempt_hash": sources.data.criterion_attempt_hash,
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
    evidence_response = _select_response(
        decision_responses,
        case_id=plan.case_id,
        sample_id=plan.decision_response_sample_id,
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
    if plan.evidence_response_marker not in evidence_response.final_response:
        raise WorkedExampleFigureError("Evidence response no longer contains its frozen marker")

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
    evidence_attempt = _load_attempt(
        correction_attempt_root / "evidence" / f"{evidence_response.response_hash}.json"
    )
    criterion_attempt = _load_attempt(
        correction_attempt_root / "criterion" / f"{criterion_response.response_hash}.json"
    )
    _validate_lineage(
        plan=plan,
        public_response=public_response,
        evidence_response=evidence_response,
        criterion_response=criterion_response,
        prediction_rows=prediction_rows,
        criterion_row=criterion_row,
        analysis=analysis,
        case_result=case_result,
        evidence_attempt=evidence_attempt,
        criterion_attempt=criterion_attempt,
    )
    predictions = _prediction_views(prediction_rows, case_result)
    revealed_at = _datetime_value(criterion_row, "revealed_at_utc")
    route = public_response.request.model_route
    content = {
        "schema_version": 1,
        "schema_id": "publication.worked_example_data.v1",
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
        "evidence_summary": plan.evidence_summary,
        "evidence_response_hash": evidence_response.response_hash,
        "evidence_passed": case_result.corrected_evidence.passed,
        "evidence_failed": case_result.corrected_evidence.failed,
        "predictions": predictions,
        "criterion_response_hash": criterion_response.response_hash,
        "criterion_passed": case_result.corrected_criterion.passed,
        "criterion_failed": case_result.corrected_criterion.failed,
        "criterion_demonstrated_performance": True,
        "criterion_revealed_at_utc": revealed_at,
        "interpretation": plan.interpretation,
        "claim_boundary": plan.claim_boundary,
        "analysis_case_result_hash": case_result.result_hash,
        "evidence_attempt_hash": evidence_attempt.attempt_hash,
        "criterion_attempt_hash": criterion_attempt.attempt_hash,
        "predictions_sealed_before_outcome_reveal": True,
        "relevant_and_unrelated_predictions_match": True,
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
    )


def _validate_lineage(
    *,
    plan: WorkedExamplePlan,
    public_response: RecordedGenerationResponse,
    evidence_response: RecordedGenerationResponse,
    criterion_response: RecordedGenerationResponse,
    prediction_rows: tuple[dict[str, object], ...],
    criterion_row: dict[str, object],
    analysis: HarnessCorrectionAnalysisReport,
    case_result: HarnessCorrectionCaseResult,
    evidence_attempt: HarnessCorrectionAttempt,
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
            evidence_attempt,
            GenerationChannel.EVIDENCE,
            evidence_response.response_hash,
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
        probability = float(row["tracker_mastery_probability"])
        if (
            not math.isclose(probability, score.mastery_score, abs_tol=1e-12)
            or str(row["input_hash"]) != score.input_hash
        ):
            raise WorkedExampleFigureError("Sealed and corrected condition scores differ")
        views.append(
            WorkedExamplePrediction(
                condition=condition,
                label=_CONDITION_LABELS[condition],
                mastery_probability=probability,
                policy_threshold=score.policy_threshold,
                predicts_success=score.binary_decision,
                correct=score.correct,
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
    height = 6.2 if profile == "report" else style.height_in
    figure = Figure(figsize=(style.width_in, height), facecolor="white")
    axis = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    axis.set_axis_off()
    figure.text(
        0.055,
        0.965,
        _TITLE,
        fontsize=style.title_size,
        fontweight="bold",
        color=INK,
        va="top",
    )
    figure.text(
        0.055,
        0.91 if profile == "report" else 0.875,
        _SUBTITLE,
        fontsize=style.font_size,
        color=MUTED,
        va="top",
    )
    if profile == "report":
        _draw_report(axis, data, font_size=style.font_size)
    else:
        _draw_presentation(axis, data, font_size=style.font_size)
    figure.text(
        0.055,
        0.018,
        _FOOTNOTE,
        fontsize=style.font_size - (1.2 if profile == "report" else 1.5),
        color=MUTED,
        va="bottom",
        wrap=True,
    )
    return figure


def _draw_report(axis: Axes, data: WorkedExampleData, *, font_size: float) -> None:
    cards = (
        (0.755, 0.13, "1  Visible answer", data.public_summary, ORANGE),
        (0.585, 0.125, "2  Executable check", data.evidence_summary, BLUE),
        (0.22, 0.095, "4  Later task", "A separate tag-removal task passed 2 of 2 tests.", GREEN),
        (0.065, 0.12, "5  What this means", data.interpretation, INK),
    )
    for y, height, title, body, colour in cards[:2]:
        _report_card(
            axis,
            y=y,
            height=height,
            title=title,
            body=body,
            colour=colour,
            font_size=font_size,
        )
    _vertical_arrow(axis, 0.748, 0.718)
    _vertical_arrow(axis, 0.578, 0.552)
    _prediction_report_card(axis, data, y=0.35, height=0.195, font_size=font_size)
    _vertical_arrow(axis, 0.343, 0.322)
    _report_card(
        axis,
        y=cards[2][0],
        height=cards[2][1],
        title=cards[2][2],
        body=cards[2][3],
        colour=cards[2][4],
        font_size=font_size,
    )
    _vertical_arrow(axis, 0.213, 0.192)
    _report_card(
        axis,
        y=cards[3][0],
        height=cards[3][1],
        title=cards[3][2],
        body=cards[3][3],
        colour=cards[3][4],
        font_size=font_size,
    )


def _report_card(
    axis: Axes,
    *,
    y: float,
    height: float,
    title: str,
    body: str,
    colour: str,
    font_size: float,
) -> None:
    _card_background(axis, x=0.055, y=y, width=0.89, height=height, colour=colour)
    axis.text(
        0.085,
        y + height - 0.028,
        title,
        transform=axis.transAxes,
        fontsize=font_size,
        fontweight="bold",
        color=INK,
        va="top",
    )
    axis.text(
        0.31,
        y + height - 0.027,
        textwrap.fill(body, width=82),
        transform=axis.transAxes,
        fontsize=font_size - 0.6,
        color=INK,
        va="top",
        linespacing=1.25,
    )


def _prediction_report_card(
    axis: Axes,
    data: WorkedExampleData,
    *,
    y: float,
    height: float,
    font_size: float,
) -> None:
    _card_background(axis, x=0.055, y=y, width=0.89, height=height, colour=PURPLE)
    axis.text(
        0.085,
        y + height - 0.028,
        "3  Predictions fixed",
        transform=axis.transAxes,
        fontsize=font_size,
        fontweight="bold",
        color=INK,
        va="top",
    )
    axis.text(
        0.085,
        y + height - 0.066,
        "Outcome still hidden",
        transform=axis.transAxes,
        fontsize=font_size - 1.0,
        color=MUTED,
        va="top",
    )
    for index, row in enumerate(data.predictions):
        column = index % 2
        row_index = index // 2
        x = 0.31 + (0.32 * column)
        row_y = y + height - 0.042 - (0.068 * row_index)
        answer = "predicts success" if row.predicts_success else "predicts no success"
        axis.text(
            x,
            row_y,
            f"{row.label}\n{row.mastery_probability:.0%}  |  {answer}",
            transform=axis.transAxes,
            fontsize=font_size - 0.7,
            color=INK,
            va="top",
            linespacing=1.25,
        )


def _draw_presentation(axis: Axes, data: WorkedExampleData, *, font_size: float) -> None:
    x_positions = (0.04, 0.285, 0.53, 0.775)
    cards = (
        (
            "1  Visible answer",
            f"{data.public_summary}\n\nIndependent category: conflicting",
            ORANGE,
        ),
        (
            "2  Executable check",
            f"{data.evidence_summary}\n\nResult: 1 of 1 tests passed",
            BLUE,
        ),
        (
            "3  Predictions fixed",
            "\n".join(_short_prediction(row) for row in data.predictions),
            PURPLE,
        ),
        (
            "4  Later task",
            "A separate tag-removal task was revealed only after the predictions.\n\n"
            "Result: 2 of 2 tests passed",
            GREEN,
        ),
    )
    for index, (x, (title, body, colour)) in enumerate(zip(x_positions, cards, strict=True)):
        _presentation_card(
            axis,
            x=x,
            y=0.35,
            width=0.205,
            height=0.40,
            title=title,
            body=body,
            colour=colour,
            font_size=font_size,
        )
        if index < len(cards) - 1:
            _horizontal_arrow(axis, x + 0.21, x_positions[index + 1] - 0.005)
    _card_background(axis, x=0.04, y=0.095, width=0.94, height=0.165, colour=INK)
    axis.text(
        0.065,
        0.225,
        "5  What this means",
        transform=axis.transAxes,
        fontsize=font_size,
        fontweight="bold",
        color=INK,
        va="top",
    )
    axis.text(
        0.24,
        0.225,
        textwrap.fill(data.interpretation, width=105),
        transform=axis.transAxes,
        fontsize=font_size - 1.0,
        color=INK,
        va="top",
        linespacing=1.3,
    )


def _presentation_card(
    axis: Axes,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str,
    colour: str,
    font_size: float,
) -> None:
    _card_background(axis, x=x, y=y, width=width, height=height, colour=colour)
    axis.text(
        x + 0.018,
        y + height - 0.035,
        title,
        transform=axis.transAxes,
        fontsize=font_size,
        fontweight="bold",
        color=INK,
        va="top",
    )
    axis.text(
        x + 0.018,
        y + height - 0.102,
        textwrap.fill(body, width=28, replace_whitespace=False),
        transform=axis.transAxes,
        fontsize=font_size - 1.2,
        color=INK,
        va="top",
        linespacing=1.3,
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


def _vertical_arrow(axis: Axes, start_y: float, end_y: float) -> None:
    axis.add_patch(
        FancyArrowPatch(
            (0.5, start_y),
            (0.5, end_y),
            transform=axis.transAxes,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.0,
            color=MUTED,
        )
    )


def _horizontal_arrow(axis: Axes, start_x: float, end_x: float) -> None:
    axis.add_patch(
        FancyArrowPatch(
            (start_x, 0.55),
            (end_x, 0.55),
            transform=axis.transAxes,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.2,
            color=MUTED,
        )
    )


def _short_prediction(row: WorkedExamplePrediction) -> str:
    answer = "yes" if row.predicts_success else "no"
    return f"{row.label}: {row.mastery_probability:.0%} -> {answer}"


def _source_table_csv(data: WorkedExampleData) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(("stage", "record", "value", "source_hash"))
    writer.writerow((1, "public_response", data.public_rating_category, data.public_response_hash))
    writer.writerow(
        (
            2,
            "executable_evidence",
            f"{data.evidence_passed}_passed_{data.evidence_failed}_failed",
            data.evidence_response_hash,
        )
    )
    for prediction in data.predictions:
        writer.writerow(
            (
                3,
                prediction.condition.value,
                f"{prediction.mastery_probability:.12g}",
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
