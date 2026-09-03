"""Reproducible study-flow figure for the sealed external benchmark."""

# pyright: reportArgumentType=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import textwrap
from collections import Counter
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Literal

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch
from pydantic import Field, ValidationError, model_validator

from socratic_tutor.benchmark.artifacts import write_immutable_bytes, write_immutable_json
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.design import BenchmarkDesignPlan, DesignStatus, load_design
from socratic_tutor.benchmark.external_criterion import (
    ExternalCriterionPlan,
    ExternalCriterionReport,
)
from socratic_tutor.benchmark.external_seal import (
    ExternalDecisionSealPlan,
    ExternalDecisionSealReport,
)
from socratic_tutor.benchmark.hashing import file_sha256, model_content_hash
from socratic_tutor.contracts import ContractModel
from socratic_tutor.csv_output import csv_bytes

_BLUE = "#0072B2"
_ORANGE = "#D55E00"
_GREEN = "#009E73"
_PURPLE = "#7A5195"
_GREY = "#8A949E"
_LIGHT_GREY = "#D7DDE2"
_PALE_GREY = "#F5F7F8"
_INK = "#17212B"
_MUTED = "#56616B"
_TITLE = "Predictions were saved before the outcome"
_STYLE: dict[str, object] = {
    "figure.facecolor": "#FAFAF7",
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans", "sans-serif"],
    "font.size": 11,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "svg.fonttype": "none",
    "svg.hashsalt": "socratic-tutor-study-design-v1",
    "text.color": _INK,
}


class StudyDesignFigureError(ValueError):
    """Study-design sources are missing, invalid, or inconsistent."""


class StudyDesignFigureManifest(ContractModel):
    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.study_design_figure_manifest.v1"] = (
        "benchmark.study_design_figure_manifest.v1"
    )
    benchmark_version: Literal["v1"] = "v1"
    run_id: str = Field(min_length=1)
    benchmark_design_sha256: Sha256
    decision_seal_plan_hash: Sha256
    decision_seal_report_hash: Sha256
    criterion_plan_hash: Sha256
    criterion_report_hash: Sha256
    execution_pixi_lock_sha256: Sha256
    publication_pixi_lock_sha256: Sha256
    publication_code_revision: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    provider_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    held_out_case_count: Literal[24] = 24
    sealed_prediction_count: Literal[96] = 96
    completed_criterion_count: int = Field(ge=0, le=24)
    missing_criterion_count: int = Field(ge=0, le=24)
    figure_formats: tuple[Literal["pdf", "svg"], Literal["pdf", "svg"]]
    pdf_sha256: Sha256
    svg_sha256: Sha256
    stage_data_csv_sha256: Sha256
    generated_at_utc: datetime
    predictions_sealed_before_criterion_reveal: Literal[True] = True
    evaluation_model_is_validated_student_simulator: Literal[False] = False
    human_learning_claim_supported: Literal[False] = False
    manifest_hash: Sha256

    @model_validator(mode="after")
    def validate_manifest(self) -> StudyDesignFigureManifest:
        _require_utc(self.generated_at_utc)
        if self.figure_formats != ("pdf", "svg"):
            raise ValueError("Study-design figure must publish PDF and SVG")
        if self.completed_criterion_count + self.missing_criterion_count != 24:
            raise ValueError("Study-design criterion counts do not reconcile")
        if self.manifest_hash != model_content_hash(self, exclude={"manifest_hash"}):
            raise ValueError("Study-design figure manifest hash does not match its content")
        return self


def render_study_design_figure(
    *,
    benchmark_design_path: Path,
    decision_seal_plan_path: Path,
    decision_seal_report_path: Path,
    criterion_plan_path: Path,
    criterion_report_path: Path,
    pixi_lock_path: Path,
    output_root: Path,
    publication_code_revision: str,
    generated_at_utc: datetime | None = None,
) -> StudyDesignFigureManifest:
    """Render the actual commit-before-reveal sequence from verified source records."""

    design = _load_design(benchmark_design_path)
    decision_plan = _load(
        decision_seal_plan_path,
        ExternalDecisionSealPlan,
        "decision-seal plan",
    )
    decision_report = _load(
        decision_seal_report_path,
        ExternalDecisionSealReport,
        "decision-seal report",
    )
    criterion_plan = _load(criterion_plan_path, ExternalCriterionPlan, "criterion plan")
    criterion_report = _load(criterion_report_path, ExternalCriterionReport, "criterion report")
    try:
        design_hash = file_sha256(benchmark_design_path.read_bytes())
        publication_lock_hash = file_sha256(pixi_lock_path.read_bytes())
    except OSError as error:
        raise StudyDesignFigureError("Could not read a study-design source") from error
    _validate_sources(
        design=design,
        decision_plan=decision_plan,
        decision_report=decision_report,
        criterion_plan=criterion_plan,
        criterion_report=criterion_report,
    )
    generated_at = _resolve_generated_at(
        generated_at_utc,
        output_root / "study_design_figure_manifest.json",
    )
    stage_data = _stage_data_csv(
        design,
        decision_plan,
        decision_report,
        criterion_plan,
        criterion_report,
    )
    pdf, svg = _render_vector_files(
        design,
        decision_plan,
        decision_report,
        criterion_report,
        generated_at,
    )
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.study_design_figure_manifest.v1",
        "benchmark_version": "v1",
        "run_id": decision_plan.run_id,
        "benchmark_design_sha256": design_hash,
        "decision_seal_plan_hash": decision_plan.plan_hash,
        "decision_seal_report_hash": decision_report.report_hash,
        "criterion_plan_hash": criterion_plan.plan_hash,
        "criterion_report_hash": criterion_report.report_hash,
        "execution_pixi_lock_sha256": decision_plan.pixi_lock_hash,
        "publication_pixi_lock_sha256": publication_lock_hash,
        "publication_code_revision": publication_code_revision,
        "provider_id": decision_plan.model_route.provider,
        "model_id": decision_plan.model_route.model,
        "held_out_case_count": 24,
        "sealed_prediction_count": decision_report.prediction_count,
        "completed_criterion_count": criterion_report.completed_execution_count,
        "missing_criterion_count": criterion_report.sandbox_missing_count,
        "figure_formats": ("pdf", "svg"),
        "pdf_sha256": file_sha256(pdf),
        "svg_sha256": file_sha256(svg),
        "stage_data_csv_sha256": file_sha256(stage_data),
        "generated_at_utc": generated_at,
        "predictions_sealed_before_criterion_reveal": True,
        "evaluation_model_is_validated_student_simulator": False,
        "human_learning_claim_supported": False,
    }
    draft = StudyDesignFigureManifest.model_construct(
        _fields_set=set(content),
        **content,
        manifest_hash="0" * 64,
    )
    manifest = StudyDesignFigureManifest.model_validate(
        {**content, "manifest_hash": model_content_hash(draft, exclude={"manifest_hash"})}
    )
    write_immutable_bytes(output_root / "external_study_design.pdf", pdf)
    write_immutable_bytes(output_root / "external_study_design.svg", svg)
    write_immutable_bytes(output_root / "study_design_stages.csv", stage_data)
    write_immutable_json(output_root / "study_design_figure_manifest.json", manifest)
    return manifest


def _stage_data_csv(
    design: BenchmarkDesignPlan,
    decision_plan: ExternalDecisionSealPlan,
    decision_report: ExternalDecisionSealReport,
    criterion_plan: ExternalCriterionPlan,
    criterion_report: ExternalCriterionReport,
) -> bytes:
    difficulties = Counter(case.difficulty.value for case in design.held_out_cases)
    rows = (
        (1, "frozen_case_design", "held_out_cases", len(design.held_out_cases)),
        (1, "frozen_case_design", "concepts", len(design.concepts)),
        (
            1,
            "frozen_case_design",
            "misconceptions",
            sum(len(concept.misconception_ids) for concept in design.concepts),
        ),
        (1, "frozen_case_design", "low_difficulty_cases", difficulties["low"]),
        (1, "frozen_case_design", "medium_difficulty_cases", difficulties["medium"]),
        (1, "frozen_case_design", "high_difficulty_cases", difficulties["high"]),
        (2, "pre_reveal_generation", "public_responses", decision_report.complete_case_count),
        (2, "pre_reveal_generation", "evidence_responses", decision_report.complete_case_count),
        (
            2,
            "pre_reveal_generation",
            "evidence_sandbox_executions",
            decision_report.evidence_execution_count,
        ),
        (3, "sealed_predictions", "condition_predictions", decision_report.prediction_count),
        (3, "sealed_predictions", "conditions_per_case", 4),
        (4, "post_seal_criterion", "planned_criterion_cases", criterion_plan.expected_case_count),
        (
            4,
            "post_seal_criterion",
            "completed_criterion_executions",
            criterion_report.completed_execution_count,
        ),
        (
            4,
            "post_seal_criterion",
            "missing_criterion_executions",
            criterion_report.sandbox_missing_count,
        ),
        (
            5,
            "paired_comparison",
            "eligible_cases",
            criterion_report.completed_execution_count,
        ),
        (5, "paired_comparison", "repeats_per_case", 1),
        (0, "model", "provider", decision_plan.model_route.provider),
        (0, "model", "model", decision_plan.model_route.model),
    )
    return csv_bytes(("stage", "stage_id", "measure", "value"), rows)


def _render_vector_files(
    design: BenchmarkDesignPlan,
    decision_plan: ExternalDecisionSealPlan,
    decision_report: ExternalDecisionSealReport,
    criterion_report: ExternalCriterionReport,
    generated_at_utc: datetime,
) -> tuple[bytes, bytes]:
    with mpl.rc_context(_STYLE):
        figure = _build_figure(design, decision_plan, decision_report, criterion_report)
        pdf_buffer = BytesIO()
        svg_buffer = BytesIO()
        figure.savefig(
            pdf_buffer,
            format="pdf",
            metadata={
                "Title": _TITLE,
                "Author": "Anas Khan",
                "Subject": "Sealed external benchmark study design",
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
                "Description": (
                    "Five-stage flow showing that condition predictions were committed before "
                    "the later criterion outcome was requested and executed."
                ),
                "Creator": "Socratic Tutor reproducible figure pipeline",
                "Date": generated_at_utc.isoformat(),
            },
        )
        figure.clear()
    return pdf_buffer.getvalue(), svg_buffer.getvalue()


def _build_figure(
    design: BenchmarkDesignPlan,
    decision_plan: ExternalDecisionSealPlan,
    decision_report: ExternalDecisionSealReport,
    criterion_report: ExternalCriterionReport,
) -> Figure:
    figure = Figure(figsize=(6.0, 8.0), facecolor="white")
    axis = figure.add_axes((0, 0, 1, 1))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    figure.text(0.07, 0.97, textwrap.fill(_TITLE, 34), fontsize=16, fontweight="bold", va="top")
    figure.text(
        0.07,
        0.885,
        f"Evaluation model: {decision_plan.model_route.model}.\n"
        "The study measured performance on authored coding tasks.",
        fontsize=10.5,
        color=_MUTED,
        va="top",
    )

    misconception_count = sum(len(concept.misconception_ids) for concept in design.concepts)
    stages = (
        (
            "Prepare the cases",
            f"{len(design.held_out_cases)} cases cover",
            f"{len(design.concepts)} concepts and {misconception_count} misconceptions.",
            "The answer, probe and outcome use separate prompts and contexts.",
            _GREY,
        ),
        (
            "Collect the answer and probe",
            "Collect one public answer and one probe response per case.",
            (f"Run {decision_report.evidence_execution_count} probe programs in sandboxes."),
            "The outcome response has not been requested.",
            _BLUE,
        ),
        (
            "Save four predictions per case",
            "Use dialogue alone, relevant evidence, unrelated evidence or inverted evidence.",
            f"Save {decision_report.prediction_count} predictions",
            "with hashes for later verification.",
            _PURPLE,
        ),
        (
            "Request and execute the outcome task",
            "After verifying the saved predictions, request the separate outcome response.",
            f"{criterion_report.completed_execution_count} executions completed. "
            f"{criterion_report.sandbox_missing_count} outcome remained unavailable.",
            "Retain the original prediction records.",
            _GREEN,
        ),
        (
            "Compare predictions with outcomes",
            "Compare all four predictions with the same outcome for each of the",
            (f"{criterion_report.completed_execution_count} eligible cases."),
            "Report differences, uncertainty and missing outcomes.",
            _INK,
        ),
    )
    y_positions = (0.70, 0.555, 0.41, 0.265, 0.12)
    for index, (stage, left, middle, right, colour) in enumerate(stages, start=1):
        _draw_stage(axis, index, y_positions[index - 1], stage, left, middle, right, colour)
        if index < len(stages):
            axis.annotate(
                "",
                xy=(0.06, y_positions[index] + 0.08),
                xytext=(0.06, y_positions[index - 1] + 0.04),
                arrowprops={"arrowstyle": "-|>", "color": _GREY, "linewidth": 1.2},
            )

    figure.text(
        0.07,
        0.065,
        "Hashes support checking for changes to saved files.\n"
        "Local files can still be edited outside the workflow.\n"
        "No human learning outcome was measured.",
        fontsize=10,
        color=_MUTED,
        va="top",
    )
    return figure


def _draw_stage(
    axis: Axes,
    number: int,
    y_position: float,
    title: str,
    left_text: str,
    middle_text: str,
    right_text: str,
    colour: str,
) -> None:
    box = FancyBboxPatch(
        (0.11, y_position),
        0.85,
        0.125,
        boxstyle="round,pad=0.004,rounding_size=0.008",
        facecolor=_PALE_GREY,
        edgecolor=_LIGHT_GREY,
        linewidth=0.9,
    )
    axis.add_patch(box)
    axis.add_patch(
        FancyBboxPatch(
            (0.11, y_position),
            0.008,
            0.125,
            boxstyle="round,pad=0.0,rounding_size=0.004",
            facecolor=colour,
            edgecolor=colour,
        )
    )
    axis.text(
        0.06,
        y_position + 0.0625,
        str(number),
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="white",
        bbox={"boxstyle": "circle,pad=0.35", "facecolor": colour, "edgecolor": colour},
    )
    axis.text(0.14, y_position + 0.101, title, fontsize=11.5, fontweight="bold")
    axis.text(
        0.14,
        y_position + 0.079,
        textwrap.fill(" ".join((left_text, middle_text, right_text)), width=64),
        fontsize=10.5,
        color=_MUTED,
        va="top",
        linespacing=1.15,
    )


def _validate_sources(
    *,
    design: BenchmarkDesignPlan,
    decision_plan: ExternalDecisionSealPlan,
    decision_report: ExternalDecisionSealReport,
    criterion_plan: ExternalCriterionPlan,
    criterion_report: ExternalCriterionReport,
) -> None:
    if design.status is not DesignStatus.FROZEN or design.benchmark_version != "v1":
        raise StudyDesignFigureError("Study-design figure requires frozen benchmark v1")
    case_count = len(design.held_out_cases)
    if decision_report.seal_plan_hash != decision_plan.plan_hash:
        raise StudyDesignFigureError("Decision report belongs to another seal plan")
    if criterion_report.criterion_plan_hash != criterion_plan.plan_hash:
        raise StudyDesignFigureError("Criterion report belongs to another criterion plan")
    if (
        len(
            {
                decision_plan.run_id,
                decision_report.run_id,
                criterion_plan.run_id,
                criterion_report.run_id,
            }
        )
        != 1
    ):
        raise StudyDesignFigureError("Study-design sources belong to different runs")
    if {decision_plan.protocol_hash, criterion_plan.protocol_hash} != {decision_plan.protocol_hash}:
        raise StudyDesignFigureError("Decision and criterion phases use different protocols")
    if decision_plan.source_manifest_hash != criterion_plan.source_manifest_hash:
        raise StudyDesignFigureError("Decision and criterion phases use different benchmarks")
    if decision_plan.model_route != criterion_plan.model_route:
        raise StudyDesignFigureError("Decision and criterion phases use different model routes")
    if decision_plan.pixi_lock_hash != criterion_plan.pixi_lock_hash:
        raise StudyDesignFigureError("Decision and criterion phases use different environments")
    if criterion_plan.global_seal_hash != decision_report.global_seal_hash:
        raise StudyDesignFigureError("Criterion phase does not follow the decision seal")
    if criterion_report.global_seal_hash != decision_report.global_seal_hash:
        raise StudyDesignFigureError("Criterion report does not follow the decision seal")
    if (
        case_count != decision_report.planned_case_count
        or case_count != criterion_plan.expected_case_count
    ):
        raise StudyDesignFigureError("Study-design case counts differ")
    if decision_report.prediction_count != case_count * 4:
        raise StudyDesignFigureError("Study design requires four sealed predictions per case")
    if not decision_report.all_cases_complete:
        raise StudyDesignFigureError("Decision phase did not seal every case")
    if criterion_plan.created_at_utc <= decision_report.sealed_at_utc:
        raise StudyDesignFigureError("Criterion plan was not created after the decision seal")
    if criterion_report.completed_at_utc < criterion_plan.created_at_utc:
        raise StudyDesignFigureError("Criterion report predates its plan")


def _resolve_generated_at(value: datetime | None, manifest_path: Path) -> datetime:
    if value is not None:
        _require_utc(value)
        return value
    if manifest_path.exists():
        existing = _load(
            manifest_path,
            StudyDesignFigureManifest,
            "study-design figure manifest",
        )
        return existing.generated_at_utc
    return datetime.now(UTC)


def _require_utc(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("Study-design figure timestamp must be timezone-aware UTC")


def _load_design(path: Path) -> BenchmarkDesignPlan:
    try:
        return load_design(path)
    except (OSError, ValidationError, ValueError) as error:
        raise StudyDesignFigureError(f"Could not verify benchmark design: {path}") from error


def _load[ModelT: ContractModel](path: Path, model: type[ModelT], label: str) -> ModelT:
    try:
        return model.model_validate_json(path.read_bytes())
    except (OSError, ValidationError) as error:
        raise StudyDesignFigureError(f"Could not verify {label}: {path}") from error
