"""Publish a case-level view of the completely corrected benchmark."""

# pyright: reportArgumentType=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Literal

import matplotlib as mpl
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure
from pydantic import Field, model_validator

from socratic_tutor.benchmark.artifacts import write_immutable_bytes, write_immutable_json
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.harness_correction_analysis import (
    HarnessCorrectionAnalysisPlan,
    HarnessCorrectionAnalysisReport,
    HarnessCorrectionCaseResult,
)
from socratic_tutor.benchmark.harness_correction_closure import (
    HarnessCorrectionClosurePlan,
    HarnessCorrectionClosureReport,
)
from socratic_tutor.benchmark.harness_correction_figure import (
    HarnessCorrectionFigureError,
    validate_harness_correction_sources,
)
from socratic_tutor.benchmark.hashing import file_sha256, model_content_hash
from socratic_tutor.benchmark.public.models import BenchmarkCondition
from socratic_tutor.contracts import ContractModel
from socratic_tutor.csv_output import csv_bytes

_BLUE = "#0072B2"
_ORANGE = "#D55E00"
_GREEN = "#009E73"
_GREY = "#8A949E"
_LIGHT_GREY = "#D7DDE2"
_INK = "#17212B"
_MUTED = "#56616B"
_TITLE = "What changed in each benchmark case?"
_CONDITION_LABELS = {
    BenchmarkCondition.DIALOGUE_ONLY: "Dialogue\nonly",
    BenchmarkCondition.PROBE_INFORMED: "Relevant\nprobe",
    BenchmarkCondition.UNRELATED_PROBE: "Unrelated\nprobe",
    BenchmarkCondition.CORRUPTED_PROBE: "Inverted\nprobe",
}
_STYLE: dict[str, object] = {
    "figure.facecolor": "#FAFAF7",
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans", "sans-serif"],
    "font.size": 11,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "svg.fonttype": "none",
    "svg.hashsalt": "socratic-tutor-harness-correction-cases-v1",
    "text.color": _INK,
    "xtick.color": _INK,
    "ytick.color": _MUTED,
}


class HarnessCorrectionCaseFigureManifest(ContractModel):
    """Source lineage and checksums for the corrected case matrix."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.harness_correction_case_figure_manifest.v1"] = (
        "benchmark.harness_correction_case_figure_manifest.v1"
    )
    benchmark_version: Literal["v1"] = "v1"
    run_id: str = Field(min_length=1)
    correction_analysis_plan_hash: Sha256
    correction_analysis_report_hash: Sha256
    correction_closure_plan_hash: Sha256
    correction_closure_report_hash: Sha256
    publication_code_revision: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    pixi_lock_sha256: Sha256
    pdf_sha256: Sha256
    svg_sha256: Sha256
    case_table_csv_sha256: Sha256
    total_case_count: Literal[24] = 24
    eligible_case_count: Literal[23] = 23
    missing_case_ids: tuple[Literal["h-c2m2-03"], ...] = ("h-c2m2-03",)
    corrected_improvement_count: Literal[1] = 1
    corrected_regression_count: Literal[0] = 0
    relevant_unrelated_decision_difference_count: Literal[0] = 0
    generated_at_utc: datetime
    corrected_result_is_post_hoc: Literal[True] = True
    human_learning_claim_supported: Literal[False] = False
    manifest_hash: Sha256

    @model_validator(mode="after")
    def validate_manifest(self) -> HarnessCorrectionCaseFigureManifest:
        _require_utc(self.generated_at_utc)
        if self.manifest_hash != model_content_hash(self, exclude={"manifest_hash"}):
            raise ValueError("Correction case figure manifest hash does not match")
        return self


def render_harness_correction_case_figure(
    *,
    correction_analysis_plan_path: Path,
    correction_analysis_report_path: Path,
    correction_closure_plan_path: Path,
    correction_closure_report_path: Path,
    pixi_lock_path: Path,
    output_root: Path,
    publication_code_revision: str,
    generated_at_utc: datetime | None = None,
) -> HarnessCorrectionCaseFigureManifest:
    """Render all corrected condition outcomes without recomputing the analysis."""

    analysis_plan = _load(
        correction_analysis_plan_path,
        HarnessCorrectionAnalysisPlan,
        "correction analysis plan",
    )
    analysis = _load(
        correction_analysis_report_path,
        HarnessCorrectionAnalysisReport,
        "correction analysis report",
    )
    closure_plan = _load(
        correction_closure_plan_path,
        HarnessCorrectionClosurePlan,
        "correction closure plan",
    )
    closure = _load(
        correction_closure_report_path,
        HarnessCorrectionClosureReport,
        "correction closure report",
    )
    try:
        lock_hash = file_sha256(pixi_lock_path.read_bytes())
    except OSError as error:
        raise HarnessCorrectionFigureError(f"Could not read Pixi lock: {pixi_lock_path}") from error
    validate_harness_correction_sources(
        analysis_plan=analysis_plan,
        analysis=analysis,
        closure_plan=closure_plan,
        closure=closure,
        pixi_lock_hash=lock_hash,
    )
    _validate_case_result(analysis, closure)
    generated_at = _resolve_generated_at(
        generated_at_utc,
        output_root / "harness_correction_case_figure_manifest.json",
    )
    case_table = _case_table_csv(analysis)
    pdf, svg = _render_vector_files(analysis, closure, generated_at)
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.harness_correction_case_figure_manifest.v1",
        "benchmark_version": "v1",
        "run_id": closure.run_id,
        "correction_analysis_plan_hash": analysis_plan.plan_hash,
        "correction_analysis_report_hash": analysis.report_hash,
        "correction_closure_plan_hash": closure_plan.plan_hash,
        "correction_closure_report_hash": closure.report_hash,
        "publication_code_revision": publication_code_revision,
        "pixi_lock_sha256": lock_hash,
        "pdf_sha256": file_sha256(pdf),
        "svg_sha256": file_sha256(svg),
        "case_table_csv_sha256": file_sha256(case_table),
        "total_case_count": 24,
        "eligible_case_count": 23,
        "missing_case_ids": ("h-c2m2-03",),
        "corrected_improvement_count": 1,
        "corrected_regression_count": 0,
        "relevant_unrelated_decision_difference_count": 0,
        "generated_at_utc": generated_at,
        "corrected_result_is_post_hoc": True,
        "human_learning_claim_supported": False,
    }
    draft = HarnessCorrectionCaseFigureManifest.model_construct(
        _fields_set=set(content),
        **content,
        manifest_hash="0" * 64,
    )
    manifest = HarnessCorrectionCaseFigureManifest.model_validate(
        {**content, "manifest_hash": model_content_hash(draft, exclude={"manifest_hash"})}
    )
    write_immutable_bytes(output_root / "harness_correction_cases.pdf", pdf)
    write_immutable_bytes(output_root / "harness_correction_cases.svg", svg)
    write_immutable_bytes(output_root / "harness_correction_cases.csv", case_table)
    write_immutable_json(
        output_root / "harness_correction_case_figure_manifest.json",
        manifest,
    )
    return manifest


def _validate_case_result(
    analysis: HarnessCorrectionAnalysisReport,
    closure: HarnessCorrectionClosureReport,
) -> None:
    if analysis.total_case_count != 24 or len(analysis.case_results) != 24:
        raise HarnessCorrectionFigureError("Case figure requires all 24 authored cases")
    if analysis.eligible_case_count != 23 or analysis.missing_case_ids != ("h-c2m2-03",):
        raise HarnessCorrectionFigureError("Case figure requires the reclosed case denominator")
    if analysis.improvement_count != 1 or analysis.regression_count != 0:
        raise HarnessCorrectionFigureError("Case figure requires the completely corrected result")
    differences = sum(
        _condition_correct(case, BenchmarkCondition.PROBE_INFORMED)
        != _condition_correct(case, BenchmarkCondition.UNRELATED_PROBE)
        for case in analysis.case_results
    )
    if differences != 0 or closure.corrected_valid_vs_unrelated_effect != 0.0:
        raise HarnessCorrectionFigureError("Relevant and unrelated corrected decisions differ")


def _case_table_csv(report: HarnessCorrectionAnalysisReport) -> bytes:
    rows: list[tuple[object, ...]] = []
    for case in report.case_results:
        rows.append(
            (
                case.case_id,
                "eligible" if case.primary_effect is not None else "missing_criterion",
                _status(_condition_correct(case, BenchmarkCondition.DIALOGUE_ONLY)),
                _status(_condition_correct(case, BenchmarkCondition.PROBE_INFORMED)),
                _status(_condition_correct(case, BenchmarkCondition.UNRELATED_PROBE)),
                _status(_condition_correct(case, BenchmarkCondition.CORRUPTED_PROBE)),
                case.primary_effect if case.primary_effect is not None else "",
            )
        )
    return csv_bytes(
        (
            "case_id",
            "analysis_status",
            "dialogue_only",
            "relevant_probe",
            "unrelated_probe",
            "inverted_probe",
            "relevant_probe_effect",
        ),
        rows,
    )


def _render_vector_files(
    analysis: HarnessCorrectionAnalysisReport,
    closure: HarnessCorrectionClosureReport,
    generated_at_utc: datetime,
) -> tuple[bytes, bytes]:
    with mpl.rc_context(_STYLE):
        figure = _build_figure(analysis, closure)
        pdf_buffer = BytesIO()
        svg_buffer = BytesIO()
        figure.savefig(
            pdf_buffer,
            format="pdf",
            metadata={
                "Title": _TITLE,
                "Author": "Anas Khan",
                "Subject": "Case-level post-hoc correction result",
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
                    "Corrected prediction outcomes for all authored benchmark cases and "
                    "four evidence conditions."
                ),
                "Creator": "Socratic Tutor reproducible figure pipeline",
                "Date": generated_at_utc.isoformat(),
            },
        )
        figure.clear()
    return pdf_buffer.getvalue(), svg_buffer.getvalue()


def _build_figure(
    analysis: HarnessCorrectionAnalysisReport,
    closure: HarnessCorrectionClosureReport,
) -> Figure:
    cases = tuple(sorted(analysis.case_results, key=lambda item: item.case_id))
    conditions = tuple(BenchmarkCondition)
    matrix = [
        [_matrix_value(_condition_correct(case, condition)) for condition in conditions]
        for case in cases
    ]
    effects = [case.primary_effect for case in cases]
    figure = Figure(figsize=(10.8, 11.6))
    grid = figure.add_gridspec(
        1,
        2,
        left=0.19,
        right=0.97,
        top=0.64,
        bottom=0.12,
        width_ratios=(3.5, 1.25),
        wspace=0.22,
    )
    matrix_axis = figure.add_subplot(grid[0, 0])
    effect_axis = figure.add_subplot(grid[0, 1], sharey=matrix_axis)
    figure.suptitle(_TITLE, x=0.08, y=0.965, ha="left", fontsize=22, fontweight="bold")
    figure.text(
        0.08,
        0.91,
        "Each cell shows whether the prediction matched the later coding task.",
        color=_MUTED,
        fontsize=12,
    )
    figure.text(
        0.08,
        0.855,
        "Only one of 23 eligible cases changed from wrong to right after adding the "
        "relevant probe; no case became worse.",
        color=_INK,
        fontsize=12,
        fontweight="bold",
    )
    figure.text(
        0.08,
        0.805,
        "Relevant and unrelated evidence led to the same decision in every eligible case.",
        color=_ORANGE,
        fontsize=12,
        fontweight="bold",
    )
    figure.text(
        0.08,
        0.755,
        "Cell key: R = correct prediction; W = wrong prediction; - = outcome not scored",
        color=_MUTED,
        fontsize=10,
    )
    figure.text(
        0.19,
        0.695,
        "A. Result for each condition",
        color=_INK,
        fontsize=12,
        fontweight="bold",
    )
    figure.text(
        0.69,
        0.695,
        "B. Change after the relevant probe",
        color=_INK,
        fontsize=12,
        fontweight="bold",
    )
    colour_map = ListedColormap((_GREY, _ORANGE, _GREEN))
    normalizer = BoundaryNorm((-1.5, -0.5, 0.5, 1.5), colour_map.N)
    matrix_axis.imshow(
        matrix,
        cmap=colour_map,
        norm=normalizer,
        aspect="auto",
        interpolation="nearest",
    )
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            matrix_axis.text(
                column_index,
                row_index,
                {-1: "-", 0: "W", 1: "R"}[value],
                ha="center",
                va="center",
                color="white",
                fontsize=9.2,
                fontweight="bold",
            )
    matrix_axis.set_xticks(
        range(len(conditions)),
        tuple(_CONDITION_LABELS[condition] for condition in conditions),
    )
    matrix_axis.xaxis.tick_top()
    matrix_axis.tick_params(axis="x", length=0, pad=7)
    matrix_axis.set_yticks(range(len(cases)), tuple(case.case_id for case in cases), fontsize=8.5)
    matrix_axis.tick_params(axis="y", length=0, pad=6)
    for boundary in (2.5, 5.5, 8.5, 11.5, 14.5, 17.5, 20.5):
        matrix_axis.axhline(boundary, color="white", linewidth=2.0)
    for boundary in (0.5, 1.5, 2.5):
        matrix_axis.axvline(boundary, color="white", linewidth=1.2)
    for spine in matrix_axis.spines.values():
        spine.set_visible(False)

    for row_index, effect in enumerate(effects):
        if effect is None:
            effect_axis.scatter(
                [-2.0],
                [row_index],
                marker="x",
                color=_GREY,
                s=34,
                linewidths=1.4,
            )
        elif effect > 0:
            effect_axis.scatter(
                [effect],
                [row_index],
                marker="s",
                color=_GREEN,
                s=38,
            )
        elif effect < 0:
            effect_axis.scatter(
                [effect],
                [row_index],
                marker="v",
                color=_ORANGE,
                s=42,
            )
        else:
            effect_axis.scatter(
                [0.0],
                [row_index],
                marker="o",
                facecolors="white",
                edgecolors=_MUTED,
                s=30,
                linewidths=1.1,
            )
    effect_axis.axvline(0.0, color=_LIGHT_GREY, linewidth=1)
    for boundary in (2.5, 5.5, 8.5, 11.5, 14.5, 17.5, 20.5):
        effect_axis.axhline(boundary, color=_LIGHT_GREY, linewidth=0.8)
    effect_axis.set_xticks(
        (-2.0, -1.0, 0.0, 1.0),
        ("Not\nscored", "Worse", "Same", "Better"),
        fontsize=7.5,
    )
    effect_axis.xaxis.tick_top()
    effect_axis.tick_params(axis="x", length=0, pad=7)
    effect_axis.tick_params(axis="y", left=False, labelleft=False)
    effect_axis.set_xlim(-2.25, 1.25)
    effect_axis.set_ylim(len(cases) - 0.5, -0.5)
    for spine in effect_axis.spines.values():
        spine.set_visible(False)
    figure.text(
        0.08,
        0.055,
        f"Post-hoc corrected replay; {closure.total_case_count} authored cases; "
        f"{closure.eligible_case_count} eligible; missing criterion: "
        f"{', '.join(closure.missing_case_ids)}. One pinned evaluation model and one run "
        "per case; no claim about human learning or tutoring effectiveness.",
        color=_MUTED,
        fontsize=9.2,
    )
    return figure


def _condition_correct(
    case: HarnessCorrectionCaseResult,
    condition: BenchmarkCondition,
) -> bool | None:
    for score in case.conditions:
        if score.condition is condition:
            return score.correct
    raise HarnessCorrectionFigureError(
        f"Case {case.case_id} is missing condition {condition.value}"
    )


def _matrix_value(correct: bool | None) -> int:
    if correct is None:
        return -1
    return int(correct)


def _status(correct: bool | None) -> str:
    if correct is None:
        return "not_scored"
    return "right" if correct else "wrong"


def _resolve_generated_at(value: datetime | None, manifest_path: Path) -> datetime:
    if value is not None:
        _require_utc(value)
        return value
    if manifest_path.exists():
        existing = _load(
            manifest_path,
            HarnessCorrectionCaseFigureManifest,
            "correction case figure manifest",
        )
        return existing.generated_at_utc
    return datetime.now(UTC)


def _load[ModelT: ContractModel](path: Path, model: type[ModelT], name: str) -> ModelT:
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise HarnessCorrectionFigureError(f"Could not load {name}: {path}") from error


def _require_utc(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("Correction case figure timestamp must be timezone-aware UTC")
