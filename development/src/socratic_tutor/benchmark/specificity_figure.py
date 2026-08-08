"""Reproducible figure for the sealed evidence-specificity analysis."""

# pyright: reportArgumentType=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import csv
import math
from datetime import UTC, datetime
from io import BytesIO, StringIO
from pathlib import Path
from typing import Literal

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure
from pydantic import Field, ValidationError, model_validator

from socratic_tutor.benchmark.artifacts import write_immutable_bytes, write_immutable_json
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.evaluator.aggregation import CaseComparison
from socratic_tutor.benchmark.evidence_specificity import EvidenceSpecificityReport
from socratic_tutor.benchmark.hashing import file_sha256, model_content_hash
from socratic_tutor.benchmark.result_interpretation import (
    ResultInterpretationPlan,
    ResultInterpretationReport,
)
from socratic_tutor.benchmark.secondary_analysis import SecondaryAnalysisReport
from socratic_tutor.contracts import ContractModel

_BLUE = "#0072B2"
_ORANGE = "#D55E00"
_GREEN = "#009E73"
_GREY = "#8A949E"
_LIGHT_GREY = "#D7DDE2"
_PALE_GREY = "#F2F4F5"
_INK = "#17212B"
_MUTED = "#56616B"
_TITLE = "Did relevant executable evidence outperform contrastive controls?"
_STYLE: dict[str, object] = {
    "axes.edgecolor": _LIGHT_GREY,
    "axes.labelcolor": _INK,
    "axes.linewidth": 0.8,
    "axes.titlecolor": _INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "figure.facecolor": "white",
    "font.family": ["DejaVu Sans", "sans-serif"],
    "font.size": 9,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "svg.fonttype": "none",
    "svg.hashsalt": "socratic-tutor-specificity-result-v1",
    "text.color": _INK,
    "xtick.color": _MUTED,
    "ytick.color": _MUTED,
}


class SpecificityFigureError(ValueError):
    """Specificity-figure sources are missing, invalid, or inconsistent."""


class SpecificityFigureManifest(ContractModel):
    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.specificity_figure_manifest.v1"] = (
        "benchmark.specificity_figure_manifest.v1"
    )
    benchmark_version: Literal["v1"] = "v1"
    run_id: str = Field(min_length=1)
    secondary_report_hash: Sha256
    specificity_report_hash: Sha256
    interpretation_plan_hash: Sha256
    interpretation_report_hash: Sha256
    publication_code_revision: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    pixi_lock_sha256: Sha256
    panel_count: Literal[3] = 3
    figure_formats: tuple[Literal["pdf", "svg"], Literal["pdf", "svg"]]
    pdf_sha256: Sha256
    svg_sha256: Sha256
    case_results_csv_sha256: Sha256
    generated_at_utc: datetime
    specificity_conclusion: Literal["specific", "non_specific"]
    confirmatory_gate_allowed: Literal[False] = False
    primary_result_rescued: Literal[False] = False
    human_learning_claim_supported: Literal[False] = False
    tutoring_efficacy_claim_supported: Literal[False] = False
    manifest_hash: Sha256

    @model_validator(mode="after")
    def validate_manifest(self) -> SpecificityFigureManifest:
        _require_utc(self.generated_at_utc)
        if self.figure_formats != ("pdf", "svg"):
            raise ValueError("Specificity figure must publish PDF and SVG")
        if self.manifest_hash != model_content_hash(self, exclude={"manifest_hash"}):
            raise ValueError("Specificity-figure manifest hash does not match its content")
        return self


def render_specificity_figure(
    *,
    secondary_report_path: Path,
    interpretation_plan_path: Path,
    interpretation_report_path: Path,
    pixi_lock_path: Path,
    output_root: Path,
    publication_code_revision: str,
    generated_at_utc: datetime | None = None,
) -> SpecificityFigureManifest:
    """Render the case-level specificity result and its claim boundary."""

    secondary = _load(secondary_report_path, SecondaryAnalysisReport, "secondary report")
    interpretation_plan = _load(
        interpretation_plan_path,
        ResultInterpretationPlan,
        "interpretation plan",
    )
    interpretation = _load(
        interpretation_report_path,
        ResultInterpretationReport,
        "interpretation report",
    )
    try:
        pixi_lock_hash = file_sha256(pixi_lock_path.read_bytes())
    except OSError as error:
        raise SpecificityFigureError(f"Could not read Pixi lock: {pixi_lock_path}") from error
    _validate_sources(
        secondary=secondary,
        interpretation_plan=interpretation_plan,
        interpretation=interpretation,
        pixi_lock_hash=pixi_lock_hash,
    )
    generated_at = _resolve_generated_at(
        generated_at_utc,
        output_root / "specificity_figure_manifest.json",
    )
    specificity = secondary.evidence_specificity
    case_csv = _case_results_csv(secondary)
    pdf, svg = _render_vector_files(specificity, secondary, interpretation, generated_at)
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.specificity_figure_manifest.v1",
        "benchmark_version": "v1",
        "run_id": secondary.run_id,
        "secondary_report_hash": secondary.report_hash,
        "specificity_report_hash": specificity.report_hash,
        "interpretation_plan_hash": interpretation_plan.plan_hash,
        "interpretation_report_hash": interpretation.report_hash,
        "publication_code_revision": publication_code_revision,
        "pixi_lock_sha256": pixi_lock_hash,
        "panel_count": 3,
        "figure_formats": ("pdf", "svg"),
        "pdf_sha256": file_sha256(pdf),
        "svg_sha256": file_sha256(svg),
        "case_results_csv_sha256": file_sha256(case_csv),
        "generated_at_utc": generated_at,
        "specificity_conclusion": interpretation.specificity_conclusion,
        "confirmatory_gate_allowed": False,
        "primary_result_rescued": False,
        "human_learning_claim_supported": False,
        "tutoring_efficacy_claim_supported": False,
    }
    draft = SpecificityFigureManifest.model_construct(
        _fields_set=set(content),
        **content,
        manifest_hash="0" * 64,
    )
    manifest = SpecificityFigureManifest.model_validate(
        {**content, "manifest_hash": model_content_hash(draft, exclude={"manifest_hash"})}
    )
    write_immutable_bytes(output_root / "evidence_specificity.pdf", pdf)
    write_immutable_bytes(output_root / "evidence_specificity.svg", svg)
    write_immutable_bytes(output_root / "specificity_case_results.csv", case_csv)
    write_immutable_json(output_root / "specificity_figure_manifest.json", manifest)
    return manifest


def _case_results_csv(report: SecondaryAnalysisReport) -> bytes:
    specificity = report.evidence_specificity
    effects = {row.case_id: row for row in specificity.case_effects}
    missing_ids = _missing_case_ids(report)
    rows: list[tuple[object, ...]] = []
    for case_id in sorted((*effects, *missing_ids)):
        if case_id in effects:
            row = effects[case_id]
            rows.append(
                (
                    case_id,
                    "eligible",
                    row.valid_vs_unrelated_effect,
                    row.valid_vs_corrupted_effect,
                    row.combined_specificity_effect,
                    _effect_label(row.combined_specificity_effect),
                )
            )
        else:
            rows.append((case_id, "missing", "", "", "", "criterion result missing"))
    return _csv_bytes(
        (
            "case_id",
            "analysis_status",
            "valid_vs_unrelated_effect",
            "valid_vs_corrupted_effect",
            "valid_vs_average_control_effect",
            "case_interpretation",
        ),
        rows,
    )


def _render_vector_files(
    specificity: EvidenceSpecificityReport,
    secondary: SecondaryAnalysisReport,
    interpretation: ResultInterpretationReport,
    generated_at_utc: datetime,
) -> tuple[bytes, bytes]:
    with mpl.rc_context(_STYLE):
        figure = _build_figure(specificity, secondary, interpretation)
        pdf_buffer = BytesIO()
        svg_buffer = BytesIO()
        figure.savefig(
            pdf_buffer,
            format="pdf",
            metadata={
                "Title": _TITLE,
                "Author": "Anas Khan",
                "Subject": "Sealed secondary evidence-specificity result",
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
                    "Case-level and aggregate comparisons of relevant executable evidence "
                    "with unrelated and deliberately corrupted controls."
                ),
                "Creator": "Socratic Tutor reproducible figure pipeline",
                "Date": generated_at_utc.isoformat(),
            },
        )
        figure.clear()
    return pdf_buffer.getvalue(), svg_buffer.getvalue()


def _build_figure(
    specificity: EvidenceSpecificityReport,
    secondary: SecondaryAnalysisReport,
    interpretation: ResultInterpretationReport,
) -> Figure:
    figure = Figure(figsize=(13.4, 9.4))
    grid = figure.add_gridspec(
        2,
        2,
        left=0.075,
        right=0.98,
        top=0.68,
        bottom=0.17,
        width_ratios=(1.35, 1.15),
        height_ratios=(1.15, 0.85),
        hspace=0.48,
        wspace=0.42,
    )
    cases_axis = figure.add_subplot(grid[:, 0])
    interval_axis = figure.add_subplot(grid[0, 1])
    counts_axis = figure.add_subplot(grid[1, 1])

    _plot_case_contrasts(cases_axis, specificity, _missing_case_ids(secondary))
    _plot_intervals(interval_axis, specificity)
    _plot_combined_counts(counts_axis, specificity)

    inference = specificity.combined_specificity_inference
    figure.text(0.075, 0.955, _TITLE, fontsize=19, fontweight="bold", color=_INK)
    figure.text(
        0.075,
        0.912,
        "The same evaluation model received valid evidence, unrelated text, or deliberately "
        "corrupted evidence for each fixed case.",
        fontsize=10.5,
        color=_MUTED,
    )
    finding = (
        "Secondary result: relevant evidence performed better than the average control."
        if interpretation.specificity_conclusion == "specific"
        else "Secondary result: relevant evidence did not outperform the average control."
    )
    figure.text(0.075, 0.852, finding, fontsize=13, fontweight="bold", color=_GREEN)
    figure.text(
        0.075,
        0.817,
        f"Mean reduction in prediction error: {inference.mean_case_effect:.3f}; "
        f"95% paired interval [{inference.interval_lower:.3f}, "
        f"{inference.interval_upper:.3f}].",
        fontsize=10.5,
        color=_INK,
    )
    figure.text(
        0.075,
        0.755,
        "This strengthens the relevance check, not the primary conclusion. "
        "The primary benchmark remains inconclusive.",
        fontsize=10,
        fontweight="bold",
        color=_ORANGE,
    )
    figure.text(
        0.075,
        0.108,
        "+1 means valid evidence produced lower prediction error; 0 means a tie; "
        "-1 means the control produced lower error; X marks the missing case.",
        fontsize=9,
        color=_MUTED,
    )
    figure.text(
        0.075,
        0.068,
        "Scope: one pinned evaluation-model run on one fixed authored corpus. "
        "This does not show human learning, reduced dependence, or tutoring efficacy.",
        fontsize=9,
        fontweight="bold",
        color=_MUTED,
    )
    figure.text(
        0.075,
        0.030,
        f"{specificity.eligible_case_count} of {specificity.total_case_count} cases were eligible. "
        "Intervals are secondary uncertainty summaries; they cannot rescue the primary endpoint.",
        fontsize=8.5,
        color=_MUTED,
    )
    return figure


def _plot_case_contrasts(
    axis: Axes,
    report: EvidenceSpecificityReport,
    missing_ids: tuple[str, ...],
) -> None:
    effects = {row.case_id: row for row in report.case_effects}
    case_ids = sorted((*effects, *missing_ids))
    values: list[list[float]] = []
    for case_id in case_ids:
        if case_id not in effects:
            values.append([math.nan, math.nan])
            continue
        row = effects[case_id]
        values.append([row.valid_vs_unrelated_effect, row.valid_vs_corrupted_effect])

    colours = ListedColormap([_ORANGE, _PALE_GREY, _GREEN]).with_extremes(bad="white")
    norm = BoundaryNorm((-1.5, -0.5, 0.5, 1.5), colours.N)
    axis.imshow(values, cmap=colours, norm=norm, aspect="auto", interpolation="none")
    for row_index, row_values in enumerate(values):
        for column_index, value in enumerate(row_values):
            label = "X" if math.isnan(value) else f"{value:+.0f}" if value else "0"
            text_colour = "white" if not math.isnan(value) and value != 0 else _INK
            axis.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                color=text_colour,
                fontsize=8,
                fontweight="bold",
            )
    for boundary in (5.5, 11.5, 17.5):
        axis.axhline(boundary, color="white", linewidth=1.5)
    axis.set_title("A. Result for every authored case", loc="left", pad=10)
    axis.set_xticks(
        (0, 1),
        ("Valid evidence vs\nunrelated text", "Valid evidence vs\ncorrupted evidence"),
        fontsize=8.5,
    )
    axis.xaxis.tick_top()
    axis.tick_params(axis="x", length=0, pad=7)
    axis.set_yticks(range(len(case_ids)), case_ids, fontsize=7.5)
    axis.tick_params(axis="y", length=0)
    axis.set_xlim(-0.5, 1.5)
    axis.set_ylim(len(case_ids) - 0.5, -0.5)
    for spine in axis.spines.values():
        spine.set_visible(False)


def _plot_intervals(axis: Axes, report: EvidenceSpecificityReport) -> None:
    comparisons = (
        ("Valid vs unrelated", report.valid_vs_unrelated_inference),
        ("Valid vs corrupted", report.valid_vs_corrupted_inference),
        ("Valid vs average control", report.combined_specificity_inference),
    )
    positions = (2, 1, 0)
    axis.axvline(0.0, color=_INK, linewidth=1)
    for position, (_label, inference) in zip(positions, comparisons, strict=True):
        mean = inference.mean_case_effect
        axis.errorbar(
            [mean],
            [position],
            xerr=([mean - inference.interval_lower], [inference.interval_upper - mean]),
            fmt="o",
            markersize=7,
            color=_BLUE,
            ecolor=_BLUE,
            elinewidth=2,
            capsize=5,
            zorder=3,
        )
        axis.text(
            min(1.02, inference.interval_upper + 0.04),
            position,
            f"{mean:.3f}",
            va="center",
            fontsize=8.5,
            color=_INK,
        )
    axis.set_title("B. Average improvement over each control", loc="left", pad=10)
    axis.set_xlabel("Reduction in prediction error from using valid evidence")
    axis.set_yticks(positions, tuple(label for label, _ in comparisons), fontsize=8.5)
    axis.set_xlim(-0.12, 1.12)
    axis.set_ylim(-0.65, 2.65)
    _style_axis(axis)


def _plot_combined_counts(axis: Axes, report: EvidenceSpecificityReport) -> None:
    effects = tuple(row.combined_specificity_effect for row in report.case_effects)
    labels = ("Valid evidence better", "Tie", "Control better", "Missing")
    values = (
        sum(value > 0 for value in effects),
        sum(value == 0 for value in effects),
        sum(value < 0 for value in effects),
        report.missing_case_count,
    )
    colours = (_GREEN, _GREY, _ORANGE, _INK)
    positions = list(range(len(values)))
    bars = axis.barh(positions, values, color=colours, height=0.62)
    axis.set_title(
        f"C. Average-control result across all {report.total_case_count} cases",
        loc="left",
        pad=10,
    )
    axis.set_xlabel("Number of cases")
    axis.set_yticks(positions, labels, fontsize=8.5)
    axis.set_xlim(0, max(values) + 3)
    axis.invert_yaxis()
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            value + 0.25,
            bar.get_y() + bar.get_height() / 2,
            str(value),
            va="center",
            fontsize=9,
            color=_INK,
        )
    _style_axis(axis)


def _style_axis(axis: Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="x", color=_LIGHT_GREY, linewidth=0.7, alpha=0.7)
    axis.set_axisbelow(True)


def _missing_case_ids(report: SecondaryAnalysisReport) -> tuple[str, ...]:
    primary = next(
        row for row in report.comparison_summaries if row.comparison is CaseComparison.PRIMARY_VALID
    )
    missing_ids = tuple(sorted(primary.missing_case_ids))
    specificity = report.evidence_specificity
    if len(missing_ids) != specificity.missing_case_count:
        raise SpecificityFigureError("Specificity and secondary missing-case counts differ")
    if set(missing_ids) & {row.case_id for row in specificity.case_effects}:
        raise SpecificityFigureError("A specificity case cannot be eligible and missing")
    return missing_ids


def _effect_label(value: float) -> str:
    if value > 0:
        return "valid evidence better"
    if value < 0:
        return "control better"
    return "tie"


def _validate_sources(
    *,
    secondary: SecondaryAnalysisReport,
    interpretation_plan: ResultInterpretationPlan,
    interpretation: ResultInterpretationReport,
    pixi_lock_hash: Sha256,
) -> None:
    if interpretation_plan.secondary_report_hash != secondary.report_hash:
        raise SpecificityFigureError("Interpretation belongs to another secondary report")
    if interpretation.interpretation_plan_hash != interpretation_plan.plan_hash:
        raise SpecificityFigureError("Interpretation report and plan differ")
    if interpretation_plan.analysis_pixi_lock_hash != pixi_lock_hash:
        raise SpecificityFigureError("Interpretation used another Pixi lock")
    if {secondary.run_id, interpretation_plan.run_id, interpretation.run_id} != {secondary.run_id}:
        raise SpecificityFigureError("Figure sources belong to different benchmark runs")
    specificity = secondary.evidence_specificity
    expected = (
        specificity.combined_specificity_inference.mean_case_effect,
        specificity.combined_specificity_inference.interval_lower,
        specificity.combined_specificity_inference.interval_upper,
    )
    observed = (
        interpretation.specificity_mean_effect,
        interpretation.specificity_interval_lower,
        interpretation.specificity_interval_upper,
    )
    if observed != expected:
        raise SpecificityFigureError("Interpretation values differ from the specificity report")
    expected_conclusion = (
        "specific"
        if specificity.interpretation == "valid_evidence_is_more_specific_than_controls"
        else "non_specific"
    )
    if interpretation.specificity_conclusion != expected_conclusion:
        raise SpecificityFigureError(
            "Interpretation conclusion differs from the specificity report"
        )
    _missing_case_ids(secondary)


def _csv_bytes(headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> bytes:
    buffer = StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(headers)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _resolve_generated_at(value: datetime | None, manifest_path: Path) -> datetime:
    if value is not None:
        _require_utc(value)
        return value
    if manifest_path.exists():
        existing = _load(
            manifest_path,
            SpecificityFigureManifest,
            "specificity figure manifest",
        )
        return existing.generated_at_utc
    return datetime.now(UTC)


def _require_utc(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("Specificity-figure timestamp must be timezone-aware UTC")


def _load[ModelT: ContractModel](path: Path, model: type[ModelT], label: str) -> ModelT:
    try:
        return model.model_validate_json(path.read_bytes())
    except (OSError, ValidationError) as error:
        raise SpecificityFigureError(f"Could not verify {label}: {path}") from error
