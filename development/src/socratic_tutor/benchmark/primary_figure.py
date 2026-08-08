"""Reproducible case-level figure for the sealed primary benchmark result."""

# pyright: reportArgumentType=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import csv
from datetime import UTC, datetime
from io import BytesIO, StringIO
from pathlib import Path
from typing import Literal

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pydantic import Field, ValidationError, model_validator

from socratic_tutor.benchmark.artifacts import write_immutable_bytes, write_immutable_json
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import file_sha256, model_content_hash
from socratic_tutor.benchmark.primary_analysis import PrimaryAnalysisReport
from socratic_tutor.benchmark.result_interpretation import (
    ResultInterpretationPlan,
    ResultInterpretationReport,
)
from socratic_tutor.contracts import ContractModel

_BLUE = "#0072B2"
_ORANGE = "#D55E00"
_GREEN = "#009E73"
_GREY = "#8A949E"
_LIGHT_GREY = "#D7DDE2"
_INK = "#17212B"
_MUTED = "#56616B"
_TITLE = "Did executable evidence improve prediction of later task performance?"
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
    "svg.hashsalt": "socratic-tutor-primary-result-v1",
    "text.color": _INK,
    "xtick.color": _MUTED,
    "ytick.color": _MUTED,
}


class PrimaryResultFigureError(ValueError):
    """Primary-result figure sources are missing, invalid, or inconsistent."""


class PrimaryResultFigureManifest(ContractModel):
    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.primary_result_figure_manifest.v1"] = (
        "benchmark.primary_result_figure_manifest.v1"
    )
    benchmark_version: Literal["v1"] = "v1"
    run_id: str = Field(min_length=1)
    primary_report_hash: Sha256
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
    primary_conclusion: Literal["positive", "negative", "inconclusive"]
    human_learning_claim_supported: Literal[False] = False
    tutoring_efficacy_claim_supported: Literal[False] = False
    reduced_cognitive_offloading_claim_supported: Literal[False] = False
    manifest_hash: Sha256

    @model_validator(mode="after")
    def validate_manifest(self) -> PrimaryResultFigureManifest:
        _require_utc(self.generated_at_utc)
        if self.figure_formats != ("pdf", "svg"):
            raise ValueError("Primary result figure must publish PDF and SVG")
        if self.manifest_hash != model_content_hash(self, exclude={"manifest_hash"}):
            raise ValueError("Primary-result figure manifest hash does not match its content")
        return self


def render_primary_result_figure(
    *,
    primary_report_path: Path,
    interpretation_plan_path: Path,
    interpretation_report_path: Path,
    pixi_lock_path: Path,
    output_root: Path,
    publication_code_revision: str,
    generated_at_utc: datetime | None = None,
) -> PrimaryResultFigureManifest:
    """Render a case-level PDF/SVG and table from the sealed primary result."""

    primary = _load(primary_report_path, PrimaryAnalysisReport, "primary report")
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
        raise PrimaryResultFigureError(f"Could not read Pixi lock: {pixi_lock_path}") from error
    _validate_sources(
        primary=primary,
        interpretation_plan=interpretation_plan,
        interpretation=interpretation,
        pixi_lock_hash=pixi_lock_hash,
    )
    generated_at = _resolve_generated_at(
        generated_at_utc,
        output_root / "primary_result_figure_manifest.json",
    )
    case_csv = _case_results_csv(primary)
    pdf, svg = _render_vector_files(primary, interpretation, generated_at)
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.primary_result_figure_manifest.v1",
        "benchmark_version": "v1",
        "run_id": primary.run_id,
        "primary_report_hash": primary.report_hash,
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
        "primary_conclusion": interpretation.primary_conclusion,
        "human_learning_claim_supported": False,
        "tutoring_efficacy_claim_supported": False,
        "reduced_cognitive_offloading_claim_supported": False,
    }
    draft = PrimaryResultFigureManifest.model_construct(
        _fields_set=set(content),
        **content,
        manifest_hash="0" * 64,
    )
    manifest = PrimaryResultFigureManifest.model_validate(
        {**content, "manifest_hash": model_content_hash(draft, exclude={"manifest_hash"})}
    )
    write_immutable_bytes(output_root / "primary_result.pdf", pdf)
    write_immutable_bytes(output_root / "primary_result.svg", svg)
    write_immutable_bytes(output_root / "primary_case_results.csv", case_csv)
    write_immutable_json(output_root / "primary_result_figure_manifest.json", manifest)
    return manifest


def _case_results_csv(report: PrimaryAnalysisReport) -> bytes:
    eligible = {row.case_id: row for row in report.case_results}
    missing = {row.case_id: row for row in report.missing_cases}
    rows: list[tuple[object, ...]] = []
    for case_id in sorted((*eligible, *missing)):
        if case_id in eligible:
            row = eligible[case_id]
            rows.append(
                (
                    case_id,
                    "eligible",
                    row.dialogue_correct,
                    row.valid_evidence_correct,
                    row.case_effect,
                    "",
                )
            )
        else:
            rows.append((case_id, "missing", "", "", "", missing[case_id].missing_reason))
    return _csv_bytes(
        (
            "case_id",
            "analysis_status",
            "dialogue_only_correct",
            "probe_informed_correct",
            "paired_case_effect",
            "missing_reason",
        ),
        rows,
    )


def _render_vector_files(
    primary: PrimaryAnalysisReport,
    interpretation: ResultInterpretationReport,
    generated_at_utc: datetime,
) -> tuple[bytes, bytes]:
    with mpl.rc_context(_STYLE):
        figure = _build_figure(primary, interpretation)
        pdf_buffer = BytesIO()
        svg_buffer = BytesIO()
        figure.savefig(
            pdf_buffer,
            format="pdf",
            metadata={
                "Title": _TITLE,
                "Author": "Anas Khan",
                "Subject": "Sealed paired benchmark result",
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
                    "Case-level and aggregate comparison of dialogue-only and executable-"
                    "evidence-informed predictions on a fixed authored benchmark."
                ),
                "Creator": "Socratic Tutor reproducible figure pipeline",
                "Date": generated_at_utc.isoformat(),
            },
        )
        figure.clear()
    return pdf_buffer.getvalue(), svg_buffer.getvalue()


def _build_figure(
    primary: PrimaryAnalysisReport,
    interpretation: ResultInterpretationReport,
) -> Figure:
    figure = Figure(figsize=(13.4, 9.4))
    grid = figure.add_gridspec(
        2,
        2,
        left=0.075,
        right=0.98,
        top=0.73,
        bottom=0.16,
        width_ratios=(1.55, 1.0),
        height_ratios=(1.0, 1.0),
        hspace=0.48,
        wspace=0.34,
    )
    cases_axis = figure.add_subplot(grid[:, 0])
    interval_axis = figure.add_subplot(grid[0, 1])
    counts_axis = figure.add_subplot(grid[1, 1])
    figure.suptitle(_TITLE, x=0.075, y=0.965, ha="left", fontsize=19, fontweight="bold")
    figure.text(
        0.075,
        0.905,
        f"One pinned evaluation model; {primary.total_case_count} fixed authored cases; "
        "one response per condition and case.",
        color=_MUTED,
        fontsize=10.5,
    )
    figure.text(
        0.075,
        0.858,
        "The prediction used either the public answer alone, or the public answer plus an "
        "executable diagnostic task.",
        color=_MUTED,
        fontsize=10.5,
    )
    figure.text(
        0.075,
        0.811,
        _conclusion_text(primary, interpretation),
        color=_conclusion_colour(interpretation),
        fontsize=10.5,
        fontweight="bold",
    )
    _plot_case_effects(cases_axis, primary)
    _plot_interval(interval_axis, primary)
    _plot_counts(counts_axis, primary)
    figure.text(
        0.075,
        0.065,
        "Scope: this result concerns prediction on one authored benchmark. It does not show "
        "that students learned more, that the tutor taught effectively, or that dependence fell.",
        color=_MUTED,
        fontsize=9.5,
        fontweight="bold",
    )
    figure.text(
        0.075,
        0.028,
        f"Across {interpretation.eligible_case_count} eligible cases, the mean paired effect was "
        f"{interpretation.primary_mean_effect:.3f}; exact McNemar p = "
        f"{interpretation.exact_mcnemar_p_value:.3f}. "
        f"{_missing_case_text(primary.missing_case_count)}",
        color=_MUTED,
        fontsize=9,
    )
    return figure


def _plot_case_effects(axis: Axes, report: PrimaryAnalysisReport) -> None:
    eligible = {row.case_id: row for row in report.case_results}
    missing = {row.case_id: row for row in report.missing_cases}
    case_ids = sorted((*eligible, *missing))
    positions = list(range(len(case_ids)))
    styles = {
        -1.0: (_ORANGE, "v", "Executable evidence harmed"),
        0.0: (_GREY, "o", "No change"),
        1.0: (_GREEN, "s", "Executable evidence helped"),
    }
    for effect, (colour, marker, _label) in styles.items():
        selected = [
            index
            for index, case_id in enumerate(case_ids)
            if case_id in eligible and eligible[case_id].case_effect == effect
        ]
        axis.scatter(
            [effect] * len(selected),
            selected,
            color=colour,
            marker=marker,
            s=44,
            zorder=3,
        )
    missing_positions = [index for index, case_id in enumerate(case_ids) if case_id in missing]
    axis.scatter(
        [-1.45] * len(missing_positions),
        missing_positions,
        color=_INK,
        marker="x",
        s=50,
        linewidths=1.6,
        zorder=3,
    )
    for boundary in (5.5, 11.5, 17.5):
        axis.axhline(boundary, color=_LIGHT_GREY, linewidth=0.8)
    axis.axvline(0.0, color=_LIGHT_GREY, linewidth=0.8)
    axis.set_title("A. Result for every authored case", loc="left", pad=10)
    axis.set_xticks(
        (-1.45, -1.0, 0.0, 1.0),
        ("Missing", "Harmed", "No change", "Helped"),
        fontsize=8,
    )
    axis.set_yticks(positions, case_ids, fontsize=7.5)
    axis.set_xlim(-1.7, 1.25)
    axis.set_ylim(-0.8, len(case_ids) - 0.2)
    axis.invert_yaxis()
    axis.set_xlabel("Effect of adding executable evidence")
    _style_axis(axis, grid=False)


def _plot_interval(axis: Axes, report: PrimaryAnalysisReport) -> None:
    inference = report.inference
    mean = inference.mean_case_effect
    axis.axvline(0.0, color=_INK, linewidth=1)
    axis.axvline(
        inference.minimum_interpretable_effect,
        color=_ORANGE,
        linewidth=1,
        linestyle=(0, (3, 3)),
    )
    axis.errorbar(
        [mean],
        [0],
        xerr=(
            [mean - inference.interval_lower],
            [inference.interval_upper - mean],
        ),
        fmt="o",
        markersize=8,
        color=_BLUE,
        ecolor=_BLUE,
        elinewidth=2,
        capsize=5,
        zorder=3,
    )
    axis.set_title("B. Average paired effect", loc="left", pad=10)
    axis.set_xlabel("Lower prediction error with executable evidence")
    axis.set_yticks(())
    axis.set_ylim(-0.6, 0.6)
    lower = min(-0.05, inference.interval_lower - 0.06)
    upper = max(0.65, inference.interval_upper + 0.06)
    axis.set_xlim(lower, upper)
    axis.annotate(
        f"Mean {mean:.3f}\n95% interval [{inference.interval_lower:.3f}, "
        f"{inference.interval_upper:.3f}]",
        (mean, 0),
        xytext=(0, -34),
        textcoords="offset points",
        ha="center",
        fontsize=8.5,
        color=_INK,
    )
    axis.annotate(
        f"Fixed practical threshold ({inference.minimum_interpretable_effect:.2f})",
        (inference.minimum_interpretable_effect, 0.42),
        xytext=(4, 0),
        textcoords="offset points",
        fontsize=8,
        color=_ORANGE,
    )
    axis.text(
        0.02,
        0.08,
        f"Exact McNemar p = {report.mcnemar.two_sided_exact_p_value:.3f}",
        transform=axis.transAxes,
        fontsize=8.5,
        color=_MUTED,
    )
    _style_axis(axis)


def _plot_counts(axis: Axes, report: PrimaryAnalysisReport) -> None:
    labels = ("Helped", "No change", "Harmed", "Missing")
    no_change = report.eligible_case_count - (
        report.valid_evidence_improvement_count + report.valid_evidence_regression_count
    )
    values = (
        report.valid_evidence_improvement_count,
        no_change,
        report.valid_evidence_regression_count,
        report.missing_case_count,
    )
    colours = (_GREEN, _GREY, _ORANGE, _INK)
    positions = list(range(len(values)))
    bars = axis.barh(positions, values, color=colours, height=0.62)
    axis.set_title(
        f"C. What happened across all {report.total_case_count} cases",
        loc="left",
        pad=10,
    )
    axis.set_xlabel("Number of cases")
    axis.set_yticks(positions, labels)
    axis.set_xlim(0, max(values) + 2)
    axis.invert_yaxis()
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            value + 0.2,
            bar.get_y() + bar.get_height() / 2,
            str(value),
            va="center",
            fontsize=9,
            color=_INK,
        )
    _style_axis(axis)


def _style_axis(axis: Axes, *, grid: bool = True) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if grid:
        axis.grid(axis="x", color=_LIGHT_GREY, linewidth=0.7, alpha=0.7)
    axis.set_axisbelow(True)


def _conclusion_text(
    primary: PrimaryAnalysisReport,
    interpretation: ResultInterpretationReport,
) -> str:
    conclusion = interpretation.primary_conclusion
    if conclusion == "inconclusive":
        return (
            "Primary conclusion: inconclusive. The observed gain was "
            f"{primary.net_improvement_count} net cases, but the uncertainty interval crossed "
            "the fixed practical threshold."
        )
    if conclusion == "positive":
        return (
            "Primary conclusion: positive. The full uncertainty interval met the fixed "
            "practical threshold."
        )
    return "Primary conclusion: negative. Executable evidence increased prediction error."


def _conclusion_colour(interpretation: ResultInterpretationReport) -> str:
    if interpretation.primary_conclusion == "positive":
        return _GREEN
    return _ORANGE


def _missing_case_text(count: int) -> str:
    if count == 0:
        return "No criterion result was missing."
    noun = "case was" if count == 1 else "cases were"
    return f"{count} {noun} missing from the paired analysis."


def _validate_sources(
    *,
    primary: PrimaryAnalysisReport,
    interpretation_plan: ResultInterpretationPlan,
    interpretation: ResultInterpretationReport,
    pixi_lock_hash: Sha256,
) -> None:
    if interpretation_plan.primary_report_hash != primary.report_hash:
        raise PrimaryResultFigureError("Interpretation belongs to another primary report")
    if interpretation.interpretation_plan_hash != interpretation_plan.plan_hash:
        raise PrimaryResultFigureError("Interpretation report and plan differ")
    if interpretation_plan.analysis_pixi_lock_hash != pixi_lock_hash:
        raise PrimaryResultFigureError("Interpretation used another Pixi lock")
    if {primary.run_id, interpretation_plan.run_id, interpretation.run_id} != {primary.run_id}:
        raise PrimaryResultFigureError("Figure sources belong to different benchmark runs")
    if primary.eligible_case_count != interpretation.eligible_case_count:
        raise PrimaryResultFigureError("Figure sources use different eligible-case counts")
    if primary.missing_case_count != interpretation.missing_case_count:
        raise PrimaryResultFigureError("Figure sources use different missing-case counts")
    expected = (
        primary.inference.mean_case_effect,
        primary.inference.interval_lower,
        primary.inference.interval_upper,
        primary.inference.minimum_interpretable_effect,
        primary.mcnemar.two_sided_exact_p_value,
    )
    observed = (
        interpretation.primary_mean_effect,
        interpretation.primary_interval_lower,
        interpretation.primary_interval_upper,
        interpretation.minimum_interpretable_effect,
        interpretation.exact_mcnemar_p_value,
    )
    if observed != expected:
        raise PrimaryResultFigureError("Interpretation values differ from the primary report")


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
            PrimaryResultFigureManifest,
            "primary-result figure manifest",
        )
        return existing.generated_at_utc
    return datetime.now(UTC)


def _require_utc(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("Primary-result figure timestamp must be timezone-aware UTC")


def _load[ModelT: ContractModel](path: Path, model: type[ModelT], label: str) -> ModelT:
    try:
        return model.model_validate_json(path.read_bytes())
    except (OSError, ValidationError) as error:
        raise PrimaryResultFigureError(f"Could not verify {label}: {path}") from error
