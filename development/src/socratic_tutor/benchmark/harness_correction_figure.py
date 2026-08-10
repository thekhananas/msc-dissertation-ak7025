"""Publish the original and corrected benchmark results in one auditable figure."""

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
from pydantic import Field, model_validator

from socratic_tutor.benchmark.artifacts import write_immutable_bytes, write_immutable_json
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.harness_correction_analysis import (
    HarnessCorrectionAnalysisPlan,
    HarnessCorrectionAnalysisReport,
)
from socratic_tutor.benchmark.harness_correction_closure import (
    HarnessCorrectionClosurePlan,
    HarnessCorrectionClosureReport,
)
from socratic_tutor.benchmark.hashing import file_sha256, model_content_hash
from socratic_tutor.benchmark.public.models import BenchmarkCondition
from socratic_tutor.contracts import ContractModel

_BLUE = "#0072B2"
_ORANGE = "#D55E00"
_GREEN = "#009E73"
_GREY = "#7A858F"
_LIGHT_GREY = "#D7DDE2"
_INK = "#17212B"
_MUTED = "#56616B"
_TITLE = "A test-harness fault changed the benchmark conclusion"
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
    "svg.hashsalt": "socratic-tutor-harness-correction-v1",
    "text.color": _INK,
    "xtick.color": _MUTED,
    "ytick.color": _MUTED,
}


class HarnessCorrectionFigureError(ValueError):
    """Figure sources are missing, invalid, or inconsistent."""


class HarnessCorrectionFigureManifest(ContractModel):
    """Source lineage and checksums for the corrected benchmark figure."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.harness_correction_figure_manifest.v1"] = (
        "benchmark.harness_correction_figure_manifest.v1"
    )
    benchmark_version: Literal["v1"] = "v1"
    run_id: str = Field(min_length=1)
    correction_analysis_plan_hash: Sha256
    correction_analysis_report_hash: Sha256
    correction_closure_plan_hash: Sha256
    correction_closure_report_hash: Sha256
    publication_code_revision: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    pixi_lock_sha256: Sha256
    figure_formats: tuple[Literal["pdf", "svg"], Literal["pdf", "svg"]]
    pdf_sha256: Sha256
    svg_sha256: Sha256
    summary_csv_sha256: Sha256
    case_results_csv_sha256: Sha256
    generated_at_utc: datetime
    corrected_result_is_post_hoc: Literal[True] = True
    original_seal_preserved: Literal[True] = True
    human_learning_claim_supported: Literal[False] = False
    tutoring_efficacy_claim_supported: Literal[False] = False
    reduced_cognitive_offloading_claim_supported: Literal[False] = False
    manifest_hash: Sha256

    @model_validator(mode="after")
    def validate_manifest(self) -> HarnessCorrectionFigureManifest:
        _require_utc(self.generated_at_utc)
        if self.figure_formats != ("pdf", "svg"):
            raise ValueError("Correction figure must publish PDF and SVG")
        if self.manifest_hash != model_content_hash(self, exclude={"manifest_hash"}):
            raise ValueError("Correction figure manifest hash does not match its content")
        return self


def render_harness_correction_figure(
    *,
    correction_analysis_plan_path: Path,
    correction_analysis_report_path: Path,
    correction_closure_plan_path: Path,
    correction_closure_report_path: Path,
    pixi_lock_path: Path,
    output_root: Path,
    publication_code_revision: str,
    generated_at_utc: datetime | None = None,
) -> HarnessCorrectionFigureManifest:
    """Render the complete correction result and bind it to verified sources."""

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
        pixi_lock_hash = file_sha256(pixi_lock_path.read_bytes())
    except OSError as error:
        raise HarnessCorrectionFigureError(f"Could not read Pixi lock: {pixi_lock_path}") from error
    _validate_sources(
        analysis_plan=analysis_plan,
        analysis=analysis,
        closure_plan=closure_plan,
        closure=closure,
        pixi_lock_hash=pixi_lock_hash,
    )
    generated_at = _resolve_generated_at(
        generated_at_utc,
        output_root / "harness_correction_figure_manifest.json",
    )
    summary_csv = _summary_csv(analysis, closure)
    case_csv = _case_results_csv(analysis)
    pdf, svg = _render_vector_files(analysis, closure, generated_at)
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.harness_correction_figure_manifest.v1",
        "benchmark_version": "v1",
        "run_id": closure.run_id,
        "correction_analysis_plan_hash": analysis_plan.plan_hash,
        "correction_analysis_report_hash": analysis.report_hash,
        "correction_closure_plan_hash": closure_plan.plan_hash,
        "correction_closure_report_hash": closure.report_hash,
        "publication_code_revision": publication_code_revision,
        "pixi_lock_sha256": pixi_lock_hash,
        "figure_formats": ("pdf", "svg"),
        "pdf_sha256": file_sha256(pdf),
        "svg_sha256": file_sha256(svg),
        "summary_csv_sha256": file_sha256(summary_csv),
        "case_results_csv_sha256": file_sha256(case_csv),
        "generated_at_utc": generated_at,
        "corrected_result_is_post_hoc": True,
        "original_seal_preserved": True,
        "human_learning_claim_supported": False,
        "tutoring_efficacy_claim_supported": False,
        "reduced_cognitive_offloading_claim_supported": False,
    }
    draft = HarnessCorrectionFigureManifest.model_construct(
        _fields_set=set(content),
        **content,
        manifest_hash="0" * 64,
    )
    manifest = HarnessCorrectionFigureManifest.model_validate(
        {**content, "manifest_hash": model_content_hash(draft, exclude={"manifest_hash"})}
    )
    write_immutable_bytes(output_root / "harness_correction_result.pdf", pdf)
    write_immutable_bytes(output_root / "harness_correction_result.svg", svg)
    write_immutable_bytes(output_root / "harness_correction_summary.csv", summary_csv)
    write_immutable_bytes(output_root / "harness_correction_case_results.csv", case_csv)
    write_immutable_json(output_root / "harness_correction_figure_manifest.json", manifest)
    return manifest


def _summary_csv(
    analysis: HarnessCorrectionAnalysisReport,
    closure: HarnessCorrectionClosureReport,
) -> bytes:
    rows = [
        (
            "condition_accuracy",
            "dialogue_only",
            "original",
            closure.original_dialogue_correct_count,
            23,
            closure.original_dialogue_correct_count / 23,
        ),
        (
            "condition_accuracy",
            "probe_informed",
            "original",
            closure.original_probe_correct_count,
            23,
            closure.original_probe_correct_count / 23,
        ),
        (
            "condition_accuracy",
            "dialogue_only",
            "corrected",
            closure.corrected_dialogue_correct_count,
            23,
            closure.corrected_dialogue_correct_count / 23,
        ),
        (
            "condition_accuracy",
            "probe_informed",
            "corrected",
            closure.corrected_probe_correct_count,
            23,
            closure.corrected_probe_correct_count / 23,
        ),
        (
            "paired_effect",
            "probe_minus_dialogue",
            "original",
            closure.original_primary_effect,
            "",
            "",
        ),
        (
            "paired_effect",
            "probe_minus_dialogue",
            "corrected",
            closure.corrected_primary_effect,
            "",
            "",
        ),
        (
            "paired_effect",
            "valid_minus_unrelated",
            "corrected",
            analysis.valid_vs_unrelated_inference.mean_case_effect,
            "",
            "",
        ),
        (
            "paired_effect",
            "valid_minus_corrupted",
            "corrected",
            analysis.valid_vs_corrupted_inference.mean_case_effect,
            "",
            "",
        ),
    ]
    return _csv_bytes(
        (
            "measure",
            "comparison",
            "result_version",
            "value_or_correct_count",
            "eligible_case_count",
            "accuracy",
        ),
        rows,
    )


def _case_results_csv(report: HarnessCorrectionAnalysisReport) -> bytes:
    rows: list[tuple[object, ...]] = []
    for case in report.case_results:
        scores = {score.condition: score for score in case.conditions}
        rows.append(
            (
                case.case_id,
                "eligible" if case.primary_effect is not None else "missing_criterion",
                scores[BenchmarkCondition.DIALOGUE_ONLY].correct,
                scores[BenchmarkCondition.PROBE_INFORMED].correct,
                scores[BenchmarkCondition.UNRELATED_PROBE].correct,
                scores[BenchmarkCondition.CORRUPTED_PROBE].correct,
                case.primary_effect if case.primary_effect is not None else "",
                case.valid_vs_unrelated_effect
                if case.valid_vs_unrelated_effect is not None
                else "",
                case.valid_vs_corrupted_effect
                if case.valid_vs_corrupted_effect is not None
                else "",
            )
        )
    return _csv_bytes(
        (
            "case_id",
            "analysis_status",
            "dialogue_only_correct",
            "probe_informed_correct",
            "unrelated_probe_correct",
            "corrupted_probe_correct",
            "probe_minus_dialogue_effect",
            "valid_minus_unrelated_effect",
            "valid_minus_corrupted_effect",
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
                "Subject": "Complete post-hoc correction of the fixed authored benchmark",
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
                    "Original sealed and completely corrected benchmark results, including "
                    "relevance-control comparisons."
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
    figure = Figure(figsize=(13.4, 8.7))
    grid = figure.add_gridspec(
        1,
        3,
        left=0.07,
        right=0.98,
        top=0.67,
        bottom=0.23,
        width_ratios=(1.1, 1.0, 1.15),
        wspace=0.38,
    )
    accuracy_axis = figure.add_subplot(grid[0, 0])
    effect_axis = figure.add_subplot(grid[0, 1])
    relevance_axis = figure.add_subplot(grid[0, 2])
    figure.suptitle(_TITLE, x=0.07, y=0.965, ha="left", fontsize=19, fontweight="bold")
    figure.text(
        0.07,
        0.895,
        "The original result is retained for transparency; the complete replay is a "
        "post-hoc correction.",
        color=_MUTED,
        fontsize=10.5,
    )
    figure.text(
        0.07,
        0.835,
        "Correcting strict tuple/list comparison changed "
        f"{closure.changed_execution_outcome_count} of 48 "
        "recorded execution outcomes.",
        color=_ORANGE,
        fontsize=10.5,
        fontweight="bold",
    )
    figure.text(
        0.07,
        0.775,
        "After correction, adding the passing probe changed only one net prediction; "
        "passing unrelated "
        "evidence produced the same decisions as relevant evidence.",
        color=_INK,
        fontsize=10.5,
        fontweight="bold",
    )
    _plot_accuracy(accuracy_axis, closure)
    _plot_primary_effect(effect_axis, analysis)
    _plot_relevance(relevance_axis, analysis)
    figure.text(
        0.07,
        0.115,
        "Reading the result: the corrected benchmark does not support a probe-based "
        "prediction advantage. The large contrast with deliberately inverted evidence "
        "is descriptive, not proof that the tracker "
        "recognises relevance.",
        color=_INK,
        fontsize=9.4,
        fontweight="bold",
    )
    figure.text(
        0.07,
        0.055,
        "Scope: one pinned evaluation model; 24 authored cases; 23 eligible; one model "
        "run per case. This does not show human learning, tutoring effectiveness, or "
        "reduced dependence on a tutor.",
        color=_MUTED,
        fontsize=9.2,
    )
    return figure


def _plot_accuracy(axis: Axes, closure: HarnessCorrectionClosureReport) -> None:
    x = (0.0, 1.0)
    width = 0.32
    original = (
        closure.original_dialogue_correct_count / closure.eligible_case_count,
        closure.original_probe_correct_count / closure.eligible_case_count,
    )
    corrected = (
        closure.corrected_dialogue_correct_count / closure.eligible_case_count,
        closure.corrected_probe_correct_count / closure.eligible_case_count,
    )
    original_bars = axis.bar(
        [value - width / 2 for value in x],
        original,
        width,
        color=_GREY,
        label="Original",
    )
    corrected_bars = axis.bar(
        [value + width / 2 for value in x],
        corrected,
        width,
        color=_BLUE,
        label="Corrected",
    )
    counts = (
        closure.original_dialogue_correct_count,
        closure.original_probe_correct_count,
        closure.corrected_dialogue_correct_count,
        closure.corrected_probe_correct_count,
    )
    for bar, count in zip((*original_bars, *corrected_bars), counts, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.025,
            f"{count}/23\n({bar.get_height():.0%})",
            ha="center",
            va="bottom",
            fontsize=8.2,
            color=_INK,
        )
    axis.set_title("A. Prediction accuracy changed", loc="left", pad=10)
    axis.set_ylabel("Correct predictions")
    axis.set_xticks(x, ("Dialogue only", "Passing probe"))
    axis.set_ylim(0.0, 1.12)
    axis.legend(frameon=False, loc="upper left")
    _style_axis(axis, grid_axis="y")


def _plot_primary_effect(axis: Axes, report: HarnessCorrectionAnalysisReport) -> None:
    means = (report.original_sealed_primary_effect, report.corrected_primary_effect)
    intervals = (
        report.original_sealed_primary_interval,
        (report.primary_inference.interval_lower, report.primary_inference.interval_upper),
    )
    y = (1.0, 0.0)
    colours = (_GREY, _BLUE)
    for mean, interval, position, colour in zip(means, intervals, y, colours, strict=True):
        axis.errorbar(
            [mean],
            [position],
            xerr=([mean - interval[0]], [interval[1] - mean]),
            fmt="o",
            markersize=8,
            color=colour,
            ecolor=colour,
            elinewidth=2,
            capsize=5,
            zorder=3,
        )
        axis.text(
            mean,
            position - 0.2,
            f"{mean:.3f}  [{interval[0]:.3f}, {interval[1]:.3f}]",
            ha="center",
            va="top",
            fontsize=8.2,
            color=_INK,
        )
    axis.axvline(0.0, color=_INK, linewidth=1)
    axis.axvline(0.1, color=_ORANGE, linewidth=1, linestyle=(0, (3, 3)))
    axis.set_title("B. Estimated advantage fell", loc="left", pad=10)
    axis.set_xlabel("Lower error after adding the probe")
    axis.set_yticks(y, ("Original: 7 net cases", "Corrected: 1 net case"))
    axis.set_xlim(-0.08, 0.62)
    axis.set_ylim(-0.55, 1.5)
    axis.text(
        0.105,
        1.38,
        "Practical threshold 0.10",
        fontsize=7.8,
        color=_ORANGE,
        ha="left",
    )
    _style_axis(axis, grid_axis="x")


def _plot_relevance(axis: Axes, report: HarnessCorrectionAnalysisReport) -> None:
    comparisons = (
        report.valid_vs_unrelated_inference,
        report.valid_vs_corrupted_inference,
    )
    labels = ("Relevant vs unrelated", "Relevant vs inverted")
    colours = (_BLUE, _ORANGE)
    y = (1.0, 0.0)
    for inference, label, colour, position in zip(
        comparisons,
        labels,
        colours,
        y,
        strict=True,
    ):
        mean = inference.mean_case_effect
        axis.errorbar(
            [mean],
            [position],
            xerr=(
                [mean - inference.interval_lower],
                [inference.interval_upper - mean],
            ),
            fmt="s" if label.endswith("unrelated") else "^",
            markersize=8,
            color=colour,
            ecolor=colour,
            elinewidth=2,
            capsize=5,
            zorder=3,
        )
        axis.text(
            mean,
            position - 0.2,
            f"{mean:.3f}  [{inference.interval_lower:.3f}, {inference.interval_upper:.3f}]",
            ha="center",
            va="top",
            fontsize=8.2,
            color=_INK,
        )
    axis.axvline(0.0, color=_INK, linewidth=1)
    axis.set_title("C. Passing evidence was not relevance-sensitive", loc="left", pad=10)
    axis.set_xlabel("Lower error with relevant evidence")
    axis.set_yticks(y, labels)
    axis.set_xlim(-0.08, 1.02)
    axis.set_ylim(-0.55, 1.5)
    _style_axis(axis, grid_axis="x")


def _style_axis(axis: Axes, *, grid_axis: Literal["x", "y"]) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis=grid_axis, color=_LIGHT_GREY, linewidth=0.7, alpha=0.7)
    axis.set_axisbelow(True)


def _validate_sources(
    *,
    analysis_plan: HarnessCorrectionAnalysisPlan,
    analysis: HarnessCorrectionAnalysisReport,
    closure_plan: HarnessCorrectionClosurePlan,
    closure: HarnessCorrectionClosureReport,
    pixi_lock_hash: Sha256,
) -> None:
    if analysis.analysis_plan_hash != analysis_plan.plan_hash:
        raise HarnessCorrectionFigureError("Correction analysis report and plan differ")
    if closure.closure_plan_hash != closure_plan.plan_hash:
        raise HarnessCorrectionFigureError("Correction closure report and plan differ")
    if closure_plan.correction_analysis_plan_hash != analysis_plan.plan_hash:
        raise HarnessCorrectionFigureError("Closure used another correction analysis plan")
    if closure_plan.correction_analysis_report_hash != analysis.report_hash:
        raise HarnessCorrectionFigureError("Closure used another correction analysis report")
    if analysis_plan.pixi_lock_sha256 != pixi_lock_hash:
        raise HarnessCorrectionFigureError("Correction analysis used another Pixi lock")
    if closure_plan.pixi_lock_sha256 != pixi_lock_hash:
        raise HarnessCorrectionFigureError("Correction closure used another Pixi lock")
    expected = (
        analysis.original_sealed_primary_effect,
        analysis.corrected_primary_effect,
        analysis.primary_inference.interval_lower,
        analysis.primary_inference.interval_upper,
        analysis.valid_vs_unrelated_inference.mean_case_effect,
        analysis.valid_vs_corrupted_inference.mean_case_effect,
    )
    observed = (
        closure.original_primary_effect,
        closure.corrected_primary_effect,
        closure.corrected_primary_interval[0],
        closure.corrected_primary_interval[1],
        closure.corrected_valid_vs_unrelated_effect,
        closure.corrected_valid_vs_corrupted_effect,
    )
    if observed != expected:
        raise HarnessCorrectionFigureError("Correction closure values differ from the analysis")
    claims = {claim.claim_id: claim.status for claim in closure.claims}
    if claims.get("probe_prediction_advantage") != "not_supported_after_complete_correction":
        raise HarnessCorrectionFigureError("Closure does not support the plotted primary wording")
    if claims.get("relevance_specific_advantage") != "not_supported_after_complete_correction":
        raise HarnessCorrectionFigureError("Closure does not support the plotted relevance wording")


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
            HarnessCorrectionFigureManifest,
            "correction figure manifest",
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
        raise ValueError("Correction figure timestamp must be timezone-aware UTC")
