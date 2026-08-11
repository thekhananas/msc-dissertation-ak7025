"""Publish corrected benchmark quality beside its observed execution burden."""

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
from socratic_tutor.benchmark.harness_correction_analysis import (
    HarnessCorrectionAnalysisPlan,
    HarnessCorrectionAnalysisReport,
)
from socratic_tutor.benchmark.harness_correction_closure import (
    HarnessCorrectionClosurePlan,
    HarnessCorrectionClosureReport,
)
from socratic_tutor.benchmark.harness_correction_figure import (
    validate_harness_correction_sources,
)
from socratic_tutor.benchmark.hashing import file_sha256, model_content_hash
from socratic_tutor.benchmark.resource_figure import validate_resource_sources
from socratic_tutor.benchmark.resource_reconciliation import (
    ResourceReconciliationPlan,
    ResourceReconciliationReport,
)
from socratic_tutor.contracts import ContractModel
from socratic_tutor.publication.figure_style import (
    BLUE,
    GREEN,
    GREY,
    INK,
    LIGHT_GREY,
    MUTED,
    ORANGE,
)

_BLUE = BLUE
_ORANGE = ORANGE
_GREEN = GREEN
_GREY = GREY
_LIGHT_GREY = LIGHT_GREY
_INK = INK
_MUTED = MUTED
_TITLE = "The corrected probe changed one decision and required more work"
_STYLE: dict[str, object] = {
    "axes.edgecolor": _LIGHT_GREY,
    "axes.labelcolor": _INK,
    "axes.linewidth": 0.8,
    "axes.titlecolor": _INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "figure.facecolor": "#FAFAF7",
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans", "sans-serif"],
    "font.size": 11,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "svg.fonttype": "none",
    "svg.hashsalt": "socratic-tutor-corrected-resource-v1",
    "text.color": _INK,
    "xtick.color": _MUTED,
    "ytick.color": _MUTED,
}


class CorrectedResourceFigureError(ValueError):
    """Correction and workload sources do not describe the same run."""


class CorrectedResourceFigureManifest(ContractModel):
    """Lineage and checksums for the corrected quality-and-work figure."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.corrected_resource_figure_manifest.v1"] = (
        "benchmark.corrected_resource_figure_manifest.v1"
    )
    benchmark_version: Literal["v1"] = "v1"
    run_id: str = Field(min_length=1)
    resource_plan_hash: Sha256
    resource_report_hash: Sha256
    correction_analysis_plan_hash: Sha256
    correction_analysis_report_hash: Sha256
    correction_closure_plan_hash: Sha256
    correction_closure_report_hash: Sha256
    publication_code_revision: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    pixi_lock_sha256: Sha256
    pdf_sha256: Sha256
    svg_sha256: Sha256
    figure_data_csv_sha256: Sha256
    generated_at_utc: datetime
    corrected_result_is_post_hoc: Literal[True] = True
    provider_cost_available: Literal[False] = False
    deployment_cost_claim_supported: Literal[False] = False
    human_learning_claim_supported: Literal[False] = False
    manifest_hash: Sha256

    @model_validator(mode="after")
    def validate_manifest(self) -> CorrectedResourceFigureManifest:
        _require_utc(self.generated_at_utc)
        if self.manifest_hash != model_content_hash(self, exclude={"manifest_hash"}):
            raise ValueError("Corrected resource-figure manifest hash does not match")
        return self


def render_corrected_resource_figure(
    *,
    resource_plan_path: Path,
    resource_report_path: Path,
    correction_analysis_plan_path: Path,
    correction_analysis_report_path: Path,
    correction_closure_plan_path: Path,
    correction_closure_report_path: Path,
    pixi_lock_path: Path,
    output_root: Path,
    publication_code_revision: str,
    generated_at_utc: datetime | None = None,
) -> CorrectedResourceFigureManifest:
    """Render corrected quality and measured workload without rerunning either study."""

    resource_plan = _load(
        resource_plan_path,
        ResourceReconciliationPlan,
        "resource plan",
    )
    resource = _load(
        resource_report_path,
        ResourceReconciliationReport,
        "resource report",
    )
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
        raise CorrectedResourceFigureError(f"Could not read Pixi lock: {pixi_lock_path}") from error

    validate_resource_sources(
        plan=resource_plan,
        report=resource,
        pixi_lock_hash=lock_hash,
    )
    validate_harness_correction_sources(
        analysis_plan=analysis_plan,
        analysis=analysis,
        closure_plan=closure_plan,
        closure=closure,
        pixi_lock_hash=lock_hash,
    )
    _validate_join(resource=resource, analysis=analysis, closure=closure)
    generated_at = _resolve_generated_at(
        generated_at_utc,
        output_root / "corrected_resource_figure_manifest.json",
    )
    data = _figure_data_csv(resource, analysis, closure)
    pdf, svg = _render_vector_files(resource, closure, generated_at)
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.corrected_resource_figure_manifest.v1",
        "benchmark_version": "v1",
        "run_id": closure.run_id,
        "resource_plan_hash": resource_plan.plan_hash,
        "resource_report_hash": resource.report_hash,
        "correction_analysis_plan_hash": analysis_plan.plan_hash,
        "correction_analysis_report_hash": analysis.report_hash,
        "correction_closure_plan_hash": closure_plan.plan_hash,
        "correction_closure_report_hash": closure.report_hash,
        "publication_code_revision": publication_code_revision,
        "pixi_lock_sha256": lock_hash,
        "pdf_sha256": file_sha256(pdf),
        "svg_sha256": file_sha256(svg),
        "figure_data_csv_sha256": file_sha256(data),
        "generated_at_utc": generated_at,
        "corrected_result_is_post_hoc": True,
        "provider_cost_available": False,
        "deployment_cost_claim_supported": False,
        "human_learning_claim_supported": False,
    }
    draft = CorrectedResourceFigureManifest.model_construct(
        _fields_set=set(content),
        **content,
        manifest_hash="0" * 64,
    )
    manifest = CorrectedResourceFigureManifest.model_validate(
        {**content, "manifest_hash": model_content_hash(draft, exclude={"manifest_hash"})}
    )
    write_immutable_bytes(output_root / "corrected_quality_and_resource_burden.pdf", pdf)
    write_immutable_bytes(output_root / "corrected_quality_and_resource_burden.svg", svg)
    write_immutable_bytes(output_root / "corrected_resource_figure_data.csv", data)
    write_immutable_json(output_root / "corrected_resource_figure_manifest.json", manifest)
    return manifest


def _validate_join(
    *,
    resource: ResourceReconciliationReport,
    analysis: HarnessCorrectionAnalysisReport,
    closure: HarnessCorrectionClosureReport,
) -> None:
    if resource.run_id != closure.run_id:
        raise CorrectedResourceFigureError(
            "Resource and correction reports belong to different runs"
        )
    if closure.corrected_dialogue_correct_count != 18:
        raise CorrectedResourceFigureError("Corrected dialogue count is not the reclosed result")
    if closure.corrected_probe_correct_count != 19:
        raise CorrectedResourceFigureError("Corrected probe count is not the reclosed result")
    if closure.eligible_case_count != 23 or analysis.improvement_count != 1:
        raise CorrectedResourceFigureError("Correction closure has the wrong case-level result")
    if analysis.regression_count != 0:
        raise CorrectedResourceFigureError("Correction closure contains an unexpected regression")


def _figure_data_csv(
    resource: ResourceReconciliationReport,
    analysis: HarnessCorrectionAnalysisReport,
    closure: HarnessCorrectionClosureReport,
) -> bytes:
    total_tokens = resource.total_input_tokens + resource.total_output_tokens
    rows = [
        (
            "corrected_quality",
            "dialogue_only_correct",
            closure.corrected_dialogue_correct_count,
            "cases",
        ),
        (
            "corrected_quality",
            "probe_informed_correct",
            closure.corrected_probe_correct_count,
            "cases",
        ),
        ("corrected_quality", "eligible", closure.eligible_case_count, "cases"),
        (
            "corrected_quality",
            "net_changed_decisions",
            analysis.improvement_count - analysis.regression_count,
            "cases",
        ),
        ("per_prediction", "dialogue_model_responses", 1, "responses"),
        ("per_prediction", "dialogue_sandbox_executions", 0, "executions"),
        ("per_prediction", "probe_model_responses", 2, "responses"),
        ("per_prediction", "probe_sandbox_executions", 1, "executions"),
        ("full_run", "model_responses", resource.total_provider_response_count, "responses"),
        ("full_run", "total_tokens", total_tokens, "tokens"),
        (
            "full_run",
            "summed_provider_latency_ms",
            resource.total_summed_provider_latency_ms,
            "milliseconds",
        ),
        ("full_run", "sandbox_attempts", resource.sandbox.total_execution_count, "attempts"),
        (
            "full_run",
            "completed_sandbox_attempts",
            resource.sandbox.total_completed_count,
            "attempts",
        ),
        (
            "full_run",
            "measured_artifact_bytes",
            resource.measured_artifact_bytes_excluding_this_report,
            "bytes",
        ),
        ("unavailable", "provider_cost", "", "not_reported"),
        ("unavailable", "sandbox_time_cpu_memory", "", "not_recorded"),
    ]
    return _csv_bytes(("scope", "metric", "value", "unit_or_status"), rows)


def _render_vector_files(
    resource: ResourceReconciliationReport,
    closure: HarnessCorrectionClosureReport,
    generated_at_utc: datetime,
) -> tuple[bytes, bytes]:
    with mpl.rc_context(_STYLE):
        figure = _build_figure(resource, closure)
        pdf_buffer = BytesIO()
        svg_buffer = BytesIO()
        figure.savefig(
            pdf_buffer,
            format="pdf",
            metadata={
                "Title": _TITLE,
                "Author": "Anas Khan",
                "Subject": "Post-hoc corrected quality and observed execution burden",
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
                    "Corrected prediction outcomes beside the model and sandbox work used "
                    "by the external benchmark."
                ),
                "Creator": "Socratic Tutor reproducible figure pipeline",
                "Date": generated_at_utc.isoformat(),
            },
        )
        figure.clear()
    return pdf_buffer.getvalue(), svg_buffer.getvalue()


def _build_figure(
    resource: ResourceReconciliationReport,
    closure: HarnessCorrectionClosureReport,
) -> Figure:
    figure = Figure(figsize=(12.8, 7.6), facecolor="#FAFAF7")
    grid = figure.add_gridspec(
        1,
        3,
        left=0.07,
        right=0.98,
        top=0.69,
        bottom=0.20,
        width_ratios=(1.05, 1.05, 1.2),
        wspace=0.42,
    )
    quality_axis = figure.add_subplot(grid[0, 0])
    burden_axis = figure.add_subplot(grid[0, 1])
    totals_axis = figure.add_subplot(grid[0, 2])
    _plot_corrected_quality(quality_axis, closure)
    _plot_per_prediction_work(burden_axis)
    _plot_run_totals(totals_axis, resource)

    figure.text(0.07, 0.95, _TITLE, fontsize=22, fontweight="bold", color=_INK)
    figure.text(
        0.07,
        0.895,
        "The recorded responses were checked again after repairing a test-harness error; no "
        "model or sandbox call was repeated.",
        fontsize=12,
        color=_MUTED,
    )
    figure.text(
        0.07,
        0.81,
        "The relevant probe changed one of 23 eligible decisions from wrong to right.",
        fontsize=15,
        fontweight="bold",
        color=_GREEN,
    )
    figure.text(
        0.07,
        0.765,
        "The probe condition used one extra model response and one code execution for each case.",
        fontsize=11.5,
        color=_INK,
    )
    figure.text(
        0.07,
        0.115,
        "Post-hoc result: the correction shows that the original result depended on the test "
        "harness; it does not replace the sealed analysis.",
        fontsize=10.5,
        fontweight="bold",
        color=_ORANGE,
    )
    figure.text(
        0.07,
        0.065,
        "Scope: one pinned evaluation model, one run per authored case. Provider cost and "
        "sandbox CPU, memory, and execution time were not recorded.",
        fontsize=10,
        color=_MUTED,
    )
    figure.text(
        0.07,
        0.025,
        "These measurements describe this experiment's workload; they do not show deployment "
        "cost, learning, or tutoring effectiveness.",
        fontsize=10,
        fontweight="bold",
        color=_MUTED,
    )
    return figure


def _plot_corrected_quality(axis: Axes, closure: HarnessCorrectionClosureReport) -> None:
    counts = (closure.corrected_dialogue_correct_count, closure.corrected_probe_correct_count)
    labels = ("Dialogue only", "Dialogue + relevant probe")
    bars = axis.barh((0, 1), counts, color=(_GREY, _BLUE), height=0.55)
    for bar, count in zip(bars, counts, strict=True):
        axis.text(
            count - 0.35,
            bar.get_y() + bar.get_height() / 2,
            f"{count}/{closure.eligible_case_count}",
            ha="right",
            va="center",
            fontsize=10,
            color="white",
            fontweight="bold",
        )
    axis.set_title("A. Corrected predictions", loc="left", pad=10)
    axis.set_xlabel("Cases predicted correctly")
    axis.set_yticks((0, 1), labels, fontsize=8.5)
    axis.set_xlim(0, 24.5)
    axis.set_xticks((0, 6, 12, 18, 23))
    axis.invert_yaxis()
    _style_axis(axis)


def _plot_per_prediction_work(axis: Axes) -> None:
    labels = ("Dialogue only", "With relevant probe")
    positions = (0, 1)
    height = 0.3
    responses = axis.barh(
        (positions[0] - height / 1.8, positions[1] - height / 1.8),
        (1, 2),
        height=height,
        color=_BLUE,
        label="Model responses",
    )
    executions = axis.barh(
        (positions[0] + height / 1.8, positions[1] + height / 1.8),
        (0, 1),
        height=height,
        color=_GREEN,
        label="Code executions",
    )
    for bars in (responses, executions):
        for bar in bars:
            axis.text(
                bar.get_width() + 0.06,
                bar.get_y() + bar.get_height() / 2,
                f"{int(bar.get_width())}",
                va="center",
                fontsize=9,
                color=_INK,
            )
    axis.set_title("B. Work used for each prediction", loc="left", pad=10)
    axis.set_xlabel("Count per case")
    axis.set_yticks(positions, labels, fontsize=8.5)
    axis.set_xlim(0, 2.6)
    axis.set_xticks((0, 1, 2))
    axis.invert_yaxis()
    axis.legend(loc="lower right", frameon=False, fontsize=8)
    _style_axis(axis)


def _plot_run_totals(axis: Axes, resource: ResourceReconciliationReport) -> None:
    axis.set_title("C. Work observed in the full run", loc="left", pad=10)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    total_tokens = resource.total_input_tokens + resource.total_output_tokens
    values = (
        (f"{resource.total_provider_response_count}", "model responses"),
        (f"{total_tokens:,}", "input + output tokens"),
        (f"{resource.total_summed_provider_latency_ms / 1000:.1f} s", "summed request latency"),
        (
            f"{resource.sandbox.total_completed_count}/{resource.sandbox.total_execution_count}",
            "completed sandbox attempts",
        ),
        (
            f"{resource.measured_artifact_bytes_excluding_this_report / 1_000_000:.2f} MB",
            "measured artefacts",
        ),
        ("Not reported", "provider cost"),
    )
    for (value, label), y in zip(values, (0.88, 0.72, 0.56, 0.40, 0.24, 0.08), strict=True):
        axis.text(
            0.0,
            y,
            value,
            fontsize=13,
            fontweight="bold",
            color=_ORANGE if value == "Not reported" else _INK,
        )
        axis.text(0.0, y - 0.06, label, fontsize=8.5, color=_MUTED)
        if y > 0.1:
            axis.axhline(y - 0.095, color=_LIGHT_GREY, linewidth=0.7)


def _style_axis(axis: Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="x", color=_LIGHT_GREY, linewidth=0.7, alpha=0.7)
    axis.set_axisbelow(True)


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
            CorrectedResourceFigureManifest,
            "corrected resource figure manifest",
        )
        return existing.generated_at_utc
    return datetime.now(UTC)


def _load[ModelT: ContractModel](path: Path, model: type[ModelT], name: str) -> ModelT:
    try:
        return model.model_validate_json(path.read_bytes())
    except (OSError, ValidationError) as error:
        raise CorrectedResourceFigureError(f"Could not verify {name}: {path}") from error


def _require_utc(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("Corrected resource-figure timestamp must be timezone-aware UTC")
