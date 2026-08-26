"""Reproducible quality-and-workload figure for the external benchmark."""

# pyright: reportArgumentType=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Literal

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pydantic import Field, ValidationError, model_validator

from socratic_tutor.benchmark.artifacts import write_immutable_bytes, write_immutable_json
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import file_sha256, model_content_hash
from socratic_tutor.benchmark.resource_reconciliation import (
    ConditionBurdenSummary,
    ResourceReconciliationPlan,
    ResourceReconciliationReport,
)
from socratic_tutor.contracts import ContractModel
from socratic_tutor.csv_output import csv_bytes

_BLUE = "#0072B2"
_ORANGE = "#D55E00"
_GREEN = "#009E73"
_GREY = "#8A949E"
_LIGHT_GREY = "#D7DDE2"
_PALE_GREY = "#F2F4F5"
_INK = "#17212B"
_MUTED = "#56616B"
_TITLE = "What did executable evidence add, and what work did it require?"
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
    "svg.hashsalt": "socratic-tutor-resource-result-v1",
    "text.color": _INK,
    "xtick.color": _MUTED,
    "ytick.color": _MUTED,
}


class ResourceFigureError(ValueError):
    """Resource-figure sources are missing, invalid, or inconsistent."""


class ResourceFigureManifest(ContractModel):
    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.resource_figure_manifest.v1"] = (
        "benchmark.resource_figure_manifest.v1"
    )
    benchmark_version: Literal["v1"] = "v1"
    run_id: str = Field(min_length=1)
    resource_plan_hash: Sha256
    resource_report_hash: Sha256
    publication_code_revision: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    pixi_lock_sha256: Sha256
    panel_count: Literal[3] = 3
    figure_formats: tuple[Literal["pdf", "svg"], Literal["pdf", "svg"]]
    pdf_sha256: Sha256
    svg_sha256: Sha256
    figure_data_csv_sha256: Sha256
    generated_at_utc: datetime
    post_hoc_result_used_as_primary: Literal[False] = False
    deployment_cost_claim_supported: Literal[False] = False
    human_learning_claim_supported: Literal[False] = False
    tutoring_efficacy_claim_supported: Literal[False] = False
    manifest_hash: Sha256

    @model_validator(mode="after")
    def validate_manifest(self) -> ResourceFigureManifest:
        _require_utc(self.generated_at_utc)
        if self.figure_formats != ("pdf", "svg"):
            raise ValueError("Resource figure must publish PDF and SVG")
        if self.manifest_hash != model_content_hash(self, exclude={"manifest_hash"}):
            raise ValueError("Resource-figure manifest hash does not match its content")
        return self


def render_resource_figure(
    *,
    resource_plan_path: Path,
    resource_report_path: Path,
    pixi_lock_path: Path,
    output_root: Path,
    publication_code_revision: str,
    generated_at_utc: datetime | None = None,
) -> ResourceFigureManifest:
    """Render observed predictive quality beside the machinery used to obtain it."""

    plan = _load(resource_plan_path, ResourceReconciliationPlan, "resource plan")
    report = _load(resource_report_path, ResourceReconciliationReport, "resource report")
    try:
        pixi_lock_hash = file_sha256(pixi_lock_path.read_bytes())
    except OSError as error:
        raise ResourceFigureError(f"Could not read Pixi lock: {pixi_lock_path}") from error
    validate_resource_sources(plan=plan, report=report, pixi_lock_hash=pixi_lock_hash)
    generated_at = _resolve_generated_at(
        generated_at_utc,
        output_root / "resource_figure_manifest.json",
    )
    figure_data = _figure_data_csv(report)
    pdf, svg = _render_vector_files(report, generated_at)
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.resource_figure_manifest.v1",
        "benchmark_version": "v1",
        "run_id": report.run_id,
        "resource_plan_hash": plan.plan_hash,
        "resource_report_hash": report.report_hash,
        "publication_code_revision": publication_code_revision,
        "pixi_lock_sha256": pixi_lock_hash,
        "panel_count": 3,
        "figure_formats": ("pdf", "svg"),
        "pdf_sha256": file_sha256(pdf),
        "svg_sha256": file_sha256(svg),
        "figure_data_csv_sha256": file_sha256(figure_data),
        "generated_at_utc": generated_at,
        "post_hoc_result_used_as_primary": False,
        "deployment_cost_claim_supported": False,
        "human_learning_claim_supported": False,
        "tutoring_efficacy_claim_supported": False,
    }
    draft = ResourceFigureManifest.model_construct(
        _fields_set=set(content),
        **content,
        manifest_hash="0" * 64,
    )
    manifest = ResourceFigureManifest.model_validate(
        {**content, "manifest_hash": model_content_hash(draft, exclude={"manifest_hash"})}
    )
    write_immutable_bytes(output_root / "quality_and_resource_burden.pdf", pdf)
    write_immutable_bytes(output_root / "quality_and_resource_burden.svg", svg)
    write_immutable_bytes(output_root / "resource_figure_data.csv", figure_data)
    write_immutable_json(output_root / "resource_figure_manifest.json", manifest)
    return manifest


def _figure_data_csv(report: ResourceReconciliationReport) -> bytes:
    rows: list[tuple[object, ...]] = []
    for condition in report.condition_burden:
        rows.extend(
            (
                (
                    "condition",
                    condition.condition,
                    "correct_cases",
                    condition.correct_case_count,
                    "cases",
                ),
                (
                    "condition",
                    condition.condition,
                    "eligible_cases",
                    condition.eligible_case_count,
                    "cases",
                ),
                (
                    "condition",
                    condition.condition,
                    "accuracy",
                    condition.accuracy_on_fixed_authored_cases,
                    "proportion",
                ),
                (
                    "condition",
                    condition.condition,
                    "external_response_channels_per_case",
                    condition.external_model_response_channels_per_case,
                    "channels_per_case",
                ),
                (
                    "condition",
                    condition.condition,
                    "sandbox_executions_per_case",
                    condition.sandbox_executions_per_case,
                    "executions_per_case",
                ),
            )
        )
    total_tokens = report.total_input_tokens + report.total_output_tokens
    rows.extend(
        (
            (
                "full_run",
                "provider",
                "responses",
                report.total_provider_response_count,
                "responses",
            ),
            ("full_run", "provider", "input_tokens", report.total_input_tokens, "tokens"),
            ("full_run", "provider", "output_tokens", report.total_output_tokens, "tokens"),
            ("full_run", "provider", "total_tokens", total_tokens, "tokens"),
            (
                "full_run",
                "provider",
                "summed_provider_latency_ms",
                report.total_summed_provider_latency_ms,
                "milliseconds",
            ),
            (
                "full_run",
                "sandbox",
                "execution_attempts",
                report.sandbox.total_execution_count,
                "attempts",
            ),
            (
                "full_run",
                "sandbox",
                "completed_executions",
                report.sandbox.total_completed_count,
                "executions",
            ),
            (
                "full_run",
                "artifacts",
                "measured_bytes",
                report.measured_artifact_bytes_excluding_this_report,
                "bytes",
            ),
            ("unavailable", "provider", "cost_usd", "", "not_reported"),
            ("unavailable", "sandbox", "execution_time", "", "not_recorded"),
            ("unavailable", "sandbox", "cpu_and_memory", "", "not_recorded"),
        )
    )
    return csv_bytes(("scope", "group", "metric", "value", "unit_or_status"), rows)


def _render_vector_files(
    report: ResourceReconciliationReport,
    generated_at_utc: datetime,
) -> tuple[bytes, bytes]:
    with mpl.rc_context(_STYLE):
        figure = _build_figure(report)
        pdf_buffer = BytesIO()
        svg_buffer = BytesIO()
        figure.savefig(
            pdf_buffer,
            format="pdf",
            metadata={
                "Title": _TITLE,
                "Author": "Anas Khan",
                "Subject": "Observed benchmark quality and resource burden",
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
                    "Observed accuracy, per-case machinery, and full-run workload for the "
                    "external benchmark."
                ),
                "Creator": "Socratic Tutor reproducible figure pipeline",
                "Date": generated_at_utc.isoformat(),
            },
        )
        figure.clear()
    return pdf_buffer.getvalue(), svg_buffer.getvalue()


def _build_figure(report: ResourceReconciliationReport) -> Figure:
    figure = Figure(figsize=(13.4, 8.8))
    grid = figure.add_gridspec(
        1,
        3,
        left=0.07,
        right=0.98,
        top=0.69,
        bottom=0.22,
        width_ratios=(1.25, 1.05, 1.2),
        wspace=0.42,
    )
    accuracy_axis = figure.add_subplot(grid[0, 0])
    burden_axis = figure.add_subplot(grid[0, 1])
    totals_axis = figure.add_subplot(grid[0, 2])

    _plot_accuracy(accuracy_axis, report.condition_burden)
    _plot_per_case_burden(burden_axis, report.condition_burden)
    _plot_full_run_totals(totals_axis, report)

    primary = _condition(report, "dialogue_only")
    informed = _condition(report, "probe_informed_observed")
    figure.text(0.07, 0.955, _TITLE, fontsize=19, fontweight="bold", color=_INK)
    figure.text(
        0.07,
        0.91,
        "The evaluation model predicted later task success from dialogue alone, then from "
        "dialogue plus a separate executable probe.",
        fontsize=10.5,
        color=_MUTED,
    )
    figure.text(
        0.07,
        0.835,
        f"Observed accuracy rose from {primary.correct_case_count}/{primary.eligible_case_count} "
        f"to {informed.correct_case_count}/{informed.eligible_case_count} fixed cases.",
        fontsize=13,
        fontweight="bold",
        color=_GREEN,
    )
    figure.text(
        0.07,
        0.792,
        "That comparison required one extra model response and one sandbox execution per case. "
        "It is a measured trade-off, not a deployment estimate.",
        fontsize=10.2,
        color=_INK,
    )
    figure.text(
        0.07,
        0.145,
        "The corrected 23/23 result followed reviewer-identified test defects. It is shown for "
        "transparency, but it does not replace the sealed 20/23 result.",
        fontsize=9.2,
        fontweight="bold",
        color=_ORANGE,
    )
    figure.text(
        0.07,
        0.096,
        "Not measured: provider cost, sandbox duration, CPU use, or memory use. Summed provider "
        "latency is the total reported request latency, not elapsed wall-clock time.",
        fontsize=8.8,
        color=_MUTED,
    )
    figure.text(
        0.07,
        0.047,
        "Scope: one pinned evaluation model and one fixed authored corpus. These measurements do "
        "not show deployment cost, human learning, or tutoring efficacy.",
        fontsize=8.8,
        fontweight="bold",
        color=_MUTED,
    )
    return figure


def _plot_accuracy(
    axis: Axes,
    conditions: tuple[ConditionBurdenSummary, ConditionBurdenSummary, ConditionBurdenSummary],
) -> None:
    ordered = (
        _condition_from(conditions, "dialogue_only"),
        _condition_from(conditions, "probe_informed_observed"),
        _condition_from(conditions, "probe_informed_post_hoc_corrected"),
    )
    labels = ("Dialogue only", "Dialogue + probe", "Corrected after review")
    colours = (_GREY, _BLUE, "white")
    bars = axis.barh(
        range(3),
        [item.accuracy_on_fixed_authored_cases for item in ordered],
        color=colours,
        edgecolor=(_GREY, _BLUE, _ORANGE),
        linewidth=1.5,
        height=0.58,
    )
    bars[2].set_hatch("///")
    for bar, item in zip(bars, ordered, strict=True):
        axis.text(
            min(1.0, item.accuracy_on_fixed_authored_cases) + 0.025,
            bar.get_y() + bar.get_height() / 2,
            f"{item.correct_case_count}/{item.eligible_case_count}",
            va="center",
            fontsize=9,
            color=_INK,
        )
    axis.set_title("A. Accuracy on the fixed cases", loc="left", pad=10)
    axis.set_xlabel("Proportion predicted correctly")
    axis.set_yticks(range(3), labels, fontsize=8.5)
    axis.set_xlim(0, 1.18)
    axis.set_xticks((0, 0.25, 0.5, 0.75, 1.0))
    axis.invert_yaxis()
    _style_axis(axis)


def _plot_per_case_burden(
    axis: Axes,
    conditions: tuple[ConditionBurdenSummary, ConditionBurdenSummary, ConditionBurdenSummary],
) -> None:
    primary = _condition_from(conditions, "dialogue_only")
    informed = _condition_from(conditions, "probe_informed_observed")
    labels = ("Dialogue only", "Dialogue + probe")
    y_positions = (0, 1)
    channels = (
        primary.external_model_response_channels_per_case,
        informed.external_model_response_channels_per_case,
    )
    executions = (primary.sandbox_executions_per_case, informed.sandbox_executions_per_case)
    height = 0.3
    channel_bars = axis.barh(
        [value - height / 1.8 for value in y_positions],
        channels,
        height=height,
        color=_BLUE,
        label="Model responses",
    )
    execution_bars = axis.barh(
        [value + height / 1.8 for value in y_positions],
        executions,
        height=height,
        color=_GREEN,
        label="Code executions",
    )
    for bars in (channel_bars, execution_bars):
        for bar in bars:
            axis.text(
                bar.get_width() + 0.05,
                bar.get_y() + bar.get_height() / 2,
                f"{int(bar.get_width())}",
                va="center",
                fontsize=9,
                color=_INK,
            )
    axis.set_title("B. Inputs used for each prediction", loc="left", pad=10)
    axis.set_xlabel("Count per case")
    axis.set_yticks(y_positions, labels, fontsize=8.5)
    axis.set_xlim(0, 2.55)
    axis.set_xticks((0, 1, 2))
    axis.invert_yaxis()
    axis.legend(loc="lower right", frameon=False, fontsize=8)
    _style_axis(axis)


def _plot_full_run_totals(axis: Axes, report: ResourceReconciliationReport) -> None:
    axis.set_title("C. Work observed across the full run", loc="left", pad=10)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    total_tokens = report.total_input_tokens + report.total_output_tokens
    decision_responses = next(
        phase.response_count
        for phase in report.provider_phases
        if phase.phase == "decision_generation"
    )
    criterion_responses = next(
        phase.response_count
        for phase in report.provider_phases
        if phase.phase == "criterion_generation"
    )
    values = (
        (
            f"{report.total_provider_response_count}",
            f"model responses ({decision_responses} inputs + {criterion_responses} outcomes)",
        ),
        (f"{total_tokens:,}", "input + output tokens"),
        (f"{report.total_summed_provider_latency_ms / 1000:.1f} s", "summed provider latency"),
        (
            f"{report.sandbox.total_completed_count}/{report.sandbox.total_execution_count}",
            "completed sandbox attempts",
        ),
        (
            f"{report.measured_artifact_bytes_excluding_this_report / 1_000_000:.2f} MB",
            "measured artefacts",
        ),
        ("Not reported", "provider cost"),
    )
    y_positions = (0.88, 0.72, 0.56, 0.40, 0.24, 0.08)
    for (value, label), y_position in zip(values, y_positions, strict=True):
        colour = _ORANGE if value == "Not reported" else _INK
        axis.text(0.0, y_position, value, fontsize=13, fontweight="bold", color=colour)
        axis.text(0.0, y_position - 0.06, label, fontsize=8.5, color=_MUTED)
        if y_position > 0.1:
            axis.axhline(y_position - 0.095, color=_LIGHT_GREY, linewidth=0.7)


def _condition(report: ResourceReconciliationReport, name: str) -> ConditionBurdenSummary:
    return _condition_from(report.condition_burden, name)


def _condition_from(
    conditions: tuple[ConditionBurdenSummary, ConditionBurdenSummary, ConditionBurdenSummary],
    name: str,
) -> ConditionBurdenSummary:
    try:
        return next(item for item in conditions if item.condition == name)
    except StopIteration as error:
        raise ResourceFigureError(f"Resource report is missing condition: {name}") from error


def validate_resource_sources(
    *,
    plan: ResourceReconciliationPlan,
    report: ResourceReconciliationReport,
    pixi_lock_hash: Sha256,
) -> None:
    if report.analysis_plan_hash != plan.plan_hash:
        raise ResourceFigureError("Resource report belongs to another resource plan")
    if plan.analysis_pixi_lock_hash != pixi_lock_hash:
        raise ResourceFigureError("Resource reconciliation used another Pixi lock")
    if plan.run_id != report.run_id:
        raise ResourceFigureError("Resource plan and report belong to different runs")
    if report.interpretation_scope != "observed_single_run_workload_not_deployment_cost":
        raise ResourceFigureError("Resource report permits an unsupported interpretation")
    primary = _condition(report, "dialogue_only")
    informed = _condition(report, "probe_informed_observed")
    corrected = _condition(report, "probe_informed_post_hoc_corrected")
    if primary.evidence_status != "sealed_primary" or informed.evidence_status != "sealed_primary":
        raise ResourceFigureError("Primary resource conditions must remain sealed")
    if corrected.evidence_status != "post_hoc_correction":
        raise ResourceFigureError("Corrected resource condition must remain post-hoc")


def _style_axis(axis: Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="x", color=_LIGHT_GREY, linewidth=0.7, alpha=0.7)
    axis.set_axisbelow(True)


def _resolve_generated_at(value: datetime | None, manifest_path: Path) -> datetime:
    if value is not None:
        _require_utc(value)
        return value
    if manifest_path.exists():
        existing = _load(manifest_path, ResourceFigureManifest, "resource figure manifest")
        return existing.generated_at_utc
    return datetime.now(UTC)


def _require_utc(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("Resource-figure timestamp must be timezone-aware UTC")


def _load[ModelT: ContractModel](path: Path, model: type[ModelT], label: str) -> ModelT:
    try:
        return model.model_validate_json(path.read_bytes())
    except (OSError, ValidationError) as error:
        raise ResourceFigureError(f"Could not verify {label}: {path}") from error
