"""Reproducible vector figures for dependence and heuristic sensitivity."""

# pyright: reportArgumentType=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Literal

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from pydantic import ValidationError, model_validator

from socratic_tutor.benchmark.artifacts import write_immutable_bytes, write_immutable_json
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.dependence_sensitivity import (
    DependenceSensitivityReport,
    HeuristicSensitivityPoint,
)
from socratic_tutor.benchmark.hashing import file_sha256, model_content_hash
from socratic_tutor.contracts import ContractModel

_BLUE = "#0072B2"
_ORANGE = "#D55E00"
_INK = "#17212B"
_MUTED = "#56616B"
_GRID = "#D7DDE2"
_FIGURE_TITLE = "Sensitivity of probe-informed performance prediction"
_SVG_METADATA = {
    "Title": _FIGURE_TITLE,
    "Description": (
        "Comparison of predictions from a public answer, an executable diagnostic task, "
        "and both sources together, across programming concepts and tracker settings."
    ),
    "Creator": "Socratic Tutor reproducible figure pipeline",
}
_STYLE: dict[str, object] = {
    "axes.edgecolor": _GRID,
    "axes.labelcolor": _INK,
    "axes.linewidth": 0.8,
    "axes.titlecolor": _INK,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "figure.facecolor": "white",
    "font.family": ["DejaVu Sans", "sans-serif"],
    "font.size": 10,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "svg.fonttype": "none",
    "svg.hashsalt": "socratic-tutor-publication-v1",
    "text.color": _INK,
    "xtick.color": _MUTED,
    "ytick.color": _MUTED,
}


class DependenceFigureError(ValueError):
    """A figure source is invalid or incomplete."""


class DependenceFigureManifest(ContractModel):
    """Content addresses for one report-ready figure in two vector formats."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.dependence_figure_manifest.v1"] = (
        "benchmark.dependence_figure_manifest.v1"
    )
    source_report_hash: Sha256
    figure_formats: tuple[Literal["pdf", "svg"], Literal["pdf", "svg"]]
    panel_count: Literal[3] = 3
    pdf_sha256: Sha256
    svg_sha256: Sha256
    generated_at_utc: datetime
    manifest_hash: Sha256

    @model_validator(mode="after")
    def validate_manifest(self) -> DependenceFigureManifest:
        _require_utc(self.generated_at_utc)
        if self.figure_formats != ("pdf", "svg"):
            raise ValueError("Dependence figure must publish PDF and SVG")
        if self.manifest_hash != model_content_hash(self, exclude={"manifest_hash"}):
            raise ValueError("Dependence-figure manifest hash does not match its content")
        return self


def render_dependence_figure(
    *,
    report_path: Path,
    pdf_output_path: Path,
    svg_output_path: Path,
    manifest_path: Path,
    generated_at_utc: datetime | None = None,
) -> DependenceFigureManifest:
    """Render matching PDF and SVG files from an immutable sensitivity report."""

    try:
        report = DependenceSensitivityReport.model_validate_json(report_path.read_bytes())
    except (OSError, ValidationError) as error:
        raise DependenceFigureError(f"Could not verify figure source: {report_path}") from error

    generated_at_utc = _resolve_generated_at_utc(generated_at_utc, manifest_path)
    pdf, svg = _render_vector_files(report, generated_at_utc)
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.dependence_figure_manifest.v1",
        "source_report_hash": report.report_hash,
        "figure_formats": ("pdf", "svg"),
        "panel_count": 3,
        "pdf_sha256": file_sha256(pdf),
        "svg_sha256": file_sha256(svg),
        "generated_at_utc": generated_at_utc,
    }
    draft = DependenceFigureManifest.model_construct(
        _fields_set=set(content), **content, manifest_hash="0" * 64
    )
    manifest = DependenceFigureManifest.model_validate(
        {**content, "manifest_hash": model_content_hash(draft, exclude={"manifest_hash"})}
    )

    write_immutable_bytes(pdf_output_path, pdf)
    write_immutable_bytes(svg_output_path, svg)
    write_immutable_json(manifest_path, manifest)
    return manifest


def _render_vector_files(
    report: DependenceSensitivityReport,
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
                "Title": _FIGURE_TITLE,
                "Author": "Anas Khan",
                "Subject": "Secondary benchmark sensitivity analysis",
                "Creator": "Socratic Tutor reproducible figure pipeline",
                "CreationDate": generated_at_utc,
                "ModDate": generated_at_utc,
            },
        )
        figure.savefig(
            svg_buffer,
            format="svg",
            metadata={**_SVG_METADATA, "Date": generated_at_utc.isoformat()},
        )
        figure.clear()
    return pdf_buffer.getvalue(), svg_buffer.getvalue()


def _build_figure(report: DependenceSensitivityReport) -> Figure:
    figure = Figure(figsize=(12, 7.6))
    axes = tuple(figure.add_subplot(1, 3, index) for index in range(1, 4))
    figure.subplots_adjust(left=0.07, right=0.98, top=0.70, bottom=0.24, wspace=0.32)
    figure.suptitle(_FIGURE_TITLE, x=0.06, y=0.975, ha="left", fontsize=19, fontweight="bold")
    figure.text(
        0.06,
        0.895,
        "Each condition predicts whether the evaluation model completes a separate coding task "
        "correctly.",
        color=_MUTED,
        fontsize=10.5,
    )
    figure.text(
        0.06,
        0.855,
        "Probe-informed prediction combines the public answer with an earlier executable "
        "diagnostic task.",
        color=_MUTED,
        fontsize=10.5,
    )
    figure.text(
        0.06,
        0.815,
        "Values show single-source error minus probe-informed error; positive values favour "
        "combining both sources.",
        color=_MUTED,
        fontsize=10.5,
    )
    _plot_concept_effects(axes[0], report)
    _plot_curve(
        axes[1],
        report.threshold_sensitivity,
        title="Across decision thresholds",
        x_label="Decision threshold",
        x_value=lambda point: point.threshold,
        frozen_value=0.6,
    )
    _plot_curve(
        axes[2],
        report.update_size_sensitivity,
        title="Across executable-evidence weights",
        x_label="Evidence-weight multiplier",
        x_value=lambda point: point.update_multiplier,
        frozen_value=1.0,
    )
    figure.legend(
        handles=(
            Line2D([0], [0], color=_BLUE, marker="o", linewidth=2.5),
            Line2D([0], [0], color=_ORANGE, marker="s", linewidth=2.5),
        ),
        labels=(
            "Line charts: public answer only",
            "Line charts: executable diagnostic task only",
        ),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.115),
        frameon=False,
        ncol=2,
        columnspacing=3,
        fontsize=9.5,
    )
    figure.text(
        0.06,
        0.055,
        "23 of 24 benchmark cases were analysed; one was excluded because its independent task "
        "could not be executed.",
        color=_MUTED,
        fontsize=9.5,
    )
    figure.text(
        0.06,
        0.025,
        "Threshold and evidence-weight settings were varied after the main result; these checks "
        "show sensitivity, not confirmation.",
        color=_MUTED,
        fontsize=9.5,
    )
    return figure


def _plot_concept_effects(axis: Axes, report: DependenceSensitivityReport) -> None:
    labels = {
        "assignment-evaluation": "Assignment",
        "conditional-control": "Conditionals",
        "function-arguments": "Functions",
        "object-aliasing": "Aliasing",
    }
    values = [item.mean_case_effect for item in report.concept_effects]
    bars = axis.bar(
        range(len(values)),
        values,
        width=0.68,
        color=[_BLUE if value >= 0 else _ORANGE for value in values],
    )
    axis.set_title("Public-answer comparison by concept", loc="left", pad=12)
    axis.set_ylabel("Single-source error minus probe-informed error")
    axis.set_xticks(
        range(len(values)),
        [labels[item.group_id] for item in report.concept_effects],
        fontsize=8,
    )
    axis.set_ylim(-1.0, 1.0)
    axis.set_yticks((-1.0, -0.5, 0.0, 0.5, 1.0))
    _style_axis(axis)
    for bar, value in zip(bars, values, strict=True):
        offset = 5 if value >= 0 else -5
        vertical_alignment = "bottom" if value >= 0 else "top"
        axis.annotate(
            f"{value:.2f}",
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=vertical_alignment,
            fontsize=8.5,
            fontweight="bold",
        )


def _plot_curve(
    axis: Axes,
    points: tuple[HeuristicSensitivityPoint, ...],
    *,
    title: str,
    x_label: str,
    x_value: Callable[[HeuristicSensitivityPoint], float],
    frozen_value: float,
) -> None:
    x_values = [x_value(point) for point in points]
    axis.plot(
        x_values,
        [point.dialogue_vs_probe_informed_effect for point in points],
        color=_BLUE,
        marker="o",
        linewidth=2.5,
        markersize=5,
    )
    axis.plot(
        x_values,
        [point.probe_informed_vs_probe_only_effect for point in points],
        color=_ORANGE,
        marker="s",
        linewidth=2.5,
        markersize=5,
    )
    axis.axvline(frozen_value, color=_MUTED, linewidth=1, linestyle=(0, (3, 3)))
    axis.annotate(
        f"Primary setting: {frozen_value:g}",
        (frozen_value, 0.49),
        xytext=(4, 0),
        textcoords="offset points",
        ha="left",
        va="top",
        color=_MUTED,
        fontsize=8,
    )
    axis.set_title(title, loc="left", pad=12)
    axis.set_xlabel(x_label)
    axis.set_ylim(-0.5, 0.5)
    axis.set_yticks((-0.5, -0.25, 0.0, 0.25, 0.5))
    axis.set_xticks(x_values)
    axis.tick_params(axis="x", labelsize=8)
    _style_axis(axis)


def _style_axis(axis: Axes) -> None:
    axis.axhline(0, color=_INK, linewidth=0.9)
    axis.grid(axis="y", color=_GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _require_utc(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("Dependence-figure timestamp must be UTC")


def _resolve_generated_at_utc(value: datetime | None, manifest_path: Path) -> datetime:
    if value is not None:
        _require_utc(value)
        return value
    if manifest_path.exists():
        try:
            manifest = DependenceFigureManifest.model_validate_json(manifest_path.read_bytes())
        except (OSError, ValidationError) as error:
            raise DependenceFigureError(
                f"Could not recover figure timestamp from manifest: {manifest_path}"
            ) from error
        return manifest.generated_at_utc
    return datetime.now(UTC)
