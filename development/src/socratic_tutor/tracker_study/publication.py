"""Report-ready tables and vector figure for tracker-study development results."""

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
from socratic_tutor.contracts import ContractModel
from socratic_tutor.csv_output import csv_bytes
from socratic_tutor.tracker_study.analysis import (
    TrackerDevelopmentAnalysisPlan,
    TrackerDevelopmentAnalysisReport,
)
from socratic_tutor.tracker_study.config import StressCondition, TrackerId
from socratic_tutor.tracker_study.runtime import (
    TrackerDevelopmentRuntimePlan,
    TrackerDevelopmentRuntimeReport,
)
from socratic_tutor.tracker_study.sensitivity import (
    SensitivityDimension,
    TrackerDevelopmentSensitivityPlan,
    TrackerDevelopmentSensitivityReport,
)
from socratic_tutor.tracker_study.simulation import StudySplit

_BLUE = "#0072B2"
_ORANGE = "#D55E00"
_GREEN = "#009E73"
_PURPLE = "#7B3294"
_GREY = "#8A949E"
_LIGHT_GREY = "#D7DDE2"
_INK = "#17212B"
_MUTED = "#56616B"
_FIGURE_TITLE = "When does bounded evidence updating help?"
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
    "svg.hashsalt": "socratic-tutor-tracker-study-v1",
    "text.color": _INK,
    "xtick.color": _MUTED,
    "ytick.color": _MUTED,
}


class TrackerPublicationError(ValueError):
    """A tracker-study publication source is invalid or inconsistent."""


class TrackerDevelopmentPublicationManifest(ContractModel):
    schema_version: Literal[1] = 1
    schema_id: Literal["tracker_study.development_publication_manifest.v1"] = (
        "tracker_study.development_publication_manifest.v1"
    )
    study_id: str
    split: Literal[StudySplit.DEVELOPMENT] = StudySplit.DEVELOPMENT
    analysis_plan_hash: Sha256
    analysis_report_hash: Sha256
    sensitivity_plan_hash: Sha256
    sensitivity_report_hash: Sha256
    runtime_plan_hash: Sha256
    runtime_report_hash: Sha256
    publication_code_revision: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    pixi_lock_sha256: Sha256
    figure_pdf_sha256: Sha256
    figure_svg_sha256: Sha256
    condition_metrics_csv_sha256: Sha256
    sensitivity_csv_sha256: Sha256
    runtime_csv_sha256: Sha256
    primary_summary_csv_sha256: Sha256
    generated_at_utc: datetime
    publication_status: Literal["development_diagnostic_not_canonical"] = (
        "development_diagnostic_not_canonical"
    )
    canonical_claim_allowed: Literal[False] = False
    human_learning_claim_supported: Literal[False] = False
    tutoring_efficacy_claim_supported: Literal[False] = False
    manifest_hash: Sha256

    @model_validator(mode="after")
    def validate_manifest(self) -> TrackerDevelopmentPublicationManifest:
        _require_utc(self.generated_at_utc)
        if self.manifest_hash != model_content_hash(self, exclude={"manifest_hash"}):
            raise ValueError("Tracker publication manifest hash does not match its content")
        return self


def publish_development_results(
    *,
    analysis_root: Path,
    sensitivity_root: Path,
    runtime_root: Path,
    pixi_lock_path: Path,
    output_root: Path,
    publication_code_revision: str,
    generated_at_utc: datetime | None = None,
) -> TrackerDevelopmentPublicationManifest:
    """Publish reconciled development tables and one explanatory vector figure."""

    analysis_plan = _load(
        analysis_root / "development_analysis_plan.json",
        TrackerDevelopmentAnalysisPlan,
        "analysis plan",
    )
    analysis_report = _load(
        analysis_root / "development_analysis_report.json",
        TrackerDevelopmentAnalysisReport,
        "analysis report",
    )
    sensitivity_plan = _load(
        sensitivity_root / "development_sensitivity_plan.json",
        TrackerDevelopmentSensitivityPlan,
        "sensitivity plan",
    )
    sensitivity_report = _load(
        sensitivity_root / "development_sensitivity_report.json",
        TrackerDevelopmentSensitivityReport,
        "sensitivity report",
    )
    runtime_plan = _load(
        runtime_root / "development_runtime_plan.json",
        TrackerDevelopmentRuntimePlan,
        "runtime plan",
    )
    runtime_report = _load(
        runtime_root / "development_runtime_report.json",
        TrackerDevelopmentRuntimeReport,
        "runtime report",
    )
    _validate_lineage(
        analysis_plan=analysis_plan,
        analysis_report=analysis_report,
        sensitivity_plan=sensitivity_plan,
        sensitivity_report=sensitivity_report,
        runtime_plan=runtime_plan,
        runtime_report=runtime_report,
    )
    try:
        pixi_lock_hash = file_sha256(pixi_lock_path.read_bytes())
    except OSError as error:
        raise TrackerPublicationError(f"Could not read Pixi lock: {pixi_lock_path}") from error
    generated_at_utc = _resolve_generated_at(
        generated_at_utc,
        output_root / "publication_manifest.json",
    )
    condition_csv = _condition_metrics_csv(analysis_report)
    sensitivity_csv = _sensitivity_csv(sensitivity_report)
    runtime_csv = _runtime_csv(runtime_report)
    primary_csv = _primary_summary_csv(analysis_report)
    pdf, svg = _render_figure(
        analysis_report,
        sensitivity_report,
        runtime_report,
        generated_at_utc,
    )
    content = {
        "schema_version": 1,
        "schema_id": "tracker_study.development_publication_manifest.v1",
        "study_id": analysis_report.study_id,
        "split": StudySplit.DEVELOPMENT,
        "analysis_plan_hash": analysis_plan.plan_hash,
        "analysis_report_hash": analysis_report.report_hash,
        "sensitivity_plan_hash": sensitivity_plan.plan_hash,
        "sensitivity_report_hash": sensitivity_report.report_hash,
        "runtime_plan_hash": runtime_plan.plan_hash,
        "runtime_report_hash": runtime_report.report_hash,
        "publication_code_revision": publication_code_revision,
        "pixi_lock_sha256": pixi_lock_hash,
        "figure_pdf_sha256": file_sha256(pdf),
        "figure_svg_sha256": file_sha256(svg),
        "condition_metrics_csv_sha256": file_sha256(condition_csv),
        "sensitivity_csv_sha256": file_sha256(sensitivity_csv),
        "runtime_csv_sha256": file_sha256(runtime_csv),
        "primary_summary_csv_sha256": file_sha256(primary_csv),
        "generated_at_utc": generated_at_utc,
        "publication_status": "development_diagnostic_not_canonical",
        "canonical_claim_allowed": False,
        "human_learning_claim_supported": False,
        "tutoring_efficacy_claim_supported": False,
    }
    draft = TrackerDevelopmentPublicationManifest.model_construct(
        _fields_set=set(content),
        **content,
        manifest_hash="0" * 64,
    )
    manifest = TrackerDevelopmentPublicationManifest.model_validate(
        {**content, "manifest_hash": model_content_hash(draft, exclude={"manifest_hash"})}
    )
    write_immutable_bytes(output_root / "tracker_study_development.pdf", pdf)
    write_immutable_bytes(output_root / "tracker_study_development.svg", svg)
    write_immutable_bytes(output_root / "condition_metrics.csv", condition_csv)
    write_immutable_bytes(output_root / "sensitivity.csv", sensitivity_csv)
    write_immutable_bytes(output_root / "runtime.csv", runtime_csv)
    write_immutable_bytes(output_root / "primary_summary.csv", primary_csv)
    write_immutable_json(output_root / "publication_manifest.json", manifest)
    return manifest


def _condition_metrics_csv(report: TrackerDevelopmentAnalysisReport) -> bytes:
    rows = [
        (
            row.condition.value,
            row.tracker_id.value,
            row.episode_count,
            row.turn_count,
            row.non_missing_update_count,
            row.metrics.brier_score,
            row.metrics.negative_log_likelihood,
            row.metrics.expected_calibration_error,
            row.metrics.classification_error,
            row.metrics.mean_absolute_posterior_update,
        )
        for row in report.condition_metrics
    ]
    return csv_bytes(
        (
            "condition",
            "tracker_id",
            "episode_count",
            "turn_count",
            "non_missing_update_count",
            "brier_score",
            "negative_log_likelihood",
            "expected_calibration_error",
            "classification_error",
            "mean_absolute_posterior_update",
        ),
        rows,
    )


def _sensitivity_csv(report: TrackerDevelopmentSensitivityReport) -> bytes:
    rows = [
        (
            point.dimension.value,
            point.value,
            point.is_configured_anchor,
            point.episode_count,
            point.candidate_mean_adverse_brier,
            point.reference_mean_adverse_brier,
            point.candidate_minus_reference_adverse_brier,
            point.candidate_clean_brier,
            point.reference_clean_brier,
            point.candidate_minus_reference_clean_brier,
        )
        for point in report.points
    ]
    return csv_bytes(
        (
            "dimension",
            "value",
            "is_configured_anchor",
            "episode_count",
            "candidate_mean_adverse_brier",
            "reference_mean_adverse_brier",
            "candidate_minus_reference_adverse_brier",
            "candidate_clean_brier",
            "reference_clean_brier",
            "candidate_minus_reference_clean_brier",
        ),
        rows,
    )


def _runtime_csv(report: TrackerDevelopmentRuntimeReport) -> bytes:
    rows = [
        (
            row.tracker_id.value,
            row.non_missing_updates_per_repetition,
            row.warmup_repetitions,
            row.measured_repetitions,
            row.median_nanoseconds_per_non_missing_update,
            row.minimum_nanoseconds_per_non_missing_update,
            row.maximum_nanoseconds_per_non_missing_update,
            row.median_nanoseconds_per_non_missing_update / 1000.0,
            row.workload_hash,
        )
        for row in report.measurements
    ]
    return csv_bytes(
        (
            "tracker_id",
            "non_missing_updates_per_repetition",
            "warmup_repetitions",
            "measured_repetitions",
            "median_nanoseconds_per_update",
            "minimum_nanoseconds_per_update",
            "maximum_nanoseconds_per_update",
            "median_microseconds_per_update",
            "workload_hash",
        ),
        rows,
    )


def _primary_summary_csv(report: TrackerDevelopmentAnalysisReport) -> bytes:
    primary = report.primary_adverse_brier
    clean = report.clean_brier_guardrail
    rows = (
        (
            "adverse_conditions",
            "bounded_minus_ordinary_brier",
            primary.mean_effect,
            primary.interval_lower,
            primary.interval_upper,
            "lower_is_better",
            "development_only_no_decision",
        ),
        (
            "clean_condition",
            "bounded_minus_ordinary_brier",
            clean.mean_effect,
            clean.interval_lower,
            clean.interval_upper,
            f"upper_interval_at_most_{report.maximum_clean_brier_degradation}",
            "development_only_no_decision",
        ),
    )
    return csv_bytes(
        (
            "comparison",
            "effect",
            "mean",
            "interval_lower",
            "interval_upper",
            "favourable_result",
            "status",
        ),
        rows,
    )


def _render_figure(
    analysis: TrackerDevelopmentAnalysisReport,
    sensitivity: TrackerDevelopmentSensitivityReport,
    runtime: TrackerDevelopmentRuntimeReport,
    generated_at_utc: datetime,
) -> tuple[bytes, bytes]:
    with mpl.rc_context(_STYLE):
        figure = _build_figure(analysis, sensitivity, runtime)
        pdf_buffer = BytesIO()
        svg_buffer = BytesIO()
        figure.savefig(
            pdf_buffer,
            format="pdf",
            metadata={
                "Title": _FIGURE_TITLE,
                "Author": "Anas Khan",
                "Subject": "Development diagnostics for the Bayesian tracker study",
                "Creator": "Socratic Tutor reproducible figure pipeline",
                "CreationDate": generated_at_utc,
                "ModDate": generated_at_utc,
            },
        )
        figure.savefig(
            svg_buffer,
            format="svg",
            metadata={
                "Title": _FIGURE_TITLE,
                "Description": (
                    "Development comparison of ordinary and bounded channel-aware mastery "
                    "trackers under six simulated evidence conditions, twelve sensitivity "
                    "settings, and one local runtime profile."
                ),
                "Creator": "Socratic Tutor reproducible figure pipeline",
                "Date": generated_at_utc.isoformat(),
            },
        )
        figure.clear()
    return pdf_buffer.getvalue(), svg_buffer.getvalue()


def _build_figure(
    analysis: TrackerDevelopmentAnalysisReport,
    sensitivity: TrackerDevelopmentSensitivityReport,
    runtime: TrackerDevelopmentRuntimeReport,
) -> Figure:
    figure = Figure(figsize=(14.0, 7.8))
    axes = tuple(figure.add_subplot(1, 3, index) for index in range(1, 4))
    figure.subplots_adjust(left=0.065, right=0.985, top=0.72, bottom=0.19, wspace=0.32)
    figure.suptitle(_FIGURE_TITLE, x=0.065, y=0.965, ha="left", fontsize=19, fontweight="bold")
    figure.text(
        0.065,
        0.895,
        "The bounded tracker limits how far one piece of evidence can move its mastery estimate.",
        color=_MUTED,
        fontsize=10.5,
    )
    figure.text(
        0.065,
        0.855,
        "Lower Brier score means more accurate probabilities; 20 matched simulated episodes "
        "were used per condition.",
        color=_MUTED,
        fontsize=10.5,
    )
    figure.text(
        0.065,
        0.815,
        "Sensitivity settings describe the trade-off only; the configured tracker was not changed.",
        color=_MUTED,
        fontsize=10.5,
    )
    _plot_condition_brier(axes[0], analysis)
    _plot_tradeoff(axes[1], sensitivity, analysis.maximum_clean_brier_degradation)
    _plot_runtime(axes[2], runtime)
    figure.text(
        0.065,
        0.065,
        "Development diagnostics only. Results describe the declared simulator, not human "
        "learning or tutoring effectiveness.",
        color=_MUTED,
        fontsize=9.5,
        fontweight="bold",
    )
    figure.text(
        0.065,
        0.028,
        "Runtime values are local CPython measurements on "
        f"{runtime.platform.operating_system} {runtime.platform.machine}; they are not "
        "end-to-end tutor latency.",
        color=_MUTED,
        fontsize=9,
    )
    return figure


def _plot_condition_brier(axis: Axes, report: TrackerDevelopmentAnalysisReport) -> None:
    conditions = tuple(StressCondition)
    rows = {(row.condition, row.tracker_id): row for row in report.condition_metrics}
    ordinary = [
        rows[(condition, TrackerId.CHANNEL_AWARE)].metrics.brier_score for condition in conditions
    ]
    bounded = [
        rows[(condition, TrackerId.BOUNDED_CHANNEL_AWARE)].metrics.brier_score
        for condition in conditions
    ]
    positions = list(range(len(conditions)))
    width = 0.36
    axis.bar(
        [position - width / 2 for position in positions],
        ordinary,
        width=width,
        color=_BLUE,
        label="Ordinary channel-aware Bayes",
    )
    axis.bar(
        [position + width / 2 for position in positions],
        bounded,
        width=width,
        color=_ORANGE,
        label="Bounded channel-aware update",
    )
    labels = {
        StressCondition.CLEAN: "Clean",
        StressCondition.UNINFORMATIVE: "No useful\ninformation",
        StressCondition.INVERTED: "Inverted",
        StressCondition.MISSING: "Missing",
        StressCondition.RELIABILITY_MISSPECIFIED: "Reliability\nmisstated",
        StressCondition.CONTRADICTORY: "Conflicting\nchannels",
    }
    axis.set_title("A. Probability error by evidence condition", loc="left", pad=10)
    axis.set_ylabel("Brier score (lower is better)")
    axis.set_xticks(positions, [labels[condition] for condition in conditions], fontsize=7.5)
    axis.set_ylim(0.0, 1.0)
    axis.legend(frameon=False, fontsize=8, loc="upper left")
    _style_axis(axis)


def _plot_tradeoff(
    axis: Axes,
    report: TrackerDevelopmentSensitivityReport,
    clean_guardrail: float,
) -> None:
    styles = {
        SensitivityDimension.TRUST_WEIGHT_MULTIPLIER: (_GREEN, "o", "Trust weight"),
        SensitivityDimension.CLIPPING_KAPPA: (_PURPLE, "s", "Clipping limit"),
        SensitivityDimension.ASSUMED_CHANNEL_RELIABILITY_SCALE: (
            _ORANGE,
            "^",
            "Assumed reliability",
        ),
    }
    configured_points: list[tuple[float, float]] = []
    for dimension, (colour, marker, label) in styles.items():
        points = tuple(point for point in report.points if point.dimension is dimension)
        x_values = [point.candidate_minus_reference_clean_brier for point in points]
        y_values = [-point.candidate_minus_reference_adverse_brier for point in points]
        axis.scatter(
            x_values,
            y_values,
            color=colour,
            marker=marker,
            s=42,
            label=label,
            zorder=3,
        )
        for point, x_value, y_value in zip(points, x_values, y_values, strict=True):
            if point.is_configured_anchor:
                configured_points.append((x_value, y_value))
                continue
            offset = (4, 3)
            if dimension is SensitivityDimension.CLIPPING_KAPPA:
                offset = (-9, -11) if point.value == 4.0 else (4, -10)
            axis.annotate(
                f"{point.value:g}",
                (x_value, y_value),
                xytext=offset,
                textcoords="offset points",
                fontsize=7,
                color=colour,
            )
    configured_x, configured_y = configured_points[0]
    axis.scatter(
        [configured_x],
        [configured_y],
        facecolors="none",
        edgecolors=_INK,
        linewidths=1.4,
        s=120,
        zorder=4,
    )
    axis.annotate(
        "Configured tracker",
        (configured_x, configured_y),
        xytext=(8, -16),
        textcoords="offset points",
        fontsize=7.5,
        color=_INK,
        fontweight="bold",
    )
    axis.axhline(0.0, color=_GREY, linewidth=1)
    axis.axvline(0.0, color=_GREY, linewidth=1)
    axis.axvline(clean_guardrail, color=_INK, linewidth=1, linestyle=(0, (3, 3)))
    axis.annotate(
        "Clean-error limit",
        (clean_guardrail, 0.068),
        xytext=(3, 0),
        textcoords="offset points",
        fontsize=7.5,
        color=_INK,
        rotation=90,
        va="top",
    )
    axis.set_title("B. Robustness versus clean-evidence cost", loc="left", pad=10)
    axis.set_xlabel("Extra clean-condition Brier error")
    axis.set_ylabel("Adverse-condition Brier error removed")
    axis.set_xlim(-0.002, 0.045)
    axis.set_ylim(-0.003, 0.071)
    axis.legend(frameon=False, fontsize=7.5, loc="lower right")
    _style_axis(axis)


def _plot_runtime(axis: Axes, report: TrackerDevelopmentRuntimeReport) -> None:
    labels = {
        TrackerId.LAST_OBSERVATION: "Last observation",
        TrackerId.HARD_BKT: "Hard BKT",
        TrackerId.LEGACY_FRACTIONAL: "Legacy fractional",
        TrackerId.CHANNEL_AWARE: "Channel-aware Bayes",
        TrackerId.BOUNDED_CHANNEL_AWARE: "Bounded channel-aware",
    }
    values = [
        measurement.median_nanoseconds_per_non_missing_update / 1000.0
        for measurement in report.measurements
    ]
    colours = [
        _BLUE
        if measurement.tracker_id is TrackerId.CHANNEL_AWARE
        else _ORANGE
        if measurement.tracker_id is TrackerId.BOUNDED_CHANNEL_AWARE
        else _GREY
        for measurement in report.measurements
    ]
    positions = list(range(len(values)))
    bars = axis.barh(positions, values, color=colours, height=0.65)
    axis.set_title("C. Local computation per evidence update", loc="left", pad=10)
    axis.set_xlabel("Median microseconds (lower is faster)")
    axis.set_yticks(
        positions,
        [labels[item.tracker_id] for item in report.measurements],
        fontsize=8,
    )
    axis.invert_yaxis()
    axis.set_xlim(0.0, max(values) * 1.22)
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            value + max(values) * 0.025,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            fontsize=8,
            color=_INK,
        )
    _style_axis(axis)


def _style_axis(axis: Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color=_LIGHT_GREY, linewidth=0.7, alpha=0.7)
    axis.set_axisbelow(True)


def _validate_lineage(
    *,
    analysis_plan: TrackerDevelopmentAnalysisPlan,
    analysis_report: TrackerDevelopmentAnalysisReport,
    sensitivity_plan: TrackerDevelopmentSensitivityPlan,
    sensitivity_report: TrackerDevelopmentSensitivityReport,
    runtime_plan: TrackerDevelopmentRuntimePlan,
    runtime_report: TrackerDevelopmentRuntimeReport,
) -> None:
    if analysis_report.analysis_plan_hash != analysis_plan.plan_hash:
        raise TrackerPublicationError("Analysis report and plan differ")
    if sensitivity_plan.development_analysis_plan_hash != analysis_plan.plan_hash:
        raise TrackerPublicationError("Sensitivity plan belongs to another analysis plan")
    if sensitivity_plan.development_analysis_report_hash != analysis_report.report_hash:
        raise TrackerPublicationError("Sensitivity plan belongs to another analysis report")
    if sensitivity_report.sensitivity_plan_hash != sensitivity_plan.plan_hash:
        raise TrackerPublicationError("Sensitivity report and plan differ")
    if runtime_plan.development_analysis_plan_hash != analysis_plan.plan_hash:
        raise TrackerPublicationError("Runtime plan belongs to another analysis plan")
    if runtime_plan.development_analysis_report_hash != analysis_report.report_hash:
        raise TrackerPublicationError("Runtime plan belongs to another analysis report")
    if runtime_report.runtime_plan_hash != runtime_plan.plan_hash:
        raise TrackerPublicationError("Runtime report and plan differ")
    study_ids = {
        analysis_report.study_id,
        sensitivity_report.study_id,
        runtime_report.study_id,
    }
    if len(study_ids) != 1:
        raise TrackerPublicationError("Publication sources belong to different studies")


def _resolve_generated_at(value: datetime | None, manifest_path: Path) -> datetime:
    if value is not None:
        _require_utc(value)
        return value
    if manifest_path.exists():
        existing = _load(
            manifest_path,
            TrackerDevelopmentPublicationManifest,
            "publication manifest",
        )
        return existing.generated_at_utc
    return datetime.now(UTC)


def _require_utc(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("Tracker publication timestamp must be UTC")


def _load[ModelT: ContractModel](path: Path, model: type[ModelT], label: str) -> ModelT:
    try:
        return model.model_validate_json(path.read_bytes())
    except (OSError, ValidationError) as error:
        raise TrackerPublicationError(f"Could not verify {label}: {path}") from error
