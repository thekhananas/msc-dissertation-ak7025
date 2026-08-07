"""Report-ready tables and vector figure for the canonical tracker study."""

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
from socratic_tutor.contracts import ContractModel
from socratic_tutor.tracker_study.analysis import (
    TrackerCanonicalAnalysisPlan,
    TrackerCanonicalAnalysisReport,
)
from socratic_tutor.tracker_study.config import StressCondition, TrackerId
from socratic_tutor.tracker_study.interpretation import (
    TrackerCanonicalInterpretationPlan,
    TrackerCanonicalInterpretationReport,
)
from socratic_tutor.tracker_study.simulation import StudySplit

_BLUE = "#0072B2"
_ORANGE = "#D55E00"
_GREEN = "#009E73"
_GREY = "#8A949E"
_LIGHT_GREY = "#D7DDE2"
_INK = "#17212B"
_MUTED = "#56616B"
_TITLE = "Bounded evidence updating under unreliable observations"
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
    "svg.hashsalt": "socratic-tutor-canonical-tracker-study-v1",
    "text.color": _INK,
    "xtick.color": _MUTED,
    "ytick.color": _MUTED,
}


class TrackerCanonicalPublicationError(ValueError):
    """Canonical publication sources are missing, invalid, or inconsistent."""


class TrackerCanonicalPublicationManifest(ContractModel):
    schema_version: Literal[1] = 1
    schema_id: Literal["tracker_study.canonical_publication_manifest.v1"] = (
        "tracker_study.canonical_publication_manifest.v1"
    )
    study_id: str
    run_id: str
    split: Literal[StudySplit.TEST] = StudySplit.TEST
    canonical_analysis_plan_hash: Sha256
    canonical_analysis_report_hash: Sha256
    canonical_interpretation_plan_hash: Sha256
    canonical_interpretation_report_hash: Sha256
    publication_code_revision: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    pixi_lock_sha256: Sha256
    figure_pdf_sha256: Sha256
    figure_svg_sha256: Sha256
    condition_metrics_csv_sha256: Sha256
    primary_summary_csv_sha256: Sha256
    claim_boundaries_csv_sha256: Sha256
    generated_at_utc: datetime
    publication_status: Literal["canonical_glass_box_simulator_result"] = (
        "canonical_glass_box_simulator_result"
    )
    primary_decision_status: Literal[
        "robust_under_declared_simulator",
        "not_robust_primary_interval",
        "not_robust_clean_guardrail",
        "not_robust_primary_and_clean_guardrail",
    ]
    bounded_robustness_result_supported: bool
    human_learning_claim_supported: Literal[False] = False
    tutoring_efficacy_claim_supported: Literal[False] = False
    reduced_cognitive_offloading_claim_supported: Literal[False] = False
    cognitive_bandwidth_validity_claim_supported: Literal[False] = False
    manifest_hash: Sha256

    @model_validator(mode="after")
    def validate_manifest(self) -> TrackerCanonicalPublicationManifest:
        _require_utc(self.generated_at_utc)
        expected_support = self.primary_decision_status == "robust_under_declared_simulator"
        if self.bounded_robustness_result_supported is not expected_support:
            raise ValueError("Publication claim status differs from the canonical decision")
        if self.manifest_hash != model_content_hash(self, exclude={"manifest_hash"}):
            raise ValueError("Canonical publication manifest hash does not match its content")
        return self


def publish_canonical_results(
    *,
    canonical_root: Path,
    interpretation_root: Path,
    pixi_lock_path: Path,
    output_root: Path,
    publication_code_revision: str,
    generated_at_utc: datetime | None = None,
) -> TrackerCanonicalPublicationManifest:
    """Publish canonical tables and a figure after reconciling their source records."""

    analysis_plan = _load(
        canonical_root / "canonical_analysis_plan.json",
        TrackerCanonicalAnalysisPlan,
        "canonical analysis plan",
    )
    analysis = _load(
        canonical_root / "canonical_analysis_report.json",
        TrackerCanonicalAnalysisReport,
        "canonical analysis report",
    )
    interpretation_plan = _load(
        interpretation_root / "canonical_interpretation_plan.json",
        TrackerCanonicalInterpretationPlan,
        "canonical interpretation plan",
    )
    interpretation = _load(
        interpretation_root / "canonical_interpretation_report.json",
        TrackerCanonicalInterpretationReport,
        "canonical interpretation report",
    )
    try:
        pixi_lock_hash = file_sha256(pixi_lock_path.read_bytes())
    except OSError as error:
        raise TrackerCanonicalPublicationError(
            f"Could not read Pixi lock: {pixi_lock_path}"
        ) from error
    _validate_lineage(
        analysis_plan=analysis_plan,
        analysis=analysis,
        interpretation_plan=interpretation_plan,
        interpretation=interpretation,
        pixi_lock_hash=pixi_lock_hash,
    )
    generated_at = _resolve_generated_at(
        generated_at_utc,
        output_root / "canonical_publication_manifest.json",
    )
    condition_csv = _condition_metrics_csv(analysis)
    primary_csv = _primary_summary_csv(analysis)
    claims_csv = _claim_boundaries_csv(analysis)
    pdf, svg = _render_figure(analysis, generated_at)
    supported = analysis.primary_decision_status == "robust_under_declared_simulator"
    content = {
        "schema_version": 1,
        "schema_id": "tracker_study.canonical_publication_manifest.v1",
        "study_id": analysis.study_id,
        "run_id": analysis.run_id,
        "split": StudySplit.TEST,
        "canonical_analysis_plan_hash": analysis_plan.plan_hash,
        "canonical_analysis_report_hash": analysis.report_hash,
        "canonical_interpretation_plan_hash": interpretation_plan.plan_hash,
        "canonical_interpretation_report_hash": interpretation.report_hash,
        "publication_code_revision": publication_code_revision,
        "pixi_lock_sha256": pixi_lock_hash,
        "figure_pdf_sha256": file_sha256(pdf),
        "figure_svg_sha256": file_sha256(svg),
        "condition_metrics_csv_sha256": file_sha256(condition_csv),
        "primary_summary_csv_sha256": file_sha256(primary_csv),
        "claim_boundaries_csv_sha256": file_sha256(claims_csv),
        "generated_at_utc": generated_at,
        "publication_status": "canonical_glass_box_simulator_result",
        "primary_decision_status": analysis.primary_decision_status,
        "bounded_robustness_result_supported": supported,
        "human_learning_claim_supported": False,
        "tutoring_efficacy_claim_supported": False,
        "reduced_cognitive_offloading_claim_supported": False,
        "cognitive_bandwidth_validity_claim_supported": False,
    }
    draft = TrackerCanonicalPublicationManifest.model_construct(
        _fields_set=set(content),
        **content,
        manifest_hash="0" * 64,
    )
    manifest = TrackerCanonicalPublicationManifest.model_validate(
        {**content, "manifest_hash": model_content_hash(draft, exclude={"manifest_hash"})}
    )
    write_immutable_bytes(output_root / "tracker_study_canonical.pdf", pdf)
    write_immutable_bytes(output_root / "tracker_study_canonical.svg", svg)
    write_immutable_bytes(output_root / "canonical_condition_metrics.csv", condition_csv)
    write_immutable_bytes(output_root / "canonical_primary_summary.csv", primary_csv)
    write_immutable_bytes(output_root / "canonical_claim_boundaries.csv", claims_csv)
    write_immutable_json(output_root / "canonical_publication_manifest.json", manifest)
    return manifest


def _condition_metrics_csv(report: TrackerCanonicalAnalysisReport) -> bytes:
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
    return _csv_bytes(
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


def _primary_summary_csv(report: TrackerCanonicalAnalysisReport) -> bytes:
    primary = report.primary_adverse_brier
    clean = report.clean_brier_guardrail
    rows = (
        (
            "five_adverse_conditions",
            "bounded_minus_ordinary_brier",
            primary.episode_count,
            primary.mean_effect,
            primary.interval_lower,
            primary.interval_upper,
            primary.two_sided_p_value,
            0.0,
            report.primary_interval_passed,
        ),
        (
            "clean_condition",
            "bounded_minus_ordinary_brier",
            clean.episode_count,
            clean.mean_effect,
            clean.interval_lower,
            clean.interval_upper,
            "",
            report.maximum_clean_brier_degradation,
            report.clean_guardrail_passed,
        ),
    )
    return _csv_bytes(
        (
            "comparison",
            "effect",
            "paired_episode_count",
            "mean",
            "interval_lower",
            "interval_upper",
            "secondary_sign_swap_p_value",
            "maximum_allowed_upper_bound",
            "prespecified_rule_passed",
        ),
        rows,
    )


def _claim_boundaries_csv(report: TrackerCanonicalAnalysisReport) -> bytes:
    supported = report.primary_decision_status == "robust_under_declared_simulator"
    rows = (
        (
            "bounded_tracker_robustness_in_declared_simulator",
            supported,
            "Supported only for the fixed glass-box simulator and stress conditions.",
        ),
        (
            "human_learning_improvement",
            False,
            "No learners took part in this tracker study.",
        ),
        (
            "tutoring_effectiveness",
            False,
            "The study compared tracker estimates, not teaching outcomes.",
        ),
        (
            "reduced_cognitive_offloading",
            False,
            "The study did not measure learner dependence on a tutor.",
        ),
        (
            "cognitive_bandwidth_validity",
            False,
            "The study did not evaluate the Cognitive Bandwidth construct.",
        ),
    )
    return _csv_bytes(("claim", "supported", "reason"), rows)


def _render_figure(
    report: TrackerCanonicalAnalysisReport,
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
                "Subject": "Canonical glass-box Bayesian tracker comparison",
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
                    "Held-out comparison of ordinary and bounded channel-aware mastery "
                    "trackers under six simulated evidence conditions."
                ),
                "Creator": "Socratic Tutor reproducible figure pipeline",
                "Date": generated_at_utc.isoformat(),
            },
        )
        figure.clear()
    return pdf_buffer.getvalue(), svg_buffer.getvalue()


def _build_figure(report: TrackerCanonicalAnalysisReport) -> Figure:
    figure = Figure(figsize=(13.4, 7.6))
    condition_axis = figure.add_subplot(1, 2, 1)
    comparison_axis = figure.add_subplot(1, 2, 2)
    figure.subplots_adjust(left=0.07, right=0.98, top=0.72, bottom=0.20, wspace=0.30)
    figure.suptitle(_TITLE, x=0.07, y=0.96, ha="left", fontsize=19, fontweight="bold")
    figure.text(
        0.07,
        0.89,
        "Held-out glass-box simulation; lower Brier error means more accurate "
        "mastery probabilities.",
        color=_MUTED,
        fontsize=10.5,
    )
    figure.text(
        0.07,
        0.845,
        f"Each condition contains {report.primary_adverse_brier.episode_count} matched episodes; "
        "both trackers saw the same latent paths and observations.",
        color=_MUTED,
        fontsize=10.5,
    )
    decision_text = (
        "Decision: passed the prespecified simulator rule."
        if report.primary_decision_status == "robust_under_declared_simulator"
        else "Decision: did not pass the prespecified simulator rule."
    )
    figure.text(
        0.07,
        0.80,
        decision_text,
        color=(
            _GREEN
            if report.primary_decision_status == "robust_under_declared_simulator"
            else _ORANGE
        ),
        fontsize=10.5,
        fontweight="bold",
    )
    _plot_condition_brier(condition_axis, report)
    _plot_primary_comparisons(comparison_axis, report)
    figure.text(
        0.07,
        0.075,
        "Scope: this result concerns robustness inside the declared simulator. It is not evidence "
        "of student learning, tutoring effectiveness, or reduced cognitive offloading.",
        color=_MUTED,
        fontsize=9.5,
        fontweight="bold",
    )
    figure.text(
        0.07,
        0.035,
        "The ordinary and bounded trackers were fixed before this one canonical test run; no "
        "parameters were selected from these results.",
        color=_MUTED,
        fontsize=9,
    )
    return figure


def _plot_condition_brier(axis: Axes, report: TrackerCanonicalAnalysisReport) -> None:
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
    axis.set_xticks(positions, [labels[condition] for condition in conditions], fontsize=8)
    maximum = max((*ordinary, *bounded))
    axis.set_ylim(0.0, min(1.0, maximum * 1.22))
    axis.legend(frameon=False, fontsize=8.5, loc="upper left")
    _style_axis(axis)


def _plot_primary_comparisons(axis: Axes, report: TrackerCanonicalAnalysisReport) -> None:
    primary = report.primary_adverse_brier
    clean = report.clean_brier_guardrail
    effects = (primary.mean_effect, clean.mean_effect)
    lower_errors = (
        primary.mean_effect - primary.interval_lower,
        clean.mean_effect - clean.interval_lower,
    )
    upper_errors = (
        primary.interval_upper - primary.mean_effect,
        clean.interval_upper - clean.mean_effect,
    )
    positions = (1, 0)
    axis.axvline(0.0, color=_INK, linewidth=1)
    axis.axvline(
        report.maximum_clean_brier_degradation,
        color=_ORANGE,
        linewidth=1,
        linestyle=(0, (3, 3)),
    )
    axis.errorbar(
        effects,
        positions,
        xerr=(lower_errors, upper_errors),
        fmt="o",
        markersize=7,
        color=_BLUE,
        ecolor=_BLUE,
        elinewidth=2,
        capsize=5,
        zorder=3,
    )
    axis.set_title("B. Prespecified paired comparisons", loc="left", pad=10)
    axis.set_xlabel("Change in Brier error (bounded minus ordinary)")
    axis.set_yticks(positions, ("Five difficult conditions", "Clean condition"))
    axis.set_ylim(-0.65, 1.65)
    all_values = (
        primary.interval_lower,
        primary.interval_upper,
        clean.interval_lower,
        clean.interval_upper,
        report.maximum_clean_brier_degradation,
    )
    padding = max(0.003, (max(all_values) - min(all_values)) * 0.22)
    axis.set_xlim(min(all_values) - padding, max(all_values) + padding)
    axis.text(
        0.02,
        0.98,
        "Left of zero favours bounded updating",
        transform=axis.transAxes,
        va="top",
        fontsize=8.5,
        color=_MUTED,
    )
    axis.annotate(
        "Clean-error limit",
        (report.maximum_clean_brier_degradation, 0.08),
        xytext=(-4, 7),
        textcoords="offset points",
        ha="right",
        fontsize=8,
        color=_ORANGE,
    )
    for position, effect, lower, upper in zip(
        positions,
        effects,
        (primary.interval_lower, clean.interval_lower),
        (primary.interval_upper, clean.interval_upper),
        strict=True,
    ):
        axis.annotate(
            f"{effect:+.4f}  [{lower:+.4f}, {upper:+.4f}]",
            (effect, position),
            xytext=(0, -18),
            textcoords="offset points",
            ha="center",
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
    analysis_plan: TrackerCanonicalAnalysisPlan,
    analysis: TrackerCanonicalAnalysisReport,
    interpretation_plan: TrackerCanonicalInterpretationPlan,
    interpretation: TrackerCanonicalInterpretationReport,
    pixi_lock_hash: Sha256,
) -> None:
    if analysis.analysis_plan_hash != analysis_plan.plan_hash:
        raise TrackerCanonicalPublicationError("Canonical analysis report and plan differ")
    if interpretation.interpretation_plan_hash != interpretation_plan.plan_hash:
        raise TrackerCanonicalPublicationError("Canonical interpretation report and plan differ")
    if interpretation_plan.canonical_analysis_plan_hash != analysis_plan.plan_hash:
        raise TrackerCanonicalPublicationError("Interpretation belongs to another analysis plan")
    if interpretation_plan.canonical_analysis_report_hash != analysis.report_hash:
        raise TrackerCanonicalPublicationError("Interpretation belongs to another analysis report")
    if analysis_plan.pixi_lock_sha256 != pixi_lock_hash:
        raise TrackerCanonicalPublicationError("Analysis used another Pixi lock")
    if interpretation_plan.pixi_lock_sha256 != pixi_lock_hash:
        raise TrackerCanonicalPublicationError("Interpretation used another Pixi lock")
    if {analysis.study_id, interpretation.study_id, interpretation_plan.study_id} != {
        analysis.study_id
    }:
        raise TrackerCanonicalPublicationError("Canonical sources belong to different studies")
    if {analysis.run_id, interpretation.run_id, interpretation_plan.run_id} != {analysis.run_id}:
        raise TrackerCanonicalPublicationError("Canonical sources belong to different runs")
    primary = analysis.primary_adverse_brier
    clean = analysis.clean_brier_guardrail
    expected = (
        primary.episode_count,
        primary.mean_effect,
        primary.interval_lower,
        primary.interval_upper,
        primary.two_sided_p_value,
        clean.mean_effect,
        clean.interval_lower,
        clean.interval_upper,
        analysis.maximum_clean_brier_degradation,
        analysis.primary_decision_status,
    )
    observed = (
        interpretation.canonical_episode_count_per_condition,
        interpretation.adverse_brier_mean_difference,
        interpretation.adverse_brier_interval_lower,
        interpretation.adverse_brier_interval_upper,
        interpretation.secondary_sign_swap_p_value,
        interpretation.clean_brier_mean_degradation,
        interpretation.clean_brier_interval_lower,
        interpretation.clean_brier_interval_upper,
        interpretation.clean_brier_maximum_allowed_degradation,
        interpretation.primary_decision_status,
    )
    if observed != expected:
        raise TrackerCanonicalPublicationError("Interpretation values differ from the analysis")


def _csv_bytes(
    headers: tuple[str, ...],
    rows: list[tuple[object, ...]] | tuple[tuple[object, ...], ...],
) -> bytes:
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
            TrackerCanonicalPublicationManifest,
            "canonical publication manifest",
        )
        return existing.generated_at_utc
    return datetime.now(UTC)


def _require_utc(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("Tracker publication timestamp must be timezone-aware UTC")


def _load[ModelT: ContractModel](path: Path, model: type[ModelT], label: str) -> ModelT:
    try:
        return model.model_validate_json(path.read_bytes())
    except (OSError, ValidationError) as error:
        raise TrackerCanonicalPublicationError(f"Could not verify {label}: {path}") from error
