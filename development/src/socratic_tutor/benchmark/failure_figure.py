"""Reproducible figure for the observable failure review."""

# pyright: reportArgumentType=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

from collections import Counter
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
from socratic_tutor.benchmark.failure_review import (
    FailureReviewLabel,
    FailureReviewRecordReport,
)
from socratic_tutor.benchmark.failure_review_reliability import (
    FailureReviewReliabilityPlan,
    FailureReviewReliabilityReport,
)
from socratic_tutor.benchmark.hashing import file_sha256, model_content_hash
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
_TITLE = "What caused the apparent benchmark failures?"
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
    "svg.hashsalt": "socratic-tutor-failure-review-v1",
    "text.color": _INK,
    "xtick.color": _MUTED,
    "ytick.color": _MUTED,
}

_PLAIN_CATEGORY = {
    FailureReviewLabel.ITEM_AMBIGUITY_OR_TEST_DEFECT: "Authored item or test defect",
    FailureReviewLabel.SYNTHETIC_STUDENT_INCONSISTENCY: "Evaluation-model inconsistency",
    FailureReviewLabel.EVIDENCE_EXTRACTION_OR_EXECUTION_ERROR: (
        "Evidence extraction or execution error"
    ),
    FailureReviewLabel.TRACKER_UPDATE_OR_CALIBRATION_ERROR: "Tracker update or calibration error",
    FailureReviewLabel.POLICY_SELECTION_ERROR: "Policy selection error",
    FailureReviewLabel.TUTOR_RESPONSE_FAILURE: "Tutor response failure",
    FailureReviewLabel.PROVIDER_OR_SANDBOX_FAILURE: "Provider or sandbox failure",
    FailureReviewLabel.UNCLASSIFIED: "Unclassified",
    FailureReviewLabel.NO_FAILURE_OBSERVED: "No failure observed",
}


class FailureFigureError(ValueError):
    """Failure-review figure sources are missing, invalid, or inconsistent."""


class FailureFigureManifest(ContractModel):
    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.failure_figure_manifest.v1"] = (
        "benchmark.failure_figure_manifest.v1"
    )
    primary_review_report_hash: Sha256
    secondary_review_report_hash: Sha256
    reliability_plan_hash: Sha256
    reliability_report_hash: Sha256
    publication_code_revision: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    pixi_lock_sha256: Sha256
    reviewed_item_count: Literal[6] = 6
    primary_failure_count: int = Field(ge=1)
    matched_success_count: int = Field(ge=1)
    panel_count: Literal[3] = 3
    figure_formats: tuple[Literal["pdf", "svg"], Literal["pdf", "svg"]]
    pdf_sha256: Sha256
    svg_sha256: Sha256
    review_rows_csv_sha256: Sha256
    generated_at_utc: datetime
    agreement_establishes_correctness: Literal[False] = False
    hidden_reasoning_used: Literal[False] = False
    human_learning_claim_supported: Literal[False] = False
    manifest_hash: Sha256

    @model_validator(mode="after")
    def validate_manifest(self) -> FailureFigureManifest:
        _require_utc(self.generated_at_utc)
        if self.figure_formats != ("pdf", "svg"):
            raise ValueError("Failure-review figure must publish PDF and SVG")
        if self.manifest_hash != model_content_hash(self, exclude={"manifest_hash"}):
            raise ValueError("Failure-review figure manifest hash does not match its content")
        return self


def render_failure_figure(
    *,
    primary_review_path: Path,
    secondary_review_path: Path,
    reliability_plan_path: Path,
    reliability_report_path: Path,
    pixi_lock_path: Path,
    output_root: Path,
    publication_code_revision: str,
    generated_at_utc: datetime | None = None,
) -> FailureFigureManifest:
    """Render the reviewed failure categories and their reliability boundary."""

    primary = _load(primary_review_path, FailureReviewRecordReport, "primary review")
    secondary = _load(secondary_review_path, FailureReviewRecordReport, "secondary review")
    plan = _load(reliability_plan_path, FailureReviewReliabilityPlan, "reliability plan")
    reliability = _load(
        reliability_report_path,
        FailureReviewReliabilityReport,
        "reliability report",
    )
    try:
        lock_hash = file_sha256(pixi_lock_path.read_bytes())
        primary_file_hash = file_sha256(primary_review_path.read_bytes())
        secondary_file_hash = file_sha256(secondary_review_path.read_bytes())
    except OSError as error:
        raise FailureFigureError("Could not hash a failure-review figure input") from error
    _validate_sources(
        primary=primary,
        secondary=secondary,
        plan=plan,
        reliability=reliability,
        lock_hash=lock_hash,
        primary_file_hash=primary_file_hash,
        secondary_file_hash=secondary_file_hash,
    )
    generated_at = _resolve_generated_at(
        generated_at_utc,
        output_root / "failure_figure_manifest.json",
    )
    review_csv = _review_rows_csv(primary, secondary)
    pdf, svg = _render_vector_files(primary, secondary, reliability, generated_at)
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.failure_figure_manifest.v1",
        "primary_review_report_hash": primary.report_hash,
        "secondary_review_report_hash": secondary.report_hash,
        "reliability_plan_hash": plan.plan_hash,
        "reliability_report_hash": reliability.report_hash,
        "publication_code_revision": publication_code_revision,
        "pixi_lock_sha256": lock_hash,
        "reviewed_item_count": reliability.item_count,
        "primary_failure_count": primary.primary_failure_count,
        "matched_success_count": primary.matched_success_count,
        "panel_count": 3,
        "figure_formats": ("pdf", "svg"),
        "pdf_sha256": file_sha256(pdf),
        "svg_sha256": file_sha256(svg),
        "review_rows_csv_sha256": file_sha256(review_csv),
        "generated_at_utc": generated_at,
        "agreement_establishes_correctness": False,
        "hidden_reasoning_used": False,
        "human_learning_claim_supported": False,
    }
    draft = FailureFigureManifest.model_construct(
        _fields_set=set(content),
        **content,
        manifest_hash="0" * 64,
    )
    manifest = FailureFigureManifest.model_validate(
        {**content, "manifest_hash": model_content_hash(draft, exclude={"manifest_hash"})}
    )
    write_immutable_bytes(output_root / "failure_analysis.pdf", pdf)
    write_immutable_bytes(output_root / "failure_analysis.svg", svg)
    write_immutable_bytes(output_root / "failure_review_rows.csv", review_csv)
    write_immutable_json(output_root / "failure_figure_manifest.json", manifest)
    return manifest


def _review_rows_csv(
    primary: FailureReviewRecordReport,
    secondary: FailureReviewRecordReport,
) -> bytes:
    secondary_by_case = {record.case_id: record for record in secondary.records}
    rows = [
        (
            record.case_id,
            record.selection_role,
            record.matched_case_id,
            record.category,
            secondary_by_case[record.case_id].category,
            record.category is secondary_by_case[record.case_id].category,
            record.rationale,
            secondary_by_case[record.case_id].rationale,
        )
        for record in sorted(primary.records, key=lambda item: item.case_id)
    ]
    return csv_bytes(
        (
            "case_id",
            "selection_role",
            "matched_case_id",
            "primary_category",
            "secondary_category",
            "reviewers_agree",
            "primary_rationale",
            "secondary_rationale",
        ),
        rows,
    )


def _render_vector_files(
    primary: FailureReviewRecordReport,
    secondary: FailureReviewRecordReport,
    reliability: FailureReviewReliabilityReport,
    generated_at_utc: datetime,
) -> tuple[bytes, bytes]:
    with mpl.rc_context(_STYLE):
        figure = _build_figure(primary, secondary, reliability)
        pdf_buffer = BytesIO()
        svg_buffer = BytesIO()
        figure.savefig(
            pdf_buffer,
            format="pdf",
            metadata={
                "Title": _TITLE,
                "Author": "Anas Khan",
                "Subject": "Observable benchmark failure review",
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
                    "Independent reviewer categories for three apparent primary regressions "
                    "and three matched successes."
                ),
                "Creator": "Socratic Tutor reproducible figure pipeline",
                "Date": generated_at_utc.isoformat(),
            },
        )
        figure.clear()
    return pdf_buffer.getvalue(), svg_buffer.getvalue()


def _build_figure(
    primary: FailureReviewRecordReport,
    secondary: FailureReviewRecordReport,
    reliability: FailureReviewReliabilityReport,
) -> Figure:
    figure = Figure(figsize=(13.4, 8.7))
    grid = figure.add_gridspec(
        2,
        2,
        left=0.075,
        right=0.98,
        top=0.66,
        bottom=0.18,
        width_ratios=(1.55, 1.0),
        height_ratios=(1.0, 0.8),
        hspace=0.48,
        wspace=0.38,
    )
    cases_axis = figure.add_subplot(grid[:, 0])
    counts_axis = figure.add_subplot(grid[0, 1])
    agreement_axis = figure.add_subplot(grid[1, 1])

    _plot_reviewed_cases(cases_axis, primary, secondary)
    _plot_category_counts(counts_axis, reliability)
    _plot_agreement(agreement_axis, reliability)

    figure.text(0.075, 0.95, _TITLE, fontsize=19, fontweight="bold", color=_INK)
    figure.text(
        0.075,
        0.902,
        "Two reviewers used visible prompts, responses, decisions, and test outcomes; "
        "hidden model reasoning was not available.",
        fontsize=10.5,
        color=_MUTED,
    )
    figure.text(
        0.075,
        0.835,
        "Both reviewers classified all three apparent regressions as authored item or "
        "test defects.",
        fontsize=13,
        fontweight="bold",
        color=_ORANGE,
    )
    figure.text(
        0.075,
        0.795,
        "One matched success also exposed a task/test mismatch; two matched successes "
        "had no observed failure.",
        fontsize=10.5,
        color=_INK,
    )
    figure.text(
        0.075,
        0.735,
        "Interpretation: some apparent model errors were measurement errors in the "
        "authored benchmark.",
        fontsize=10,
        fontweight="bold",
        color=_BLUE,
    )
    figure.text(
        0.075,
        0.160,
        "TD = authored item or test defect; OK = no failure observed. "
        "Seven other taxonomy categories received no labels.",
        fontsize=8.5,
        color=_MUTED,
    )
    figure.text(
        0.075,
        0.120,
        "Scope: six deliberately selected records, three apparent regressions and three "
        "matched successes. They are not a random sample of all cases.",
        fontsize=9,
        fontweight="bold",
        color=_MUTED,
    )
    figure.text(
        0.075,
        0.075,
        "Perfect agreement shows that the final labels were reproducible under the shared guide; "
        "it does not prove that the labels or taxonomy are objectively correct.",
        fontsize=9,
        color=_MUTED,
    )
    figure.text(
        0.075,
        0.035,
        "This review supports a post-hoc measurement correction only. It cannot replace "
        "the sealed primary result or support claims about human learning.",
        fontsize=8.5,
        color=_MUTED,
    )
    return figure


def _plot_reviewed_cases(
    axis: Axes,
    primary: FailureReviewRecordReport,
    secondary: FailureReviewRecordReport,
) -> None:
    primary_rows = sorted(
        primary.records,
        key=lambda item: (item.selection_role != "primary_failure", item.case_id),
    )
    secondary_by_case = {record.case_id: record for record in secondary.records}
    positions = list(range(len(primary_rows)))
    axis.set_title("A. Independent labels for each reviewed record", loc="left", pad=12)
    axis.set_xlim(-0.1, 2.1)
    axis.set_ylim(len(primary_rows) - 0.5, -0.8)
    axis.set_xticks(
        (0.48, 1.52),
        ("Researcher review", "Independent review"),
        fontsize=8.5,
    )
    axis.xaxis.tick_top()
    axis.tick_params(axis="x", length=0, pad=8)
    labels = [f"{record.case_id}\n{_plain_role(record.selection_role)}" for record in primary_rows]
    axis.set_yticks(positions, labels, fontsize=8)
    axis.tick_params(axis="y", length=0)
    for index, primary_record in enumerate(primary_rows):
        secondary_record = secondary_by_case[primary_record.case_id]
        for column, record in ((0, primary_record), (1, secondary_record)):
            x = 0.48 + column * 1.04
            colour = _category_colour(record.category)
            axis.scatter(x, index, s=430, marker="s", color=colour, zorder=2)
            axis.text(
                x,
                index,
                _category_code(record.category),
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="white"
                if record.category is not FailureReviewLabel.NO_FAILURE_OBSERVED
                else _INK,
                zorder=3,
            )
        axis.plot((0.62, 1.38), (index, index), color=_LIGHT_GREY, linewidth=1, zorder=1)
    for spine in axis.spines.values():
        spine.set_visible(False)


def _plot_category_counts(axis: Axes, report: FailureReviewReliabilityReport) -> None:
    occupied = tuple(
        row for row in report.category_counts if row.primary_count or row.secondary_count
    )
    positions = list(range(len(occupied)))
    labels = tuple(_PLAIN_CATEGORY[row.category] for row in occupied)
    values = [row.primary_count for row in occupied]
    colours = [
        _ORANGE if row.category is FailureReviewLabel.ITEM_AMBIGUITY_OR_TEST_DEFECT else _GREEN
        for row in occupied
    ]
    bars = axis.barh(positions, values, height=0.52, color=colours)
    axis.set_title("B. Categories assigned by both reviewers", loc="left", pad=10)
    axis.set_xlabel("Number of reviewed records")
    axis.set_yticks(positions, labels, fontsize=8.5)
    axis.set_xlim(0, max(row.primary_count for row in occupied) + 1)
    axis.invert_yaxis()
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            value + 0.08,
            bar.get_y() + bar.get_height() / 2,
            str(value),
            va="center",
            fontsize=9,
            color=_INK,
        )
    _style_axis(axis)


def _plot_agreement(axis: Axes, report: FailureReviewReliabilityReport) -> None:
    axis.set_title("C. Reviewer consistency", loc="left", pad=10)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.text(
        0.02,
        0.72,
        f"{report.agreement_count}/{report.item_count}",
        fontsize=26,
        fontweight="bold",
        color=_GREEN,
    )
    axis.text(0.02, 0.55, "records received the same final category", fontsize=9.5, color=_INK)
    kappa = "undefined" if report.cohen_kappa is None else f"{report.cohen_kappa:.2f}"
    axis.text(0.02, 0.32, f"Raw agreement: {report.raw_agreement:.2f}", fontsize=10, color=_INK)
    axis.text(0.02, 0.17, f"Unweighted Cohen's kappa: {kappa}", fontsize=10, color=_INK)
    axis.text(0.02, 0.00, "Denominator: six selected records", fontsize=9, color=_ORANGE)


def _validate_sources(
    *,
    primary: FailureReviewRecordReport,
    secondary: FailureReviewRecordReport,
    plan: FailureReviewReliabilityPlan,
    reliability: FailureReviewReliabilityReport,
    lock_hash: Sha256,
    primary_file_hash: Sha256,
    secondary_file_hash: Sha256,
) -> None:
    if reliability.analysis_plan_hash != plan.plan_hash:
        raise FailureFigureError("Reliability report and plan differ")
    expected_plan_sources = (
        primary.report_hash,
        secondary.report_hash,
        primary_file_hash,
        secondary_file_hash,
        lock_hash,
    )
    observed_plan_sources = (
        plan.primary_review_report_hash,
        plan.secondary_review_report_hash,
        plan.primary_review_file_hash,
        plan.secondary_review_file_hash,
        plan.analysis_pixi_lock_hash,
    )
    if observed_plan_sources != expected_plan_sources:
        raise FailureFigureError(
            "Reliability plan does not match the supplied reviews or Pixi lock"
        )
    if (
        primary.packet_hash != secondary.packet_hash
        or primary.taxonomy_hash != secondary.taxonomy_hash
    ):
        raise FailureFigureError("Failure reviews use different packet provenance")
    if (plan.packet_hash, plan.taxonomy_hash) != (primary.packet_hash, primary.taxonomy_hash):
        raise FailureFigureError("Reliability plan uses different review provenance")
    first = {record.case_id: record for record in primary.records}
    second = {record.case_id: record for record in secondary.records}
    if set(first) != set(second):
        raise FailureFigureError("Failure reviews contain different cases")
    pairs = tuple((first[case_id], second[case_id]) for case_id in sorted(first))
    if tuple(report_id for report_id in reliability.rater_ids) != (
        primary.rater_id,
        secondary.rater_id,
    ):
        raise FailureFigureError("Reliability report uses different reviewers")
    agreements = sum(left.category is right.category for left, right in pairs)
    if (agreements, len(pairs) - agreements) != (
        reliability.agreement_count,
        reliability.disagreement_count,
    ):
        raise FailureFigureError("Reliability counts do not match the review records")
    primary_counts = Counter(left.category for left, _ in pairs)
    secondary_counts = Counter(right.category for _, right in pairs)
    for row in reliability.category_counts:
        if (row.primary_count, row.secondary_count) != (
            primary_counts[row.category],
            secondary_counts[row.category],
        ):
            raise FailureFigureError("Reliability category counts do not match the reviews")
    for left, right in pairs:
        if (
            left.selection_role != right.selection_role
            or left.matched_case_id != right.matched_case_id
            or left.evidence_references != right.evidence_references
        ):
            raise FailureFigureError("Reviewers did not receive the same visible record")


def _plain_role(value: str) -> str:
    return "Apparent regression" if value == "primary_failure" else "Matched success"


def _category_code(category: FailureReviewLabel) -> str:
    if category is FailureReviewLabel.ITEM_AMBIGUITY_OR_TEST_DEFECT:
        return "TD"
    if category is FailureReviewLabel.NO_FAILURE_OBSERVED:
        return "OK"
    return "OT"


def _category_colour(category: FailureReviewLabel) -> str:
    if category is FailureReviewLabel.ITEM_AMBIGUITY_OR_TEST_DEFECT:
        return _ORANGE
    if category is FailureReviewLabel.NO_FAILURE_OBSERVED:
        return _PALE_GREY
    return _GREY


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
        existing = _load(
            manifest_path,
            FailureFigureManifest,
            "failure-review figure manifest",
        )
        return existing.generated_at_utc
    return datetime.now(UTC)


def _require_utc(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("Failure-review figure timestamp must be timezone-aware UTC")


def _load[ModelT: ContractModel](path: Path, model: type[ModelT], label: str) -> ModelT:
    try:
        return model.model_validate_json(path.read_bytes())
    except (OSError, ValidationError) as error:
        raise FailureFigureError(f"Could not verify {label}: {path}") from error
