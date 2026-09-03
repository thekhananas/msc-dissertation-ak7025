# pyright: reportArgumentType=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
"""Publication-safe table and figure for the CSEDM companion study."""

from __future__ import annotations

import argparse
import json
import sys
from io import BytesIO
from pathlib import Path
from typing import Literal, Self

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
from pydantic import Field, model_validator

from socratic_tutor.benchmark.artifacts import (
    artifact_locations,
    write_immutable_bytes,
    write_immutable_json,
)
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import canonical_sha256, file_sha256
from socratic_tutor.contracts import ContractModel
from socratic_tutor.csedm_study.analysis import CSEDMAnalysisManifest, CSEDMAnalysisReport
from socratic_tutor.csedm_study.features import CSEDMAdapterError
from socratic_tutor.csedm_study.models import (
    LOGISTIC_LEARNER_HISTORY,
    TRAINING_LABEL_PREVALENCE,
    TRAINING_PROBLEM_FREQUENCY,
)
from socratic_tutor.csv_output import csv_bytes
from socratic_tutor.publication.figure_style import (
    BLUE,
    GREEN,
    GREY,
    INK,
    LIGHT_GREY,
    MUTED,
    REPORT_STYLE,
)
from socratic_tutor.repository_state import current_clean_revision

_MODEL_NAMES = {
    TRAINING_LABEL_PREVALENCE: "Overall success rate",
    TRAINING_PROBLEM_FREQUENCY: "Problem success rate",
    LOGISTIC_LEARNER_HISTORY: "Learner-history logistic model",
}
_STYLE: dict[str, object] = {
    "axes.edgecolor": LIGHT_GREY,
    "axes.labelcolor": INK,
    "axes.linewidth": 0.8,
    "axes.titlecolor": INK,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "figure.facecolor": "white",
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans", "sans-serif"],
    "font.size": 9.0,
    "pdf.fonttype": 42,
    "savefig.facecolor": "white",
    "svg.fonttype": "none",
    "svg.hashsalt": "socratic-tutor-csedm-publication-v1",
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
}
_FORBIDDEN_PUBLIC_MARKERS = (
    b"learner_id",
    b"subjectid",
    b"source_code",
    b"codestateid",
    b"target_id",
)


class CSEDMPublicationError(ValueError):
    """Publication inputs or outputs are missing, inconsistent, or unsafe."""


class CSEDMPublicationManifest(ContractModel):
    """Hashes and claim boundaries for the public M7C evidence package."""

    schema_id: Literal["csedm.publication_manifest.v1"] = "csedm.publication_manifest.v1"
    schema_version: Literal[1] = 1
    study_id: Literal["csedm-uncertainty-triage-v1"] = "csedm-uncertainty-triage-v1"
    publication_code_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    pixi_lock_sha256: Sha256
    analysis_manifest_sha256: Sha256
    analysis_manifest_hash: Sha256
    analysis_report_sha256: Sha256
    analysis_report_hash: Sha256
    model_table_csv_sha256: Sha256
    model_table_tex_sha256: Sha256
    figure_source_csv_sha256: Sha256
    figure_pdf_sha256: Sha256
    figure_svg_sha256: Sha256
    prediction_target_count: Literal[729]
    learner_count: Literal[86]
    problem_count: Literal[19]
    public_source: Literal["aggregate_analysis_report_only"]
    network_calls: Literal[0]
    sandbox_calls: Literal[0]
    public_artifacts_contain_learner_identifiers: Literal[False]
    public_artifacts_contain_raw_events: Literal[False]
    public_artifacts_contain_source_code: Literal[False]
    executable_probe_claim_supported: Literal[False]
    tutoring_effect_claim_supported: Literal[False]
    human_learning_claim_supported: Literal[False]
    manifest_hash: Sha256

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        expected = canonical_sha256(self.model_dump(mode="json", exclude={"manifest_hash"}))
        if self.manifest_hash != expected:
            raise ValueError("CSEDM publication manifest hash does not match its content")
        return self


def publish_csedm_results(
    *,
    analysis_root: Path,
    pixi_lock_path: Path,
    output_root: Path,
    code_revision: str,
) -> CSEDMPublicationManifest:
    """Build the public table, figure, and manifest from aggregate results only."""

    analysis_manifest_path = analysis_root / "analysis_manifest.json"
    analysis_report_path = analysis_root / "uncertainty_analysis_report.json"
    analysis_manifest_bytes = _read(analysis_manifest_path, "analysis manifest")
    analysis_report_bytes = _read(analysis_report_path, "analysis report")
    analysis_manifest = CSEDMAnalysisManifest.model_validate_json(analysis_manifest_bytes)
    report = CSEDMAnalysisReport.model_validate_json(analysis_report_bytes)
    pixi_lock_hash = file_sha256(_read(pixi_lock_path, "Pixi lock"))
    _validate_sources(
        manifest=analysis_manifest,
        report=report,
        report_bytes=analysis_report_bytes,
        pixi_lock_hash=pixi_lock_hash,
    )

    table_csv = _model_table_csv(report)
    table_tex = _model_table_tex(report)
    source_csv = _figure_source_csv(report)
    figure_pdf, figure_svg = _render_figure(report)
    public_files = (table_csv, table_tex, source_csv, figure_pdf, figure_svg)
    _check_public_payloads(public_files)

    write_immutable_bytes(output_root / "model_comparison.csv", table_csv)
    write_immutable_bytes(output_root / "model_comparison.tex", table_tex)
    write_immutable_bytes(output_root / "uncertainty_error_review_source.csv", source_csv)
    write_immutable_bytes(output_root / "uncertainty_error_review.pdf", figure_pdf)
    write_immutable_bytes(output_root / "uncertainty_error_review.svg", figure_svg)

    payload = {
        "schema_id": "csedm.publication_manifest.v1",
        "schema_version": 1,
        "study_id": report.study_id,
        "publication_code_revision": code_revision,
        "pixi_lock_sha256": pixi_lock_hash,
        "analysis_manifest_sha256": file_sha256(analysis_manifest_bytes),
        "analysis_manifest_hash": analysis_manifest.manifest_hash,
        "analysis_report_sha256": file_sha256(analysis_report_bytes),
        "analysis_report_hash": report.report_hash,
        "model_table_csv_sha256": file_sha256(table_csv),
        "model_table_tex_sha256": file_sha256(table_tex),
        "figure_source_csv_sha256": file_sha256(source_csv),
        "figure_pdf_sha256": file_sha256(figure_pdf),
        "figure_svg_sha256": file_sha256(figure_svg),
        "prediction_target_count": report.prediction_target_count,
        "learner_count": report.learner_count,
        "problem_count": report.problem_count,
        "public_source": "aggregate_analysis_report_only",
        "network_calls": 0,
        "sandbox_calls": 0,
        "public_artifacts_contain_learner_identifiers": False,
        "public_artifacts_contain_raw_events": False,
        "public_artifacts_contain_source_code": False,
        "executable_probe_claim_supported": False,
        "tutoring_effect_claim_supported": False,
        "human_learning_claim_supported": False,
    }
    payload["manifest_hash"] = canonical_sha256(payload)
    manifest = CSEDMPublicationManifest.model_validate(payload)
    write_immutable_json(output_root / "publication_manifest.json", manifest)
    return manifest


def _validate_sources(
    *,
    manifest: CSEDMAnalysisManifest,
    report: CSEDMAnalysisReport,
    report_bytes: bytes,
    pixi_lock_hash: Sha256,
) -> None:
    if manifest.analysis_report_sha256 != file_sha256(report_bytes):
        raise CSEDMPublicationError("Analysis report differs from its manifest")
    if manifest.report_content_hash != report.report_hash:
        raise CSEDMPublicationError("Analysis report content hash differs from its manifest")
    if manifest.pixi_lock_sha256 != pixi_lock_hash:
        raise CSEDMPublicationError("Analysis and publication use different Pixi locks")
    if (
        report.executable_probe_claim_supported
        or report.tutoring_effect_claim_supported
        or report.human_learning_claim_supported
    ):
        raise CSEDMPublicationError("Analysis report exceeds the M7C claim boundary")


def _model_table_csv(report: CSEDMAnalysisReport) -> bytes:
    return csv_bytes(
        (
            "model",
            "predictions",
            "errors",
            "accuracy",
            "brier_score",
            "log_loss",
            "calibration_error",
        ),
        (
            (
                _MODEL_NAMES[row.model_id],
                row.prediction_count,
                row.error_count,
                row.accuracy,
                row.brier_score,
                row.log_loss,
                row.expected_calibration_error,
            )
            for row in report.model_metrics
        ),
    )


def _model_table_tex(report: CSEDMAnalysisReport) -> bytes:
    rows = "\n".join(
        f"{_MODEL_NAMES[row.model_id]} & {row.error_count} & {row.accuracy:.3f} & "
        f"{row.brier_score:.3f} & {row.log_loss:.3f} & "
        f"{row.expected_calibration_error:.3f} \\\\"
        for row in report.model_metrics
    )
    content = f"""% Generated from the frozen aggregate CSEDM analysis report.
\\begin{{table}}[tb]
\\centering
\\caption{{Prediction results on 729 historical targets from 86 learners and 19 Python problems.
Each learner was evaluated using models fitted on other learners. Higher accuracy and lower errors, Brier score,
log loss and calibration error indicate better predictions. Values are descriptive estimates without intervals.}}
\\label{{tab:csedm-model-comparison}}
\\begin{{tabular}}{{lrrrrr}}
\\toprule
Model & Errors & Accuracy & Brier & Log loss & Cal. error \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}
\\begin{{minipage}}{{0.96\\linewidth}}
\\footnotesize Historical observations were separated by learner. These figures describe
prediction quality; they do not measure executable-probe benefit, tutoring effects, or learning.
\\end{{minipage}}
\\end{{table}}
"""
    return content.encode("ascii")


def _figure_source_csv(report: CSEDMAnalysisReport) -> bytes:
    primary = report.primary
    return csv_bytes(
        (
            "measure",
            "estimate",
            "interval_lower",
            "interval_upper",
            "selected_errors",
            "total_errors",
        ),
        (
            (
                "random_review_expectation",
                primary.expected_random_captured_error_recall,
                "",
                "",
                primary.expected_random_selected_error_count,
                primary.total_error_count,
            ),
            (
                "uncertainty_ranked_review",
                primary.captured_error_recall,
                "",
                "",
                primary.selected_error_count,
                primary.total_error_count,
            ),
            (
                "uncertainty_minus_random",
                primary.captured_error_recall_difference,
                primary.interval_lower,
                primary.interval_upper,
                "",
                "",
            ),
        ),
    )


def _render_figure(report: CSEDMAnalysisReport) -> tuple[bytes, bytes]:
    with mpl.rc_context(_STYLE):
        figure = _build_figure(report)
        pdf_buffer = BytesIO()
        svg_buffer = BytesIO()
        figure.savefig(
            pdf_buffer,
            format="pdf",
            metadata={"Creator": "Matplotlib", "CreationDate": None},
        )
        figure.savefig(
            svg_buffer,
            format="svg",
            metadata={"Creator": "Matplotlib", "Date": None},
        )
        return pdf_buffer.getvalue(), svg_buffer.getvalue()


def _build_figure(report: CSEDMAnalysisReport) -> Figure:
    figure = Figure(figsize=(REPORT_STYLE.width_in, 4.65))
    grid = figure.add_gridspec(
        1,
        2,
        left=0.175,
        right=0.97,
        bottom=0.24,
        top=0.70,
        width_ratios=(1.45, 1.0),
        wspace=0.52,
    )
    capture_axis = figure.add_subplot(grid[0, 0])
    difference_axis = figure.add_subplot(grid[0, 1])

    figure.text(
        0.055,
        0.94,
        "Does uncertainty help find prediction errors?",
        fontsize=12.0,
        fontweight="bold",
        ha="left",
        va="top",
    )
    figure.text(
        0.055,
        0.865,
        "Both approaches select 361 of 729 predictions, using the same fold-level allocation.",
        fontsize=9.5,
        color=MUTED,
        ha="left",
        va="top",
    )
    _draw_capture_panel(capture_axis, report)
    _draw_difference_panel(difference_axis, report)
    figure.text(
        0.055,
        0.115,
        "Historical observations: 729 predictions from 86 learners across 19 Python problems.",
        fontsize=8.4,
        color=INK,
        ha="left",
        va="top",
    )
    figure.text(
        0.055,
        0.065,
        "No corrective review, executable-probe intervention or learning gain was tested.",
        fontsize=8.4,
        color=MUTED,
        ha="left",
        va="top",
    )
    return figure


def _draw_capture_panel(axis: Axes, report: CSEDMAnalysisReport) -> None:
    primary = report.primary
    values = (primary.expected_random_captured_error_recall, primary.captured_error_recall)
    labels = ("Random selection\n(expected)", "Uncertainty\nranking")
    bars = axis.barh((1, 0), values, color=(GREY, BLUE), height=0.52)
    axis.set_title("Share of the model's errors found", loc="left", pad=10)
    axis.set_xlim(0.0, 0.82)
    axis.set_yticks((1, 0), labels)
    axis.set_xlabel("Errors found")
    axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.set_xticks((0.0, 0.2, 0.4, 0.6, 0.8))
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            value + 0.018,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1%}",
            va="center",
            ha="left",
            fontsize=9.2,
            fontweight="bold",
            color=INK,
        )
    _plain_axis(axis)


def _draw_difference_panel(axis: Axes, report: CSEDMAnalysisReport) -> None:
    primary = report.primary
    difference = primary.captured_error_recall_difference
    lower = primary.interval_lower
    upper = primary.interval_upper
    axis.axvline(0.0, color=GREY, linewidth=1.0, linestyle="--")
    axis.errorbar(
        difference,
        0.5,
        xerr=((difference - lower,), (upper - difference,)),
        fmt="o",
        color=GREEN,
        ecolor=GREEN,
        elinewidth=2.2,
        capsize=5,
        markersize=7,
    )
    axis.set_title("Gain over random selection", loc="left", pad=10)
    axis.set_xlim(-0.04, 0.34)
    axis.set_ylim(0.0, 1.0)
    axis.set_yticks(())
    axis.set_xlabel("Difference in errors found\n(percentage points)")
    axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0, symbol=""))
    axis.set_xticks((0.0, 0.1, 0.2, 0.3))
    axis.text(
        difference,
        0.72,
        f"{difference * 100:+.1f} points",
        ha="center",
        va="bottom",
        fontsize=10.0,
        fontweight="bold",
        color=INK,
    )
    axis.text(
        0.15,
        0.14,
        f"95% learner interval\n{lower * 100:.1f} to {upper * 100:.1f} points",
        ha="center",
        va="bottom",
        fontsize=8.4,
        color=MUTED,
    )
    _plain_axis(axis)


def _plain_axis(axis: Axes) -> None:
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    axis.grid(axis="x", color=LIGHT_GREY, linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)


def _check_public_payloads(payloads: tuple[bytes, ...]) -> None:
    lowered = b"\n".join(payloads).lower()
    leaked = [marker.decode("ascii") for marker in _FORBIDDEN_PUBLIC_MARKERS if marker in lowered]
    if leaked:
        raise CSEDMPublicationError(f"Public publication contains restricted fields: {leaked}")


def _read(path: Path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise CSEDMPublicationError(f"Could not read {label}: {path}") from error


def build_parser(project_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish the CSEDM result table and figure")
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--pixi-lock", type=Path, default=project_root / "pixi.lock")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--code-revision")
    return parser


def main(argv: list[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parents[3]
    args = build_parser(project_root).parse_args(argv)
    try:
        revision = args.code_revision or current_clean_revision(
            project_root,
            dirty_message="Commit tracked changes before publishing CSEDM results",
            untracked_files="no",
            operation_name="CSEDM result publication",
            error_factory=CSEDMPublicationError,
        )
        output_root = args.output_root or (
            project_root / "artifacts" / "csedm-study" / f"publication-{revision[:7]}"
        )
        manifest = publish_csedm_results(
            analysis_root=args.analysis_root,
            pixi_lock_path=args.pixi_lock,
            output_root=output_root,
            code_revision=revision,
        )
    except (CSEDMPublicationError, CSEDMAdapterError, OSError, ValueError) as error:
        print(
            json.dumps(
                {
                    "command": "csedm-publish",
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "status": "error",
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "artifact_locations": artifact_locations(output_root),
                "command": "csedm-publish",
                "result": manifest.model_dump(mode="json"),
                "status": "ok",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0
