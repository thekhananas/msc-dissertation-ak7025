"""Publication figure for the development acquisition ablation."""

# pyright: reportArgumentType=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from textwrap import fill

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from pydantic import ValidationError

from socratic_tutor.acquisition_study.ablation_analysis import (
    DevelopmentAblationAnalysisReport,
)
from socratic_tutor.acquisition_study.ablations import AblationPolicyCellId
from socratic_tutor.benchmark.artifacts import (
    artifact_locations,
    write_immutable_bytes,
    write_immutable_json,
)
from socratic_tutor.benchmark.hashing import canonical_sha256, file_sha256
from socratic_tutor.publication.figure_style import (
    BLUE,
    GREEN,
    INK,
    LIGHT_GREY,
    MUTED,
    ORANGE,
    FigureProfile,
    style_for,
)
from socratic_tutor.repository_state import current_clean_revision

_CELL_LAYOUT = (
    (
        AblationPolicyCellId.FULL_CANDIDATE,
        "Conservative reliability",
        "Limited update",
    ),
    (
        AblationPolicyCellId.LOWER_QUANTILE_UNBOUNDED,
        "Conservative reliability",
        "Unrestricted update",
    ),
    (
        AblationPolicyCellId.POSTERIOR_MEAN_BOUNDED,
        "Average reliability",
        "Limited update",
    ),
    (
        AblationPolicyCellId.POSTERIOR_MEAN_UNBOUNDED,
        "Average reliability",
        "Unrestricted update",
    ),
)
_TITLE = "The two safeguards did not improve prediction together in development"
_SUBTITLE = (
    "Mean classification error across six held-out simulated settings at the 50% probe budget; "
    "lower is better."
)
_FOOTNOTE = (
    "Development diagnostic based on 300 paired simulated episodes. Settings receive equal weight. "
    "This hand-specified simulator does not measure human learning or tutoring quality."
)
_STYLE: dict[str, object] = {
    "figure.facecolor": "white",
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans", "sans-serif"],
    "pdf.fonttype": 42,
    "savefig.facecolor": "white",
    "svg.fonttype": "none",
    "svg.hashsalt": "socratic-tutor-acquisition-ablation-v1",
    "text.color": INK,
}


class AcquisitionAblationFigureError(ValueError):
    """The development report cannot support the ablation figure."""


@dataclass(frozen=True, slots=True)
class AblationCell:
    cell_id: str
    reliability_label: str
    update_label: str
    classification_error: float
    difference_from_candidate: float
    brier_score: float
    episode_count: int


def render_acquisition_ablation_figure(
    *,
    ablation_report_path: Path,
    pixi_lock_path: Path,
    output_root: Path,
    publication_code_revision: str,
    generated_at_utc: datetime | None = None,
) -> dict[str, object]:
    """Publish report and presentation figures from one verified diagnostic report."""

    report_bytes = _read(ablation_report_path, "development ablation report")
    lock_bytes = _read(pixi_lock_path, "Pixi lock")
    report = _load_report(report_bytes)
    cells = _cells(report)
    manifest_path = output_root / "acquisition_ablation_figure_manifest.json"
    generated_at = _resolve_generated_at(generated_at_utc, manifest_path)
    source_data = _source_data_csv(cells, report)

    generated_files: list[dict[str, object]] = []
    vector_files: list[tuple[str, bytes]] = []
    for profile in ("report", "presentation"):
        pdf, svg = _render_vector_files(
            cells,
            report,
            profile=profile,
        )
        for file_format, file_content in (("pdf", pdf), ("svg", svg)):
            name = f"acquisition_ablation_{profile}.{file_format}"
            vector_files.append((name, file_content))
            generated_files.append(
                {
                    "profile": profile,
                    "format": file_format,
                    "file": name,
                    "sha256": file_sha256(file_content),
                }
            )

    manifest_content: dict[str, object] = {
        "schema_version": 1,
        "schema_id": "acquisition_study.ablation_figure_manifest.v1",
        "study_id": report.study_id,
        "split": report.split,
        "figure_id": "FIG-M7B-03",
        "result_status": report.result_status,
        "ablation_report_hash": report.report_hash,
        "ablation_report_sha256": file_sha256(report_bytes),
        "pixi_lock_sha256": file_sha256(lock_bytes),
        "publication_code_revision": _validate_revision(publication_code_revision),
        "files": generated_files,
        "source_data_file": "acquisition_ablation_figure_data.csv",
        "source_data_sha256": file_sha256(source_data),
        "generated_at_utc": generated_at.isoformat().replace("+00:00", "Z"),
        "canonical_claim_allowed": False,
        "human_learning_claim_supported": False,
        "tutoring_efficacy_claim_supported": False,
    }
    manifest = {
        **manifest_content,
        "manifest_hash": canonical_sha256(manifest_content),
    }
    for name, data in vector_files:
        write_immutable_bytes(output_root / name, data)
    write_immutable_bytes(output_root / str(manifest_content["source_data_file"]), source_data)
    write_immutable_json(manifest_path, manifest)
    return manifest


def _load_report(content: bytes) -> DevelopmentAblationAnalysisReport:
    try:
        report = DevelopmentAblationAnalysisReport.model_validate_json(content)
    except ValidationError as error:
        raise AcquisitionAblationFigureError("Development ablation report is invalid") from error
    if report.canonical_claim_allowed:
        raise AcquisitionAblationFigureError("A development ablation cannot be canonical")
    return report


def _cells(report: DevelopmentAblationAnalysisReport) -> tuple[AblationCell, ...]:
    summaries = {row.cell_id: row for row in report.held_out_cell_summaries}
    candidate_error = summaries[AblationPolicyCellId.FULL_CANDIDATE].macro_mean_classification_error
    cells = tuple(
        AblationCell(
            cell_id=cell_id.value,
            reliability_label=reliability_label,
            update_label=update_label,
            classification_error=summaries[cell_id].macro_mean_classification_error,
            difference_from_candidate=(
                summaries[cell_id].macro_mean_classification_error - candidate_error
            ),
            brier_score=summaries[cell_id].macro_mean_brier_score,
            episode_count=(
                summaries[cell_id].environment_count * summaries[cell_id].episodes_per_environment
            ),
        )
        for cell_id, reliability_label, update_label in _CELL_LAYOUT
    )
    if len(summaries) != 4 or any(cell.episode_count != 300 for cell in cells):
        raise AcquisitionAblationFigureError("Ablation cells do not cover the planned workload")
    return cells


def _source_data_csv(
    cells: tuple[AblationCell, ...],
    report: DevelopmentAblationAnalysisReport,
) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        (
            "cell_id",
            "reliability_estimate",
            "update_rule",
            "held_out_classification_error",
            "difference_from_full_candidate",
            "held_out_brier_score",
            "paired_episode_count",
            "interaction_difference",
            "evidence_status",
        )
    )
    for cell in cells:
        writer.writerow(
            (
                cell.cell_id,
                cell.reliability_label,
                cell.update_label,
                f"{cell.classification_error:.12f}",
                f"{cell.difference_from_candidate:.12f}",
                f"{cell.brier_score:.12f}",
                cell.episode_count,
                f"{report.held_out_factorial_interaction.interaction_difference:.12f}",
                "development_diagnostic_not_canonical",
            )
        )
    return output.getvalue().encode("utf-8")


def _render_vector_files(
    cells: tuple[AblationCell, ...],
    report: DevelopmentAblationAnalysisReport,
    *,
    profile: FigureProfile,
) -> tuple[bytes, bytes]:
    style = style_for(profile)
    if profile == "report":
        matrix_bounds = (0.22, 0.18, 0.46, 0.52)
        notes_bounds = (0.72, 0.18, 0.25, 0.52)
        note_line_width = 34
        footnote_line_width = 112
    else:
        matrix_bounds = (0.18, 0.18, 0.50, 0.52)
        notes_bounds = (0.72, 0.18, 0.25, 0.52)
        note_line_width = 42
        footnote_line_width = 158

    with mpl.rc_context(_STYLE):
        figure = Figure(figsize=(style.width_in, style.height_in), layout="none")
        axis = figure.add_axes(matrix_bounds)
        notes = figure.add_axes(notes_bounds)
        _draw_matrix(axis, cells, profile)
        _draw_notes(notes, cells, report, profile, line_width=note_line_width)
        figure.text(
            0.055,
            0.95,
            _TITLE,
            fontsize=style.title_size,
            fontweight="bold",
            ha="left",
            va="top",
        )
        figure.text(
            0.055,
            0.885,
            _SUBTITLE,
            fontsize=style.font_size,
            color=MUTED,
            ha="left",
            va="top",
            wrap=True,
        )
        figure.text(
            0.055,
            0.095,
            fill(_FOOTNOTE, width=footnote_line_width, break_on_hyphens=False),
            fontsize=style.font_size * 0.86,
            color=MUTED,
            ha="left",
            va="top",
        )
        return _save(figure, "pdf"), _save(figure, "svg")


def _draw_matrix(axis: Axes, cells: tuple[AblationCell, ...], profile: FigureProfile) -> None:
    style = style_for(profile)
    lookup = {(cell.reliability_label, cell.update_label): cell for cell in cells}
    rows = ("Conservative reliability", "Average reliability")
    columns = ("Limited update", "Unrestricted update")
    candidate_error = lookup[(rows[0], columns[0])].classification_error

    axis.set_xlim(0, 2)
    axis.set_ylim(0, 2)
    axis.invert_yaxis()
    axis.set_xticks((0.5, 1.5), ("Limited evidence\nupdate", "Unrestricted evidence\nupdate"))
    axis.set_yticks(
        (0.5, 1.5),
        ("Conservative reliability\nestimate", "Average reliability\nestimate"),
    )
    axis.tick_params(axis="both", length=0, labelsize=style.font_size)
    axis.tick_params(axis="x", pad=8)
    axis.tick_params(axis="y", pad=8)
    axis.xaxis.tick_top()

    for row_index, reliability in enumerate(rows):
        for column_index, update in enumerate(columns):
            cell = lookup[(reliability, update)]
            is_candidate = cell.cell_id == AblationPolicyCellId.FULL_CANDIDATE.value
            difference = cell.classification_error - candidate_error
            edge = BLUE if is_candidate else GREEN if difference < 0 else ORANGE
            fill = "#E9F4FA" if is_candidate else "#EEF7F3" if difference < 0 else "#FBEFE9"
            axis.add_patch(
                Rectangle(
                    (column_index + 0.04, row_index + 0.04),
                    0.92,
                    0.92,
                    facecolor=fill,
                    edgecolor=edge,
                    linewidth=2.0 if is_candidate else 1.4,
                )
            )
            axis.text(
                column_index + 0.5,
                row_index + 0.43,
                f"{cell.classification_error:.2%}",
                fontsize=style.font_size * 1.55,
                fontweight="bold",
                ha="center",
                va="center",
                color=INK,
            )
            label = "Proposed method" if is_candidate else _difference_label(difference)
            axis.text(
                column_index + 0.5,
                row_index + 0.68,
                label,
                fontsize=style.font_size * 0.82,
                ha="center",
                va="center",
                color=edge,
            )
    for spine in axis.spines.values():
        spine.set_visible(False)


def _draw_notes(
    axis: Axes,
    cells: tuple[AblationCell, ...],
    report: DevelopmentAblationAnalysisReport,
    profile: FigureProfile,
    *,
    line_width: int,
) -> None:
    style = style_for(profile)
    lookup = {(cell.reliability_label, cell.update_label): cell for cell in cells}
    candidate = lookup[("Conservative reliability", "Limited update")]
    average_bounded = lookup[("Average reliability", "Limited update")]
    conservative_unbounded = lookup[("Conservative reliability", "Unrestricted update")]
    interaction = report.held_out_factorial_interaction.interaction_difference

    axis.set_axis_off()
    axis.add_patch(
        Rectangle(
            (0, 0),
            1,
            1,
            transform=axis.transAxes,
            facecolor="#F7F8F9",
            edgecolor=LIGHT_GREY,
            linewidth=1.0,
        )
    )
    axis.text(
        0.08,
        0.91,
        "What this shows",
        fontsize=style.font_size * 1.05,
        fontweight="bold",
        transform=axis.transAxes,
        va="top",
    )
    notes = (
        (
            "Reliability caution",
            _component_sentence(
                candidate.classification_error - average_bounded.classification_error,
                "when updates were limited",
            ),
        ),
        (
            "Update limit",
            _component_sentence(
                candidate.classification_error - conservative_unbounded.classification_error,
                "with the conservative estimate",
            ),
        ),
        (
            "Joint effect",
            (
                f"The interaction was {abs(interaction) * 100:.02f} percentage points; "
                "descriptive only."
            ),
        ),
    )
    y = 0.77
    for heading, sentence in notes:
        axis.text(
            0.08,
            y,
            heading,
            fontsize=style.font_size * 0.9,
            fontweight="bold",
            transform=axis.transAxes,
            va="top",
        )
        axis.text(
            0.08,
            y - 0.075,
            fill(sentence, width=line_width),
            fontsize=style.font_size * 0.82,
            color=MUTED,
            transform=axis.transAxes,
            va="top",
        )
        y -= 0.255


def _difference_label(difference: float) -> str:
    if abs(difference) < 0.00005:
        return "No change from proposed method"
    direction = "lower" if difference < 0 else "higher"
    return f"{abs(difference) * 100:.2f} points {direction}"


def _component_sentence(effect: float, context: str) -> str:
    direction = "increased" if effect > 0 else "reduced"
    return f"It {direction} error by {abs(effect) * 100:.2f} percentage points {context}."


def _save(figure: Figure, file_format: str) -> bytes:
    output = BytesIO()
    figure.savefig(output, format=file_format, metadata={"Creator": "Matplotlib"})
    return output.getvalue()


def _read(path: Path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise AcquisitionAblationFigureError(f"Could not read {label}: {path}") from error


def _resolve_generated_at(value: datetime | None, manifest_path: Path) -> datetime:
    if value is not None:
        _require_utc(value)
        return value
    if not manifest_path.exists():
        return datetime.now(UTC)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        stored_hash = manifest.pop("manifest_hash")
        if canonical_sha256(manifest) != stored_hash:
            raise ValueError("manifest hash mismatch")
        generated_at = datetime.fromisoformat(
            str(manifest["generated_at_utc"]).replace("Z", "+00:00")
        )
        _require_utc(generated_at)
        return generated_at
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise AcquisitionAblationFigureError(
            f"Could not verify existing ablation figure manifest: {manifest_path}"
        ) from error


def _require_utc(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("Ablation figure timestamp must be timezone-aware UTC")


def _validate_revision(value: str) -> str:
    if re.fullmatch(r"[0-9a-f]{7,40}", value) is None:
        raise AcquisitionAblationFigureError("Publication revision must be a Git SHA")
    return value


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ABLATION_REPORT = (
    PROJECT_ROOT
    / "artifacts"
    / "acquisition-study"
    / "development-aede5af"
    / "ablations-bfdf385"
    / "analysis-bfdf385"
    / "development_ablation_analysis_report.json"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "acquisition-study"
    / "development-aede5af"
    / "publication-ablation-v1"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish the development acquisition ablation figure"
    )
    parser.add_argument("--ablation-report", type=Path, default=DEFAULT_ABLATION_REPORT)
    parser.add_argument("--pixi-lock", type=Path, default=PROJECT_ROOT / "pixi.lock")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--generated-at-utc", type=_utc_datetime)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = render_acquisition_ablation_figure(
            ablation_report_path=args.ablation_report,
            pixi_lock_path=args.pixi_lock,
            output_root=args.output_root,
            publication_code_revision=current_clean_revision(
                PROJECT_ROOT,
                dirty_message="Commit or remove all visible changes before publishing the figure",
                untracked_files="all",
                required_branch="main",
                operation_name="Figure publication",
                invalid_revision_message="Publication revision must be a Git SHA",
                invalid_revision_error_factory=AcquisitionAblationFigureError,
            ),
            generated_at_utc=args.generated_at_utc,
        )
    except Exception as error:
        print(
            json.dumps(
                {
                    "command": "acquisition-development-ablation-figure",
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
                "artifact_locations": artifact_locations(args.output_root),
                "command": "acquisition-development-ablation-figure",
                "result": manifest,
                "status": "ok",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _utc_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    _require_utc(parsed)
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
