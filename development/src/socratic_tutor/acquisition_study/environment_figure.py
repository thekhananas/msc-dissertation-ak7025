"""Environment-level figure for the canonical acquisition study."""

# pyright: reportArgumentType=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from pydantic import ValidationError

from socratic_tutor.acquisition_study.canonical_primary_analysis import (
    CanonicalPrimaryAnalysisReport,
)
from socratic_tutor.acquisition_study.contracts import PolicyId
from socratic_tutor.acquisition_study.publication_labels import COMPARATOR_LABELS
from socratic_tutor.benchmark.artifacts import (
    artifact_locations,
    write_immutable_bytes,
    write_immutable_json,
)
from socratic_tutor.benchmark.hashing import canonical_sha256, file_sha256
from socratic_tutor.filesystem import read_bytes
from socratic_tutor.publication.figure_style import (
    BLUE,
    GREY,
    INK,
    LIGHT_GREY,
    MUTED,
    ORANGE,
    FigureProfile,
    style_for,
)
from socratic_tutor.repository_state import current_clean_revision, validate_git_revision

_COMPARATORS = tuple(
    (policy_id, COMPARATOR_LABELS[policy_id], marker)
    for policy_id, marker in (
        (PolicyId.SEEDED_RANDOM_BOUNDED, "o"),
        (PolicyId.UNCERTAINTY_ONLY_BOUNDED, "s"),
        (PolicyId.PLUG_IN_EVSI_BOUNDED, "D"),
    )
)
_ENVIRONMENTS = (
    ("matched", "Evidence behaves as assumed"),
    ("lower_reliability", "Lower reliability"),
    ("asymmetric_errors", "Unequal error rates"),
    ("irrelevant_evidence", "Irrelevant evidence"),
    ("inverted_evidence", "Inverted evidence"),
    ("difficulty_missingness", "Missing on difficult cases"),
    ("correlated_family_failures", "Correlated family failures"),
    ("held_out_mean", "Held-out mean (six settings)"),
)
_TITLE = "The proposed selector beat random choice overall, but not simpler selection rules"
_SUBTITLE = (
    "Difference in prediction error at the planned 50% probe budget; "
    "values left of zero favour the proposed selector."
)
_FOOTNOTE = (
    "Setting rows show paired means; held-out intervals hold calibration fixed and adjust for "
    "three comparisons. Outlined points mark worst settings. No human learning was measured."
)
_STYLE: dict[str, object] = {
    "figure.facecolor": "white",
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans", "sans-serif"],
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "svg.fonttype": "none",
    "svg.hashsalt": "socratic-tutor-acquisition-environments-v1",
    "text.color": INK,
}


class AcquisitionEnvironmentFigureError(ValueError):
    """The canonical primary report cannot support the environment figure."""


@dataclass(frozen=True, slots=True)
class EffectPoint:
    environment_id: str
    environment_label: str
    environment_role: str
    comparator_id: str
    comparator_label: str
    marker: str
    effect: float
    interval_lower: float | None
    interval_upper: float | None
    episode_count: int
    worst_setting: bool


def render_acquisition_environment_figure(
    *,
    primary_report_path: Path,
    pixi_lock_path: Path,
    output_root: Path,
    publication_code_revision: str,
    generated_at_utc: datetime | None = None,
) -> dict[str, object]:
    """Publish report and presentation figures from the sealed primary report."""

    primary_bytes = _read(primary_report_path, "canonical primary report")
    pixi_lock_bytes = _read(pixi_lock_path, "Pixi lock")
    primary = _load_primary(primary_bytes)
    points = _effect_points(primary)
    manifest_path = output_root / "acquisition_environment_figure_manifest.json"
    generated_at = _resolve_generated_at(generated_at_utc, manifest_path)
    source_data = _source_data_csv(points)

    generated_files: list[dict[str, object]] = []
    vector_files: list[tuple[str, bytes]] = []
    for profile in ("report", "presentation"):
        pdf, svg = _render_vector_files(points, profile=profile, generated_at_utc=generated_at)
        for file_format, content in (("pdf", pdf), ("svg", svg)):
            name = f"acquisition_environment_effects_{profile}.{file_format}"
            vector_files.append((name, content))
            generated_files.append(
                {
                    "profile": profile,
                    "format": file_format,
                    "file": name,
                    "sha256": file_sha256(content),
                }
            )

    manifest_content: dict[str, object] = {
        "schema_version": 1,
        "schema_id": "acquisition_study.environment_figure_manifest.v1",
        "study_id": primary.study_id,
        "split": primary.split,
        "figure_id": "FIG-M7B-02",
        "primary_report_hash": primary.report_hash,
        "primary_report_sha256": file_sha256(primary_bytes),
        "pixi_lock_sha256": file_sha256(pixi_lock_bytes),
        "publication_code_revision": validate_git_revision(
            publication_code_revision,
            invalid_message="Publication revision must be a Git SHA",
            error_factory=AcquisitionEnvironmentFigureError,
            allow_abbreviated=True,
        ),
        "files": generated_files,
        "source_data_file": "acquisition_environment_figure_data.csv",
        "source_data_sha256": file_sha256(source_data),
        "generated_at_utc": generated_at.isoformat().replace("+00:00", "Z"),
        "individual_environment_intervals_reported": False,
        "held_out_macro_interval_reported": True,
        "human_learning_claim_supported": False,
        "tutoring_efficacy_claim_supported": False,
    }
    manifest = {**manifest_content, "manifest_hash": canonical_sha256(manifest_content)}
    for name, content in vector_files:
        write_immutable_bytes(output_root / name, content)
    write_immutable_bytes(output_root / str(manifest["source_data_file"]), source_data)
    write_immutable_json(manifest_path, manifest)
    return manifest


def _load_primary(content: bytes) -> CanonicalPrimaryAnalysisReport:
    try:
        report = CanonicalPrimaryAnalysisReport.model_validate_json(content)
    except ValidationError as error:
        raise AcquisitionEnvironmentFigureError("Canonical primary report is invalid") from error
    if report.restricted_stream_accessed or report.human_record_count:
        raise AcquisitionEnvironmentFigureError("Primary report crosses the publication boundary")
    return report


def _effect_points(report: CanonicalPrimaryAnalysisReport) -> tuple[EffectPoint, ...]:
    labels = dict(_ENVIRONMENTS)
    matched = {row.comparator_policy_id: row for row in report.matched_environment_effects}
    intervals = {row.comparator_policy_id: row for row in report.held_out_primary_intervals}
    points: list[EffectPoint] = []
    for comparator_id, comparator_label, marker in _COMPARATORS:
        matched_row = matched[comparator_id]
        points.append(
            EffectPoint(
                environment_id="matched",
                environment_label=labels["matched"],
                environment_role="matched_sanity_check",
                comparator_id=comparator_id.value,
                comparator_label=comparator_label,
                marker=marker,
                effect=matched_row.mean_paired_effect,
                interval_lower=None,
                interval_upper=None,
                episode_count=matched_row.episode_count,
                worst_setting=False,
            )
        )
        interval = intervals[comparator_id]
        for row in interval.environment_effects:
            points.append(
                EffectPoint(
                    environment_id=row.environment_id.value,
                    environment_label=labels[row.environment_id.value],
                    environment_role=row.environment_role.value,
                    comparator_id=comparator_id.value,
                    comparator_label=comparator_label,
                    marker=marker,
                    effect=row.mean_paired_effect,
                    interval_lower=None,
                    interval_upper=None,
                    episode_count=row.episode_count,
                    worst_setting=row.environment_id is interval.worst_environment_id,
                )
            )
        points.append(
            EffectPoint(
                environment_id="held_out_mean",
                environment_label=labels["held_out_mean"],
                environment_role="held_out_equal_mean",
                comparator_id=comparator_id.value,
                comparator_label=comparator_label,
                marker=marker,
                effect=interval.macro_mean_paired_effect,
                interval_lower=interval.interval_lower,
                interval_upper=interval.interval_upper,
                episode_count=interval.episodes_per_environment * len(interval.environment_effects),
                worst_setting=False,
            )
        )
    if len(points) != len(_COMPARATORS) * len(_ENVIRONMENTS):
        raise AcquisitionEnvironmentFigureError("Primary environment comparison is incomplete")
    return tuple(points)


def _source_data_csv(points: tuple[EffectPoint, ...]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        (
            "environment_id",
            "environment_label",
            "environment_role",
            "comparator_id",
            "comparator_label",
            "candidate_minus_comparator_error_percentage_points",
            "interval_lower_percentage_points",
            "interval_upper_percentage_points",
            "episode_count",
            "worst_setting_for_comparator",
        )
    )
    for point in points:
        writer.writerow(
            (
                point.environment_id,
                point.environment_label,
                point.environment_role,
                point.comparator_id,
                point.comparator_label,
                100.0 * point.effect,
                "" if point.interval_lower is None else 100.0 * point.interval_lower,
                "" if point.interval_upper is None else 100.0 * point.interval_upper,
                point.episode_count,
                str(point.worst_setting).lower(),
            )
        )
    return output.getvalue().encode("utf-8")


def _render_vector_files(
    points: tuple[EffectPoint, ...],
    *,
    profile: FigureProfile,
    generated_at_utc: datetime,
) -> tuple[bytes, bytes]:
    style = style_for(profile)
    with mpl.rc_context({**_STYLE, "font.size": style.font_size}):
        figure = _build_figure(points, profile=profile)
        pdf_buffer = BytesIO()
        svg_buffer = BytesIO()
        figure.savefig(
            pdf_buffer,
            format="pdf",
            metadata={
                "Title": _TITLE,
                "Author": "Anas Khan",
                "Subject": "Canonical acquisition effects by simulated environment",
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
                "Description": f"{_SUBTITLE} {_FOOTNOTE}",
                "Creator": "Socratic Tutor reproducible figure pipeline",
                "Date": generated_at_utc.isoformat(),
            },
        )
        figure.clear()
    return pdf_buffer.getvalue(), svg_buffer.getvalue()


def _build_figure(points: tuple[EffectPoint, ...], *, profile: FigureProfile) -> Figure:
    style = style_for(profile)
    height = 5.7 if profile == "report" else style.height_in
    figure = Figure(figsize=(style.width_in, height), facecolor="white")
    left = 0.34 if profile == "report" else 0.24
    axis = figure.add_axes((left, 0.17, 0.96 - left, 0.55))

    figure.text(
        0.06,
        0.965,
        _TITLE,
        fontsize=style.title_size,
        fontweight="bold",
        color=INK,
        va="top",
    )
    figure.text(0.06, 0.89, _SUBTITLE, fontsize=style.font_size, color=MUTED, va="top")
    handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="none",
            markerfacecolor=INK,
            markeredgecolor=INK,
            markersize=6 if profile == "report" else 9,
            label=label,
        )
        for _, label, marker in _COMPARATORS
    ]
    figure.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.055, 0.825),
        frameon=False,
        ncols=3,
        columnspacing=1.5,
        handletextpad=0.5,
    )
    _draw_effects(axis, points, profile=profile)
    figure.text(
        0.06,
        0.025,
        _FOOTNOTE,
        fontsize=style.font_size - (0.9 if profile == "report" else 1.2),
        color=MUTED,
        va="bottom",
        wrap=True,
    )
    return figure


def _draw_effects(axis: Axes, points: tuple[EffectPoint, ...], *, profile: FigureProfile) -> None:
    environment_ids = [environment_id for environment_id, _ in _ENVIRONMENTS]
    labels = [label for _, label in _ENVIRONMENTS]
    y_by_environment = {
        environment_id: len(environment_ids) - index - 1
        for index, environment_id in enumerate(environment_ids)
    }
    offsets = dict(
        zip(
            (comparator.value for comparator, _, _ in _COMPARATORS),
            (0.22, 0.0, -0.22),
            strict=True,
        )
    )
    axis.axhspan(6.55, 7.45, color="#F1F7F4", zorder=0)
    axis.axhspan(-0.45, 0.45, color="#F0F3F5", zorder=0)
    axis.axvline(0.0, color=INK, linewidth=1.1, zorder=1)
    axis.grid(axis="x", color=LIGHT_GREY, linewidth=0.8, zorder=0)

    for point in points:
        x = 100.0 * point.effect
        y = y_by_environment[point.environment_id] + offsets[point.comparator_id]
        colour = BLUE if x < 0.0 else ORANGE
        size = 36 if profile == "report" else 72
        edge_colour = INK if point.worst_setting else colour
        edge_width = 1.4 if point.worst_setting else 0.7
        if point.interval_lower is None or point.interval_upper is None:
            axis.scatter(
                x,
                y,
                marker=point.marker,
                s=size,
                facecolor=colour,
                edgecolor=edge_colour,
                linewidth=edge_width,
                zorder=3,
            )
        else:
            lower = 100.0 * point.interval_lower
            upper = 100.0 * point.interval_upper
            axis.errorbar(
                x,
                y,
                xerr=((x - lower,), (upper - x,)),
                marker=point.marker,
                color=colour,
                markeredgecolor=colour,
                markersize=6 if profile == "report" else 9,
                linewidth=1.8,
                capsize=3.0,
                zorder=4,
            )

    axis.set_yticks(list(reversed(range(len(labels)))), labels)
    axis.set_xlim(-4.5, 2.75)
    axis.set_ylim(-0.65, 7.65)
    axis.set_xlabel("Proposed selector minus comparison error (percentage points)")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(GREY)
    axis.spines["bottom"].set_color(GREY)
    axis.tick_params(colors=INK)


def _read(path: Path, label: str) -> bytes:
    return read_bytes(path, label, AcquisitionEnvironmentFigureError)


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
        raise AcquisitionEnvironmentFigureError(
            f"Could not verify existing environment figure manifest: {manifest_path}"
        ) from error


def _require_utc(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("Environment figure timestamp must be timezone-aware UTC")


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PRIMARY_REPORT = (
    PROJECT_ROOT
    / "artifacts"
    / "acquisition-study"
    / "canonical-v1"
    / "primary-analysis-v1"
    / "canonical_primary_analysis_report.json"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "acquisition-study"
    / "canonical-v1"
    / "publication-environments-v1"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish canonical acquisition effects by simulated environment"
    )
    parser.add_argument("--primary-report", type=Path, default=DEFAULT_PRIMARY_REPORT)
    parser.add_argument("--pixi-lock", type=Path, default=PROJECT_ROOT / "pixi.lock")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--generated-at-utc", type=_utc_datetime)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = render_acquisition_environment_figure(
            primary_report_path=args.primary_report,
            pixi_lock_path=args.pixi_lock,
            output_root=args.output_root,
            publication_code_revision=current_clean_revision(
                PROJECT_ROOT,
                dirty_message="Commit or remove all visible changes before publishing the figure",
                untracked_files="all",
                required_branch="main",
                operation_name="Figure publication",
                invalid_revision_message="Publication revision must be a Git SHA",
                invalid_revision_error_factory=AcquisitionEnvironmentFigureError,
            ),
            generated_at_utc=args.generated_at_utc,
        )
    except Exception as error:
        print(
            json.dumps(
                {
                    "command": "acquisition-canonical-environment-figure",
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
                "command": "acquisition-canonical-environment-figure",
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
