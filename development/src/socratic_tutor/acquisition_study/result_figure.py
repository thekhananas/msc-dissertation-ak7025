"""Publication figure for the canonical acquisition-study result."""

# pyright: reportArgumentType=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Literal, Self

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pydantic import Field, ValidationError, model_validator

from socratic_tutor.acquisition_study.canonical_primary_analysis import (
    CanonicalPrimaryAnalysisReport,
)
from socratic_tutor.acquisition_study.contracts import PolicyId
from socratic_tutor.benchmark.artifacts import (
    artifact_locations,
    write_immutable_bytes,
    write_immutable_json,
)
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import canonical_sha256, file_sha256, model_content_hash
from socratic_tutor.contracts import ContractModel
from socratic_tutor.filesystem import read_bytes
from socratic_tutor.publication.figure_style import (
    BLUE,
    GREEN,
    GREY,
    INK,
    LIGHT_GREY,
    MUTED,
    ORANGE,
    FigureProfile,
    style_for,
)
from socratic_tutor.repository_state import current_clean_revision

_CANDIDATE = PolicyId.RELIABILITY_AWARE_BOUNDED.value
_BUDGETS = (0.0, 0.25, 0.5, 0.75, 1.0)
_COMPARATOR_LABELS = {
    PolicyId.SEEDED_RANDOM_BOUNDED: "Random choice",
    PolicyId.UNCERTAINTY_ONLY_BOUNDED: "Highest uncertainty",
    PolicyId.PLUG_IN_EVSI_BOUNDED: "Standard value of information",
}
_TITLE = "Executable checks helped only when evidence behaved as assumed"
_SUBTITLE = "Prediction error across 2,000 simulated episodes per setting; lower values are better."
_FOOTNOTE = (
    "Hand-specified simulator; six altered failure settings receive equal weight. "
    "This does not measure human learning or tutoring quality."
)
_STYLE: dict[str, object] = {
    "figure.facecolor": "white",
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans", "sans-serif"],
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "svg.fonttype": "none",
    "svg.hashsalt": "socratic-tutor-acquisition-result-v1",
    "text.color": INK,
}


class AcquisitionResultFigureError(ValueError):
    """Canonical result sources cannot support the publication figure."""


class FigureFile(ContractModel):
    """One generated vector file and its digest."""

    profile: FigureProfile
    format: Literal["pdf", "svg"]
    file: str
    sha256: Sha256


class AcquisitionResultFigureManifest(ContractModel):
    """Source and output identities for the canonical result figure."""

    schema_version: Literal[1] = 1
    schema_id: Literal["acquisition_study.result_figure_manifest.v1"] = (
        "acquisition_study.result_figure_manifest.v1"
    )
    study_id: Literal["reliability-aware-probing-v1"] = "reliability-aware-probing-v1"
    split: Literal["evaluation"] = "evaluation"
    figure_id: Literal["FIG-M7B-01"] = "FIG-M7B-01"
    primary_report_hash: Sha256
    primary_report_sha256: Sha256
    budget_report_hash: Sha256
    budget_report_sha256: Sha256
    budget_environment_table_sha256: Sha256
    pixi_lock_sha256: Sha256
    publication_code_revision: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    files: tuple[FigureFile, FigureFile, FigureFile, FigureFile]
    source_data_file: Literal["acquisition_result_figure_data.csv"] = (
        "acquisition_result_figure_data.csv"
    )
    source_data_sha256: Sha256
    generated_at_utc: datetime
    result_classification: Literal["useful_when_matched_harmful_under_important_misspecification"]
    human_learning_claim_supported: Literal[False] = False
    tutoring_efficacy_claim_supported: Literal[False] = False
    manifest_hash: Sha256

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        _require_utc(self.generated_at_utc)
        expected = (
            ("report", "pdf"),
            ("report", "svg"),
            ("presentation", "pdf"),
            ("presentation", "svg"),
        )
        if tuple((item.profile, item.format) for item in self.files) != expected:
            raise ValueError("Acquisition figure files differ from the required profiles")
        if self.manifest_hash != model_content_hash(self, exclude={"manifest_hash"}):
            raise ValueError("Acquisition figure manifest hash does not match its content")
        return self


class _BudgetPoint(ContractModel):
    budget_fraction: float = Field(ge=0.0, le=1.0)
    matched_error: float = Field(ge=0.0, le=1.0)
    held_out_error: float = Field(ge=0.0, le=1.0)


class _ComparisonPoint(ContractModel):
    comparator: str
    label: str
    effect: float = Field(ge=-1.0, le=1.0)
    lower: float = Field(ge=-1.0, le=1.0)
    upper: float = Field(ge=-1.0, le=1.0)
    confidence_level: float = Field(gt=0.95, lt=1.0)

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        if not self.lower <= self.effect <= self.upper:
            raise ValueError("Primary comparison point lies outside its interval")
        return self


def render_acquisition_result_figure(
    *,
    primary_report_path: Path,
    budget_report_path: Path,
    pixi_lock_path: Path,
    output_root: Path,
    publication_code_revision: str,
    generated_at_utc: datetime | None = None,
) -> AcquisitionResultFigureManifest:
    """Render report and presentation versions from verified canonical reports."""

    primary_bytes = _read(primary_report_path, "canonical primary report")
    budget_bytes = _read(budget_report_path, "canonical budget report")
    pixi_lock_bytes = _read(pixi_lock_path, "Pixi lock")
    primary = _load_primary(primary_bytes)
    budget = _load_budget_report(budget_bytes)
    environment_table_path = budget_report_path.parent / str(budget["environment_table_file"])
    environment_bytes = _read(environment_table_path, "canonical budget environment table")
    if file_sha256(environment_bytes) != budget["environment_table_sha256"]:
        raise AcquisitionResultFigureError("Budget environment table hash does not match")
    if budget["primary_analysis_report_hash"] != primary.report_hash:
        raise AcquisitionResultFigureError("Primary and budget reports do not describe one run")

    budget_points = _budget_points(budget, environment_bytes)
    comparisons = _comparison_points(primary)
    generated_at = _resolve_generated_at(
        generated_at_utc,
        output_root / "acquisition_result_figure_manifest.json",
    )
    source_data = _source_data_csv(budget_points, comparisons)
    outputs: list[tuple[FigureProfile, str, bytes]] = []
    for profile in ("report", "presentation"):
        pdf, svg = _render_vector_files(
            budget_points,
            comparisons,
            profile=profile,
            generated_at_utc=generated_at,
        )
        outputs.extend(((profile, "pdf", pdf), (profile, "svg", svg)))

    files = tuple(
        FigureFile(
            profile=profile,
            format=file_format,
            file=f"acquisition_result_{profile}.{file_format}",
            sha256=file_sha256(content),
        )
        for profile, file_format, content in outputs
    )
    content = {
        "schema_version": 1,
        "schema_id": "acquisition_study.result_figure_manifest.v1",
        "study_id": "reliability-aware-probing-v1",
        "split": "evaluation",
        "figure_id": "FIG-M7B-01",
        "primary_report_hash": primary.report_hash,
        "primary_report_sha256": file_sha256(primary_bytes),
        "budget_report_hash": budget["report_hash"],
        "budget_report_sha256": file_sha256(budget_bytes),
        "budget_environment_table_sha256": file_sha256(environment_bytes),
        "pixi_lock_sha256": file_sha256(pixi_lock_bytes),
        "publication_code_revision": publication_code_revision,
        "files": files,
        "source_data_file": "acquisition_result_figure_data.csv",
        "source_data_sha256": file_sha256(source_data),
        "generated_at_utc": generated_at,
        "result_classification": ("useful_when_matched_harmful_under_important_misspecification"),
        "human_learning_claim_supported": False,
        "tutoring_efficacy_claim_supported": False,
    }
    draft = AcquisitionResultFigureManifest.model_construct(
        _fields_set=set(content),
        **content,
        manifest_hash="0" * 64,
    )
    manifest = AcquisitionResultFigureManifest.model_validate(
        {**content, "manifest_hash": model_content_hash(draft, exclude={"manifest_hash"})}
    )
    for item, (_, _, data) in zip(files, outputs, strict=True):
        write_immutable_bytes(output_root / item.file, data)
    write_immutable_bytes(output_root / manifest.source_data_file, source_data)
    write_immutable_json(output_root / "acquisition_result_figure_manifest.json", manifest)
    return manifest


def _load_primary(content: bytes) -> CanonicalPrimaryAnalysisReport:
    try:
        return CanonicalPrimaryAnalysisReport.model_validate_json(content)
    except ValidationError as error:
        raise AcquisitionResultFigureError("Canonical primary report is invalid") from error


def _load_budget_report(content: bytes) -> dict[str, object]:
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AcquisitionResultFigureError("Canonical budget report is invalid JSON") from error
    if not isinstance(value, dict):
        raise AcquisitionResultFigureError("Canonical budget report must be an object")
    required = {
        "schema_id": "acquisition_study.canonical_budget_curve_analysis.v1",
        "study_id": "reliability-aware-probing-v1",
        "split": "evaluation",
        "result_status": "canonical_secondary_budget_analysis_complete",
        "episodes_per_environment": 2000,
        "primary_effect_reproduction_verified": True,
        "restricted_stream_accessed": False,
        "human_learning_claim_supported": False,
        "tutoring_efficacy_claim_supported": False,
    }
    if any(value.get(key) != expected for key, expected in required.items()):
        raise AcquisitionResultFigureError("Canonical budget report has the wrong study identity")
    report_hash = value.get("report_hash")
    unsigned = {key: item for key, item in value.items() if key != "report_hash"}
    if not isinstance(report_hash, str) or canonical_sha256(unsigned) != report_hash:
        raise AcquisitionResultFigureError("Canonical budget report hash does not match")
    if value.get("budget_fractions") != list(_BUDGETS):
        raise AcquisitionResultFigureError("Canonical budget levels have changed")
    for key in (
        "environment_table_file",
        "environment_table_sha256",
        "primary_analysis_report_hash",
    ):
        if not isinstance(value.get(key), str):
            raise AcquisitionResultFigureError(f"Canonical budget report is missing {key}")
    if not isinstance(value.get("held_out_macro_summaries"), list):
        raise AcquisitionResultFigureError("Canonical budget report is missing held-out summaries")
    return value


def _budget_points(
    budget: dict[str, object],
    environment_table: bytes,
) -> tuple[_BudgetPoint, ...]:
    try:
        rows = tuple(csv.DictReader(io.StringIO(environment_table.decode("utf-8"))))
    except UnicodeDecodeError as error:
        raise AcquisitionResultFigureError("Budget environment table is not UTF-8") from error
    matched = {
        float(row["budget_fraction"]): float(row["mean_classification_error"])
        for row in rows
        if row["environment_id"] == "matched" and row["policy_id"] == _CANDIDATE
    }
    raw_held_out = budget["held_out_macro_summaries"]
    assert isinstance(raw_held_out, list)
    held_out: dict[float, float] = {}
    for row in raw_held_out:
        if not isinstance(row, dict) or row.get("policy_id") != _CANDIDATE:
            continue
        try:
            held_out[float(row["budget_fraction"])] = float(row["macro_mean_classification_error"])
        except (KeyError, TypeError, ValueError) as error:
            raise AcquisitionResultFigureError("Held-out budget summary is invalid") from error
    if set(matched) != set(_BUDGETS) or set(held_out) != set(_BUDGETS):
        raise AcquisitionResultFigureError("Candidate budget curve is incomplete")
    return tuple(
        _BudgetPoint(
            budget_fraction=budget_fraction,
            matched_error=matched[budget_fraction],
            held_out_error=held_out[budget_fraction],
        )
        for budget_fraction in _BUDGETS
    )


def _comparison_points(
    report: CanonicalPrimaryAnalysisReport,
) -> tuple[_ComparisonPoint, ...]:
    points = tuple(
        _ComparisonPoint(
            comparator=row.comparator_policy_id.value,
            label=_COMPARATOR_LABELS[row.comparator_policy_id],
            effect=row.macro_mean_paired_effect,
            lower=row.interval_lower,
            upper=row.interval_upper,
            confidence_level=row.confidence_level,
        )
        for row in report.held_out_primary_intervals
    )
    if len(points) != 3:
        raise AcquisitionResultFigureError("Primary comparison is incomplete")
    if len({point.confidence_level for point in points}) != 1:
        raise AcquisitionResultFigureError("Primary intervals use different confidence levels")
    return points


def _source_data_csv(
    budgets: tuple[_BudgetPoint, ...],
    comparisons: tuple[_ComparisonPoint, ...],
) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        (
            "panel",
            "series",
            "budget_percent",
            "value_percentage_points",
            "interval_lower_percentage_points",
            "interval_upper_percentage_points",
            "confidence_percent",
        )
    )
    for point in budgets:
        budget_percent = 100.0 * point.budget_fraction
        writer.writerow(
            ("budget", "matched", budget_percent, 100.0 * point.matched_error, "", "", "")
        )
        writer.writerow(
            (
                "budget",
                "held_out_equal_mean",
                budget_percent,
                100.0 * point.held_out_error,
                "",
                "",
                "",
            )
        )
    for point in comparisons:
        writer.writerow(
            (
                "fixed_budget_comparison",
                point.comparator,
                50.0,
                100.0 * point.effect,
                100.0 * point.lower,
                100.0 * point.upper,
                100.0 * point.confidence_level,
            )
        )
    return output.getvalue().encode("utf-8")


def _render_vector_files(
    budgets: tuple[_BudgetPoint, ...],
    comparisons: tuple[_ComparisonPoint, ...],
    *,
    profile: FigureProfile,
    generated_at_utc: datetime,
) -> tuple[bytes, bytes]:
    style = style_for(profile)
    with mpl.rc_context({**_STYLE, "font.size": style.font_size}):
        figure = _build_figure(budgets, comparisons, profile=profile)
        pdf_buffer = BytesIO()
        svg_buffer = BytesIO()
        figure.savefig(
            pdf_buffer,
            format="pdf",
            metadata={
                "Title": _TITLE,
                "Author": "Anas Khan",
                "Subject": "Canonical reliability-aware acquisition study result",
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


def _build_figure(
    budgets: tuple[_BudgetPoint, ...],
    comparisons: tuple[_ComparisonPoint, ...],
    *,
    profile: FigureProfile,
) -> Figure:
    style = style_for(profile)
    if profile == "report":
        figure = Figure(figsize=(style.width_in, 6.2), facecolor="white")
        grid = figure.add_gridspec(
            2,
            1,
            left=0.13,
            right=0.97,
            top=0.79,
            bottom=0.14,
            hspace=0.62,
        )
        budget_axis = figure.add_subplot(grid[0, 0])
        comparison_axis = figure.add_subplot(grid[1, 0])
        title_y, subtitle_y, footer_y = 0.965, 0.913, 0.025
    else:
        figure = Figure(figsize=(style.width_in, style.height_in), facecolor="white")
        grid = figure.add_gridspec(
            1,
            2,
            left=0.075,
            right=0.975,
            top=0.76,
            bottom=0.23,
            wspace=0.38,
        )
        budget_axis = figure.add_subplot(grid[0, 0])
        comparison_axis = figure.add_subplot(grid[0, 1])
        title_y, subtitle_y, footer_y = 0.955, 0.875, 0.035

    figure.text(
        0.06,
        title_y,
        _TITLE,
        fontsize=style.title_size,
        fontweight="bold",
        color=INK,
        va="top",
    )
    figure.text(0.06, subtitle_y, _SUBTITLE, fontsize=style.font_size, color=MUTED, va="top")
    _draw_budget_panel(budget_axis, budgets, profile=profile)
    _draw_comparison_panel(comparison_axis, comparisons, profile=profile)
    figure.text(
        0.06,
        footer_y,
        _FOOTNOTE,
        fontsize=style.font_size - (0.8 if profile == "report" else 1.0),
        color=MUTED,
        va="bottom",
        wrap=True,
    )
    return figure


def _draw_budget_panel(
    axis: Axes,
    points: tuple[_BudgetPoint, ...],
    *,
    profile: FigureProfile,
) -> None:
    budgets = [100.0 * point.budget_fraction for point in points]
    matched = [100.0 * point.matched_error for point in points]
    held_out = [100.0 * point.held_out_error for point in points]
    marker_size = 5.2 if profile == "report" else 8.0
    axis.plot(
        budgets,
        matched,
        color=GREEN,
        marker="o",
        markersize=marker_size,
        linewidth=2.0,
        label="Evidence behaved as assumed",
    )
    axis.plot(
        budgets,
        held_out,
        color=ORANGE,
        marker="s",
        markersize=marker_size,
        linewidth=2.0,
        linestyle="--",
        label="Six altered failure settings",
    )
    axis.axvline(50.0, color=GREY, linewidth=1.0, linestyle=":", zorder=0)
    axis.text(
        50.8,
        10.9,
        "planned 50% budget",
        color=MUTED,
        fontsize="small",
        va="bottom",
    )
    axis.set_title("A. More probing helped only in the matched setting", loc="left", pad=8)
    axis.set_xlabel("Cases given an executable check (%)")
    axis.set_ylabel("Prediction error (%)")
    axis.set_xticks(budgets)
    axis.set_ylim(10.0, 45.0)
    axis.grid(axis="y", color=LIGHT_GREY, linewidth=0.8)
    axis.legend(frameon=False, loc="upper left", ncols=2 if profile == "report" else 1)
    _plain_axes(axis)


def _draw_comparison_panel(
    axis: Axes,
    points: tuple[_ComparisonPoint, ...],
    *,
    profile: FigureProfile,
) -> None:
    y_positions = list(reversed(range(len(points))))
    markers = ("o", "s", "D")
    for point, y_position, marker in zip(points, y_positions, markers, strict=True):
        effect = 100.0 * point.effect
        lower = 100.0 * point.lower
        upper = 100.0 * point.upper
        colour = BLUE if effect < 0.0 else ORANGE
        axis.errorbar(
            effect,
            y_position,
            xerr=((effect - lower,), (upper - effect,)),
            color=colour,
            marker=marker,
            markersize=5.8 if profile == "report" else 9.0,
            linewidth=1.8,
            capsize=3.0,
            zorder=3,
        )
        label_x = lower if effect < 0.0 else upper
        axis.annotate(
            f"{effect:+.2f}",
            (label_x, y_position),
            xytext=(5 if effect >= 0 else -5, 0),
            textcoords="offset points",
            ha="left" if effect >= 0 else "right",
            va="center",
            color=colour,
            fontweight="bold",
        )
    axis.axvline(0.0, color=INK, linewidth=1.0)
    axis.set_title("B. At 50%, only random choice performed worse", loc="left", pad=8)
    axis.set_yticks(y_positions, [point.label for point in points])
    axis.set_xlabel("Change in prediction error (percentage points)")
    axis.set_xlim(-1.65, 1.25)
    axis.set_ylim(-0.75, len(points) - 0.15)
    axis.grid(axis="x", color=LIGHT_GREY, linewidth=0.8)
    axis.text(
        0.02,
        0.98,
        "Proposed selector better",
        transform=axis.transAxes,
        color=BLUE,
        ha="left",
        va="top",
        fontsize="small",
    )
    axis.text(
        0.98,
        0.98,
        "Comparator better",
        transform=axis.transAxes,
        color=ORANGE,
        ha="right",
        va="top",
        fontsize="small",
    )
    axis.text(
        0.5,
        0.88,
        f"{100.0 * points[0].confidence_level:.1f}% simultaneous intervals",
        transform=axis.transAxes,
        color=MUTED,
        ha="center",
        va="top",
        fontsize="small",
    )
    _plain_axes(axis)


def _plain_axes(axis: Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(GREY)
    axis.spines["bottom"].set_color(GREY)
    axis.tick_params(colors=INK)


def _read(path: Path, label: str) -> bytes:
    return read_bytes(path, label, AcquisitionResultFigureError)


def _resolve_generated_at(value: datetime | None, manifest_path: Path) -> datetime:
    if value is not None:
        _require_utc(value)
        return value
    if manifest_path.exists():
        try:
            return AcquisitionResultFigureManifest.model_validate_json(
                manifest_path.read_bytes()
            ).generated_at_utc
        except (OSError, ValidationError) as error:
            raise AcquisitionResultFigureError(
                f"Could not verify existing acquisition figure manifest: {manifest_path}"
            ) from error
    return datetime.now(UTC)


def _require_utc(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("Acquisition figure timestamp must be timezone-aware UTC")


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CANONICAL_ROOT = PROJECT_ROOT / "artifacts" / "acquisition-study" / "canonical-v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish the canonical acquisition result for report and presentation use"
    )
    parser.add_argument(
        "--primary-report",
        type=Path,
        default=DEFAULT_CANONICAL_ROOT
        / "primary-analysis-v1"
        / "canonical_primary_analysis_report.json",
    )
    parser.add_argument(
        "--budget-report",
        type=Path,
        default=DEFAULT_CANONICAL_ROOT
        / "secondary-budget-v1"
        / "analysis-v1"
        / "canonical_budget_curve_analysis_report.json",
    )
    parser.add_argument("--pixi-lock", type=Path, default=PROJECT_ROOT / "pixi.lock")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_CANONICAL_ROOT / "publication-v1",
    )
    parser.add_argument("--generated-at-utc", type=_utc_datetime)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = render_acquisition_result_figure(
            primary_report_path=args.primary_report,
            budget_report_path=args.budget_report,
            pixi_lock_path=args.pixi_lock,
            output_root=args.output_root,
            publication_code_revision=current_clean_revision(
                PROJECT_ROOT,
                dirty_message="Commit or remove all visible changes before publishing the figure",
                untracked_files="all",
                required_branch="main",
                operation_name="Figure publication",
            ),
            generated_at_utc=args.generated_at_utc,
        )
    except Exception as error:
        print(
            json.dumps(
                {
                    "command": "acquisition-canonical-result-figure",
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
                "command": "acquisition-canonical-result-figure",
                "result": manifest.model_dump(mode="json"),
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
