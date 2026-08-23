"""Report-ready primary table for the canonical acquisition study."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean

from pydantic import ValidationError

from socratic_tutor.acquisition_study.canonical_primary_analysis import (
    CanonicalPrimaryAnalysisReport,
)
from socratic_tutor.acquisition_study.contracts import PolicyId
from socratic_tutor.benchmark.artifacts import (
    artifact_locations,
    write_immutable_bytes,
    write_immutable_json,
)
from socratic_tutor.benchmark.hashing import canonical_sha256, file_sha256

_COMPARATOR_LABELS = {
    PolicyId.SEEDED_RANDOM_BOUNDED: "Random selection",
    PolicyId.UNCERTAINTY_ONLY_BOUNDED: "Highest uncertainty",
    PolicyId.PLUG_IN_EVSI_BOUNDED: "Standard value of information",
}
_CAPTION = (
    "Primary comparison at the fixed 50\\% executable-probe budget across six "
    "held-out simulated settings."
)
_NOTE = (
    "Difference is proposed minus comparator classification error, so negative values favour "
    "the proposed selector. The six settings receive equal weight, with 2,000 paired episodes "
    "per setting. The planned 98.3\\% intervals account for the three primary comparisons. "
    "These results come from a hand-specified simulator and do not measure human learning."
)


class AcquisitionResultTableError(ValueError):
    """The canonical report cannot support the primary result table."""


@dataclass(frozen=True, slots=True)
class PrimaryTableRow:
    comparator_id: str
    comparator_label: str
    candidate_error: float
    comparator_error: float
    difference: float
    interval_lower: float
    interval_upper: float
    confidence_level: float
    environment_count: int
    episodes_per_environment: int
    outcome: str


def publish_acquisition_result_table(
    *,
    primary_report_path: Path,
    pixi_lock_path: Path,
    output_root: Path,
    publication_code_revision: str,
    generated_at_utc: datetime | None = None,
) -> dict[str, object]:
    """Write CSV, LaTeX and Markdown views of the sealed primary comparison."""

    report_bytes = _read(primary_report_path, "canonical primary report")
    lock_bytes = _read(pixi_lock_path, "Pixi lock")
    report = _load_report(report_bytes)
    rows = _rows(report)
    manifest_path = output_root / "acquisition_primary_result_table_manifest.json"
    generated_at = _resolve_generated_at(generated_at_utc, manifest_path)
    outputs = {
        "acquisition_primary_result_table.csv": _csv(rows),
        "acquisition_primary_result_table.tex": _latex(rows),
        "acquisition_primary_result_table.md": _markdown(rows),
    }
    files = [
        {"format": name.rsplit(".", 1)[1], "file": name, "sha256": file_sha256(data)}
        for name, data in outputs.items()
    ]
    content: dict[str, object] = {
        "schema_version": 1,
        "schema_id": "acquisition_study.primary_result_table_manifest.v1",
        "study_id": report.study_id,
        "split": report.split,
        "table_id": "TAB-M7B-01",
        "primary_report_hash": report.report_hash,
        "primary_report_sha256": file_sha256(report_bytes),
        "pixi_lock_sha256": file_sha256(lock_bytes),
        "publication_code_revision": _validate_revision(publication_code_revision),
        "files": files,
        "row_count": len(rows),
        "held_out_environment_count": rows[0].environment_count,
        "episodes_per_environment": report.episodes_per_environment,
        "generated_at_utc": generated_at.isoformat().replace("+00:00", "Z"),
        "overall_advantage_rule_met": report.overall_advantage_rule_met,
        "robustness_rule_met": report.robustness_rule_met,
        "human_learning_claim_supported": False,
        "tutoring_efficacy_claim_supported": False,
    }
    manifest = {**content, "manifest_hash": canonical_sha256(content)}
    for name, data in outputs.items():
        write_immutable_bytes(output_root / name, data)
    write_immutable_json(manifest_path, manifest)
    return manifest


def _load_report(content: bytes) -> CanonicalPrimaryAnalysisReport:
    try:
        report = CanonicalPrimaryAnalysisReport.model_validate_json(content)
    except ValidationError as error:
        raise AcquisitionResultTableError("Canonical primary report is invalid") from error
    if report.restricted_stream_accessed or report.human_record_count:
        raise AcquisitionResultTableError("Primary report crosses the publication boundary")
    if len(report.held_out_primary_intervals) != 3:
        raise AcquisitionResultTableError("Primary report does not contain three comparisons")
    return report


def _rows(report: CanonicalPrimaryAnalysisReport) -> tuple[PrimaryTableRow, ...]:
    rows: list[PrimaryTableRow] = []
    candidate_errors: list[float] = []
    for comparison in report.held_out_primary_intervals:
        if comparison.comparator_policy_id not in _COMPARATOR_LABELS:
            raise AcquisitionResultTableError("Primary report contains an unknown comparator")
        candidate_error = fmean(
            item.candidate_mean_classification_error for item in comparison.environment_effects
        )
        comparator_error = fmean(
            item.comparator_mean_classification_error for item in comparison.environment_effects
        )
        if not math.isclose(
            candidate_error - comparator_error,
            comparison.macro_mean_paired_effect,
            abs_tol=1e-12,
        ):
            raise AcquisitionResultTableError("Primary table values do not reconcile")
        candidate_errors.append(candidate_error)
        rows.append(
            PrimaryTableRow(
                comparator_id=comparison.comparator_policy_id.value,
                comparator_label=_COMPARATOR_LABELS[comparison.comparator_policy_id],
                candidate_error=candidate_error,
                comparator_error=comparator_error,
                difference=comparison.macro_mean_paired_effect,
                interval_lower=comparison.interval_lower,
                interval_upper=comparison.interval_upper,
                confidence_level=comparison.confidence_level,
                environment_count=len(comparison.environment_effects),
                episodes_per_environment=comparison.episodes_per_environment,
                outcome=(
                    "Proposed lower error"
                    if comparison.macro_mean_paired_effect < 0
                    else "Proposed higher error"
                ),
            )
        )
    if any(
        not math.isclose(value, candidate_errors[0], abs_tol=1e-12)
        for value in candidate_errors[1:]
    ):
        raise AcquisitionResultTableError("Candidate error differs across comparisons")
    if any(row.environment_count != 6 for row in rows):
        raise AcquisitionResultTableError("Primary table does not cover six held-out settings")
    return tuple(rows)


def _csv(rows: tuple[PrimaryTableRow, ...]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        (
            "comparator_policy_id",
            "comparator_label",
            "proposed_classification_error",
            "comparator_classification_error",
            "proposed_minus_comparator_error",
            "simultaneous_interval_lower",
            "simultaneous_interval_upper",
            "confidence_level",
            "held_out_environment_count",
            "episodes_per_environment",
            "plain_outcome",
        )
    )
    for row in rows:
        writer.writerow(
            (
                row.comparator_id,
                row.comparator_label,
                f"{row.candidate_error:.12f}",
                f"{row.comparator_error:.12f}",
                f"{row.difference:.12f}",
                f"{row.interval_lower:.12f}",
                f"{row.interval_upper:.12f}",
                f"{row.confidence_level:.12f}",
                row.environment_count,
                row.episodes_per_environment,
                row.outcome,
            )
        )
    return output.getvalue().encode("utf-8")


def _latex(rows: tuple[PrimaryTableRow, ...]) -> bytes:
    proposed_error = rows[0].candidate_error * 100
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{_CAPTION} The proposed selector's error was {proposed_error:.2f}\%.}}",
        r"\label{tab:m7b-primary-comparison}",
        r"\begin{tabular}{@{}lrrl@{}}",
        r"\toprule",
        (
            r"Comparator & Error (\%) & "
            r"\shortstack{Difference [98.3\% interval]\\(percentage points)} "
            r"& Proposed \\"
        ),
        r"\midrule",
    ]
    lines.extend(
        (
            f"{_latex_escape(row.comparator_label)} & "
            f"{row.comparator_error * 100:.2f} & "
            f"{row.difference * 100:+.2f} "
            f"[{row.interval_lower * 100:+.2f}, {row.interval_upper * 100:+.2f}] & "
            f"{'Lower' if row.difference < 0 else 'Higher'} \\\\"
        )
        for row in rows
    )
    lines.extend(
        (
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{0.4em}",
            r"\begin{minipage}{0.98\linewidth}",
            r"\footnotesize " + _NOTE,
            r"\end{minipage}",
            r"\end{table}",
            "",
        )
    )
    return "\n".join(lines).encode("utf-8")


def _markdown(rows: tuple[PrimaryTableRow, ...]) -> bytes:
    lines = [
        "<!-- TAB-M7B-01 -->",
        "",
        "**Primary comparison at the fixed 50% probe budget**",
        "",
        (
            "| Comparator | Proposed error | Comparator error | Difference | "
            "98.3% interval | Outcome |"
        ),
        "|---|---:|---:|---:|---:|---|",
    ]
    lines.extend(
        (
            f"| {row.comparator_label} | {row.candidate_error:.2%} | "
            f"{row.comparator_error:.2%} | {row.difference * 100:+.2f} pp | "
            f"[{row.interval_lower * 100:+.2f}, {row.interval_upper * 100:+.2f}] pp | "
            f"{row.outcome} |"
        )
        for row in rows
    )
    lines.extend(
        (
            "",
            (
                "Difference is proposed minus comparator error, so negative values favour the "
                "proposed selector. Six held-out simulated settings receive equal weight, with "
                "2,000 paired episodes per setting. The planned 98.3% intervals account for the "
                "three comparisons. This does not measure human learning or tutoring quality."
            ),
            "",
        )
    )
    return "\n".join(lines).encode("utf-8")


def _latex_escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
    )


def _read(path: Path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise AcquisitionResultTableError(f"Could not read {label}: {path}") from error


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
        raise AcquisitionResultTableError(
            f"Could not verify existing result table manifest: {manifest_path}"
        ) from error


def _require_utc(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("Result table timestamp must be timezone-aware UTC")


def _validate_revision(value: str) -> str:
    if re.fullmatch(r"[0-9a-f]{7,40}", value) is None:
        raise AcquisitionResultTableError("Publication revision must be a Git SHA")
    return value


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
    PROJECT_ROOT / "artifacts" / "acquisition-study" / "canonical-v1" / "publication-table-v1"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish the canonical acquisition result table")
    parser.add_argument("--primary-report", type=Path, default=DEFAULT_PRIMARY_REPORT)
    parser.add_argument("--pixi-lock", type=Path, default=PROJECT_ROOT / "pixi.lock")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--generated-at-utc", type=_utc_datetime)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = publish_acquisition_result_table(
            primary_report_path=args.primary_report,
            pixi_lock_path=args.pixi_lock,
            output_root=args.output_root,
            publication_code_revision=_current_clean_revision(),
            generated_at_utc=args.generated_at_utc,
        )
    except Exception as error:
        print(
            json.dumps(
                {
                    "command": "acquisition-canonical-result-table",
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
                "command": "acquisition-canonical-result-table",
                "result": manifest,
                "status": "ok",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _current_clean_revision() -> str:
    branch = _git("branch", "--show-current")
    status = _git("status", "--porcelain", "--untracked-files=all")
    revision = _git("rev-parse", "HEAD")
    if branch != "main":
        raise ValueError(f"Table publication requires branch main, not {branch or 'detached HEAD'}")
    if status:
        raise ValueError("Commit or remove all visible changes before publishing the table")
    return _validate_revision(revision)


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _utc_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    _require_utc(parsed)
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
