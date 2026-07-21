"""Verified reconstruction of compact benchmark summaries from published datasets."""

from pathlib import Path
from statistics import fmean
from typing import Literal

from pydantic import Field, model_validator

from socratic_tutor.benchmark.artifacts import write_immutable_json
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.evaluator.aggregation import CaseComparison
from socratic_tutor.benchmark.evaluator.datasets import (
    read_baseline_results,
    read_case_aggregates,
    read_criterion_records,
    read_repeat_metrics,
)
from socratic_tutor.benchmark.hashing import model_content_hash
from socratic_tutor.benchmark.parquet import AtomicParquetDatasetStore
from socratic_tutor.benchmark.public.datasets import read_condition_predictions
from socratic_tutor.contracts import ContractModel


class EvaluationReportError(ValueError):
    """Published datasets cannot be joined into one trustworthy run report."""


class ComparisonSummary(ContractModel):
    comparison: CaseComparison
    case_count: int = Field(ge=0)
    eligible_case_count: int = Field(ge=0)
    mean_case_effect: float | None = Field(default=None, ge=-1.0, le=1.0)
    median_case_effect: float | None = Field(default=None, ge=-1.0, le=1.0)


class EvaluationSummary(ContractModel):
    """Content-addressed descriptive report; inferential analysis remains a later gate."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.evaluation_summary.v1"] = "benchmark.evaluation_summary.v1"
    benchmark_version: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    decision_seal_hash: Sha256
    row_counts: dict[str, int]
    demonstrated_count: int = Field(ge=0)
    not_demonstrated_count: int = Field(ge=0)
    missing_criterion_count: int = Field(ge=0)
    comparisons: tuple[ComparisonSummary, ...]
    publication_hashes: dict[str, Sha256]
    report_hash: Sha256

    @model_validator(mode="after")
    def validate_report(self) -> "EvaluationSummary":
        if self.report_hash != model_content_hash(self, exclude={"report_hash"}):
            raise ValueError("Evaluation report hash does not match its content")
        return self


def evaluate_published_run(
    *,
    dataset_root: Path,
    output_path: Path,
) -> EvaluationSummary:
    """Verify every publication and rebuild descriptive case-level results."""

    store = AtomicParquetDatasetStore(dataset_root)
    loaded = (
        read_condition_predictions(store),
        read_criterion_records(store),
        read_baseline_results(store),
        read_repeat_metrics(store),
        read_case_aggregates(store),
    )
    manifests = tuple(item.manifest for item in loaded)
    identities = {
        (manifest.benchmark_version, manifest.run_id, manifest.decision_seal_hash)
        for manifest in manifests
    }
    if len(identities) != 1:
        raise EvaluationReportError("Published datasets do not share one sealed run identity")
    benchmark_version, run_id, decision_seal_hash = identities.pop()
    names = tuple(manifest.dataset.value for manifest in manifests)
    criterion_rows = loaded[1].table.to_pylist()
    aggregate_rows = loaded[4].table.to_pylist()
    comparisons = tuple(
        _comparison_summary(comparison, aggregate_rows) for comparison in CaseComparison
    )
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.evaluation_summary.v1",
        "benchmark_version": benchmark_version,
        "run_id": run_id,
        "decision_seal_hash": decision_seal_hash,
        "row_counts": {name: item.table.num_rows for name, item in zip(names, loaded, strict=True)},
        "demonstrated_count": sum(
            row["demonstrated_performance"] is True for row in criterion_rows
        ),
        "not_demonstrated_count": sum(
            row["demonstrated_performance"] is False for row in criterion_rows
        ),
        "missing_criterion_count": sum(
            row["demonstrated_performance"] is None for row in criterion_rows
        ),
        "comparisons": comparisons,
        "publication_hashes": {
            name: manifest.publication_hash for name, manifest in zip(names, manifests, strict=True)
        },
    }
    draft = EvaluationSummary.model_construct(
        _fields_set=set(content),
        **content,
        report_hash="0" * 64,
    )
    report = EvaluationSummary.model_validate(
        {**content, "report_hash": model_content_hash(draft, exclude={"report_hash"})}
    )
    write_immutable_json(output_path, report)
    return report


def _comparison_summary(
    comparison: CaseComparison,
    rows: list[dict[str, object]],
) -> ComparisonSummary:
    selected = tuple(row for row in rows if row["comparison"] == comparison.value)
    raw_effects = tuple(row["case_effect"] for row in selected)
    if any(value is not None and not isinstance(value, (float, int)) for value in raw_effects):
        raise EvaluationReportError("Case aggregate contains a non-numeric effect")
    effects = sorted(float(value) for value in raw_effects if isinstance(value, (float, int)))
    return ComparisonSummary(
        comparison=comparison,
        case_count=len(selected),
        eligible_case_count=len(effects),
        mean_case_effect=fmean(effects) if effects else None,
        median_case_effect=_median(effects) if effects else None,
    )


def _median(values: list[float]) -> float:
    midpoint = len(values) // 2
    if len(values) % 2:
        return values[midpoint]
    return (values[midpoint - 1] + values[midpoint]) / 2
