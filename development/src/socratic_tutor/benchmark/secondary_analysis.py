# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
"""Frozen baseline, negative-control, and evidence-specificity analysis."""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError, model_validator

from socratic_tutor.benchmark.analysis_spec import (
    AnalysisSpecification,
    analysis_specification_hash,
    load_analysis_specification,
)
from socratic_tutor.benchmark.artifacts import write_immutable_json
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.evaluator.aggregation import CaseAggregate, CaseComparison
from socratic_tutor.benchmark.evaluator.datasets import read_case_aggregates
from socratic_tutor.benchmark.evidence_specificity import (
    EvidenceSpecificityAmendment,
    EvidenceSpecificityReport,
    analyze_evidence_specificity,
    load_evidence_specificity_amendment,
)
from socratic_tutor.benchmark.external_scoring import ExternalScoringSummary
from socratic_tutor.benchmark.hashing import file_sha256, model_content_hash
from socratic_tutor.benchmark.inferential_hierarchy import InferentialHierarchy
from socratic_tutor.benchmark.parquet import AtomicParquetDatasetStore
from socratic_tutor.benchmark.primary_analysis import (
    PrimaryAnalysisPlan,
    PrimaryAnalysisReport,
)
from socratic_tutor.benchmark.statistics import PairedCaseInference, paired_case_inference
from socratic_tutor.contracts import ContractModel


class SecondaryAnalysisError(ValueError):
    """Secondary-analysis inputs violate the frozen paired design."""


class SecondaryComparisonSummary(ContractModel):
    """One case-paired comparison with its complete denominator."""

    comparison: CaseComparison
    total_case_count: Literal[24]
    eligible_case_count: int = Field(ge=1, le=24)
    missing_case_ids: tuple[str, ...]
    favourable_case_count: int = Field(ge=0)
    adverse_case_count: int = Field(ge=0)
    unchanged_case_count: int = Field(ge=0)
    inference: PairedCaseInference
    source_aggregate_hashes: tuple[Sha256, ...] = Field(min_length=24, max_length=24)

    @model_validator(mode="after")
    def validate_summary(self) -> SecondaryComparisonSummary:
        if self.eligible_case_count + len(self.missing_case_ids) != self.total_case_count:
            raise ValueError("Secondary comparison case counts do not reconcile")
        if len(set(self.missing_case_ids)) != len(self.missing_case_ids):
            raise ValueError("Secondary comparison has duplicate missing cases")
        if (
            self.favourable_case_count + self.adverse_case_count + self.unchanged_case_count
            != self.eligible_case_count
        ):
            raise ValueError("Secondary comparison effect counts do not reconcile")
        if self.inference.case_count != self.eligible_case_count:
            raise ValueError("Secondary inference uses another case denominator")
        if len(set(self.source_aggregate_hashes)) != self.total_case_count:
            raise ValueError("Secondary comparison aggregate sources must be unique")
        return self


class SecondaryAnalysisPlan(ContractModel):
    """Frozen inputs and software identity for the secondary comparisons."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.secondary_analysis_plan.v1"] = (
        "benchmark.secondary_analysis_plan.v1"
    )
    benchmark_version: Literal["v1"] = "v1"
    run_id: str = Field(min_length=1)
    scoring_summary_hash: Sha256
    primary_analysis_plan_hash: Sha256
    primary_analysis_report_hash: Sha256
    aggregate_publication_hash: Sha256
    analysis_specification_hash: Sha256
    inferential_hierarchy_hash: Sha256
    evidence_specificity_amendment_hash: Sha256
    comparisons: tuple[CaseComparison, ...] = Field(min_length=5, max_length=5)
    secondary_role: Literal["interpretation_only_cannot_rescue_primary"]
    analysis_code_revision: str = Field(min_length=1)
    analysis_pixi_lock_hash: Sha256
    created_at_utc: datetime
    plan_hash: Sha256

    @model_validator(mode="after")
    def validate_plan(self) -> SecondaryAnalysisPlan:
        _require_utc(self.created_at_utc)
        if set(self.comparisons) != set(CaseComparison):
            raise ValueError("Secondary plan must include all five frozen comparisons")
        if self.plan_hash != model_content_hash(self, exclude={"plan_hash"}):
            raise ValueError("Secondary analysis-plan hash does not match its content")
        return self


class SecondaryAnalysisReport(ContractModel):
    """Auditable secondary comparisons that cannot change the primary conclusion."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.secondary_analysis_report.v1"] = (
        "benchmark.secondary_analysis_report.v1"
    )
    benchmark_version: Literal["v1"] = "v1"
    run_id: str = Field(min_length=1)
    analysis_plan_hash: Sha256
    primary_analysis_report_hash: Sha256
    comparison_summaries: tuple[SecondaryComparisonSummary, ...] = Field(min_length=5, max_length=5)
    evidence_specificity: EvidenceSpecificityReport
    secondary_can_rescue_primary: Literal[False] = False
    multiple_testing_role: Literal["descriptive_secondary_not_confirmatory"] = (
        "descriptive_secondary_not_confirmatory"
    )
    network_calls_made: Literal[0] = 0
    sandbox_calls_made: Literal[0] = 0
    completed_at_utc: datetime
    report_hash: Sha256

    @model_validator(mode="after")
    def validate_report(self) -> SecondaryAnalysisReport:
        _require_utc(self.completed_at_utc)
        comparisons = tuple(item.comparison for item in self.comparison_summaries)
        if set(comparisons) != set(CaseComparison) or len(set(comparisons)) != 5:
            raise ValueError("Secondary report must contain each frozen comparison once")
        if self.evidence_specificity.confirmatory_gate_allowed is not False:
            raise ValueError("Evidence specificity cannot become a confirmatory gate")
        if self.report_hash != model_content_hash(self, exclude={"report_hash"}):
            raise ValueError("Secondary analysis-report hash does not match its content")
        return self


def run_secondary_analysis(
    *,
    scoring_summary_path: Path,
    primary_analysis_plan_path: Path,
    primary_analysis_report_path: Path,
    analysis_specification_path: Path,
    inferential_hierarchy_path: Path,
    evidence_specificity_amendment_path: Path,
    dataset_root: Path,
    pixi_lock_path: Path,
    output_root: Path,
    analysis_code_revision: str,
    created_at_utc: datetime,
) -> SecondaryAnalysisReport:
    """Calculate all predeclared secondary comparisons from verified case rows."""

    _require_utc(created_at_utc)
    scoring = _load_json(scoring_summary_path, ExternalScoringSummary)
    primary_plan = _load_json(primary_analysis_plan_path, PrimaryAnalysisPlan)
    primary_report = _load_json(primary_analysis_report_path, PrimaryAnalysisReport)
    specification = load_analysis_specification(analysis_specification_path)
    hierarchy = _load_json(inferential_hierarchy_path, InferentialHierarchy)
    amendment = load_evidence_specificity_amendment(evidence_specificity_amendment_path)
    try:
        pixi_lock_hash = file_sha256(pixi_lock_path.read_bytes())
    except OSError as error:
        raise SecondaryAnalysisError(f"Could not read Pixi lock: {pixi_lock_path}") from error

    published = read_case_aggregates(AtomicParquetDatasetStore(dataset_root))
    aggregates = tuple(CaseAggregate.model_validate(row) for row in published.table.to_pylist())
    _validate_sources(
        scoring=scoring,
        primary_plan=primary_plan,
        primary_report=primary_report,
        specification=specification,
        hierarchy=hierarchy,
        amendment=amendment,
        aggregate_publication_hash=published.manifest.publication_hash,
        aggregate_identity=(published.manifest.benchmark_version, published.manifest.run_id),
        aggregates=aggregates,
        created_at_utc=created_at_utc,
    )
    plan = _create_plan(
        scoring=scoring,
        primary_plan=primary_plan,
        primary_report=primary_report,
        specification=specification,
        hierarchy=hierarchy,
        amendment=amendment,
        pixi_lock_hash=pixi_lock_hash,
        analysis_code_revision=analysis_code_revision,
        created_at_utc=created_at_utc,
    )
    root = output_root.resolve()
    report_path = root / "secondary_analysis_report.json"
    if report_path.exists():
        report = _load_json(report_path, SecondaryAnalysisReport)
        if report.analysis_plan_hash != plan.plan_hash:
            raise SecondaryAnalysisError("Existing secondary report belongs to another plan")
        return report
    write_immutable_json(root / "secondary_analysis_plan.json", plan)

    summaries = _comparison_summaries(aggregates, specification)
    specificity = analyze_evidence_specificity(
        aggregates=aggregates,
        amendment=amendment,
        output_path=root / "evidence_specificity_report.json",
    )
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.secondary_analysis_report.v1",
        "benchmark_version": "v1",
        "run_id": scoring.run_id,
        "analysis_plan_hash": plan.plan_hash,
        "primary_analysis_report_hash": primary_report.report_hash,
        "comparison_summaries": summaries,
        "evidence_specificity": specificity,
        "secondary_can_rescue_primary": False,
        "multiple_testing_role": "descriptive_secondary_not_confirmatory",
        "network_calls_made": 0,
        "sandbox_calls_made": 0,
        "completed_at_utc": created_at_utc,
    }
    draft = SecondaryAnalysisReport.model_construct(
        _fields_set=set(content), **content, report_hash="0" * 64
    )
    report = SecondaryAnalysisReport.model_validate(
        {**content, "report_hash": model_content_hash(draft, exclude={"report_hash"})}
    )
    write_immutable_json(report_path, report)
    return report


def _comparison_summaries(
    aggregates: tuple[CaseAggregate, ...],
    specification: AnalysisSpecification,
) -> tuple[SecondaryComparisonSummary, ...]:
    grouped: defaultdict[CaseComparison, list[CaseAggregate]] = defaultdict(list)
    for aggregate in aggregates:
        grouped[aggregate.comparison].append(aggregate)
    summaries: list[SecondaryComparisonSummary] = []
    for comparison in CaseComparison:
        rows = sorted(grouped[comparison], key=lambda item: item.case_id)
        effects = tuple(row.case_effect for row in rows if row.case_effect is not None)
        missing = tuple(row.case_id for row in rows if row.case_effect is None)
        summaries.append(
            SecondaryComparisonSummary(
                comparison=comparison,
                total_case_count=24,
                eligible_case_count=len(effects),
                missing_case_ids=missing,
                favourable_case_count=sum(effect > 0 for effect in effects),
                adverse_case_count=sum(effect < 0 for effect in effects),
                unchanged_case_count=sum(effect == 0 for effect in effects),
                inference=paired_case_inference(
                    effects,
                    bootstrap_resamples=specification.bootstrap_resamples,
                    permutation_resamples=specification.permutation_resamples,
                    minimum_interpretable_effect=specification.minimum_interpretable_effect,
                    random_seed=specification.random_seed,
                ),
                source_aggregate_hashes=tuple(row.record_hash for row in rows),
            )
        )
    return tuple(summaries)


def _validate_sources(
    *,
    scoring: ExternalScoringSummary,
    primary_plan: PrimaryAnalysisPlan,
    primary_report: PrimaryAnalysisReport,
    specification: AnalysisSpecification,
    hierarchy: InferentialHierarchy,
    amendment: EvidenceSpecificityAmendment,
    aggregate_publication_hash: Sha256,
    aggregate_identity: tuple[str, str],
    aggregates: tuple[CaseAggregate, ...],
    created_at_utc: datetime,
) -> None:
    specification_hash = analysis_specification_hash(specification)
    if created_at_utc <= primary_report.completed_at_utc:
        raise SecondaryAnalysisError("Secondary analysis must follow the primary report")
    if primary_report.analysis_plan_hash != primary_plan.plan_hash:
        raise SecondaryAnalysisError("Primary report belongs to another primary plan")
    if primary_plan.scoring_summary_hash != scoring.summary_hash:
        raise SecondaryAnalysisError("Primary plan belongs to another scoring summary")
    if primary_report.run_id != scoring.run_id or primary_plan.run_id != scoring.run_id:
        raise SecondaryAnalysisError("Primary artifacts belong to another scored run")
    if aggregate_identity != (scoring.benchmark_version, scoring.run_id):
        raise SecondaryAnalysisError("Aggregate dataset belongs to another scored run")
    if scoring.aggregate_publication_hash != aggregate_publication_hash:
        raise SecondaryAnalysisError("Aggregate dataset differs from the scoring summary")
    if primary_plan.aggregate_publication_hash != aggregate_publication_hash:
        raise SecondaryAnalysisError("Aggregate dataset differs from the primary plan")
    if hierarchy.analysis_specification_hash != specification_hash:
        raise SecondaryAnalysisError("Inferential hierarchy belongs to another analysis plan")
    if amendment.base_analysis_specification_hash != specification_hash:
        raise SecondaryAnalysisError("Specificity amendment belongs to another analysis plan")
    if hierarchy.evidence_specificity_amendment_hash != amendment.amendment_hash:
        raise SecondaryAnalysisError("Inferential hierarchy names another specificity amendment")
    if amendment.secondary_role != "interpretation_only_cannot_rescue_primary":
        raise SecondaryAnalysisError("Specificity amendment has an invalid analytical role")
    expected_rows = 24 * len(CaseComparison)
    identities = {(row.case_id, row.comparison) for row in aggregates}
    if len(aggregates) != expected_rows or len(identities) != expected_rows:
        raise SecondaryAnalysisError("Secondary analysis requires 24 cases by five comparisons")
    case_ids = {row.case_id for row in aggregates}
    if len(case_ids) != 24:
        raise SecondaryAnalysisError("Secondary analysis requires all 24 case identities")
    missing_by_comparison = {
        comparison: {
            row.case_id
            for row in aggregates
            if row.comparison is comparison and row.case_effect is None
        }
        for comparison in CaseComparison
    }
    if any(missing != set(scoring.missing_case_ids) for missing in missing_by_comparison.values()):
        raise SecondaryAnalysisError("Secondary comparisons do not share the frozen missing cases")


def _create_plan(
    *,
    scoring: ExternalScoringSummary,
    primary_plan: PrimaryAnalysisPlan,
    primary_report: PrimaryAnalysisReport,
    specification: AnalysisSpecification,
    hierarchy: InferentialHierarchy,
    amendment: EvidenceSpecificityAmendment,
    pixi_lock_hash: Sha256,
    analysis_code_revision: str,
    created_at_utc: datetime,
) -> SecondaryAnalysisPlan:
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.secondary_analysis_plan.v1",
        "benchmark_version": "v1",
        "run_id": scoring.run_id,
        "scoring_summary_hash": scoring.summary_hash,
        "primary_analysis_plan_hash": primary_plan.plan_hash,
        "primary_analysis_report_hash": primary_report.report_hash,
        "aggregate_publication_hash": scoring.aggregate_publication_hash,
        "analysis_specification_hash": analysis_specification_hash(specification),
        "inferential_hierarchy_hash": hierarchy.hierarchy_hash,
        "evidence_specificity_amendment_hash": amendment.amendment_hash,
        "comparisons": tuple(CaseComparison),
        "secondary_role": amendment.secondary_role,
        "analysis_code_revision": analysis_code_revision,
        "analysis_pixi_lock_hash": pixi_lock_hash,
        "created_at_utc": created_at_utc,
    }
    draft = SecondaryAnalysisPlan.model_construct(
        _fields_set=set(content), **content, plan_hash="0" * 64
    )
    return SecondaryAnalysisPlan.model_validate(
        {**content, "plan_hash": model_content_hash(draft, exclude={"plan_hash"})}
    )


def _load_json[ModelT: ContractModel](path: Path, model: type[ModelT]) -> ModelT:
    try:
        return model.model_validate_json(path.read_bytes())
    except (OSError, ValidationError) as error:
        message = f"Could not verify secondary-analysis source: {path}"
        raise SecondaryAnalysisError(message) from error


def _require_utc(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("Secondary analysis timestamp must be UTC")
