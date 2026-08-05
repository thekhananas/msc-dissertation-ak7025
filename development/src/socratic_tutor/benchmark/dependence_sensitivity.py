# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
"""Descriptive dependence and heuristic sensitivity checks for external v1."""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any, Literal

from pydantic import Field, ValidationError, model_validator
from scipy.stats import wilcoxon

from socratic_tutor.benchmark.artifacts import write_immutable_json
from socratic_tutor.benchmark.command_io import load_command_model
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.design import BenchmarkDesignPlan, load_design
from socratic_tutor.benchmark.evaluator.datasets import read_repeat_metrics
from socratic_tutor.benchmark.evaluator.scoring import RepeatEstimator
from socratic_tutor.benchmark.hashing import file_sha256, model_content_hash
from socratic_tutor.benchmark.inferential_hierarchy import InferentialHierarchy
from socratic_tutor.benchmark.parquet import AtomicParquetDatasetStore
from socratic_tutor.benchmark.primary_analysis import PrimaryAnalysisPlan, PrimaryAnalysisReport
from socratic_tutor.benchmark.secondary_analysis import SecondaryAnalysisReport
from socratic_tutor.contracts import ContractModel


class DependenceSensitivityError(ValueError):
    """Sensitivity inputs do not match the frozen external experiment."""


class DependenceSensitivityGrid(ContractModel):
    """Transparent post-reveal operational grid for predeclared checks."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.dependence_sensitivity_grid.v1"] = (
        "benchmark.dependence_sensitivity_grid.v1"
    )
    grid_version: Literal["dependence-sensitivity-v1"]
    status: Literal["post_reveal_exploratory_grid"]
    thresholds: tuple[float, ...] = Field(min_length=3)
    frozen_threshold: float = Field(ge=0.6, le=0.6)
    update_multipliers: tuple[float, ...] = Field(min_length=3)
    implemented_update_multiplier: float = Field(ge=1.0, le=1.0)
    interpretation_role: Literal["descriptive_fragility_only_cannot_rescue_primary"]
    selection_note: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_grid(self) -> DependenceSensitivityGrid:
        if tuple(sorted(set(self.thresholds))) != self.thresholds:
            raise ValueError("Sensitivity thresholds must be unique and increasing")
        if tuple(sorted(set(self.update_multipliers))) != self.update_multipliers:
            raise ValueError("Update multipliers must be unique and increasing")
        if self.frozen_threshold not in self.thresholds:
            raise ValueError("Threshold grid must contain the frozen threshold")
        if self.implemented_update_multiplier not in self.update_multipliers:
            raise ValueError("Update grid must contain the implemented multiplier")
        if any(not 0.0 < value < 1.0 for value in self.thresholds):
            raise ValueError("Sensitivity thresholds must lie strictly within (0, 1)")
        if any(not 0.0 < value <= 2.0 for value in self.update_multipliers):
            raise ValueError("Update multipliers must lie within (0, 2]")
        return self


class GroupEffectSummary(ContractModel):
    """Primary effect within one declared cluster or retained subset."""

    group_type: Literal["concept", "misconception", "leave_one_concept_out"]
    group_id: str = Field(min_length=1)
    planned_case_count: int = Field(ge=1)
    eligible_case_count: int = Field(ge=1)
    missing_case_ids: tuple[str, ...]
    favourable_case_count: int = Field(ge=0)
    adverse_case_count: int = Field(ge=0)
    unchanged_case_count: int = Field(ge=0)
    mean_case_effect: float = Field(ge=-1.0, le=1.0)

    @model_validator(mode="after")
    def validate_counts(self) -> GroupEffectSummary:
        if self.planned_case_count != self.eligible_case_count + len(self.missing_case_ids):
            raise ValueError("Group case counts do not reconcile")
        if self.eligible_case_count != (
            self.favourable_case_count + self.adverse_case_count + self.unchanged_case_count
        ):
            raise ValueError("Group effect counts do not reconcile")
        return self


class HeuristicSensitivityPoint(ContractModel):
    """Three paired effects under one threshold and update multiplier."""

    threshold: float = Field(gt=0.0, lt=1.0)
    update_multiplier: float = Field(gt=0.0, le=2.0)
    eligible_case_count: int = Field(ge=1)
    dialogue_vs_probe_informed_effect: float = Field(ge=-1.0, le=1.0)
    dialogue_vs_probe_only_effect: float = Field(ge=-1.0, le=1.0)
    probe_informed_vs_probe_only_effect: float = Field(ge=-1.0, le=1.0)
    probe_informed_probe_only_disagreement_count: int = Field(ge=0)


class WilcoxonSensitivity(ContractModel):
    """Secondary signed-rank check, including its zero handling."""

    case_count: int = Field(ge=1)
    nonzero_case_count: int = Field(ge=1)
    zero_method: Literal["wilcox"] = "wilcox"
    method: Literal["auto"] = "auto"
    statistic: float = Field(ge=0.0)
    two_sided_p_value: float = Field(ge=0.0, le=1.0)


class DependenceSensitivityPlan(ContractModel):
    """Immutable provenance for one descriptive sensitivity run."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.dependence_sensitivity_plan.v1"] = (
        "benchmark.dependence_sensitivity_plan.v1"
    )
    benchmark_version: Literal["v1"] = "v1"
    run_id: str = Field(min_length=1)
    primary_analysis_plan_hash: Sha256
    primary_report_hash: Sha256
    secondary_report_hash: Sha256
    inferential_hierarchy_hash: Sha256
    benchmark_design_hash: Sha256
    repeat_publication_hash: Sha256
    sensitivity_grid_hash: Sha256
    analysis_code_revision: str = Field(min_length=1)
    analysis_pixi_lock_hash: Sha256
    created_at_utc: datetime
    plan_hash: Sha256

    @model_validator(mode="after")
    def validate_plan(self) -> DependenceSensitivityPlan:
        _require_utc(self.created_at_utc)
        if self.plan_hash != model_content_hash(self, exclude={"plan_hash"}):
            raise ValueError("Dependence-sensitivity plan hash does not match its content")
        return self


class DependenceSensitivityReport(ContractModel):
    """All predeclared descriptive fragility checks in one artifact."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.dependence_sensitivity_report.v1"] = (
        "benchmark.dependence_sensitivity_report.v1"
    )
    benchmark_version: Literal["v1"] = "v1"
    run_id: str = Field(min_length=1)
    analysis_plan_hash: Sha256
    concept_effects: tuple[GroupEffectSummary, ...] = Field(min_length=4, max_length=4)
    misconception_effects: tuple[GroupEffectSummary, ...] = Field(min_length=8, max_length=8)
    leave_one_concept_out: tuple[GroupEffectSummary, ...] = Field(min_length=4, max_length=4)
    threshold_sensitivity: tuple[HeuristicSensitivityPoint, ...] = Field(min_length=3)
    update_size_sensitivity: tuple[HeuristicSensitivityPoint, ...] = Field(min_length=3)
    primary_wilcoxon: WilcoxonSensitivity
    grid_values_selected_after_reveal: Literal[True] = True
    secondary_can_rescue_primary: Literal[False] = False
    repeat_variability_estimable: Literal[False] = False
    network_calls_made: Literal[0] = 0
    sandbox_calls_made: Literal[0] = 0
    completed_at_utc: datetime
    report_hash: Sha256

    @model_validator(mode="after")
    def validate_report(self) -> DependenceSensitivityReport:
        _require_utc(self.completed_at_utc)
        if self.report_hash != model_content_hash(self, exclude={"report_hash"}):
            raise ValueError("Dependence-sensitivity report hash does not match its content")
        return self


def run_dependence_sensitivity(
    *,
    grid_path: Path,
    primary_analysis_plan_path: Path,
    primary_report_path: Path,
    secondary_report_path: Path,
    inferential_hierarchy_path: Path,
    benchmark_design_path: Path,
    dataset_root: Path,
    pixi_lock_path: Path,
    output_root: Path,
    analysis_code_revision: str,
    created_at_utc: datetime,
) -> DependenceSensitivityReport:
    """Run every declared dependence and heuristic fragility check."""

    grid = load_command_model(grid_path, DependenceSensitivityGrid)
    primary_plan = _load_json(primary_analysis_plan_path, PrimaryAnalysisPlan)
    primary = _load_json(primary_report_path, PrimaryAnalysisReport)
    secondary = _load_json(secondary_report_path, SecondaryAnalysisReport)
    hierarchy = _load_json(inferential_hierarchy_path, InferentialHierarchy)
    design = load_design(benchmark_design_path)
    try:
        lock_hash = file_sha256(pixi_lock_path.read_bytes())
    except OSError as error:
        raise DependenceSensitivityError(f"Could not read Pixi lock: {pixi_lock_path}") from error
    published = read_repeat_metrics(AtomicParquetDatasetStore(dataset_root))
    rows = published.table.to_pylist()
    _validate_sources(
        primary=primary,
        primary_plan=primary_plan,
        secondary=secondary,
        hierarchy=hierarchy,
        design=design,
        repeat_publication_hash=published.manifest.publication_hash,
        repeat_identity=(published.manifest.benchmark_version, published.manifest.run_id),
        rows=rows,
        created_at_utc=created_at_utc,
    )
    plan = _create_plan(
        grid=grid,
        primary_plan=primary_plan,
        primary=primary,
        secondary=secondary,
        hierarchy=hierarchy,
        design=design,
        repeat_publication_hash=published.manifest.publication_hash,
        lock_hash=lock_hash,
        analysis_code_revision=analysis_code_revision,
        created_at_utc=created_at_utc,
    )
    root = output_root.resolve()
    report_path = root / "dependence_sensitivity_report.json"
    if report_path.exists():
        report = _load_json(report_path, DependenceSensitivityReport)
        if report.analysis_plan_hash != plan.plan_hash:
            raise DependenceSensitivityError("Existing sensitivity report belongs to another plan")
        return report
    write_immutable_json(root / "dependence_sensitivity_plan.json", plan)

    effects = {item.case_id: item.case_effect for item in primary.case_results}
    missing = {item.case_id for item in primary.missing_cases}
    concepts = _group_effects(design, effects, missing, group_type="concept")
    misconceptions = _group_effects(design, effects, missing, group_type="misconception")
    leave_one_out = _leave_one_concept_out(design, effects, missing)
    scores = _score_rows_by_case(
        rows, expected_missing={item.case_id for item in primary.missing_cases}
    )
    thresholds = tuple(
        _sensitivity_point(scores, threshold=value, update_multiplier=1.0)
        for value in grid.thresholds
    )
    updates = tuple(
        _sensitivity_point(
            scores,
            threshold=grid.frozen_threshold,
            update_multiplier=value,
        )
        for value in grid.update_multipliers
    )
    wilcoxon_result = _wilcoxon(tuple(effects.values()))
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.dependence_sensitivity_report.v1",
        "benchmark_version": "v1",
        "run_id": primary.run_id,
        "analysis_plan_hash": plan.plan_hash,
        "concept_effects": concepts,
        "misconception_effects": misconceptions,
        "leave_one_concept_out": leave_one_out,
        "threshold_sensitivity": thresholds,
        "update_size_sensitivity": updates,
        "primary_wilcoxon": wilcoxon_result,
        "grid_values_selected_after_reveal": True,
        "secondary_can_rescue_primary": False,
        "repeat_variability_estimable": False,
        "network_calls_made": 0,
        "sandbox_calls_made": 0,
        "completed_at_utc": created_at_utc,
    }
    draft = DependenceSensitivityReport.model_construct(
        _fields_set=set(content), **content, report_hash="0" * 64
    )
    report = DependenceSensitivityReport.model_validate(
        {**content, "report_hash": model_content_hash(draft, exclude={"report_hash"})}
    )
    write_immutable_json(report_path, report)
    return report


def _group_effects(
    design: BenchmarkDesignPlan,
    effects: dict[str, float],
    missing: set[str],
    *,
    group_type: Literal["concept", "misconception"],
) -> tuple[GroupEffectSummary, ...]:
    grouped: defaultdict[str, list[str]] = defaultdict(list)
    for case in design.held_out_cases:
        key = case.concept_id if group_type == "concept" else case.misconception_id
        grouped[key].append(case.case_id)
    return tuple(
        _group_summary(group_type, group_id, case_ids, effects, missing)
        for group_id, case_ids in sorted(grouped.items())
    )


def _leave_one_concept_out(
    design: BenchmarkDesignPlan,
    effects: dict[str, float],
    missing: set[str],
) -> tuple[GroupEffectSummary, ...]:
    concepts = sorted({case.concept_id for case in design.held_out_cases})
    return tuple(
        _group_summary(
            "leave_one_concept_out",
            omitted,
            [case.case_id for case in design.held_out_cases if case.concept_id != omitted],
            effects,
            missing,
        )
        for omitted in concepts
    )


def _group_summary(
    group_type: Literal["concept", "misconception", "leave_one_concept_out"],
    group_id: str,
    case_ids: list[str],
    effects: dict[str, float],
    missing: set[str],
) -> GroupEffectSummary:
    eligible = tuple(effects[case_id] for case_id in case_ids if case_id in effects)
    missing_ids = tuple(sorted(set(case_ids) & missing))
    return GroupEffectSummary(
        group_type=group_type,
        group_id=group_id,
        planned_case_count=len(case_ids),
        eligible_case_count=len(eligible),
        missing_case_ids=missing_ids,
        favourable_case_count=sum(value > 0 for value in eligible),
        adverse_case_count=sum(value < 0 for value in eligible),
        unchanged_case_count=sum(value == 0 for value in eligible),
        mean_case_effect=fmean(eligible),
    )


def _score_rows_by_case(
    rows: list[dict[str, Any]],
    *,
    expected_missing: set[str],
) -> dict[str, dict[RepeatEstimator, dict[str, Any]]]:
    grouped: defaultdict[str, dict[RepeatEstimator, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        case_id = str(row["case_id"])
        estimator = RepeatEstimator(str(row["estimator"]))
        if estimator in grouped[case_id]:
            raise DependenceSensitivityError("Sensitivity data contain duplicate estimators")
        grouped[case_id][estimator] = row
    expected = set(RepeatEstimator)
    if len(grouped) != 24 or any(set(case_rows) != expected for case_rows in grouped.values()):
        raise DependenceSensitivityError("Sensitivity requires six estimators for all 24 cases")
    missing: set[str] = set()
    for case_id, case_rows in grouped.items():
        outcomes = {row["criterion_demonstrated_performance"] for row in case_rows.values()}
        if len(outcomes) != 1:
            raise DependenceSensitivityError("Estimator rows disagree on the criterion outcome")
        if outcomes == {None}:
            missing.add(case_id)
    if missing != expected_missing:
        raise DependenceSensitivityError("Sensitivity rows have another missing-case identity")
    return dict(grouped)


def _sensitivity_point(
    scores: dict[str, dict[RepeatEstimator, dict[str, Any]]],
    *,
    threshold: float,
    update_multiplier: float,
) -> HeuristicSensitivityPoint:
    effects: list[tuple[float, float, float, bool]] = []
    for rows in scores.values():
        outcome = rows[RepeatEstimator.DIALOGUE_ONLY]["criterion_demonstrated_performance"]
        if outcome is None:
            continue
        dialogue_original = float(rows[RepeatEstimator.DIALOGUE_ONLY]["score"])
        informed_original = float(rows[RepeatEstimator.PROBE_INFORMED]["score"])
        probe_only_original = float(rows[RepeatEstimator.PROBE_ONLY]["score"])
        dialogue = _clamp(0.5 + update_multiplier * (dialogue_original - 0.5))
        informed = _clamp(dialogue + update_multiplier * (informed_original - dialogue_original))
        probe_only = _clamp(0.5 + update_multiplier * (probe_only_original - 0.5))
        dialogue_error = float((dialogue >= threshold) is not outcome)
        informed_error = float((informed >= threshold) is not outcome)
        probe_only_error = float((probe_only >= threshold) is not outcome)
        effects.append(
            (
                dialogue_error - informed_error,
                dialogue_error - probe_only_error,
                probe_only_error - informed_error,
                (informed >= threshold) is not (probe_only >= threshold),
            )
        )
    return HeuristicSensitivityPoint(
        threshold=threshold,
        update_multiplier=update_multiplier,
        eligible_case_count=len(effects),
        dialogue_vs_probe_informed_effect=fmean(item[0] for item in effects),
        dialogue_vs_probe_only_effect=fmean(item[1] for item in effects),
        probe_informed_vs_probe_only_effect=fmean(item[2] for item in effects),
        probe_informed_probe_only_disagreement_count=sum(item[3] for item in effects),
    )


def _wilcoxon(effects: tuple[float, ...]) -> WilcoxonSensitivity:
    result: Any = wilcoxon(
        effects,
        zero_method="wilcox",
        alternative="two-sided",
        method="auto",
    )
    return WilcoxonSensitivity(
        case_count=len(effects),
        nonzero_case_count=sum(value != 0 for value in effects),
        statistic=float(result.statistic),
        two_sided_p_value=float(result.pvalue),
    )


def _validate_sources(
    *,
    primary: PrimaryAnalysisReport,
    primary_plan: PrimaryAnalysisPlan,
    secondary: SecondaryAnalysisReport,
    hierarchy: InferentialHierarchy,
    design: BenchmarkDesignPlan,
    repeat_publication_hash: Sha256,
    repeat_identity: tuple[str, str],
    rows: list[dict[str, Any]],
    created_at_utc: datetime,
) -> None:
    _require_utc(created_at_utc)
    if created_at_utc <= secondary.completed_at_utc:
        raise DependenceSensitivityError("Sensitivity analysis must follow secondary analysis")
    if secondary.primary_analysis_report_hash != primary.report_hash:
        raise DependenceSensitivityError("Secondary report belongs to another primary result")
    if primary.analysis_plan_hash != primary_plan.plan_hash:
        raise DependenceSensitivityError("Primary report belongs to another primary plan")
    if primary_plan.repeat_publication_hash != repeat_publication_hash:
        raise DependenceSensitivityError("Repeat dataset differs from the primary plan")
    if primary.run_id != secondary.run_id or repeat_identity != ("v1", primary.run_id):
        raise DependenceSensitivityError("Sensitivity sources belong to another run")
    if hierarchy.benchmark_design_hash != model_content_hash(design):
        raise DependenceSensitivityError("Design differs from the frozen hierarchy")
    required = {
        "wilcoxon_signed_rank",
        "policy_threshold_sensitivity",
        "tracker_update_size_sensitivity",
        "misconception_group_summary",
        "concept_group_summary",
        "leave_one_concept_out",
    }
    if not required.issubset(set(hierarchy.secondary_analyses)):
        raise DependenceSensitivityError("Hierarchy omits required sensitivity analyses")
    if len(rows) != 144 or repeat_publication_hash == "0" * 64:
        raise DependenceSensitivityError("Sensitivity requires the published 144-row dataset")


def _create_plan(
    *,
    grid: DependenceSensitivityGrid,
    primary_plan: PrimaryAnalysisPlan,
    primary: PrimaryAnalysisReport,
    secondary: SecondaryAnalysisReport,
    hierarchy: InferentialHierarchy,
    design: BenchmarkDesignPlan,
    repeat_publication_hash: Sha256,
    lock_hash: Sha256,
    analysis_code_revision: str,
    created_at_utc: datetime,
) -> DependenceSensitivityPlan:
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.dependence_sensitivity_plan.v1",
        "benchmark_version": "v1",
        "run_id": primary.run_id,
        "primary_analysis_plan_hash": primary_plan.plan_hash,
        "primary_report_hash": primary.report_hash,
        "secondary_report_hash": secondary.report_hash,
        "inferential_hierarchy_hash": hierarchy.hierarchy_hash,
        "benchmark_design_hash": model_content_hash(design),
        "repeat_publication_hash": repeat_publication_hash,
        "sensitivity_grid_hash": model_content_hash(grid),
        "analysis_code_revision": analysis_code_revision,
        "analysis_pixi_lock_hash": lock_hash,
        "created_at_utc": created_at_utc,
    }
    draft = DependenceSensitivityPlan.model_construct(
        _fields_set=set(content), **content, plan_hash="0" * 64
    )
    return DependenceSensitivityPlan.model_validate(
        {**content, "plan_hash": model_content_hash(draft, exclude={"plan_hash"})}
    )


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))


def _load_json[ModelT: ContractModel](path: Path, model: type[ModelT]) -> ModelT:
    try:
        return model.model_validate_json(path.read_bytes())
    except (OSError, ValidationError) as error:
        message = f"Could not verify dependence-sensitivity source: {path}"
        raise DependenceSensitivityError(message) from error


def _require_utc(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("Dependence-sensitivity timestamp must be UTC")
