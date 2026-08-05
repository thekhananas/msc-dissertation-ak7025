"""Deterministic bounds for missing outcomes in the external primary comparison."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import Field, ValidationError, model_validator

from socratic_tutor.benchmark.artifacts import write_immutable_json
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.evaluator.datasets import read_repeat_metrics
from socratic_tutor.benchmark.evaluator.scoring import RepeatEstimator
from socratic_tutor.benchmark.hashing import file_sha256, model_content_hash
from socratic_tutor.benchmark.parquet import AtomicParquetDatasetStore
from socratic_tutor.benchmark.primary_analysis import PrimaryAnalysisPlan, PrimaryAnalysisReport
from socratic_tutor.contracts import ContractModel


class MissingnessSensitivityError(ValueError):
    """Missingness-bound inputs differ from the frozen primary analysis."""


type BoundAssignment = Literal["criterion_false", "criterion_true", "either"]


class MissingCaseBound(ContractModel):
    """Both possible contributions from one unavailable binary criterion outcome."""

    case_id: str = Field(min_length=1)
    missing_reason: str = Field(min_length=1)
    dialogue_binary_decision: bool
    valid_evidence_binary_decision: bool
    effect_if_criterion_false: Literal[-1, 0, 1]
    effect_if_criterion_true: Literal[-1, 0, 1]
    lower_bound_contribution: Literal[-1, 0, 1]
    upper_bound_contribution: Literal[-1, 0, 1]
    lower_bound_assignment: BoundAssignment
    upper_bound_assignment: BoundAssignment
    dialogue_repeat_hash: Sha256
    valid_evidence_repeat_hash: Sha256
    criterion_record_hash: Sha256

    @model_validator(mode="after")
    def validate_bound(self) -> MissingCaseBound:
        effects = (self.effect_if_criterion_false, self.effect_if_criterion_true)
        if self.lower_bound_contribution != min(effects):
            raise ValueError("Missing-case lower contribution is not the minimum possible effect")
        if self.upper_bound_contribution != max(effects):
            raise ValueError("Missing-case upper contribution is not the maximum possible effect")
        if self.lower_bound_assignment != _assignment_for(effects, lower=True):
            raise ValueError("Missing-case lower assignment differs from its possible effects")
        if self.upper_bound_assignment != _assignment_for(effects, lower=False):
            raise ValueError("Missing-case upper assignment differs from its possible effects")
        return self


class MissingnessSensitivityPlan(ContractModel):
    """Source and software identities for a read-only missingness analysis."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.missingness_sensitivity_plan.v1"] = (
        "benchmark.missingness_sensitivity_plan.v1"
    )
    benchmark_version: Literal["v1"] = "v1"
    run_id: str = Field(min_length=1)
    primary_analysis_plan_hash: Sha256
    primary_analysis_report_hash: Sha256
    repeat_publication_hash: Sha256
    method: Literal["all_binary_outcomes_case_effect_bounds_v1"] = (
        "all_binary_outcomes_case_effect_bounds_v1"
    )
    analysis_code_revision: str = Field(min_length=1)
    analysis_pixi_lock_hash: Sha256
    created_at_utc: datetime
    plan_hash: Sha256

    @model_validator(mode="after")
    def validate_plan(self) -> MissingnessSensitivityPlan:
        _require_utc(self.created_at_utc)
        if self.plan_hash != model_content_hash(self, exclude={"plan_hash"}):
            raise ValueError("Missingness-sensitivity plan hash does not match its content")
        return self


class MissingnessSensitivityReport(ContractModel):
    """Complete-case result and exact bounds over every missing binary outcome."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.missingness_sensitivity_report.v1"] = (
        "benchmark.missingness_sensitivity_report.v1"
    )
    benchmark_version: Literal["v1"] = "v1"
    run_id: str = Field(min_length=1)
    analysis_plan_hash: Sha256
    total_case_count: Literal[24]
    complete_case_count: int = Field(ge=1, le=24)
    missing_case_count: int = Field(ge=1, le=24)
    missing_case_bounds: tuple[MissingCaseBound, ...] = Field(min_length=1)
    complete_case_net_improvement_count: int
    complete_case_effect: float = Field(ge=-1.0, le=1.0)
    lower_bound_net_improvement_count: int
    upper_bound_net_improvement_count: int
    lower_bound_full_corpus_effect: float = Field(ge=-1.0, le=1.0)
    upper_bound_full_corpus_effect: float = Field(ge=-1.0, le=1.0)
    bound_width: float = Field(ge=0.0, le=2.0)
    complete_case_remains_primary: Literal[True] = True
    sensitivity_scope: Literal["fixed_authored_corpus_only"] = "fixed_authored_corpus_only"
    network_calls_made: Literal[0] = 0
    sandbox_calls_made: Literal[0] = 0
    completed_at_utc: datetime
    report_hash: Sha256

    @model_validator(mode="after")
    def validate_report(self) -> MissingnessSensitivityReport:
        _require_utc(self.completed_at_utc)
        if self.total_case_count != self.complete_case_count + self.missing_case_count:
            raise ValueError("Missingness-sensitivity case counts do not reconcile")
        if self.missing_case_count != len(self.missing_case_bounds):
            raise ValueError("Missingness count differs from its case records")
        if len({record.case_id for record in self.missing_case_bounds}) != self.missing_case_count:
            raise ValueError("Missingness bounds contain duplicate case identities")
        expected_complete = self.complete_case_net_improvement_count / self.complete_case_count
        expected_lower = self.lower_bound_net_improvement_count / self.total_case_count
        expected_upper = self.upper_bound_net_improvement_count / self.total_case_count
        if abs(self.complete_case_effect - expected_complete) > 1e-12:
            raise ValueError("Complete-case effect does not reconcile")
        if abs(self.lower_bound_full_corpus_effect - expected_lower) > 1e-12:
            raise ValueError("Lower missingness bound does not reconcile")
        if abs(self.upper_bound_full_corpus_effect - expected_upper) > 1e-12:
            raise ValueError("Upper missingness bound does not reconcile")
        if self.lower_bound_full_corpus_effect > self.upper_bound_full_corpus_effect:
            raise ValueError("Missingness lower bound exceeds upper bound")
        if (
            abs(
                self.bound_width
                - (self.upper_bound_full_corpus_effect - self.lower_bound_full_corpus_effect)
            )
            > 1e-12
        ):
            raise ValueError("Missingness bound width does not reconcile")
        if self.report_hash != model_content_hash(self, exclude={"report_hash"}):
            raise ValueError("Missingness-sensitivity report hash does not match its content")
        return self


def run_missingness_sensitivity(
    *,
    primary_analysis_plan_path: Path,
    primary_report_path: Path,
    dataset_root: Path,
    pixi_lock_path: Path,
    output_root: Path,
    analysis_code_revision: str,
    created_at_utc: datetime,
) -> MissingnessSensitivityReport:
    """Bound the primary effect over every possible missing binary criterion outcome."""

    _require_utc(created_at_utc)
    primary_plan = _load_json(primary_analysis_plan_path, PrimaryAnalysisPlan)
    primary = _load_json(primary_report_path, PrimaryAnalysisReport)
    if primary.analysis_plan_hash != primary_plan.plan_hash:
        raise MissingnessSensitivityError("Primary report belongs to another analysis plan")
    if not primary.missing_cases:
        raise MissingnessSensitivityError("Missingness analysis requires at least one missing case")
    if created_at_utc <= primary.completed_at_utc:
        raise MissingnessSensitivityError("Missingness analysis must follow the primary result")
    try:
        pixi_lock_hash = file_sha256(pixi_lock_path.read_bytes())
    except OSError as error:
        raise MissingnessSensitivityError(f"Could not read Pixi lock: {pixi_lock_path}") from error

    repeats = read_repeat_metrics(AtomicParquetDatasetStore(dataset_root))
    if repeats.manifest.publication_hash != primary_plan.repeat_publication_hash:
        raise MissingnessSensitivityError("Repeat metrics differ from the primary analysis plan")
    if (
        repeats.manifest.benchmark_version != primary.benchmark_version
        or repeats.manifest.run_id != primary.run_id
    ):
        raise MissingnessSensitivityError("Repeat metrics belong to another benchmark run")

    plan = _create_plan(
        primary_plan=primary_plan,
        primary=primary,
        repeat_publication_hash=repeats.manifest.publication_hash,
        pixi_lock_hash=pixi_lock_hash,
        analysis_code_revision=analysis_code_revision,
        created_at_utc=created_at_utc,
    )
    root = output_root.resolve()
    report_path = root / "missingness_sensitivity_report.json"
    if report_path.exists():
        report = _load_json(report_path, MissingnessSensitivityReport)
        if report.analysis_plan_hash != plan.plan_hash:
            raise MissingnessSensitivityError("Existing missingness report belongs to another plan")
        return report
    write_immutable_json(root / "missingness_sensitivity_plan.json", plan)

    bounds = _missing_case_bounds(primary, repeats.table.to_pylist())
    lower_net = primary.net_improvement_count + sum(
        record.lower_bound_contribution for record in bounds
    )
    upper_net = primary.net_improvement_count + sum(
        record.upper_bound_contribution for record in bounds
    )
    lower_effect = lower_net / primary.total_case_count
    upper_effect = upper_net / primary.total_case_count
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.missingness_sensitivity_report.v1",
        "benchmark_version": primary.benchmark_version,
        "run_id": primary.run_id,
        "analysis_plan_hash": plan.plan_hash,
        "total_case_count": primary.total_case_count,
        "complete_case_count": primary.eligible_case_count,
        "missing_case_count": primary.missing_case_count,
        "missing_case_bounds": bounds,
        "complete_case_net_improvement_count": primary.net_improvement_count,
        "complete_case_effect": primary.paired_effect_from_counts,
        "lower_bound_net_improvement_count": lower_net,
        "upper_bound_net_improvement_count": upper_net,
        "lower_bound_full_corpus_effect": lower_effect,
        "upper_bound_full_corpus_effect": upper_effect,
        "bound_width": upper_effect - lower_effect,
        "complete_case_remains_primary": True,
        "sensitivity_scope": "fixed_authored_corpus_only",
        "network_calls_made": 0,
        "sandbox_calls_made": 0,
        "completed_at_utc": created_at_utc,
    }
    draft = MissingnessSensitivityReport.model_construct(
        _fields_set=set(content), **content, report_hash="0" * 64
    )
    report = MissingnessSensitivityReport.model_validate(
        {**content, "report_hash": model_content_hash(draft, exclude={"report_hash"})}
    )
    write_immutable_json(report_path, report)
    return report


def _create_plan(
    *,
    primary_plan: PrimaryAnalysisPlan,
    primary: PrimaryAnalysisReport,
    repeat_publication_hash: Sha256,
    pixi_lock_hash: Sha256,
    analysis_code_revision: str,
    created_at_utc: datetime,
) -> MissingnessSensitivityPlan:
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.missingness_sensitivity_plan.v1",
        "benchmark_version": primary.benchmark_version,
        "run_id": primary.run_id,
        "primary_analysis_plan_hash": primary_plan.plan_hash,
        "primary_analysis_report_hash": primary.report_hash,
        "repeat_publication_hash": repeat_publication_hash,
        "method": "all_binary_outcomes_case_effect_bounds_v1",
        "analysis_code_revision": analysis_code_revision,
        "analysis_pixi_lock_hash": pixi_lock_hash,
        "created_at_utc": created_at_utc,
    }
    draft = MissingnessSensitivityPlan.model_construct(
        _fields_set=set(content), **content, plan_hash="0" * 64
    )
    return MissingnessSensitivityPlan.model_validate(
        {**content, "plan_hash": model_content_hash(draft, exclude={"plan_hash"})}
    )


def _missing_case_bounds(
    primary: PrimaryAnalysisReport,
    repeat_rows: list[dict[str, Any]],
) -> tuple[MissingCaseBound, ...]:
    missing_by_id = {record.case_id: record for record in primary.missing_cases}
    relevant = [
        row
        for row in repeat_rows
        if row["case_id"] in missing_by_id
        and row["estimator"]
        in {RepeatEstimator.DIALOGUE_ONLY.value, RepeatEstimator.PROBE_INFORMED.value}
    ]
    by_case: dict[str, dict[str, dict[str, Any]]] = {}
    for row in relevant:
        estimator = str(row["estimator"])
        if estimator in by_case.setdefault(str(row["case_id"]), {}):
            raise MissingnessSensitivityError("Missing case contains duplicate primary estimators")
        by_case[str(row["case_id"])][estimator] = row
    if set(by_case) != set(missing_by_id):
        raise MissingnessSensitivityError("Repeat metrics lack a reported missing case")

    bounds: list[MissingCaseBound] = []
    for case_id in sorted(missing_by_id):
        missing = missing_by_id[case_id]
        rows = by_case[case_id]
        if set(rows) != {
            RepeatEstimator.DIALOGUE_ONLY.value,
            RepeatEstimator.PROBE_INFORMED.value,
        }:
            raise MissingnessSensitivityError("Missing case lacks its two primary estimators")
        dialogue = rows[RepeatEstimator.DIALOGUE_ONLY.value]
        valid = rows[RepeatEstimator.PROBE_INFORMED.value]
        for row in (dialogue, valid):
            if (
                row["eligible"] is not False
                or row["criterion_demonstrated_performance"] is not None
                or row["classification_error"] is not None
                or row["missing_reason"] != missing.missing_reason
                or row["criterion_record_hash"] != missing.criterion_record_hash
            ):
                raise MissingnessSensitivityError(
                    "Missing repeat conflicts with the primary report"
                )
        dialogue_decision = _bool(row=dialogue, field="binary_decision")
        valid_decision = _bool(row=valid, field="binary_decision")
        effects = (
            _case_effect(dialogue_decision, valid_decision, criterion=False),
            _case_effect(dialogue_decision, valid_decision, criterion=True),
        )
        bounds.append(
            MissingCaseBound(
                case_id=case_id,
                missing_reason=missing.missing_reason,
                dialogue_binary_decision=dialogue_decision,
                valid_evidence_binary_decision=valid_decision,
                effect_if_criterion_false=effects[0],
                effect_if_criterion_true=effects[1],
                lower_bound_contribution=min(effects),
                upper_bound_contribution=max(effects),
                lower_bound_assignment=_assignment_for(effects, lower=True),
                upper_bound_assignment=_assignment_for(effects, lower=False),
                dialogue_repeat_hash=str(dialogue["record_hash"]),
                valid_evidence_repeat_hash=str(valid["record_hash"]),
                criterion_record_hash=missing.criterion_record_hash,
            )
        )
    return tuple(bounds)


def _case_effect(dialogue: bool, valid: bool, *, criterion: bool) -> Literal[-1, 0, 1]:
    effect = int(dialogue is not criterion) - int(valid is not criterion)
    if effect not in {-1, 0, 1}:  # pragma: no cover - arithmetic invariant
        raise AssertionError("Binary case effect escaped its valid range")
    return cast(Literal[-1, 0, 1], effect)


def _assignment_for(effects: tuple[int, int], *, lower: bool) -> BoundAssignment:
    if effects[0] == effects[1]:
        return "either"
    selected = min(effects) if lower else max(effects)
    return "criterion_false" if effects[0] == selected else "criterion_true"


def _bool(*, row: dict[str, Any], field: str) -> bool:
    value = row[field]
    if type(value) is not bool:
        raise MissingnessSensitivityError(f"Missing repeat {field} must be boolean")
    return value


def _load_json(path: Path, model: type[ContractModel]) -> Any:
    try:
        return model.model_validate_json(path.read_bytes())
    except (OSError, ValidationError) as error:
        raise MissingnessSensitivityError(f"Could not verify missingness source: {path}") from error


def _require_utc(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("Missingness-sensitivity timestamp must be UTC")
