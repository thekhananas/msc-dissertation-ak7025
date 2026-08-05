# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
"""Frozen primary analysis for one scored external benchmark run."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, ValidationError, model_validator

from socratic_tutor.benchmark.analysis_spec import (
    AnalysisSpecification,
    analysis_specification_hash,
    load_analysis_specification,
)
from socratic_tutor.benchmark.artifacts import write_immutable_json
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.evaluator.aggregation import CaseComparison
from socratic_tutor.benchmark.evaluator.datasets import (
    read_case_aggregates,
    read_criterion_records,
    read_repeat_metrics,
)
from socratic_tutor.benchmark.evaluator.scoring import RepeatEstimator
from socratic_tutor.benchmark.external_scoring import ExternalScoringSummary
from socratic_tutor.benchmark.hashing import file_sha256, model_content_hash
from socratic_tutor.benchmark.inferential_hierarchy import InferentialHierarchy
from socratic_tutor.benchmark.parquet import AtomicParquetDatasetStore
from socratic_tutor.benchmark.statistics import (
    McNemarResult,
    PairedCaseInference,
    exact_mcnemar,
    paired_case_inference,
)
from socratic_tutor.contracts import ContractModel


class PrimaryAnalysisError(ValueError):
    """Primary-analysis inputs violate the frozen paired design."""


class PrimaryCaseResult(ContractModel):
    """One eligible case's hand-checkable paired binary result."""

    case_id: str = Field(min_length=1)
    criterion_outcome: bool
    dialogue_binary_decision: bool
    valid_evidence_binary_decision: bool
    dialogue_correct: bool
    valid_evidence_correct: bool
    dialogue_classification_error: float = Field(ge=0.0, le=1.0)
    valid_evidence_classification_error: float = Field(ge=0.0, le=1.0)
    case_effect: float = Field(ge=-1.0, le=1.0)
    aggregate_record_hash: Sha256
    dialogue_repeat_hash: Sha256
    valid_evidence_repeat_hash: Sha256
    criterion_record_hash: Sha256

    @model_validator(mode="after")
    def validate_case(self) -> PrimaryCaseResult:
        if self.dialogue_classification_error not in {0.0, 1.0}:
            raise ValueError("Dialogue classification error must be binary")
        if self.valid_evidence_classification_error not in {0.0, 1.0}:
            raise ValueError("Valid-evidence classification error must be binary")
        if self.case_effect not in {-1.0, 0.0, 1.0}:
            raise ValueError("Primary case effect must be -1, 0, or 1")
        if self.dialogue_correct is not (self.dialogue_binary_decision is self.criterion_outcome):
            raise ValueError("Dialogue correctness differs from its paired outcome")
        if self.valid_evidence_correct is not (
            self.valid_evidence_binary_decision is self.criterion_outcome
        ):
            raise ValueError("Valid-evidence correctness differs from its paired outcome")
        if self.dialogue_classification_error != float(not self.dialogue_correct):
            raise ValueError("Dialogue error differs from dialogue correctness")
        if self.valid_evidence_classification_error != float(not self.valid_evidence_correct):
            raise ValueError("Valid-evidence error differs from valid-evidence correctness")
        expected = self.dialogue_classification_error - self.valid_evidence_classification_error
        if self.case_effect != expected:
            raise ValueError("Primary case effect differs from paired classification errors")
        return self


class MissingPrimaryCase(ContractModel):
    """One retained case without an invented criterion outcome."""

    case_id: str = Field(min_length=1)
    missing_reason: str = Field(min_length=1)
    aggregate_record_hash: Sha256
    criterion_record_hash: Sha256


class PrimaryAnalysisPlan(ContractModel):
    """Frozen sources and software identity used for one primary result."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.primary_analysis_plan.v1"] = "benchmark.primary_analysis_plan.v1"
    benchmark_version: Literal["v1"] = "v1"
    run_id: str = Field(min_length=1)
    scoring_summary_hash: Sha256
    repeat_publication_hash: Sha256
    aggregate_publication_hash: Sha256
    analysis_specification_hash: Sha256
    inferential_hierarchy_hash: Sha256
    primary_comparison: Literal["primary_valid"] = "primary_valid"
    primary_estimand: Literal["mean_paired_classification_error_difference_by_case"]
    primary_test: Literal["exact_two_sided_mcnemar"]
    bootstrap_resamples: int = Field(ge=1000)
    random_seed: int = Field(ge=0, le=2**32 - 1)
    minimum_interpretable_effect: float = Field(gt=0.0, le=1.0)
    analysis_code_revision: str = Field(min_length=1)
    analysis_pixi_lock_hash: Sha256
    created_at_utc: datetime
    plan_hash: Sha256

    @model_validator(mode="after")
    def validate_plan(self) -> PrimaryAnalysisPlan:
        _require_utc(self.created_at_utc)
        if self.plan_hash != model_content_hash(self, exclude={"plan_hash"}):
            raise ValueError("Primary analysis-plan hash does not match its content")
        return self


class PrimaryAnalysisReport(ContractModel):
    """Primary effect, uncertainty, and exact paired test under frozen rules."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.primary_analysis_report.v1"] = (
        "benchmark.primary_analysis_report.v1"
    )
    benchmark_version: Literal["v1"] = "v1"
    run_id: str = Field(min_length=1)
    analysis_plan_hash: Sha256
    total_case_count: Literal[24]
    eligible_case_count: int = Field(ge=1, le=24)
    missing_case_count: int = Field(ge=0, le=24)
    case_results: tuple[PrimaryCaseResult, ...] = Field(min_length=1)
    missing_cases: tuple[MissingPrimaryCase, ...]
    inference: PairedCaseInference
    mcnemar: McNemarResult
    valid_evidence_improvement_count: int = Field(ge=0)
    valid_evidence_regression_count: int = Field(ge=0)
    net_improvement_count: int
    paired_effect_from_counts: float = Field(ge=-1.0, le=1.0)
    interval_threshold_status: Literal[
        "interval_above_minimum",
        "interval_crosses_minimum",
        "interval_below_minimum",
    ]
    mcnemar_rejects_equal_discordance: bool
    mcnemar_alpha: float = Field(default=0.05, ge=0.05, le=0.05)
    mcnemar_scope: Literal["tests_equality_of_discordant_probabilities"] = (
        "tests_equality_of_discordant_probabilities"
    )
    threshold_scope: Literal["tests_whether_mean_reaches_frozen_practical_threshold"] = (
        "tests_whether_mean_reaches_frozen_practical_threshold"
    )
    significance_is_not_success_gate: Literal[True] = True
    sign_swap_role: Literal["secondary_sensitivity_only"] = "secondary_sensitivity_only"
    interval_scope: Literal["conditional_on_this_fixed_authored_corpus"] = (
        "conditional_on_this_fixed_authored_corpus"
    )
    repeat_variability: Literal["not_estimable_with_one_model_run_per_case"] = (
        "not_estimable_with_one_model_run_per_case"
    )
    completed_at_utc: datetime
    report_hash: Sha256

    @model_validator(mode="after")
    def validate_report(self) -> PrimaryAnalysisReport:
        _require_utc(self.completed_at_utc)
        if self.total_case_count != self.eligible_case_count + self.missing_case_count:
            raise ValueError("Primary case counts do not reconcile")
        if self.eligible_case_count != len(self.case_results):
            raise ValueError("Eligible count differs from primary case records")
        if self.missing_case_count != len(self.missing_cases):
            raise ValueError("Missing count differs from missing case records")
        all_case_ids = tuple(record.case_id for record in (*self.case_results, *self.missing_cases))
        if len(set(all_case_ids)) != self.total_case_count:
            raise ValueError("Primary report does not contain 24 unique case identities")
        if self.inference.case_count != self.eligible_case_count:
            raise ValueError("Primary inference uses another case denominator")
        if self.mcnemar.pair_count != self.eligible_case_count:
            raise ValueError("McNemar result uses another case denominator")
        if self.valid_evidence_improvement_count != self.mcnemar.right_only_correct_count:
            raise ValueError("Improvement count differs from the paired table")
        if self.valid_evidence_regression_count != self.mcnemar.left_only_correct_count:
            raise ValueError("Regression count differs from the paired table")
        if self.net_improvement_count != (
            self.valid_evidence_improvement_count - self.valid_evidence_regression_count
        ):
            raise ValueError("Net improvement count does not reconcile")
        expected_effect = self.net_improvement_count / self.eligible_case_count
        if abs(self.paired_effect_from_counts - expected_effect) > 1e-12:
            raise ValueError("Count-derived primary effect does not reconcile")
        if abs(self.paired_effect_from_counts - self.inference.mean_case_effect) > 1e-12:
            raise ValueError("Count-derived and case-mean effects differ")
        expected_rejection = self.mcnemar.two_sided_exact_p_value < self.mcnemar_alpha
        if self.mcnemar_rejects_equal_discordance is not expected_rejection:
            raise ValueError("McNemar rejection flag differs from its exact p-value")
        if self.interval_threshold_status != _interval_threshold_status(self.inference):
            raise ValueError("Interval-threshold status differs from the primary interval")
        if self.report_hash != model_content_hash(self, exclude={"report_hash"}):
            raise ValueError("Primary analysis-report hash does not match its content")
        return self


def run_primary_analysis(
    *,
    scoring_summary_path: Path,
    analysis_specification_path: Path,
    inferential_hierarchy_path: Path,
    dataset_root: Path,
    pixi_lock_path: Path,
    output_root: Path,
    analysis_code_revision: str,
    created_at_utc: datetime,
) -> PrimaryAnalysisReport:
    """Calculate the frozen primary comparison from verified published rows."""

    _require_utc(created_at_utc)
    scoring = _load_json(scoring_summary_path, ExternalScoringSummary)
    specification = load_analysis_specification(analysis_specification_path)
    hierarchy = _load_json(inferential_hierarchy_path, InferentialHierarchy)
    try:
        pixi_lock_hash = file_sha256(pixi_lock_path.read_bytes())
    except OSError as error:
        raise PrimaryAnalysisError(f"Could not read Pixi lock: {pixi_lock_path}") from error
    store = AtomicParquetDatasetStore(dataset_root)
    criteria = read_criterion_records(store)
    repeats = read_repeat_metrics(store)
    aggregates = read_case_aggregates(store)
    _validate_sources(
        scoring=scoring,
        specification=specification,
        hierarchy=hierarchy,
        repeat_publication_hash=repeats.manifest.publication_hash,
        aggregate_publication_hash=aggregates.manifest.publication_hash,
        dataset_identities=(
            (criteria.manifest.benchmark_version, criteria.manifest.run_id),
            (repeats.manifest.benchmark_version, repeats.manifest.run_id),
            (aggregates.manifest.benchmark_version, aggregates.manifest.run_id),
        ),
        created_at_utc=created_at_utc,
    )
    plan = _create_plan(
        scoring=scoring,
        specification=specification,
        hierarchy=hierarchy,
        pixi_lock_hash=pixi_lock_hash,
        analysis_code_revision=analysis_code_revision,
        created_at_utc=created_at_utc,
    )
    root = output_root.resolve()
    report_path = root / "primary_analysis_report.json"
    if report_path.exists():
        report = _load_json(report_path, PrimaryAnalysisReport)
        if report.analysis_plan_hash != plan.plan_hash:
            raise PrimaryAnalysisError("Existing primary report belongs to another plan")
        return report
    write_immutable_json(root / "primary_analysis_plan.json", plan)

    case_results, missing_cases = _extract_primary_cases(
        criterion_rows=criteria.table.to_pylist(),
        repeat_rows=repeats.table.to_pylist(),
        aggregate_rows=aggregates.table.to_pylist(),
    )
    effects = tuple(record.case_effect for record in case_results)
    inference = paired_case_inference(
        effects,
        bootstrap_resamples=specification.bootstrap_resamples,
        permutation_resamples=specification.permutation_resamples,
        minimum_interpretable_effect=specification.minimum_interpretable_effect,
        random_seed=specification.random_seed,
    )
    paired = tuple(
        (record.dialogue_correct, record.valid_evidence_correct) for record in case_results
    )
    mcnemar = exact_mcnemar(paired)
    improvement_count = mcnemar.right_only_correct_count
    regression_count = mcnemar.left_only_correct_count
    net_improvement_count = improvement_count - regression_count
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.primary_analysis_report.v1",
        "benchmark_version": "v1",
        "run_id": scoring.run_id,
        "analysis_plan_hash": plan.plan_hash,
        "total_case_count": 24,
        "eligible_case_count": len(case_results),
        "missing_case_count": len(missing_cases),
        "case_results": case_results,
        "missing_cases": missing_cases,
        "inference": inference,
        "mcnemar": mcnemar,
        "valid_evidence_improvement_count": improvement_count,
        "valid_evidence_regression_count": regression_count,
        "net_improvement_count": net_improvement_count,
        "paired_effect_from_counts": net_improvement_count / len(case_results),
        "interval_threshold_status": _interval_threshold_status(inference),
        "mcnemar_rejects_equal_discordance": (
            mcnemar.two_sided_exact_p_value < hierarchy.primary_test_alpha
        ),
        "mcnemar_alpha": hierarchy.primary_test_alpha,
        "mcnemar_scope": "tests_equality_of_discordant_probabilities",
        "threshold_scope": "tests_whether_mean_reaches_frozen_practical_threshold",
        "significance_is_not_success_gate": True,
        "sign_swap_role": "secondary_sensitivity_only",
        "interval_scope": hierarchy.case_interval_scope,
        "repeat_variability": hierarchy.repeat_variability,
        "completed_at_utc": created_at_utc,
    }
    draft = PrimaryAnalysisReport.model_construct(
        _fields_set=set(content), **content, report_hash="0" * 64
    )
    report = PrimaryAnalysisReport.model_validate(
        {**content, "report_hash": model_content_hash(draft, exclude={"report_hash"})}
    )
    write_immutable_json(report_path, report)
    return report


def _extract_primary_cases(
    *,
    criterion_rows: list[dict[str, Any]],
    repeat_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
) -> tuple[tuple[PrimaryCaseResult, ...], tuple[MissingPrimaryCase, ...]]:
    criteria = _unique_by_case(criterion_rows, "criterion")
    primary_aggregates = _unique_by_case(
        [row for row in aggregate_rows if row["comparison"] == CaseComparison.PRIMARY_VALID.value],
        "primary aggregate",
    )
    dialogue = _unique_by_case(
        [row for row in repeat_rows if row["estimator"] == RepeatEstimator.DIALOGUE_ONLY.value],
        "dialogue repeat",
    )
    valid = _unique_by_case(
        [row for row in repeat_rows if row["estimator"] == RepeatEstimator.PROBE_INFORMED.value],
        "valid-evidence repeat",
    )
    case_sets = tuple(map(set, (criteria, primary_aggregates, dialogue, valid)))
    if not case_sets or any(case_ids != case_sets[0] for case_ids in case_sets[1:]):
        raise PrimaryAnalysisError("Primary datasets do not contain the same case identities")
    if len(case_sets[0]) != 24:
        raise PrimaryAnalysisError("Primary analysis requires all 24 authored case identities")

    eligible: list[PrimaryCaseResult] = []
    missing: list[MissingPrimaryCase] = []
    for case_id in sorted(case_sets[0]):
        criterion = criteria[case_id]
        aggregate = primary_aggregates[case_id]
        dialogue_row = dialogue[case_id]
        valid_row = valid[case_id]
        criterion_hash = str(criterion["record_hash"])
        dialogue_hash = str(dialogue_row["record_hash"])
        valid_hash = str(valid_row["record_hash"])
        if (
            str(dialogue_row["criterion_record_hash"]) != criterion_hash
            or str(valid_row["criterion_record_hash"]) != criterion_hash
        ):
            raise PrimaryAnalysisError("Primary repeat belongs to another criterion record")
        source_hashes = {str(value) for value in aggregate["source_repeat_hashes"]}
        if not {dialogue_hash, valid_hash}.issubset(source_hashes):
            raise PrimaryAnalysisError("Primary aggregate lacks its paired repeat sources")
        if (
            dialogue_row["selected_loss_name"] != "classification_error"
            or valid_row["selected_loss_name"] != "classification_error"
        ):
            raise PrimaryAnalysisError("Primary repeats use another selected loss")
        outcome = criterion["demonstrated_performance"]
        if outcome is None:
            reason = criterion["missing_reason"]
            if not isinstance(reason, str) or not reason:
                raise PrimaryAnalysisError("Missing criterion lacks a typed reason")
            if aggregate["case_effect"] is not None:
                raise PrimaryAnalysisError("Missing criterion contains a primary case effect")
            if (
                dialogue_row["eligible"] is not False
                or valid_row["eligible"] is not False
                or dialogue_row["classification_error"] is not None
                or valid_row["classification_error"] is not None
                or aggregate["eligible_repeat_count"] != 0
            ):
                raise PrimaryAnalysisError("Missing criterion contains invented primary losses")
            missing.append(
                MissingPrimaryCase(
                    case_id=case_id,
                    missing_reason=reason,
                    aggregate_record_hash=str(aggregate["record_hash"]),
                    criterion_record_hash=criterion_hash,
                )
            )
            continue
        if type(outcome) is not bool:
            raise PrimaryAnalysisError("Criterion outcome must be boolean or missing")
        if (
            dialogue_row["eligible"] is not True
            or valid_row["eligible"] is not True
            or aggregate["eligible_repeat_count"] != 1
        ):
            raise PrimaryAnalysisError("Eligible criterion lacks one paired primary loss")
        dialogue_error = _binary_loss(dialogue_row["classification_error"])
        valid_error = _binary_loss(valid_row["classification_error"])
        case_effect = _signed_binary_effect(aggregate["case_effect"])
        dialogue_decision = _strict_bool(dialogue_row["binary_decision"], "dialogue decision")
        valid_decision = _strict_bool(valid_row["binary_decision"], "valid-evidence decision")
        result = PrimaryCaseResult(
            case_id=case_id,
            criterion_outcome=outcome,
            dialogue_binary_decision=dialogue_decision,
            valid_evidence_binary_decision=valid_decision,
            dialogue_correct=dialogue_decision is outcome,
            valid_evidence_correct=valid_decision is outcome,
            dialogue_classification_error=dialogue_error,
            valid_evidence_classification_error=valid_error,
            case_effect=case_effect,
            aggregate_record_hash=str(aggregate["record_hash"]),
            dialogue_repeat_hash=dialogue_hash,
            valid_evidence_repeat_hash=valid_hash,
            criterion_record_hash=criterion_hash,
        )
        if abs(result.case_effect - (dialogue_error - valid_error)) > 1e-12:
            raise PrimaryAnalysisError("Aggregate and repeat-level primary effects differ")
        eligible.append(result)
    if len(eligible) + len(missing) != 24:
        raise PrimaryAnalysisError("Primary eligible and missing cases do not reconcile")
    return tuple(eligible), tuple(missing)


def _validate_sources(
    *,
    scoring: ExternalScoringSummary,
    specification: AnalysisSpecification,
    hierarchy: InferentialHierarchy,
    repeat_publication_hash: Sha256,
    aggregate_publication_hash: Sha256,
    dataset_identities: tuple[tuple[str, str], ...],
    created_at_utc: datetime,
) -> None:
    specification_hash = analysis_specification_hash(specification)
    if not scoring.gate_passed or scoring.effect_interpretation_status != "not_computed":
        raise PrimaryAnalysisError("Primary analysis requires uninterpreted scored datasets")
    if created_at_utc <= scoring.completed_at_utc:
        raise PrimaryAnalysisError("Primary analysis must follow scoring publication")
    if hierarchy.analysis_specification_hash != specification_hash:
        raise PrimaryAnalysisError("Inferential hierarchy belongs to another analysis plan")
    if hierarchy.bootstrap_resamples != specification.bootstrap_resamples:
        raise PrimaryAnalysisError("Bootstrap count differs from the frozen hierarchy")
    if hierarchy.minimum_interpretable_effect != specification.minimum_interpretable_effect:
        raise PrimaryAnalysisError("Effect threshold differs from the frozen hierarchy")
    if hierarchy.case_count != scoring.criterion_count:
        raise PrimaryAnalysisError("Scored case count differs from the frozen hierarchy")
    if scoring.repeat_publication_hash != repeat_publication_hash:
        raise PrimaryAnalysisError("Repeat dataset differs from the scoring summary")
    if scoring.aggregate_publication_hash != aggregate_publication_hash:
        raise PrimaryAnalysisError("Aggregate dataset differs from the scoring summary")
    identities = set(dataset_identities)
    if identities != {(scoring.benchmark_version, scoring.run_id)}:
        raise PrimaryAnalysisError("Primary datasets belong to another scored run")


def _create_plan(
    *,
    scoring: ExternalScoringSummary,
    specification: AnalysisSpecification,
    hierarchy: InferentialHierarchy,
    pixi_lock_hash: Sha256,
    analysis_code_revision: str,
    created_at_utc: datetime,
) -> PrimaryAnalysisPlan:
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.primary_analysis_plan.v1",
        "benchmark_version": "v1",
        "run_id": scoring.run_id,
        "scoring_summary_hash": scoring.summary_hash,
        "repeat_publication_hash": scoring.repeat_publication_hash,
        "aggregate_publication_hash": scoring.aggregate_publication_hash,
        "analysis_specification_hash": analysis_specification_hash(specification),
        "inferential_hierarchy_hash": hierarchy.hierarchy_hash,
        "primary_comparison": "primary_valid",
        "primary_estimand": hierarchy.primary_estimand,
        "primary_test": hierarchy.primary_test,
        "bootstrap_resamples": specification.bootstrap_resamples,
        "random_seed": specification.random_seed,
        "minimum_interpretable_effect": specification.minimum_interpretable_effect,
        "analysis_code_revision": analysis_code_revision,
        "analysis_pixi_lock_hash": pixi_lock_hash,
        "created_at_utc": created_at_utc,
    }
    draft = PrimaryAnalysisPlan.model_construct(
        _fields_set=set(content), **content, plan_hash="0" * 64
    )
    return PrimaryAnalysisPlan.model_validate(
        {**content, "plan_hash": model_content_hash(draft, exclude={"plan_hash"})}
    )


def _interval_threshold_status(
    inference: PairedCaseInference,
) -> Literal["interval_above_minimum", "interval_crosses_minimum", "interval_below_minimum"]:
    threshold = inference.minimum_interpretable_effect
    if inference.interval_lower >= threshold:
        return "interval_above_minimum"
    if inference.interval_upper < threshold:
        return "interval_below_minimum"
    return "interval_crosses_minimum"


def _unique_by_case(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise PrimaryAnalysisError(f"{label} row lacks a case ID")
        if case_id in indexed:
            raise PrimaryAnalysisError(f"Duplicate {label} row for {case_id}")
        indexed[case_id] = row
    return indexed


def _binary_loss(value: object) -> float:
    if type(value) in {int, float} and value == 0.0:
        return 0.0
    if type(value) in {int, float} and value == 1.0:
        return 1.0
    raise PrimaryAnalysisError("Eligible classification loss must be zero or one")


def _signed_binary_effect(value: object) -> float:
    if type(value) in {int, float} and value == -1.0:
        return -1.0
    if type(value) in {int, float} and value == 0.0:
        return 0.0
    if type(value) in {int, float} and value == 1.0:
        return 1.0
    raise PrimaryAnalysisError("Eligible primary case effect must be -1, 0, or 1")


def _strict_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise PrimaryAnalysisError(f"{label} must be boolean")
    return value


def _load_json[ModelT: ContractModel](path: Path, model: type[ModelT]) -> ModelT:
    try:
        return model.model_validate_json(path.read_bytes())
    except (OSError, ValidationError) as error:
        raise PrimaryAnalysisError(f"Could not verify primary-analysis source: {path}") from error


def _require_utc(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("Primary analysis timestamp must be UTC")
