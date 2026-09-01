# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
"""Publish the frozen CSEDM uncertainty analysis."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Literal, Self

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field, model_validator

from socratic_tutor.benchmark.artifacts import (
    artifact_locations,
    write_immutable_bytes,
    write_immutable_json,
)
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import canonical_sha256, file_sha256
from socratic_tutor.contracts import ContractModel
from socratic_tutor.csedm_study.analysis_plan import CSEDMAnalysisPlan, load_analysis_plan
from socratic_tutor.csedm_study.evaluation import (
    PredictionMetrics,
    PredictionRecord,
    RiskCoveragePoint,
    learner_bootstrap_differences,
    prediction_metrics,
    risk_coverage_curve,
    triage_summary,
)
from socratic_tutor.csedm_study.features import CSEDMAdapterError
from socratic_tutor.csedm_study.models import LOGISTIC_LEARNER_HISTORY, MODEL_IDS
from socratic_tutor.csedm_study.prediction import CSEDMPredictionManifest
from socratic_tutor.csv_output import csv_bytes
from socratic_tutor.repository_state import current_clean_revision


class PrimaryTriageResult(ContractModel):
    """Primary fixed-budget comparison with learner-level uncertainty."""

    model_id: Literal["logistic_learner_history"]
    review_budget_fraction: float = Field(gt=0.0, lt=1.0)
    eligible_prediction_count: int = Field(ge=1)
    selected_prediction_count: int = Field(ge=1)
    total_error_count: int = Field(ge=1)
    selected_error_count: int = Field(ge=0)
    expected_random_selected_error_count: float = Field(ge=0.0, allow_inf_nan=False)
    captured_error_recall: float = Field(ge=0.0, le=1.0)
    expected_random_captured_error_recall: float = Field(ge=0.0, le=1.0)
    captured_error_recall_difference: float = Field(ge=-1.0, le=1.0)
    confidence_level: float = Field(gt=0.0, lt=1.0)
    interval_lower: float = Field(ge=-1.0, le=1.0)
    interval_upper: float = Field(ge=-1.0, le=1.0)
    bootstrap_method: Literal["learner_cluster_percentile_bootstrap"]
    bootstrap_repetitions: Literal[10000]
    bootstrap_seed: int = Field(ge=0)
    bootstrap_distribution_hash: Sha256
    selected_status: Literal["fixed_from_original_out_of_fold_ranking"]
    claim_supported: bool

    @model_validator(mode="after")
    def validate_claim(self) -> Self:
        if (self.review_budget_fraction, self.confidence_level) != (0.5, 0.95):
            raise ValueError("Primary CSEDM budget or confidence level changed")
        if self.claim_supported != (self.interval_lower > 0.0):
            raise ValueError("Primary CSEDM claim does not match its frozen interval rule")
        return self


class FoldTriageResult(ContractModel):
    """Fixed-budget error capture in one official test fold."""

    fold_id: int = Field(ge=0, lt=10)
    eligible_prediction_count: int = Field(ge=1)
    selected_prediction_count: int = Field(ge=1)
    total_error_count: int = Field(ge=1)
    selected_error_count: int = Field(ge=0)
    expected_random_selected_error_count: float = Field(ge=0.0)
    captured_error_recall: float = Field(ge=0.0, le=1.0)
    expected_random_captured_error_recall: float = Field(ge=0.0, le=1.0)
    captured_error_recall_difference: float = Field(ge=-1.0, le=1.0)


class ModelMetricsResult(ContractModel):
    """Descriptive prediction quality for one frozen model."""

    model_id: str = Field(min_length=1)
    prediction_count: int = Field(ge=1)
    positive_count: int = Field(ge=0)
    predicted_positive_count: int = Field(ge=0)
    error_count: int = Field(ge=0)
    accuracy: float = Field(ge=0.0, le=1.0)
    brier_score: float = Field(ge=0.0, le=1.0)
    log_loss: float = Field(ge=0.0, allow_inf_nan=False)
    expected_calibration_error: float = Field(ge=0.0, le=1.0)


class RiskCoverageResult(ContractModel):
    """Error among logistic predictions retained without review."""

    requested_coverage: float = Field(gt=0.0, le=1.0)
    retained_prediction_count: int = Field(ge=1)
    actual_coverage: float = Field(gt=0.0, le=1.0)
    retained_error_count: int = Field(ge=0)
    retained_error_rate: float = Field(ge=0.0, le=1.0)


class FoldSensitivityResult(ContractModel):
    """Primary contrast after omitting one complete official fold."""

    omitted_fold_id: int = Field(ge=0, lt=10)
    eligible_prediction_count: int = Field(ge=1)
    total_error_count: int = Field(ge=1)
    captured_error_recall_difference: float = Field(ge=-1.0, le=1.0)


class CSEDMAnalysisReport(ContractModel):
    """Aggregate, publication-safe result of the frozen M7C analysis."""

    schema_id: Literal["csedm.uncertainty_analysis_report.v1"] = (
        "csedm.uncertainty_analysis_report.v1"
    )
    schema_version: Literal[1] = 1
    study_id: Literal["csedm-uncertainty-triage-v1"] = "csedm-uncertainty-triage-v1"
    analysis_plan_hash: Sha256
    prediction_manifest_hash: Sha256
    prediction_target_count: Literal[729]
    learner_count: Literal[86]
    problem_count: Literal[19]
    official_fold_count: Literal[10]
    positive_target_count: int = Field(ge=0)
    negative_target_count: int = Field(ge=0)
    exclusion_count: Literal[0]
    primary: PrimaryTriageResult
    fold_results: tuple[FoldTriageResult, ...] = Field(min_length=10, max_length=10)
    model_metrics: tuple[ModelMetricsResult, ...] = Field(min_length=3, max_length=3)
    risk_coverage: tuple[RiskCoverageResult, ...] = Field(min_length=10, max_length=10)
    fold_sensitivity: tuple[FoldSensitivityResult, ...] = Field(min_length=10, max_length=10)
    result_classification: Literal[
        "uncertainty_captured_more_errors_than_random_expectation",
        "uncertainty_captured_fewer_errors_than_random_expectation",
        "interval_includes_zero",
    ]
    executable_probe_claim_supported: Literal[False]
    tutoring_effect_claim_supported: Literal[False]
    human_learning_claim_supported: Literal[False]
    report_hash: Sha256

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        if self.positive_target_count + self.negative_target_count != 729:
            raise ValueError("CSEDM class counts do not cover every target")
        if tuple(row.fold_id for row in self.fold_results) != tuple(range(10)):
            raise ValueError("CSEDM report does not contain every fold in order")
        if tuple(row.model_id for row in self.model_metrics) != MODEL_IDS:
            raise ValueError("CSEDM report model metrics differ from the frozen order")
        expected_hash = canonical_sha256(self.model_dump(mode="json", exclude={"report_hash"}))
        if self.report_hash != expected_hash:
            raise ValueError("CSEDM analysis report hash does not match its content")
        return self


class CSEDMAnalysisManifest(ContractModel):
    """Immutable file identity for the M7C analysis package."""

    schema_id: Literal["csedm.uncertainty_analysis_manifest.v1"] = (
        "csedm.uncertainty_analysis_manifest.v1"
    )
    schema_version: Literal[1] = 1
    study_id: Literal["csedm-uncertainty-triage-v1"] = "csedm-uncertainty-triage-v1"
    code_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    pixi_lock_sha256: Sha256
    analysis_plan_hash: Sha256
    prediction_manifest_hash: Sha256
    analysis_report_sha256: Sha256
    report_content_hash: Sha256
    deterministic_replay_hash: Sha256
    model_metrics_csv_sha256: Sha256
    fold_results_csv_sha256: Sha256
    risk_coverage_csv_sha256: Sha256
    network_calls: Literal[0]
    sandbox_calls: Literal[0]
    public_artifacts_contain_learner_values: Literal[False]
    manifest_hash: Sha256

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        if self.report_content_hash != self.deterministic_replay_hash:
            raise ValueError("CSEDM analysis replay differs from the primary calculation")
        expected_hash = canonical_sha256(self.model_dump(mode="json", exclude={"manifest_hash"}))
        if self.manifest_hash != expected_hash:
            raise ValueError("CSEDM analysis manifest hash does not match its content")
        return self


def publish_uncertainty_analysis(
    *,
    prediction_root: Path,
    analysis_plan_path: Path,
    pixi_lock_path: Path,
    output_root: Path,
    code_revision: str,
) -> CSEDMAnalysisManifest:
    """Calculate, replay, and publish the prespecified M7C analysis."""

    plan = load_analysis_plan(analysis_plan_path)
    prediction_manifest = _load_prediction_manifest(prediction_root, plan)
    records = {
        model_id: _load_predictions(prediction_root, prediction_manifest, model_id)
        for model_id in MODEL_IDS
    }
    _validate_matched_records(records)
    report = _calculate_report(records, plan, prediction_manifest)
    replay = _calculate_report(records, plan, prediction_manifest)
    if report.report_hash != replay.report_hash:
        raise CSEDMAdapterError("CSEDM analysis did not reproduce in memory")

    model_csv = _model_metrics_csv(report.model_metrics)
    fold_csv = _fold_results_csv(report.fold_results)
    risk_csv = _risk_coverage_csv(report.risk_coverage)
    report_path = output_root / "uncertainty_analysis_report.json"
    model_path = output_root / "model_metrics.csv"
    fold_path = output_root / "fold_results.csv"
    risk_path = output_root / "risk_coverage.csv"
    write_immutable_json(report_path, report)
    write_immutable_bytes(model_path, model_csv)
    write_immutable_bytes(fold_path, fold_csv)
    write_immutable_bytes(risk_path, risk_csv)

    manifest_payload = {
        "schema_id": "csedm.uncertainty_analysis_manifest.v1",
        "schema_version": 1,
        "study_id": "csedm-uncertainty-triage-v1",
        "code_revision": code_revision,
        "pixi_lock_sha256": file_sha256(pixi_lock_path.read_bytes()),
        "analysis_plan_hash": plan.plan_hash,
        "prediction_manifest_hash": prediction_manifest.manifest_hash,
        "analysis_report_sha256": file_sha256(report_path.read_bytes()),
        "report_content_hash": report.report_hash,
        "deterministic_replay_hash": replay.report_hash,
        "model_metrics_csv_sha256": file_sha256(model_csv),
        "fold_results_csv_sha256": file_sha256(fold_csv),
        "risk_coverage_csv_sha256": file_sha256(risk_csv),
        "network_calls": 0,
        "sandbox_calls": 0,
        "public_artifacts_contain_learner_values": False,
    }
    manifest_payload["manifest_hash"] = canonical_sha256(manifest_payload)
    manifest = CSEDMAnalysisManifest.model_validate(manifest_payload)
    write_immutable_json(output_root / "analysis_manifest.json", manifest)
    return manifest


def _calculate_report(
    records: dict[str, tuple[PredictionRecord, ...]],
    plan: CSEDMAnalysisPlan,
    prediction_manifest: CSEDMPredictionManifest,
) -> CSEDMAnalysisReport:
    primary_records = records[LOGISTIC_LEARNER_HISTORY]
    budget = plan.primary.review_budget_fraction
    triage = triage_summary(primary_records, budget)
    distribution = learner_bootstrap_differences(
        primary_records,
        triage.selected_target_ids,
        budget_fraction=budget,
        repetitions=plan.bootstrap.repetitions,
        seed=plan.bootstrap.seed,
    )
    alpha = 1.0 - plan.bootstrap.confidence_level
    lower, upper = np.quantile(
        distribution,
        (alpha / 2.0, 1.0 - alpha / 2.0),
        method="linear",
    )
    primary = PrimaryTriageResult(
        model_id=LOGISTIC_LEARNER_HISTORY,
        review_budget_fraction=0.5,
        eligible_prediction_count=triage.eligible_count,
        selected_prediction_count=triage.selected_count,
        total_error_count=triage.error_count,
        selected_error_count=triage.selected_error_count,
        expected_random_selected_error_count=triage.expected_random_selected_error_count,
        captured_error_recall=triage.captured_error_recall,
        expected_random_captured_error_recall=triage.expected_random_captured_error_recall,
        captured_error_recall_difference=triage.captured_error_recall_difference,
        confidence_level=0.95,
        interval_lower=float(lower),
        interval_upper=float(upper),
        bootstrap_method=plan.bootstrap.method,
        bootstrap_repetitions=plan.bootstrap.repetitions,
        bootstrap_seed=plan.bootstrap.seed,
        bootstrap_distribution_hash=canonical_sha256(tuple(float(value) for value in distribution)),
        selected_status=plan.bootstrap.selected_status,
        claim_supported=float(lower) > 0.0,
    )
    fold_results = tuple(
        _fold_result(
            fold_id,
            tuple(row for row in primary_records if row.fold_id == fold_id),
            budget,
        )
        for fold_id in plan.official_fold_ids
    )
    metrics = tuple(
        _metrics_result(
            model_id,
            prediction_metrics(
                records[model_id],
                probability_floor=plan.secondary.log_loss_probability_floor,
                calibration_bins=plan.secondary.ece_equal_width_bin_count,
            ),
        )
        for model_id in MODEL_IDS
    )
    risk = tuple(
        _risk_result(point)
        for point in risk_coverage_curve(
            primary_records,
            plan.secondary.risk_coverage_fractions,
        )
    )
    sensitivity = tuple(
        _fold_sensitivity(primary_records, omitted_fold_id, budget)
        for omitted_fold_id in plan.official_fold_ids
    )
    classification = _result_classification(float(lower), float(upper))
    payload = {
        "schema_id": "csedm.uncertainty_analysis_report.v1",
        "schema_version": 1,
        "study_id": "csedm-uncertainty-triage-v1",
        "analysis_plan_hash": plan.plan_hash,
        "prediction_manifest_hash": prediction_manifest.manifest_hash,
        "prediction_target_count": len(primary_records),
        "learner_count": len({row.learner_id for row in primary_records}),
        "problem_count": len({row.problem_id for row in primary_records}),
        "official_fold_count": len({row.fold_id for row in primary_records}),
        "positive_target_count": sum(row.first_correct for row in primary_records),
        "negative_target_count": sum(not row.first_correct for row in primary_records),
        "exclusion_count": 0,
        "primary": primary,
        "fold_results": fold_results,
        "model_metrics": metrics,
        "risk_coverage": risk,
        "fold_sensitivity": sensitivity,
        "result_classification": classification,
        "executable_probe_claim_supported": False,
        "tutoring_effect_claim_supported": False,
        "human_learning_claim_supported": False,
    }
    payload["report_hash"] = canonical_sha256(payload)
    return CSEDMAnalysisReport.model_validate(payload)


def _fold_result(
    fold_id: int,
    records: tuple[PredictionRecord, ...],
    budget: float,
) -> FoldTriageResult:
    summary = triage_summary(records, budget)
    return FoldTriageResult(
        fold_id=fold_id,
        eligible_prediction_count=summary.eligible_count,
        selected_prediction_count=summary.selected_count,
        total_error_count=summary.error_count,
        selected_error_count=summary.selected_error_count,
        expected_random_selected_error_count=summary.expected_random_selected_error_count,
        captured_error_recall=summary.captured_error_recall,
        expected_random_captured_error_recall=summary.expected_random_captured_error_recall,
        captured_error_recall_difference=summary.captured_error_recall_difference,
    )


def _metrics_result(model_id: str, metrics: PredictionMetrics) -> ModelMetricsResult:
    return ModelMetricsResult(
        model_id=model_id,
        prediction_count=metrics.prediction_count,
        positive_count=metrics.positive_count,
        predicted_positive_count=metrics.predicted_positive_count,
        error_count=metrics.error_count,
        accuracy=metrics.accuracy,
        brier_score=metrics.brier_score,
        log_loss=metrics.log_loss,
        expected_calibration_error=metrics.expected_calibration_error,
    )


def _risk_result(point: RiskCoveragePoint) -> RiskCoverageResult:
    return RiskCoverageResult(
        requested_coverage=point.requested_coverage,
        retained_prediction_count=point.retained_count,
        actual_coverage=point.actual_coverage,
        retained_error_count=point.retained_error_count,
        retained_error_rate=point.retained_error_rate,
    )


def _fold_sensitivity(
    records: tuple[PredictionRecord, ...],
    omitted_fold_id: int,
    budget: float,
) -> FoldSensitivityResult:
    retained = tuple(row for row in records if row.fold_id != omitted_fold_id)
    summary = triage_summary(retained, budget)
    return FoldSensitivityResult(
        omitted_fold_id=omitted_fold_id,
        eligible_prediction_count=summary.eligible_count,
        total_error_count=summary.error_count,
        captured_error_recall_difference=summary.captured_error_recall_difference,
    )


def _result_classification(
    interval_lower: float,
    interval_upper: float,
) -> str:
    if interval_lower > 0.0:
        return "uncertainty_captured_more_errors_than_random_expectation"
    if interval_upper < 0.0:
        return "uncertainty_captured_fewer_errors_than_random_expectation"
    return "interval_includes_zero"


def _load_prediction_manifest(
    prediction_root: Path,
    plan: CSEDMAnalysisPlan,
) -> CSEDMPredictionManifest:
    manifest = CSEDMPredictionManifest.model_validate_json(
        (prediction_root / "prediction_manifest.json").read_bytes()
    )
    if manifest.analysis_plan_hash != plan.plan_hash:
        raise CSEDMAdapterError("Prediction manifest and analysis plan hashes differ")
    return manifest


def _load_predictions(
    prediction_root: Path,
    manifest: CSEDMPredictionManifest,
    model_id: str,
) -> tuple[PredictionRecord, ...]:
    path = prediction_root / "restricted" / "predictions" / f"{model_id}.parquet"
    content = path.read_bytes()
    if file_sha256(content) != manifest.prediction_file_sha256[model_id]:
        raise CSEDMAdapterError(f"Prediction file differs from its manifest: {model_id}")
    rows: tuple[dict[str, object], ...] = tuple(pq.read_table(pa.BufferReader(content)).to_pylist())
    records: list[PredictionRecord] = []
    for row in rows:
        _validate_prediction_row(row, model_id)
        records.append(
            PredictionRecord(
                model_id=model_id,
                fold_id=_integer(row, "fold_id"),
                target_id=_text(row, "target_id"),
                learner_id=_text(row, "learner_id"),
                problem_id=_text(row, "problem_id"),
                first_correct=_boolean(row, "first_correct"),
                probability_correct=_number(row, "probability_correct"),
            )
        )
    return tuple(sorted(records, key=lambda row: (row.fold_id, row.target_id)))


def _validate_prediction_row(row: dict[str, object], expected_model_id: str) -> None:
    if _text(row, "model_id") != expected_model_id:
        raise CSEDMAdapterError("Prediction row model differs from its file")
    payload = {key: value for key, value in row.items() if key != "record_hash"}
    if canonical_sha256(payload) != _text(row, "record_hash"):
        raise CSEDMAdapterError("Prediction row has an invalid record hash")
    probability = _number(row, "probability_correct")
    if _boolean(row, "predicted_correct") != (probability >= 0.5):
        raise CSEDMAdapterError("Prediction label differs from its probability")
    if not math.isclose(
        _number(row, "uncertainty"), abs(probability - 0.5), rel_tol=0.0, abs_tol=1e-15
    ):
        raise CSEDMAdapterError("Prediction uncertainty differs from its probability")


def _validate_matched_records(
    records: dict[str, tuple[PredictionRecord, ...]],
) -> None:
    reference = records[MODEL_IDS[0]]
    reference_values = {
        row.target_id: (row.fold_id, row.learner_id, row.first_correct) for row in reference
    }
    for model_id in MODEL_IDS:
        rows = records[model_id]
        values = {row.target_id: (row.fold_id, row.learner_id, row.first_correct) for row in rows}
        if values != reference_values:
            raise CSEDMAdapterError(f"{model_id} predictions do not match the reference targets")


def _model_metrics_csv(rows: tuple[ModelMetricsResult, ...]) -> bytes:
    return csv_bytes(
        (
            "model_id",
            "prediction_count",
            "positive_count",
            "predicted_positive_count",
            "error_count",
            "accuracy",
            "brier_score",
            "log_loss",
            "expected_calibration_error",
        ),
        (
            (
                row.model_id,
                row.prediction_count,
                row.positive_count,
                row.predicted_positive_count,
                row.error_count,
                row.accuracy,
                row.brier_score,
                row.log_loss,
                row.expected_calibration_error,
            )
            for row in rows
        ),
    )


def _fold_results_csv(rows: tuple[FoldTriageResult, ...]) -> bytes:
    headers = tuple(rows[0].model_dump(mode="json"))
    return csv_bytes(headers, (tuple(row.model_dump(mode="json").values()) for row in rows))


def _risk_coverage_csv(rows: tuple[RiskCoverageResult, ...]) -> bytes:
    headers = tuple(rows[0].model_dump(mode="json"))
    return csv_bytes(headers, (tuple(row.model_dump(mode="json").values()) for row in rows))


def _text(row: dict[str, object], name: str) -> str:
    value = row.get(name)
    if not isinstance(value, str) or not value:
        raise CSEDMAdapterError(f"Prediction field {name} must be non-empty text")
    return value


def _boolean(row: dict[str, object], name: str) -> bool:
    value = row.get(name)
    if not isinstance(value, bool):
        raise CSEDMAdapterError(f"Prediction field {name} must be Boolean")
    return value


def _integer(row: dict[str, object], name: str) -> int:
    value = row.get(name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise CSEDMAdapterError(f"Prediction field {name} must be an integer")
    return value


def _number(row: dict[str, object], name: str) -> float:
    value = row.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise CSEDMAdapterError(f"Prediction field {name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise CSEDMAdapterError(f"Prediction field {name} must be finite")
    return numeric


def build_parser(project_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish the frozen CSEDM uncertainty analysis")
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument(
        "--analysis-plan",
        type=Path,
        default=project_root / "configs" / "csedm-study" / "v1-analysis.yaml",
    )
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
            dirty_message="Commit tracked changes before publishing the CSEDM analysis",
            untracked_files="no",
            operation_name="CSEDM analysis publication",
            error_factory=CSEDMAdapterError,
        )
        output_root = args.output_root or (
            project_root / "artifacts" / "csedm-study" / f"analysis-{revision[:7]}"
        )
        manifest = publish_uncertainty_analysis(
            prediction_root=args.prediction_root,
            analysis_plan_path=args.analysis_plan,
            pixi_lock_path=args.pixi_lock,
            output_root=output_root,
            code_revision=revision,
        )
    except Exception as error:
        print(
            json.dumps(
                {
                    "command": "csedm-analyse",
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
                "command": "csedm-analyse",
                "result": manifest.model_dump(mode="json"),
                "status": "ok",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0
