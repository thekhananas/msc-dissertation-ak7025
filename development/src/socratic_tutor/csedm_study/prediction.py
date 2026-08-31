# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
"""Out-of-fold prediction pipeline for the CSEDM companion study."""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path
from typing import Any, Literal, Self

import pyarrow as pa
import pyarrow.parquet as pq
import sklearn
from pydantic import Field, model_validator

from socratic_tutor.benchmark.artifacts import (
    artifact_locations,
    write_immutable_bytes,
    write_immutable_json,
)
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import canonical_sha256, file_sha256
from socratic_tutor.contracts import ContractModel
from socratic_tutor.csedm_study.adapter import CSEDMAdapterManifest
from socratic_tutor.csedm_study.analysis_plan import (
    PRIMARY_FEATURES,
    CSEDMAnalysisPlan,
    load_analysis_plan,
)
from socratic_tutor.csedm_study.features import CSEDMAdapterError
from socratic_tutor.csedm_study.inventory import load_inventory_specification
from socratic_tutor.csedm_study.models import (
    MODEL_IDS,
    ModelInput,
    ModelPrediction,
    predict_fold,
)
from socratic_tutor.csedm_study.reference import check_reference_baseline
from socratic_tutor.csedm_study.tabular import parquet_bytes
from socratic_tutor.repository_state import current_clean_revision

MODEL_FEATURE_COLUMNS = tuple(f"scaled_{name}" for name in PRIMARY_FEATURES) + tuple(
    f"missing_{name}" for name in PRIMARY_FEATURES
)


class FoldPredictionAudit(ContractModel):
    """Aggregate evidence that all models completed one official fold."""

    fold_id: int = Field(ge=0, lt=10)
    training_target_count: int = Field(ge=1)
    test_target_count: int = Field(ge=1)
    training_positive_count: int = Field(ge=0)
    test_positive_count: int = Field(ge=0)
    training_prevalence: float = Field(ge=0.0, le=1.0)
    problem_frequency_fallback_count: int = Field(ge=0)
    logistic_iterations: int = Field(ge=1, le=1000)
    completed_models: tuple[str, ...]

    @model_validator(mode="after")
    def validate_models(self) -> Self:
        if self.completed_models != MODEL_IDS:
            raise ValueError("A CSEDM fold did not complete every frozen model")
        return self


class CSEDMPredictionManifest(ContractModel):
    """Public identity and completion record for restricted predictions."""

    schema_id: Literal["csedm.prediction_manifest.v1"] = "csedm.prediction_manifest.v1"
    schema_version: Literal[1] = 1
    study_id: Literal["csedm-uncertainty-triage-v1"] = "csedm-uncertainty-triage-v1"
    code_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    pixi_lock_sha256: Sha256
    analysis_plan_hash: Sha256
    analysis_plan_file_sha256: Sha256
    adapter_manifest_hash: Sha256
    adapter_manifest_file_sha256: Sha256
    adapter_fold_features_sha256: Sha256
    adapter_out_of_fold_sha256: Sha256
    model_ids: tuple[str, ...]
    prediction_row_count_per_model: int = Field(ge=1)
    unique_target_count_per_model: int = Field(ge=1)
    prediction_file_sha256: dict[str, Sha256]
    prediction_schema_sha256: Sha256
    model_specification_sha256: Sha256
    fold_completion_sha256: Sha256
    reference_baseline_sha256: Sha256
    all_official_folds_completed: Literal[True]
    public_artifacts_contain_learner_values: Literal[False]
    restricted_predictions_contain_learner_identifiers: Literal[True]
    manifest_hash: Sha256

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        if self.model_ids != MODEL_IDS:
            raise ValueError("Prediction manifest model order differs from the frozen order")
        if set(self.prediction_file_sha256) != set(MODEL_IDS):
            raise ValueError("Prediction files differ from the frozen model order")
        if self.prediction_row_count_per_model != self.unique_target_count_per_model:
            raise ValueError("Every model must predict each target exactly once")
        expected_hash = canonical_sha256(self.model_dump(mode="json", exclude={"manifest_hash"}))
        if self.manifest_hash != expected_hash:
            raise ValueError("CSEDM prediction manifest hash does not match its content")
        return self


def publish_out_of_fold_predictions(
    *,
    adapter_root: Path,
    analysis_plan_path: Path,
    inventory_specification_path: Path,
    project_root: Path,
    pixi_lock_path: Path,
    output_root: Path,
    code_revision: str,
) -> CSEDMPredictionManifest:
    """Run every frozen model on the same ten untouched test folds."""

    plan = load_analysis_plan(analysis_plan_path)
    adapter, adapter_bytes = _load_adapter(adapter_root, plan, analysis_plan_path)
    fold_rows = _load_verified_table(
        adapter_root / "restricted" / "fold_features.parquet",
        adapter.fold_features_parquet_sha256,
    )
    out_of_fold_rows = _load_verified_table(
        adapter_root / "restricted" / "out_of_fold_targets.parquet",
        adapter.out_of_fold_parquet_sha256,
    )
    expected_test_hashes = {_required_text(row, "record_hash") for row in out_of_fold_rows}
    predictions: dict[str, list[ModelPrediction]] = {model_id: [] for model_id in MODEL_IDS}
    audits: list[FoldPredictionAudit] = []

    for fold_id in plan.official_fold_ids:
        training = _model_inputs(fold_rows, fold_id=fold_id, split="training")
        test = _model_inputs(fold_rows, fold_id=fold_id, split="test")
        result = predict_fold(training, test, plan.primary.model)
        for model_id in MODEL_IDS:
            predictions[model_id].extend(result.predictions[model_id])
        audits.append(
            FoldPredictionAudit(
                fold_id=fold_id,
                training_target_count=len(training),
                test_target_count=len(test),
                training_positive_count=sum(row.first_correct for row in training),
                test_positive_count=sum(row.first_correct for row in test),
                training_prevalence=result.training_prevalence,
                problem_frequency_fallback_count=result.problem_frequency_fallback_count,
                logistic_iterations=result.logistic_iterations,
                completed_models=MODEL_IDS,
            )
        )

    ordered_predictions = {
        model_id: tuple(sorted(predictions[model_id], key=lambda row: (row.fold_id, row.target_id)))
        for model_id in MODEL_IDS
    }
    _validate_prediction_coverage(ordered_predictions, expected_test_hashes)
    schema = _prediction_schema()
    prediction_content = {
        model_id: parquet_bytes(schema, _prediction_rows(rows))
        for model_id, rows in ordered_predictions.items()
    }

    reference = _reference_report(project_root, inventory_specification_path, plan)
    model_specification = _model_specification(plan)
    fold_completion = _fold_completion(audits)
    restricted_root = output_root / "restricted" / "predictions"
    prediction_paths = {model_id: restricted_root / f"{model_id}.parquet" for model_id in MODEL_IDS}
    for model_id in MODEL_IDS:
        write_immutable_bytes(prediction_paths[model_id], prediction_content[model_id])
    model_path = output_root / "model_specification.json"
    fold_path = output_root / "fold_completion.json"
    reference_path = output_root / "reference_baseline.json"
    write_immutable_json(model_path, model_specification)
    write_immutable_json(fold_path, fold_completion)
    write_immutable_json(reference_path, reference)

    target_count = len(next(iter(ordered_predictions.values())))
    manifest_payload = {
        "schema_id": "csedm.prediction_manifest.v1",
        "schema_version": 1,
        "study_id": "csedm-uncertainty-triage-v1",
        "code_revision": code_revision,
        "pixi_lock_sha256": file_sha256(pixi_lock_path.read_bytes()),
        "analysis_plan_hash": plan.plan_hash,
        "analysis_plan_file_sha256": file_sha256(analysis_plan_path.read_bytes()),
        "adapter_manifest_hash": adapter.manifest_hash,
        "adapter_manifest_file_sha256": file_sha256(adapter_bytes),
        "adapter_fold_features_sha256": adapter.fold_features_parquet_sha256,
        "adapter_out_of_fold_sha256": adapter.out_of_fold_parquet_sha256,
        "model_ids": MODEL_IDS,
        "prediction_row_count_per_model": target_count,
        "unique_target_count_per_model": target_count,
        "prediction_file_sha256": {
            model_id: file_sha256(prediction_content[model_id]) for model_id in MODEL_IDS
        },
        "prediction_schema_sha256": canonical_sha256(_schema_payload(schema)),
        "model_specification_sha256": file_sha256(model_path.read_bytes()),
        "fold_completion_sha256": file_sha256(fold_path.read_bytes()),
        "reference_baseline_sha256": file_sha256(reference_path.read_bytes()),
        "all_official_folds_completed": True,
        "public_artifacts_contain_learner_values": False,
        "restricted_predictions_contain_learner_identifiers": True,
    }
    manifest_payload["manifest_hash"] = canonical_sha256(manifest_payload)
    manifest = CSEDMPredictionManifest.model_validate(manifest_payload)
    write_immutable_json(output_root / "prediction_manifest.json", manifest)
    return manifest


def _load_adapter(
    adapter_root: Path,
    plan: CSEDMAnalysisPlan,
    analysis_plan_path: Path,
) -> tuple[CSEDMAdapterManifest, bytes]:
    path = adapter_root / "adapter_manifest.json"
    content = path.read_bytes()
    adapter = CSEDMAdapterManifest.model_validate_json(content)
    if adapter.analysis_plan_hash != plan.plan_hash:
        raise CSEDMAdapterError("Adapter and analysis plan hashes differ")
    if adapter.analysis_plan_file_sha256 != file_sha256(analysis_plan_path.read_bytes()):
        raise CSEDMAdapterError("Adapter and analysis plan files differ")
    return adapter, content


def _load_verified_table(path: Path, expected_sha256: str) -> tuple[dict[str, object], ...]:
    content = path.read_bytes()
    if file_sha256(content) != expected_sha256:
        raise CSEDMAdapterError(f"Adapter table differs from its manifest: {path.name}")
    table = pq.read_table(pa.BufferReader(content))
    return tuple(table.to_pylist())


def _model_inputs(
    rows: tuple[dict[str, object], ...],
    *,
    fold_id: int,
    split: Literal["training", "test"],
) -> tuple[ModelInput, ...]:
    selected: list[ModelInput] = []
    for row in rows:
        if row.get("fold_id") != fold_id or row.get("split") != split:
            continue
        _validate_source_record(row)
        selected.append(
            ModelInput(
                fold_id=fold_id,
                target_id=_required_text(row, "target_id"),
                learner_id=_required_text(row, "learner_id"),
                problem_id=_required_text(row, "problem_id"),
                first_correct=_required_bool(row, "first_correct"),
                feature_values=tuple(_numeric_feature(row, name) for name in MODEL_FEATURE_COLUMNS),
                source_record_hash=_required_text(row, "record_hash"),
            )
        )
    return tuple(sorted(selected, key=lambda item: item.target_id))


def _validate_source_record(row: dict[str, object]) -> None:
    record_hash = _required_text(row, "record_hash")
    payload = {key: value for key, value in row.items() if key != "record_hash"}
    if canonical_sha256(payload) != record_hash:
        raise CSEDMAdapterError("An adapter row has an invalid record hash")


def _numeric_feature(row: dict[str, object], name: str) -> float:
    value = row.get(name)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    raise CSEDMAdapterError(f"Adapter feature {name} must be numeric")


def _required_text(row: dict[str, object], name: str) -> str:
    value = row.get(name)
    if not isinstance(value, str) or not value:
        raise CSEDMAdapterError(f"Adapter field {name} must be non-empty text")
    return value


def _required_bool(row: dict[str, object], name: str) -> bool:
    value = row.get(name)
    if not isinstance(value, bool):
        raise CSEDMAdapterError(f"Adapter field {name} must be Boolean")
    return value


def _validate_prediction_coverage(
    predictions: dict[str, tuple[ModelPrediction, ...]],
    expected_source_hashes: set[str],
) -> None:
    expected_count = len(expected_source_hashes)
    if expected_count == 0:
        raise CSEDMAdapterError("The adapter contains no out-of-fold targets")
    expected_targets: set[str] | None = None
    for model_id in MODEL_IDS:
        rows = predictions[model_id]
        target_ids = {row.target_id for row in rows}
        source_hashes = {row.source_record_hash for row in rows}
        if len(rows) != expected_count or len(target_ids) != expected_count:
            raise CSEDMAdapterError(f"{model_id} did not predict every target exactly once")
        if source_hashes != expected_source_hashes:
            raise CSEDMAdapterError(f"{model_id} predictions differ from adapter test rows")
        if expected_targets is None:
            expected_targets = target_ids
        elif target_ids != expected_targets:
            raise CSEDMAdapterError("Frozen models predicted different target sets")


def _prediction_rows(
    predictions: tuple[ModelPrediction, ...],
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for prediction in predictions:
        row: dict[str, object] = {
            "schema_id": "csedm.model_prediction.v1",
            "schema_version": 1,
            "model_id": prediction.model_id,
            "fold_id": prediction.fold_id,
            "target_id": prediction.target_id,
            "learner_id": prediction.learner_id,
            "problem_id": prediction.problem_id,
            "first_correct": prediction.first_correct,
            "probability_correct": prediction.probability_correct,
            "predicted_correct": prediction.predicted_correct,
            "uncertainty": prediction.uncertainty,
            "source_record_hash": prediction.source_record_hash,
        }
        row["record_hash"] = canonical_sha256(row)
        rows.append(row)
    return tuple(rows)


def _prediction_schema() -> Any:
    return pa.schema(
        [
            pa.field("schema_id", pa.string(), nullable=False),
            pa.field("schema_version", pa.int16(), nullable=False),
            pa.field("model_id", pa.string(), nullable=False),
            pa.field("fold_id", pa.int8(), nullable=False),
            pa.field("target_id", pa.string(), nullable=False),
            pa.field("learner_id", pa.string(), nullable=False),
            pa.field("problem_id", pa.string(), nullable=False),
            pa.field("first_correct", pa.bool_(), nullable=False),
            pa.field("probability_correct", pa.float64(), nullable=False),
            pa.field("predicted_correct", pa.bool_(), nullable=False),
            pa.field("uncertainty", pa.float64(), nullable=False),
            pa.field("source_record_hash", pa.string(), nullable=False),
            pa.field("record_hash", pa.string(), nullable=False),
        ],
        metadata={b"schema_id": b"csedm.model_prediction.v1"},
    )


def _schema_payload(schema: Any) -> dict[str, object]:
    return {
        "fields": [
            {"name": field.name, "type": str(field.type), "nullable": field.nullable}
            for field in schema
        ],
        "metadata": {
            key.decode("utf-8"): value.decode("utf-8")
            for key, value in sorted((schema.metadata or {}).items())
        },
    }


def _model_specification(plan: CSEDMAnalysisPlan) -> dict[str, object]:
    return {
        "schema_id": "csedm.model_specification_report.v1",
        "schema_version": 1,
        "model_order": list(MODEL_IDS),
        "training_label_prevalence": {
            "fit_scope": "current official training fold only",
            "probability": "share of successful training targets",
        },
        "training_problem_frequency": {
            "fit_scope": "current official training fold only",
            "probability": "share of successful training targets for the same problem",
            "unseen_problem_fallback": "training-fold label prevalence",
        },
        "logistic_learner_history": {
            "implementation_version": sklearn.__version__,
            "settings": plan.primary.model.model_dump(mode="json"),
            "input_columns": list(MODEL_FEATURE_COLUMNS),
        },
        "hyperparameter_search_performed": False,
        "test_fold_model_selection_performed": False,
        "calibration_model_fitted": False,
    }


def _fold_completion(audits: list[FoldPredictionAudit]) -> dict[str, object]:
    if tuple(row.fold_id for row in audits) != tuple(range(10)):
        raise CSEDMAdapterError("Prediction run did not complete all official folds in order")
    return {
        "schema_id": "csedm.fold_prediction_completion.v1",
        "schema_version": 1,
        "fold_count": len(audits),
        "all_official_folds_completed": True,
        "total_test_targets": sum(row.test_target_count for row in audits),
        "problem_frequency_fallback_count": sum(
            row.problem_frequency_fallback_count for row in audits
        ),
        "folds": [row.model_dump(mode="json") for row in audits],
    }


def _reference_report(
    project_root: Path,
    inventory_specification_path: Path,
    plan: CSEDMAnalysisPlan,
) -> ContractModel:
    specification = load_inventory_specification(inventory_specification_path)
    archive_path = (project_root / specification.archive_path).resolve()
    archive_bytes = archive_path.read_bytes()
    if file_sha256(archive_bytes) != plan.archive_sha256:
        raise CSEDMAdapterError("Reference archive differs from the frozen analysis plan")
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            return check_reference_baseline(archive, plan.reference_baseline)
    except zipfile.BadZipFile as error:
        raise CSEDMAdapterError("CSEDM source is not a valid ZIP archive") from error


def build_parser(project_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish frozen CSEDM out-of-fold predictions")
    parser.add_argument(
        "--adapter-root",
        type=Path,
        default=project_root / "artifacts" / "csedm-study" / "adapter-v1",
    )
    parser.add_argument(
        "--analysis-plan",
        type=Path,
        default=project_root / "configs" / "csedm-study" / "v1-analysis.yaml",
    )
    parser.add_argument(
        "--inventory-specification",
        type=Path,
        default=project_root / "configs" / "csedm-study" / "v1-inventory.yaml",
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
            dirty_message="Commit tracked changes before publishing CSEDM predictions",
            untracked_files="no",
            operation_name="CSEDM prediction publication",
            error_factory=CSEDMAdapterError,
        )
        output_root = args.output_root or (
            project_root / "artifacts" / "csedm-study" / f"predictions-{revision[:7]}"
        )
        manifest = publish_out_of_fold_predictions(
            adapter_root=args.adapter_root,
            analysis_plan_path=args.analysis_plan,
            inventory_specification_path=args.inventory_specification,
            project_root=project_root,
            pixi_lock_path=args.pixi_lock,
            output_root=output_root,
            code_revision=revision,
        )
    except Exception as error:
        print(
            json.dumps(
                {
                    "command": "csedm-predict",
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
                "command": "csedm-predict",
                "result": manifest.model_dump(mode="json"),
                "status": "ok",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0
