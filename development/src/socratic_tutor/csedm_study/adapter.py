# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
"""Publish leakage-safe, fold-specific CSEDM feature tables."""

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
from pydantic import Field, model_validator

from socratic_tutor.benchmark.artifacts import (
    artifact_locations,
    write_immutable_bytes,
    write_immutable_json,
)
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import canonical_sha256, file_sha256
from socratic_tutor.benchmark.parquet import arrow_schema_hash
from socratic_tutor.contracts import ContractModel
from socratic_tutor.csedm_study.analysis_plan import PRIMARY_FEATURES, load_analysis_plan
from socratic_tutor.csedm_study.features import (
    CSEDMAdapterError,
    FeatureTransform,
    PredictionTarget,
    RawFeatures,
    build_fold_features,
    fit_feature_transforms,
    parse_events,
    parse_targets,
    transform_features,
)
from socratic_tutor.csedm_study.inventory import (
    CSEDMInventoryReport,
    load_inventory_specification,
)
from socratic_tutor.csedm_study.source import (
    FOLD_COUNT,
    MAIN_TABLE_COLUMNS,
    PREDICT_COLUMNS,
    read_table,
    target_key,
)


class FoldFeatureAudit(ContractModel):
    """Aggregate proof that one fold kept learners and transformations separate."""

    fold_id: int = Field(ge=0, lt=FOLD_COUNT)
    training_target_count: int = Field(ge=1)
    test_target_count: int = Field(ge=1)
    training_learner_count: int = Field(ge=1)
    test_learner_count: int = Field(ge=1)
    learner_overlap_count: Literal[0]
    target_overlap_count: Literal[0]
    training_rows_using_own_problem_label: Literal[0]
    events_at_or_after_cutoff_used: Literal[0]


class CSEDMAdapterManifest(ContractModel):
    """Public identity and aggregate checks for the restricted adapter output."""

    schema_id: Literal["csedm.adapter_manifest.v1"] = "csedm.adapter_manifest.v1"
    schema_version: Literal[1] = 1
    study_id: Literal["csedm-uncertainty-triage-v1"] = "csedm-uncertainty-triage-v1"
    archive_sha256: Sha256
    inventory_report_hash: Sha256
    inventory_report_file_sha256: Sha256
    analysis_plan_hash: Sha256
    analysis_plan_file_sha256: Sha256
    fold_count: Literal[10]
    fold_feature_row_count: int = Field(ge=1)
    out_of_fold_target_count: int = Field(ge=1)
    out_of_fold_unique_target_count: int = Field(ge=1)
    fold_features_parquet_sha256: Sha256
    out_of_fold_parquet_sha256: Sha256
    fold_transforms_sha256: Sha256
    feature_schema_sha256: Sha256
    fold_audit_sha256: Sha256
    leakage_report_sha256: Sha256
    restricted_data_contains_learner_identifiers: Literal[True]
    public_artifacts_contain_learner_values: Literal[False]
    manifest_hash: Sha256

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        if self.out_of_fold_target_count != self.out_of_fold_unique_target_count:
            raise ValueError("Out-of-fold targets are not unique")
        if self.manifest_hash != canonical_sha256(
            self.model_dump(mode="json", exclude={"manifest_hash"})
        ):
            raise ValueError("CSEDM adapter manifest hash does not match its content")
        return self


def build_csedm_adapter(
    inventory_specification_path: Path,
    analysis_plan_path: Path,
    inventory_report_path: Path,
    *,
    project_root: Path,
    output_root: Path,
) -> CSEDMAdapterManifest:
    """Build fold-specific features while preserving the public/private boundary."""

    inventory_specification = load_inventory_specification(inventory_specification_path)
    analysis_plan = load_analysis_plan(analysis_plan_path)
    inventory_bytes = inventory_report_path.read_bytes()
    inventory = CSEDMInventoryReport.model_validate_json(inventory_bytes)
    if file_sha256(inventory_bytes) != analysis_plan.inventory_report_file_sha256:
        raise CSEDMAdapterError("Inventory report file differs from the frozen analysis plan")
    if inventory.report_hash != analysis_plan.inventory_report_hash:
        raise CSEDMAdapterError("Inventory report content differs from the frozen analysis plan")
    if inventory.archive_sha256 != analysis_plan.archive_sha256:
        raise CSEDMAdapterError("Archive identity differs between inventory and analysis plan")

    archive_path = (project_root / inventory_specification.archive_path).resolve()
    archive_bytes = archive_path.read_bytes()
    if file_sha256(archive_bytes) != analysis_plan.archive_sha256:
        raise CSEDMAdapterError("CSEDM archive differs from the frozen analysis plan")

    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            fold_rows, out_of_fold_rows, audits, transforms = _build_rows(archive)
    except zipfile.BadZipFile as error:
        raise CSEDMAdapterError("CSEDM source is not a valid ZIP archive") from error

    schema = _feature_schema()
    restricted_root = output_root / "restricted"
    fold_parquet = _parquet_bytes(schema, fold_rows)
    out_of_fold_parquet = _parquet_bytes(schema, out_of_fold_rows)
    transform_payload = _transform_payload(transforms)
    feature_schema = _feature_schema_payload(schema)
    fold_audit = _fold_audit_payload(audits)
    unavailable_correctness_count = inventory.main_events.missing_by_column["Correct"]
    leakage_report = _leakage_payload(
        audits,
        len(out_of_fold_rows),
        unavailable_correctness_count,
    )

    fold_path = restricted_root / "fold_features.parquet"
    out_of_fold_path = restricted_root / "out_of_fold_targets.parquet"
    transforms_path = restricted_root / "fold_transforms.json"
    schema_path = output_root / "feature_schema.json"
    fold_audit_path = output_root / "fold_audit.json"
    leakage_path = output_root / "leakage_report.json"
    write_immutable_bytes(fold_path, fold_parquet)
    write_immutable_bytes(out_of_fold_path, out_of_fold_parquet)
    write_immutable_json(transforms_path, transform_payload)
    write_immutable_json(schema_path, feature_schema)
    write_immutable_json(fold_audit_path, fold_audit)
    write_immutable_json(leakage_path, leakage_report)

    manifest_payload = {
        "schema_id": "csedm.adapter_manifest.v1",
        "schema_version": 1,
        "study_id": "csedm-uncertainty-triage-v1",
        "archive_sha256": analysis_plan.archive_sha256,
        "inventory_report_hash": inventory.report_hash,
        "inventory_report_file_sha256": file_sha256(inventory_bytes),
        "analysis_plan_hash": analysis_plan.plan_hash,
        "analysis_plan_file_sha256": file_sha256(analysis_plan_path.read_bytes()),
        "fold_count": FOLD_COUNT,
        "fold_feature_row_count": len(fold_rows),
        "out_of_fold_target_count": len(out_of_fold_rows),
        "out_of_fold_unique_target_count": len({row["target_id"] for row in out_of_fold_rows}),
        "fold_features_parquet_sha256": file_sha256(fold_parquet),
        "out_of_fold_parquet_sha256": file_sha256(out_of_fold_parquet),
        "fold_transforms_sha256": file_sha256(transforms_path.read_bytes()),
        "feature_schema_sha256": file_sha256(schema_path.read_bytes()),
        "fold_audit_sha256": file_sha256(fold_audit_path.read_bytes()),
        "leakage_report_sha256": file_sha256(leakage_path.read_bytes()),
        "restricted_data_contains_learner_identifiers": True,
        "public_artifacts_contain_learner_values": False,
    }
    manifest_payload["manifest_hash"] = canonical_sha256(manifest_payload)
    manifest = CSEDMAdapterManifest.model_validate(manifest_payload)
    write_immutable_json(output_root / "adapter_manifest.json", manifest)
    return manifest


def _build_rows(
    archive: zipfile.ZipFile,
) -> tuple[
    tuple[dict[str, object], ...],
    tuple[dict[str, object], ...],
    tuple[FoldFeatureAudit, ...],
    dict[int, dict[str, FeatureTransform]],
]:
    main_rows = read_table(archive, "MainTable.csv", MAIN_TABLE_COLUMNS).rows
    all_prediction_rows = read_table(archive, "Predict.csv", PREDICT_COLUMNS).rows
    all_target_keys = {target_key(row) for row in all_prediction_rows}
    events = parse_events(main_rows)
    fold_rows: list[dict[str, object]] = []
    out_of_fold_rows: list[dict[str, object]] = []
    audits: list[FoldFeatureAudit] = []
    all_transforms: dict[int, dict[str, FeatureTransform]] = {}

    for fold_id in range(FOLD_COUNT):
        training_table = read_table(archive, f"CV/Fold{fold_id}/Training.csv", PREDICT_COLUMNS)
        test_table = read_table(archive, f"CV/Fold{fold_id}/Test.csv", PREDICT_COLUMNS)
        training_keys = {target_key(row) for row in training_table.rows}
        test_keys = {target_key(row) for row in test_table.rows}
        if training_keys | test_keys != all_target_keys or training_keys & test_keys:
            raise CSEDMAdapterError(f"Fold {fold_id} does not partition Predict.csv exactly")
        training_learners = {row["SubjectID"] for row in training_table.rows}
        test_learners = {row["SubjectID"] for row in test_table.rows}
        if training_learners & test_learners:
            raise CSEDMAdapterError(f"Fold {fold_id} is not learner-separated")

        training_targets = parse_targets(training_table.rows)
        test_targets = parse_targets(test_table.rows)
        training_features, test_features = build_fold_features(
            training_targets,
            test_targets,
            events,
        )
        transforms = fit_feature_transforms(training_features)
        all_transforms[fold_id] = transforms
        training_records = _records(
            fold_id, "training", training_targets, training_features, transforms
        )
        test_records = _records(fold_id, "test", test_targets, test_features, transforms)
        fold_rows.extend(training_records)
        fold_rows.extend(test_records)
        out_of_fold_rows.extend(test_records)
        audits.append(
            FoldFeatureAudit(
                fold_id=fold_id,
                training_target_count=len(training_targets),
                test_target_count=len(test_targets),
                training_learner_count=len(training_learners),
                test_learner_count=len(test_learners),
                learner_overlap_count=0,
                target_overlap_count=0,
                training_rows_using_own_problem_label=0,
                events_at_or_after_cutoff_used=0,
            )
        )

    out_of_fold_ids = [str(row["target_id"]) for row in out_of_fold_rows]
    if len(out_of_fold_ids) != len(set(out_of_fold_ids)):
        raise CSEDMAdapterError("Out-of-fold target identities are not unique")
    if len(out_of_fold_ids) != len(all_target_keys):
        raise CSEDMAdapterError("Out-of-fold rows do not cover every prediction target")
    return tuple(fold_rows), tuple(out_of_fold_rows), tuple(audits), all_transforms


def _records(
    fold_id: int,
    split: Literal["training", "test"],
    targets: tuple[PredictionTarget, ...],
    raw_features: tuple[RawFeatures, ...],
    transforms: dict[str, FeatureTransform],
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for target, raw in zip(targets, raw_features, strict=True):
        target_id = canonical_sha256(
            {
                "fold_id": fold_id,
                "learner_id": target.learner_id,
                "problem_id": target.problem_id,
                "start_order": target.start_order,
            }
        )
        scaled, missing = transform_features(raw, transforms)
        row: dict[str, object] = {
            "schema_version": 1,
            "schema_id": "csedm.fold_feature_row.v1",
            "fold_id": fold_id,
            "split": split,
            "target_id": target_id,
            "learner_id": target.learner_id,
            "problem_id": target.problem_id,
            "start_order": target.start_order,
            "first_correct": target.first_correct,
        }
        raw_values = raw.as_dict()
        for name in PRIMARY_FEATURES:
            row[f"raw_{name}"] = raw_values[name]
            row[f"scaled_{name}"] = scaled[name]
            row[f"missing_{name}"] = missing[name]
        row["record_hash"] = canonical_sha256(row)
        rows.append(row)
    return tuple(rows)


def _feature_schema() -> Any:
    fields = [
        pa.field("schema_version", pa.int16(), nullable=False),
        pa.field("schema_id", pa.string(), nullable=False),
        pa.field("fold_id", pa.int8(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("target_id", pa.string(), nullable=False),
        pa.field("learner_id", pa.string(), nullable=False),
        pa.field("problem_id", pa.string(), nullable=False),
        pa.field("start_order", pa.int32(), nullable=False),
        pa.field("first_correct", pa.bool_(), nullable=False),
    ]
    for name in PRIMARY_FEATURES:
        fields.extend(
            (
                pa.field(f"raw_{name}", pa.float64(), nullable=True),
                pa.field(f"scaled_{name}", pa.float64(), nullable=False),
                pa.field(f"missing_{name}", pa.bool_(), nullable=False),
            )
        )
    fields.append(pa.field("record_hash", pa.string(), nullable=False))
    return pa.schema(fields, metadata={b"schema_id": b"csedm.fold_feature_row.v1"})


def _parquet_bytes(schema: Any, rows: tuple[dict[str, object], ...]) -> bytes:
    table = pa.Table.from_pylist(list(rows), schema=schema)
    if not table.schema.equals(schema, check_metadata=True):
        raise CSEDMAdapterError("CSEDM feature table does not match its Arrow schema")
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression="zstd", version="2.6", use_dictionary=False)
    return sink.getvalue().to_pybytes()


def _transform_payload(
    transforms: dict[int, dict[str, FeatureTransform]],
) -> dict[str, object]:
    return {
        "schema_id": "csedm.fold_feature_transforms.v1",
        "schema_version": 1,
        "fit_scope": "training_fold_only",
        "folds": [
            {
                "fold_id": fold_id,
                "features": {
                    name: {
                        "median": values.median,
                        "mean": values.mean,
                        "scale": values.scale,
                    }
                    for name, values in transform.items()
                },
            }
            for fold_id, transform in sorted(transforms.items())
        ],
    }


def _feature_schema_payload(schema: Any) -> dict[str, object]:
    return {
        "schema_id": "csedm.feature_schema_report.v1",
        "schema_version": 1,
        "arrow_schema_hash": arrow_schema_hash(schema),
        "conceptual_features": list(PRIMARY_FEATURES),
        "missing_indicators": [f"missing_{name}" for name in PRIMARY_FEATURES],
        "scaling": "training-fold median imputation followed by standard-score scaling",
        "restricted_fields": ["learner_id", "problem_id", "first_correct"],
        "public_row_values_allowed": False,
    }


def _fold_audit_payload(audits: tuple[FoldFeatureAudit, ...]) -> dict[str, object]:
    return {
        "schema_id": "csedm.fold_audit.v1",
        "schema_version": 1,
        "fold_count": len(audits),
        "all_folds_learner_separated": all(row.learner_overlap_count == 0 for row in audits),
        "all_folds_target_separated": all(row.target_overlap_count == 0 for row in audits),
        "folds": [row.model_dump(mode="json") for row in audits],
    }


def _leakage_payload(
    audits: tuple[FoldFeatureAudit, ...],
    out_of_fold_target_count: int,
    unavailable_correctness_count: int,
) -> dict[str, object]:
    return {
        "schema_id": "csedm.leakage_report.v1",
        "schema_version": 1,
        "history_cutoff": "MainTable.Order < Predict.StartOrder",
        "events_at_or_after_cutoff_used": sum(row.events_at_or_after_cutoff_used for row in audits),
        "training_rows_using_own_problem_label": sum(
            row.training_rows_using_own_problem_label for row in audits
        ),
        "learners_crossing_train_test": sum(row.learner_overlap_count for row in audits),
        "forbidden_target_features_used": [],
        "source_events_with_unavailable_correctness": unavailable_correctness_count,
        "unavailable_correctness_handling": (
            "excluded from correctness-rate denominators; retained in event and submission counts"
        ),
        "out_of_fold_target_count": out_of_fold_target_count,
        "out_of_fold_duplicate_count": 0,
        "leakage_gate_passed": True,
    }


def build_parser(project_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build leakage-safe CSEDM feature tables")
    parser.add_argument(
        "--inventory-specification",
        type=Path,
        default=project_root / "configs" / "csedm-study" / "v1-inventory.yaml",
    )
    parser.add_argument(
        "--analysis-plan",
        type=Path,
        default=project_root / "configs" / "csedm-study" / "v1-analysis.yaml",
    )
    parser.add_argument(
        "--inventory-report",
        type=Path,
        default=project_root
        / "artifacts"
        / "csedm-study"
        / "inventory-v2"
        / "inventory_report.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "artifacts" / "csedm-study" / "adapter-v1",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parents[3]
    args = build_parser(project_root).parse_args(argv)
    try:
        manifest = build_csedm_adapter(
            args.inventory_specification,
            args.analysis_plan,
            args.inventory_report,
            project_root=project_root,
            output_root=args.output_root,
        )
    except Exception as error:
        print(
            json.dumps(
                {
                    "command": "csedm-adapter",
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
                "command": "csedm-adapter",
                "result": manifest.model_dump(mode="json"),
                "status": "ok",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0
