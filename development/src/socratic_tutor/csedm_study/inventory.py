"""Read-only inventory for the CSEDM 2019 Python data challenge archive."""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import Field, model_validator

from socratic_tutor.benchmark.artifacts import (
    artifact_locations,
    write_immutable_bytes,
    write_immutable_json,
)
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import canonical_sha256, file_sha256
from socratic_tutor.contracts import ContractModel
from socratic_tutor.csedm_study.source import (
    CODE_STATE_COLUMNS,
    FOLD_COUNT,
    MAIN_TABLE_COLUMNS,
    PREDICT_COLUMNS,
    CSEDMInventoryError,
    CsvRow,
    TableInventory,
    TargetKey,
)
from socratic_tutor.csedm_study.source import (
    read_table as _read_table,
)
from socratic_tutor.csedm_study.source import (
    target_key as _target_key,
)


class CSEDMInventorySpecification(ContractModel):
    """Frozen identity, access boundary, and required contents of the source archive."""

    schema_id: Literal["csedm.inventory_specification.v1"]
    schema_version: Literal[1]
    dataset_id: Literal["csedm-2019-data-challenge"]
    dataset_version: Literal["1.1"]
    archive_path: str = Field(min_length=1)
    archive_sha256: Sha256
    source_url: str = Field(min_length=1)
    access_terms: str = Field(min_length=20)
    licence_status: str = Field(min_length=20)
    public_data_policy: str = Field(min_length=20)
    required_members: tuple[str, ...] = Field(min_length=8)
    excluded_local_sources: tuple[str, ...] = Field(min_length=1)
    known_caveats: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_entries(self) -> Self:
        for entries in (
            self.required_members,
            self.excluded_local_sources,
            self.known_caveats,
        ):
            if len(entries) != len(set(entries)):
                raise ValueError("Inventory specification entries must be unique")
        return self


class FoldInventory(ContractModel):
    """Aggregate membership checks for one official learner-separated fold."""

    fold_id: int = Field(ge=0, lt=FOLD_COUNT)
    training_target_count: int = Field(ge=1)
    test_target_count: int = Field(ge=1)
    training_learner_count: int = Field(ge=1)
    test_learner_count: int = Field(ge=1)
    learner_overlap_count: Literal[0]
    target_overlap_count: Literal[0]
    covers_all_targets: Literal[True]


class CSEDMInventoryReport(ContractModel):
    """Publication-safe aggregate inventory of the local CSEDM archive."""

    schema_id: Literal["csedm.inventory_report.v1"] = "csedm.inventory_report.v1"
    schema_version: Literal[1] = 1
    dataset_id: Literal["csedm-2019-data-challenge"]
    dataset_version: Literal["1.1"]
    archive_sha256: Sha256
    archive_size_bytes: int = Field(gt=0)
    specification_sha256: Sha256
    source_url: str
    access_terms: str
    licence_status: str
    public_data_policy: str
    main_events: TableInventory
    code_states: TableInventory
    prediction_targets: TableInventory
    main_learner_count: int = Field(ge=1)
    eligible_learner_count: int = Field(ge=1)
    main_problem_count: int = Field(ge=1)
    target_problem_count: int = Field(ge=1)
    first_correct_true_count: int = Field(ge=0)
    first_correct_false_count: int = Field(ge=0)
    event_type_counts: dict[str, int]
    folds: tuple[FoldInventory, ...] = Field(min_length=FOLD_COUNT, max_length=FOLD_COUNT)
    test_folds_cover_each_target_once: Literal[True]
    excluded_local_sources: tuple[str, ...]
    known_caveats: tuple[str, ...]
    report_hash: Sha256

    @model_validator(mode="after")
    def validate_counts_and_hash(self) -> Self:
        if self.first_correct_true_count + self.first_correct_false_count != (
            self.prediction_targets.row_count
        ):
            raise ValueError("FirstCorrect counts do not cover every prediction target")
        expected_hash = canonical_sha256(self.model_dump(mode="json", exclude={"report_hash"}))
        if self.report_hash != expected_hash:
            raise ValueError("Inventory report hash does not match its content")
        return self


def load_inventory_specification(path: Path) -> CSEDMInventorySpecification:
    """Load and validate the frozen archive specification."""

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise CSEDMInventoryError(f"Could not read inventory specification: {path}") from error
    return CSEDMInventorySpecification.model_validate(payload)


def create_inventory(
    specification_path: Path,
    *,
    project_root: Path,
    output_root: Path,
) -> CSEDMInventoryReport:
    """Verify the source archive and publish aggregate inventory artifacts."""

    specification_bytes = specification_path.read_bytes()
    specification = load_inventory_specification(specification_path)
    archive_path = (project_root / specification.archive_path).resolve()
    archive_bytes = archive_path.read_bytes()
    if file_sha256(archive_bytes) != specification.archive_sha256:
        raise CSEDMInventoryError("CSEDM archive hash differs from the frozen specification")

    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            report_payload = _inventory_payload(archive, specification)
    except zipfile.BadZipFile as error:
        raise CSEDMInventoryError("CSEDM source is not a valid ZIP archive") from error

    report_payload.update(
        {
            "schema_id": "csedm.inventory_report.v1",
            "schema_version": 1,
            "archive_sha256": specification.archive_sha256,
            "archive_size_bytes": len(archive_bytes),
            "specification_sha256": file_sha256(specification_bytes),
        }
    )
    report_payload["report_hash"] = canonical_sha256(report_payload)
    report = CSEDMInventoryReport.model_validate(report_payload)

    write_immutable_json(output_root / "inventory_report.json", report)
    write_immutable_bytes(output_root / "data_card.md", _data_card(report).encode("utf-8"))
    return report


def _inventory_payload(
    archive: zipfile.ZipFile,
    specification: CSEDMInventorySpecification,
) -> dict[str, object]:
    members = archive.namelist()
    if len(members) != len(set(members)):
        raise CSEDMInventoryError("CSEDM archive contains duplicate member names")
    missing_members = sorted(set(specification.required_members) - set(members))
    if missing_members:
        raise CSEDMInventoryError(f"CSEDM archive is missing required members: {missing_members}")

    predict = _read_table(archive, "Predict.csv", PREDICT_COLUMNS)
    main = _read_table(archive, "MainTable.csv", MAIN_TABLE_COLUMNS)
    code_states = _read_table(archive, "CodeStates/CodeState.csv", CODE_STATE_COLUMNS)
    folds = _inventory_folds(archive, predict.rows)
    labels = Counter(row["FirstCorrect"].upper() for row in predict.rows)
    if set(labels) != {"TRUE", "FALSE"}:
        raise CSEDMInventoryError("FirstCorrect must contain only TRUE and FALSE")

    return {
        "dataset_id": specification.dataset_id,
        "dataset_version": specification.dataset_version,
        "source_url": specification.source_url,
        "access_terms": specification.access_terms,
        "licence_status": specification.licence_status,
        "public_data_policy": specification.public_data_policy,
        "main_events": main.summary,
        "code_states": code_states.summary,
        "prediction_targets": predict.summary,
        "main_learner_count": len({row["SubjectID"] for row in main.rows}),
        "eligible_learner_count": len({row["SubjectID"] for row in predict.rows}),
        "main_problem_count": len({row["ProblemID"] for row in main.rows}),
        "target_problem_count": len({row["ProblemID"] for row in predict.rows}),
        "first_correct_true_count": labels["TRUE"],
        "first_correct_false_count": labels["FALSE"],
        "event_type_counts": dict(sorted(Counter(row["EventType"] for row in main.rows).items())),
        "folds": folds,
        "test_folds_cover_each_target_once": True,
        "excluded_local_sources": specification.excluded_local_sources,
        "known_caveats": specification.known_caveats,
    }


def _inventory_folds(
    archive: zipfile.ZipFile,
    prediction_rows: tuple[CsvRow, ...],
) -> tuple[FoldInventory, ...]:
    all_keys = {_target_key(row) for row in prediction_rows}
    if len(all_keys) != len(prediction_rows):
        raise CSEDMInventoryError("Predict.csv contains duplicate prediction targets")

    seen_test_keys: set[TargetKey] = set()
    summaries: list[FoldInventory] = []
    for fold_id in range(FOLD_COUNT):
        training = _read_table(archive, f"CV/Fold{fold_id}/Training.csv", PREDICT_COLUMNS)
        test = _read_table(archive, f"CV/Fold{fold_id}/Test.csv", PREDICT_COLUMNS)
        training_keys = {_target_key(row) for row in training.rows}
        test_keys = {_target_key(row) for row in test.rows}
        if len(training_keys) != len(training.rows) or len(test_keys) != len(test.rows):
            raise CSEDMInventoryError(f"Fold {fold_id} contains duplicate targets")
        if training_keys & test_keys:
            raise CSEDMInventoryError(f"Fold {fold_id} reuses targets across training and test")
        if training_keys | test_keys != all_keys:
            raise CSEDMInventoryError(f"Fold {fold_id} does not cover Predict.csv exactly")
        training_learners = {row["SubjectID"] for row in training.rows}
        test_learners = {row["SubjectID"] for row in test.rows}
        learner_overlap = training_learners & test_learners
        if learner_overlap:
            raise CSEDMInventoryError(f"Fold {fold_id} is not learner-separated")
        if seen_test_keys & test_keys:
            raise CSEDMInventoryError("A prediction target appears in more than one test fold")
        seen_test_keys.update(test_keys)
        summaries.append(
            FoldInventory(
                fold_id=fold_id,
                training_target_count=len(training.rows),
                test_target_count=len(test.rows),
                training_learner_count=len(training_learners),
                test_learner_count=len(test_learners),
                learner_overlap_count=0,
                target_overlap_count=0,
                covers_all_targets=True,
            )
        )
    if seen_test_keys != all_keys:
        raise CSEDMInventoryError("Official test folds do not cover every target exactly once")
    return tuple(summaries)


def _data_card(report: CSEDMInventoryReport) -> str:
    sample = "\n".join(
        (
            f"- {report.prediction_targets.row_count} first-attempt prediction targets",
            f"- {report.eligible_learner_count} eligible learners",
            f"- {report.target_problem_count} target problems",
            (
                f"- {report.first_correct_true_count} successful and "
                f"{report.first_correct_false_count} unsuccessful first attempts"
            ),
            (
                f"- {len(report.folds)} official learner-separated folds; "
                "every target appears in one test fold"
            ),
            (
                f"- {report.main_events.row_count} earlier activity events from "
                f"{report.main_learner_count} learners across "
                f"{report.main_problem_count} problems"
            ),
        )
    )
    caveats = "\n".join(f"- {item}" for item in report.known_caveats)
    return (
        "# CSEDM Companion Study Data Card\n\n"
        "## Source\n\n"
        f"- Dataset: CSEDM 2019 Data Challenge, version {report.dataset_version}\n"
        f"- Source: {report.source_url}\n"
        f"- Archive SHA-256: `{report.archive_sha256}`\n"
        f"- Access: {report.access_terms}\n"
        f"- Licence: {report.licence_status}\n\n"
        "## Study sample\n\n"
        f"{sample}\n\n"
        "## Use in this dissertation\n\n"
        "The primary label is first-attempt correctness. Prediction features may use only "
        "activity recorded before each target row's `StartOrder`. Learners remain separated "
        "by the supplied folds.\n\n"
        "This companion study tests whether uncertain predictions identify likely errors for "
        "review. It does not evaluate executable probing, a tutoring intervention, or learning "
        "caused by a tutor.\n\n"
        "## Access boundary\n\n"
        f"{report.public_data_policy}\n\n"
        "The separate Java CodeWorkout release is excluded from this study.\n\n"
        "## Known caveats\n\n"
        f"{caveats}\n"
    )


def build_parser(project_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify and inventory the local CSEDM archive")
    parser.add_argument(
        "--specification",
        type=Path,
        default=project_root / "configs" / "csedm-study" / "v1-inventory.yaml",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "artifacts" / "csedm-study" / "inventory-v1",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parents[3]
    args = build_parser(project_root).parse_args(argv)
    try:
        report = create_inventory(
            args.specification,
            project_root=project_root,
            output_root=args.output_root,
        )
    except Exception as error:
        print(
            json.dumps(
                {
                    "command": "csedm-inventory",
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
                "command": "csedm-inventory",
                "result": report.model_dump(mode="json"),
                "status": "ok",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0
