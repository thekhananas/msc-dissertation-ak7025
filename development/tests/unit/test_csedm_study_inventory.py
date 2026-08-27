"""Scientific boundary tests for the CSEDM source inventory."""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

import pytest
import yaml

from socratic_tutor.benchmark.hashing import file_sha256
from socratic_tutor.csedm_study.inventory import CSEDMInventoryError, create_inventory

PREDICT_COLUMNS = (
    "SubjectID",
    "ProblemID",
    "StartOrder",
    "FirstCorrect",
    "EverCorrect",
    "UsedHint",
    "Attempts",
)
MAIN_COLUMNS = (
    "EventType",
    "EventID",
    "Order",
    "SubjectID",
    "ToolInstances",
    "CodeStateID",
    "ServerTimestamp",
    "ProblemID",
    "Correct",
)


def test_inventory_publishes_aggregates_and_rejects_learner_leakage(tmp_path: Path) -> None:
    valid_archive = _write_archive(tmp_path / "valid.zip", leak_fold_zero=False)
    specification = _write_specification(tmp_path / "valid.yaml", valid_archive)
    output_root = tmp_path / "output"

    report = create_inventory(specification, project_root=tmp_path, output_root=output_root)

    assert report.prediction_targets.row_count == 20
    assert report.eligible_learner_count == 10
    assert report.target_problem_count == 2
    assert report.test_folds_cover_each_target_once is True
    assert all(fold.learner_overlap_count == 0 for fold in report.folds)
    assert "learner-00" not in (output_root / "data_card.md").read_text(encoding="utf-8")

    leaking_archive = _write_archive(tmp_path / "leaking.zip", leak_fold_zero=True)
    leaking_specification = _write_specification(tmp_path / "leaking.yaml", leaking_archive)
    with pytest.raises(CSEDMInventoryError, match="not learner-separated"):
        create_inventory(
            leaking_specification,
            project_root=tmp_path,
            output_root=tmp_path / "leaking-output",
        )


def _write_archive(path: Path, *, leak_fold_zero: bool) -> Path:
    predictions = _prediction_rows()
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("README.md", "Synthetic inventory fixture")
        archive.writestr("DatasetMetadata.csv", "Property,Value\nVersion,3\n")
        archive.writestr("Predict.csv", _csv_text(PREDICT_COLUMNS, predictions))
        archive.writestr(
            "MainTable.csv",
            _csv_text(
                MAIN_COLUMNS,
                [
                    {
                        "EventType": "Submit",
                        "EventID": f"event-{index}",
                        "Order": str(index + 1),
                        "SubjectID": row["SubjectID"],
                        "ToolInstances": "ITAP; Python",
                        "CodeStateID": f"code-{index}",
                        "ServerTimestamp": "2016-01-01T00:00:00",
                        "ProblemID": row["ProblemID"],
                        "Correct": row["FirstCorrect"],
                    }
                    for index, row in enumerate(predictions)
                ],
            ),
        )
        archive.writestr(
            "CodeStates/CodeState.csv",
            _csv_text(
                ("CodeStateID", "Code"),
                [
                    {"CodeStateID": f"code-{index}", "Code": "print(1)"}
                    for index in range(len(predictions))
                ],
            ),
        )
        for member in (
            "Example/cv_predict.csv",
            "Example/evaluation_overall.csv",
            "Example/model.txt",
        ):
            archive.writestr(member, "fixture\n")
        for fold_id in range(10):
            test = [row for row in predictions if row["SubjectID"] == f"learner-{fold_id:02d}"]
            training = [row for row in predictions if row not in test]
            if leak_fold_zero and fold_id == 0:
                training.append(test.pop())
            archive.writestr(
                f"CV/Fold{fold_id}/Training.csv",
                _csv_text(PREDICT_COLUMNS, training),
            )
            archive.writestr(
                f"CV/Fold{fold_id}/Test.csv",
                _csv_text(PREDICT_COLUMNS, test),
            )
    return path


def _prediction_rows() -> list[dict[str, str]]:
    return [
        {
            "SubjectID": f"learner-{learner_id:02d}",
            "ProblemID": f"problem-{problem_id}",
            "StartOrder": str(learner_id * 2 + problem_id + 1),
            "FirstCorrect": "TRUE" if (learner_id + problem_id) % 2 == 0 else "FALSE",
            "EverCorrect": "TRUE",
            "UsedHint": "FALSE",
            "Attempts": "1",
        }
        for learner_id in range(10)
        for problem_id in range(2)
    ]


def _csv_text(columns: tuple[str, ...], rows: list[dict[str, str]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _write_specification(path: Path, archive_path: Path) -> Path:
    payload = {
        "schema_id": "csedm.inventory_specification.v1",
        "schema_version": 1,
        "dataset_id": "csedm-2019-data-challenge",
        "dataset_version": "1.1",
        "archive_path": archive_path.name,
        "archive_sha256": file_sha256(archive_path.read_bytes()),
        "source_url": "https://example.test/csedm",
        "access_terms": "Permission is required to access this test dataset.",
        "licence_status": "No redistribution licence is supplied with this fixture.",
        "public_data_policy": "Publish aggregate values only and keep all row data private.",
        "required_members": [
            "README.md",
            "DatasetMetadata.csv",
            "MainTable.csv",
            "CodeStates/CodeState.csv",
            "Predict.csv",
            "Example/cv_predict.csv",
            "Example/evaluation_overall.csv",
            "Example/model.txt",
        ],
        "excluded_local_sources": ["data/All/"],
        "known_caveats": ["This is a synthetic archive used only for a contract test."],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path
