"""Tests for the labelled, network-free v1 integrity replay."""

import json
from pathlib import Path

from pytest import CaptureFixture

from socratic_tutor.benchmark.cli import main
from socratic_tutor.benchmark.reference_integrity import (
    load_reference_integrity_plan,
    run_reference_integrity_replay,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = WORKSPACE_ROOT / "data" / "benchmarks" / "v1" / "manifest.yaml"
PLAN_PATH = WORKSPACE_ROOT / "configs" / "benchmark" / "v1-reference-integrity.yaml"


def test_reference_integrity_replay_covers_all_frozen_routes(tmp_path: Path) -> None:
    plan = load_reference_integrity_plan(PLAN_PATH)

    report = run_reference_integrity_replay(
        plan=plan,
        manifest_path=MANIFEST_PATH,
        output_root=tmp_path,
    )

    assert report.case_count == 24
    assert report.condition_prediction_count == 96
    assert report.criterion_record_count == 24
    assert (tmp_path / "decision" / "commitment_manifest.json").exists()
    assert (tmp_path / "criterion" / "recorded_responses.jsonl").exists()
    evaluation = json.loads((tmp_path / "evaluation_summary.json").read_text(encoding="utf-8"))
    assert evaluation["row_counts"] == {
        "baseline_results": 48,
        "case_aggregates": 120,
        "condition_predictions": 96,
        "criterion_records": 24,
        "repeat_metrics": 144,
    }

    repeated = run_reference_integrity_replay(
        plan=plan,
        manifest_path=MANIFEST_PATH,
        output_root=tmp_path,
    )
    assert repeated.integrity_hash == report.integrity_hash


def test_reference_integrity_command_is_replayable(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    arguments = [
        "reference-integrity",
        "--plan",
        str(PLAN_PATH),
        "--manifest",
        str(MANIFEST_PATH),
        "--output-root",
        str(tmp_path),
    ]

    assert main(arguments) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["result"]["condition_prediction_count"] == 96

    assert main(arguments) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["result"]["integrity_hash"] == first["result"]["integrity_hash"]
