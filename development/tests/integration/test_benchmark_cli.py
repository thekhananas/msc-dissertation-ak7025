"""Command-level benchmark workflow and failure-exit tests."""

import json
import subprocess
import sys
from pathlib import Path

import yaml
from pytest import CaptureFixture

from socratic_tutor.benchmark.cli import main

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = WORKSPACE_ROOT / "data" / "benchmarks" / "dev-v0"
MANIFEST = BENCHMARK_ROOT / "manifest.yaml"
DECISION_PLAN = WORKSPACE_ROOT / "configs" / "benchmark" / "dev-v0-decision.yaml"
CRITERION_PLAN = WORKSPACE_ROOT / "configs" / "benchmark" / "dev-v0-criterion.yaml"
SHORTCUT_PLAN = WORKSPACE_ROOT / "configs" / "benchmark" / "dev-v0-shortcuts.yaml"
SENSITIVITY_PLAN = WORKSPACE_ROOT / "configs" / "benchmark" / "dev-v0-sensitivity.yaml"


def test_smoke_command_publishes_and_replays_complete_two_case_artifact(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    output = tmp_path / "smoke"
    arguments = [
        "smoke",
        "--manifest",
        str(MANIFEST),
        "--decision-plan",
        str(DECISION_PLAN),
        "--criterion-plan",
        str(CRITERION_PLAN),
        "--output-root",
        str(output),
    ]

    assert main(arguments) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["status"] == "ok"
    smoke_hash = first["result"]["smoke_hash"]
    evaluation = json.loads((output / "evaluation_summary.json").read_text(encoding="utf-8"))
    assert evaluation["row_counts"] == {
        "baseline_results": 4,
        "case_aggregates": 10,
        "condition_predictions": 8,
        "criterion_records": 2,
        "repeat_metrics": 12,
    }
    assert evaluation["demonstrated_count"] == 1
    assert evaluation["not_demonstrated_count"] == 1
    assert (output / "decision" / "commitment_manifest.json").exists()
    assert (output / "decision" / "recorded_responses.jsonl").exists()
    assert (output / "criterion" / "recorded_responses.jsonl").exists()

    assert main(arguments) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["result"]["smoke_hash"] == smoke_hash


def test_explicit_generate_phases_and_evaluate_command(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    projected = tmp_path / "projected"
    assert (
        main(
            [
                "smoke",
                "--manifest",
                str(MANIFEST),
                "--decision-plan",
                str(DECISION_PLAN),
                "--criterion-plan",
                str(CRITERION_PLAN),
                "--output-root",
                str(projected),
            ]
        )
        == 0
    )
    capsys.readouterr()
    output = tmp_path / "phased"
    decision_arguments = [
        "generate",
        "decision",
        "--public-manifest",
        str(projected / "public_manifest.json"),
        "--benchmark-root",
        str(BENCHMARK_ROOT),
        "--plan",
        str(DECISION_PLAN),
        "--output-root",
        str(output),
    ]
    decision_process = _run_generate(decision_arguments[1:])
    assert decision_process.returncode == 0, decision_process.stderr
    decision_result = json.loads(decision_process.stdout)
    assert decision_result["result"]["prediction_count"] == 8

    criterion_arguments = [
        "generate",
        "criterion",
        "--public-manifest",
        str(projected / "public_manifest.json"),
        "--evaluator-manifest",
        str(projected / "evaluator_manifest.json"),
        "--benchmark-root",
        str(BENCHMARK_ROOT),
        "--plan",
        str(CRITERION_PLAN),
        "--output-root",
        str(output),
    ]
    criterion_process = _run_generate(criterion_arguments[1:])
    assert criterion_process.returncode == 0, criterion_process.stderr
    criterion_result = json.loads(criterion_process.stdout)
    assert criterion_result["result"]["criterion_count"] == 2

    report_path = output / "rebuilt-evaluation.json"
    assert (
        main(
            [
                "evaluate",
                "--dataset-root",
                str(output / "datasets"),
                "--output",
                str(report_path),
            ]
        )
        == 0
    )
    evaluation_result = json.loads(capsys.readouterr().out)
    assert evaluation_result["result"]["row_counts"]["case_aggregates"] == 10
    assert report_path.exists()


def test_research_check_commands_emit_gate_status_and_nonzero_failure(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    shortcut_output = tmp_path / "shortcuts.json"
    assert (
        main(
            [
                "shortcuts",
                "--input",
                str(SHORTCUT_PLAN),
                "--output",
                str(shortcut_output),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["result"]["surface_accuracy"] == 0.5

    strict = yaml.safe_load(SHORTCUT_PLAN.read_text(encoding="utf-8"))
    strict["maximum_surface_accuracy"] = 0.25
    strict_path = tmp_path / "strict-shortcuts.yaml"
    strict_path.write_text(yaml.safe_dump(strict, sort_keys=False), encoding="utf-8")
    assert (
        main(
            [
                "shortcuts",
                "--input",
                str(strict_path),
                "--output",
                str(tmp_path / "strict-output.json"),
            ]
        )
        == 2
    )
    assert json.loads(capsys.readouterr().out)["status"] == "gate_failed"

    assert (
        main(
            [
                "sensitivity",
                "--input",
                str(SENSITIVITY_PLAN),
                "--output",
                str(tmp_path / "sensitivity.json"),
            ]
        )
        == 0
    )
    sensitivity = json.loads(capsys.readouterr().out)["result"]
    assert sensitivity["gate_passed"] is True
    assert sensitivity["case_count"] == 24


def test_commands_fail_closed_for_draft_or_invalid_inputs(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    assert (
        main(
            [
                "validate",
                "--manifest",
                str(MANIFEST),
                "--require-frozen",
            ]
        )
        == 1
    )
    validation_error = json.loads(capsys.readouterr().err)
    assert validation_error["status"] == "error"
    assert "frozen benchmark" in validation_error["message"]

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("schema_id: wrong\n", encoding="utf-8")
    assert (
        main(
            [
                "sensitivity",
                "--input",
                str(invalid),
                "--output",
                str(tmp_path / "never-created.json"),
            ]
        )
        == 1
    )
    input_error = json.loads(capsys.readouterr().err)
    assert input_error["status"] == "error"
    assert input_error["error_type"] == "ValidationError"


def test_smoke_does_not_persist_evaluator_inputs_before_decision_seal(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    output = tmp_path / "stopped-after-decision"

    assert (
        main(
            [
                "smoke",
                "--manifest",
                str(MANIFEST),
                "--decision-plan",
                str(DECISION_PLAN),
                "--criterion-plan",
                str(tmp_path / "missing-criterion-plan.yaml"),
                "--output-root",
                str(output),
            ]
        )
        == 1
    )
    capsys.readouterr()
    assert (output / "decision" / "commitment_manifest.json").exists()
    assert (output / "decision_summary.json").exists()
    assert not (output / "evaluator_manifest.json").exists()
    assert not (output / "offline_criterion_plan.json").exists()


def test_offline_generation_rejects_held_out_projection(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    projected = tmp_path / "projected"
    assert (
        main(
            [
                "smoke",
                "--manifest",
                str(MANIFEST),
                "--decision-plan",
                str(DECISION_PLAN),
                "--criterion-plan",
                str(CRITERION_PLAN),
                "--output-root",
                str(projected),
            ]
        )
        == 0
    )
    capsys.readouterr()
    public = json.loads((projected / "public_manifest.json").read_text(encoding="utf-8"))
    public["cases"][0]["split"] = "held_out"
    held_out = tmp_path / "held-out-public.json"
    held_out.write_text(json.dumps(public), encoding="utf-8")

    process = _run_generate(
        [
            "decision",
            "--public-manifest",
            str(held_out),
            "--benchmark-root",
            str(BENCHMARK_ROOT),
            "--plan",
            str(DECISION_PLAN),
            "--output-root",
            str(tmp_path / "held-out-output"),
        ]
    )
    assert process.returncode == 1
    failure = json.loads(process.stderr)
    assert "held-out generation is disabled" in failure["message"]


def test_external_decision_seal_fails_closed_before_modal_for_missing_sources(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    missing = tmp_path / "missing.json"
    output = tmp_path / "never-created"

    assert (
        main(
            [
                "external-decision-seal",
                "--protocol",
                str(missing),
                "--preflight",
                str(missing),
                "--generation-report",
                str(missing),
                "--recorded-responses",
                str(missing),
                "--public-rating-report",
                str(missing),
                "--methodology-clarification",
                str(missing),
                "--inferential-hierarchy",
                str(missing),
                "--manifest",
                str(missing),
                "--analysis-specification",
                str(missing),
                "--pixi-lock",
                str(missing),
                "--output-root",
                str(output),
                "--code-revision",
                "external-seal-cli-test",
                "--created-at-utc",
                "2026-09-01T12:00:00Z",
            ]
        )
        == 1
    )
    error = json.loads(capsys.readouterr().err)
    assert error["command"] == "external-decision-seal"
    assert error["status"] == "error"
    assert not output.exists()


def test_external_criterion_run_fails_closed_before_provider_for_missing_audit(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    missing = tmp_path / "missing.json"

    assert (
        main(
            [
                "external-criterion-run",
                "--protocol",
                str(missing),
                "--accounting-report",
                str(missing),
                "--manifest",
                str(missing),
                "--system-prompt",
                str(missing),
                "--seal-root",
                str(tmp_path / "seal"),
                "--pixi-lock",
                str(missing),
                "--code-revision",
                "criterion-cli-test",
                "--created-at-utc",
                "2026-09-01T10:00:00Z",
            ]
        )
        == 1
    )
    error = json.loads(capsys.readouterr().err)
    assert error["command"] == "external-criterion-run"
    assert error["status"] == "error"
    assert not (tmp_path / "seal").exists()


def test_external_criterion_audit_fails_closed_for_missing_sources(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    missing = tmp_path / "missing.json"
    output = tmp_path / "never-created.json"

    assert (
        main(
            [
                "external-criterion-audit",
                "--protocol",
                str(missing),
                "--accounting-report",
                str(missing),
                "--manifest",
                str(missing),
                "--system-prompt",
                str(missing),
                "--seal-root",
                str(tmp_path / "seal"),
                "--pixi-lock",
                str(missing),
                "--output",
                str(output),
                "--audit-code-revision",
                "criterion-audit-cli-test",
                "--audited-at-utc",
                "2026-09-01T11:00:00Z",
            ]
        )
        == 1
    )
    error = json.loads(capsys.readouterr().err)
    assert error["command"] == "external-criterion-audit"
    assert error["status"] == "error"
    assert not output.exists()


def test_external_replay_fails_closed_for_missing_sources(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    missing = tmp_path / "missing.json"
    output = tmp_path / "never-created"

    assert (
        main(
            [
                "external-replay",
                "--integrity-report",
                str(missing),
                "--decision-responses",
                str(missing),
                "--seal-root",
                str(tmp_path / "seal"),
                "--pixi-lock",
                str(missing),
                "--output-root",
                str(output),
                "--code-revision",
                "external-replay-cli-test",
                "--created-at-utc",
                "2026-09-01T12:00:00Z",
            ]
        )
        == 1
    )
    error = json.loads(capsys.readouterr().err)
    assert error["command"] == "external-replay"
    assert error["status"] == "error"
    assert not output.exists()


def test_external_scoring_fails_closed_for_missing_sources(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    missing = tmp_path / "missing.json"
    seal = tmp_path / "seal"

    assert (
        main(
            [
                "external-score",
                "--replay-report",
                str(missing),
                "--analysis-specification",
                str(missing),
                "--calibration-report",
                str(missing),
                "--public-manifest",
                str(missing),
                "--benchmark-root",
                str(tmp_path / "benchmark"),
                "--seal-root",
                str(seal),
                "--pixi-lock",
                str(missing),
                "--code-revision",
                "external-score-cli-test",
                "--created-at-utc",
                "2026-09-01T13:00:00Z",
            ]
        )
        == 1
    )
    error = json.loads(capsys.readouterr().err)
    assert error["command"] == "external-score"
    assert error["status"] == "error"
    assert not seal.exists()


def test_external_primary_analysis_fails_closed_for_missing_sources(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    missing = tmp_path / "missing.json"
    output = tmp_path / "primary"

    assert (
        main(
            [
                "external-primary-analyze",
                "--scoring-summary",
                str(missing),
                "--analysis-specification",
                str(missing),
                "--inferential-hierarchy",
                str(missing),
                "--dataset-root",
                str(tmp_path / "datasets"),
                "--pixi-lock",
                str(missing),
                "--output-root",
                str(output),
                "--code-revision",
                "primary-analysis-cli-test",
                "--created-at-utc",
                "2026-09-01T14:00:00Z",
            ]
        )
        == 1
    )
    error = json.loads(capsys.readouterr().err)
    assert error["command"] == "external-primary-analyze"
    assert error["status"] == "error"
    assert not output.exists()


def test_external_secondary_analysis_fails_closed_for_missing_sources(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    missing = tmp_path / "missing.json"
    output = tmp_path / "secondary"

    assert (
        main(
            [
                "external-secondary-analyze",
                "--scoring-summary",
                str(missing),
                "--primary-plan",
                str(missing),
                "--primary-report",
                str(missing),
                "--analysis-specification",
                str(missing),
                "--inferential-hierarchy",
                str(missing),
                "--specificity-amendment",
                str(missing),
                "--dataset-root",
                str(tmp_path / "datasets"),
                "--pixi-lock",
                str(missing),
                "--output-root",
                str(output),
                "--code-revision",
                "secondary-analysis-cli-test",
                "--created-at-utc",
                "2026-09-01T15:00:00Z",
            ]
        )
        == 1
    )
    error = json.loads(capsys.readouterr().err)
    assert error["command"] == "external-secondary-analyze"
    assert error["status"] == "error"
    assert not output.exists()


def test_external_dependence_sensitivity_fails_closed_for_missing_sources(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    missing = tmp_path / "missing.json"
    output = tmp_path / "dependence"

    assert (
        main(
            [
                "external-dependence-sensitivity",
                "--grid",
                str(missing),
                "--primary-plan",
                str(missing),
                "--primary-report",
                str(missing),
                "--secondary-report",
                str(missing),
                "--inferential-hierarchy",
                str(missing),
                "--benchmark-design",
                str(missing),
                "--dataset-root",
                str(tmp_path / "datasets"),
                "--pixi-lock",
                str(missing),
                "--output-root",
                str(output),
                "--code-revision",
                "dependence-sensitivity-cli-test",
                "--created-at-utc",
                "2026-09-01T16:00:00Z",
            ]
        )
        == 1
    )
    error = json.loads(capsys.readouterr().err)
    assert error["command"] == "external-dependence-sensitivity"
    assert error["status"] == "error"
    assert not output.exists()


def test_external_dependence_figure_fails_closed_for_missing_source(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    pdf_output = tmp_path / "dependence.pdf"
    svg_output = tmp_path / "dependence.svg"
    manifest = tmp_path / "figure_manifest.json"

    assert (
        main(
            [
                "external-dependence-figure",
                "--report",
                str(tmp_path / "missing.json"),
                "--pdf-output",
                str(pdf_output),
                "--svg-output",
                str(svg_output),
                "--manifest",
                str(manifest),
            ]
        )
        == 1
    )
    error = json.loads(capsys.readouterr().err)
    assert error["command"] == "external-dependence-figure"
    assert error["status"] == "error"
    assert not pdf_output.exists()
    assert not svg_output.exists()
    assert not manifest.exists()


def test_external_missingness_bound_fails_closed_for_missing_sources(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    missing = tmp_path / "missing.json"
    output = tmp_path / "missingness"

    assert (
        main(
            [
                "external-missingness-bound",
                "--primary-plan",
                str(missing),
                "--primary-report",
                str(missing),
                "--dataset-root",
                str(tmp_path / "datasets"),
                "--pixi-lock",
                str(missing),
                "--output-root",
                str(output),
                "--code-revision",
                "missingness-cli-test",
                "--created-at-utc",
                "2026-09-01T16:02:00Z",
            ]
        )
        == 1
    )
    error = json.loads(capsys.readouterr().err)
    assert error["command"] == "external-missingness-bound"
    assert error["status"] == "error"
    assert not output.exists()


def test_public_rating_reliability_fails_closed_for_missing_sources(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    missing = tmp_path / "missing.json"
    output = tmp_path / "reliability"

    assert (
        main(
            [
                "public-rating-reliability",
                "--rating-report",
                str(missing),
                "--procedure-record",
                str(missing),
                "--ratings-a",
                str(missing),
                "--ratings-b",
                str(missing),
                "--pixi-lock",
                str(missing),
                "--output-root",
                str(output),
                "--code-revision",
                "rating-reliability-cli-test",
                "--created-at-utc",
                "2026-09-01T17:00:00Z",
            ]
        )
        == 1
    )
    error = json.loads(capsys.readouterr().err)
    assert error["command"] == "public-rating-reliability"
    assert error["status"] == "error"
    assert not output.exists()


def test_external_diagnostics_fails_closed_for_missing_sources(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    missing = tmp_path / "missing.json"
    output = tmp_path / "diagnostics"

    assert (
        main(
            [
                "external-diagnostics",
                "--scoring-summary",
                str(missing),
                "--primary-report",
                str(missing),
                "--decision-seal-report",
                str(missing),
                "--dataset-root",
                str(tmp_path / "datasets"),
                "--pixi-lock",
                str(missing),
                "--output-root",
                str(output),
                "--code-revision",
                "external-diagnostics-cli-test",
                "--created-at-utc",
                "2026-09-01T16:03:00Z",
            ]
        )
        == 1
    )
    error = json.loads(capsys.readouterr().err)
    assert error["command"] == "external-diagnostics"
    assert error["status"] == "error"
    assert not output.exists()


def test_failure_review_prepare_fails_closed_for_missing_sources(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    missing = tmp_path / "missing.json"
    output = tmp_path / "failure-review"

    assert (
        main(
            [
                "failure-review-prepare",
                "--taxonomy",
                str(missing),
                "--analysis-specification",
                str(missing),
                "--manifest",
                str(missing),
                "--primary-report",
                str(missing),
                "--dataset-root",
                str(tmp_path / "datasets"),
                "--generation-responses",
                str(missing),
                "--criterion-responses",
                str(missing),
                "--pixi-lock",
                str(missing),
                "--output-root",
                str(output),
                "--code-revision",
                "failure-review-cli-test",
                "--created-at-utc",
                "2026-09-01T17:00:00Z",
            ]
        )
        == 1
    )
    error = json.loads(capsys.readouterr().err)
    assert error["command"] == "failure-review-prepare"
    assert error["status"] == "error"
    assert not output.exists()


def test_failure_review_record_fails_closed_for_missing_sources(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    missing = tmp_path / "missing.json"
    output = tmp_path / "record.json"

    assert (
        main(
            [
                "failure-review-record",
                "--taxonomy",
                str(missing),
                "--packet",
                str(missing),
                "--completed-sheet",
                str(missing),
                "--rater-id",
                "reviewer-a",
                "--output",
                str(output),
                "--recorded-at-utc",
                "2026-09-01T18:00:00Z",
            ]
        )
        == 1
    )
    error = json.loads(capsys.readouterr().err)
    assert error["command"] == "failure-review-record"
    assert error["status"] == "error"
    assert not output.exists()


def test_probe_correction_sensitivity_fails_closed_for_missing_sources(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    missing = tmp_path / "missing.json"
    output = tmp_path / "probe-correction"

    assert (
        main(
            [
                "probe-correction-sensitivity",
                "--specification",
                str(missing),
                "--primary-report",
                str(missing),
                "--failure-review-report",
                str(missing),
                "--evidence-execution-root",
                str(tmp_path / "evidence"),
                "--pixi-lock",
                str(missing),
                "--output-root",
                str(output),
                "--code-revision",
                "probe-correction-cli-test",
                "--created-at-utc",
                "2026-09-01T19:00:00Z",
            ]
        )
        == 1
    )
    error = json.loads(capsys.readouterr().err)
    assert error["command"] == "probe-correction-sensitivity"
    assert error["status"] == "error"
    assert not output.exists()


def test_external_resource_reconciliation_fails_closed_for_missing_sources(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    missing = tmp_path / "missing.json"
    output = tmp_path / "resource-reconciliation"

    assert (
        main(
            [
                "external-resource-reconcile",
                "--generation-report",
                str(missing),
                "--criterion-report",
                str(missing),
                "--decision-seal-report",
                str(missing),
                "--primary-report",
                str(missing),
                "--correction-report",
                str(missing),
                "--decision-responses",
                str(missing),
                "--criterion-responses",
                str(missing),
                "--evidence-execution-root",
                str(tmp_path / "evidence"),
                "--measured-artifact-root",
                str(tmp_path / "artifacts"),
                "--pixi-lock",
                str(missing),
                "--output-root",
                str(output),
                "--code-revision",
                "resource-reconciliation-cli-test",
                "--created-at-utc",
                "2026-09-01T20:00:00Z",
            ]
        )
        == 1
    )
    error = json.loads(capsys.readouterr().err)
    assert error["command"] == "external-resource-reconcile"
    assert error["status"] == "error"
    assert not output.exists()


def _run_generate(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/benchmark_generate.py", *arguments],
        cwd=WORKSPACE_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
