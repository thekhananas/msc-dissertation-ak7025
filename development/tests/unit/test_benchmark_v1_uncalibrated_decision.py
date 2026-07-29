"""Tests for the explicit uncalibrated scoring decision used by v1."""

from pathlib import Path

import pytest
import yaml

from socratic_tutor.benchmark.calibration import (
    UncalibratedDecisionPlan,
    load_uncalibrated_decision_plan,
    record_uncalibrated_decision,
)
from socratic_tutor.benchmark.evaluator.scoring import (
    CalibrationStatus,
    PrimaryMetric,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = WORKSPACE_ROOT / "data" / "benchmarks"
V1_MANIFEST = BENCHMARK_ROOT / "v1" / "manifest.yaml"
ANALYSIS_SPEC = WORKSPACE_ROOT / "configs" / "benchmark" / "v1-analysis-spec.yaml"
PLAN_PATH = WORKSPACE_ROOT / "configs" / "benchmark" / "v1-uncalibrated-decision.yaml"


def test_v1_records_the_predeclared_uncalibrated_metric(tmp_path: Path) -> None:
    report = record_uncalibrated_decision(
        plan=load_uncalibrated_decision_plan(PLAN_PATH),
        heldout_manifest_path=V1_MANIFEST,
        benchmark_root=BENCHMARK_ROOT,
        analysis_specification_path=ANALYSIS_SPEC,
        output_path=tmp_path / "decision.json",
    )

    assert report.calibration_case_count == 0
    assert len(report.source_manifests) == 2
    assert report.decision.status is CalibrationStatus.UNCALIBRATED_SCORE
    assert report.decision.primary_metric is PrimaryMetric.PAIRED_CLASSIFICATION_ERROR_DIFFERENCE
    assert len(report.report_hash) == 64


def test_uncalibrated_plan_rejects_an_unexpected_calibration_corpus() -> None:
    raw = yaml.safe_load(PLAN_PATH.read_text(encoding="utf-8"))
    raw["expected_calibration_case_count"] = 1

    with pytest.raises(ValueError):
        UncalibratedDecisionPlan.model_validate(raw)


def test_uncalibrated_plan_rejects_a_threshold_change_after_freeze(tmp_path: Path) -> None:
    raw = yaml.safe_load(PLAN_PATH.read_text(encoding="utf-8"))
    raw["policy_threshold"] = 0.70
    changed_plan = tmp_path / "changed-plan.yaml"
    changed_plan.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="threshold differs"):
        record_uncalibrated_decision(
            plan=load_uncalibrated_decision_plan(changed_plan),
            heldout_manifest_path=V1_MANIFEST,
            benchmark_root=BENCHMARK_ROOT,
            analysis_specification_path=ANALYSIS_SPEC,
            output_path=tmp_path / "never-written.json",
        )
