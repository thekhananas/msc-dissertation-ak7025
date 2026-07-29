"""Record the pre-held-out scoring choice when calibration data are unavailable."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from socratic_tutor.benchmark.analysis_spec import (
    AnalysisSpecification,
    analysis_specification_hash,
    load_analysis_specification,
)
from socratic_tutor.benchmark.artifacts import write_immutable_json
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.evaluator.loader import load_and_verify_manifest
from socratic_tutor.benchmark.evaluator.models import ManifestStatus
from socratic_tutor.benchmark.evaluator.scoring import (
    CalibrationDecision,
    CalibrationStatus,
    PrimaryMetric,
    create_calibration_decision,
)
from socratic_tutor.benchmark.hashing import canonical_sha256, model_content_hash
from socratic_tutor.benchmark.public import BenchmarkSplit
from socratic_tutor.contracts import ContractModel


class UncalibratedDecisionPlan(ContractModel):
    """A deliberate fallback, valid only when independent calibration data are absent."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.uncalibrated_decision_plan.v1"] = (
        "benchmark.uncalibrated_decision_plan.v1"
    )
    benchmark_version: str = Field(min_length=1)
    tracker_version: str = Field(min_length=1)
    policy_threshold: float = Field(ge=0.0, le=1.0)
    decision_owner: str = Field(min_length=1)
    decided_at_utc: datetime
    reason: Literal["no_independent_calibration_cases"]
    expected_calibration_case_count: Literal[0] = 0

    @model_validator(mode="after")
    def require_utc_time(self) -> UncalibratedDecisionPlan:
        if self.decided_at_utc.tzinfo is None or self.decided_at_utc.utcoffset() != UTC.utcoffset(
            self.decided_at_utc
        ):
            raise ValueError("Decision time must be UTC-aware")
        return self


class CalibrationManifestInventory(ContractModel):
    """One verified authored manifest inspected before the scoring decision."""

    manifest_ref: str = Field(min_length=1)
    benchmark_version: str = Field(min_length=1)
    manifest_hash: Sha256
    calibration_case_count: int = Field(ge=0)
    calibration_task_families: tuple[str, ...]


class UncalibratedDecisionReport(ContractModel):
    """Immutable evidence for selecting classification error over a calibration loss."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.uncalibrated_decision_report.v1"] = (
        "benchmark.uncalibrated_decision_report.v1"
    )
    benchmark_version: str = Field(min_length=1)
    heldout_manifest_hash: Sha256
    analysis_specification_hash: Sha256
    source_manifests: tuple[CalibrationManifestInventory, ...] = Field(min_length=1)
    calibration_case_count: Literal[0] = 0
    reason: Literal["no_independent_calibration_cases"]
    decision: CalibrationDecision
    report_hash: Sha256

    @model_validator(mode="after")
    def validate_report(self) -> UncalibratedDecisionReport:
        if self.decision.status is not CalibrationStatus.UNCALIBRATED_SCORE:
            raise ValueError("Uncalibrated reports must select the uncalibrated score")
        if self.decision.primary_metric is not PrimaryMetric.PAIRED_CLASSIFICATION_ERROR_DIFFERENCE:
            raise ValueError("Uncalibrated reports must select classification error")
        if self.calibration_case_count != sum(
            item.calibration_case_count for item in self.source_manifests
        ):
            raise ValueError("Calibration case count does not match the source inventory")
        if self.report_hash != model_content_hash(self, exclude={"report_hash"}):
            raise ValueError("Uncalibrated decision report hash does not match its content")
        return self


def load_uncalibrated_decision_plan(path: Path) -> UncalibratedDecisionPlan:
    """Load a decision plan from a reviewed YAML file."""

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"Could not read uncalibrated decision plan: {path}") from error
    return UncalibratedDecisionPlan.model_validate(raw)


def record_uncalibrated_decision(
    *,
    plan: UncalibratedDecisionPlan,
    heldout_manifest_path: Path,
    benchmark_root: Path,
    analysis_specification_path: Path,
    output_path: Path,
) -> UncalibratedDecisionReport:
    """Inspect all authored manifests and record the permitted fallback metric.

    This is intentionally fail-closed. A future calibration corpus makes this plan
    invalid; it cannot silently continue with the weaker classification metric.
    """

    heldout = load_and_verify_manifest(heldout_manifest_path)
    if heldout.status is not ManifestStatus.FROZEN:
        raise ValueError("An uncalibrated decision requires a frozen held-out manifest")
    if heldout.benchmark_version != plan.benchmark_version:
        raise ValueError("Decision plan belongs to another benchmark version")

    specification = load_analysis_specification(analysis_specification_path)
    _validate_analysis_specification(specification, plan)

    root = benchmark_root.resolve()
    inventory = _manifest_inventory(root)
    calibration_case_count = sum(item.calibration_case_count for item in inventory)
    if calibration_case_count != plan.expected_calibration_case_count:
        raise ValueError(
            "Uncalibrated fallback requires exactly "
            f"{plan.expected_calibration_case_count} calibration cases, "
            f"found {calibration_case_count}"
        )

    task_families = tuple(family for item in inventory for family in item.calibration_task_families)
    calibration_task_family_hash = canonical_sha256(
        {"schema_id": "benchmark.calibration_task_families.v1", "task_families": task_families}
    )
    calibration_data_hash = canonical_sha256(
        {
            "schema_id": "benchmark.calibration_inventory.v1",
            "source_manifests": [item.model_dump(mode="json") for item in inventory],
        }
    )
    specification_hash = analysis_specification_hash(specification)
    diagnostics_hash = canonical_sha256(
        {
            "schema_id": "benchmark.uncalibrated_diagnostics.v1",
            "analysis_specification_hash": specification_hash,
            "calibration_case_count": calibration_case_count,
            "reason": plan.reason,
        }
    )
    decision = create_calibration_decision(
        status=CalibrationStatus.UNCALIBRATED_SCORE,
        policy_threshold=plan.policy_threshold,
        calibration_task_family_hash=calibration_task_family_hash,
        calibration_data_hash=calibration_data_hash,
        tracker_version=plan.tracker_version,
        diagnostics_hash=diagnostics_hash,
        decision_owner=plan.decision_owner,
        decided_at_utc=plan.decided_at_utc,
    )
    content: dict[str, object] = {
        "schema_version": 1,
        "schema_id": "benchmark.uncalibrated_decision_report.v1",
        "benchmark_version": plan.benchmark_version,
        "heldout_manifest_hash": model_content_hash(heldout),
        "analysis_specification_hash": specification_hash,
        "source_manifests": inventory,
        "calibration_case_count": calibration_case_count,
        "reason": plan.reason,
        "decision": decision,
    }
    draft = UncalibratedDecisionReport.model_construct(
        _fields_set=set(content), **content, report_hash="0" * 64
    )
    report = UncalibratedDecisionReport.model_validate(
        {**content, "report_hash": model_content_hash(draft, exclude={"report_hash"})}
    )
    write_immutable_json(output_path, report)
    return report


def _validate_analysis_specification(
    specification: AnalysisSpecification,
    plan: UncalibratedDecisionPlan,
) -> None:
    if specification.benchmark_version != plan.benchmark_version:
        raise ValueError("Analysis specification belongs to another benchmark version")
    if specification.uncalibrated_fallback_metric != "paired_classification_error_difference":
        raise ValueError("Analysis specification has no supported uncalibrated fallback metric")
    if specification.policy_threshold != plan.policy_threshold:
        raise ValueError("Decision plan threshold differs from the analysis specification")


def _manifest_inventory(benchmark_root: Path) -> tuple[CalibrationManifestInventory, ...]:
    paths = tuple(sorted(benchmark_root.rglob("manifest.yaml")))
    if not paths:
        raise ValueError(f"No authored manifests found under {benchmark_root}")
    inventory: list[CalibrationManifestInventory] = []
    for path in paths:
        manifest = load_and_verify_manifest(path)
        families = tuple(
            case.public.task_family
            for case in manifest.cases
            if case.public.split is BenchmarkSplit.CALIBRATION
        )
        inventory.append(
            CalibrationManifestInventory(
                manifest_ref=path.relative_to(benchmark_root).as_posix(),
                benchmark_version=manifest.benchmark_version,
                manifest_hash=model_content_hash(manifest),
                calibration_case_count=len(families),
                calibration_task_families=families,
            )
        )
    return tuple(inventory)
