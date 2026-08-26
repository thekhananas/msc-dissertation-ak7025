"""Runtime and memory forecast for the compact acquisition-study stream."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path
from statistics import median
from typing import Literal, Self

from pydantic import Field, ValidationError, model_validator

from socratic_tutor.acquisition_study.compact_stream import (
    DevelopmentCompactParityPlan,
    DevelopmentCompactParityReport,
)
from socratic_tutor.acquisition_study.plan import (
    AcquisitionAnalysisSpecification,
    AcquisitionEnvironmentSpecification,
)
from socratic_tutor.acquisition_study.runtime_forecast import AcquisitionRuntimePlatform
from socratic_tutor.acquisition_study.runtime_metrics import (
    NANOSECONDS_PER_SECOND,
)
from socratic_tutor.acquisition_study.runtime_metrics import (
    runtime_seconds as runtime_seconds,
)
from socratic_tutor.benchmark.artifacts import write_immutable_json
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import canonical_sha256, file_sha256, model_content_hash
from socratic_tutor.contracts import ContractModel

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class CompactRuntimeForecastError(ValueError):
    """A compact-stream measurement cannot support the feasibility decision."""


class CompactRuntimeSample(ContractModel):
    """One fresh-process measurement of the complete development stream."""

    repetition_index: int = Field(ge=0)
    episodes_per_environment: int = Field(ge=1)
    environment_count: Literal[7] = 7
    policy_count: Literal[7] = 7
    episode_count: int = Field(ge=1)
    row_count: int = Field(ge=1)
    case_prediction_count: int = Field(ge=1)
    output_bytes: int = Field(ge=1)
    elapsed_nanoseconds: int = Field(ge=1)
    baseline_peak_rss_bytes: int = Field(ge=1)
    observed_peak_rss_bytes: int = Field(ge=1)
    incremental_peak_rss_bytes: int = Field(ge=0)
    stream_file_sha256: Sha256

    @model_validator(mode="after")
    def validate_measurement(self) -> Self:
        if self.episode_count != self.environment_count * self.episodes_per_environment:
            raise ValueError("Compact runtime episode count does not reconcile")
        if self.row_count != self.episode_count * self.policy_count:
            raise ValueError("Compact runtime row count does not reconcile")
        if self.case_prediction_count != self.row_count * 40:
            raise ValueError("Compact runtime prediction count does not reconcile")
        if self.observed_peak_rss_bytes < self.baseline_peak_rss_bytes:
            raise ValueError("Observed peak memory cannot be below its baseline")
        if self.incremental_peak_rss_bytes != (
            self.observed_peak_rss_bytes - self.baseline_peak_rss_bytes
        ):
            raise ValueError("Incremental peak memory does not reconcile")
        return self


class DevelopmentCompactRuntimeForecastPlan(ContractModel):
    """Frozen lineage and planning rules for the compact-stream forecast."""

    schema_version: Literal[1] = 1
    schema_id: Literal["acquisition_study.development_compact_runtime_plan.v1"] = (
        "acquisition_study.development_compact_runtime_plan.v1"
    )
    study_id: Literal["reliability-aware-probing-v1"] = "reliability-aware-probing-v1"
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,99}$")
    split: Literal["development"] = "development"
    compact_parity_report_hash: Sha256
    compact_parity_report_file_sha256: Sha256
    compact_parity_plan_hash: Sha256
    compact_stream_file_sha256: Sha256
    compact_stream_file_bytes: int = Field(ge=1)
    source_matrix_content_hash: Sha256
    environment_specification_hash: Sha256
    frozen_analysis_specification_hash: Sha256
    compact_code_revision: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    runtime_code_revision: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    pixi_lock_sha256: Sha256
    measurement_repetitions: Literal[1, 3]
    fresh_process_per_repetition: Literal[True] = True
    development_episodes_per_environment: int = Field(ge=1)
    canonical_episodes_per_environment: int = Field(ge=1)
    canonical_workload_scale_factor: Literal[40] = 40
    runtime_safety_multiplier: Literal[2] = 2
    peak_memory_safety_multiplier: Literal[2] = 2
    remaining_m7b_seconds: Literal[57600] = 57600
    planning_physical_memory_divisor: Literal[2] = 2
    time_projection_method: Literal["median_development_time_times_episode_scale"] = (
        "median_development_time_times_episode_scale"
    )
    memory_projection_method: Literal[
        "maximum_full_development_stream_peak_times_safety_multiplier"
    ] = "maximum_full_development_stream_peak_times_safety_multiplier"
    memory_projection_assumption: Literal[
        "one_episode_is_retained_and_full_development_shows_no_cross_episode_collection"
    ] = "one_episode_is_retained_and_full_development_shows_no_cross_episode_collection"
    result_scope: Literal["operational_forecast_not_scientific_result"] = (
        "operational_forecast_not_scientific_result"
    )
    canonical_claim_allowed: Literal[False] = False
    plan_hash: Sha256

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        if (
            self.canonical_episodes_per_environment
            != self.development_episodes_per_environment * self.canonical_workload_scale_factor
        ):
            raise ValueError("Compact runtime episode scale does not reconcile")
        if self.plan_hash != model_content_hash(self, exclude={"plan_hash"}):
            raise ValueError("Compact runtime plan hash does not match its content")
        return self


class DevelopmentCompactRuntimeForecastReport(ContractModel):
    """Measured compact cost and cautious canonical planning estimate."""

    schema_version: Literal[1] = 1
    schema_id: Literal["acquisition_study.development_compact_runtime_report.v1"] = (
        "acquisition_study.development_compact_runtime_report.v1"
    )
    study_id: Literal["reliability-aware-probing-v1"] = "reliability-aware-probing-v1"
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,99}$")
    split: Literal["development"] = "development"
    runtime_plan_hash: Sha256
    compact_parity_report_hash: Sha256
    compact_stream_file_sha256: Sha256
    platform: AcquisitionRuntimePlatform
    measurement_count: int = Field(ge=1)
    measurements: tuple[CompactRuntimeSample, ...] = Field(min_length=1)
    exact_stream_parity_verified: Literal[True] = True
    development_episodes_per_environment: int = Field(ge=1)
    canonical_episodes_per_environment: int = Field(ge=1)
    canonical_workload_scale_factor: Literal[40] = 40
    runtime_safety_multiplier: Literal[2] = 2
    peak_memory_safety_multiplier: Literal[2] = 2
    remaining_m7b_seconds: Literal[57600] = 57600
    planning_physical_memory_divisor: Literal[2] = 2
    median_development_elapsed_nanoseconds: int = Field(ge=1)
    maximum_development_peak_rss_bytes: int = Field(ge=1)
    maximum_incremental_peak_rss_bytes: int = Field(ge=0)
    linear_projected_canonical_elapsed_nanoseconds: int = Field(ge=1)
    guarded_projected_canonical_elapsed_nanoseconds: int = Field(ge=1)
    guarded_planning_canonical_peak_rss_bytes: int = Field(ge=1)
    planning_memory_limit_bytes: int = Field(ge=1)
    runtime_within_remaining_m7b_time: bool
    memory_within_planning_limit: bool
    compact_runner_feasible_for_canonical_run: bool
    canonical_peak_memory_measured: Literal[False] = False
    projection_limit: Literal["planning_estimate_not_a_guarantee_and_not_a_scientific_outcome"] = (
        "planning_estimate_not_a_guarantee_and_not_a_scientific_outcome"
    )
    result_status: Literal["development_compact_operational_forecast"] = (
        "development_compact_operational_forecast"
    )
    external_model_call_count: Literal[0] = 0
    sandbox_call_count: Literal[0] = 0
    human_record_count: Literal[0] = 0
    canonical_claim_allowed: Literal[False] = False
    report_hash: Sha256

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        if self.measurement_count != len(self.measurements):
            raise ValueError("Compact runtime measurement count does not reconcile")
        if tuple(row.repetition_index for row in self.measurements) != tuple(
            range(self.measurement_count)
        ):
            raise ValueError("Compact runtime repetitions are incomplete or out of order")
        if any(
            row.episodes_per_environment != self.development_episodes_per_environment
            for row in self.measurements
        ):
            raise ValueError("Compact runtime measurements use the wrong development size")
        if any(
            row.stream_file_sha256 != self.compact_stream_file_sha256 for row in self.measurements
        ):
            raise ValueError("A measured compact stream differs from the parity artifact")
        if (
            self.canonical_episodes_per_environment
            != self.development_episodes_per_environment * self.canonical_workload_scale_factor
        ):
            raise ValueError("Compact runtime report episode scale does not reconcile")
        expected_median = int(median(row.elapsed_nanoseconds for row in self.measurements))
        if self.median_development_elapsed_nanoseconds != expected_median:
            raise ValueError("Compact median runtime does not reconcile")
        expected_peak = max(row.observed_peak_rss_bytes for row in self.measurements)
        if self.maximum_development_peak_rss_bytes != expected_peak:
            raise ValueError("Compact maximum memory does not reconcile")
        expected_increment = max(row.incremental_peak_rss_bytes for row in self.measurements)
        if self.maximum_incremental_peak_rss_bytes != expected_increment:
            raise ValueError("Compact incremental memory does not reconcile")
        linear_time = expected_median * self.canonical_workload_scale_factor
        if self.linear_projected_canonical_elapsed_nanoseconds != linear_time:
            raise ValueError("Compact linear runtime projection does not reconcile")
        if self.guarded_projected_canonical_elapsed_nanoseconds != (
            linear_time * self.runtime_safety_multiplier
        ):
            raise ValueError("Compact guarded runtime projection does not reconcile")
        guarded_memory = expected_peak * self.peak_memory_safety_multiplier
        if self.guarded_planning_canonical_peak_rss_bytes != guarded_memory:
            raise ValueError("Compact guarded memory estimate does not reconcile")
        memory_limit = self.platform.physical_memory_bytes // self.planning_physical_memory_divisor
        if self.planning_memory_limit_bytes != memory_limit:
            raise ValueError("Compact planning memory limit does not reconcile")
        runtime_gate = (
            self.guarded_projected_canonical_elapsed_nanoseconds
            <= self.remaining_m7b_seconds * NANOSECONDS_PER_SECOND
        )
        memory_gate = self.guarded_planning_canonical_peak_rss_bytes <= memory_limit
        if self.runtime_within_remaining_m7b_time != runtime_gate:
            raise ValueError("Compact runtime gate does not reconcile")
        if self.memory_within_planning_limit != memory_gate:
            raise ValueError("Compact memory gate does not reconcile")
        if self.compact_runner_feasible_for_canonical_run != (runtime_gate and memory_gate):
            raise ValueError("Compact feasibility gate does not reconcile")
        if self.report_hash != model_content_hash(self, exclude={"report_hash"}):
            raise ValueError("Compact runtime report hash does not match its content")
        return self


def run_development_compact_runtime_forecast(
    *,
    compact_parity_report_path: Path,
    environment_path: Path,
    analysis_path: Path,
    specification: AcquisitionEnvironmentSpecification,
    analysis: AcquisitionAnalysisSpecification,
    pixi_lock_path: Path,
    output_root: Path,
    run_id: str,
    runtime_code_revision: str,
    measurement_repetitions: Literal[1, 3] = 3,
) -> DevelopmentCompactRuntimeForecastReport:
    """Measure the verified stream and forecast the configured evaluation workload."""

    parity_plan, parity_report, compact_path = _load_verified_parity(
        compact_parity_report_path,
        specification=specification,
        analysis=analysis,
    )
    development_count = specification.episodes.development_episodes_per_environment
    canonical_count = specification.episodes.evaluation_episodes_per_environment
    if parity_report.episodes_per_environment != development_count:
        raise CompactRuntimeForecastError(
            "Runtime forecast requires the complete configured development stream"
        )
    if canonical_count != development_count * 40:
        raise CompactRuntimeForecastError(
            "Compact runtime forecast expects the frozen forty-times workload"
        )
    try:
        parity_report_file_hash = file_sha256(compact_parity_report_path.read_bytes())
        pixi_lock_hash = file_sha256(pixi_lock_path.read_bytes())
    except OSError as error:
        raise CompactRuntimeForecastError("Could not hash a compact runtime source") from error

    plan_content = {
        "schema_version": 1,
        "schema_id": "acquisition_study.development_compact_runtime_plan.v1",
        "study_id": parity_report.study_id,
        "run_id": run_id,
        "split": parity_report.split,
        "compact_parity_report_hash": parity_report.report_hash,
        "compact_parity_report_file_sha256": parity_report_file_hash,
        "compact_parity_plan_hash": parity_plan.plan_hash,
        "compact_stream_file_sha256": parity_report.compact_file_sha256,
        "compact_stream_file_bytes": parity_report.compact_file_bytes,
        "source_matrix_content_hash": parity_plan.source_matrix_content_hash,
        "environment_specification_hash": specification.specification_hash,
        "frozen_analysis_specification_hash": analysis.plan_hash,
        "compact_code_revision": parity_plan.compact_code_revision,
        "runtime_code_revision": runtime_code_revision,
        "pixi_lock_sha256": pixi_lock_hash,
        "measurement_repetitions": measurement_repetitions,
        "fresh_process_per_repetition": True,
        "development_episodes_per_environment": development_count,
        "canonical_episodes_per_environment": canonical_count,
        "canonical_workload_scale_factor": 40,
        "runtime_safety_multiplier": 2,
        "peak_memory_safety_multiplier": 2,
        "remaining_m7b_seconds": 57600,
        "planning_physical_memory_divisor": 2,
        "time_projection_method": "median_development_time_times_episode_scale",
        "memory_projection_method": (
            "maximum_full_development_stream_peak_times_safety_multiplier"
        ),
        "memory_projection_assumption": (
            "one_episode_is_retained_and_full_development_shows_no_cross_episode_collection"
        ),
        "result_scope": "operational_forecast_not_scientific_result",
        "canonical_claim_allowed": False,
    }
    plan = DevelopmentCompactRuntimeForecastPlan.model_validate(
        {**plan_content, "plan_hash": canonical_sha256(plan_content)}
    )
    samples = tuple(
        _run_fresh_process_measurement(
            repetition_index=index,
            environment_path=environment_path,
            analysis_path=analysis_path,
            episodes_per_environment=development_count,
        )
        for index in range(measurement_repetitions)
    )
    if any(row.stream_file_sha256 != parity_report.compact_file_sha256 for row in samples):
        raise CompactRuntimeForecastError(
            "A measured compact stream differs from the exact-parity artifact"
        )
    if any(row.output_bytes != compact_path.stat().st_size for row in samples):
        raise CompactRuntimeForecastError("A measured compact stream has a different size")

    median_elapsed = int(median(row.elapsed_nanoseconds for row in samples))
    maximum_observed = max(row.observed_peak_rss_bytes for row in samples)
    maximum_increment = max(row.incremental_peak_rss_bytes for row in samples)
    linear_time = median_elapsed * plan.canonical_workload_scale_factor
    guarded_time = linear_time * plan.runtime_safety_multiplier
    guarded_memory = maximum_observed * plan.peak_memory_safety_multiplier
    runtime_platform = _platform()
    planning_memory_limit = (
        runtime_platform.physical_memory_bytes // plan.planning_physical_memory_divisor
    )
    runtime_within_gate = guarded_time <= plan.remaining_m7b_seconds * NANOSECONDS_PER_SECOND
    memory_within_limit = guarded_memory <= planning_memory_limit
    report_content = {
        "schema_version": 1,
        "schema_id": "acquisition_study.development_compact_runtime_report.v1",
        "study_id": parity_report.study_id,
        "run_id": run_id,
        "split": parity_report.split,
        "runtime_plan_hash": plan.plan_hash,
        "compact_parity_report_hash": parity_report.report_hash,
        "compact_stream_file_sha256": parity_report.compact_file_sha256,
        "platform": runtime_platform,
        "measurement_count": len(samples),
        "measurements": samples,
        "exact_stream_parity_verified": True,
        "development_episodes_per_environment": development_count,
        "canonical_episodes_per_environment": canonical_count,
        "canonical_workload_scale_factor": plan.canonical_workload_scale_factor,
        "runtime_safety_multiplier": plan.runtime_safety_multiplier,
        "peak_memory_safety_multiplier": plan.peak_memory_safety_multiplier,
        "remaining_m7b_seconds": plan.remaining_m7b_seconds,
        "planning_physical_memory_divisor": plan.planning_physical_memory_divisor,
        "median_development_elapsed_nanoseconds": median_elapsed,
        "maximum_development_peak_rss_bytes": maximum_observed,
        "maximum_incremental_peak_rss_bytes": maximum_increment,
        "linear_projected_canonical_elapsed_nanoseconds": linear_time,
        "guarded_projected_canonical_elapsed_nanoseconds": guarded_time,
        "guarded_planning_canonical_peak_rss_bytes": guarded_memory,
        "planning_memory_limit_bytes": planning_memory_limit,
        "runtime_within_remaining_m7b_time": runtime_within_gate,
        "memory_within_planning_limit": memory_within_limit,
        "compact_runner_feasible_for_canonical_run": (runtime_within_gate and memory_within_limit),
        "canonical_peak_memory_measured": False,
        "projection_limit": ("planning_estimate_not_a_guarantee_and_not_a_scientific_outcome"),
        "result_status": "development_compact_operational_forecast",
        "external_model_call_count": 0,
        "sandbox_call_count": 0,
        "human_record_count": 0,
        "canonical_claim_allowed": False,
    }
    report = DevelopmentCompactRuntimeForecastReport.model_validate(
        {**report_content, "report_hash": canonical_sha256(report_content)}
    )
    write_immutable_json(output_root / "development_compact_runtime_plan.json", plan)
    write_immutable_json(output_root / "development_compact_runtime_report.json", report)
    return report


def _load_verified_parity(
    report_path: Path,
    *,
    specification: AcquisitionEnvironmentSpecification,
    analysis: AcquisitionAnalysisSpecification,
) -> tuple[DevelopmentCompactParityPlan, DevelopmentCompactParityReport, Path]:
    plan_path = report_path.parent / "development_compact_parity_plan.json"
    try:
        report = DevelopmentCompactParityReport.model_validate_json(
            report_path.read_text(encoding="utf-8")
        )
        plan = DevelopmentCompactParityPlan.model_validate_json(
            plan_path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError) as error:
        raise CompactRuntimeForecastError("Could not load verified compact parity") from error
    if report.parity_plan_hash != plan.plan_hash:
        raise CompactRuntimeForecastError("Compact parity report and plan do not match")
    if plan.environment_specification_hash != specification.specification_hash:
        raise CompactRuntimeForecastError("Compact parity uses another environment specification")
    if plan.frozen_analysis_specification_hash != analysis.plan_hash:
        raise CompactRuntimeForecastError("Compact parity uses another analysis specification")
    compact_path = report_path.parent / report.compact_file
    try:
        compact_content = compact_path.read_bytes()
    except OSError as error:
        raise CompactRuntimeForecastError("Could not read compact parity stream") from error
    if file_sha256(compact_content) != report.compact_file_sha256:
        raise CompactRuntimeForecastError("Compact parity stream hash does not match")
    if len(compact_content) != report.compact_file_bytes:
        raise CompactRuntimeForecastError("Compact parity stream size does not match")
    return plan, report, compact_path


def _run_fresh_process_measurement(
    *,
    repetition_index: int,
    environment_path: Path,
    analysis_path: Path,
    episodes_per_environment: int,
) -> CompactRuntimeSample:
    command = [
        sys.executable,
        "-m",
        "socratic_tutor.acquisition_study.compact_runtime_worker",
        "--environments",
        str(environment_path.resolve()),
        "--analysis",
        str(analysis_path.resolve()),
        "--episodes-per-environment",
        str(episodes_per_environment),
        "--repetition-index",
        str(repetition_index),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise CompactRuntimeForecastError(
            f"Compact runtime measurement {repetition_index} could not complete"
        ) from error
    if completed.returncode != 0:
        message = completed.stderr.strip() or "worker returned no error detail"
        raise CompactRuntimeForecastError(
            f"Compact runtime measurement {repetition_index} failed: {message}"
        )
    try:
        return CompactRuntimeSample.model_validate_json(completed.stdout)
    except ValidationError as error:
        raise CompactRuntimeForecastError(
            f"Compact runtime measurement {repetition_index} returned invalid data"
        ) from error


def _platform() -> AcquisitionRuntimePlatform:
    cpu_count = os.cpu_count()
    if cpu_count is None or cpu_count < 1:
        raise CompactRuntimeForecastError("Could not determine logical CPU count")
    try:
        physical_memory = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError) as error:
        raise CompactRuntimeForecastError("Could not determine physical memory") from error
    return AcquisitionRuntimePlatform(
        operating_system=platform.platform(),
        machine=platform.machine(),
        processor=platform.processor(),
        python_implementation=platform.python_implementation(),
        python_version=platform.python_version(),
        logical_cpu_count=cpu_count,
        physical_memory_bytes=physical_memory,
    )
