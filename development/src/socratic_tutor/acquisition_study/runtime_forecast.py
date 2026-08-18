"""Development runtime measurement and canonical-workload forecast."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path
from statistics import median
from typing import Literal, Self

from pydantic import Field, ValidationError, model_validator

from socratic_tutor.acquisition_study.plan import (
    AcquisitionAnalysisSpecification,
    AcquisitionEnvironmentSpecification,
)
from socratic_tutor.acquisition_study.primary_analysis import (
    load_verified_development_policy_matrix,
)
from socratic_tutor.benchmark.artifacts import write_immutable_json
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import canonical_sha256, file_sha256, model_content_hash
from socratic_tutor.contracts import ContractModel

PROJECT_ROOT = Path(__file__).resolve().parents[3]
_NANOSECONDS_PER_SECOND = 1_000_000_000


class AcquisitionRuntimeForecastError(ValueError):
    """A runtime measurement cannot support the declared feasibility check."""


class AcquisitionRuntimePlatform(ContractModel):
    """Machine details needed to interpret a local performance measurement."""

    operating_system: str = Field(min_length=1)
    machine: str = Field(min_length=1)
    processor: str
    python_implementation: str = Field(min_length=1)
    python_version: str = Field(min_length=1)
    logical_cpu_count: int = Field(ge=1)
    physical_memory_bytes: int = Field(ge=1)


class AcquisitionRuntimeSample(ContractModel):
    """One fresh-process measurement of the complete development matrix."""

    repetition_index: int = Field(ge=0)
    episodes_per_environment: int = Field(ge=1)
    comparison_count: int = Field(ge=1)
    policy_result_count: int = Field(ge=1)
    case_prediction_count: int = Field(ge=1)
    elapsed_nanoseconds: int = Field(ge=1)
    baseline_peak_rss_bytes: int = Field(ge=1)
    observed_peak_rss_bytes: int = Field(ge=1)
    incremental_peak_rss_bytes: int = Field(ge=0)
    matrix_content_hash: Sha256

    @model_validator(mode="after")
    def validate_measurement(self) -> Self:
        if self.observed_peak_rss_bytes < self.baseline_peak_rss_bytes:
            raise ValueError("Observed peak memory cannot be below its baseline")
        if self.incremental_peak_rss_bytes != (
            self.observed_peak_rss_bytes - self.baseline_peak_rss_bytes
        ):
            raise ValueError("Incremental peak memory does not reconcile")
        return self


class DevelopmentRuntimeForecastPlan(ContractModel):
    """Provenance and fixed rules for the operational forecast."""

    schema_version: Literal[1] = 1
    schema_id: Literal["acquisition_study.development_runtime_forecast_plan.v1"] = (
        "acquisition_study.development_runtime_forecast_plan.v1"
    )
    study_id: Literal["reliability-aware-probing-v1"] = "reliability-aware-probing-v1"
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,99}$")
    split: Literal["development"] = "development"
    source_run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,99}$")
    source_manifest_hash: Sha256
    source_manifest_file_sha256: Sha256
    source_matrix_content_hash: Sha256
    source_code_revision: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    environment_specification_hash: Sha256
    frozen_analysis_specification_hash: Sha256
    runtime_code_revision: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    pixi_lock_sha256: Sha256
    measurement_repetitions: Literal[1, 3]
    fresh_process_per_repetition: Literal[True] = True
    timing_clock: Literal["time.perf_counter_ns"] = "time.perf_counter_ns"
    peak_memory_source: Literal["resource.getrusage_rusage_self"] = "resource.getrusage_rusage_self"
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
        "maximum_baseline_plus_maximum_increment_times_episode_scale"
    ] = "maximum_baseline_plus_maximum_increment_times_episode_scale"
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
            raise ValueError("Canonical episode scale does not reconcile")
        if self.plan_hash != model_content_hash(self, exclude={"plan_hash"}):
            raise ValueError("Runtime forecast plan hash does not match its content")
        return self


class DevelopmentRuntimeForecastReport(ContractModel):
    """Measured development cost and cautious linear canonical forecast."""

    schema_version: Literal[1] = 1
    schema_id: Literal["acquisition_study.development_runtime_forecast_report.v1"] = (
        "acquisition_study.development_runtime_forecast_report.v1"
    )
    study_id: Literal["reliability-aware-probing-v1"] = "reliability-aware-probing-v1"
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,99}$")
    split: Literal["development"] = "development"
    runtime_plan_hash: Sha256
    source_matrix_content_hash: Sha256
    platform: AcquisitionRuntimePlatform
    measurement_count: int = Field(ge=1)
    measurements: tuple[AcquisitionRuntimeSample, ...] = Field(min_length=1)
    source_matrix_parity_verified: Literal[True] = True
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
    linear_projected_canonical_peak_rss_bytes: int = Field(ge=1)
    guarded_projected_canonical_peak_rss_bytes: int = Field(ge=1)
    planning_memory_limit_bytes: int = Field(ge=1)
    runtime_within_remaining_m7b_time: bool
    memory_within_planning_limit: bool
    current_runner_feasible_for_canonical_run: bool
    projection_limit: Literal[
        "linear_planning_estimate_not_a_guarantee_and_not_a_scientific_outcome"
    ] = "linear_planning_estimate_not_a_guarantee_and_not_a_scientific_outcome"
    result_status: Literal["development_operational_forecast"] = "development_operational_forecast"
    external_model_call_count: Literal[0] = 0
    sandbox_call_count: Literal[0] = 0
    human_record_count: Literal[0] = 0
    canonical_claim_allowed: Literal[False] = False
    report_hash: Sha256

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        plan_count = len(self.measurements)
        if self.measurement_count != plan_count:
            raise ValueError("Runtime measurement count does not reconcile")
        if tuple(row.repetition_index for row in self.measurements) != tuple(range(plan_count)):
            raise ValueError("Runtime repetitions are incomplete or out of order")
        if len({row.matrix_content_hash for row in self.measurements}) != 1:
            raise ValueError("Runtime repetitions produced different matrices")
        if any(
            row.matrix_content_hash != self.source_matrix_content_hash for row in self.measurements
        ):
            raise ValueError("Runtime measurement differs from its source matrix")
        if any(
            row.episodes_per_environment != self.development_episodes_per_environment
            for row in self.measurements
        ):
            raise ValueError("Runtime measurements use the wrong development size")
        if (
            self.canonical_episodes_per_environment
            != self.development_episodes_per_environment * self.canonical_workload_scale_factor
        ):
            raise ValueError("Runtime report episode scale does not reconcile")
        expected_median = int(median(row.elapsed_nanoseconds for row in self.measurements))
        if self.median_development_elapsed_nanoseconds != expected_median:
            raise ValueError("Median development runtime does not reconcile")
        if self.maximum_development_peak_rss_bytes != max(
            row.observed_peak_rss_bytes for row in self.measurements
        ):
            raise ValueError("Maximum measured memory does not reconcile")
        if self.maximum_incremental_peak_rss_bytes != max(
            row.incremental_peak_rss_bytes for row in self.measurements
        ):
            raise ValueError("Incremental measured memory does not reconcile")
        expected_linear_time = expected_median * self.canonical_workload_scale_factor
        if self.linear_projected_canonical_elapsed_nanoseconds != expected_linear_time:
            raise ValueError("Linear runtime projection does not reconcile")
        if self.guarded_projected_canonical_elapsed_nanoseconds != (
            expected_linear_time * self.runtime_safety_multiplier
        ):
            raise ValueError("Guarded runtime projection does not reconcile")
        maximum_baseline = max(row.baseline_peak_rss_bytes for row in self.measurements)
        expected_linear_memory = maximum_baseline + (
            self.maximum_incremental_peak_rss_bytes * self.canonical_workload_scale_factor
        )
        expected_guarded_memory = maximum_baseline + (
            self.maximum_incremental_peak_rss_bytes
            * self.canonical_workload_scale_factor
            * self.peak_memory_safety_multiplier
        )
        if self.linear_projected_canonical_peak_rss_bytes != expected_linear_memory:
            raise ValueError("Linear memory projection does not reconcile")
        if self.guarded_projected_canonical_peak_rss_bytes != expected_guarded_memory:
            raise ValueError("Guarded memory projection does not reconcile")
        expected_memory_limit = (
            self.platform.physical_memory_bytes // self.planning_physical_memory_divisor
        )
        if self.planning_memory_limit_bytes != expected_memory_limit:
            raise ValueError("Planning memory limit does not reconcile")
        expected_runtime_gate = (
            self.guarded_projected_canonical_elapsed_nanoseconds
            <= self.remaining_m7b_seconds * _NANOSECONDS_PER_SECOND
        )
        expected_memory_gate = (
            self.guarded_projected_canonical_peak_rss_bytes <= expected_memory_limit
        )
        if self.runtime_within_remaining_m7b_time != expected_runtime_gate:
            raise ValueError("Runtime gate does not reconcile")
        if self.memory_within_planning_limit != expected_memory_gate:
            raise ValueError("Memory gate does not reconcile")
        if self.current_runner_feasible_for_canonical_run != (
            self.runtime_within_remaining_m7b_time and self.memory_within_planning_limit
        ):
            raise ValueError("Overall runtime feasibility does not reconcile")
        if self.report_hash != model_content_hash(self, exclude={"report_hash"}):
            raise ValueError("Runtime forecast report hash does not match its content")
        return self


def run_development_runtime_forecast(
    *,
    source_manifest_path: Path,
    environment_path: Path,
    analysis_path: Path,
    specification: AcquisitionEnvironmentSpecification,
    analysis: AcquisitionAnalysisSpecification,
    pixi_lock_path: Path,
    output_root: Path,
    run_id: str,
    runtime_code_revision: str,
    measurement_repetitions: Literal[1, 3] = 3,
) -> DevelopmentRuntimeForecastReport:
    """Measure the full development workload and project the frozen evaluation size."""

    source_manifest, source_matrix = load_verified_development_policy_matrix(
        source_manifest_path,
        specification=specification,
        analysis=analysis,
    )
    development_count = specification.episodes.development_episodes_per_environment
    canonical_count = specification.episodes.evaluation_episodes_per_environment
    if source_matrix.episodes_per_environment != development_count:
        raise AcquisitionRuntimeForecastError(
            "Runtime forecast requires the complete configured development matrix"
        )
    if canonical_count != development_count * 40:
        raise AcquisitionRuntimeForecastError(
            "Runtime forecast expects the frozen forty-times canonical workload"
        )
    try:
        source_file_hash = file_sha256(source_manifest_path.read_bytes())
        pixi_lock_hash = file_sha256(pixi_lock_path.read_bytes())
    except OSError as error:
        raise AcquisitionRuntimeForecastError("Could not hash a runtime source file") from error

    plan_content = {
        "schema_version": 1,
        "schema_id": "acquisition_study.development_runtime_forecast_plan.v1",
        "study_id": source_matrix.study_id,
        "run_id": run_id,
        "split": source_matrix.split,
        "source_run_id": source_manifest.run_id,
        "source_manifest_hash": source_manifest.manifest_hash,
        "source_manifest_file_sha256": source_file_hash,
        "source_matrix_content_hash": source_manifest.matrix_content_hash,
        "source_code_revision": source_manifest.code_revision,
        "environment_specification_hash": specification.specification_hash,
        "frozen_analysis_specification_hash": analysis.plan_hash,
        "runtime_code_revision": runtime_code_revision,
        "pixi_lock_sha256": pixi_lock_hash,
        "measurement_repetitions": measurement_repetitions,
        "fresh_process_per_repetition": True,
        "timing_clock": "time.perf_counter_ns",
        "peak_memory_source": "resource.getrusage_rusage_self",
        "development_episodes_per_environment": development_count,
        "canonical_episodes_per_environment": canonical_count,
        "canonical_workload_scale_factor": 40,
        "runtime_safety_multiplier": 2,
        "peak_memory_safety_multiplier": 2,
        "remaining_m7b_seconds": 57600,
        "planning_physical_memory_divisor": 2,
        "time_projection_method": "median_development_time_times_episode_scale",
        "memory_projection_method": ("maximum_baseline_plus_maximum_increment_times_episode_scale"),
        "result_scope": "operational_forecast_not_scientific_result",
        "canonical_claim_allowed": False,
    }
    plan = DevelopmentRuntimeForecastPlan.model_validate(
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
    if any(row.matrix_content_hash != source_manifest.matrix_content_hash for row in samples):
        raise AcquisitionRuntimeForecastError(
            "Measured workload differs from the verified development matrix"
        )

    median_elapsed = int(median(row.elapsed_nanoseconds for row in samples))
    maximum_baseline = max(row.baseline_peak_rss_bytes for row in samples)
    maximum_observed = max(row.observed_peak_rss_bytes for row in samples)
    maximum_increment = max(row.incremental_peak_rss_bytes for row in samples)
    linear_time = median_elapsed * plan.canonical_workload_scale_factor
    guarded_time = linear_time * plan.runtime_safety_multiplier
    linear_memory = maximum_baseline + (maximum_increment * plan.canonical_workload_scale_factor)
    guarded_memory = maximum_baseline + (
        maximum_increment
        * plan.canonical_workload_scale_factor
        * plan.peak_memory_safety_multiplier
    )
    runtime_platform = _platform()
    planning_memory_limit = (
        runtime_platform.physical_memory_bytes // plan.planning_physical_memory_divisor
    )
    runtime_within_gate = guarded_time <= plan.remaining_m7b_seconds * _NANOSECONDS_PER_SECOND
    memory_within_limit = guarded_memory <= planning_memory_limit
    report_content = {
        "schema_version": 1,
        "schema_id": "acquisition_study.development_runtime_forecast_report.v1",
        "study_id": source_matrix.study_id,
        "run_id": run_id,
        "split": source_matrix.split,
        "runtime_plan_hash": plan.plan_hash,
        "source_matrix_content_hash": source_manifest.matrix_content_hash,
        "platform": runtime_platform,
        "measurement_count": len(samples),
        "measurements": samples,
        "source_matrix_parity_verified": True,
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
        "linear_projected_canonical_peak_rss_bytes": linear_memory,
        "guarded_projected_canonical_peak_rss_bytes": guarded_memory,
        "planning_memory_limit_bytes": planning_memory_limit,
        "runtime_within_remaining_m7b_time": runtime_within_gate,
        "memory_within_planning_limit": memory_within_limit,
        "current_runner_feasible_for_canonical_run": (runtime_within_gate and memory_within_limit),
        "projection_limit": (
            "linear_planning_estimate_not_a_guarantee_and_not_a_scientific_outcome"
        ),
        "result_status": "development_operational_forecast",
        "external_model_call_count": 0,
        "sandbox_call_count": 0,
        "human_record_count": 0,
        "canonical_claim_allowed": False,
    }
    report = DevelopmentRuntimeForecastReport.model_validate(
        {**report_content, "report_hash": canonical_sha256(report_content)}
    )
    write_immutable_json(output_root / "development_runtime_forecast_plan.json", plan)
    write_immutable_json(output_root / "development_runtime_forecast_report.json", report)
    return report


def _run_fresh_process_measurement(
    *,
    repetition_index: int,
    environment_path: Path,
    analysis_path: Path,
    episodes_per_environment: int,
) -> AcquisitionRuntimeSample:
    command = [
        sys.executable,
        "-m",
        "socratic_tutor.acquisition_study.runtime_worker",
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
        raise AcquisitionRuntimeForecastError(
            f"Runtime measurement {repetition_index} could not complete"
        ) from error
    if completed.returncode != 0:
        message = completed.stderr.strip() or "worker returned no error detail"
        raise AcquisitionRuntimeForecastError(
            f"Runtime measurement {repetition_index} failed: {message}"
        )
    try:
        return AcquisitionRuntimeSample.model_validate_json(completed.stdout)
    except ValidationError as error:
        raise AcquisitionRuntimeForecastError(
            f"Runtime measurement {repetition_index} returned invalid data"
        ) from error


def _platform() -> AcquisitionRuntimePlatform:
    cpu_count = os.cpu_count()
    if cpu_count is None or cpu_count < 1:
        raise AcquisitionRuntimeForecastError("Could not determine logical CPU count")
    try:
        physical_memory = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError) as error:
        raise AcquisitionRuntimeForecastError("Could not determine physical memory") from error
    return AcquisitionRuntimePlatform(
        operating_system=platform.platform(),
        machine=platform.machine(),
        processor=platform.processor(),
        python_implementation=platform.python_implementation(),
        python_version=platform.python_version(),
        logical_cpu_count=cpu_count,
        physical_memory_bytes=physical_memory,
    )


def runtime_seconds(nanoseconds: int) -> float:
    """Convert a stored integer duration for concise command output."""

    return nanoseconds / _NANOSECONDS_PER_SECOND
