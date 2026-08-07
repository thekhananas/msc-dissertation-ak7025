"""Platform-specific runtime profiling for configured mastery trackers."""

from __future__ import annotations

import gc
import json
import math
import platform
import sys
from pathlib import Path
from statistics import median
from time import perf_counter_ns
from typing import Literal

from pydantic import Field, ValidationError, model_validator

from socratic_tutor.benchmark.artifacts import write_immutable_json
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import canonical_sha256, file_sha256, model_content_hash
from socratic_tutor.contracts import ContractModel
from socratic_tutor.tracker_study.analysis import (
    TrackerDevelopmentAnalysisPlan,
    TrackerDevelopmentAnalysisReport,
    load_verified_stress_matrix,
)
from socratic_tutor.tracker_study.analysis_spec import TrackerStudyAnalysisSpecification
from socratic_tutor.tracker_study.config import (
    EvidenceCategory,
    TrackerId,
    TrackerStudyConfiguration,
)
from socratic_tutor.tracker_study.simulation import SimulatedEpisode, StudySplit
from socratic_tutor.tracker_study.trackers import (
    ConfiguredMasteryTracker,
    EvidenceObservation,
    build_trackers,
    propagate_mastery_probability,
)


class TrackerRuntimeError(ValueError):
    """A runtime-profile input or measurement violates the declared protocol."""


class RuntimePlatform(ContractModel):
    operating_system: str = Field(min_length=1)
    machine: str = Field(min_length=1)
    processor: str
    python_implementation: str = Field(min_length=1)
    python_version: str = Field(min_length=1)


class TrackerRuntimeMeasurement(ContractModel):
    tracker_id: TrackerId
    workload_hash: Sha256
    non_missing_updates_per_repetition: int = Field(ge=1)
    warmup_repetitions: Literal[3] = 3
    measured_repetitions: Literal[20] = 20
    elapsed_nanoseconds: tuple[int, ...] = Field(min_length=20, max_length=20)
    median_nanoseconds_per_non_missing_update: float = Field(gt=0.0, allow_inf_nan=False)
    minimum_nanoseconds_per_non_missing_update: float = Field(gt=0.0, allow_inf_nan=False)
    maximum_nanoseconds_per_non_missing_update: float = Field(gt=0.0, allow_inf_nan=False)
    posterior_checksum: float = Field(allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_summary(self) -> TrackerRuntimeMeasurement:
        if any(value <= 0 for value in self.elapsed_nanoseconds):
            raise ValueError("Runtime samples must be positive")
        per_update = tuple(
            value / self.non_missing_updates_per_repetition for value in self.elapsed_nanoseconds
        )
        expected = (median(per_update), min(per_update), max(per_update))
        actual = (
            self.median_nanoseconds_per_non_missing_update,
            self.minimum_nanoseconds_per_non_missing_update,
            self.maximum_nanoseconds_per_non_missing_update,
        )
        if any(
            not math.isclose(left, right, abs_tol=1e-12)
            for left, right in zip(actual, expected, strict=True)
        ):
            raise ValueError("Runtime summary does not reconcile with raw samples")
        return self


class TrackerDevelopmentRuntimePlan(ContractModel):
    schema_version: Literal[1] = 1
    schema_id: Literal["tracker_study.development_runtime_plan.v1"] = (
        "tracker_study.development_runtime_plan.v1"
    )
    study_id: str
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,99}$")
    split: Literal[StudySplit.DEVELOPMENT] = StudySplit.DEVELOPMENT
    simulation_manifest_hash: Sha256
    development_analysis_plan_hash: Sha256
    development_analysis_report_hash: Sha256
    configuration_hash: Sha256
    analysis_specification_hash: Sha256
    runtime_code_revision: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    pixi_lock_sha256: Sha256
    clock: Literal["time.perf_counter_ns"] = "time.perf_counter_ns"
    workload: Literal["configured_update_method_on_recorded_non_missing_observations"] = (
        "configured_update_method_on_recorded_non_missing_observations"
    )
    garbage_collection_during_timing: Literal["disabled_then_restored"] = "disabled_then_restored"
    warmup_repetitions: Literal[3] = 3
    measured_repetitions: Literal[20] = 20
    comparison_role: Literal["descriptive_platform_specific_not_inferential"] = (
        "descriptive_platform_specific_not_inferential"
    )
    canonical_claim_allowed: Literal[False] = False
    external_model_call_count: Literal[0] = 0
    sandbox_call_count: Literal[0] = 0
    human_record_count: Literal[0] = 0
    plan_hash: Sha256

    @model_validator(mode="after")
    def validate_plan(self) -> TrackerDevelopmentRuntimePlan:
        if self.plan_hash != model_content_hash(self, exclude={"plan_hash"}):
            raise ValueError("Development runtime plan hash does not match its content")
        return self


class TrackerDevelopmentRuntimeReport(ContractModel):
    schema_version: Literal[1] = 1
    schema_id: Literal["tracker_study.development_runtime_report.v1"] = (
        "tracker_study.development_runtime_report.v1"
    )
    study_id: str
    run_id: str
    split: Literal[StudySplit.DEVELOPMENT] = StudySplit.DEVELOPMENT
    runtime_plan_hash: Sha256
    platform: RuntimePlatform
    measurement_count: Literal[5] = 5
    measurements: tuple[TrackerRuntimeMeasurement, ...] = Field(min_length=5, max_length=5)
    non_missing_updates_per_tracker_repetition: int = Field(ge=1)
    measured_update_invocation_count: int = Field(ge=1)
    warmup_update_invocation_count: int = Field(ge=1)
    result_status: Literal["development_platform_profile_not_inferential"] = (
        "development_platform_profile_not_inferential"
    )
    exact_replay_expected: Literal[False] = False
    canonical_claim_allowed: Literal[False] = False
    human_learning_claim_supported: Literal[False] = False
    tutoring_efficacy_claim_supported: Literal[False] = False
    report_hash: Sha256

    @model_validator(mode="after")
    def validate_report(self) -> TrackerDevelopmentRuntimeReport:
        if tuple(item.tracker_id for item in self.measurements) != tuple(TrackerId):
            raise ValueError("Runtime report differs from the configured tracker order")
        counts = {item.non_missing_updates_per_repetition for item in self.measurements}
        if counts != {self.non_missing_updates_per_tracker_repetition}:
            raise ValueError("Trackers were profiled with different operation counts")
        expected_measured = (
            self.non_missing_updates_per_tracker_repetition
            * len(self.measurements)
            * self.measurements[0].measured_repetitions
        )
        expected_warmup = (
            self.non_missing_updates_per_tracker_repetition
            * len(self.measurements)
            * self.measurements[0].warmup_repetitions
        )
        if self.measured_update_invocation_count != expected_measured:
            raise ValueError("Measured update invocation count does not reconcile")
        if self.warmup_update_invocation_count != expected_warmup:
            raise ValueError("Warm-up update invocation count does not reconcile")
        if self.report_hash != model_content_hash(self, exclude={"report_hash"}):
            raise ValueError("Development runtime report hash does not match its content")
        return self


def run_development_runtime_profile(
    *,
    simulation_manifest_path: Path,
    development_analysis_plan_path: Path,
    development_analysis_report_path: Path,
    configuration: TrackerStudyConfiguration,
    specification: TrackerStudyAnalysisSpecification,
    pixi_lock_path: Path,
    output_root: Path,
    run_id: str,
    runtime_code_revision: str,
) -> TrackerDevelopmentRuntimeReport:
    """Profile configured tracker updates over one fixed development workload."""

    manifest, matrix = load_verified_stress_matrix(simulation_manifest_path, configuration)
    source_plan = _load_model(
        development_analysis_plan_path,
        TrackerDevelopmentAnalysisPlan,
        "development analysis plan",
    )
    source_report = _load_model(
        development_analysis_report_path,
        TrackerDevelopmentAnalysisReport,
        "development analysis report",
    )
    _validate_lineage(
        manifest_hash=manifest.manifest_hash,
        configuration=configuration,
        specification=specification,
        source_plan=source_plan,
        source_report=source_report,
    )
    try:
        pixi_lock_hash = file_sha256(pixi_lock_path.read_bytes())
    except OSError as error:
        raise TrackerRuntimeError(f"Could not read Pixi lock: {pixi_lock_path}") from error
    runtime = specification.runtime
    plan = _build_plan(
        manifest_hash=manifest.manifest_hash,
        source_plan=source_plan,
        source_report=source_report,
        configuration=configuration,
        specification=specification,
        pixi_lock_hash=pixi_lock_hash,
        run_id=run_id,
        runtime_code_revision=runtime_code_revision,
    )
    measurements = tuple(
        _profile_tracker(
            tracker,
            matrix.episodes,
            configuration,
            warmup_repetitions=runtime.warmup_repetitions,
            measured_repetitions=runtime.measured_repetitions,
        )
        for tracker in build_trackers(configuration)
    )
    count = measurements[0].non_missing_updates_per_repetition
    content = {
        "schema_version": 1,
        "schema_id": "tracker_study.development_runtime_report.v1",
        "study_id": configuration.study_id,
        "run_id": run_id,
        "split": StudySplit.DEVELOPMENT,
        "runtime_plan_hash": plan.plan_hash,
        "platform": _platform(),
        "measurement_count": 5,
        "measurements": measurements,
        "non_missing_updates_per_tracker_repetition": count,
        "measured_update_invocation_count": (
            count * len(measurements) * runtime.measured_repetitions
        ),
        "warmup_update_invocation_count": (count * len(measurements) * runtime.warmup_repetitions),
        "result_status": "development_platform_profile_not_inferential",
        "exact_replay_expected": False,
        "canonical_claim_allowed": False,
        "human_learning_claim_supported": False,
        "tutoring_efficacy_claim_supported": False,
    }
    report = TrackerDevelopmentRuntimeReport.model_validate(
        {**content, "report_hash": canonical_sha256(content)}
    )
    write_immutable_json(output_root / "development_runtime_plan.json", plan)
    write_immutable_json(output_root / "development_runtime_report.json", report)
    return report


def _profile_tracker(
    tracker: ConfiguredMasteryTracker,
    episodes: tuple[SimulatedEpisode, ...],
    configuration: TrackerStudyConfiguration,
    *,
    warmup_repetitions: Literal[3],
    measured_repetitions: Literal[20],
) -> TrackerRuntimeMeasurement:
    workload = _build_workload(tracker, episodes, configuration)
    expected_checksum = _execute_workload(tracker, workload)
    for _ in range(warmup_repetitions):
        checksum = _execute_workload(tracker, workload)
        _check_checksum(checksum, expected_checksum)
    elapsed: list[int] = []
    collection_was_enabled = gc.isenabled()
    try:
        gc.disable()
        for _ in range(measured_repetitions):
            started = perf_counter_ns()
            checksum = _execute_workload(tracker, workload)
            elapsed.append(perf_counter_ns() - started)
            _check_checksum(checksum, expected_checksum)
    finally:
        if collection_was_enabled:
            gc.enable()
    per_update = tuple(value / len(workload) for value in elapsed)
    return TrackerRuntimeMeasurement(
        tracker_id=tracker.tracker_id,
        workload_hash=canonical_sha256(workload),
        non_missing_updates_per_repetition=len(workload),
        warmup_repetitions=warmup_repetitions,
        measured_repetitions=measured_repetitions,
        elapsed_nanoseconds=tuple(elapsed),
        median_nanoseconds_per_non_missing_update=median(per_update),
        minimum_nanoseconds_per_non_missing_update=min(per_update),
        maximum_nanoseconds_per_non_missing_update=max(per_update),
        posterior_checksum=expected_checksum,
    )


def _build_workload(
    tracker: ConfiguredMasteryTracker,
    episodes: tuple[SimulatedEpisode, ...],
    configuration: TrackerStudyConfiguration,
) -> tuple[tuple[float, EvidenceObservation], ...]:
    workload: list[tuple[float, EvidenceObservation]] = []
    for episode in episodes:
        prior = configuration.latent_dynamics.initial_mastery_probability
        for turn in episode.turns:
            belief = prior
            for observation in turn.observations:
                if observation.category is not EvidenceCategory.MISSING:
                    workload.append((belief, observation))
                belief = tracker.update(belief, observation).posterior_mastery_probability
            prior = propagate_mastery_probability(
                belief,
                configuration.latent_dynamics,
                configuration.trackers.probability_floor,
            )
    if not workload:
        raise TrackerRuntimeError("Runtime workload contains no non-missing observations")
    return tuple(workload)


def _execute_workload(
    tracker: ConfiguredMasteryTracker,
    workload: tuple[tuple[float, EvidenceObservation], ...],
) -> float:
    return math.fsum(
        tracker.update(prior, observation).posterior_mastery_probability
        for prior, observation in workload
    )


def _check_checksum(actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise TrackerRuntimeError("Timed update workload produced a different checksum")


def _platform() -> RuntimePlatform:
    return RuntimePlatform(
        operating_system=platform.system(),
        machine=platform.machine(),
        processor=platform.processor(),
        python_implementation=platform.python_implementation(),
        python_version=sys.version.split()[0],
    )


def _build_plan(
    *,
    manifest_hash: Sha256,
    source_plan: TrackerDevelopmentAnalysisPlan,
    source_report: TrackerDevelopmentAnalysisReport,
    configuration: TrackerStudyConfiguration,
    specification: TrackerStudyAnalysisSpecification,
    pixi_lock_hash: Sha256,
    run_id: str,
    runtime_code_revision: str,
) -> TrackerDevelopmentRuntimePlan:
    content = {
        "schema_version": 1,
        "schema_id": "tracker_study.development_runtime_plan.v1",
        "study_id": configuration.study_id,
        "run_id": run_id,
        "split": StudySplit.DEVELOPMENT,
        "simulation_manifest_hash": manifest_hash,
        "development_analysis_plan_hash": source_plan.plan_hash,
        "development_analysis_report_hash": source_report.report_hash,
        "configuration_hash": configuration.configuration_hash,
        "analysis_specification_hash": specification.analysis_specification_hash,
        "runtime_code_revision": runtime_code_revision,
        "pixi_lock_sha256": pixi_lock_hash,
        "clock": "time.perf_counter_ns",
        "workload": "configured_update_method_on_recorded_non_missing_observations",
        "garbage_collection_during_timing": "disabled_then_restored",
        "warmup_repetitions": specification.runtime.warmup_repetitions,
        "measured_repetitions": specification.runtime.measured_repetitions,
        "comparison_role": "descriptive_platform_specific_not_inferential",
        "canonical_claim_allowed": False,
        "external_model_call_count": 0,
        "sandbox_call_count": 0,
        "human_record_count": 0,
    }
    return TrackerDevelopmentRuntimePlan.model_validate(
        {**content, "plan_hash": canonical_sha256(content)}
    )


def _validate_lineage(
    *,
    manifest_hash: Sha256,
    configuration: TrackerStudyConfiguration,
    specification: TrackerStudyAnalysisSpecification,
    source_plan: TrackerDevelopmentAnalysisPlan,
    source_report: TrackerDevelopmentAnalysisReport,
) -> None:
    if source_plan.simulation_manifest_hash != manifest_hash:
        raise TrackerRuntimeError("Development analysis belongs to another simulation")
    if source_plan.configuration_hash != configuration.configuration_hash:
        raise TrackerRuntimeError("Development analysis belongs to another configuration")
    if source_plan.analysis_specification_hash != specification.analysis_specification_hash:
        raise TrackerRuntimeError("Development analysis belongs to another specification")
    if source_report.analysis_plan_hash != source_plan.plan_hash:
        raise TrackerRuntimeError("Development analysis report and plan differ")
    if (
        source_plan.split is not StudySplit.DEVELOPMENT
        or source_report.split is not StudySplit.DEVELOPMENT
    ):
        raise TrackerRuntimeError("Runtime profiling accepts development sources only")


def _load_model[ModelT: ContractModel](
    path: Path,
    model: type[ModelT],
    label: str,
) -> ModelT:
    try:
        return model.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValidationError) as error:
        raise TrackerRuntimeError(f"Could not verify {label}: {path}") from error
