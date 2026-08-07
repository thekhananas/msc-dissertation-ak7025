"""One-factor sensitivity analysis for the tracker-study development split."""

from __future__ import annotations

import json
import math
from enum import StrEnum
from pathlib import Path
from statistics import fmean
from typing import Literal

from pydantic import Field, ValidationError, model_validator

from socratic_tutor.benchmark.artifacts import write_immutable_json
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import canonical_sha256, file_sha256, model_content_hash
from socratic_tutor.contracts import ContractModel
from socratic_tutor.tracker_study.analysis import (
    TrackerDevelopmentAnalysisPlan,
    TrackerDevelopmentAnalysisReport,
    TrackerStudyAnalysisError,
    load_verified_stress_matrix,
)
from socratic_tutor.tracker_study.analysis_spec import TrackerStudyAnalysisSpecification
from socratic_tutor.tracker_study.config import (
    StressCondition,
    TrackerId,
    TrackerStudyConfiguration,
)
from socratic_tutor.tracker_study.simulation import SimulatedEpisode, StudySplit
from socratic_tutor.tracker_study.trackers import (
    ConfiguredMasteryTracker,
    propagate_mastery_probability,
)


class TrackerSensitivityError(ValueError):
    """A sensitivity input or result violates the declared development design."""


class SensitivityDimension(StrEnum):
    TRUST_WEIGHT_MULTIPLIER = "trust_weight_multiplier"
    CLIPPING_KAPPA = "clipping_kappa"
    ASSUMED_CHANNEL_RELIABILITY_SCALE = "assumed_channel_reliability_scale"


class TrackerSensitivityPoint(ContractModel):
    dimension: SensitivityDimension
    value: float = Field(ge=0.0, allow_inf_nan=False)
    is_configured_anchor: bool
    episode_count: int = Field(ge=1)
    candidate_mean_adverse_brier: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    reference_mean_adverse_brier: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    candidate_minus_reference_adverse_brier: float = Field(
        ge=-1.0,
        le=1.0,
        allow_inf_nan=False,
    )
    candidate_clean_brier: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    reference_clean_brier: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    candidate_minus_reference_clean_brier: float = Field(
        ge=-1.0,
        le=1.0,
        allow_inf_nan=False,
    )

    @model_validator(mode="after")
    def validate_differences(self) -> TrackerSensitivityPoint:
        adverse = self.candidate_mean_adverse_brier - self.reference_mean_adverse_brier
        clean = self.candidate_clean_brier - self.reference_clean_brier
        if not math.isclose(
            self.candidate_minus_reference_adverse_brier,
            adverse,
            abs_tol=1e-12,
        ):
            raise ValueError("Adverse sensitivity difference does not reconcile")
        if not math.isclose(
            self.candidate_minus_reference_clean_brier,
            clean,
            abs_tol=1e-12,
        ):
            raise ValueError("Clean sensitivity difference does not reconcile")
        return self


class AnchorReconciliation(ContractModel):
    configured_anchor_count: Literal[3] = 3
    tolerance: float = Field(default=1e-12, gt=0.0, le=1e-12)
    maximum_absolute_difference: float = Field(ge=0.0, le=1e-12, allow_inf_nan=False)
    passed: Literal[True] = True


class TrackerDevelopmentSensitivityPlan(ContractModel):
    schema_version: Literal[1] = 1
    schema_id: Literal["tracker_study.development_sensitivity_plan.v1"] = (
        "tracker_study.development_sensitivity_plan.v1"
    )
    study_id: str
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,99}$")
    split: Literal[StudySplit.DEVELOPMENT] = StudySplit.DEVELOPMENT
    simulation_manifest_hash: Sha256
    development_analysis_plan_hash: Sha256
    development_analysis_report_hash: Sha256
    configuration_hash: Sha256
    analysis_specification_hash: Sha256
    sensitivity_grid_hash: Sha256
    sensitivity_code_revision: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    pixi_lock_sha256: Sha256
    method: Literal["one_factor_at_a_time_on_recorded_development_episodes"] = (
        "one_factor_at_a_time_on_recorded_development_episodes"
    )
    reliability_interpretation: Literal[
        "scale_assumed_channel_likelihoods_towards_half_without_regenerating_evidence"
    ] = "scale_assumed_channel_likelihoods_towards_half_without_regenerating_evidence"
    parameter_selection_allowed: Literal[False] = False
    canonical_claim_allowed: Literal[False] = False
    external_model_call_count: Literal[0] = 0
    sandbox_call_count: Literal[0] = 0
    human_record_count: Literal[0] = 0
    plan_hash: Sha256

    @model_validator(mode="after")
    def validate_plan(self) -> TrackerDevelopmentSensitivityPlan:
        if self.plan_hash != model_content_hash(self, exclude={"plan_hash"}):
            raise ValueError("Development sensitivity plan hash does not match its content")
        return self


class TrackerDevelopmentSensitivityReport(ContractModel):
    schema_version: Literal[1] = 1
    schema_id: Literal["tracker_study.development_sensitivity_report.v1"] = (
        "tracker_study.development_sensitivity_report.v1"
    )
    study_id: str
    run_id: str
    split: Literal[StudySplit.DEVELOPMENT] = StudySplit.DEVELOPMENT
    sensitivity_plan_hash: Sha256
    result_status: Literal["development_descriptive_not_for_parameter_selection"] = (
        "development_descriptive_not_for_parameter_selection"
    )
    point_count: Literal[12] = 12
    points: tuple[TrackerSensitivityPoint, ...] = Field(min_length=12, max_length=12)
    anchor_reconciliation: AnchorReconciliation
    parameter_selected: Literal[False] = False
    configured_tracker_unchanged: Literal[True] = True
    canonical_claim_allowed: Literal[False] = False
    inference_scope: Literal["declared_simulator_and_recorded_development_episodes_only"] = (
        "declared_simulator_and_recorded_development_episodes_only"
    )
    human_learning_claim_supported: Literal[False] = False
    tutoring_efficacy_claim_supported: Literal[False] = False
    report_hash: Sha256

    @model_validator(mode="after")
    def validate_report(self) -> TrackerDevelopmentSensitivityReport:
        keys = tuple((point.dimension, point.value) for point in self.points)
        if len(set(keys)) != len(keys):
            raise ValueError("Sensitivity report contains duplicate grid points")
        if sum(point.is_configured_anchor for point in self.points) != 3:
            raise ValueError("Sensitivity report requires one configured anchor per dimension")
        if self.report_hash != model_content_hash(self, exclude={"report_hash"}):
            raise ValueError("Development sensitivity report hash does not match its content")
        return self


def run_development_sensitivity(
    *,
    simulation_manifest_path: Path,
    development_analysis_plan_path: Path,
    development_analysis_report_path: Path,
    configuration: TrackerStudyConfiguration,
    specification: TrackerStudyAnalysisSpecification,
    pixi_lock_path: Path,
    output_root: Path,
    run_id: str,
    sensitivity_code_revision: str,
) -> TrackerDevelopmentSensitivityReport:
    """Run every frozen one-factor grid without selecting a parameter value."""

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
        raise TrackerSensitivityError(f"Could not read Pixi lock: {pixi_lock_path}") from error
    grid = _grid(configuration)
    plan = _build_plan(
        manifest_hash=manifest.manifest_hash,
        source_plan=source_plan,
        source_report=source_report,
        configuration=configuration,
        specification=specification,
        grid=grid,
        pixi_lock_hash=pixi_lock_hash,
        run_id=run_id,
        sensitivity_code_revision=sensitivity_code_revision,
    )
    points = tuple(
        _evaluate_point(
            matrix.episodes,
            episodes_per_condition=matrix.episodes_per_condition,
            configuration=configuration,
            specification=specification,
            dimension=dimension,
            value=value,
        )
        for dimension, value in grid
    )
    reconciliation = _reconcile_anchors(points, source_report)
    content = {
        "schema_version": 1,
        "schema_id": "tracker_study.development_sensitivity_report.v1",
        "study_id": configuration.study_id,
        "run_id": run_id,
        "split": StudySplit.DEVELOPMENT,
        "sensitivity_plan_hash": plan.plan_hash,
        "result_status": "development_descriptive_not_for_parameter_selection",
        "point_count": 12,
        "points": points,
        "anchor_reconciliation": reconciliation,
        "parameter_selected": False,
        "configured_tracker_unchanged": True,
        "canonical_claim_allowed": False,
        "inference_scope": "declared_simulator_and_recorded_development_episodes_only",
        "human_learning_claim_supported": False,
        "tutoring_efficacy_claim_supported": False,
    }
    report = TrackerDevelopmentSensitivityReport.model_validate(
        {**content, "report_hash": canonical_sha256(content)}
    )
    write_immutable_json(output_root / "development_sensitivity_plan.json", plan)
    write_immutable_json(output_root / "development_sensitivity_report.json", report)
    return report


def _grid(
    configuration: TrackerStudyConfiguration,
) -> tuple[tuple[SensitivityDimension, float], ...]:
    sensitivity = configuration.sensitivity
    return tuple(
        (dimension, value)
        for dimension, values in (
            (SensitivityDimension.TRUST_WEIGHT_MULTIPLIER, sensitivity.trust_weight_multipliers),
            (SensitivityDimension.CLIPPING_KAPPA, sensitivity.clipping_kappas),
            (
                SensitivityDimension.ASSUMED_CHANNEL_RELIABILITY_SCALE,
                sensitivity.reliability_scales,
            ),
        )
        for value in values
    )


def _build_plan(
    *,
    manifest_hash: Sha256,
    source_plan: TrackerDevelopmentAnalysisPlan,
    source_report: TrackerDevelopmentAnalysisReport,
    configuration: TrackerStudyConfiguration,
    specification: TrackerStudyAnalysisSpecification,
    grid: tuple[tuple[SensitivityDimension, float], ...],
    pixi_lock_hash: Sha256,
    run_id: str,
    sensitivity_code_revision: str,
) -> TrackerDevelopmentSensitivityPlan:
    content = {
        "schema_version": 1,
        "schema_id": "tracker_study.development_sensitivity_plan.v1",
        "study_id": configuration.study_id,
        "run_id": run_id,
        "split": StudySplit.DEVELOPMENT,
        "simulation_manifest_hash": manifest_hash,
        "development_analysis_plan_hash": source_plan.plan_hash,
        "development_analysis_report_hash": source_report.report_hash,
        "configuration_hash": configuration.configuration_hash,
        "analysis_specification_hash": specification.analysis_specification_hash,
        "sensitivity_grid_hash": canonical_sha256(grid),
        "sensitivity_code_revision": sensitivity_code_revision,
        "pixi_lock_sha256": pixi_lock_hash,
        "method": "one_factor_at_a_time_on_recorded_development_episodes",
        "reliability_interpretation": (
            "scale_assumed_channel_likelihoods_towards_half_without_regenerating_evidence"
        ),
        "parameter_selection_allowed": False,
        "canonical_claim_allowed": False,
        "external_model_call_count": 0,
        "sandbox_call_count": 0,
        "human_record_count": 0,
    }
    return TrackerDevelopmentSensitivityPlan.model_validate(
        {**content, "plan_hash": canonical_sha256(content)}
    )


def _evaluate_point(
    episodes: tuple[SimulatedEpisode, ...],
    *,
    episodes_per_condition: int,
    configuration: TrackerStudyConfiguration,
    specification: TrackerStudyAnalysisSpecification,
    dimension: SensitivityDimension,
    value: float,
) -> TrackerSensitivityPoint:
    candidate_by_condition: dict[StressCondition, list[float]] = {
        condition: [] for condition in StressCondition
    }
    reference_by_condition: dict[StressCondition, list[float]] = {
        condition: [] for condition in StressCondition
    }
    candidate, reference = _trackers_for_point(configuration, dimension, value)
    for episode in episodes:
        candidate_by_condition[episode.condition].append(
            _episode_brier(episode, candidate, configuration)
        )
        reference_by_condition[episode.condition].append(
            _episode_brier(episode, reference, configuration)
        )
    adverse = specification.primary.adverse_conditions
    candidate_adverse = fmean(
        value for condition in adverse for value in candidate_by_condition[condition]
    )
    reference_adverse = fmean(
        value for condition in adverse for value in reference_by_condition[condition]
    )
    candidate_clean = fmean(candidate_by_condition[StressCondition.CLEAN])
    reference_clean = fmean(reference_by_condition[StressCondition.CLEAN])
    return TrackerSensitivityPoint(
        dimension=dimension,
        value=value,
        is_configured_anchor=_is_configured_anchor(dimension, value, configuration),
        episode_count=episodes_per_condition,
        candidate_mean_adverse_brier=candidate_adverse,
        reference_mean_adverse_brier=reference_adverse,
        candidate_minus_reference_adverse_brier=candidate_adverse - reference_adverse,
        candidate_clean_brier=candidate_clean,
        reference_clean_brier=reference_clean,
        candidate_minus_reference_clean_brier=candidate_clean - reference_clean,
    )


def _trackers_for_point(
    configuration: TrackerStudyConfiguration,
    dimension: SensitivityDimension,
    value: float,
) -> tuple[ConfiguredMasteryTracker, ConfiguredMasteryTracker]:
    trust_multiplier = value if dimension is SensitivityDimension.TRUST_WEIGHT_MULTIPLIER else 1.0
    clipping_kappa = value if dimension is SensitivityDimension.CLIPPING_KAPPA else None
    reliability_scale = (
        value if dimension is SensitivityDimension.ASSUMED_CHANNEL_RELIABILITY_SCALE else 1.0
    )
    candidate = ConfiguredMasteryTracker(
        tracker_id=TrackerId.BOUNDED_CHANNEL_AWARE,
        configuration=configuration,
        trust_weight_multiplier=trust_multiplier,
        clipping_kappa_override=clipping_kappa,
        assumed_channel_reliability_scale=reliability_scale,
    )
    reference = ConfiguredMasteryTracker(
        tracker_id=TrackerId.CHANNEL_AWARE,
        configuration=configuration,
        assumed_channel_reliability_scale=reliability_scale,
    )
    return candidate, reference


def _episode_brier(
    episode: SimulatedEpisode,
    tracker: ConfiguredMasteryTracker,
    configuration: TrackerStudyConfiguration,
) -> float:
    prior = configuration.latent_dynamics.initial_mastery_probability
    errors: list[float] = []
    for turn in episode.turns:
        belief = prior
        for observation in turn.observations:
            belief = tracker.update(belief, observation).posterior_mastery_probability
        errors.append((belief - float(turn.latent_mastery)) ** 2)
        prior = propagate_mastery_probability(
            belief,
            configuration.latent_dynamics,
            configuration.trackers.probability_floor,
        )
    return fmean(errors)


def _is_configured_anchor(
    dimension: SensitivityDimension,
    value: float,
    configuration: TrackerStudyConfiguration,
) -> bool:
    expected = {
        SensitivityDimension.TRUST_WEIGHT_MULTIPLIER: 1.0,
        SensitivityDimension.CLIPPING_KAPPA: (configuration.trackers.bounded_log_likelihood_kappa),
        SensitivityDimension.ASSUMED_CHANNEL_RELIABILITY_SCALE: 1.0,
    }[dimension]
    return math.isclose(value, expected, abs_tol=1e-12)


def _reconcile_anchors(
    points: tuple[TrackerSensitivityPoint, ...],
    source_report: TrackerDevelopmentAnalysisReport,
) -> AnchorReconciliation:
    anchors = tuple(point for point in points if point.is_configured_anchor)
    expected = (
        source_report.primary_adverse_brier.mean_effect,
        source_report.clean_brier_guardrail.mean_effect,
    )
    differences = tuple(
        abs(actual - expected_value)
        for point in anchors
        for actual, expected_value in zip(
            (
                point.candidate_minus_reference_adverse_brier,
                point.candidate_minus_reference_clean_brier,
            ),
            expected,
            strict=True,
        )
    )
    maximum = max(differences)
    if maximum > 1e-12:
        raise TrackerSensitivityError(
            "Configured sensitivity anchors do not reproduce the development analysis"
        )
    return AnchorReconciliation(maximum_absolute_difference=maximum)


def _validate_lineage(
    *,
    manifest_hash: Sha256,
    configuration: TrackerStudyConfiguration,
    specification: TrackerStudyAnalysisSpecification,
    source_plan: TrackerDevelopmentAnalysisPlan,
    source_report: TrackerDevelopmentAnalysisReport,
) -> None:
    if source_plan.simulation_manifest_hash != manifest_hash:
        raise TrackerSensitivityError("Development analysis belongs to another simulation")
    if source_plan.configuration_hash != configuration.configuration_hash:
        raise TrackerSensitivityError("Development analysis belongs to another configuration")
    if source_plan.analysis_specification_hash != specification.analysis_specification_hash:
        raise TrackerSensitivityError("Development analysis belongs to another specification")
    if source_report.analysis_plan_hash != source_plan.plan_hash:
        raise TrackerSensitivityError("Development analysis report and plan differ")
    if (
        source_plan.split is not StudySplit.DEVELOPMENT
        or source_report.split is not StudySplit.DEVELOPMENT
    ):
        raise TrackerSensitivityError("Sensitivity analysis accepts development sources only")


def _load_model[ModelT: ContractModel](
    path: Path,
    model: type[ModelT],
    label: str,
) -> ModelT:
    try:
        return model.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValidationError, TrackerStudyAnalysisError) as error:
        raise TrackerSensitivityError(f"Could not verify {label}: {path}") from error
