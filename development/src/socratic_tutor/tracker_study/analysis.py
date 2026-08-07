"""Deterministic metric analysis for tracker-study development trajectories."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import fmean, median
from typing import Literal

import numpy as np
from pydantic import Field, ValidationError, model_validator

from socratic_tutor.benchmark.artifacts import write_immutable_json
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import canonical_sha256, file_sha256, model_content_hash
from socratic_tutor.contracts import ContractModel
from socratic_tutor.tracker_study.analysis_spec import (
    TrackerStudyAnalysisSpecification,
)
from socratic_tutor.tracker_study.config import (
    StressCondition,
    TrackerId,
    TrackerStudyConfiguration,
)
from socratic_tutor.tracker_study.simulation import (
    SimulatedEpisode,
    StressMatrix,
    StressMatrixManifest,
    StudySplit,
)


class TrackerStudyAnalysisError(ValueError):
    """A tracker-study trajectory or analysis result violates the frozen design."""


class BinaryMetricSummary(ContractModel):
    observation_count: int = Field(ge=1)
    brier_score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    negative_log_likelihood: float = Field(ge=0.0, allow_inf_nan=False)
    expected_calibration_error: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    classification_error: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    mean_absolute_posterior_update: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class TrackerConditionMetrics(ContractModel):
    condition: StressCondition
    tracker_id: TrackerId
    episode_count: int = Field(ge=1)
    turn_count: int = Field(ge=1)
    non_missing_update_count: int = Field(ge=0)
    metrics: BinaryMetricSummary

    @model_validator(mode="after")
    def validate_counts(self) -> TrackerConditionMetrics:
        if self.metrics.observation_count != self.turn_count:
            raise ValueError("Metric denominator differs from condition turn count")
        return self


class PairedEpisodeInterval(ContractModel):
    episode_count: int = Field(ge=1)
    effects_hash: Sha256
    mean_effect: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    median_effect: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    confidence_level: float = Field(ge=0.95, le=0.95)
    interval_method: Literal["paired_episode_percentile_bootstrap"] = (
        "paired_episode_percentile_bootstrap"
    )
    bootstrap_repetitions: int = Field(ge=1)
    bootstrap_seed: int = Field(ge=0)
    interval_lower: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    interval_upper: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_interval(self) -> PairedEpisodeInterval:
        if self.interval_lower > self.interval_upper:
            raise ValueError("Paired interval bounds are reversed")
        return self


class PairedEpisodeInference(PairedEpisodeInterval):
    test_method: Literal["paired_episode_sign_swap_mean"] = "paired_episode_sign_swap_mean"
    test_repetitions: int = Field(ge=1)
    test_seed: int = Field(ge=0)
    two_sided_p_value: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    test_role: Literal["secondary_evidence_not_the_success_gate"] = (
        "secondary_evidence_not_the_success_gate"
    )


class TrackerDevelopmentAnalysisPlan(ContractModel):
    schema_version: Literal[1] = 1
    schema_id: Literal["tracker_study.development_analysis_plan.v1"] = (
        "tracker_study.development_analysis_plan.v1"
    )
    study_id: str
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,99}$")
    split: Literal[StudySplit.DEVELOPMENT] = StudySplit.DEVELOPMENT
    simulation_manifest_hash: Sha256
    trajectory_file_sha256: Sha256
    trajectory_content_hash: Sha256
    configuration_hash: Sha256
    analysis_specification_hash: Sha256
    analysis_code_revision: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    pixi_lock_sha256: Sha256
    canonical_claim_allowed: Literal[False] = False
    external_model_call_count: Literal[0] = 0
    sandbox_call_count: Literal[0] = 0
    human_record_count: Literal[0] = 0
    plan_hash: Sha256

    @model_validator(mode="after")
    def validate_plan(self) -> TrackerDevelopmentAnalysisPlan:
        if self.plan_hash != model_content_hash(self, exclude={"plan_hash"}):
            raise ValueError("Development analysis plan hash does not match its content")
        return self


class TrackerDevelopmentAnalysisReport(ContractModel):
    schema_version: Literal[1] = 1
    schema_id: Literal["tracker_study.development_analysis_report.v1"] = (
        "tracker_study.development_analysis_report.v1"
    )
    study_id: str
    run_id: str
    split: Literal[StudySplit.DEVELOPMENT] = StudySplit.DEVELOPMENT
    analysis_plan_hash: Sha256
    result_status: Literal["development_diagnostic_not_canonical"] = (
        "development_diagnostic_not_canonical"
    )
    canonical_claim_allowed: Literal[False] = False
    condition_metric_count: Literal[30] = 30
    condition_metrics: tuple[TrackerConditionMetrics, ...] = Field(
        min_length=30,
        max_length=30,
    )
    primary_adverse_brier: PairedEpisodeInference
    clean_brier_guardrail: PairedEpisodeInterval
    maximum_clean_brier_degradation: float = Field(ge=0.0, le=1.0)
    primary_decision_status: Literal["not_evaluated_on_development_split"] = (
        "not_evaluated_on_development_split"
    )
    sensitivity_status: Literal["pending_separate_analysis"] = "pending_separate_analysis"
    runtime_status: Literal["pending_separate_platform_profile"] = (
        "pending_separate_platform_profile"
    )
    inference_scope: Literal["repeated_episodes_from_declared_simulator_only"] = (
        "repeated_episodes_from_declared_simulator_only"
    )
    human_learning_claim_supported: Literal[False] = False
    tutoring_efficacy_claim_supported: Literal[False] = False
    report_hash: Sha256

    @model_validator(mode="after")
    def validate_report(self) -> TrackerDevelopmentAnalysisReport:
        expected_order = tuple(
            (condition, tracker_id) for condition in StressCondition for tracker_id in TrackerId
        )
        actual_order = tuple((row.condition, row.tracker_id) for row in self.condition_metrics)
        if actual_order != expected_order:
            raise ValueError("Condition metrics differ from the frozen comparison order")
        if self.primary_adverse_brier.episode_count != self.clean_brier_guardrail.episode_count:
            raise ValueError("Primary and clean comparisons use different episode counts")
        if self.report_hash != model_content_hash(self, exclude={"report_hash"}):
            raise ValueError("Development analysis report hash does not match its content")
        return self


def calculate_binary_metrics(
    *,
    probabilities: tuple[float, ...],
    outcomes: tuple[bool, ...],
    priors: tuple[float, ...],
    probability_floor: float,
    classification_threshold: float,
    ece_bin_count: int,
) -> BinaryMetricSummary:
    """Calculate the five frozen descriptive metrics over one set of turns."""

    if not probabilities or len(probabilities) != len(outcomes) or len(priors) != len(outcomes):
        raise TrackerStudyAnalysisError("Metric vectors must be non-empty and equal in length")
    if not 0.0 < probability_floor < 0.5:
        raise TrackerStudyAnalysisError("Metric probability floor must lie in (0, 0.5)")
    if not 0.0 < classification_threshold < 1.0:
        raise TrackerStudyAnalysisError("Classification threshold must lie in (0, 1)")
    if ece_bin_count < 1:
        raise TrackerStudyAnalysisError("Expected calibration error requires at least one bin")
    probability_array = np.asarray(probabilities, dtype=np.float64)
    outcome_array = np.asarray(outcomes, dtype=np.float64)
    outcome_bool_array = np.asarray(outcomes, dtype=np.bool_)
    prior_array = np.asarray(priors, dtype=np.float64)
    if not (
        np.all(np.isfinite(probability_array))
        and np.all(np.isfinite(prior_array))
        and np.all((probability_array >= 0.0) & (probability_array <= 1.0))
        and np.all((prior_array >= 0.0) & (prior_array <= 1.0))
    ):
        raise TrackerStudyAnalysisError("Metric probabilities must be finite values in [0, 1]")
    clipped = np.clip(probability_array, probability_floor, 1.0 - probability_floor)
    brier = float(np.mean(np.square(probability_array - outcome_array)))
    nll = float(
        -np.mean(outcome_array * np.log(clipped) + (1.0 - outcome_array) * np.log(1.0 - clipped))
    )
    classifications = probability_array >= classification_threshold
    classification_error = float(
        np.count_nonzero(classifications != outcome_bool_array) / len(outcomes)
    )
    update_magnitude = float(np.mean(np.abs(probability_array - prior_array)))
    ece = _expected_calibration_error(
        probability_array,
        outcome_array,
        bin_count=ece_bin_count,
    )
    return BinaryMetricSummary(
        observation_count=len(probabilities),
        brier_score=brier,
        negative_log_likelihood=nll,
        expected_calibration_error=ece,
        classification_error=classification_error,
        mean_absolute_posterior_update=update_magnitude,
    )


def run_development_analysis(
    *,
    simulation_manifest_path: Path,
    configuration: TrackerStudyConfiguration,
    specification: TrackerStudyAnalysisSpecification,
    pixi_lock_path: Path,
    output_root: Path,
    run_id: str,
    analysis_code_revision: str,
) -> TrackerDevelopmentAnalysisReport:
    """Validate, analyse, and immutably publish development-only tracker metrics."""

    manifest, matrix = load_verified_stress_matrix(
        simulation_manifest_path,
        configuration,
    )
    if specification.configuration_hash != configuration.configuration_hash:
        raise TrackerStudyAnalysisError("Analysis specification and configuration differ")
    try:
        pixi_lock_hash = file_sha256(pixi_lock_path.read_bytes())
    except OSError as error:
        raise TrackerStudyAnalysisError(f"Could not read Pixi lock: {pixi_lock_path}") from error
    plan = _build_analysis_plan(
        manifest=manifest,
        specification=specification,
        pixi_lock_hash=pixi_lock_hash,
        run_id=run_id,
        analysis_code_revision=analysis_code_revision,
    )
    report = analyse_development_matrix(
        matrix,
        specification=specification,
        plan=plan,
    )
    write_immutable_json(output_root / "development_analysis_plan.json", plan)
    write_immutable_json(output_root / "development_analysis_report.json", report)
    return report


def load_verified_stress_matrix(
    manifest_path: Path,
    configuration: TrackerStudyConfiguration,
) -> tuple[StressMatrixManifest, StressMatrix]:
    """Load restricted trajectories only after all source hashes reconcile."""

    try:
        manifest_raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = StressMatrixManifest.model_validate(manifest_raw)
        trajectory_path = manifest_path.parent / manifest.trajectory_file
        trajectory_bytes = trajectory_path.read_bytes()
        if file_sha256(trajectory_bytes) != manifest.trajectory_file_sha256:
            raise TrackerStudyAnalysisError("Trajectory file hash differs from its manifest")
        episodes = tuple(
            SimulatedEpisode.model_validate(json.loads(line))
            for line in trajectory_bytes.decode("utf-8").splitlines()
            if line
        )
        matrix = StressMatrix(
            study_id=manifest.study_id,
            configuration_hash=manifest.configuration_hash,
            split=manifest.split,
            episodes_per_condition=manifest.episodes_per_condition,
            turns_per_episode=manifest.turns_per_episode,
            episodes=episodes,
        )
    except TrackerStudyAnalysisError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError) as error:
        raise TrackerStudyAnalysisError(
            f"Could not verify tracker-study trajectories: {manifest_path}"
        ) from error
    if manifest.configuration_hash != configuration.configuration_hash:
        raise TrackerStudyAnalysisError("Simulation manifest belongs to another configuration")
    if matrix.content_hash != manifest.trajectory_content_hash:
        raise TrackerStudyAnalysisError("Trajectory content hash differs from its manifest")
    return manifest, matrix


def analyse_development_matrix(
    matrix: StressMatrix,
    *,
    specification: TrackerStudyAnalysisSpecification,
    plan: TrackerDevelopmentAnalysisPlan,
) -> TrackerDevelopmentAnalysisReport:
    if matrix.split is not StudySplit.DEVELOPMENT:
        raise TrackerStudyAnalysisError("Development analysis cannot inspect the test split")
    rows, episode_brier = _condition_metrics(matrix, specification)
    primary_effects = _primary_episode_effects(
        episode_brier,
        episode_count=matrix.episodes_per_condition,
        specification=specification,
    )
    clean_effects = _clean_episode_effects(
        episode_brier,
        episode_count=matrix.episodes_per_condition,
        specification=specification,
    )
    primary = _paired_inference(primary_effects, specification)
    clean = _paired_interval(clean_effects, specification)
    content = {
        "schema_version": 1,
        "schema_id": "tracker_study.development_analysis_report.v1",
        "study_id": matrix.study_id,
        "run_id": plan.run_id,
        "split": StudySplit.DEVELOPMENT,
        "analysis_plan_hash": plan.plan_hash,
        "result_status": "development_diagnostic_not_canonical",
        "canonical_claim_allowed": False,
        "condition_metric_count": 30,
        "condition_metrics": rows,
        "primary_adverse_brier": primary,
        "clean_brier_guardrail": clean,
        "maximum_clean_brier_degradation": (specification.primary.maximum_clean_brier_degradation),
        "primary_decision_status": "not_evaluated_on_development_split",
        "sensitivity_status": "pending_separate_analysis",
        "runtime_status": "pending_separate_platform_profile",
        "inference_scope": "repeated_episodes_from_declared_simulator_only",
        "human_learning_claim_supported": False,
        "tutoring_efficacy_claim_supported": False,
    }
    return TrackerDevelopmentAnalysisReport.model_validate(
        {**content, "report_hash": canonical_sha256(content)}
    )


def _build_analysis_plan(
    *,
    manifest: StressMatrixManifest,
    specification: TrackerStudyAnalysisSpecification,
    pixi_lock_hash: Sha256,
    run_id: str,
    analysis_code_revision: str,
) -> TrackerDevelopmentAnalysisPlan:
    content = {
        "schema_version": 1,
        "schema_id": "tracker_study.development_analysis_plan.v1",
        "study_id": manifest.study_id,
        "run_id": run_id,
        "split": StudySplit.DEVELOPMENT,
        "simulation_manifest_hash": manifest.manifest_hash,
        "trajectory_file_sha256": manifest.trajectory_file_sha256,
        "trajectory_content_hash": manifest.trajectory_content_hash,
        "configuration_hash": manifest.configuration_hash,
        "analysis_specification_hash": specification.analysis_specification_hash,
        "analysis_code_revision": analysis_code_revision,
        "pixi_lock_sha256": pixi_lock_hash,
        "canonical_claim_allowed": False,
        "external_model_call_count": 0,
        "sandbox_call_count": 0,
        "human_record_count": 0,
    }
    return TrackerDevelopmentAnalysisPlan.model_validate(
        {**content, "plan_hash": canonical_sha256(content)}
    )


def _condition_metrics(
    matrix: StressMatrix,
    specification: TrackerStudyAnalysisSpecification,
) -> tuple[
    tuple[TrackerConditionMetrics, ...],
    dict[tuple[StressCondition, TrackerId, int], float],
]:
    grouped = {(episode.condition, episode.episode_index): episode for episode in matrix.episodes}
    rows: list[TrackerConditionMetrics] = []
    episode_brier: dict[tuple[StressCondition, TrackerId, int], float] = {}
    for condition in StressCondition:
        for tracker_id in TrackerId:
            probabilities: list[float] = []
            outcomes: list[bool] = []
            priors: list[float] = []
            update_count = 0
            for episode_index in range(matrix.episodes_per_condition):
                episode = grouped[(condition, episode_index)]
                episode_probabilities: list[float] = []
                episode_outcomes: list[bool] = []
                for turn in episode.turns:
                    estimate = next(
                        item for item in turn.tracker_estimates if item.tracker_id is tracker_id
                    )
                    probabilities.append(estimate.posterior_mastery_probability)
                    episode_probabilities.append(estimate.posterior_mastery_probability)
                    outcomes.append(turn.latent_mastery)
                    episode_outcomes.append(turn.latent_mastery)
                    priors.append(estimate.prior_mastery_probability)
                    update_count += estimate.evidence_used_count
                episode_brier[(condition, tracker_id, episode_index)] = _brier_score(
                    episode_probabilities,
                    episode_outcomes,
                )
            metrics = calculate_binary_metrics(
                probabilities=tuple(probabilities),
                outcomes=tuple(outcomes),
                priors=tuple(priors),
                probability_floor=specification.metrics.probability_floor,
                classification_threshold=specification.metrics.classification_threshold,
                ece_bin_count=specification.metrics.ece_bin_count,
            )
            rows.append(
                TrackerConditionMetrics(
                    condition=condition,
                    tracker_id=tracker_id,
                    episode_count=matrix.episodes_per_condition,
                    turn_count=len(probabilities),
                    non_missing_update_count=update_count,
                    metrics=metrics,
                )
            )
    return tuple(rows), episode_brier


def _primary_episode_effects(
    episode_brier: dict[tuple[StressCondition, TrackerId, int], float],
    *,
    episode_count: int,
    specification: TrackerStudyAnalysisSpecification,
) -> tuple[float, ...]:
    primary = specification.primary
    return tuple(
        fmean(
            episode_brier[(condition, primary.candidate_tracker, episode_index)]
            - episode_brier[(condition, primary.reference_tracker, episode_index)]
            for condition in primary.adverse_conditions
        )
        for episode_index in range(episode_count)
    )


def _clean_episode_effects(
    episode_brier: dict[tuple[StressCondition, TrackerId, int], float],
    *,
    episode_count: int,
    specification: TrackerStudyAnalysisSpecification,
) -> tuple[float, ...]:
    primary = specification.primary
    return tuple(
        episode_brier[(StressCondition.CLEAN, primary.candidate_tracker, episode_index)]
        - episode_brier[(StressCondition.CLEAN, primary.reference_tracker, episode_index)]
        for episode_index in range(episode_count)
    )


def _paired_inference(
    effects: tuple[float, ...],
    specification: TrackerStudyAnalysisSpecification,
) -> PairedEpisodeInference:
    interval = _paired_interval(effects, specification)
    p_value = _sign_swap_p_value(
        effects,
        repetitions=specification.primary.test_repetitions,
        seed=specification.primary.test_seed,
    )
    return PairedEpisodeInference(
        **interval.model_dump(),
        test_repetitions=specification.primary.test_repetitions,
        test_seed=specification.primary.test_seed,
        two_sided_p_value=p_value,
    )


def _paired_interval(
    effects: tuple[float, ...],
    specification: TrackerStudyAnalysisSpecification,
) -> PairedEpisodeInterval:
    if not effects or any(
        not math.isfinite(value) or not -1.0 <= value <= 1.0 for value in effects
    ):
        raise TrackerStudyAnalysisError("Episode effects must be finite values in [-1, 1]")
    primary = specification.primary
    generator = np.random.Generator(np.random.PCG64(primary.bootstrap_seed))
    values = np.asarray(effects, dtype=np.float64)
    indices = generator.integers(
        0,
        len(values),
        size=(primary.bootstrap_repetitions, len(values)),
    )
    bootstrap_means = np.mean(values[indices], axis=1)
    alpha = 1.0 - primary.confidence_level
    lower, upper = np.quantile(
        bootstrap_means,
        (alpha / 2.0, 1.0 - alpha / 2.0),
        method="linear",
    )
    return PairedEpisodeInterval(
        episode_count=len(effects),
        effects_hash=canonical_sha256(effects),
        mean_effect=fmean(effects),
        median_effect=median(effects),
        confidence_level=primary.confidence_level,
        bootstrap_repetitions=primary.bootstrap_repetitions,
        bootstrap_seed=primary.bootstrap_seed,
        interval_lower=float(lower),
        interval_upper=float(upper),
    )


def _sign_swap_p_value(
    effects: tuple[float, ...],
    *,
    repetitions: int,
    seed: int,
) -> float:
    observed = abs(fmean(effects))
    values = np.asarray(effects, dtype=np.float64)
    generator = np.random.Generator(np.random.PCG64(seed))
    exceedances = 0
    remaining = repetitions
    while remaining:
        batch_size = min(1000, remaining)
        signs = generator.integers(0, 2, size=(batch_size, len(values)), dtype=np.int8)
        signed_means = np.mean((2.0 * signs - 1.0) * values, axis=1)
        exceedances += int(np.count_nonzero(np.abs(signed_means) >= observed - 1e-15))
        remaining -= batch_size
    return (exceedances + 1) / (repetitions + 1)


def _brier_score(probabilities: list[float], outcomes: list[bool]) -> float:
    return fmean(
        (probability - float(outcome)) ** 2
        for probability, outcome in zip(probabilities, outcomes, strict=True)
    )


def _expected_calibration_error(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    *,
    bin_count: int,
) -> float:
    bin_indices = np.minimum((probabilities * bin_count).astype(np.int64), bin_count - 1)
    total = len(probabilities)
    error = 0.0
    for bin_index in range(bin_count):
        mask = bin_indices == bin_index
        count = int(np.count_nonzero(mask))
        if count == 0:
            continue
        error += (count / total) * abs(
            float(np.mean(probabilities[mask])) - float(np.mean(outcomes[mask]))
        )
    return error
