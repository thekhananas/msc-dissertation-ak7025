"""Deterministic glass-box episodes with policy-safe and privileged records."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self

import numpy as np
from pydantic import Field, model_validator

from socratic_tutor.acquisition_study.calibration import ReliabilityCalibrationRun
from socratic_tutor.acquisition_study.contracts import (
    AcquisitionCandidate,
    AcquisitionRequest,
    ProbeClassId,
)
from socratic_tutor.acquisition_study.plan import (
    AcquisitionEnvironmentSpecification,
    EvaluationEnvironment,
    EvaluationEnvironmentId,
)
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.contracts import ContractModel


class AcquisitionSimulationError(ValueError):
    """An episode request conflicts with the frozen simulator design."""


class SimulatedProbeOutcome(StrEnum):
    """Observable result of one requested simulated probe."""

    PASS = "pass"
    FAIL = "fail"
    MISSING = "missing"


class PolicyEpisodeRecord(ContractModel):
    """Episode metadata and request that contain no evaluation-world truth."""

    schema_version: Literal[1] = 1
    schema_id: Literal["acquisition_study.policy_episode.v1"] = (
        "acquisition_study.policy_episode.v1"
    )
    study_id: Literal["reliability-aware-probing-v1"] = "reliability-aware-probing-v1"
    episode_id: str = Field(min_length=1)
    environment_specification_hash: Sha256
    calibration_hash: Sha256
    request: AcquisitionRequest


class SimulatedCaseTruth(ContractModel):
    """Privileged generating values and outcomes for one simulated case."""

    case_id: str = Field(min_length=1)
    family_id: str = Field(min_length=1)
    probe_class: ProbeClassId
    prior_success_probability: float = Field(gt=0.0, lt=1.0, allow_inf_nan=False)
    latent_success: bool
    base_sensitivity: float = Field(gt=0.0, lt=1.0, allow_inf_nan=False)
    base_specificity: float = Field(gt=0.0, lt=1.0, allow_inf_nan=False)
    effective_sensitivity: float = Field(gt=0.0, lt=1.0, allow_inf_nan=False)
    effective_specificity: float = Field(gt=0.0, lt=1.0, allow_inf_nan=False)
    raw_probe_passed: bool
    case_inversion_applied: bool
    family_inversion_applied: bool
    probe_passed_before_missingness: bool
    missing_probability: float = Field(ge=0.0, lt=1.0, allow_inf_nan=False)
    observed_probe_outcome: SimulatedProbeOutcome


class PrivilegedEpisodeRecord(ContractModel):
    """Restricted simulator truth that must never be passed to an ordinary policy."""

    schema_version: Literal[1] = 1
    schema_id: Literal["acquisition_study.privileged_episode.v1"] = (
        "acquisition_study.privileged_episode.v1"
    )
    study_id: Literal["reliability-aware-probing-v1"] = "reliability-aware-probing-v1"
    episode_id: str = Field(min_length=1)
    environment_id: EvaluationEnvironmentId
    environment_specification_hash: Sha256
    cases: tuple[SimulatedCaseTruth, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_case_layout(self) -> Self:
        case_ids = tuple(case.case_id for case in self.cases)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Privileged case IDs must be unique")

        classes_by_family: dict[str, list[ProbeClassId]] = {}
        for case in self.cases:
            classes_by_family.setdefault(case.family_id, []).append(case.probe_class)
        expected_classes = set(ProbeClassId)
        if any(
            len(classes) != len(expected_classes) or set(classes) != expected_classes
            for classes in classes_by_family.values()
        ):
            raise ValueError("Every family must contain each probe class exactly once")
        return self


def generate_acquisition_episode(
    specification: AcquisitionEnvironmentSpecification,
    calibration: ReliabilityCalibrationRun,
    *,
    environment_id: EvaluationEnvironmentId,
    episode_index: int,
    probe_budget: int,
) -> tuple[PolicyEpisodeRecord, PrivilegedEpisodeRecord]:
    """Generate one matched episode and return its safe and restricted projections."""

    if episode_index < 0:
        raise AcquisitionSimulationError("Episode index must be non-negative")
    if calibration.environment_specification_hash != specification.specification_hash:
        raise AcquisitionSimulationError("Calibration belongs to another environment specification")

    environment_index, environment = _environment(specification, environment_id)
    episode_id = f"episode-{episode_index:06d}"
    family_faults = _family_faults(specification, environment, episode_index)
    latent_rng = _rng(
        specification.episodes.latent_evaluation_seed,
        environment_index,
        episode_index,
    )
    evidence_rng = _rng(environment.evidence_seed, episode_index, 0)
    case_inversion_rng = _rng(environment.evidence_seed, episode_index, 1)
    missingness_rng = _rng(environment.evidence_seed, episode_index, 2)

    layout = [
        (family_index, probe_class)
        for family_index in range(specification.episodes.families_per_episode)
        for probe_class in ProbeClassId
    ]
    ordered_layout = [layout[int(index)] for index in latent_rng.permutation(len(layout))]
    calibration_by_class = {
        item.probe_class: item for item in specification.calibration.probe_classes
    }
    candidates: list[AcquisitionCandidate] = []
    truths: list[SimulatedCaseTruth] = []

    for case_index, (family_index, probe_class) in enumerate(ordered_layout):
        case_id = f"{episode_id}:case-{case_index:02d}"
        prior = float(
            np.clip(
                latent_rng.beta(
                    specification.episodes.prior_beta_alpha,
                    specification.episodes.prior_beta_beta,
                ),
                specification.episodes.prior_probability_floor,
                1.0 - specification.episodes.prior_probability_floor,
            )
        )
        latent_success = bool(latent_rng.random() < prior)
        base = calibration_by_class[probe_class]
        sensitivity = _scale_from_chance(
            base.sensitivity,
            environment.reliability.sensitivity_scale_from_chance,
        )
        specificity = _scale_from_chance(
            base.specificity,
            environment.reliability.specificity_scale_from_chance,
        )
        pass_probability = sensitivity if latent_success else 1.0 - specificity
        raw_probe_passed = bool(evidence_rng.random() < pass_probability)
        case_inversion = bool(
            case_inversion_rng.random() < environment.reliability.outcome_inversion_probability
        )
        family_inversion = family_faults[family_index]
        probe_passed = raw_probe_passed ^ case_inversion ^ family_inversion
        missing_probability = _missing_probability(environment, prior)
        is_missing = bool(missingness_rng.random() < missing_probability)
        observed_outcome = (
            SimulatedProbeOutcome.MISSING
            if is_missing
            else SimulatedProbeOutcome.PASS
            if probe_passed
            else SimulatedProbeOutcome.FAIL
        )

        candidates.append(
            AcquisitionCandidate(
                case_id=case_id,
                prior_success_probability=prior,
                probe_class=probe_class,
                calibration_beta_posterior=calibration.posterior_for(probe_class),
            )
        )
        truths.append(
            SimulatedCaseTruth(
                case_id=case_id,
                family_id=f"family-{family_index:02d}",
                probe_class=probe_class,
                prior_success_probability=prior,
                latent_success=latent_success,
                base_sensitivity=base.sensitivity,
                base_specificity=base.specificity,
                effective_sensitivity=sensitivity,
                effective_specificity=specificity,
                raw_probe_passed=raw_probe_passed,
                case_inversion_applied=case_inversion,
                family_inversion_applied=family_inversion,
                probe_passed_before_missingness=probe_passed,
                missing_probability=missing_probability,
                observed_probe_outcome=observed_outcome,
            )
        )

    request = AcquisitionRequest(
        candidates=tuple(candidates),
        remaining_probe_budget=probe_budget,
    )
    policy_record = PolicyEpisodeRecord(
        episode_id=episode_id,
        environment_specification_hash=specification.specification_hash,
        calibration_hash=calibration.calibration_hash,
        request=request,
    )
    privileged_record = PrivilegedEpisodeRecord(
        episode_id=episode_id,
        environment_id=environment_id,
        environment_specification_hash=specification.specification_hash,
        cases=tuple(truths),
    )
    _validate_projections(policy_record, privileged_record)
    return policy_record, privileged_record


def _environment(
    specification: AcquisitionEnvironmentSpecification,
    environment_id: EvaluationEnvironmentId,
) -> tuple[int, EvaluationEnvironment]:
    for index, environment in enumerate(specification.evaluation_environments):
        if environment.environment_id is environment_id:
            return index, environment
    raise AcquisitionSimulationError(f"Unknown evaluation environment: {environment_id}")


def _family_faults(
    specification: AcquisitionEnvironmentSpecification,
    environment: EvaluationEnvironment,
    episode_index: int,
) -> tuple[bool, ...]:
    generator = _rng(environment.evidence_seed, episode_index, 3)
    return tuple(
        bool(generator.random() < environment.family_fault.probability)
        for _ in range(specification.episodes.families_per_episode)
    )


def _rng(*entropy: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))


def _scale_from_chance(base: float, scale: float) -> float:
    return 0.5 + scale * (base - 0.5)


def _missing_probability(environment: EvaluationEnvironment, prior: float) -> float:
    difficulty = 1.0 - 2.0 * abs(prior - 0.5)
    return min(
        environment.missingness.maximum_probability,
        environment.missingness.base_probability
        + environment.missingness.prior_difficulty_slope * difficulty,
    )


def _validate_projections(
    policy_record: PolicyEpisodeRecord,
    privileged_record: PrivilegedEpisodeRecord,
) -> None:
    if policy_record.episode_id != privileged_record.episode_id:
        raise AcquisitionSimulationError("Safe and privileged episode IDs differ")
    candidates = {candidate.case_id: candidate for candidate in policy_record.request.candidates}
    truths = {case.case_id: case for case in privileged_record.cases}
    if candidates.keys() != truths.keys():
        raise AcquisitionSimulationError("Safe and privileged cases differ")
    for case_id, candidate in candidates.items():
        truth = truths[case_id]
        if (
            candidate.probe_class is not truth.probe_class
            or candidate.prior_success_probability != truth.prior_success_probability
        ):
            raise AcquisitionSimulationError("Safe and privileged case projections disagree")
