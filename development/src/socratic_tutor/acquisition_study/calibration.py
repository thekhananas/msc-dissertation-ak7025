"""Calibration-only estimation of executable-probe reliability."""

from __future__ import annotations

import math
from typing import Literal, Self

import numpy as np
from pydantic import Field, model_validator

from socratic_tutor.acquisition_study.contracts import (
    BetaPosterior,
    CalibrationBetaPosterior,
    ProbeClassId,
)
from socratic_tutor.acquisition_study.plan import AcquisitionEnvironmentSpecification
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import canonical_sha256, model_content_hash
from socratic_tutor.contracts import ContractModel


class CalibrationError(ValueError):
    """A reliability calibration request violates the frozen study design."""


class BernoulliCalibrationCount(ContractModel):
    """Observed successes and trials for one reliability parameter."""

    successes: int = Field(ge=0)
    trials: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_successes(self) -> Self:
        if self.successes > self.trials:
            raise ValueError("Calibration successes cannot exceed trials")
        return self


class ProbeClassCalibrationEstimate(ContractModel):
    """Calibration counts and posterior supplied for one probe class."""

    probe_class: ProbeClassId
    sensitivity_count: BernoulliCalibrationCount
    specificity_count: BernoulliCalibrationCount
    posterior: CalibrationBetaPosterior


class ReliabilityCalibrationRun(ContractModel):
    """Immutable reliability estimates produced before evaluation."""

    schema_version: Literal[1] = 1
    schema_id: Literal["acquisition_study.reliability_calibration.v1"] = (
        "acquisition_study.reliability_calibration.v1"
    )
    study_id: Literal["reliability-aware-probing-v1"] = "reliability-aware-probing-v1"
    environment_specification_hash: Sha256
    calibration_environment_id: Literal["calibration_assumed_model"] = "calibration_assumed_model"
    calibration_seed: int = Field(ge=0)
    random_generator: Literal["numpy.PCG64.SeedSequence"] = "numpy.PCG64.SeedSequence"
    beta_prior_alpha: float = Field(gt=0.0, allow_inf_nan=False)
    beta_prior_beta: float = Field(gt=0.0, allow_inf_nan=False)
    estimates: tuple[ProbeClassCalibrationEstimate, ...] = Field(min_length=4, max_length=4)
    calibration_hash: Sha256

    @model_validator(mode="after")
    def validate_run(self) -> Self:
        if tuple(estimate.probe_class for estimate in self.estimates) != tuple(ProbeClassId):
            raise ValueError(
                "Calibration estimates must contain each probe class once and in order"
            )
        for estimate in self.estimates:
            expected_sensitivity_alpha = (
                self.beta_prior_alpha + estimate.sensitivity_count.successes
            )
            expected_sensitivity_beta = self.beta_prior_beta + (
                estimate.sensitivity_count.trials - estimate.sensitivity_count.successes
            )
            expected_specificity_alpha = (
                self.beta_prior_alpha + estimate.specificity_count.successes
            )
            expected_specificity_beta = self.beta_prior_beta + (
                estimate.specificity_count.trials - estimate.specificity_count.successes
            )
            posterior = estimate.posterior
            if not (
                math.isclose(posterior.sensitivity.alpha, expected_sensitivity_alpha)
                and math.isclose(posterior.sensitivity.beta, expected_sensitivity_beta)
                and math.isclose(posterior.specificity.alpha, expected_specificity_alpha)
                and math.isclose(posterior.specificity.beta, expected_specificity_beta)
            ):
                raise ValueError("Calibration posterior does not match its prior and counts")
        if self.calibration_hash != model_content_hash(self, exclude={"calibration_hash"}):
            raise ValueError("Calibration hash does not match its content")
        return self

    def posterior_for(self, probe_class: ProbeClassId) -> CalibrationBetaPosterior:
        """Return the policy-safe posterior for one probe class."""

        for estimate in self.estimates:
            if estimate.probe_class is probe_class:
                return estimate.posterior
        raise KeyError(probe_class)


def estimate_probe_reliability(
    specification: AcquisitionEnvironmentSpecification,
    *,
    seed: int | None = None,
) -> ReliabilityCalibrationRun:
    """Estimate reliability using only a predeclared calibration seed."""

    design = specification.calibration
    calibration_seed = design.seed if seed is None else seed
    allowed_seeds = {design.seed, *design.sensitivity_seeds}
    if calibration_seed not in allowed_seeds:
        raise CalibrationError("Calibration seed was not declared in the frozen design")

    estimates: list[ProbeClassCalibrationEstimate] = []
    for probe_index, probe_design in enumerate(design.probe_classes):
        sensitivity_successes = _sample_successes(
            seed=calibration_seed,
            probe_index=probe_index,
            parameter_index=0,
            trials=probe_design.mastered_samples,
            probability=probe_design.sensitivity,
        )
        specificity_successes = _sample_successes(
            seed=calibration_seed,
            probe_index=probe_index,
            parameter_index=1,
            trials=probe_design.non_mastered_samples,
            probability=probe_design.specificity,
        )
        estimates.append(
            ProbeClassCalibrationEstimate(
                probe_class=probe_design.probe_class,
                sensitivity_count=BernoulliCalibrationCount(
                    successes=sensitivity_successes,
                    trials=probe_design.mastered_samples,
                ),
                specificity_count=BernoulliCalibrationCount(
                    successes=specificity_successes,
                    trials=probe_design.non_mastered_samples,
                ),
                posterior=CalibrationBetaPosterior(
                    sensitivity=BetaPosterior(
                        alpha=design.beta_prior_alpha + sensitivity_successes,
                        beta=design.beta_prior_beta
                        + probe_design.mastered_samples
                        - sensitivity_successes,
                    ),
                    specificity=BetaPosterior(
                        alpha=design.beta_prior_alpha + specificity_successes,
                        beta=design.beta_prior_beta
                        + probe_design.non_mastered_samples
                        - specificity_successes,
                    ),
                ),
            )
        )

    payload = {
        "schema_version": 1,
        "schema_id": "acquisition_study.reliability_calibration.v1",
        "study_id": specification.study_id,
        "environment_specification_hash": specification.specification_hash,
        "calibration_environment_id": design.environment_id,
        "calibration_seed": calibration_seed,
        "random_generator": "numpy.PCG64.SeedSequence",
        "beta_prior_alpha": design.beta_prior_alpha,
        "beta_prior_beta": design.beta_prior_beta,
        "estimates": estimates,
    }
    return ReliabilityCalibrationRun.model_validate(
        {**payload, "calibration_hash": canonical_sha256(payload)}
    )


def _sample_successes(
    *,
    seed: int,
    probe_index: int,
    parameter_index: int,
    trials: int,
    probability: float,
) -> int:
    seed_sequence = np.random.SeedSequence([seed, probe_index, parameter_index])
    generator = np.random.Generator(np.random.PCG64(seed_sequence))
    return int(generator.binomial(trials, probability))
