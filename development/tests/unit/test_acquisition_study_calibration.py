"""Checks for pre-evaluation probe-reliability calibration."""

from __future__ import annotations

from pathlib import Path

import pytest

from socratic_tutor.acquisition_study import (
    CalibrationError,
    ProbeClassId,
    estimate_probe_reliability,
    load_acquisition_study_plan,
)

ROOT = Path(__file__).parents[2]
ENVIRONMENTS = ROOT / "configs" / "acquisition-study" / "v1-environments.yaml"
ANALYSIS = ROOT / "configs" / "acquisition-study" / "v1-analysis.yaml"


def test_calibration_is_repeatable_and_preserves_declared_sample_sizes() -> None:
    specification, _ = load_acquisition_study_plan(ENVIRONMENTS, ANALYSIS)

    first = estimate_probe_reliability(specification)
    second = estimate_probe_reliability(specification)

    assert first == second
    for design, estimate in zip(
        specification.calibration.probe_classes, first.estimates, strict=True
    ):
        assert estimate.sensitivity_count.trials == design.mastered_samples
        assert estimate.specificity_count.trials == design.non_mastered_samples
        assert (
            estimate.posterior.sensitivity.alpha + estimate.posterior.sensitivity.beta
            == design.mastered_samples
            + specification.calibration.beta_prior_alpha
            + specification.calibration.beta_prior_beta
        )
        assert (
            estimate.posterior.specificity.alpha + estimate.posterior.specificity.beta
            == design.non_mastered_samples
            + specification.calibration.beta_prior_alpha
            + specification.calibration.beta_prior_beta
        )

    dense = first.posterior_for(ProbeClassId.HIGH_RELIABILITY_DENSE)
    sparse = first.posterior_for(ProbeClassId.HIGH_RELIABILITY_SPARSE)
    assert dense.sensitivity.alpha + dense.sensitivity.beta > (
        sparse.sensitivity.alpha + sparse.sensitivity.beta
    )


def test_calibration_rejects_a_seed_chosen_after_the_design_freeze() -> None:
    specification, _ = load_acquisition_study_plan(ENVIRONMENTS, ANALYSIS)

    with pytest.raises(CalibrationError, match="not declared"):
        estimate_probe_reliability(specification, seed=123456789)
