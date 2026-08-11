"""Tests for the scientific boundary around acquisition-policy inputs."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from socratic_tutor.acquisition_study import (
    AcquisitionCandidate,
    AcquisitionContractError,
    AcquisitionDecision,
    AcquisitionRequest,
    BetaPosterior,
    CalibrationBetaPosterior,
    PolicyId,
    ProbeClassId,
)


def _candidate(case_id: str) -> AcquisitionCandidate:
    return AcquisitionCandidate(
        case_id=case_id,
        prior_success_probability=0.45,
        probe_class=ProbeClassId.HIGH_RELIABILITY_DENSE,
        calibration_beta_posterior=CalibrationBetaPosterior(
            sensitivity=BetaPosterior(alpha=145.0, beta=17.0),
            specificity=BetaPosterior(alpha=149.0, beta=13.0),
        ),
    )


@pytest.mark.parametrize(
    ("forbidden_field", "forbidden_value"),
    [
        ("family_id", "family-01"),
        ("evaluation_environment_id", "matched"),
        ("latent_outcome", True),
        ("unrequested_probe_outcome", "pass"),
        ("criterion_outcome", True),
    ],
)
def test_candidate_rejects_simulator_owned_information(
    forbidden_field: str,
    forbidden_value: object,
) -> None:
    payload = _candidate("case-01").model_dump(mode="json")
    payload[forbidden_field] = forbidden_value

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        AcquisitionCandidate.model_validate(payload)


def test_policy_input_fields_match_the_frozen_information_boundary() -> None:
    flattened_fields = set(AcquisitionCandidate.model_fields) | {"remaining_probe_budget"}

    assert flattened_fields == {
        "case_id",
        "prior_success_probability",
        "probe_class",
        "calibration_beta_posterior",
        "remaining_probe_budget",
    }


def test_decision_must_use_visible_cases_and_the_exact_budget() -> None:
    request = AcquisitionRequest(
        candidates=(_candidate("case-01"), _candidate("case-02")),
        remaining_probe_budget=1,
    )
    valid = AcquisitionDecision(
        policy_id=PolicyId.UNCERTAINTY_ONLY_BOUNDED,
        selected_case_ids=("case-02",),
    )

    assert valid.checked_against(request) is valid

    with pytest.raises(AcquisitionContractError, match="count must equal"):
        AcquisitionDecision(
            policy_id=PolicyId.NEVER_PROBE,
            selected_case_ids=(),
        ).checked_against(request)

    with pytest.raises(AcquisitionContractError, match="unknown case IDs: hidden-case"):
        AcquisitionDecision(
            policy_id=PolicyId.UNCERTAINTY_ONLY_BOUNDED,
            selected_case_ids=("hidden-case",),
        ).checked_against(request)
