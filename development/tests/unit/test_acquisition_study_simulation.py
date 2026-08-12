"""Scientific checks for safe and privileged simulator projections."""

from __future__ import annotations

import json
import math
from pathlib import Path

from socratic_tutor.acquisition_study import (
    EvaluationEnvironmentId,
    ProbeClassId,
    SimulatedProbeOutcome,
    estimate_probe_reliability,
    generate_acquisition_episode,
    load_acquisition_study_plan,
)

ROOT = Path(__file__).parents[2]
ENVIRONMENTS = ROOT / "configs" / "acquisition-study" / "v1-environments.yaml"
ANALYSIS = ROOT / "configs" / "acquisition-study" / "v1-analysis.yaml"
SPECIFICATION, ANALYSIS_PLAN = load_acquisition_study_plan(ENVIRONMENTS, ANALYSIS)
CALIBRATION = estimate_probe_reliability(SPECIFICATION)


def _episode(environment_id: EvaluationEnvironmentId):
    return generate_acquisition_episode(
        SPECIFICATION,
        CALIBRATION,
        environment_id=environment_id,
        episode_index=7,
        probe_budget=ANALYSIS_PLAN.primary.exact_selected_probes_per_episode,
    )


def test_episode_is_repeatable_and_policy_projection_contains_no_world_truth() -> None:
    first_policy, first_truth = _episode(EvaluationEnvironmentId.MATCHED)
    second_policy, second_truth = _episode(EvaluationEnvironmentId.MATCHED)

    assert first_policy == second_policy
    assert first_truth == second_truth
    assert len(first_policy.request.candidates) == 40
    assert first_policy.request.remaining_probe_budget == 20

    serialised_policy_record = json.dumps(first_policy.model_dump(mode="json"), sort_keys=True)
    for forbidden_name in (
        "environment_id",
        "family_id",
        "latent_success",
        "observed_probe_outcome",
        "effective_sensitivity",
        "effective_specificity",
    ):
        assert forbidden_name not in serialised_policy_record


def test_held_out_worlds_apply_the_frozen_failure_rules() -> None:
    _, matched = _episode(EvaluationEnvironmentId.MATCHED)
    assert all(not case.case_inversion_applied for case in matched.cases)
    assert all(not case.family_inversion_applied for case in matched.cases)
    assert all(
        case.observed_probe_outcome is not SimulatedProbeOutcome.MISSING for case in matched.cases
    )
    assert all(
        math.isclose(case.effective_sensitivity, case.base_sensitivity)
        and math.isclose(case.effective_specificity, case.base_specificity)
        for case in matched.cases
    )

    _, irrelevant = _episode(EvaluationEnvironmentId.IRRELEVANT_EVIDENCE)
    assert all(math.isclose(case.effective_sensitivity, 0.5) for case in irrelevant.cases)
    assert all(math.isclose(case.effective_specificity, 0.5) for case in irrelevant.cases)

    _, inverted = _episode(EvaluationEnvironmentId.INVERTED_EVIDENCE)
    assert all(case.case_inversion_applied for case in inverted.cases)

    _, missing = _episode(EvaluationEnvironmentId.DIFFICULTY_MISSINGNESS)
    for case in missing.cases:
        difficulty = 1.0 - 2.0 * abs(case.prior_success_probability - 0.5)
        expected = min(0.60, 0.05 + 0.55 * difficulty)
        assert math.isclose(case.missing_probability, expected, abs_tol=1e-12)

    _, correlated = _episode(EvaluationEnvironmentId.CORRELATED_FAMILY_FAILURES)
    family_faults: dict[str, set[bool]] = {}
    family_classes: dict[str, set[ProbeClassId]] = {}
    for case in correlated.cases:
        family_faults.setdefault(case.family_id, set()).add(case.family_inversion_applied)
        family_classes.setdefault(case.family_id, set()).add(case.probe_class)
    assert all(len(values) == 1 for values in family_faults.values())
    assert all(values == set(ProbeClassId) for values in family_classes.values())
