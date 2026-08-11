from __future__ import annotations

import math
from pathlib import Path
from statistics import NormalDist

import pytest

from socratic_tutor.acquisition_study import (
    AcquisitionStudyPlanError,
    EvaluationEnvironmentId,
    load_acquisition_study_plan,
)

ROOT = Path(__file__).parents[2]
ENVIRONMENTS = ROOT / "configs" / "acquisition-study" / "v1-environments.yaml"
ANALYSIS = ROOT / "configs" / "acquisition-study" / "v1-analysis.yaml"


def test_frozen_plan_separates_policy_inputs_and_held_out_worlds() -> None:
    environments, analysis = load_acquisition_study_plan(ENVIRONMENTS, ANALYSIS)

    assert analysis.primary.held_out_environments == tuple(EvaluationEnvironmentId)[1:]
    assert analysis.primary.exact_selected_probes_per_episode == 20
    assert environments.episodes.candidates_per_episode == 40
    assert not set(analysis.policy_information_allowed) & set(analysis.policy_information_forbidden)
    assert "evaluation_reliability_parameters" in analysis.policy_information_forbidden
    assert "latent_outcome" in analysis.policy_information_forbidden
    assert not analysis.claims.human_learning_claim_allowed
    assert not analysis.claims.deployed_cost_saving_claim_allowed


def test_episode_count_meets_the_declared_family_cluster_precision_target() -> None:
    environments, analysis = load_acquisition_study_plan(ENVIRONMENTS, ANALYSIS)

    alpha_per_comparison = analysis.primary.familywise_alpha / 3.0
    critical_value = NormalDist().inv_cdf(1.0 - alpha_per_comparison / 2.0)
    worst_case_episode_sd = math.sqrt(1.0 / environments.episodes.families_per_episode)
    half_width = (
        critical_value
        * worst_case_episode_sd
        / math.sqrt(environments.episodes.evaluation_episodes_per_environment)
    )

    assert half_width <= analysis.primary.precision_target_half_width


def test_environment_hash_rejects_a_changed_failure_setting(tmp_path: Path) -> None:
    changed = ENVIRONMENTS.read_text(encoding="utf-8").replace(
        "probability: 0.30",
        "probability: 0.25",
    )
    changed_path = tmp_path / "changed-environments.yaml"
    changed_path.write_text(changed, encoding="utf-8")

    with pytest.raises(AcquisitionStudyPlanError):
        load_acquisition_study_plan(changed_path, ANALYSIS)
