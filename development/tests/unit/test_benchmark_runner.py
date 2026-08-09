"""Paired benchmark condition execution and parity tests."""

from math import isclose
from pathlib import Path
from typing import TypedDict

import pytest
from pydantic import ValidationError

from socratic_tutor.benchmark.evaluator.loader import load_and_verify_manifest
from socratic_tutor.benchmark.evaluator.projection import project_public_manifest
from socratic_tutor.benchmark.public import (
    EXPECTED_CONDITIONS,
    BenchmarkCondition,
    ConditionRunError,
    ControlEvidenceRecord,
    ControlFactory,
    PairedConditionRun,
    PairedConditionRunner,
    ProbeExecutionSummary,
    find_public_payload_violations,
)
from socratic_tutor.contracts import (
    Evidence,
    EvidenceCategory,
    PolicyDecision,
    TrackerState,
)
from socratic_tutor.policies import choose_action
from socratic_tutor.tracking import update_tracker

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = WORKSPACE_ROOT / "data" / "benchmarks" / "dev-v0"
MANIFEST_PATH = BENCHMARK_ROOT / "manifest.yaml"


class RunCaseInputs(TypedDict):
    case_id: str
    initial_tracker_state: TrackerState
    public_evidence: Evidence
    probe_evidence: ProbeExecutionSummary
    unrelated_evidence: ControlEvidenceRecord
    corrupted_evidence: ControlEvidenceRecord


def paired_inputs() -> tuple[PairedConditionRunner, RunCaseInputs]:
    authored = load_and_verify_manifest(MANIFEST_PATH)
    public = project_public_manifest(authored)
    controls = ControlFactory(BENCHMARK_ROOT, public)
    aliasing = controls.probe_summary(case_id="dev-aliasing-001", passed=0, failed=2)
    none_falsy = controls.probe_summary(case_id="dev-none-falsy-001", passed=2, failed=0)
    aliasing_case = next(case for case in public.cases if case.case_id == aliasing.case_id)
    evidence_by_case = {
        aliasing.case_id: aliasing,
        none_falsy.case_id: none_falsy,
    }
    runner = PairedConditionRunner(
        public,
        tracker_update=update_tracker,
        policy_decision=choose_action,
        tracker_version="simple-v1",
        policy_version="heuristic-v1",
    )
    inputs: RunCaseInputs = {
        "case_id": aliasing.case_id,
        "initial_tracker_state": TrackerState(
            concept=aliasing_case.target_concept,
            mastery_probability=0.5,
            observations=0,
        ),
        "public_evidence": Evidence(
            category=EvidenceCategory.CORRECT,
            confidence=0.8,
            rationale="The public explanation appears correct.",
        ),
        "probe_evidence": aliasing,
        "unrelated_evidence": controls.unrelated(
            case_id=aliasing.case_id,
            evidence_by_case=evidence_by_case,
        ),
        "corrupted_evidence": controls.corrupted(
            case_id=aliasing.case_id,
            evidence=aliasing,
        ),
    }
    return runner, inputs


def test_all_conditions_clone_one_post_public_state() -> None:
    authored = load_and_verify_manifest(MANIFEST_PATH)
    public = project_public_manifest(authored)
    controls = ControlFactory(BENCHMARK_ROOT, public)
    aliasing = controls.probe_summary(case_id="dev-aliasing-001", passed=0, failed=2)
    none_falsy = controls.probe_summary(case_id="dev-none-falsy-001", passed=2, failed=0)
    aliasing_case = next(case for case in public.cases if case.case_id == aliasing.case_id)
    tracker_inputs: list[TrackerState] = []

    def tracking_spy(state: TrackerState, evidence: Evidence) -> TrackerState:
        tracker_inputs.append(state)
        return update_tracker(state, evidence)

    runner = PairedConditionRunner(
        public,
        tracker_update=tracking_spy,
        policy_decision=choose_action,
        tracker_version="simple-v1",
        policy_version="heuristic-v1",
    )
    initial = TrackerState(
        concept=aliasing_case.target_concept,
        mastery_probability=0.5,
        observations=0,
    )
    result = runner.run_case(
        case_id=aliasing.case_id,
        initial_tracker_state=initial,
        public_evidence=Evidence(
            category=EvidenceCategory.CORRECT,
            confidence=0.8,
            rationale="The public explanation appears correct.",
        ),
        probe_evidence=aliasing,
        unrelated_evidence=controls.unrelated(
            case_id=aliasing.case_id,
            evidence_by_case={aliasing.case_id: aliasing, none_falsy.case_id: none_falsy},
        ),
        corrupted_evidence=controls.corrupted(
            case_id=aliasing.case_id,
            evidence=aliasing,
        ),
    )

    assert initial.mastery_probability == 0.5
    assert initial.observations == 0
    assert isclose(result.post_public_state.mastery_probability, 0.7)
    assert result.post_public_state.observations == 1
    assert tuple(outcome.condition for outcome in result.outcomes) == EXPECTED_CONDITIONS
    assert {outcome.initial_state_hash for outcome in result.outcomes} == {
        result.initial_state_hash
    }
    assert {outcome.tracker_version for outcome in result.outcomes} == {"simple-v1"}
    assert {outcome.policy_version for outcome in result.outcomes} == {"heuristic-v1"}
    assert all(outcome.tracker_before == result.post_public_state for outcome in result.outcomes)
    assert len({id(outcome.tracker_before) for outcome in result.outcomes}) == 4
    assert len(tracker_inputs) == 4
    assert tracker_inputs[0] is initial
    assert all(state == result.post_public_state for state in tracker_inputs[1:])
    assert len({id(state) for state in tracker_inputs[1:]}) == 3

    outcomes = {outcome.condition: outcome for outcome in result.outcomes}
    probe_evidence = outcomes[BenchmarkCondition.PROBE_INFORMED].additional_evidence
    assert probe_evidence is not None
    assert probe_evidence.category is EvidenceCategory.INCORRECT
    assert isclose(
        outcomes[BenchmarkCondition.DIALOGUE_ONLY].tracker_after.mastery_probability,
        0.7,
    )
    assert isclose(
        outcomes[BenchmarkCondition.PROBE_INFORMED].tracker_after.mastery_probability,
        0.5,
    )
    assert isclose(
        outcomes[BenchmarkCondition.UNRELATED_PROBE].tracker_after.mastery_probability,
        0.9,
    )
    assert isclose(
        outcomes[BenchmarkCondition.CORRUPTED_PROBE].tracker_after.mastery_probability,
        0.9,
    )
    assert len({outcome.input_hash for outcome in result.outcomes}) == 4
    assert find_public_payload_violations(result.model_dump(mode="json")) == ()


def test_runner_uses_one_policy_callable_for_every_condition() -> None:
    authored = load_and_verify_manifest(MANIFEST_PATH)
    public = project_public_manifest(authored)
    controls = ControlFactory(BENCHMARK_ROOT, public)
    aliasing = controls.probe_summary(case_id="dev-aliasing-001", passed=0, failed=2)
    none_falsy = controls.probe_summary(case_id="dev-none-falsy-001", passed=2, failed=0)
    aliasing_case = next(case for case in public.cases if case.case_id == aliasing.case_id)
    policy_inputs: list[Evidence] = []

    def policy_spy(evidence: Evidence) -> PolicyDecision:
        policy_inputs.append(evidence)
        return choose_action(evidence)

    runner = PairedConditionRunner(
        public,
        tracker_update=update_tracker,
        policy_decision=policy_spy,
        tracker_version="simple-v1",
        policy_version="heuristic-v1",
    )
    runner.run_case(
        case_id=aliasing.case_id,
        initial_tracker_state=TrackerState(
            concept=aliasing_case.target_concept,
            mastery_probability=0.5,
            observations=0,
        ),
        public_evidence=Evidence(
            category=EvidenceCategory.CORRECT,
            confidence=0.8,
            rationale="The public explanation appears correct.",
        ),
        probe_evidence=aliasing,
        unrelated_evidence=controls.unrelated(
            case_id=aliasing.case_id,
            evidence_by_case={aliasing.case_id: aliasing, none_falsy.case_id: none_falsy},
        ),
        corrupted_evidence=controls.corrupted(case_id=aliasing.case_id, evidence=aliasing),
    )

    assert len(policy_inputs) == 4


def test_runner_rejects_control_from_another_target_case() -> None:
    runner, inputs = paired_inputs()
    unrelated = inputs["unrelated_evidence"]
    inputs["unrelated_evidence"] = unrelated.model_copy(
        update={"target_case_id": "dev-none-falsy-001"}
    )

    with pytest.raises(ConditionRunError, match="Unrelated evidence"):
        runner.run_case(**inputs)


def test_paired_run_rejects_state_or_version_parity_tampering() -> None:
    runner, inputs = paired_inputs()
    result = runner.run_case(**inputs)

    changed_version = result.model_dump(mode="json")
    changed_version["tracker_version"] = "different-tracker-v1"
    with pytest.raises(ValidationError, match="different tracker version"):
        PairedConditionRun.model_validate(changed_version)

    missing_condition = result.model_dump(mode="json")
    missing_condition["outcomes"].pop()
    with pytest.raises(ValidationError, match="complete ordered condition set"):
        PairedConditionRun.model_validate(missing_condition)

    changed_state = result.model_dump(mode="json")
    changed_state["outcomes"][1]["initial_state_hash"] = "0" * 64
    with pytest.raises(ValidationError, match="shared initial hash"):
        PairedConditionRun.model_validate(changed_state)


@pytest.mark.parametrize("version_field", ["tracker_version", "policy_version"])
def test_runner_requires_explicit_component_versions(version_field: str) -> None:
    authored = load_and_verify_manifest(MANIFEST_PATH)
    public = project_public_manifest(authored)
    tracker_version = "" if version_field == "tracker_version" else "simple-v1"
    policy_version = "" if version_field == "policy_version" else "heuristic-v1"

    with pytest.raises(ConditionRunError, match="versions must be non-empty"):
        PairedConditionRunner(
            public,
            tracker_update=update_tracker,
            policy_decision=choose_action,
            tracker_version=tracker_version,
            policy_version=policy_version,
        )
