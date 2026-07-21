"""Pure paired-condition execution over one cloned post-public tracker state."""

from collections.abc import Callable
from typing import Literal

from pydantic import Field, model_validator

from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import canonical_sha256, model_content_hash
from socratic_tutor.benchmark.public.controls import (
    ControlEvidenceRecord,
    ProbeExecutionSummary,
)
from socratic_tutor.benchmark.public.models import (
    EXPECTED_CONDITIONS,
    BenchmarkCaseView,
    BenchmarkCondition,
    PublicArtifactClass,
    PublicBenchmarkManifest,
)
from socratic_tutor.benchmark.public.safety import assert_public_payload_safe
from socratic_tutor.contracts import (
    ContractModel,
    Evidence,
    EvidenceCategory,
    PolicyDecision,
    TrackerState,
)

type TrackerUpdate = Callable[[TrackerState, Evidence], TrackerState]
type PolicyDecisionFunction = Callable[[Evidence], PolicyDecision]


class ConditionRunError(ValueError):
    """A paired condition cannot be executed without violating parity."""


class ConditionOutcome(ContractModel):
    """In-memory decision result produced before persistence or outcome reveal."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.condition_outcome.v1"] = "benchmark.condition_outcome.v1"
    case_id: str = Field(min_length=1)
    case_content_hash: Sha256
    condition: BenchmarkCondition
    public_evidence_hash: Sha256
    tracker_before: TrackerState
    initial_state_hash: Sha256
    additional_evidence: Evidence | None
    evidence_source_hash: Sha256 | None
    tracker_after: TrackerState
    decision: PolicyDecision
    tracker_version: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)
    input_hash: Sha256

    @model_validator(mode="after")
    def validate_condition_input(self) -> "ConditionOutcome":
        if model_content_hash(self.tracker_before) != self.initial_state_hash:
            raise ValueError("Condition tracker state does not match the shared initial hash")
        if self.tracker_after.concept != self.tracker_before.concept:
            raise ValueError("Condition tracker update changed the target concept")
        if self.condition is BenchmarkCondition.DIALOGUE_ONLY:
            if self.additional_evidence is not None or self.evidence_source_hash is not None:
                raise ValueError("Dialogue-only condition cannot contain additional evidence")
            if self.tracker_after != self.tracker_before:
                raise ValueError("Dialogue-only condition must not update the cloned state")
        elif self.additional_evidence is None or self.evidence_source_hash is None:
            raise ValueError("Probe conditions require additional evidence and its source hash")
        if self.input_hash != condition_input_hash(self):
            raise ValueError("Condition input hash does not match its content")
        return self


def condition_input_hash(outcome: ConditionOutcome) -> Sha256:
    """Hash only the policy-safe values available before a condition decision."""

    return canonical_sha256(
        outcome.model_dump(
            mode="json",
            include={
                "schema_version",
                "schema_id",
                "case_id",
                "case_content_hash",
                "condition",
                "public_evidence_hash",
                "tracker_before",
                "initial_state_hash",
                "additional_evidence",
                "evidence_source_hash",
                "tracker_version",
                "policy_version",
            },
        )
    )


class PairedConditionRun(ContractModel):
    """Complete four-condition result sharing one public update and branch state."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.paired_condition_run.v1"] = "benchmark.paired_condition_run.v1"
    case_id: str = Field(min_length=1)
    case_content_hash: Sha256
    initial_tracker_state: TrackerState
    public_evidence: Evidence
    public_evidence_hash: Sha256
    post_public_state: TrackerState
    initial_state_hash: Sha256
    tracker_version: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)
    outcomes: tuple[ConditionOutcome, ...]

    @model_validator(mode="after")
    def validate_condition_parity(self) -> "PairedConditionRun":
        if self.public_evidence_hash != model_content_hash(self.public_evidence):
            raise ValueError("Public evidence hash does not match its content")
        if self.initial_state_hash != model_content_hash(self.post_public_state):
            raise ValueError("Post-public state hash does not match its content")
        if tuple(outcome.condition for outcome in self.outcomes) != EXPECTED_CONDITIONS:
            raise ValueError("Paired run must contain the complete ordered condition set")
        for outcome in self.outcomes:
            if outcome.case_id != self.case_id:
                raise ValueError("Condition outcome belongs to a different case")
            if outcome.case_content_hash != self.case_content_hash:
                raise ValueError("Condition outcome has a different case hash")
            if outcome.public_evidence_hash != self.public_evidence_hash:
                raise ValueError("Condition outcome has different public evidence")
            if outcome.initial_state_hash != self.initial_state_hash:
                raise ValueError("Condition outcome has a different initial state hash")
            if outcome.tracker_before != self.post_public_state:
                raise ValueError("Condition did not start from the shared post-public state")
            if outcome.tracker_version != self.tracker_version:
                raise ValueError("Condition used a different tracker version")
            if outcome.policy_version != self.policy_version:
                raise ValueError("Condition used a different policy version")
        assert_public_payload_safe(self.model_dump(mode="json"))
        return self


class PairedConditionRunner:
    """Run all prespecified conditions with one tracker and one policy implementation."""

    def __init__(
        self,
        manifest: PublicBenchmarkManifest,
        *,
        tracker_update: TrackerUpdate,
        policy_decision: PolicyDecisionFunction,
        tracker_version: str,
        policy_version: str,
    ) -> None:
        if not tracker_version.strip() or not policy_version.strip():
            raise ConditionRunError("Tracker and policy versions must be non-empty")
        self._cases = {case.case_id: case for case in manifest.cases}
        self._files = {entry.path: entry for entry in manifest.files}
        self._tracker_update = tracker_update
        self._policy_decision = policy_decision
        self._tracker_version = tracker_version
        self._policy_version = policy_version

    def run_case(
        self,
        *,
        case_id: str,
        initial_tracker_state: TrackerState,
        public_evidence: Evidence,
        probe_evidence: ProbeExecutionSummary,
        unrelated_evidence: ControlEvidenceRecord,
        corrupted_evidence: ControlEvidenceRecord,
    ) -> PairedConditionRun:
        """Update public state once, clone it four times, and run paired decisions."""

        case = self._case(case_id)
        self._validate_inputs(
            case=case,
            initial_tracker_state=initial_tracker_state,
            probe_evidence=probe_evidence,
            unrelated_evidence=unrelated_evidence,
            corrupted_evidence=corrupted_evidence,
        )
        post_public_state = self._tracker_update(initial_tracker_state, public_evidence)
        if post_public_state.concept != case.target_concept:
            raise ConditionRunError("Public tracker update changed the target concept")
        post_public_bytes = post_public_state.model_dump_json().encode("utf-8")
        initial_state_hash = model_content_hash(post_public_state)
        public_evidence_hash = model_content_hash(public_evidence)

        evidence_by_condition: dict[
            BenchmarkCondition,
            tuple[Evidence | None, Sha256 | None],
        ] = {
            BenchmarkCondition.DIALOGUE_ONLY: (None, None),
            BenchmarkCondition.PROBE_INFORMED: (
                evidence_from_test_counts(
                    passed=probe_evidence.passed,
                    failed=probe_evidence.failed,
                    source="evidence probe",
                ),
                probe_evidence.summary_hash,
            ),
            BenchmarkCondition.UNRELATED_PROBE: (
                evidence_from_test_counts(
                    passed=unrelated_evidence.passed,
                    failed=unrelated_evidence.failed,
                    source="unrelated probe control",
                ),
                unrelated_evidence.record_hash,
            ),
            BenchmarkCondition.CORRUPTED_PROBE: (
                evidence_from_test_counts(
                    passed=corrupted_evidence.passed,
                    failed=corrupted_evidence.failed,
                    source="corrupted probe control",
                ),
                corrupted_evidence.record_hash,
            ),
        }

        outcomes: list[ConditionOutcome] = []
        for condition in EXPECTED_CONDITIONS:
            tracker_before = TrackerState.model_validate_json(post_public_bytes)
            if model_content_hash(tracker_before) != initial_state_hash:
                raise ConditionRunError("Cloned tracker state changed during reconstruction")
            additional_evidence, evidence_source_hash = evidence_by_condition[condition]
            tracker_after = (
                tracker_before
                if additional_evidence is None
                else self._tracker_update(tracker_before, additional_evidence)
            )
            decision_evidence = additional_evidence or public_evidence
            decision = self._policy_decision(decision_evidence)
            outcomes.append(
                _create_condition_outcome(
                    case=case,
                    condition=condition,
                    public_evidence_hash=public_evidence_hash,
                    tracker_before=tracker_before,
                    initial_state_hash=initial_state_hash,
                    additional_evidence=additional_evidence,
                    evidence_source_hash=evidence_source_hash,
                    tracker_after=tracker_after,
                    decision=decision,
                    tracker_version=self._tracker_version,
                    policy_version=self._policy_version,
                )
            )

        return PairedConditionRun(
            case_id=case.case_id,
            case_content_hash=case.case_content_hash,
            initial_tracker_state=initial_tracker_state,
            public_evidence=public_evidence,
            public_evidence_hash=public_evidence_hash,
            post_public_state=post_public_state,
            initial_state_hash=initial_state_hash,
            tracker_version=self._tracker_version,
            policy_version=self._policy_version,
            outcomes=tuple(outcomes),
        )

    def _validate_inputs(
        self,
        *,
        case: BenchmarkCaseView,
        initial_tracker_state: TrackerState,
        probe_evidence: ProbeExecutionSummary,
        unrelated_evidence: ControlEvidenceRecord,
        corrupted_evidence: ControlEvidenceRecord,
    ) -> None:
        if initial_tracker_state.concept != case.target_concept:
            raise ConditionRunError("Initial tracker concept does not match benchmark case")
        evidence_test = self._files.get(case.evidence_test_ref)
        if (
            evidence_test is None
            or evidence_test.artifact_class is not PublicArtifactClass.EVIDENCE_TEST
            or probe_evidence.evidence_test_sha256 != evidence_test.sha256
        ):
            raise ConditionRunError("Probe evidence does not match the case evidence test")
        if (
            probe_evidence.case_id != case.case_id
            or probe_evidence.target_concept != case.target_concept
        ):
            raise ConditionRunError("Probe evidence belongs to a different case")
        if (
            unrelated_evidence.condition is not BenchmarkCondition.UNRELATED_PROBE
            or unrelated_evidence.target_case_id != case.case_id
            or unrelated_evidence.source_case_id == case.case_id
            or unrelated_evidence.source_target_concept == case.target_concept
        ):
            raise ConditionRunError("Unrelated evidence does not match the case control")
        if (
            corrupted_evidence.condition is not BenchmarkCondition.CORRUPTED_PROBE
            or corrupted_evidence.target_case_id != case.case_id
            or corrupted_evidence.source_case_id != case.case_id
            or corrupted_evidence.source_target_concept != case.target_concept
            or corrupted_evidence.source_summary_hash != probe_evidence.summary_hash
        ):
            raise ConditionRunError("Corrupted evidence does not match the case probe")

    def _case(self, case_id: str) -> BenchmarkCaseView:
        try:
            return self._cases[case_id]
        except KeyError as error:
            raise ConditionRunError(f"Unknown benchmark case: {case_id}") from error


def evidence_from_test_counts(*, passed: int, failed: int, source: str) -> Evidence:
    """Map normalized test counts to the one frozen evidence vocabulary."""

    total = passed + failed
    if total <= 0:
        raise ConditionRunError("Condition evidence requires at least one test result")
    if failed == 0:
        category = EvidenceCategory.CORRECT
    elif passed == 0:
        category = EvidenceCategory.INCORRECT
    else:
        category = EvidenceCategory.CONFLICTING
    confidence = abs(passed - failed) / total
    return Evidence(
        category=category,
        confidence=confidence,
        rationale=f"The {source} produced {passed} passed and {failed} failed checks.",
        observed_signals=(f"passed:{passed}", f"failed:{failed}"),
    )


def _create_condition_outcome(
    *,
    case: BenchmarkCaseView,
    condition: BenchmarkCondition,
    public_evidence_hash: Sha256,
    tracker_before: TrackerState,
    initial_state_hash: Sha256,
    additional_evidence: Evidence | None,
    evidence_source_hash: Sha256 | None,
    tracker_after: TrackerState,
    decision: PolicyDecision,
    tracker_version: str,
    policy_version: str,
) -> ConditionOutcome:
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.condition_outcome.v1",
        "case_id": case.case_id,
        "case_content_hash": case.case_content_hash,
        "condition": condition,
        "public_evidence_hash": public_evidence_hash,
        "tracker_before": tracker_before,
        "initial_state_hash": initial_state_hash,
        "additional_evidence": additional_evidence,
        "evidence_source_hash": evidence_source_hash,
        "tracker_after": tracker_after,
        "decision": decision,
        "tracker_version": tracker_version,
        "policy_version": policy_version,
    }
    hash_content = {
        key: value for key, value in content.items() if key not in {"tracker_after", "decision"}
    }
    return ConditionOutcome.model_validate(
        {**content, "input_hash": canonical_sha256(hash_content)}
    )
