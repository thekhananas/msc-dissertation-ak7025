"""Bounded LangGraph execution for one deterministic tutoring turn."""

# pyright: reportUnknownMemberType=false

from typing import TypedDict, cast

from langgraph.graph import END, START, StateGraph

from socratic_tutor.contracts import (
    Evidence,
    GuardrailResult,
    PolicyDecision,
    StudentSubmission,
    TaskDefinition,
    TrackerState,
    TurnResult,
)
from socratic_tutor.evaluation import classify_evidence
from socratic_tutor.generation import generate_prompt
from socratic_tutor.guardrails import check_prompt
from socratic_tutor.policies import choose_action
from socratic_tutor.tracking import update_tracker


class TurnGraphState(TypedDict):
    """Internal state for the fixed, acyclic turn graph."""

    task: TaskDefinition
    submission: StudentSubmission
    question_prompt: str | None
    tracker_before: TrackerState
    evidence: Evidence | None
    tracker_after: TrackerState | None
    decision: PolicyDecision | None
    candidate_prompt: str | None
    guardrail: GuardrailResult | None


def _classify(state: TurnGraphState) -> dict[str, Evidence]:
    return {
        "evidence": classify_evidence(
            state["task"],
            state["submission"],
            observation_number=state["tracker_before"].observations,
            question_prompt=state["question_prompt"],
        )
    }


def _track(state: TurnGraphState) -> dict[str, TrackerState]:
    evidence = state["evidence"]
    if evidence is None:
        raise RuntimeError("Evidence node did not produce a result")
    return {"tracker_after": update_tracker(state["tracker_before"], evidence)}


def _decide(state: TurnGraphState) -> dict[str, PolicyDecision]:
    evidence = state["evidence"]
    tracker_after = state["tracker_after"]
    if evidence is None:
        raise RuntimeError("Evidence node did not produce a result")
    if tracker_after is None:
        raise RuntimeError("Tracker node did not produce a result")
    return {
        "decision": choose_action(
            evidence,
            observation_number=tracker_after.observations,
        )
    }


def _generate(state: TurnGraphState) -> dict[str, str]:
    decision = state["decision"]
    tracker_after = state["tracker_after"]
    if decision is None or tracker_after is None:
        raise RuntimeError("Policy and tracker nodes must complete before generation")
    return {
        "candidate_prompt": generate_prompt(
            state["task"],
            decision,
            tracker_after.observations,
            evidence_category=state["evidence"].category,
            question_prompt=state["question_prompt"],
        ),
    }


def _guard(state: TurnGraphState) -> dict[str, GuardrailResult]:
    prompt = state["candidate_prompt"]
    if prompt is None:
        raise RuntimeError("Generation node did not produce a prompt")
    return {"guardrail": check_prompt(prompt)}


def _build_graph():  # type: ignore[no-untyped-def]
    builder = StateGraph(TurnGraphState)
    builder.add_node("classify_evidence", _classify)
    builder.add_node("update_tracker", _track)
    builder.add_node("choose_action", _decide)
    builder.add_node("generate_prompt", _generate)
    builder.add_node("check_guardrail", _guard)
    builder.add_edge(START, "classify_evidence")
    builder.add_edge("classify_evidence", "update_tracker")
    builder.add_edge("update_tracker", "choose_action")
    builder.add_edge("choose_action", "generate_prompt")
    builder.add_edge("generate_prompt", "check_guardrail")
    builder.add_edge("check_guardrail", END)
    return builder.compile(name="deterministic_tutoring_turn")


_TURN_GRAPH = _build_graph()


def run_graph_turn(
    task: TaskDefinition,
    submission: StudentSubmission,
    tracker: TrackerState,
    *,
    question_prompt: str | None = None,
) -> TurnResult:
    """Run exactly one bounded graph pass and return its validated result."""

    if tracker.concept != task.concept:
        raise ValueError("Tracker concept must match the task concept")

    initial: TurnGraphState = {
        "task": task,
        "submission": submission,
        "question_prompt": question_prompt,
        "tracker_before": tracker,
        "evidence": None,
        "tracker_after": None,
        "decision": None,
        "candidate_prompt": None,
        "guardrail": None,
    }
    state = cast(TurnGraphState, _TURN_GRAPH.invoke(initial))
    evidence = state["evidence"]
    tracker_after = state["tracker_after"]
    decision = state["decision"]
    guardrail = state["guardrail"]
    if evidence is None or tracker_after is None or decision is None or guardrail is None:
        raise RuntimeError("Tutoring graph completed with an incomplete state")
    return TurnResult(
        evidence=evidence,
        tracker_before=state["tracker_before"],
        tracker_after=tracker_after,
        decision=decision,
        next_prompt=guardrail.output_prompt,
        guardrail=guardrail,
    )
