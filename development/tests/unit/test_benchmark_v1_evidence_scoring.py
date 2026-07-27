"""Tests for scoring recorded responses to the v1 evidence probes."""

import json
from pathlib import Path

import pytest
import yaml

from socratic_tutor.benchmark.evidence_scoring import (
    EvidenceScoringError,
    score_evidence_responses,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = WORKSPACE_ROOT / "data" / "benchmarks" / "v1"
MANIFEST_PATH = BENCHMARK_ROOT / "manifest.yaml"


def _responses() -> list[dict[str, str]]:
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    result: list[dict[str, str]] = []
    for case in manifest["cases"]:
        tests = yaml.safe_load(
            (BENCHMARK_ROOT / case["public"]["evidence_test_ref"]).read_text(encoding="utf-8")
        )
        expected = tests["checks"][0]["expected"]
        result.append({"case_id": case["public"]["case_id"], "response": _as_response(expected)})
    return result


def _as_response(value: object) -> str:
    if isinstance(value, str):
        return value
    return repr(value)


def test_v1_evidence_score_accepts_one_correct_response_per_case(tmp_path: Path) -> None:
    response_path = tmp_path / "responses.jsonl"
    response_path.write_text(
        "".join(json.dumps(response) + "\n" for response in _responses()),
        encoding="utf-8",
    )

    summary = score_evidence_responses(
        manifest_path=MANIFEST_PATH,
        responses_path=response_path,
        output_path=tmp_path / "summary.json",
    )

    assert summary.case_count == 24
    assert summary.response_count == 24
    assert summary.correct_count == 24
    assert summary.incorrect_count == 0
    assert summary.invalid_count == 0
    assert summary.accuracy == 1.0
    assert len(summary.concept_scores) == 4
    assert len(summary.task_family_scores) == 24
    assert all(group.accuracy == 1.0 for group in summary.concept_scores)
    assert (tmp_path / "summary.json").exists()


def test_v1_evidence_score_rejects_incomplete_response_file(tmp_path: Path) -> None:
    response_path = tmp_path / "responses.jsonl"
    response_path.write_text(json.dumps(_responses()[0]) + "\n", encoding="utf-8")

    with pytest.raises(EvidenceScoringError, match="missing cases"):
        score_evidence_responses(
            manifest_path=MANIFEST_PATH,
            responses_path=response_path,
            output_path=tmp_path / "summary.json",
        )


def test_v1_evidence_score_records_invalid_and_incorrect_answers(tmp_path: Path) -> None:
    responses = _responses()
    responses[0]["response"] = "not-an-integer"
    responses[1]["response"] = "5"
    response_path = tmp_path / "responses.jsonl"
    response_path.write_text(
        "".join(json.dumps(response) + "\n" for response in responses),
        encoding="utf-8",
    )

    summary = score_evidence_responses(
        manifest_path=MANIFEST_PATH,
        responses_path=response_path,
        output_path=tmp_path / "summary.json",
    )

    assert summary.invalid_count == 1
    assert summary.incorrect_count == 1
    assert summary.correct_count == 22


def test_v1_evidence_score_does_not_treat_arbitrary_text_as_false(tmp_path: Path) -> None:
    responses = _responses()
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    boolean_index = None
    for index, case in enumerate(manifest["cases"]):
        tests = yaml.safe_load(
            (BENCHMARK_ROOT / case["public"]["evidence_test_ref"]).read_text(encoding="utf-8")
        )
        if isinstance(tests["checks"][0]["expected"], bool):
            boolean_index = index
            break
    assert boolean_index is not None
    responses[boolean_index]["response"] = "maybe"
    response_path = tmp_path / "responses.jsonl"
    response_path.write_text(
        "".join(json.dumps(response) + "\n" for response in responses),
        encoding="utf-8",
    )

    summary = score_evidence_responses(
        manifest_path=MANIFEST_PATH,
        responses_path=response_path,
        output_path=tmp_path / "summary.json",
    )

    assert summary.invalid_count == 1
