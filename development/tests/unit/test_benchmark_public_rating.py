"""Tests for the blinded public-answer rating boundary."""

import csv
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from socratic_tutor.benchmark.artifacts import write_immutable_jsonl
from socratic_tutor.benchmark.external_protocol import ExternalModelExecutionProtocol
from socratic_tutor.benchmark.generation import (
    CriterionTaskPayload,
    GenerationRequestSpec,
    ModelRoute,
    PublicTaskPayload,
    SamplingConfig,
    create_student_generation_request,
)
from socratic_tutor.benchmark.hashing import file_sha256, model_content_hash
from socratic_tutor.benchmark.public_rating import (
    PublicAnswerRating,
    PublicAnswerRatingGuidePlan,
    build_public_answer_rating_packet,
    create_public_answer_adjudication,
    create_public_answer_rating,
    export_public_rating_workbook,
    finalize_public_answer_ratings,
    freeze_public_rating_boundary,
    record_public_answer_ratings,
)
from socratic_tutor.benchmark.replay import (
    OriginalGenerationSource,
    create_recorded_response,
)
from socratic_tutor.contracts import EvidenceCategory

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
GUIDE_PLAN = WORKSPACE_ROOT / "configs" / "benchmark" / "v2-public-answer-rating-guide.yaml"


def _protocol() -> ExternalModelExecutionProtocol:
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.external_model_protocol.v1",
        "protocol_version": "v2",
        "source_benchmark_version": "v1",
        "source_manifest_hash": "1" * 64,
        "source_public_projection_hash": "2" * 64,
        "analysis_specification_hash": "3" * 64,
        "provider_id": "cerebras",
        "adapter_version": "cerebras-chat-completions-v1",
        "requested_model_id": "gpt-oss-120b",
        "system_prompt_version": "external-evaluation-system-v1",
        "system_prompt_sha256": file_sha256(b"Answer the visible task."),
        "temperature": 0.0,
        "max_output_tokens": 512,
        "run_seed": 20260823,
        "reasoning_effort": "low",
        "timeout_seconds": 60.0,
        "max_attempts": 2,
        "request_spacing_seconds": 13.0,
        "repeats_per_case": 1,
        "request_budget": 72,
        "cost_budget_usd": Decimal("5.00"),
        "missingness_rule": "exclude_missing_criterion_and_report_denominator",
        "created_at_utc": datetime(2026, 8, 23, 1, 15, tzinfo=UTC),
    }
    draft = ExternalModelExecutionProtocol.model_construct(
        _fields_set=set(content),
        **cast(dict[str, Any], content),
        protocol_hash="0" * 64,
    )
    return ExternalModelExecutionProtocol.model_validate(
        {**content, "protocol_hash": model_content_hash(draft, exclude={"protocol_hash"})}
    )


def _guide_plan(protocol: ExternalModelExecutionProtocol) -> PublicAnswerRatingGuidePlan:
    raw = yaml.safe_load(GUIDE_PLAN.read_text(encoding="utf-8"))
    raw["base_protocol_hash"] = protocol.protocol_hash
    return PublicAnswerRatingGuidePlan.model_validate(raw)


def _freeze(tmp_path: Path):
    protocol = _protocol()
    result = freeze_public_rating_boundary(
        plan=_guide_plan(protocol),
        protocol=protocol,
        guide_output_path=tmp_path / "guide.json",
        amendment_output_path=tmp_path / "amendment.json",
    )
    from socratic_tutor.benchmark.public_rating import (
        load_public_answer_rating_guide,
        load_public_rating_protocol_amendment,
    )

    guide = load_public_answer_rating_guide(tmp_path / "guide.json")
    amendment = load_public_rating_protocol_amendment(tmp_path / "amendment.json")
    assert result.rating_guide_hash == guide.guide_hash
    return protocol, guide, amendment


def _record(protocol: ExternalModelExecutionProtocol, case_id: str, *, criterion: bool = False):
    prompt = "Answer the visible task."
    spec = GenerationRequestSpec(
        run_id="rating-test",
        sample_id=f"{case_id}-sample",
        system_prompt_version=protocol.system_prompt_version,
        system_prompt=prompt,
        system_prompt_sha256=file_sha256(prompt.encode()),
        model_route=ModelRoute(provider=protocol.provider_id, model=protocol.requested_model_id),
        sampling=SamplingConfig(
            temperature=protocol.temperature,
            max_output_tokens=protocol.max_output_tokens,
            seed=protocol.run_seed,
        ),
    )
    payload = (
        CriterionTaskPayload(
            benchmark_version="dev-v0",
            case_id=case_id,
            criterion_probe_id=f"{case_id}-criterion",
            criterion_probe="Hidden transfer question.",
        )
        if criterion
        else PublicTaskPayload(
            benchmark_version="dev-v0",
            case_id=case_id,
            task_family="python",
            target_concept="visible concept",
            public_interaction=f"Visible question for {case_id}?",
        )
    )
    request = create_student_generation_request(spec=spec, payload=payload)
    return create_recorded_response(
        request=request,
        original_source=OriginalGenerationSource.AUTHORED,
        final_response=f"Visible answer for {case_id}.",
        captured_at_utc=datetime(2026, 8, 24, 9, 0, tzinfo=UTC),
    )


def _packet(tmp_path: Path):
    protocol, guide, amendment = _freeze(tmp_path)
    responses = (_record(protocol, "case-001"), _record(protocol, "case-002"))
    write_immutable_jsonl(tmp_path / "responses.jsonl", responses)
    packet = build_public_answer_rating_packet(
        protocol=protocol,
        amendment=amendment,
        recorded_responses_path=tmp_path / "responses.jsonl",
        output_path=tmp_path / "packet.json",
    )
    return guide, amendment, packet


def test_packet_contains_only_the_four_reviewed_public_fields(tmp_path: Path) -> None:
    _, _, packet = _packet(tmp_path)

    raw = json.loads((tmp_path / "packet.json").read_text(encoding="utf-8"))
    assert set(raw["items"][0]) == {
        "case_id",
        "public_prompt",
        "visible_response",
        "response_hash",
    }
    assert "criterion" not in json.dumps(raw).casefold()
    assert len(packet.items) == 2


def test_packet_projects_public_records_from_a_mixed_source_without_leaking_criterion(
    tmp_path: Path,
) -> None:
    protocol, _, amendment = _freeze(tmp_path)
    write_immutable_jsonl(
        tmp_path / "mixed.jsonl",
        (_record(protocol, "case-001"), _record(protocol, "case-001", criterion=True)),
    )

    packet = build_public_answer_rating_packet(
        protocol=protocol,
        amendment=amendment,
        recorded_responses_path=tmp_path / "mixed.jsonl",
        output_path=tmp_path / "packet.json",
    )

    assert len(packet.items) == 1
    assert "criterion" not in (tmp_path / "packet.json").read_text(encoding="utf-8").casefold()


def test_workbook_records_one_complete_human_rating_set(tmp_path: Path) -> None:
    guide, _, packet = _packet(tmp_path)
    workbook = export_public_rating_workbook(
        guide=guide,
        packet=packet,
        output_root=tmp_path / "workbook",
    )
    sheet_path = tmp_path / "workbook" / "rating_sheet.csv"
    with sheet_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        fieldnames = tuple(reader.fieldnames)
        rows = list(reader)
    for row in rows:
        row["category"] = "correct"
        row["rationale"] = "The visible answer matches the visible prompt."
    completed_path = tmp_path / "completed.csv"
    with completed_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    result = record_public_answer_ratings(
        guide=guide,
        packet=packet,
        completed_sheet_path=completed_path,
        rater_id="independent-rater-a",
        output_path=tmp_path / "ratings-a.jsonl",
    )

    assert workbook.item_count == result.rating_count == 2
    assert result.rater_id == "independent-rater-a"
    recorded = [
        json.loads(line)
        for line in (tmp_path / "ratings-a.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {row["category"] for row in recorded} == {"correct"}
    assert {row["case_id"] for row in recorded} == {"case-001", "case-002"}

    rows[0]["visible_response"] = "Changed after review."
    tampered_path = tmp_path / "tampered.csv"
    with tampered_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="changed blinded content"):
        record_public_answer_ratings(
            guide=guide,
            packet=packet,
            completed_sheet_path=tampered_path,
            rater_id="independent-rater-a",
            output_path=tmp_path / "tampered.jsonl",
        )


def test_two_raters_require_adjudication_and_report_agreement(tmp_path: Path) -> None:
    guide, amendment, packet = _packet(tmp_path)
    ratings_a = (
        create_public_answer_rating(
            packet=packet,
            guide=guide,
            case_id="case-001",
            rater_id="rater-a",
            category=EvidenceCategory.CORRECT,
            rationale="The answer is correct.",
        ),
        create_public_answer_rating(
            packet=packet,
            guide=guide,
            case_id="case-002",
            rater_id="rater-a",
            category=EvidenceCategory.INCORRECT,
            rationale="The answer is wrong without a stated false rule.",
        ),
    )
    ratings_b = (
        create_public_answer_rating(
            packet=packet,
            guide=guide,
            case_id="case-001",
            rater_id="rater-b",
            category=EvidenceCategory.CORRECT,
            rationale="The answer is correct.",
        ),
        create_public_answer_rating(
            packet=packet,
            guide=guide,
            case_id="case-002",
            rater_id="rater-b",
            category=EvidenceCategory.MISCONCEPTION,
            rationale="The answer states a specific false rule.",
        ),
    )
    write_immutable_jsonl(tmp_path / "ratings-a.jsonl", ratings_a)
    write_immutable_jsonl(tmp_path / "ratings-b.jsonl", ratings_b)

    with pytest.raises(ValueError, match="requires one adjudication"):
        finalize_public_answer_ratings(
            guide=guide,
            amendment=amendment,
            packet=packet,
            ratings_a_path=tmp_path / "ratings-a.jsonl",
            ratings_b_path=tmp_path / "ratings-b.jsonl",
            adjudications_path=None,
            output_path=tmp_path / "report.json",
        )

    adjudication = create_public_answer_adjudication(
        packet=packet,
        guide=guide,
        left=ratings_a[1],
        right=ratings_b[1],
        final_category=EvidenceCategory.MISCONCEPTION,
        rationale="The response states a specific false rule, so misconception takes precedence.",
    )
    write_immutable_jsonl(tmp_path / "adjudications.jsonl", (adjudication,))
    report = finalize_public_answer_ratings(
        guide=guide,
        amendment=amendment,
        packet=packet,
        ratings_a_path=tmp_path / "ratings-a.jsonl",
        ratings_b_path=tmp_path / "ratings-b.jsonl",
        adjudications_path=tmp_path / "adjudications.jsonl",
        output_path=tmp_path / "report.json",
    )

    assert report.raw_agreement == 0.5
    assert report.cohen_kappa is not None
    assert abs(report.cohen_kappa - (1 / 3)) < 1e-12
    assert report.disagreement_count == 1
    assert report.final_ratings[1].resolution == "adjudicated"
    assert report.final_ratings[1].confidence == 0.9


def test_kappa_is_explicitly_undefined_for_identical_single_category_marginals(
    tmp_path: Path,
) -> None:
    guide, amendment, packet = _packet(tmp_path)
    rating_sets: list[tuple[PublicAnswerRating, ...]] = []
    for rater_id in ("rater-a", "rater-b"):
        rating_sets.append(
            tuple(
                create_public_answer_rating(
                    packet=packet,
                    guide=guide,
                    case_id=item.case_id,
                    rater_id=rater_id,
                    category=EvidenceCategory.CORRECT,
                    rationale="The visible answer is correct.",
                )
                for item in packet.items
            )
        )
    write_immutable_jsonl(tmp_path / "ratings-a.jsonl", rating_sets[0])
    write_immutable_jsonl(tmp_path / "ratings-b.jsonl", rating_sets[1])

    report = finalize_public_answer_ratings(
        guide=guide,
        amendment=amendment,
        packet=packet,
        ratings_a_path=tmp_path / "ratings-a.jsonl",
        ratings_b_path=tmp_path / "ratings-b.jsonl",
        adjudications_path=None,
        output_path=tmp_path / "report.json",
    )

    assert report.raw_agreement == 1.0
    assert report.cohen_kappa is None
    assert report.kappa_status == "undefined_degenerate_marginals"
