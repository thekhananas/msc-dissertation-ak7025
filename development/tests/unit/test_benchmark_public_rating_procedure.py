"""Tests for the complete public-answer rating procedure record."""

from pathlib import Path

import pytest

from socratic_tutor.benchmark.public_rating import PublicAnswerRatingReport
from socratic_tutor.benchmark.public_rating_procedure import (
    freeze_public_rating_procedure,
    load_public_rating_procedure_plan,
)

WORKSPACE_ROOT = Path(__file__).parents[2]
PLAN_PATH = WORKSPACE_ROOT / "configs" / "benchmark" / "v2-public-rating-procedure.yaml"


def _rating_report(*, disagreements: int = 0) -> PublicAnswerRatingReport:
    return PublicAnswerRatingReport.model_construct(
        report_hash="a" * 64,
        packet_hash="b" * 64,
        rating_guide_hash="c" * 64,
        protocol_amendment_hash="d" * 64,
        item_count=24,
        disagreement_count=disagreements,
        rater_ids=("rater-a", "rater-b"),
    )


def _plan_for_fixture():
    return load_public_rating_procedure_plan(PLAN_PATH).model_copy(
        update={"public_rating_report_hash": "a" * 64}
    )


def test_versioned_plan_records_roles_independence_and_access() -> None:
    plan = load_public_rating_procedure_plan(PLAN_PATH)

    assert tuple(item.stable_pseudonym for item in plan.raters) == ("OT", "AD")
    assert tuple(item.completion_order for item in plan.raters) == (1, 2)
    assert all(item.completed_independently for item in plan.raters)
    assert all(not item.additional_tools_used for item in plan.raters)
    assert plan.public_category_guide_present is True
    assert plan.case_specific_scoring_key_present is False


def test_freezes_procedure_against_the_exact_rating_report(tmp_path: Path) -> None:
    record = freeze_public_rating_procedure(
        plan=_plan_for_fixture(),
        rating_report=_rating_report(),
        output_path=tmp_path / "procedure.json",
    )

    assert record.actual_disagreement_count == 0
    assert record.actual_adjudication_required is False
    assert record.planned_adjudicator == "dissertation_author"
    assert record.criterion_material_remained_restricted is True
    assert record.sufficient_to_unlock_criterion_access is False


def test_rejects_raters_that_do_not_match_the_completed_report(tmp_path: Path) -> None:
    report = _rating_report().model_copy(update={"rater_ids": ("rater-b", "rater-a")})

    with pytest.raises(ValueError, match="raters do not match"):
        freeze_public_rating_procedure(
            plan=_plan_for_fixture(),
            rating_report=report,
            output_path=tmp_path / "invalid.json",
        )


def test_requires_separate_record_when_disagreement_exists(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="separate adjudication"):
        freeze_public_rating_procedure(
            plan=_plan_for_fixture(),
            rating_report=_rating_report(disagreements=1),
            output_path=tmp_path / "invalid.json",
        )
