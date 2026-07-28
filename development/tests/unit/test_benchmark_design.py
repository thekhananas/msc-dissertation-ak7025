"""Tests for the pre-authoring benchmark allocation."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from socratic_tutor.benchmark.design import BenchmarkDesignPlan, DesignStatus, load_design

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DESIGN_ROOT = WORKSPACE_ROOT / "data" / "benchmark-design" / "v1"
PLAN_PATH = DESIGN_ROOT / "case-allocation.yaml"


def test_v1_design_has_balanced_24_case_allocation() -> None:
    plan = load_design(PLAN_PATH)

    assert plan.status is DesignStatus.FROZEN
    assert len(plan.concepts) == 4
    assert sum(len(concept.misconception_ids) for concept in plan.concepts) == 8
    assert len(plan.held_out_cases) == 24
    assert len({case.task_family for case in plan.held_out_cases}) == 24


def test_design_rejects_an_incomplete_misconception_allocation() -> None:
    raw = yaml.safe_load(PLAN_PATH.read_text(encoding="utf-8"))
    raw["held_out_cases"] = raw["held_out_cases"][:-1]

    with pytest.raises(ValidationError, match="at least 24 items"):
        BenchmarkDesignPlan.model_validate(raw)


def test_design_rejects_split_family_overlap() -> None:
    raw = yaml.safe_load(PLAN_PATH.read_text(encoding="utf-8"))
    raw["auxiliary_families"][0]["task_family"] = raw["held_out_cases"][0]["task_family"]

    with pytest.raises(ValidationError, match="must be disjoint"):
        BenchmarkDesignPlan.model_validate(raw)


def test_design_rejects_reused_channel_context() -> None:
    raw = yaml.safe_load(PLAN_PATH.read_text(encoding="utf-8"))
    first = raw["held_out_cases"][0]
    first["public_context"] = first["evidence_context"]

    with pytest.raises(ValidationError, match="contexts must differ"):
        BenchmarkDesignPlan.model_validate(raw)


def test_design_rejects_private_blueprint_metadata() -> None:
    raw = yaml.safe_load(PLAN_PATH.read_text(encoding="utf-8"))
    raw["concept_blueprint_ref"] = "private-note.md"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        BenchmarkDesignPlan.model_validate(raw)
