"""Tests for the pre-authoring benchmark allocation."""

from pathlib import Path
from shutil import copytree

import pytest
import yaml
from pydantic import ValidationError

from socratic_tutor.benchmark.design import (
    BenchmarkDesignPlan,
    BenchmarkDesignReadError,
    load_and_verify_design,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DESIGN_ROOT = WORKSPACE_ROOT / "data" / "benchmark-design" / "v1"
PLAN_PATH = DESIGN_ROOT / "case-allocation.yaml"


def test_v1_design_has_balanced_24_case_allocation() -> None:
    plan = load_and_verify_design(PLAN_PATH)

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


def test_design_detects_blueprint_mutation(tmp_path: Path) -> None:
    copied_root = tmp_path / "v1"
    copytree(DESIGN_ROOT, copied_root)
    blueprint = copied_root / "concept-blueprint.md"
    blueprint.write_text(blueprint.read_text(encoding="utf-8") + "\nchanged\n", encoding="utf-8")

    with pytest.raises(BenchmarkDesignReadError, match="hash does not match"):
        load_and_verify_design(copied_root / "case-allocation.yaml")
