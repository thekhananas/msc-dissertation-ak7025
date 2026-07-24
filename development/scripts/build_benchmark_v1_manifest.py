"""Build the draft v1 benchmark manifest from its authored files."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from socratic_tutor.benchmark.design import (
    BenchmarkDesignPlan,
    HeldOutCaseAllocation,
    load_design,
)
from socratic_tutor.benchmark.evaluator.models import ArtifactClass
from socratic_tutor.benchmark.hashing import file_sha256, model_content_hash
from socratic_tutor.benchmark.public.models import BenchmarkCaseView

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = WORKSPACE_ROOT / "data" / "benchmarks" / "v1"
DESIGN_PATH = WORKSPACE_ROOT / "data" / "benchmark-design" / "v1" / "case-allocation.yaml"
BUILD_TIME = datetime(2026, 8, 12, 16, 0, tzinfo=UTC)
ZERO_HASH = "0" * 64


def main() -> None:
    plan = load_design(DESIGN_PATH)
    cases = {case.case_id: case for case in plan.held_out_cases}
    _write_reviews(tuple(cases))

    case_rows: list[dict[str, Any]] = []
    for case in plan.held_out_cases:
        public = _public_case(case)
        criterion = _criterion_case(case)
        case_rows.append({"public": public, "criterion": criterion})

    file_classes = _file_classes(cases)
    files = [_inventory_entry(path, file_classes[path]) for path in sorted(file_classes)]
    file_lookup: dict[str, dict[str, Any]] = {str(entry["path"]): entry for entry in files}
    generation_prompt_ref = "prompts/student-system-v1.md"
    review_rows = [_review(case, file_lookup) for case in plan.held_out_cases]

    manifest = {
        "schema_version": 1,
        "schema_id": "benchmark.manifest.v1",
        "benchmark_version": "v1",
        "status": "draft",
        "frozen_at_utc": None,
        "parent_version": None,
        "manifest_hash": None,
        "conditions": ["dialogue_only", "probe_informed", "unrelated_probe", "corrupted_probe"],
        "cases": case_rows,
        "splits": _splits(plan),
        "files": files,
        "reviews": review_rows,
        "primary_metric": {
            "name": "paired_brier_difference",
            "favourable_direction": "positive",
            "aggregation_unit": "case",
            "interval_method": "paired_case_bootstrap",
            "missingness_rule": "exclude_missing_criterion_and_report_denominator",
        },
        "generation": {
            "provider": "recorded_fixture",
            "model": "authored-deterministic",
            "system_prompt_version": "student-system-v1",
            "system_prompt_ref": generation_prompt_ref,
            "system_prompt_sha256": file_lookup[generation_prompt_ref]["sha256"],
            "repeats_per_case": 1,
            "temperature": 0.0,
            "max_output_tokens": 512,
        },
        "limitations_ref": "limitations.md",
    }
    (BENCHMARK_ROOT / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )


def _public_case(case: HeldOutCaseAllocation) -> dict[str, Any]:
    case_id = case.case_id
    raw = {
        "schema_version": 1,
        "schema_id": "benchmark.case_public.v1",
        "benchmark_version": "v1",
        "case_id": case_id,
        "task_family": case.task_family,
        "split": "held_out",
        "target_concept": case.concept_id,
        "public_fixture_ref": f"public/{case_id}-dialogue.md",
        "evidence_probe_ref": f"evidence/{case_id}-prompt.md",
        "evidence_test_ref": f"evidence/{case_id}-tests.yaml",
        "initial_tracker_ref": "public/initial-tracker.json",
        "initial_tracker_hash": file_sha256(
            (BENCHMARK_ROOT / "public/initial-tracker.json").read_bytes()
        ),
        "unrelated_control_ref": "controls/unrelated.yaml",
        "corruption_spec_ref": "controls/corruption.yaml",
        "case_content_hash": ZERO_HASH,
    }
    view = BenchmarkCaseView.model_validate(raw)
    raw["case_content_hash"] = model_content_hash(view, exclude={"case_content_hash"})
    return raw


def _criterion_case(case: HeldOutCaseAllocation) -> dict[str, Any]:
    case_id = case.case_id
    criterion_root = "evaluator/criterion"
    return {
        "schema_version": 1,
        "schema_id": "benchmark.criterion_spec.v1",
        "benchmark_version": "v1",
        "case_id": case_id,
        "misconception_id": case.misconception_id,
        "transfer_case_id": case.task_family,
        "evidence_pattern": case.evidence_pattern.value,
        "criterion_probe_id": f"{case_id}-criterion-v1",
        "criterion_prompt_ref": f"{criterion_root}/{case_id}-prompt.md",
        "test_bundle_ref": f"{criterion_root}/{case_id}-tests.yaml",
        "test_bundle_sha256": file_sha256(
            (BENCHMARK_ROOT / f"{criterion_root}/{case_id}-tests.yaml").read_bytes()
        ),
        "rubric_ref": f"{criterion_root}/{case_id}-rubric.yaml",
        "rubric_sha256": file_sha256(
            (BENCHMARK_ROOT / f"{criterion_root}/{case_id}-rubric.yaml").read_bytes()
        ),
        "label_rationale_ref": f"{criterion_root}/{case_id}-label-rationale.md",
        "review_record_ref": f"evaluator/reviews/{case_id}.yaml",
        "structural_difference_record_ref": f"evaluator/structural/{case_id}.md",
    }


def _splits(plan: BenchmarkDesignPlan) -> list[dict[str, Any]]:
    rows = [
        {
            "schema_version": 1,
            "schema_id": "benchmark.split.v1",
            "split": "held_out",
            "task_families": [case.task_family for case in plan.held_out_cases],
        }
    ]
    for split in ("development", "calibration"):
        families = [
            item.task_family for item in plan.auxiliary_families if item.split.value == split
        ]
        rows.append(
            {
                "schema_version": 1,
                "schema_id": "benchmark.split.v1",
                "split": split,
                "task_families": families,
            }
        )
    return rows


def _file_classes(
    cases: Mapping[str, HeldOutCaseAllocation],
) -> dict[str, ArtifactClass]:
    classes: dict[str, ArtifactClass] = {
        "public/initial-tracker.json": ArtifactClass.INITIAL_TRACKER,
        "splits.yaml": ArtifactClass.SPLIT,
        "limitations.md": ArtifactClass.LIMITATIONS,
        "prompts/student-system-v1.md": ArtifactClass.PROMPT,
        "controls/unrelated.yaml": ArtifactClass.CONTROL,
        "controls/corruption.yaml": ArtifactClass.CONTROL,
    }
    for case_id in cases:
        classes[f"public/{case_id}-dialogue.md"] = ArtifactClass.PUBLIC_FIXTURE
        classes[f"evidence/{case_id}-prompt.md"] = ArtifactClass.EVIDENCE_PROBE
        classes[f"evidence/{case_id}-tests.yaml"] = ArtifactClass.EVIDENCE_TEST
        classes[f"evaluator/criterion/{case_id}-prompt.md"] = ArtifactClass.CRITERION_PROBE
        classes[f"evaluator/criterion/{case_id}-tests.yaml"] = ArtifactClass.CRITERION_TEST
        classes[f"evaluator/criterion/{case_id}-rubric.yaml"] = ArtifactClass.CRITERION_RUBRIC
        classes[f"evaluator/criterion/{case_id}-label-rationale.md"] = ArtifactClass.LABEL_RATIONALE
        classes[f"evaluator/structural/{case_id}.md"] = ArtifactClass.STRUCTURAL_DIFFERENCE
        classes[f"evaluator/reviews/{case_id}.yaml"] = ArtifactClass.REVIEW
    return classes


def _inventory_entry(path: str, artifact_class: ArtifactClass) -> dict[str, object]:
    content = (BENCHMARK_ROOT / path).read_bytes()
    schema_ids = {
        ArtifactClass.PUBLIC_FIXTURE: "benchmark.public_fixture.v1",
        ArtifactClass.EVIDENCE_PROBE: "benchmark.evidence_probe.v1",
        ArtifactClass.EVIDENCE_TEST: "benchmark.authored_tests.v1",
        ArtifactClass.INITIAL_TRACKER: "benchmark.initial_tracker.v1",
        ArtifactClass.CONTROL: "benchmark.control.v1",
        ArtifactClass.SPLIT: "benchmark.split_inventory.v1",
        ArtifactClass.LIMITATIONS: "benchmark.limitations.v1",
        ArtifactClass.PROMPT: "benchmark.student_system_prompt.v1",
        ArtifactClass.CRITERION_PROBE: "benchmark.criterion_probe.v1",
        ArtifactClass.CRITERION_TEST: "benchmark.authored_tests.v1",
        ArtifactClass.CRITERION_RUBRIC: "benchmark.criterion_rubric.v1",
        ArtifactClass.LABEL_RATIONALE: "benchmark.label_rationale.v1",
        ArtifactClass.REVIEW: "benchmark.review_file.v1",
        ArtifactClass.STRUCTURAL_DIFFERENCE: "benchmark.structural_difference.v1",
    }
    return {
        "path": path,
        "artifact_class": artifact_class.value,
        "schema_id": schema_ids[artifact_class],
        "sha256": file_sha256(content),
        "byte_size": len(content),
    }


def _review(
    case: HeldOutCaseAllocation,
    file_lookup: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    case_id = case.case_id
    paths = [
        f"public/{case_id}-dialogue.md",
        "public/initial-tracker.json",
        "controls/unrelated.yaml",
        "controls/corruption.yaml",
        "prompts/student-system-v1.md",
        f"evidence/{case_id}-prompt.md",
        f"evidence/{case_id}-tests.yaml",
        f"evaluator/criterion/{case_id}-prompt.md",
        f"evaluator/criterion/{case_id}-tests.yaml",
        f"evaluator/criterion/{case_id}-rubric.yaml",
        f"evaluator/criterion/{case_id}-label-rationale.md",
        f"evaluator/structural/{case_id}.md",
    ]
    return {
        "schema_version": 1,
        "schema_id": "benchmark.review.v1",
        "review_id": f"v1-review-{case_id}",
        "case_id": case_id,
        "author": "Anas Khan",
        "reviewer": "independent-reviewer-tbd",
        "reviewed_at_utc": None,
        "decision": "pending",
        "concerns": [],
        "adjudication": None,
        "reviewed_files": [{"path": path, "sha256": file_lookup[path]["sha256"]} for path in paths],
    }


def _write_reviews(case_ids: tuple[str, ...]) -> None:
    review_root = BENCHMARK_ROOT / "evaluator" / "reviews"
    review_root.mkdir(parents=True, exist_ok=True)
    for case_id in case_ids:
        review = {
            "schema_id": "benchmark.review_file.v1",
            "case_id": case_id,
            "status": "pending",
            "author": "Anas Khan",
            "reviewer": "independent-reviewer-tbd",
            "review_scope": [
                "public_dialogue_matches_declared_pattern",
                "evidence_and_criterion_measure_the_same_concept",
                "criterion_does_not_repeat_the_evidence_solution",
                "authored_tests_match_the_prompts",
                "controls_do_not_use_criterion_labels",
            ],
        }
        (review_root / f"{case_id}.yaml").write_text(
            yaml.safe_dump(review, sort_keys=False), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
