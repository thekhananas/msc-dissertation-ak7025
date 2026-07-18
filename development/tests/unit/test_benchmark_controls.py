"""Negative-control construction tests for the development benchmark."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from socratic_tutor.benchmark.evaluator.loader import load_and_verify_manifest
from socratic_tutor.benchmark.evaluator.projection import project_public_manifest
from socratic_tutor.benchmark.public import (
    BenchmarkCondition,
    ControlConstructionError,
    ControlEvidenceRecord,
    ControlFactory,
    CorruptionControlSpec,
    UnrelatedControlSpec,
    find_public_payload_violations,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = WORKSPACE_ROOT / "data" / "benchmarks" / "dev-v0"
MANIFEST_PATH = BENCHMARK_ROOT / "manifest.yaml"


def control_factory() -> ControlFactory:
    authored = load_and_verify_manifest(MANIFEST_PATH)
    return ControlFactory(BENCHMARK_ROOT, project_public_manifest(authored))


def test_controls_are_stable_and_content_addressed() -> None:
    factory = control_factory()
    aliasing = factory.probe_summary(case_id="dev-aliasing-001", passed=0, failed=2)
    none_falsy = factory.probe_summary(case_id="dev-none-falsy-001", passed=2, failed=0)
    evidence_by_case = {
        aliasing.case_id: aliasing,
        none_falsy.case_id: none_falsy,
    }

    first_unrelated = factory.unrelated(
        case_id=aliasing.case_id,
        evidence_by_case=evidence_by_case,
    )
    second_unrelated = factory.unrelated(
        case_id=aliasing.case_id,
        evidence_by_case=evidence_by_case,
    )
    first_corrupted = factory.corrupted(case_id=aliasing.case_id, evidence=aliasing)
    second_corrupted = factory.corrupted(case_id=aliasing.case_id, evidence=aliasing)

    assert first_unrelated == second_unrelated
    assert first_unrelated.condition is BenchmarkCondition.UNRELATED_PROBE
    assert first_unrelated.source_case_id == none_falsy.case_id
    assert (first_unrelated.passed, first_unrelated.failed) == (2, 0)
    assert first_corrupted == second_corrupted
    assert first_corrupted.condition is BenchmarkCondition.CORRUPTED_PROBE
    assert first_corrupted.source_case_id == aliasing.case_id
    assert (first_corrupted.passed, first_corrupted.failed) == (2, 0)
    assert first_corrupted.seed == 20260725
    assert first_corrupted.record_hash != first_unrelated.record_hash
    assert find_public_payload_violations(first_unrelated.model_dump(mode="json")) == ()
    assert find_public_payload_violations(first_corrupted.model_dump(mode="json")) == ()

    changed_seed = first_corrupted.model_dump(mode="json")
    changed_seed["seed"] = 20260726
    with pytest.raises(ValidationError, match="hash does not match"):
        ControlEvidenceRecord.model_validate(changed_seed)


def test_control_factory_rejects_wrong_source_evidence() -> None:
    factory = control_factory()
    aliasing = factory.probe_summary(case_id="dev-aliasing-001", passed=0, failed=2)

    with pytest.raises(ControlConstructionError, match="Missing unrelated source evidence"):
        factory.unrelated(
            case_id=aliasing.case_id,
            evidence_by_case={aliasing.case_id: aliasing},
        )

    with pytest.raises(ControlConstructionError, match="does not match its source case"):
        factory.corrupted(case_id="dev-none-falsy-001", evidence=aliasing)


def test_control_factory_requires_exact_frozen_assignment_coverage() -> None:
    authored = load_and_verify_manifest(MANIFEST_PATH)
    public = project_public_manifest(authored)
    incomplete_public = public.model_copy(update={"cases": public.cases[:1]})

    with pytest.raises(ControlConstructionError, match="do not match referencing cases"):
        ControlFactory(BENCHMARK_ROOT, incomplete_public)


def test_control_specs_reject_evaluator_owned_fields() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        UnrelatedControlSpec.model_validate(
            {
                "schema_id": "benchmark.unrelated_control.v1",
                "assignments": {"case-a": "case-b"},
                "rule": "use_the_other_cases_evidence_summary_without_criterion_access",
                "criterion_label": True,
            }
        )

    with pytest.raises(ValidationError, match="extra_forbidden"):
        CorruptionControlSpec.model_validate(
            {
                "schema_id": "benchmark.corruption_control.v1",
                "transformation": "invert_pass_fail_summary",
                "input_fields": ["passed", "failed"],
                "reject_undeclared_fields": True,
                "seed": 20260725,
                "criterion_label": False,
            }
        )


def test_probe_summary_rejects_empty_execution() -> None:
    factory = control_factory()

    with pytest.raises(ValidationError, match="at least one test result"):
        factory.probe_summary(case_id="dev-aliasing-001", passed=0, failed=0)
