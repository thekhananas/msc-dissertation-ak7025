from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from socratic_tutor.benchmark.artifacts import write_immutable_json
from socratic_tutor.benchmark.external_criterion import (
    ExternalCriterionPlan,
    ExternalCriterionReport,
)
from socratic_tutor.benchmark.external_seal import (
    ExternalDecisionSealPlan,
    ExternalDecisionSealReport,
)
from socratic_tutor.benchmark.generation import ModelRoute
from socratic_tutor.benchmark.hashing import file_sha256, model_content_hash
from socratic_tutor.benchmark.study_design_figure import (
    StudyDesignFigureError,
    render_study_design_figure,
)
from socratic_tutor.contracts import ContractModel

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DESIGN = WORKSPACE_ROOT / "data" / "benchmark-design" / "v1" / "case-allocation.yaml"
RUN_ID = "study-design-figure-test"
ROUTE = ModelRoute(provider="cerebras", model="gpt-oss-120b")


def test_renders_commit_before_reveal_flow_and_retries_exactly(tmp_path: Path) -> None:
    decision_plan, decision_report, criterion_plan, criterion_report, pixi_lock = _write_sources(
        tmp_path
    )
    output_root = tmp_path / "publication"
    generated_at = datetime(2026, 9, 3, 19, 0, tzinfo=UTC)

    first = render_study_design_figure(
        benchmark_design_path=BENCHMARK_DESIGN,
        decision_seal_plan_path=decision_plan,
        decision_seal_report_path=decision_report,
        criterion_plan_path=criterion_plan,
        criterion_report_path=criterion_report,
        pixi_lock_path=pixi_lock,
        output_root=output_root,
        publication_code_revision="a" * 40,
        generated_at_utc=generated_at,
    )
    retry = render_study_design_figure(
        benchmark_design_path=BENCHMARK_DESIGN,
        decision_seal_plan_path=decision_plan,
        decision_seal_report_path=decision_report,
        criterion_plan_path=criterion_plan,
        criterion_report_path=criterion_report,
        pixi_lock_path=pixi_lock,
        output_root=output_root,
        publication_code_revision="a" * 40,
    )

    assert first == retry
    assert first.predictions_sealed_before_criterion_reveal
    assert not first.evaluation_model_is_validated_student_simulator
    assert first.completed_criterion_count == 23
    assert first.missing_criterion_count == 1
    assert (output_root / "external_study_design.pdf").read_bytes().startswith(b"%PDF")
    svg = (output_root / "external_study_design.svg").read_text(encoding="utf-8")
    assert "No human learner took part" in svg
    assert "Fix four predictions per case" in svg
    assert "No prediction request was allowed afterwards" in svg
    csv_text = (output_root / "study_design_stages.csv").read_text(encoding="utf-8")
    assert "sealed_predictions,condition_predictions,96" in csv_text
    assert "post_seal_criterion,missing_criterion_executions,1" in csv_text


def test_rejects_a_criterion_plan_created_before_the_seal(tmp_path: Path) -> None:
    decision_plan, decision_report, criterion_plan_path, criterion_report_path, pixi_lock = (
        _write_sources(tmp_path)
    )
    criterion_plan = ExternalCriterionPlan.model_validate_json(criterion_plan_path.read_bytes())
    changed = criterion_plan.model_dump(mode="python", exclude={"plan_hash"})
    changed["model_route"] = criterion_plan.model_route
    changed["created_at_utc"] = datetime(2026, 9, 1, 9, 0, tzinfo=UTC)
    invalid_plan = _hashed_model(ExternalCriterionPlan, changed, "plan_hash")
    invalid_plan_path = tmp_path / "invalid_criterion_plan.json"
    write_immutable_json(invalid_plan_path, invalid_plan)

    criterion_report = ExternalCriterionReport.model_validate_json(
        criterion_report_path.read_bytes()
    )
    changed_report = criterion_report.model_dump(mode="python", exclude={"report_hash"})
    changed_report["criterion_plan_hash"] = invalid_plan.plan_hash
    invalid_report = _hashed_model(
        ExternalCriterionReport,
        changed_report,
        "report_hash",
    )
    invalid_report_path = tmp_path / "invalid_criterion_report.json"
    write_immutable_json(invalid_report_path, invalid_report)

    with pytest.raises(StudyDesignFigureError, match="not created after"):
        render_study_design_figure(
            benchmark_design_path=BENCHMARK_DESIGN,
            decision_seal_plan_path=decision_plan,
            decision_seal_report_path=decision_report,
            criterion_plan_path=invalid_plan_path,
            criterion_report_path=invalid_report_path,
            pixi_lock_path=pixi_lock,
            output_root=tmp_path / "publication",
            publication_code_revision="a" * 40,
        )


def _write_sources(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    pixi_lock = root / "pixi.lock"
    pixi_lock.write_text("execution environment\n", encoding="utf-8")
    lock_hash = file_sha256(pixi_lock.read_bytes())
    created_at = datetime(2026, 9, 1, 9, 0, tzinfo=UTC)
    decision_plan_content = {
        "schema_version": 1,
        "schema_id": "benchmark.external_decision_seal_plan.v1",
        "run_id": RUN_ID,
        "benchmark_version": "v1",
        "source_manifest_hash": _sha(1),
        "public_projection_hash": _sha(2),
        "protocol_hash": _sha(3),
        "preflight_hash": _sha(4),
        "generation_report_hash": _sha(5),
        "recorded_responses_hash": _sha(6),
        "public_rating_report_hash": _sha(7),
        "methodology_clarification_hash": _sha(8),
        "inferential_hierarchy_hash": _sha(9),
        "analysis_specification_hash": _sha(10),
        "calibration_decision_hash": _sha(11),
        "model_route": ROUTE,
        "tracker_version": "simple-v1",
        "policy_version": "heuristic-v1",
        "system_prompt_version": "external-evaluation-system-v1",
        "system_prompt_sha256": _sha(12),
        "root_seed": 20260823,
        "code_revision": "decision-unit-test",
        "dirty_worktree": False,
        "pixi_lock_hash": lock_hash,
        "created_at_utc": created_at,
    }
    decision_plan = _hashed_model(
        ExternalDecisionSealPlan,
        decision_plan_content,
        "plan_hash",
    )
    sealed_at = created_at + timedelta(minutes=10)
    decision_report_content = {
        "schema_version": 1,
        "schema_id": "benchmark.external_decision_seal_report.v1",
        "run_id": RUN_ID,
        "benchmark_version": "v1",
        "seal_plan_hash": decision_plan.plan_hash,
        "decision_run_plan_hash": _sha(13),
        "generation_report_hash": decision_plan.generation_report_hash,
        "public_rating_report_hash": decision_plan.public_rating_report_hash,
        "methodology_clarification_hash": decision_plan.methodology_clarification_hash,
        "inferential_hierarchy_hash": decision_plan.inferential_hierarchy_hash,
        "planned_case_count": 24,
        "complete_case_count": 24,
        "precriterion_missing_case_count": 0,
        "invalid_case_count": 0,
        "evidence_execution_count": 24,
        "usable_evidence_count": 24,
        "prediction_count": 96,
        "global_seal_hash": _sha(14),
        "decision_publication_hash": _sha(15),
        "sealed_at_utc": sealed_at,
        "all_cases_complete": True,
        "gate_passed": True,
    }
    decision_report = _hashed_model(
        ExternalDecisionSealReport,
        decision_report_content,
        "report_hash",
    )
    criterion_plan_content = {
        "schema_version": 1,
        "schema_id": "benchmark.external_criterion_plan.v1",
        "run_id": RUN_ID,
        "benchmark_version": "v1",
        "protocol_hash": decision_plan.protocol_hash,
        "accounting_report_hash": _sha(16),
        "source_manifest_hash": decision_plan.source_manifest_hash,
        "public_projection_hash": decision_plan.public_projection_hash,
        "evaluator_projection_hash": _sha(17),
        "global_seal_hash": decision_report.global_seal_hash,
        "expected_case_count": 24,
        "model_route": ROUTE,
        "system_prompt_version": decision_plan.system_prompt_version,
        "system_prompt_sha256": decision_plan.system_prompt_sha256,
        "temperature": 0.0,
        "max_output_tokens": 512,
        "root_seed": decision_plan.root_seed,
        "request_spacing_seconds": 13.0,
        "prior_request_count": 48,
        "request_budget": 72,
        "code_revision": "criterion-unit-test",
        "pixi_lock_hash": lock_hash,
        "created_at_utc": sealed_at + timedelta(minutes=10),
    }
    criterion_plan = _hashed_model(
        ExternalCriterionPlan,
        criterion_plan_content,
        "plan_hash",
    )
    criterion_report_content = {
        "schema_version": 1,
        "schema_id": "benchmark.external_criterion_report.v1",
        "run_id": RUN_ID,
        "benchmark_version": "v1",
        "criterion_plan_hash": criterion_plan.plan_hash,
        "accounting_report_hash": criterion_plan.accounting_report_hash,
        "global_seal_hash": decision_report.global_seal_hash,
        "planned_case_count": 24,
        "provider_success_count": 24,
        "provider_missing_count": 0,
        "completed_execution_count": 23,
        "sandbox_missing_count": 1,
        "demonstrated_count": 14,
        "criterion_record_count": 24,
        "recorded_response_count": 24,
        "criterion_publication_hash": _sha(18),
        "criterion_record_root_hash": _sha(19),
        "completed_at_utc": criterion_plan.created_at_utc + timedelta(minutes=10),
        "gate_passed": True,
    }
    criterion_report = _hashed_model(
        ExternalCriterionReport,
        criterion_report_content,
        "report_hash",
    )

    paths = (
        root / "decision_plan.json",
        root / "decision_report.json",
        root / "criterion_plan.json",
        root / "criterion_report.json",
    )
    for path, model in zip(
        paths,
        (decision_plan, decision_report, criterion_plan, criterion_report),
        strict=True,
    ):
        write_immutable_json(path, model)
    return (*paths, pixi_lock)


def _hashed_model[ModelT: ContractModel](
    model: type[ModelT],
    content: Mapping[str, object],
    hash_field: str,
) -> ModelT:
    draft = model.model_construct(
        _fields_set=set(content),
        **content,
        **{hash_field: "0" * 64},
    )
    return model.model_validate(
        {**content, hash_field: model_content_hash(draft, exclude={hash_field})}
    )


def _sha(value: int) -> str:
    return f"{value:064x}"
