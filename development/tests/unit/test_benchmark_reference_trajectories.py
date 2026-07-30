"""Held-out deterministic reference-path guards."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pytest

from socratic_tutor.benchmark.evaluator.loader import load_and_verify_manifest
from socratic_tutor.benchmark.evaluator.projection import project_public_manifest
from socratic_tutor.benchmark.generation import ModelRoute, SamplingConfig
from socratic_tutor.benchmark.hashing import file_sha256
from socratic_tutor.benchmark.public.offline import (
    OfflineDecisionCase,
    OfflineDecisionError,
    OfflineDecisionPlan,
    run_offline_decision_phase,
)
from socratic_tutor.contracts import Evidence, EvidenceCategory

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = WORKSPACE_ROOT / "data" / "benchmarks" / "v1"
MANIFEST_PATH = BENCHMARK_ROOT / "manifest.yaml"


def _reference_plan(
    *, mode: Literal["development_offline", "reference_heldout"]
) -> OfflineDecisionPlan:
    authored = load_and_verify_manifest(MANIFEST_PATH)
    public = project_public_manifest(authored)
    prompt = (BENCHMARK_ROOT / "prompts" / "student-system-v1.md").read_text(encoding="utf-8")
    return OfflineDecisionPlan(
        mode=mode,
        run_id="reference-path-test",
        sample_ids=("reference-001",),
        model_route=ModelRoute(
            provider="recorded_fixture",
            model="authored-deterministic",
        ),
        sampling=SamplingConfig(temperature=0.0, max_output_tokens=512, seed=20260823),
        system_prompt_version="student-system-v1",
        system_prompt_ref="prompts/student-system-v1.md",
        system_prompt_sha256=file_sha256(prompt.encode("utf-8")),
        tracker_version="simple-v1",
        policy_version="heuristic-v1",
        code_revision="reference-path-test",
        dirty_worktree=False,
        environment_lock_hash="0" * 64,
        root_seed=20260823,
        created_at_utc=datetime(2026, 8, 23, tzinfo=UTC),
        cases=tuple(
            OfflineDecisionCase(
                case_id=case.case_id,
                public_response="A deterministic reference response for this public task.",
                evidence_response="def reference_probe():\n    return True\n",
                public_evidence=Evidence(
                    category=EvidenceCategory.CORRECT,
                    confidence=0.8,
                    rationale="Reference-path scope test only.",
                ),
                probe_passed=1,
                probe_failed=0,
            )
            for case in public.cases
        ),
    )


def test_reference_mode_runs_all_held_out_cases_without_network(tmp_path: Path) -> None:
    authored = load_and_verify_manifest(MANIFEST_PATH)
    public = project_public_manifest(authored)

    summary = run_offline_decision_phase(
        manifest=public,
        benchmark_root=BENCHMARK_ROOT,
        plan=_reference_plan(mode="reference_heldout"),
        output_root=tmp_path,
    )

    assert summary.case_count == 24
    assert summary.sample_count == 24
    assert summary.prediction_count == 96
    assert summary.recorded_response_count == 48
    assert (tmp_path / "decision" / "commitment_manifest.json").exists()
    assert (tmp_path / "datasets" / "condition_predictions.parquet").exists()


def test_development_mode_still_rejects_held_out_cases(tmp_path: Path) -> None:
    authored = load_and_verify_manifest(MANIFEST_PATH)
    public = project_public_manifest(authored)

    with pytest.raises(OfflineDecisionError, match="held-out generation is disabled"):
        run_offline_decision_phase(
            manifest=public,
            benchmark_root=BENCHMARK_ROOT,
            plan=_reference_plan(mode="development_offline"),
            output_root=tmp_path,
        )


def test_reference_mode_rejects_non_reference_route(tmp_path: Path) -> None:
    authored = load_and_verify_manifest(MANIFEST_PATH)
    public = project_public_manifest(authored)
    plan = _reference_plan(mode="reference_heldout").model_copy(
        update={"model_route": ModelRoute(provider="cerebras", model="gpt-oss-120b")}
    )

    with pytest.raises(OfflineDecisionError, match="recorded_fixture/authored-deterministic"):
        run_offline_decision_phase(
            manifest=public,
            benchmark_root=BENCHMARK_ROOT,
            plan=plan,
            output_root=tmp_path,
        )
