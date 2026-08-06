from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import socratic_tutor.benchmark.negative_result_audit as audit_module
from socratic_tutor.benchmark.analysis_spec import AnalysisSpecification
from socratic_tutor.benchmark.evaluator.models import ArtifactClass, ManifestStatus
from socratic_tutor.benchmark.negative_result_audit import (
    GitHistoryInspector,
    NegativeResultAuditError,
    run_negative_result_audit,
)

STAMP = datetime(2026, 9, 3, 10, 0, tzinfo=UTC)
REVISION = "a" * 40
BASE_REVISION = "b" * 40


class FakeGitInspector:
    def __init__(
        self,
        *,
        changed_between: tuple[str, ...] = (),
        changed_in_worktree: tuple[str, ...] = (),
    ) -> None:
        self._changed_between = changed_between
        self._changed_in_worktree = changed_in_worktree

    def resolve_commit(self, repository_root: Path, revision: str) -> str:
        del repository_root
        return BASE_REVISION if revision == "criterion-revision" else revision

    def changed_between(
        self,
        repository_root: Path,
        base_revision: str,
        target_revision: str,
        protected_paths: tuple[str, ...],
    ) -> tuple[str, ...]:
        del repository_root, base_revision, target_revision
        assert protected_paths == (
            "data/benchmarks/v1",
            "configs/benchmark/v1-analysis-spec.yaml",
            "configs/benchmark/v1-uncalibrated-decision.yaml",
        )
        return self._changed_between

    def changed_in_worktree(
        self,
        repository_root: Path,
        revision: str,
        protected_paths: tuple[str, ...],
    ) -> tuple[str, ...]:
        del repository_root, revision, protected_paths
        return self._changed_in_worktree


def test_reports_unchanged_inputs_and_discloses_dirty_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _sources(tmp_path, monkeypatch, dirty_seal=True)

    report = run_negative_result_audit(
        **paths,
        output_root=tmp_path / "audit",
        analysis_code_revision=REVISION,
        created_at_utc=STAMP,
        git_inspector=cast(GitHistoryInspector, FakeGitInspector()),
    )

    assert report.case_count == 24
    assert report.criterion_label_count == 24
    assert report.evidence_test_count == 24
    assert report.criterion_test_count == 24
    assert report.policy_threshold == 0.6
    assert report.post_reveal_changed_paths == ()
    assert report.current_uncommitted_protected_paths == ()
    assert report.audit_status == "passed_with_dirty_seal_disclosed"
    assert report.decision_seal_dirty_worktree is True
    assert report.network_calls_made == 0
    assert report.sandbox_calls_made == 0


def test_rejects_post_reveal_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _sources(tmp_path, monkeypatch, dirty_seal=True)
    inspector = FakeGitInspector(
        changed_between=("data/benchmarks/v1/evidence/example-tests.yaml",)
    )

    with pytest.raises(NegativeResultAuditError, match="changed after reveal"):
        run_negative_result_audit(
            **paths,
            output_root=tmp_path / "audit",
            analysis_code_revision=REVISION,
            created_at_utc=STAMP,
            git_inspector=cast(GitHistoryInspector, inspector),
        )


def test_rejects_uncommitted_protected_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _sources(tmp_path, monkeypatch, dirty_seal=True)
    inspector = FakeGitInspector(changed_in_worktree=("configs/benchmark/v1-analysis-spec.yaml",))

    with pytest.raises(NegativeResultAuditError, match="uncommitted changes"):
        run_negative_result_audit(
            **paths,
            output_root=tmp_path / "audit",
            analysis_code_revision=REVISION,
            created_at_utc=STAMP,
            git_inspector=cast(GitHistoryInspector, inspector),
        )


def test_rejects_analysis_rules_that_differ_from_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _sources(tmp_path, monkeypatch, dirty_seal=False, wrong_analysis_hash=True)

    with pytest.raises(NegativeResultAuditError, match="Analysis rules differ"):
        run_negative_result_audit(
            **paths,
            output_root=tmp_path / "audit",
            analysis_code_revision=REVISION,
            created_at_utc=STAMP,
            git_inspector=cast(GitHistoryInspector, FakeGitInspector()),
        )


def _sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    dirty_seal: bool,
    wrong_analysis_hash: bool = False,
) -> dict[str, Path]:
    names = (
        "manifest",
        "analysis",
        "calibration",
        "seal-plan",
        "seal-report",
        "criterion-plan",
        "criterion-integrity",
        "pixi-lock",
    )
    paths = {name.replace("-", "_") + "_path": tmp_path / name for name in names}
    for path in paths.values():
        path.write_text(path.name, encoding="utf-8")
    manifest_hash = "1" * 64
    specification = AnalysisSpecification(
        benchmark_version="v1",
        minimum_interpretable_effect=0.1,
        policy_threshold=0.6,
        bootstrap_resamples=10_000,
        permutation_resamples=10_000,
        random_seed=1,
        aggregation_version="case-v1",
        confirmatory_exclusions=("exclude missing case outcomes",),
        exploratory_metrics=("descriptive subgroup summaries",),
    )
    specification_hash = audit_module.analysis_specification_hash(specification)
    calibration_hash = "3" * 64
    global_seal_hash = "4" * 64
    seal_plan_hash = "5" * 64
    criterion_plan_hash = "6" * 64
    seal_plan = SimpleNamespace(
        plan_hash=seal_plan_hash,
        source_manifest_hash=manifest_hash,
        analysis_specification_hash=("9" * 64 if wrong_analysis_hash else specification_hash),
        calibration_decision_hash=calibration_hash,
        dirty_worktree=dirty_seal,
    )
    seal_report = SimpleNamespace(
        plan_hash="unused",
        seal_plan_hash=seal_plan_hash,
        global_seal_hash=global_seal_hash,
        sealed_at_utc=STAMP - timedelta(hours=2),
        report_hash="7" * 64,
    )
    criterion_plan = SimpleNamespace(
        plan_hash=criterion_plan_hash,
        source_manifest_hash=manifest_hash,
        global_seal_hash=global_seal_hash,
        code_revision="criterion-revision",
        created_at_utc=STAMP - timedelta(hours=1),
    )
    criterion_integrity = SimpleNamespace(
        criterion_plan_hash=criterion_plan_hash,
        global_seal_hash=global_seal_hash,
        gate_passed=True,
        report_hash="8" * 64,
    )
    calibration = SimpleNamespace(
        report_hash="0" * 64,
        analysis_specification_hash=specification_hash,
        decision=SimpleNamespace(decision_hash=calibration_hash),
    )
    manifest = SimpleNamespace(
        status=ManifestStatus.FROZEN,
        benchmark_version="v1",
        manifest_hash=manifest_hash,
        cases=tuple(range(24)),
        files=tuple(
            SimpleNamespace(artifact_class=artifact_class)
            for artifact_class in (
                *([ArtifactClass.LABEL_RATIONALE] * 24),
                *([ArtifactClass.EVIDENCE_TEST] * 24),
                *([ArtifactClass.CRITERION_TEST] * 24),
            )
        ),
    )
    loaded: dict[type[object], object] = {
        audit_module.ExternalDecisionSealPlan: seal_plan,
        audit_module.ExternalDecisionSealReport: seal_report,
        audit_module.ExternalCriterionPlan: criterion_plan,
        audit_module.ExternalCriterionIntegrityReport: criterion_integrity,
        audit_module.UncalibratedDecisionReport: calibration,
    }

    def fake_load_json(path: Path, model: type[object]) -> object:
        del path
        return loaded[model]

    def fake_load_analysis_specification(path: Path) -> AnalysisSpecification:
        del path
        return specification

    def fake_load_manifest(path: Path) -> object:
        del path
        return manifest

    monkeypatch.setattr(audit_module, "_load_json", fake_load_json)
    monkeypatch.setattr(
        audit_module, "load_analysis_specification", fake_load_analysis_specification
    )
    monkeypatch.setattr(audit_module, "load_and_verify_manifest", fake_load_manifest)
    return {
        "repository_root": tmp_path,
        "manifest_path": paths["manifest_path"],
        "analysis_specification_path": paths["analysis_path"],
        "calibration_report_path": paths["calibration_path"],
        "decision_seal_plan_path": paths["seal_plan_path"],
        "decision_seal_report_path": paths["seal_report_path"],
        "criterion_plan_path": paths["criterion_plan_path"],
        "criterion_integrity_report_path": paths["criterion_integrity_path"],
        "pixi_lock_path": paths["pixi_lock_path"],
    }
