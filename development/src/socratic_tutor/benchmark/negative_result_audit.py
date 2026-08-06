"""Post-reveal audit of frozen benchmark and analysis inputs."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from pydantic import Field, ValidationError, model_validator

from socratic_tutor.benchmark.analysis_spec import (
    analysis_specification_hash,
    load_analysis_specification,
)
from socratic_tutor.benchmark.artifacts import write_immutable_json
from socratic_tutor.benchmark.calibration import UncalibratedDecisionReport
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.evaluator.loader import load_and_verify_manifest
from socratic_tutor.benchmark.evaluator.models import ArtifactClass, ManifestStatus
from socratic_tutor.benchmark.external_criterion import ExternalCriterionPlan
from socratic_tutor.benchmark.external_criterion_audit import ExternalCriterionIntegrityReport
from socratic_tutor.benchmark.external_seal import (
    ExternalDecisionSealPlan,
    ExternalDecisionSealReport,
)
from socratic_tutor.benchmark.hashing import file_sha256, model_content_hash
from socratic_tutor.contracts import ContractModel

_PROTECTED_PATHS = (
    "data/benchmarks/v1",
    "configs/benchmark/v1-analysis-spec.yaml",
    "configs/benchmark/v1-uncalibrated-decision.yaml",
)


class NegativeResultAuditError(ValueError):
    """Post-reveal inputs or Git history violate the frozen boundary."""


class GitHistoryInspector(Protocol):
    """Minimal Git operations needed by the audit."""

    def resolve_commit(self, repository_root: Path, revision: str) -> str: ...

    def changed_between(
        self,
        repository_root: Path,
        base_revision: str,
        target_revision: str,
        protected_paths: tuple[str, ...],
    ) -> tuple[str, ...]: ...

    def changed_in_worktree(
        self,
        repository_root: Path,
        revision: str,
        protected_paths: tuple[str, ...],
    ) -> tuple[str, ...]: ...


class SubprocessGitHistoryInspector:
    """Inspect Git without invoking a shell."""

    def resolve_commit(self, repository_root: Path, revision: str) -> str:
        output = self._run(repository_root, "rev-parse", "--verify", f"{revision}^{{commit}}")
        return output.strip()

    def changed_between(
        self,
        repository_root: Path,
        base_revision: str,
        target_revision: str,
        protected_paths: tuple[str, ...],
    ) -> tuple[str, ...]:
        output = self._run(
            repository_root,
            "diff",
            "--name-only",
            f"{base_revision}..{target_revision}",
            "--",
            *protected_paths,
        )
        return _path_lines(output)

    def changed_in_worktree(
        self,
        repository_root: Path,
        revision: str,
        protected_paths: tuple[str, ...],
    ) -> tuple[str, ...]:
        output = self._run(
            repository_root,
            "diff",
            "--name-only",
            revision,
            "--",
            *protected_paths,
        )
        return _path_lines(output)

    @staticmethod
    def _run(repository_root: Path, *arguments: str) -> str:
        try:
            completed = subprocess.run(
                ("git", *arguments),
                cwd=repository_root,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise NegativeResultAuditError("Could not inspect protected Git history") from error
        return completed.stdout


class NegativeResultAuditPlan(ContractModel):
    """Immutable inputs, revisions, and protected scope for the audit."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.negative_result_audit_plan.v1"] = (
        "benchmark.negative_result_audit_plan.v1"
    )
    decision_seal_plan_hash: Sha256
    decision_seal_report_hash: Sha256
    criterion_plan_hash: Sha256
    criterion_integrity_report_hash: Sha256
    source_manifest_hash: Sha256
    analysis_specification_hash: Sha256
    calibration_decision_hash: Sha256
    post_reveal_base_revision: str = Field(min_length=1)
    analysis_code_revision: str = Field(min_length=1)
    protected_paths: tuple[str, ...] = Field(min_length=3, max_length=3)
    analysis_pixi_lock_hash: Sha256
    created_at_utc: datetime
    plan_hash: Sha256

    @model_validator(mode="after")
    def validate_plan(self) -> NegativeResultAuditPlan:
        _require_utc(self.created_at_utc)
        if self.protected_paths != _PROTECTED_PATHS:
            raise ValueError("Negative-result audit protected paths are incomplete")
        if self.plan_hash != model_content_hash(self, exclude={"plan_hash"}):
            raise ValueError("Negative-result audit plan hash does not match its content")
        return self


class NegativeResultAuditReport(ContractModel):
    """Evidence that protected inputs remained unchanged after reveal."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.negative_result_audit_report.v1"] = (
        "benchmark.negative_result_audit_report.v1"
    )
    analysis_plan_hash: Sha256
    benchmark_version: Literal["v1"] = "v1"
    case_count: Literal[24]
    criterion_label_count: Literal[24]
    evidence_test_count: Literal[24]
    criterion_test_count: Literal[24]
    policy_threshold: float = Field(ge=0.0, le=1.0)
    missingness_rule: str = Field(min_length=1)
    confirmatory_exclusions: tuple[str, ...] = Field(min_length=1)
    post_reveal_changed_paths: tuple[str, ...]
    current_uncommitted_protected_paths: tuple[str, ...]
    decision_seal_dirty_worktree: bool
    audit_status: Literal["passed_clean_seal", "passed_with_dirty_seal_disclosed"]
    checks_passed: tuple[
        Literal["recursive_manifest_matches_decision_seal"],
        Literal["analysis_rules_match_decision_seal"],
        Literal["calibration_decision_matches_decision_seal"],
        Literal["criterion_reveal_followed_decision_seal"],
        Literal["criterion_integrity_report_passed"],
        Literal["protected_git_paths_unchanged_after_reveal"],
        Literal["protected_worktree_clean_at_audit"],
    ]
    interpretation_limits: tuple[
        Literal["seal_cleanliness_is_reported_from_original_plan"],
        Literal["content_hashes_bound_protected_inputs_not_the_entire_worktree"],
        Literal["post_hoc_analyses_remain_separate_from_primary_result"],
    ]
    network_calls_made: Literal[0] = 0
    sandbox_calls_made: Literal[0] = 0
    completed_at_utc: datetime
    report_hash: Sha256

    @model_validator(mode="after")
    def validate_report(self) -> NegativeResultAuditReport:
        _require_utc(self.completed_at_utc)
        if self.post_reveal_changed_paths or self.current_uncommitted_protected_paths:
            raise ValueError("A passing negative-result audit cannot contain protected changes")
        expected_status = (
            "passed_with_dirty_seal_disclosed"
            if self.decision_seal_dirty_worktree
            else "passed_clean_seal"
        )
        if self.audit_status != expected_status:
            raise ValueError("Negative-result audit status does not match seal cleanliness")
        if self.report_hash != model_content_hash(self, exclude={"report_hash"}):
            raise ValueError("Negative-result audit report hash does not match its content")
        return self


def run_negative_result_audit(
    *,
    repository_root: Path,
    manifest_path: Path,
    analysis_specification_path: Path,
    calibration_report_path: Path,
    decision_seal_plan_path: Path,
    decision_seal_report_path: Path,
    criterion_plan_path: Path,
    criterion_integrity_report_path: Path,
    pixi_lock_path: Path,
    output_root: Path,
    analysis_code_revision: str,
    created_at_utc: datetime,
    git_inspector: GitHistoryInspector | None = None,
) -> NegativeResultAuditReport:
    """Verify content hashes and Git history across the criterion reveal boundary."""

    _require_utc(created_at_utc)
    inspector = git_inspector or SubprocessGitHistoryInspector()
    repository = repository_root.resolve()
    seal_plan = _load_json(decision_seal_plan_path, ExternalDecisionSealPlan)
    seal_report = _load_json(decision_seal_report_path, ExternalDecisionSealReport)
    criterion_plan = _load_json(criterion_plan_path, ExternalCriterionPlan)
    criterion_integrity = _load_json(
        criterion_integrity_report_path, ExternalCriterionIntegrityReport
    )
    calibration = _load_json(calibration_report_path, UncalibratedDecisionReport)
    specification = load_analysis_specification(analysis_specification_path)
    manifest = load_and_verify_manifest(manifest_path)
    if manifest.status is not ManifestStatus.FROZEN or manifest.benchmark_version != "v1":
        raise NegativeResultAuditError("Negative-result audit requires the frozen v1 manifest")

    if manifest.manifest_hash is None:
        raise NegativeResultAuditError("Frozen manifest is missing its recursive hash")
    manifest_hash = manifest.manifest_hash
    try:
        lock_hash = file_sha256(pixi_lock_path.read_bytes())
    except OSError as error:
        raise NegativeResultAuditError("Could not hash negative-result audit input") from error
    specification_hash = analysis_specification_hash(specification)
    _validate_lineage(
        seal_plan=seal_plan,
        seal_report=seal_report,
        criterion_plan=criterion_plan,
        criterion_integrity=criterion_integrity,
        calibration=calibration,
        manifest_hash=manifest_hash,
        specification_hash=specification_hash,
    )

    base_revision = inspector.resolve_commit(repository, criterion_plan.code_revision)
    current_revision = inspector.resolve_commit(repository, analysis_code_revision)
    changed_paths = inspector.changed_between(
        repository, base_revision, current_revision, _PROTECTED_PATHS
    )
    worktree_paths = inspector.changed_in_worktree(repository, current_revision, _PROTECTED_PATHS)
    if changed_paths:
        raise NegativeResultAuditError(
            "Protected inputs changed after reveal: " + ", ".join(changed_paths)
        )
    if worktree_paths:
        raise NegativeResultAuditError(
            "Protected inputs have uncommitted changes: " + ", ".join(worktree_paths)
        )

    plan = _create_plan(
        seal_plan=seal_plan,
        seal_report=seal_report,
        criterion_plan=criterion_plan,
        criterion_integrity=criterion_integrity,
        manifest_hash=manifest_hash,
        specification_hash=specification_hash,
        calibration=calibration,
        base_revision=base_revision,
        current_revision=current_revision,
        lock_hash=lock_hash,
        created_at_utc=created_at_utc,
    )
    root = output_root.resolve()
    report_path = root / "negative_result_audit_report.json"
    if report_path.exists():
        report = _load_json(report_path, NegativeResultAuditReport)
        if report.analysis_plan_hash != plan.plan_hash:
            raise NegativeResultAuditError("Existing negative-result audit belongs to another plan")
        return report
    write_immutable_json(root / "negative_result_audit_plan.json", plan)

    file_counts = {
        artifact_class: sum(item.artifact_class is artifact_class for item in manifest.files)
        for artifact_class in ArtifactClass
    }
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.negative_result_audit_report.v1",
        "analysis_plan_hash": plan.plan_hash,
        "benchmark_version": "v1",
        "case_count": len(manifest.cases),
        "criterion_label_count": file_counts[ArtifactClass.LABEL_RATIONALE],
        "evidence_test_count": file_counts[ArtifactClass.EVIDENCE_TEST],
        "criterion_test_count": file_counts[ArtifactClass.CRITERION_TEST],
        "policy_threshold": specification.policy_threshold,
        "missingness_rule": specification.missingness_rule,
        "confirmatory_exclusions": specification.confirmatory_exclusions,
        "post_reveal_changed_paths": changed_paths,
        "current_uncommitted_protected_paths": worktree_paths,
        "decision_seal_dirty_worktree": seal_plan.dirty_worktree,
        "audit_status": (
            "passed_with_dirty_seal_disclosed" if seal_plan.dirty_worktree else "passed_clean_seal"
        ),
        "checks_passed": (
            "recursive_manifest_matches_decision_seal",
            "analysis_rules_match_decision_seal",
            "calibration_decision_matches_decision_seal",
            "criterion_reveal_followed_decision_seal",
            "criterion_integrity_report_passed",
            "protected_git_paths_unchanged_after_reveal",
            "protected_worktree_clean_at_audit",
        ),
        "interpretation_limits": (
            "seal_cleanliness_is_reported_from_original_plan",
            "content_hashes_bound_protected_inputs_not_the_entire_worktree",
            "post_hoc_analyses_remain_separate_from_primary_result",
        ),
        "network_calls_made": 0,
        "sandbox_calls_made": 0,
        "completed_at_utc": created_at_utc,
    }
    draft = NegativeResultAuditReport.model_construct(
        _fields_set=set(content), **content, report_hash="0" * 64
    )
    report = NegativeResultAuditReport.model_validate(
        {**content, "report_hash": model_content_hash(draft, exclude={"report_hash"})}
    )
    write_immutable_json(report_path, report)
    return report


def _validate_lineage(
    *,
    seal_plan: ExternalDecisionSealPlan,
    seal_report: ExternalDecisionSealReport,
    criterion_plan: ExternalCriterionPlan,
    criterion_integrity: ExternalCriterionIntegrityReport,
    calibration: UncalibratedDecisionReport,
    manifest_hash: Sha256,
    specification_hash: Sha256,
) -> None:
    if seal_report.seal_plan_hash != seal_plan.plan_hash:
        raise NegativeResultAuditError("Decision-seal report belongs to another plan")
    if manifest_hash != seal_plan.source_manifest_hash:
        raise NegativeResultAuditError("Recursive manifest differs from the decision seal")
    if specification_hash != seal_plan.analysis_specification_hash:
        raise NegativeResultAuditError("Analysis rules differ from the decision seal")
    if calibration.decision.decision_hash != seal_plan.calibration_decision_hash:
        raise NegativeResultAuditError("Calibration decision differs from the decision seal")
    if calibration.analysis_specification_hash != specification_hash:
        raise NegativeResultAuditError("Calibration report uses different analysis rules")
    if (
        criterion_plan.source_manifest_hash != manifest_hash
        or criterion_plan.global_seal_hash != seal_report.global_seal_hash
        or criterion_integrity.criterion_plan_hash != criterion_plan.plan_hash
        or criterion_integrity.global_seal_hash != seal_report.global_seal_hash
    ):
        raise NegativeResultAuditError("Criterion reveal lineage is inconsistent")
    if criterion_plan.created_at_utc < seal_report.sealed_at_utc:
        raise NegativeResultAuditError("Criterion access preceded the decision seal")
    if not criterion_integrity.gate_passed:
        raise NegativeResultAuditError("Criterion integrity audit did not pass")


def _create_plan(
    *,
    seal_plan: ExternalDecisionSealPlan,
    seal_report: ExternalDecisionSealReport,
    criterion_plan: ExternalCriterionPlan,
    criterion_integrity: ExternalCriterionIntegrityReport,
    manifest_hash: Sha256,
    specification_hash: Sha256,
    calibration: UncalibratedDecisionReport,
    base_revision: str,
    current_revision: str,
    lock_hash: Sha256,
    created_at_utc: datetime,
) -> NegativeResultAuditPlan:
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.negative_result_audit_plan.v1",
        "decision_seal_plan_hash": seal_plan.plan_hash,
        "decision_seal_report_hash": seal_report.report_hash,
        "criterion_plan_hash": criterion_plan.plan_hash,
        "criterion_integrity_report_hash": criterion_integrity.report_hash,
        "source_manifest_hash": manifest_hash,
        "analysis_specification_hash": specification_hash,
        "calibration_decision_hash": calibration.decision.decision_hash,
        "post_reveal_base_revision": base_revision,
        "analysis_code_revision": current_revision,
        "protected_paths": _PROTECTED_PATHS,
        "analysis_pixi_lock_hash": lock_hash,
        "created_at_utc": created_at_utc,
    }
    draft = NegativeResultAuditPlan.model_construct(
        _fields_set=set(content), **content, plan_hash="0" * 64
    )
    return NegativeResultAuditPlan.model_validate(
        {**content, "plan_hash": model_content_hash(draft, exclude={"plan_hash"})}
    )


def _load_json[ModelT: ContractModel](path: Path, model: type[ModelT]) -> ModelT:
    try:
        return model.model_validate_json(path.read_bytes())
    except (OSError, ValidationError) as error:
        raise NegativeResultAuditError(f"Could not verify audit source: {path}") from error


def _path_lines(output: str) -> tuple[str, ...]:
    return tuple(sorted({line.strip() for line in output.splitlines() if line.strip()}))


def _require_utc(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("Negative-result audit timestamp must be UTC")
