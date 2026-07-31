"""Network-free, non-empirical integrity replay for the frozen v1 corpus."""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from socratic_tutor.benchmark.artifacts import write_immutable_json
from socratic_tutor.benchmark.evaluator.loader import load_and_verify_manifest
from socratic_tutor.benchmark.evaluator.models import ManifestStatus
from socratic_tutor.benchmark.evaluator.offline import (
    OfflineCriterionCase,
    OfflineCriterionPlan,
    run_offline_criterion_phase,
)
from socratic_tutor.benchmark.evaluator.projection import (
    project_evaluator_manifest,
    project_public_manifest,
)
from socratic_tutor.benchmark.evaluator.reporting import evaluate_published_run
from socratic_tutor.benchmark.evaluator.scoring import CalibrationStatus
from socratic_tutor.benchmark.generation import ModelRoute, SamplingConfig
from socratic_tutor.benchmark.hashing import file_sha256, model_content_hash
from socratic_tutor.benchmark.public.models import PublicBenchmarkManifest
from socratic_tutor.benchmark.public.offline import (
    OfflineDecisionCase,
    OfflineDecisionPlan,
    run_offline_decision_phase,
)
from socratic_tutor.contracts import ContractModel, Evidence, EvidenceCategory


class ReferenceIntegrityPlan(ContractModel):
    """Pinned inputs for one all-case engineering replay, never a primary result."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.reference_integrity_plan.v1"] = (
        "benchmark.reference_integrity_plan.v1"
    )
    benchmark_version: Literal["v1"] = "v1"
    source_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_id: str = Field(min_length=1)
    code_revision: str = Field(min_length=1)
    dirty_worktree: bool
    environment_lock_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    root_seed: int = Field(ge=0, le=2**32 - 1)
    created_at_utc: datetime
    fixture_version: Literal["reference-integrity-v1"] = "reference-integrity-v1"

    @model_validator(mode="after")
    def require_utc_timestamp(self) -> "ReferenceIntegrityPlan":
        if self.created_at_utc.tzinfo is None or self.created_at_utc.utcoffset() != timedelta(0):
            raise ValueError("Reference integrity plan timestamp must use UTC")
        return self


class ReferenceIntegrityReport(ContractModel):
    """Content-addressed summary of a complete non-network integrity replay."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.reference_integrity_report.v1"] = (
        "benchmark.reference_integrity_report.v1"
    )
    benchmark_version: Literal["v1"] = "v1"
    run_id: str
    source_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision_summary_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    criterion_summary_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_report_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_count: int = Field(ge=1)
    condition_prediction_count: int = Field(ge=1)
    criterion_record_count: int = Field(ge=1)
    integrity_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_integrity_hash(self) -> "ReferenceIntegrityReport":
        if self.integrity_hash != model_content_hash(self, exclude={"integrity_hash"}):
            raise ValueError("Reference integrity report hash does not match its content")
        return self


def load_reference_integrity_plan(path: Path) -> ReferenceIntegrityPlan:
    """Load one versioned integrity plan without accepting hidden defaults."""

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"Could not read reference integrity plan: {path}") from error
    return ReferenceIntegrityPlan.model_validate(raw)


def run_reference_integrity_replay(
    *,
    plan: ReferenceIntegrityPlan,
    manifest_path: Path,
    output_root: Path,
) -> ReferenceIntegrityReport:
    """Exercise every v1 route with labelled fixed fixtures and no network calls."""

    authored = load_and_verify_manifest(manifest_path)
    if authored.status is not ManifestStatus.FROZEN:
        raise ValueError("Reference integrity replay requires a frozen manifest")
    if authored.benchmark_version != plan.benchmark_version:
        raise ValueError("Reference integrity plan belongs to another benchmark version")
    if authored.manifest_hash != plan.source_manifest_hash:
        raise ValueError("Reference integrity plan belongs to a different frozen manifest")

    root = output_root.resolve()
    benchmark_root = manifest_path.resolve().parent
    created_at = plan.created_at_utc
    public = project_public_manifest(authored, projected_at_utc=created_at)
    write_immutable_json(root / "public_manifest.json", public)
    write_immutable_json(root / "reference_integrity_plan.json", plan)

    decision_plan = _decision_plan(plan, public, benchmark_root)
    write_immutable_json(root / "offline_decision_plan.json", decision_plan)
    decision_summary = run_offline_decision_phase(
        manifest=public,
        benchmark_root=benchmark_root,
        plan=decision_plan,
        output_root=root,
    )

    evaluator = project_evaluator_manifest(authored, public, projected_at_utc=created_at)
    write_immutable_json(root / "evaluator_manifest.json", evaluator)
    criterion_plan = _criterion_plan(plan, public, benchmark_root)
    write_immutable_json(root / "offline_criterion_plan.json", criterion_plan)
    criterion_summary = run_offline_criterion_phase(
        public_manifest=public,
        evaluator_manifest=evaluator,
        benchmark_root=benchmark_root,
        plan=criterion_plan,
        output_root=root,
    )
    evaluation = evaluate_published_run(
        dataset_root=root / "datasets",
        output_path=root / "evaluation_summary.json",
    )
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.reference_integrity_report.v1",
        "benchmark_version": "v1",
        "run_id": plan.run_id,
        "source_manifest_hash": plan.source_manifest_hash,
        "reference_plan_hash": model_content_hash(plan),
        "decision_summary_hash": model_content_hash(decision_summary),
        "criterion_summary_hash": model_content_hash(criterion_summary),
        "evaluation_report_hash": evaluation.report_hash,
        "case_count": decision_summary.case_count,
        "condition_prediction_count": decision_summary.prediction_count,
        "criterion_record_count": criterion_summary.criterion_count,
    }
    draft = ReferenceIntegrityReport.model_construct(
        _fields_set=set(content),
        **content,
        integrity_hash="0" * 64,
    )
    report = ReferenceIntegrityReport.model_validate(
        {**content, "integrity_hash": model_content_hash(draft, exclude={"integrity_hash"})}
    )
    write_immutable_json(root / "reference_integrity_report.json", report)
    return report


def _decision_plan(
    plan: ReferenceIntegrityPlan,
    public: PublicBenchmarkManifest,
    benchmark_root: Path,
) -> OfflineDecisionPlan:
    prompt_path = benchmark_root / "prompts" / "student-system-v1.md"
    prompt_hash = file_sha256(prompt_path.read_bytes())
    return OfflineDecisionPlan(
        mode="reference_heldout",
        run_id=plan.run_id,
        sample_ids=("reference-001",),
        model_route=ModelRoute(provider="recorded_fixture", model="authored-deterministic"),
        sampling=SamplingConfig(temperature=0.0, max_output_tokens=512, seed=plan.root_seed),
        system_prompt_version="student-system-v1",
        system_prompt_ref="prompts/student-system-v1.md",
        system_prompt_sha256=prompt_hash,
        tracker_version="simple-v1",
        policy_version="heuristic-v1",
        code_revision=plan.code_revision,
        dirty_worktree=plan.dirty_worktree,
        environment_lock_hash=plan.environment_lock_hash,
        root_seed=plan.root_seed,
        created_at_utc=plan.created_at_utc,
        cases=tuple(_decision_case(case.case_id, index) for index, case in enumerate(public.cases)),
    )


def _criterion_plan(
    plan: ReferenceIntegrityPlan,
    public: PublicBenchmarkManifest,
    benchmark_root: Path,
) -> OfflineCriterionPlan:
    prompt_hash = file_sha256((benchmark_root / "prompts" / "student-system-v1.md").read_bytes())
    return OfflineCriterionPlan(
        mode="reference_heldout",
        run_id=plan.run_id,
        model_route=ModelRoute(provider="recorded_fixture", model="authored-deterministic"),
        sampling=SamplingConfig(temperature=0.0, max_output_tokens=512, seed=plan.root_seed),
        system_prompt_version="student-system-v1",
        system_prompt_ref="prompts/student-system-v1.md",
        system_prompt_sha256=prompt_hash,
        tracker_version="simple-v1",
        calibration_status=CalibrationStatus.UNCALIBRATED_SCORE,
        policy_threshold=0.6,
        calibration_outcomes=(True, False),
        calibration_owner="reference-integrity-fixture",
        scorer_version="scorer-v1",
        constant_baseline_version="constant-prevalence-v1",
        probe_baseline_version="probe-only-simple-v1",
        aggregation_version="case-mean-v1",
        created_at_utc=plan.created_at_utc - timedelta(seconds=1),
        cases=tuple(
            _criterion_case(case.case_id, index) for index, case in enumerate(public.cases)
        ),
    )


def _decision_case(case_id: str, index: int) -> OfflineDecisionCase:
    category = (
        EvidenceCategory.CORRECT,
        EvidenceCategory.MISCONCEPTION,
        EvidenceCategory.UNCERTAIN,
        EvidenceCategory.CONFLICTING,
    )[index % 4]
    return OfflineDecisionCase(
        case_id=case_id,
        public_response=f"Reference integrity fixture response for {case_id}.",
        evidence_response="def reference_integrity_evidence() -> bool:\n    return True\n",
        public_evidence=Evidence(
            category=category,
            confidence=0.75,
            rationale="Fixed non-empirical integrity fixture.",
        ),
        probe_passed=1 if index % 2 == 0 else 0,
        probe_failed=0 if index % 2 == 0 else 1,
    )


def _criterion_case(case_id: str, index: int) -> OfflineCriterionCase:
    return OfflineCriterionCase(
        case_id=case_id,
        criterion_response=f"# Reference integrity criterion fixture for {case_id}\n",
        criterion_passed=1 if index % 2 == 0 else 0,
        criterion_failed=0 if index % 2 == 0 else 1,
    )
