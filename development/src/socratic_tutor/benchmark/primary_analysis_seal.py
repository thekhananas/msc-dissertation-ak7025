"""Close the primary benchmark analysis with one verified artifact manifest."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import Field, ValidationError, model_validator

from socratic_tutor.benchmark.artifacts import write_immutable_json
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import canonical_sha256, file_sha256, model_content_hash
from socratic_tutor.contracts import ContractModel


class PrimaryAnalysisSealError(ValueError):
    """The analysis artifacts do not form one internally consistent result."""


@dataclass(frozen=True)
class _SourceSpec:
    role: str
    relative_path: str
    schema_id: str
    hash_field: Literal["plan_hash", "report_hash", "summary_hash", "manifest_hash"]


PRIMARY_ANALYSIS_SOURCE_SPECS = (
    _SourceSpec(
        "scoring_plan",
        "scoring-v1/external_scoring_plan.json",
        "benchmark.external_scoring_plan.v1",
        "plan_hash",
    ),
    _SourceSpec(
        "scoring_summary",
        "scoring-v1/external_scoring_summary.json",
        "benchmark.external_scoring_summary.v1",
        "summary_hash",
    ),
    _SourceSpec(
        "primary_plan",
        "primary-v1/primary_analysis_plan.json",
        "benchmark.primary_analysis_plan.v1",
        "plan_hash",
    ),
    _SourceSpec(
        "primary_report",
        "primary-v1/primary_analysis_report.json",
        "benchmark.primary_analysis_report.v1",
        "report_hash",
    ),
    _SourceSpec(
        "secondary_plan",
        "secondary-v1/secondary_analysis_plan.json",
        "benchmark.secondary_analysis_plan.v1",
        "plan_hash",
    ),
    _SourceSpec(
        "secondary_report",
        "secondary-v1/secondary_analysis_report.json",
        "benchmark.secondary_analysis_report.v1",
        "report_hash",
    ),
    _SourceSpec(
        "specificity_report",
        "secondary-v1/evidence_specificity_report.json",
        "benchmark.evidence_specificity_report.v1",
        "report_hash",
    ),
    _SourceSpec(
        "diagnostics_plan",
        "diagnostics-v1/external_diagnostics_plan.json",
        "benchmark.external_diagnostics_plan.v1",
        "plan_hash",
    ),
    _SourceSpec(
        "diagnostics_report",
        "diagnostics-v1/external_diagnostics_report.json",
        "benchmark.external_diagnostics_report.v1",
        "report_hash",
    ),
    _SourceSpec(
        "missingness_plan",
        "missingness-v1/missingness_sensitivity_plan.json",
        "benchmark.missingness_sensitivity_plan.v1",
        "plan_hash",
    ),
    _SourceSpec(
        "missingness_report",
        "missingness-v1/missingness_sensitivity_report.json",
        "benchmark.missingness_sensitivity_report.v1",
        "report_hash",
    ),
    _SourceSpec(
        "rating_reliability_plan",
        "rating-reliability-v1/rating_reliability_plan.json",
        "benchmark.rating_reliability_plan.v1",
        "plan_hash",
    ),
    _SourceSpec(
        "rating_reliability_report",
        "rating-reliability-v1/rating_reliability_report.json",
        "benchmark.rating_reliability_report.v1",
        "report_hash",
    ),
    _SourceSpec(
        "failure_reliability_plan",
        "failure-review-reliability-v1/failure_review_reliability_plan.json",
        "benchmark.failure_review_reliability_plan.v1",
        "plan_hash",
    ),
    _SourceSpec(
        "failure_reliability_report",
        "failure-review-reliability-v1/failure_review_reliability_report.json",
        "benchmark.failure_review_reliability_report.v1",
        "report_hash",
    ),
    _SourceSpec(
        "resource_plan",
        "resource-reconciliation-v1/resource_reconciliation_plan.json",
        "benchmark.resource_reconciliation_plan.v1",
        "plan_hash",
    ),
    _SourceSpec(
        "resource_report",
        "resource-reconciliation-v1/resource_reconciliation_report.json",
        "benchmark.resource_reconciliation_report.v1",
        "report_hash",
    ),
    _SourceSpec(
        "negative_audit_plan",
        "negative-result-audit-v1/negative_result_audit_plan.json",
        "benchmark.negative_result_audit_plan.v1",
        "plan_hash",
    ),
    _SourceSpec(
        "negative_audit_report",
        "negative-result-audit-v1/negative_result_audit_report.json",
        "benchmark.negative_result_audit_report.v1",
        "report_hash",
    ),
    _SourceSpec(
        "correction_plan",
        "probe-correction-v1/probe_correction_analysis_plan.json",
        "benchmark.probe_correction_analysis_plan.v1",
        "plan_hash",
    ),
    _SourceSpec(
        "correction_report",
        "probe-correction-v1/probe_correction_sensitivity_report.json",
        "benchmark.probe_correction_sensitivity_report.v1",
        "report_hash",
    ),
    _SourceSpec(
        "dependence_plan",
        "dependence-v1/dependence_sensitivity_plan.json",
        "benchmark.dependence_sensitivity_plan.v1",
        "plan_hash",
    ),
    _SourceSpec(
        "dependence_report",
        "dependence-v1/dependence_sensitivity_report.json",
        "benchmark.dependence_sensitivity_report.v1",
        "report_hash",
    ),
    _SourceSpec(
        "interpretation_plan",
        "result-interpretation-v1/result_interpretation_plan.json",
        "benchmark.result_interpretation_plan.v1",
        "plan_hash",
    ),
    _SourceSpec(
        "interpretation_report",
        "result-interpretation-v1/result_interpretation_report.json",
        "benchmark.result_interpretation_report.v1",
        "report_hash",
    ),
    _SourceSpec(
        "final_replay_plan",
        "final-replay-v1/final_replay_plan.json",
        "benchmark.final_replay_plan.v1",
        "plan_hash",
    ),
    _SourceSpec(
        "final_replay_report",
        "final-replay-v1/final_replay_report.json",
        "benchmark.final_replay_report.v1",
        "report_hash",
    ),
    _SourceSpec(
        "publication_figure_manifest",
        "dependence-v1/publication-v5/dependence_figure_manifest.json",
        "benchmark.dependence_figure_manifest.v1",
        "manifest_hash",
    ),
)

PRIMARY_ANALYSIS_ARTIFACT_COUNT = len(PRIMARY_ANALYSIS_SOURCE_SPECS)


class AnalysisArtifactBinding(ContractModel):
    """One verified, content-addressed analysis artifact."""

    role: str = Field(min_length=1)
    relative_path: str = Field(min_length=1)
    schema_id: str = Field(min_length=1)
    content_hash: Sha256
    file_sha256: Sha256


class PrimaryAnalysisManifest(ContractModel):
    """One closure record for the complete primary-analysis evidence chain."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.primary_analysis_manifest.v1"] = (
        "benchmark.primary_analysis_manifest.v1"
    )
    benchmark_version: Literal["v1"] = "v1"
    run_id: str = Field(min_length=1)
    artifacts: tuple[AnalysisArtifactBinding, ...] = Field(
        min_length=PRIMARY_ANALYSIS_ARTIFACT_COUNT,
        max_length=PRIMARY_ANALYSIS_ARTIFACT_COUNT,
    )
    primary_conclusion: Literal["inconclusive"] = "inconclusive"
    specificity_conclusion: Literal["specific"] = "specific"
    closure_status: Literal["closed_with_inconclusive_primary_result"] = (
        "closed_with_inconclusive_primary_result"
    )
    eligible_case_count: Literal[23] = 23
    missing_case_count: Literal[1] = 1
    final_replay_gate_passed: Literal[True] = True
    negative_result_audit_status: Literal["passed_with_dirty_seal_disclosed"] = (
        "passed_with_dirty_seal_disclosed"
    )
    result_scope: Literal["one_pinned_model_run_on_one_fixed_authored_corpus"] = (
        "one_pinned_model_run_on_one_fixed_authored_corpus"
    )
    human_learning_claim_allowed: Literal[False] = False
    tutoring_efficacy_claim_allowed: Literal[False] = False
    reduced_cognitive_offloading_claim_allowed: Literal[False] = False
    network_calls_made: Literal[0] = 0
    sandbox_calls_made: Literal[0] = 0
    seal_code_revision: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    seal_pixi_lock_hash: Sha256
    created_at_utc: datetime
    manifest_hash: Sha256

    @model_validator(mode="after")
    def validate_manifest(self) -> PrimaryAnalysisManifest:
        _require_utc(self.created_at_utc)
        roles = tuple(binding.role for binding in self.artifacts)
        if roles != tuple(spec.role for spec in PRIMARY_ANALYSIS_SOURCE_SPECS):
            raise ValueError("Primary-analysis artifact roles are incomplete or out of order")
        if self.manifest_hash != model_content_hash(self, exclude={"manifest_hash"}):
            raise ValueError("Primary-analysis manifest hash does not match its content")
        return self


def seal_primary_analysis(
    *,
    analysis_root: Path,
    pixi_lock_path: Path,
    output_root: Path,
    code_revision: str,
    created_at_utc: datetime,
) -> PrimaryAnalysisManifest:
    """Validate and bind every declared analysis artifact without recomputation."""

    _require_utc(created_at_utc)
    root = analysis_root.resolve()
    sources: dict[str, dict[str, Any]] = {}
    bindings: list[AnalysisArtifactBinding] = []
    for spec in PRIMARY_ANALYSIS_SOURCE_SPECS:
        source, binding = _load_and_bind(root, spec)
        sources[spec.role] = source
        bindings.append(binding)

    run_id = _validate_source_graph(sources)
    _validate_figure_files(root, sources["publication_figure_manifest"])
    try:
        lock_hash = file_sha256(pixi_lock_path.read_bytes())
    except OSError as error:
        raise PrimaryAnalysisSealError("Could not hash the seal environment") from error

    content = {
        "schema_version": 1,
        "schema_id": "benchmark.primary_analysis_manifest.v1",
        "benchmark_version": "v1",
        "run_id": run_id,
        "artifacts": tuple(bindings),
        "primary_conclusion": "inconclusive",
        "specificity_conclusion": "specific",
        "closure_status": "closed_with_inconclusive_primary_result",
        "eligible_case_count": 23,
        "missing_case_count": 1,
        "final_replay_gate_passed": True,
        "negative_result_audit_status": "passed_with_dirty_seal_disclosed",
        "result_scope": "one_pinned_model_run_on_one_fixed_authored_corpus",
        "human_learning_claim_allowed": False,
        "tutoring_efficacy_claim_allowed": False,
        "reduced_cognitive_offloading_claim_allowed": False,
        "network_calls_made": 0,
        "sandbox_calls_made": 0,
        "seal_code_revision": code_revision,
        "seal_pixi_lock_hash": lock_hash,
        "created_at_utc": created_at_utc,
    }
    draft = PrimaryAnalysisManifest.model_construct(
        _fields_set=set(content), **content, manifest_hash="0" * 64
    )
    manifest = PrimaryAnalysisManifest.model_validate(
        {**content, "manifest_hash": model_content_hash(draft, exclude={"manifest_hash"})}
    )
    output_path = output_root.resolve() / "primary_analysis_manifest.json"
    if output_path.exists():
        existing = _load_manifest(output_path)
        if existing.manifest_hash != manifest.manifest_hash:
            raise PrimaryAnalysisSealError("Existing primary-analysis seal differs")
        return existing
    write_immutable_json(output_path, manifest)
    return manifest


def _load_and_bind(root: Path, spec: _SourceSpec) -> tuple[dict[str, Any], AnalysisArtifactBinding]:
    path = root / spec.relative_path
    try:
        raw = path.read_bytes()
        parsed = cast(object, json.loads(raw))
    except (OSError, json.JSONDecodeError) as error:
        raise PrimaryAnalysisSealError(f"Could not verify analysis artifact: {path}") from error
    if not isinstance(parsed, dict):
        raise PrimaryAnalysisSealError(f"Analysis artifact must be a JSON object: {path}")
    source = cast(dict[str, Any], parsed)
    if source.get("schema_id") != spec.schema_id:
        raise PrimaryAnalysisSealError(f"Unexpected schema for analysis artifact: {path}")
    declared_hash = source.get(spec.hash_field)
    if not isinstance(declared_hash, str):
        raise PrimaryAnalysisSealError(f"Analysis artifact has no {spec.hash_field}: {path}")
    payload = {key: value for key, value in source.items() if key != spec.hash_field}
    if declared_hash != canonical_sha256(payload):
        raise PrimaryAnalysisSealError(f"Analysis artifact self-hash differs: {path}")
    return source, AnalysisArtifactBinding(
        role=spec.role,
        relative_path=spec.relative_path,
        schema_id=spec.schema_id,
        content_hash=declared_hash,
        file_sha256=file_sha256(raw),
    )


def _validate_source_graph(sources: dict[str, dict[str, Any]]) -> str:
    run_ids = {
        value["run_id"] for value in sources.values() if isinstance(value.get("run_id"), str)
    }
    if len(run_ids) != 1:
        raise PrimaryAnalysisSealError("Analysis artifacts do not share one run identity")
    versions = {
        value["benchmark_version"]
        for value in sources.values()
        if isinstance(value.get("benchmark_version"), str)
    }
    if versions != {"v1"}:
        raise PrimaryAnalysisSealError("Analysis artifacts do not share benchmark v1")

    _require_link(sources, "scoring_summary", "scoring_plan_hash", "scoring_plan")
    _require_link(sources, "primary_plan", "scoring_summary_hash", "scoring_summary")
    _require_link(sources, "primary_report", "analysis_plan_hash", "primary_plan")
    _require_link(sources, "secondary_plan", "primary_analysis_report_hash", "primary_report")
    _require_link(sources, "secondary_report", "analysis_plan_hash", "secondary_plan")
    _require_link(sources, "secondary_report", "primary_analysis_report_hash", "primary_report")
    if (
        sources["secondary_report"].get("evidence_specificity", {}).get("report_hash")
        != sources["specificity_report"]["report_hash"]
    ):
        raise PrimaryAnalysisSealError(
            "Standalone specificity report differs from the secondary report"
        )
    _require_link(sources, "diagnostics_plan", "primary_analysis_report_hash", "primary_report")
    _require_link(sources, "diagnostics_report", "analysis_plan_hash", "diagnostics_plan")
    _require_link(sources, "missingness_plan", "primary_analysis_report_hash", "primary_report")
    _require_link(sources, "missingness_report", "analysis_plan_hash", "missingness_plan")
    _require_link(
        sources, "rating_reliability_report", "analysis_plan_hash", "rating_reliability_plan"
    )
    _require_link(
        sources,
        "failure_reliability_report",
        "analysis_plan_hash",
        "failure_reliability_plan",
    )
    _require_link(sources, "resource_report", "analysis_plan_hash", "resource_plan")
    _require_link(sources, "negative_audit_report", "analysis_plan_hash", "negative_audit_plan")
    _require_link(sources, "correction_report", "analysis_plan_hash", "correction_plan")
    _require_link(sources, "dependence_report", "analysis_plan_hash", "dependence_plan")
    _require_link(sources, "interpretation_plan", "primary_report_hash", "primary_report")
    _require_link(sources, "interpretation_plan", "secondary_report_hash", "secondary_report")
    _require_link(
        sources,
        "interpretation_plan",
        "probe_correction_report_hash",
        "correction_report",
    )
    _require_link(
        sources,
        "interpretation_plan",
        "negative_result_audit_report_hash",
        "negative_audit_report",
    )
    _require_link(
        sources, "interpretation_report", "interpretation_plan_hash", "interpretation_plan"
    )
    _require_link(sources, "final_replay_report", "replay_plan_hash", "final_replay_plan")
    _require_link(
        sources,
        "publication_figure_manifest",
        "source_report_hash",
        "dependence_report",
    )

    interpretation = sources["interpretation_report"]
    if interpretation.get("primary_conclusion") != "inconclusive":
        raise PrimaryAnalysisSealError("Primary conclusion is not the recorded inconclusive result")
    if interpretation.get("specificity_conclusion") != "specific":
        raise PrimaryAnalysisSealError(
            "Specificity conclusion is not the recorded secondary result"
        )
    if (
        interpretation.get("eligible_case_count") != 23
        or interpretation.get("missing_case_count") != 1
    ):
        raise PrimaryAnalysisSealError("Interpretation case counts do not match the sealed result")
    if any(
        interpretation.get(field) is not False
        for field in (
            "human_learning_claim_allowed",
            "tutoring_efficacy_claim_allowed",
            "reduced_cognitive_offloading_claim_allowed",
        )
    ):
        raise PrimaryAnalysisSealError("Interpretation permits an unsupported human claim")
    if sources["negative_audit_report"].get("audit_status") != ("passed_with_dirty_seal_disclosed"):
        raise PrimaryAnalysisSealError("Negative-result audit status is not acceptable")
    replay = sources["final_replay_report"]
    if replay.get("gate_passed") is not True or replay.get("matched_comparison_count") != 10:
        raise PrimaryAnalysisSealError("Final deterministic replay did not pass all comparisons")
    return run_ids.pop()


def _require_link(
    sources: dict[str, dict[str, Any]], source_role: str, field: str, target_role: str
) -> None:
    target_spec = next(spec for spec in PRIMARY_ANALYSIS_SOURCE_SPECS if spec.role == target_role)
    expected = sources[target_role].get(target_spec.hash_field)
    if sources[source_role].get(field) != expected:
        raise PrimaryAnalysisSealError(f"{source_role}.{field} does not identify {target_role}")


def _validate_figure_files(root: Path, manifest: dict[str, Any]) -> None:
    figure_root = root / "dependence-v1" / "publication-v5"
    figure_files = (
        ("dependence_sensitivity.pdf", "pdf_sha256"),
        ("dependence_sensitivity.svg", "svg_sha256"),
    )
    for name, field in figure_files:
        try:
            actual = file_sha256((figure_root / name).read_bytes())
        except OSError as error:
            raise PrimaryAnalysisSealError(
                f"Could not verify publication figure: {name}"
            ) from error
        if actual != manifest.get(field):
            raise PrimaryAnalysisSealError(f"Publication figure hash differs: {name}")


def _load_manifest(path: Path) -> PrimaryAnalysisManifest:
    try:
        return PrimaryAnalysisManifest.model_validate_json(path.read_bytes())
    except (OSError, ValidationError) as error:
        raise PrimaryAnalysisSealError("Could not verify existing primary-analysis seal") from error


def _require_utc(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("Primary-analysis seal timestamp must be UTC")
