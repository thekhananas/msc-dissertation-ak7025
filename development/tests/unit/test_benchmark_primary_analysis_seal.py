from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from socratic_tutor.benchmark.hashing import canonical_sha256, file_sha256
from socratic_tutor.benchmark.primary_analysis_seal import (
    PRIMARY_ANALYSIS_ARTIFACT_COUNT,
    PRIMARY_ANALYSIS_SOURCE_SPECS,
    PrimaryAnalysisSealError,
    seal_primary_analysis,
)

RUN_ID = "test-v1-run"
CREATED_AT = datetime(2026, 9, 3, 2, 30, tzinfo=UTC)


def test_seals_complete_analysis_chain_and_returns_exact_retry(tmp_path: Path) -> None:
    analysis_root, pixi_lock = _write_source_graph(tmp_path)
    output_root = tmp_path / "seal"

    manifest = seal_primary_analysis(
        analysis_root=analysis_root,
        pixi_lock_path=pixi_lock,
        output_root=output_root,
        code_revision="a" * 40,
        created_at_utc=CREATED_AT,
    )
    retry = seal_primary_analysis(
        analysis_root=analysis_root,
        pixi_lock_path=pixi_lock,
        output_root=output_root,
        code_revision="a" * 40,
        created_at_utc=CREATED_AT,
    )

    assert manifest.manifest_hash == retry.manifest_hash
    assert len(manifest.artifacts) == PRIMARY_ANALYSIS_ARTIFACT_COUNT
    assert manifest.closure_status == "closed_with_inconclusive_primary_result"
    assert manifest.human_learning_claim_allowed is False
    assert manifest.network_calls_made == 0
    assert (output_root / "primary_analysis_manifest.json").is_file()


def test_rejects_self_consistent_but_broken_lineage(tmp_path: Path) -> None:
    analysis_root, pixi_lock = _write_source_graph(tmp_path)
    path = analysis_root / "primary-v1" / "primary_analysis_report.json"
    source = json.loads(path.read_text(encoding="utf-8"))
    source["analysis_plan_hash"] = "f" * 64
    source["report_hash"] = _content_hash(source, "report_hash")
    path.write_text(json.dumps(source, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(PrimaryAnalysisSealError, match="does not identify"):
        seal_primary_analysis(
            analysis_root=analysis_root,
            pixi_lock_path=pixi_lock,
            output_root=tmp_path / "seal",
            code_revision="a" * 40,
            created_at_utc=CREATED_AT,
        )


def test_rejects_an_unsupported_human_claim(tmp_path: Path) -> None:
    analysis_root, pixi_lock = _write_source_graph(tmp_path)
    path = analysis_root / "result-interpretation-v1" / "result_interpretation_report.json"
    source = json.loads(path.read_text(encoding="utf-8"))
    source["human_learning_claim_allowed"] = True
    source["report_hash"] = _content_hash(source, "report_hash")
    path.write_text(json.dumps(source, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(PrimaryAnalysisSealError, match="unsupported human claim"):
        seal_primary_analysis(
            analysis_root=analysis_root,
            pixi_lock_path=pixi_lock,
            output_root=tmp_path / "seal",
            code_revision="a" * 40,
            created_at_utc=CREATED_AT,
        )


def _write_source_graph(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "analysis"
    specs = {spec.role: spec for spec in PRIMARY_ANALYSIS_SOURCE_SPECS}
    sources: dict[str, dict[str, Any]] = {}

    def emit(role: str, **extra: object) -> dict[str, Any]:
        spec = specs[role]
        source: dict[str, Any] = {
            "schema_version": 1,
            "schema_id": spec.schema_id,
            "benchmark_version": "v1",
            "run_id": RUN_ID,
            **extra,
        }
        source[spec.hash_field] = _content_hash(source, spec.hash_field)
        path = root / spec.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(source, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        sources[role] = source
        return source

    scoring_plan = emit("scoring_plan")
    scoring_summary = emit("scoring_summary", scoring_plan_hash=scoring_plan["plan_hash"])
    primary_plan = emit("primary_plan", scoring_summary_hash=scoring_summary["summary_hash"])
    primary_report = emit("primary_report", analysis_plan_hash=primary_plan["plan_hash"])
    secondary_plan = emit(
        "secondary_plan", primary_analysis_report_hash=primary_report["report_hash"]
    )
    specificity = emit("specificity_report")
    secondary_report = emit(
        "secondary_report",
        analysis_plan_hash=secondary_plan["plan_hash"],
        primary_analysis_report_hash=primary_report["report_hash"],
        evidence_specificity={"report_hash": specificity["report_hash"]},
    )
    diagnostics_plan = emit(
        "diagnostics_plan", primary_analysis_report_hash=primary_report["report_hash"]
    )
    emit("diagnostics_report", analysis_plan_hash=diagnostics_plan["plan_hash"])
    missingness_plan = emit(
        "missingness_plan", primary_analysis_report_hash=primary_report["report_hash"]
    )
    emit("missingness_report", analysis_plan_hash=missingness_plan["plan_hash"])
    rating_plan = emit("rating_reliability_plan")
    emit("rating_reliability_report", analysis_plan_hash=rating_plan["plan_hash"])
    failure_plan = emit("failure_reliability_plan")
    emit("failure_reliability_report", analysis_plan_hash=failure_plan["plan_hash"])
    resource_plan = emit("resource_plan")
    emit("resource_report", analysis_plan_hash=resource_plan["plan_hash"])
    audit_plan = emit("negative_audit_plan")
    audit_report = emit(
        "negative_audit_report",
        analysis_plan_hash=audit_plan["plan_hash"],
        audit_status="passed_with_dirty_seal_disclosed",
    )
    correction_plan = emit("correction_plan")
    correction_report = emit("correction_report", analysis_plan_hash=correction_plan["plan_hash"])
    dependence_plan = emit("dependence_plan")
    dependence_report = emit("dependence_report", analysis_plan_hash=dependence_plan["plan_hash"])
    interpretation_plan = emit(
        "interpretation_plan",
        primary_report_hash=primary_report["report_hash"],
        secondary_report_hash=secondary_report["report_hash"],
        probe_correction_report_hash=correction_report["report_hash"],
        negative_result_audit_report_hash=audit_report["report_hash"],
    )
    emit(
        "interpretation_report",
        interpretation_plan_hash=interpretation_plan["plan_hash"],
        primary_conclusion="inconclusive",
        specificity_conclusion="specific",
        eligible_case_count=23,
        missing_case_count=1,
        human_learning_claim_allowed=False,
        tutoring_efficacy_claim_allowed=False,
        reduced_cognitive_offloading_claim_allowed=False,
    )
    replay_plan = emit("final_replay_plan")
    emit(
        "final_replay_report",
        replay_plan_hash=replay_plan["plan_hash"],
        gate_passed=True,
        matched_comparison_count=10,
    )

    figure_root = root / "dependence-v1" / "publication-v5"
    figure_root.mkdir(parents=True, exist_ok=True)
    pdf = b"test-pdf"
    svg = b"<svg>test</svg>"
    (figure_root / "dependence_sensitivity.pdf").write_bytes(pdf)
    (figure_root / "dependence_sensitivity.svg").write_bytes(svg)
    emit(
        "publication_figure_manifest",
        source_report_hash=dependence_report["report_hash"],
        pdf_sha256=file_sha256(pdf),
        svg_sha256=file_sha256(svg),
    )

    pixi_lock = tmp_path / "pixi.lock"
    pixi_lock.write_text("locked\n", encoding="utf-8")
    return root, pixi_lock


def _content_hash(source: dict[str, Any], hash_field: str) -> str:
    return canonical_sha256({key: value for key, value in source.items() if key != hash_field})
