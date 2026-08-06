"""Network-free reconstruction of scored datasets and analysis reports."""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from pydantic import Field, ValidationError, model_validator

from socratic_tutor.benchmark.artifacts import write_immutable_json
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.external_replay import ExternalReplayReport
from socratic_tutor.benchmark.external_scoring import (
    ExternalScoringPlan,
    ExternalScoringSummary,
)
from socratic_tutor.benchmark.hashing import file_sha256, model_content_hash
from socratic_tutor.benchmark.primary_analysis import (
    PrimaryAnalysisPlan,
    PrimaryAnalysisReport,
)
from socratic_tutor.benchmark.secondary_analysis import (
    SecondaryAnalysisPlan,
    SecondaryAnalysisReport,
)
from socratic_tutor.contracts import ContractModel


class FinalReplayError(ValueError):
    """Fresh reconstruction differs from the canonical scored result."""


ReplayArtifact = Literal[
    "baseline_publication",
    "repeat_publication",
    "case_aggregate_publication",
    "scoring_plan",
    "scoring_summary",
    "primary_plan",
    "primary_report",
    "secondary_plan",
    "secondary_report",
    "evidence_specificity_report",
]


class HashComparison(ContractModel):
    """One canonical hash and its freshly reconstructed counterpart."""

    artifact: ReplayArtifact
    canonical_hash: Sha256
    replayed_hash: Sha256
    matches: Literal[True] = True

    @model_validator(mode="after")
    def validate_comparison(self) -> HashComparison:
        if self.canonical_hash != self.replayed_hash:
            raise ValueError(f"Replayed {self.artifact} hash differs from canonical")
        return self


class FinalReplayPlan(ContractModel):
    """Canonical sources and current software identity for one final replay."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.final_replay_plan.v1"] = "benchmark.final_replay_plan.v1"
    benchmark_version: Literal["v1"] = "v1"
    run_id: str = Field(min_length=1)
    external_replay_report_hash: Sha256
    canonical_scoring_plan_hash: Sha256
    canonical_scoring_summary_hash: Sha256
    canonical_primary_plan_hash: Sha256
    canonical_primary_report_hash: Sha256
    canonical_secondary_plan_hash: Sha256
    canonical_secondary_report_hash: Sha256
    replay_code_revision: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    replay_pixi_lock_hash: Sha256
    created_at_utc: datetime
    plan_hash: Sha256

    @model_validator(mode="after")
    def validate_plan(self) -> FinalReplayPlan:
        _require_utc(self.created_at_utc)
        if self.plan_hash != model_content_hash(self, exclude={"plan_hash"}):
            raise ValueError("Final replay-plan hash does not match its content")
        return self


class FinalReplayReport(ContractModel):
    """Proof that deterministic analysis reproduces every canonical hash."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.final_replay_report.v1"] = "benchmark.final_replay_report.v1"
    benchmark_version: Literal["v1"] = "v1"
    run_id: str = Field(min_length=1)
    replay_plan_hash: Sha256
    comparisons: tuple[HashComparison, ...] = Field(min_length=10, max_length=10)
    comparison_count: Literal[10]
    matched_comparison_count: Literal[10]
    network_calls_made: Literal[0] = 0
    sandbox_calls_made: Literal[0] = 0
    historical_source_revisions_used: Literal[True] = True
    historical_lock_hashes_verified: Literal[True] = True
    dependency_execution_mode: Literal["current_interpreter_with_historical_lock_identity"] = (
        "current_interpreter_with_historical_lock_identity"
    )
    full_historical_environment_recreated: Literal[False] = False
    reconstruction_workspace_retained: Literal[False] = False
    gate_passed: Literal[True] = True
    completed_at_utc: datetime
    report_hash: Sha256

    @model_validator(mode="after")
    def validate_report(self) -> FinalReplayReport:
        _require_utc(self.completed_at_utc)
        if len({item.artifact for item in self.comparisons}) != self.comparison_count:
            raise ValueError("Final replay must compare each required artifact once")
        if self.matched_comparison_count != sum(item.matches for item in self.comparisons):
            raise ValueError("Final replay match count does not reconcile")
        if self.report_hash != model_content_hash(self, exclude={"report_hash"}):
            raise ValueError("Final replay-report hash does not match its content")
        return self


def run_final_replay(
    *,
    repository_root: Path,
    external_replay_report_path: Path,
    canonical_scoring_plan_path: Path,
    canonical_scoring_summary_path: Path,
    canonical_primary_plan_path: Path,
    canonical_primary_report_path: Path,
    canonical_secondary_plan_path: Path,
    canonical_secondary_report_path: Path,
    analysis_specification_path: Path,
    calibration_report_path: Path,
    public_manifest_path: Path,
    benchmark_root: Path,
    source_seal_root: Path,
    inferential_hierarchy_path: Path,
    specificity_amendment_path: Path,
    pixi_lock_path: Path,
    output_root: Path,
    replay_code_revision: str,
    created_at_utc: datetime,
) -> FinalReplayReport:
    """Rebuild scored and analysed outputs in temporary isolated storage."""

    _require_utc(created_at_utc)
    external_replay = _load_json(external_replay_report_path, ExternalReplayReport)
    scoring_plan = _load_json(canonical_scoring_plan_path, ExternalScoringPlan)
    scoring = _load_json(canonical_scoring_summary_path, ExternalScoringSummary)
    primary_plan = _load_json(canonical_primary_plan_path, PrimaryAnalysisPlan)
    primary = _load_json(canonical_primary_report_path, PrimaryAnalysisReport)
    secondary_plan = _load_json(canonical_secondary_plan_path, SecondaryAnalysisPlan)
    secondary = _load_json(canonical_secondary_report_path, SecondaryAnalysisReport)
    _validate_canonical_lineage(
        external_replay,
        scoring_plan,
        scoring,
        primary_plan,
        primary,
        secondary_plan,
        secondary,
    )
    if created_at_utc <= max(
        scoring.completed_at_utc,
        primary.completed_at_utc,
        secondary.completed_at_utc,
    ):
        raise FinalReplayError("Final replay must follow the canonical analyses")
    try:
        lock_hash = file_sha256(pixi_lock_path.read_bytes())
    except OSError as error:
        raise FinalReplayError("Could not hash the final replay environment") from error

    plan = _create_plan(
        external_replay=external_replay,
        scoring_plan=scoring_plan,
        scoring=scoring,
        primary_plan=primary_plan,
        primary=primary,
        secondary_plan=secondary_plan,
        secondary=secondary,
        replay_code_revision=replay_code_revision,
        lock_hash=lock_hash,
        created_at_utc=created_at_utc,
    )
    root = output_root.resolve()
    report_path = root / "final_replay_report.json"
    if report_path.exists():
        report = _load_json(report_path, FinalReplayReport)
        if report.replay_plan_hash != plan.plan_hash:
            raise FinalReplayError("Existing final replay belongs to another plan")
        return report

    with tempfile.TemporaryDirectory(prefix="socratic-tutor-final-replay-") as temporary:
        temporary_root = Path(temporary)
        replay_seal = temporary_root / "decision-seal"
        _copy_scoring_sources(source_seal_root.resolve(), replay_seal)
        scoring_checkout = _extract_historical_source(
            repository_root=repository_root.resolve(),
            revision=scoring_plan.scoring_code_revision,
            destination=temporary_root / "scoring-source",
        )
        _verify_historical_lock(scoring_checkout, scoring_plan.scoring_pixi_lock_hash)
        _run_historical_cli(
            scoring_checkout,
            (
                "external-score",
                "--replay-report",
                str(external_replay_report_path.resolve()),
                "--analysis-specification",
                str(analysis_specification_path.resolve()),
                "--calibration-report",
                str(calibration_report_path.resolve()),
                "--public-manifest",
                str(public_manifest_path.resolve()),
                "--benchmark-root",
                str(benchmark_root.resolve()),
                "--seal-root",
                str(replay_seal),
                "--pixi-lock",
                str(scoring_checkout / "development" / "pixi.lock"),
                "--code-revision",
                scoring_plan.scoring_code_revision,
                "--created-at-utc",
                _utc_text(scoring_plan.created_at_utc),
            ),
        )
        primary_checkout = _extract_historical_source(
            repository_root=repository_root.resolve(),
            revision=primary_plan.analysis_code_revision,
            destination=temporary_root / "primary-source",
        )
        _verify_historical_lock(primary_checkout, primary_plan.analysis_pixi_lock_hash)
        _run_historical_cli(
            primary_checkout,
            (
                "external-primary-analyze",
                "--scoring-summary",
                str(replay_seal / "analysis" / "scoring-v1" / "external_scoring_summary.json"),
                "--analysis-specification",
                str(analysis_specification_path.resolve()),
                "--inferential-hierarchy",
                str(inferential_hierarchy_path.resolve()),
                "--dataset-root",
                str(replay_seal / "datasets"),
                "--pixi-lock",
                str(primary_checkout / "development" / "pixi.lock"),
                "--output-root",
                str(replay_seal / "analysis" / "primary-v1"),
                "--code-revision",
                primary_plan.analysis_code_revision,
                "--created-at-utc",
                _utc_text(primary_plan.created_at_utc),
            ),
        )
        secondary_checkout = _extract_historical_source(
            repository_root=repository_root.resolve(),
            revision=secondary_plan.analysis_code_revision,
            destination=temporary_root / "secondary-source",
        )
        _verify_historical_lock(secondary_checkout, secondary_plan.analysis_pixi_lock_hash)
        _run_historical_cli(
            secondary_checkout,
            (
                "external-secondary-analyze",
                "--scoring-summary",
                str(replay_seal / "analysis" / "scoring-v1" / "external_scoring_summary.json"),
                "--primary-plan",
                str(replay_seal / "analysis" / "primary-v1" / "primary_analysis_plan.json"),
                "--primary-report",
                str(replay_seal / "analysis" / "primary-v1" / "primary_analysis_report.json"),
                "--analysis-specification",
                str(analysis_specification_path.resolve()),
                "--inferential-hierarchy",
                str(inferential_hierarchy_path.resolve()),
                "--specificity-amendment",
                str(specificity_amendment_path.resolve()),
                "--dataset-root",
                str(replay_seal / "datasets"),
                "--pixi-lock",
                str(secondary_checkout / "development" / "pixi.lock"),
                "--output-root",
                str(replay_seal / "analysis" / "secondary-v1"),
                "--code-revision",
                secondary_plan.analysis_code_revision,
                "--created-at-utc",
                _utc_text(secondary_plan.created_at_utc),
            ),
        )
        replayed_scoring = _load_json(
            replay_seal / "analysis" / "scoring-v1" / "external_scoring_summary.json",
            ExternalScoringSummary,
        )
        replayed_primary = _load_json(
            replay_seal / "analysis" / "primary-v1" / "primary_analysis_report.json",
            PrimaryAnalysisReport,
        )
        replayed_secondary = _load_json(
            replay_seal / "analysis" / "secondary-v1" / "secondary_analysis_report.json",
            SecondaryAnalysisReport,
        )
        replayed_scoring_plan = _load_json(
            replay_seal / "analysis" / "scoring-v1" / "external_scoring_plan.json",
            ExternalScoringPlan,
        )
        replayed_primary_plan = _load_json(
            replay_seal / "analysis" / "primary-v1" / "primary_analysis_plan.json",
            PrimaryAnalysisPlan,
        )
        replayed_secondary_plan = _load_json(
            replay_seal / "analysis" / "secondary-v1" / "secondary_analysis_plan.json",
            SecondaryAnalysisPlan,
        )
        comparisons = _comparisons(
            scoring_plan=scoring_plan,
            scoring=scoring,
            primary_plan=primary_plan,
            primary=primary,
            secondary_plan=secondary_plan,
            secondary=secondary,
            replayed_scoring_plan=replayed_scoring_plan,
            replayed_scoring=replayed_scoring,
            replayed_primary_plan=replayed_primary_plan,
            replayed_primary=replayed_primary,
            replayed_secondary_plan=replayed_secondary_plan,
            replayed_secondary=replayed_secondary,
        )

    content = {
        "schema_version": 1,
        "schema_id": "benchmark.final_replay_report.v1",
        "benchmark_version": "v1",
        "run_id": scoring.run_id,
        "replay_plan_hash": plan.plan_hash,
        "comparisons": comparisons,
        "comparison_count": len(comparisons),
        "matched_comparison_count": sum(item.matches for item in comparisons),
        "network_calls_made": 0,
        "sandbox_calls_made": 0,
        "historical_source_revisions_used": True,
        "historical_lock_hashes_verified": True,
        "dependency_execution_mode": "current_interpreter_with_historical_lock_identity",
        "full_historical_environment_recreated": False,
        "reconstruction_workspace_retained": False,
        "gate_passed": True,
        "completed_at_utc": created_at_utc,
    }
    draft = FinalReplayReport.model_construct(
        _fields_set=set(content), **content, report_hash="0" * 64
    )
    report = FinalReplayReport.model_validate(
        {**content, "report_hash": model_content_hash(draft, exclude={"report_hash"})}
    )
    write_immutable_json(root / "final_replay_plan.json", plan)
    write_immutable_json(report_path, report)
    return report


def _extract_historical_source(*, repository_root: Path, revision: str, destination: Path) -> Path:
    if re.fullmatch(r"[0-9a-f]{7,40}", revision) is None:
        raise FinalReplayError("Historical source revision is not a Git object ID")
    try:
        archive = subprocess.run(
            (
                "git",
                "archive",
                "--format=tar",
                revision,
                "development/src",
                "development/scripts/benchmark_cli.py",
                "development/pixi.lock",
            ),
            cwd=repository_root,
            check=True,
            capture_output=True,
        ).stdout
        destination.mkdir(parents=True)
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
            bundle.extractall(destination, filter="data")
    except (OSError, subprocess.CalledProcessError, tarfile.TarError) as error:
        raise FinalReplayError(
            f"Could not extract historical source revision {revision}"
        ) from error
    return destination


def _verify_historical_lock(checkout: Path, expected_hash: Sha256) -> None:
    try:
        actual_hash = file_sha256((checkout / "development" / "pixi.lock").read_bytes())
    except OSError as error:
        raise FinalReplayError("Could not verify historical Pixi lock") from error
    if actual_hash != expected_hash:
        raise FinalReplayError(
            f"Historical Pixi lock differs from its plan: expected={expected_hash}, "
            f"actual={actual_hash}"
        )


def _run_historical_cli(checkout: Path, arguments: tuple[str, ...]) -> None:
    development = checkout / "development"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(development / "src")
    for secret in ("CEREBRAS_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        environment.pop(secret, None)
    try:
        completed = subprocess.run(
            (sys.executable, str(development / "scripts" / "benchmark_cli.py"), *arguments),
            cwd=development,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        stdout = error.stdout if isinstance(error, subprocess.CalledProcessError) else None
        stderr = error.stderr if isinstance(error, subprocess.CalledProcessError) else None
        detail = "\n".join(part.strip() for part in (stdout, stderr) if part and part.strip())
        raise FinalReplayError(f"Historical replay stage failed: {detail[-2000:]}") from error
    try:
        decoded = cast(object, json.loads(completed.stdout))
    except json.JSONDecodeError as error:
        raise FinalReplayError("Historical replay stage returned invalid JSON") from error
    if not isinstance(decoded, dict):
        raise FinalReplayError("Historical replay stage returned invalid JSON")
    result = cast(dict[str, object], decoded)
    if result.get("status") != "ok":
        raise FinalReplayError("Historical replay stage did not report success")


def _utc_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _copy_scoring_sources(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FinalReplayError("Final replay destination already exists")
    destination.mkdir(parents=True)
    for name in ("run_plan.json", "probe_summaries.json"):
        try:
            shutil.copy2(source / name, destination / name)
        except OSError as error:
            raise FinalReplayError(f"Could not copy sealed source: {name}") from error
    try:
        shutil.copytree(source / "decision", destination / "decision")
        shutil.copytree(source / "criterion" / "records", destination / "criterion" / "records")
        datasets = destination / "datasets"
        datasets.mkdir()
        for stem in ("condition_predictions", "criterion_records"):
            for suffix in (".parquet", ".publication.json"):
                shutil.copy2(source / "datasets" / f"{stem}{suffix}", datasets)
    except OSError as error:
        raise FinalReplayError("Could not copy sealed scoring inputs") from error


def _validate_canonical_lineage(
    replay: ExternalReplayReport,
    scoring_plan: ExternalScoringPlan,
    scoring: ExternalScoringSummary,
    primary_plan: PrimaryAnalysisPlan,
    primary: PrimaryAnalysisReport,
    secondary_plan: SecondaryAnalysisPlan,
    secondary: SecondaryAnalysisReport,
) -> None:
    valid = (
        scoring_plan.replay_report_hash == replay.report_hash
        and scoring.scoring_plan_hash == scoring_plan.plan_hash
        and primary_plan.scoring_summary_hash == scoring.summary_hash
        and primary.analysis_plan_hash == primary_plan.plan_hash
        and secondary_plan.scoring_summary_hash == scoring.summary_hash
        and secondary_plan.primary_analysis_plan_hash == primary_plan.plan_hash
        and secondary_plan.primary_analysis_report_hash == primary.report_hash
        and secondary.analysis_plan_hash == secondary_plan.plan_hash
        and secondary.primary_analysis_report_hash == primary.report_hash
    )
    if not valid:
        raise FinalReplayError("Canonical scoring and analysis lineage is inconsistent")
    if len({replay.run_id, scoring.run_id, primary.run_id, secondary.run_id}) != 1:
        raise FinalReplayError("Canonical replay sources belong to different runs")
    if replay.network_calls_made or replay.sandbox_calls_made:
        raise FinalReplayError("Source replay used an external service")


def _comparisons(
    *,
    scoring_plan: ExternalScoringPlan,
    scoring: ExternalScoringSummary,
    primary_plan: PrimaryAnalysisPlan,
    primary: PrimaryAnalysisReport,
    secondary_plan: SecondaryAnalysisPlan,
    secondary: SecondaryAnalysisReport,
    replayed_scoring_plan: ExternalScoringPlan,
    replayed_scoring: ExternalScoringSummary,
    replayed_primary_plan: PrimaryAnalysisPlan,
    replayed_primary: PrimaryAnalysisReport,
    replayed_secondary_plan: SecondaryAnalysisPlan,
    replayed_secondary: SecondaryAnalysisReport,
) -> tuple[HashComparison, ...]:
    pairs: tuple[tuple[ReplayArtifact, Sha256, Sha256], ...] = (
        (
            "baseline_publication",
            scoring.baseline_publication_hash,
            replayed_scoring.baseline_publication_hash,
        ),
        (
            "repeat_publication",
            scoring.repeat_publication_hash,
            replayed_scoring.repeat_publication_hash,
        ),
        (
            "case_aggregate_publication",
            scoring.aggregate_publication_hash,
            replayed_scoring.aggregate_publication_hash,
        ),
        ("scoring_plan", scoring_plan.plan_hash, replayed_scoring_plan.plan_hash),
        ("scoring_summary", scoring.summary_hash, replayed_scoring.summary_hash),
        ("primary_plan", primary_plan.plan_hash, replayed_primary_plan.plan_hash),
        ("primary_report", primary.report_hash, replayed_primary.report_hash),
        ("secondary_plan", secondary_plan.plan_hash, replayed_secondary_plan.plan_hash),
        ("secondary_report", secondary.report_hash, replayed_secondary.report_hash),
        (
            "evidence_specificity_report",
            secondary.evidence_specificity.report_hash,
            replayed_secondary.evidence_specificity.report_hash,
        ),
    )
    mismatches = tuple(
        (artifact, canonical, replayed)
        for artifact, canonical, replayed in pairs
        if canonical != replayed
    )
    if mismatches:
        details = "; ".join(
            f"{artifact}: canonical={canonical}, replayed={replayed}"
            for artifact, canonical, replayed in mismatches
        )
        raise FinalReplayError(f"Fresh analysis did not reproduce canonical hashes: {details}")
    return tuple(
        HashComparison(
            artifact=artifact,
            canonical_hash=canonical,
            replayed_hash=replayed,
            matches=True,
        )
        for artifact, canonical, replayed in pairs
    )


def _create_plan(
    *,
    external_replay: ExternalReplayReport,
    scoring_plan: ExternalScoringPlan,
    scoring: ExternalScoringSummary,
    primary_plan: PrimaryAnalysisPlan,
    primary: PrimaryAnalysisReport,
    secondary_plan: SecondaryAnalysisPlan,
    secondary: SecondaryAnalysisReport,
    replay_code_revision: str,
    lock_hash: Sha256,
    created_at_utc: datetime,
) -> FinalReplayPlan:
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.final_replay_plan.v1",
        "benchmark_version": "v1",
        "run_id": scoring.run_id,
        "external_replay_report_hash": external_replay.report_hash,
        "canonical_scoring_plan_hash": scoring_plan.plan_hash,
        "canonical_scoring_summary_hash": scoring.summary_hash,
        "canonical_primary_plan_hash": primary_plan.plan_hash,
        "canonical_primary_report_hash": primary.report_hash,
        "canonical_secondary_plan_hash": secondary_plan.plan_hash,
        "canonical_secondary_report_hash": secondary.report_hash,
        "replay_code_revision": replay_code_revision,
        "replay_pixi_lock_hash": lock_hash,
        "created_at_utc": created_at_utc,
    }
    draft = FinalReplayPlan.model_construct(_fields_set=set(content), **content, plan_hash="0" * 64)
    return FinalReplayPlan.model_validate(
        {**content, "plan_hash": model_content_hash(draft, exclude={"plan_hash"})}
    )


def _load_json[ModelT: ContractModel](path: Path, model: type[ModelT]) -> ModelT:
    try:
        return model.model_validate_json(path.read_bytes())
    except (OSError, ValidationError) as error:
        raise FinalReplayError(f"Could not verify final replay source: {path}") from error


def _require_utc(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("Final replay timestamp must be UTC")
