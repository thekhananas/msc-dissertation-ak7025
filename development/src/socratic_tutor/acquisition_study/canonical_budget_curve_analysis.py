"""Verified descriptive analysis of the canonical probe-budget curve."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Protocol

from pydantic import ValidationError

from socratic_tutor.acquisition_study.budget_curve import BUDGET_CURVE_POLICY_ORDER
from socratic_tutor.acquisition_study.budget_curve_analysis import (
    EnvironmentBudgetSummary,
    HeldOutBudgetPairedDifference,
    HeldOutBudgetSummary,
)
from socratic_tutor.acquisition_study.budget_curve_runner import BudgetCurveEpisodeMetric
from socratic_tutor.acquisition_study.canonical_budget_curve import (
    CanonicalBudgetCurveExecutionPlan,
    CanonicalBudgetCurveManifest,
    CanonicalBudgetCurveRecord,
    CanonicalBudgetCurveReport,
)
from socratic_tutor.acquisition_study.canonical_primary_analysis import (
    CanonicalPrimaryAnalysisManifest,
    CanonicalPrimaryAnalysisReport,
)
from socratic_tutor.acquisition_study.contracts import PolicyId
from socratic_tutor.acquisition_study.plan import (
    EnvironmentRole,
    EvaluationEnvironmentId,
    load_acquisition_study_plan,
)
from socratic_tutor.benchmark.artifacts import (
    artifact_locations,
    write_immutable_bytes,
    write_immutable_json,
)
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import canonical_sha256, file_sha256
from socratic_tutor.contracts import ContractModel
from socratic_tutor.repository_state import current_clean_revision, validate_git_revision

_BUDGETS = (0.0, 0.25, 0.5, 0.75, 1.0)
_POLICIES = BUDGET_CURVE_POLICY_ORDER
_REFERENCES = _POLICIES[1:]
_PRIMARY_REFERENCES = (
    PolicyId.SEEDED_RANDOM_BOUNDED,
    PolicyId.UNCERTAINTY_ONLY_BOUNDED,
    PolicyId.PLUG_IN_EVSI_BOUNDED,
)
_CANDIDATE = PolicyId.RELIABILITY_AWARE_BOUNDED
_ENVIRONMENTS = tuple(EvaluationEnvironmentId)
_HELD_OUT = _ENVIRONMENTS[1:]
_REPORT_FILE = "canonical_budget_curve_analysis_report.json"
_ENVIRONMENT_TABLE_FILE = "canonical_budget_environment_summaries.csv"
_HELD_OUT_TABLE_FILE = "canonical_budget_held_out_summaries.csv"
_PAIRED_TABLE_FILE = "canonical_budget_paired_differences.csv"


class CanonicalBudgetCurveAnalysisError(ValueError):
    """The canonical curve cannot support the frozen descriptive analysis."""


@dataclass(frozen=True, slots=True)
class BudgetStreamExpectation:
    """Expected identity and fixed layout of the canonical stream."""

    file_sha256: Sha256
    episode_index_start: int
    episodes_per_environment: int
    candidates_per_episode: int

    @property
    def row_count(self) -> int:
        return len(_ENVIRONMENTS) * self.episodes_per_environment * len(_BUDGETS) * len(_POLICIES)


class _ByteDigest(Protocol):
    def update(self, value: bytes, /) -> None: ...

    def hexdigest(self) -> str: ...


@dataclass(slots=True)
class _Sum:
    total: float = 0.0
    correction: float = 0.0

    def add(self, value: float) -> None:
        adjusted = value - self.correction
        updated = self.total + adjusted
        self.correction = (updated - self.total) - adjusted
        self.total = updated


@dataclass(slots=True)
class _Totals:
    rows: int = 0
    selected: int = 0
    passed: int = 0
    failed: int = 0
    missing: int = 0
    errors: int = 0
    squared_error: _Sum = field(default_factory=_Sum)
    log_loss: _Sum = field(default_factory=_Sum)
    bin_counts: list[int] = field(default_factory=lambda: [0] * 10)
    bin_probability: list[_Sum] = field(default_factory=lambda: [_Sum() for _ in range(10)])
    bin_positive: list[int] = field(default_factory=lambda: [0] * 10)
    digest: _ByteDigest = field(default_factory=hashlib.sha256)

    def add(self, metric: BudgetCurveEpisodeMetric, raw_line: bytes) -> None:
        self.rows += 1
        self.selected += metric.selected_probe_count
        self.passed += metric.observed_probe_pass_count
        self.failed += metric.observed_probe_fail_count
        self.missing += metric.missing_probe_result_count
        self.errors += metric.classification_error_count
        self.squared_error.add(metric.squared_error_sum)
        self.log_loss.add(metric.negative_log_likelihood_sum)
        for index in range(10):
            self.bin_counts[index] += metric.calibration_bin_counts[index]
            self.bin_probability[index].add(metric.calibration_probability_sums[index])
            self.bin_positive[index] += metric.calibration_positive_counts[index]
        self.digest.update(raw_line)


@dataclass(frozen=True, slots=True)
class _LoadedCurve:
    totals: dict[tuple[EvaluationEnvironmentId, float, PolicyId], _Totals]
    effects: dict[tuple[EvaluationEnvironmentId, float, PolicyId], tuple[float, ...]]
    row_count: int
    byte_count: int
    file_sha256: Sha256


@dataclass(frozen=True, slots=True)
class _Tables:
    environments: tuple[EnvironmentBudgetSummary, ...]
    held_out: tuple[HeldOutBudgetSummary, ...]
    paired: tuple[HeldOutBudgetPairedDifference, ...]


def load_verified_budget_stream(path: Path, expected: BudgetStreamExpectation) -> _LoadedCurve:
    """Verify and aggregate the large JSONL stream in one pass."""

    totals = {
        (environment_id, budget, policy_id): _Totals()
        for environment_id in _ENVIRONMENTS
        for budget in _BUDGETS
        for policy_id in _POLICIES
    }
    effects: dict[tuple[EvaluationEnvironmentId, float, PolicyId], list[float]] = {
        (environment_id, budget, reference): []
        for environment_id in _HELD_OUT
        for budget in _BUDGETS
        for reference in _REFERENCES
    }
    digest = hashlib.sha256()
    byte_count = 0
    candidate_hash: Sha256 | None = None
    candidate_error: float | None = None
    endpoint: tuple[object, ...] | None = None
    rows_per_episode = len(_BUDGETS) * len(_POLICIES)
    rows_per_environment = expected.episodes_per_environment * rows_per_episode
    try:
        with path.open("rb") as handle:
            for row_index, raw_line in enumerate(handle):
                if row_index >= expected.row_count:
                    raise CanonicalBudgetCurveAnalysisError("Budget stream contains extra rows")
                if not raw_line.endswith(b"\n") or raw_line == b"\n":
                    raise CanonicalBudgetCurveAnalysisError("Budget stream contains invalid JSONL")
                digest.update(raw_line)
                byte_count += len(raw_line)
                environment_index, remainder = divmod(row_index, rows_per_environment)
                episode_offset, within_episode = divmod(remainder, rows_per_episode)
                budget_index, policy_index = divmod(within_episode, len(_POLICIES))
                environment_id = _ENVIRONMENTS[environment_index]
                budget = _BUDGETS[budget_index]
                policy_id = _POLICIES[policy_index]
                row = CanonicalBudgetCurveRecord.model_validate_json(raw_line)
                metric = row.metric
                if (
                    metric.environment_id is not environment_id
                    or metric.episode_index != expected.episode_index_start + episode_offset
                    or not math.isclose(metric.budget_fraction, budget, abs_tol=1e-12)
                    or metric.policy_id is not policy_id
                    or metric.candidate_count != expected.candidates_per_episode
                ):
                    raise CanonicalBudgetCurveAnalysisError(
                        f"Budget stream row {row_index + 1} violates the fixed order"
                    )
                if budget_index == 0 and policy_index == 0:
                    candidate_hash = row.shared_candidate_hash
                elif row.shared_candidate_hash != candidate_hash:
                    raise CanonicalBudgetCurveAnalysisError("Budget stream breaks episode pairing")
                if policy_index == 0:
                    candidate_error = metric.final_classification_error
                    endpoint = _outcome(metric) if budget in (0.0, 1.0) else None
                elif budget in (0.0, 1.0) and _outcome(metric) != endpoint:
                    raise CanonicalBudgetCurveAnalysisError("Budget endpoint policies disagree")
                if environment_id in _HELD_OUT and policy_index > 0:
                    if candidate_error is None:
                        raise CanonicalBudgetCurveAnalysisError("Candidate result is missing")
                    effects[(environment_id, budget, policy_id)].append(
                        candidate_error - metric.final_classification_error
                    )
                totals[(environment_id, budget, policy_id)].add(metric, raw_line)
    except CanonicalBudgetCurveAnalysisError:
        raise
    except (OSError, ValidationError) as error:
        raise CanonicalBudgetCurveAnalysisError(
            f"Could not verify canonical budget stream: {path}"
        ) from error
    file_hash = digest.hexdigest()
    row_count = sum(item.rows for item in totals.values())
    if row_count != expected.row_count:
        raise CanonicalBudgetCurveAnalysisError("Budget stream is incomplete")
    if file_hash != expected.file_sha256:
        raise CanonicalBudgetCurveAnalysisError("Budget stream hash differs")
    if any(item.rows != expected.episodes_per_environment for item in totals.values()):
        raise CanonicalBudgetCurveAnalysisError("Budget groups have incomplete coverage")
    return _LoadedCurve(
        totals=totals,
        effects={key: tuple(values) for key, values in effects.items()},
        row_count=row_count,
        byte_count=byte_count,
        file_sha256=file_hash,
    )


def analyse_budget_stream(
    loaded: _LoadedCurve,
    *,
    primary: CanonicalPrimaryAnalysisReport,
    episodes_per_environment: int,
    candidates_per_episode: int,
) -> _Tables:
    """Create descriptive summaries and require exact primary-result parity."""

    environments = tuple(
        _environment_summary(
            loaded.totals[(environment_id, budget, policy_id)],
            environment_id,
            budget,
            policy_id,
            candidates_per_episode,
        )
        for environment_id in _ENVIRONMENTS
        for budget in _BUDGETS
        for policy_id in _POLICIES
    )
    by_key = {(row.environment_id, row.budget_fraction, row.policy_id): row for row in environments}
    held_out = tuple(
        _held_out_summary(by_key, budget, policy_id, episodes_per_environment)
        for budget in _BUDGETS
        for policy_id in _POLICIES
    )
    macro_by_key = {(row.budget_fraction, row.policy_id): row for row in held_out}
    paired = tuple(
        _paired_summary(
            loaded.effects,
            by_key,
            macro_by_key,
            budget,
            reference,
            episodes_per_environment,
        )
        for budget in _BUDGETS
        for reference in _REFERENCES
    )
    _verify_primary_effects(loaded.effects, paired, primary)
    return _Tables(environments=environments, held_out=held_out, paired=paired)


def run_canonical_budget_curve_analysis(
    *,
    source_manifest_path: Path,
    primary_manifest_path: Path,
    environment_path: Path,
    analysis_path: Path,
    pixi_lock_path: Path,
    output_root: Path,
    analysis_code_revision: str,
) -> dict[str, object]:
    """Verify the sealed sources and publish compact analysis outputs."""

    validate_git_revision(
        analysis_code_revision,
        invalid_message="Analysis revision must be a full Git SHA",
        error_factory=CanonicalBudgetCurveAnalysisError,
    )
    source_manifest = _load(source_manifest_path, CanonicalBudgetCurveManifest)
    source_root = source_manifest_path.parent
    source_plan_path = source_root / source_manifest.execution_plan_file
    source_report_path = source_root / source_manifest.report_file
    stream_path = source_root / source_manifest.budget_curve_file
    source_plan = _load(source_plan_path, CanonicalBudgetCurveExecutionPlan)
    source_report = _load(source_report_path, CanonicalBudgetCurveReport)
    primary_manifest = _load(primary_manifest_path, CanonicalPrimaryAnalysisManifest)
    primary_report_path = primary_manifest_path.parent / primary_manifest.report_file
    primary = _load(primary_report_path, CanonicalPrimaryAnalysisReport)
    _, analysis = load_acquisition_study_plan(environment_path, analysis_path)
    pixi_hash = _hash_small_file(pixi_lock_path, "Pixi lock")
    _verify_sources(
        source_manifest,
        source_plan,
        source_report,
        primary_manifest,
        primary,
        source_plan_path=source_plan_path,
        source_report_path=source_report_path,
        primary_manifest_path=primary_manifest_path,
        primary_report_path=primary_report_path,
        stream_path=stream_path,
        analysis_hash=analysis.plan_hash,
        pixi_hash=pixi_hash,
    )
    loaded = load_verified_budget_stream(
        stream_path,
        BudgetStreamExpectation(
            file_sha256=source_report.budget_curve_file_sha256,
            episode_index_start=source_plan.evaluation_episode_index_start,
            episodes_per_environment=source_plan.episodes_per_environment,
            candidates_per_episode=source_plan.candidates_per_episode,
        ),
    )
    tables = analyse_budget_stream(
        loaded,
        primary=primary,
        episodes_per_environment=source_plan.episodes_per_environment,
        candidates_per_episode=source_plan.candidates_per_episode,
    )
    environment_csv = _csv_table(
        tables.environments,
        (
            "environment_id",
            "environment_role",
            "budget_fraction",
            "policy_id",
            "episode_count",
            "selected_probes_per_episode",
            "mean_classification_error",
            "brier_score",
            "negative_log_likelihood",
            "expected_calibration_error",
            "observed_probe_pass_count",
            "observed_probe_fail_count",
            "missing_probe_result_count",
        ),
    )
    held_out_csv = _csv_table(
        tables.held_out,
        (
            "budget_fraction",
            "policy_id",
            "macro_mean_classification_error",
            "macro_mean_brier_score",
            "macro_mean_negative_log_likelihood",
            "macro_mean_expected_calibration_error",
            "worst_environment_id",
            "worst_environment_classification_error",
        ),
    )
    paired_csv = _csv_table(
        tables.paired,
        (
            "budget_fraction",
            "candidate_policy_id",
            "reference_policy_id",
            "reference_role",
            "candidate_macro_mean_classification_error",
            "reference_macro_mean_classification_error",
            "macro_mean_paired_difference",
            "candidate_better_episode_count",
            "tied_episode_count",
            "reference_better_episode_count",
            "worst_environment_id",
            "worst_environment_mean_difference",
        ),
    )
    content = {
        "schema_version": 1,
        "schema_id": "acquisition_study.canonical_budget_curve_analysis.v1",
        "study_id": source_manifest.study_id,
        "split": "evaluation",
        "result_status": "canonical_secondary_budget_analysis_complete",
        "analysis_code_revision": analysis_code_revision,
        "source_budget_manifest_hash": source_manifest.manifest_hash,
        "source_budget_execution_plan_hash": source_plan.plan_hash,
        "source_budget_report_hash": source_report.report_hash,
        "source_budget_stream_sha256": loaded.file_sha256,
        "primary_analysis_report_hash": primary.report_hash,
        "frozen_analysis_specification_hash": analysis.plan_hash,
        "pixi_lock_sha256": pixi_hash,
        "source_row_count": loaded.row_count,
        "source_stream_bytes": loaded.byte_count,
        "episodes_per_environment": source_plan.episodes_per_environment,
        "candidates_per_episode": source_plan.candidates_per_episode,
        "budget_fractions": _BUDGETS,
        "policy_ids": tuple(policy_id.value for policy_id in _POLICIES),
        "environment_summary_count": len(tables.environments),
        "held_out_macro_summaries": tuple(row.model_dump(mode="json") for row in tables.held_out),
        "held_out_paired_differences": tuple(row.model_dump(mode="json") for row in tables.paired),
        "environment_table_file": _ENVIRONMENT_TABLE_FILE,
        "environment_table_sha256": file_sha256(environment_csv),
        "held_out_table_file": _HELD_OUT_TABLE_FILE,
        "held_out_table_sha256": file_sha256(held_out_csv),
        "paired_table_file": _PAIRED_TABLE_FILE,
        "paired_table_sha256": file_sha256(paired_csv),
        "primary_effect_reproduction_verified": True,
        "endpoint_agreement_verified": True,
        "analysis_role": "descriptive_secondary_cannot_rescue_primary",
        "primary_result_known_before_analysis": True,
        "post_primary_policy_tuning_allowed": False,
        "restricted_stream_accessed": False,
        "external_model_call_count": 0,
        "sandbox_call_count": 0,
        "human_learning_claim_supported": False,
        "tutoring_efficacy_claim_supported": False,
        "cognitive_offloading_claim_supported": False,
        "deployed_cost_saving_claim_supported": False,
        "primary_claim_rescued": False,
    }
    report: dict[str, object] = {
        **content,
        "report_hash": canonical_sha256(content),
    }
    write_immutable_bytes(output_root / _ENVIRONMENT_TABLE_FILE, environment_csv)
    write_immutable_bytes(output_root / _HELD_OUT_TABLE_FILE, held_out_csv)
    write_immutable_bytes(output_root / _PAIRED_TABLE_FILE, paired_csv)
    write_immutable_json(output_root / _REPORT_FILE, report)
    return report


def _environment_summary(
    totals: _Totals,
    environment_id: EvaluationEnvironmentId,
    budget: float,
    policy_id: PolicyId,
    candidates_per_episode: int,
) -> EnvironmentBudgetSummary:
    case_count = totals.rows * candidates_per_episode
    if not case_count:
        raise CanonicalBudgetCurveAnalysisError("Budget summary contains no cases")
    return EnvironmentBudgetSummary(
        environment_id=environment_id,
        environment_role=(
            EnvironmentRole.MATCHED_SANITY_CHECK
            if environment_id is EvaluationEnvironmentId.MATCHED
            else EnvironmentRole.HELD_OUT_MISMATCH
        ),
        budget_fraction=budget,
        policy_id=policy_id,
        episode_count=totals.rows,
        candidates_per_episode=candidates_per_episode,
        selected_probes_per_episode=int(budget * candidates_per_episode),
        case_prediction_count=case_count,
        selected_probe_count=totals.selected,
        observed_probe_pass_count=totals.passed,
        observed_probe_fail_count=totals.failed,
        missing_probe_result_count=totals.missing,
        mean_classification_error=totals.errors / case_count,
        brier_score=totals.squared_error.total / case_count,
        negative_log_likelihood=totals.log_loss.total / case_count,
        expected_calibration_error=_calibration_error(totals),
        source_rows_hash=totals.digest.hexdigest(),
    )


def _held_out_summary(
    summaries: dict[tuple[EvaluationEnvironmentId, float, PolicyId], EnvironmentBudgetSummary],
    budget: float,
    policy_id: PolicyId,
    episodes_per_environment: int,
) -> HeldOutBudgetSummary:
    rows = tuple(summaries[(environment_id, budget, policy_id)] for environment_id in _HELD_OUT)
    worst = max(rows, key=lambda row: row.mean_classification_error)
    return HeldOutBudgetSummary(
        budget_fraction=budget,
        policy_id=policy_id,
        episodes_per_environment=episodes_per_environment,
        macro_mean_classification_error=fmean(row.mean_classification_error for row in rows),
        macro_mean_brier_score=fmean(row.brier_score for row in rows),
        macro_mean_negative_log_likelihood=fmean(row.negative_log_likelihood for row in rows),
        macro_mean_expected_calibration_error=fmean(row.expected_calibration_error for row in rows),
        worst_environment_id=worst.environment_id,
        worst_environment_classification_error=worst.mean_classification_error,
        environment_summaries_hash=canonical_sha256(rows),
    )


def _paired_summary(
    effects: dict[tuple[EvaluationEnvironmentId, float, PolicyId], tuple[float, ...]],
    summaries: dict[tuple[EvaluationEnvironmentId, float, PolicyId], EnvironmentBudgetSummary],
    macros: dict[tuple[float, PolicyId], HeldOutBudgetSummary],
    budget: float,
    reference: PolicyId,
    episodes_per_environment: int,
) -> HeldOutBudgetPairedDifference:
    by_environment = {
        environment_id: effects[(environment_id, budget, reference)] for environment_id in _HELD_OUT
    }
    all_effects = tuple(
        effect for environment_id in _HELD_OUT for effect in by_environment[environment_id]
    )
    environment_means = {
        environment_id: fmean(values) for environment_id, values in by_environment.items()
    }
    worst = max(_HELD_OUT, key=environment_means.__getitem__)
    candidate = macros[(budget, _CANDIDATE)]
    comparator = macros[(budget, reference)]
    macro_effect = fmean(
        summaries[(environment_id, budget, _CANDIDATE)].mean_classification_error
        - summaries[(environment_id, budget, reference)].mean_classification_error
        for environment_id in _HELD_OUT
    )
    return HeldOutBudgetPairedDifference(
        budget_fraction=budget,
        reference_policy_id=reference,
        reference_role=(
            "oracle_reference"
            if reference is PolicyId.ORACLE_TRUE_RELIABILITY_BOUNDED
            else "matched_budget_comparator"
        ),
        episodes_per_environment=episodes_per_environment,
        paired_episode_count=len(all_effects),
        candidate_macro_mean_classification_error=candidate.macro_mean_classification_error,
        reference_macro_mean_classification_error=comparator.macro_mean_classification_error,
        macro_mean_paired_difference=macro_effect,
        candidate_better_episode_count=sum(value < 0.0 for value in all_effects),
        tied_episode_count=sum(value == 0.0 for value in all_effects),
        reference_better_episode_count=sum(value > 0.0 for value in all_effects),
        worst_environment_id=worst,
        worst_environment_mean_difference=environment_means[worst],
        paired_differences_hash=canonical_sha256(
            tuple((key, by_environment[key]) for key in _HELD_OUT)
        ),
    )


def _verify_primary_effects(
    effects: dict[tuple[EvaluationEnvironmentId, float, PolicyId], tuple[float, ...]],
    paired: tuple[HeldOutBudgetPairedDifference, ...],
    primary: CanonicalPrimaryAnalysisReport,
) -> None:
    primary_by_reference = {
        row.comparator_policy_id: row for row in primary.held_out_primary_intervals
    }
    paired_by_key = {(row.budget_fraction, row.reference_policy_id): row for row in paired}
    for reference in _PRIMARY_REFERENCES:
        source = primary_by_reference[reference]
        reproduced = paired_by_key[(0.5, reference)]
        if not math.isclose(
            source.macro_mean_paired_effect,
            reproduced.macro_mean_paired_difference,
            abs_tol=1e-12,
        ):
            raise CanonicalBudgetCurveAnalysisError("Budget analysis changes a primary effect")
        source_by_environment = {
            row.environment_id: row.mean_paired_effect for row in source.environment_effects
        }
        for environment_id in _HELD_OUT:
            if not math.isclose(
                source_by_environment[environment_id],
                fmean(effects[(environment_id, 0.5, reference)]),
                abs_tol=1e-12,
            ):
                raise CanonicalBudgetCurveAnalysisError(
                    "Budget analysis changes a primary environment effect"
                )


def _verify_sources(
    source_manifest: CanonicalBudgetCurveManifest,
    source_plan: CanonicalBudgetCurveExecutionPlan,
    source_report: CanonicalBudgetCurveReport,
    primary_manifest: CanonicalPrimaryAnalysisManifest,
    primary: CanonicalPrimaryAnalysisReport,
    *,
    source_plan_path: Path,
    source_report_path: Path,
    primary_manifest_path: Path,
    primary_report_path: Path,
    stream_path: Path,
    analysis_hash: Sha256,
    pixi_hash: Sha256,
) -> None:
    primary_report_file_hash = _hash_small_file(primary_report_path, "primary report")
    checks = (
        _hash_small_file(source_plan_path, "budget execution plan")
        == source_manifest.execution_plan_file_sha256,
        source_plan.plan_hash == source_manifest.execution_plan_hash,
        _hash_small_file(source_report_path, "budget report") == source_manifest.report_file_sha256,
        source_report.report_hash == source_manifest.report_hash,
        source_report.execution_plan_hash == source_plan.plan_hash,
        source_report.budget_curve_file_sha256 == source_manifest.budget_curve_file_sha256,
        source_report.source_public_stream_sha256 == source_plan.source_public_stream_sha256,
        source_manifest.source_public_stream_sha256 == source_plan.source_public_stream_sha256,
        _hash_small_file(primary_manifest_path, "primary manifest")
        == source_plan.primary_analysis_manifest_file_sha256,
        primary_manifest.manifest_hash == source_plan.primary_analysis_manifest_hash,
        primary_report_file_hash == source_plan.primary_analysis_report_file_sha256,
        primary_report_file_hash == primary_manifest.report_file_sha256,
        primary.report_hash == source_plan.primary_analysis_report_hash,
        primary.report_hash == primary_manifest.report_hash,
        source_plan.frozen_analysis_specification_hash == analysis_hash,
        source_plan.pixi_lock_sha256 == pixi_hash,
        source_report.primary_budget_reproduction_verified,
        stream_path.is_file(),
    )
    if not all(checks):
        raise CanonicalBudgetCurveAnalysisError("Canonical budget sources do not reconcile")


def _outcome(metric: BudgetCurveEpisodeMetric) -> tuple[object, ...]:
    return (
        metric.selected_probe_count,
        metric.observed_probe_pass_count,
        metric.observed_probe_fail_count,
        metric.missing_probe_result_count,
        metric.classification_error_count,
        metric.final_classification_error,
        metric.squared_error_sum,
        metric.negative_log_likelihood_sum,
        metric.calibration_bin_counts,
        metric.calibration_probability_sums,
        metric.calibration_positive_counts,
        metric.case_results_hash,
    )


def _calibration_error(totals: _Totals) -> float:
    count = sum(totals.bin_counts)
    if not count:
        raise CanonicalBudgetCurveAnalysisError("Calibration summary contains no cases")
    return math.fsum(
        bin_count / count * abs(probability.total / bin_count - positive / bin_count)
        for bin_count, probability, positive in zip(
            totals.bin_counts,
            totals.bin_probability,
            totals.bin_positive,
            strict=True,
        )
        if bin_count
    )


def _csv_table(rows: Iterable[ContractModel], fields: tuple[str, ...]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(fields)
    for row in rows:
        values = row.model_dump(mode="json")
        writer.writerow(values[field] for field in fields)
    return output.getvalue().encode("utf-8")


def _load[ModelT: ContractModel](path: Path, model: type[ModelT]) -> ModelT:
    try:
        return model.model_validate_json(path.read_bytes())
    except (OSError, ValidationError) as error:
        raise CanonicalBudgetCurveAnalysisError(f"Could not load source: {path}") from error


def _hash_small_file(path: Path, label: str) -> Sha256:
    try:
        return file_sha256(path.read_bytes())
    except OSError as error:
        raise CanonicalBudgetCurveAnalysisError(f"Could not read {label}: {path}") from error


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BUDGET_ROOT = (
    PROJECT_ROOT / "artifacts" / "acquisition-study" / "canonical-v1" / "secondary-budget-v1"
)
DEFAULT_PRIMARY_ROOT = (
    PROJECT_ROOT / "artifacts" / "acquisition-study" / "canonical-v1" / "primary-analysis-v1"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyse the canonical probe-budget curve")
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=DEFAULT_BUDGET_ROOT / "canonical_budget_curve_manifest.json",
    )
    parser.add_argument(
        "--primary-manifest",
        type=Path,
        default=DEFAULT_PRIMARY_ROOT / "canonical_primary_analysis_manifest.json",
    )
    parser.add_argument(
        "--environments",
        type=Path,
        default=PROJECT_ROOT / "configs" / "acquisition-study" / "v1-environments.yaml",
    )
    parser.add_argument(
        "--analysis",
        type=Path,
        default=PROJECT_ROOT / "configs" / "acquisition-study" / "v1-analysis.yaml",
    )
    parser.add_argument("--pixi-lock", type=Path, default=PROJECT_ROOT / "pixi.lock")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_BUDGET_ROOT / "analysis-v1")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_canonical_budget_curve_analysis(
            source_manifest_path=args.source_manifest,
            primary_manifest_path=args.primary_manifest,
            environment_path=args.environments,
            analysis_path=args.analysis,
            pixi_lock_path=args.pixi_lock,
            output_root=args.output_root,
            analysis_code_revision=current_clean_revision(
                PROJECT_ROOT,
                dirty_message="Commit or remove all visible changes before canonical analysis",
                untracked_files="all",
                required_branch="main",
                operation_name="Canonical analysis",
            ),
        )
    except Exception as error:
        print(
            json.dumps(
                {
                    "command": "acquisition-canonical-analyse-budget-curve",
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "status": "error",
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "artifact_locations": artifact_locations(args.output_root),
                "command": "acquisition-canonical-analyse-budget-curve",
                "result": report,
                "status": "ok",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0
