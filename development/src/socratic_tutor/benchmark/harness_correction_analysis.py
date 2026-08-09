# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
"""Deterministic post-hoc analysis of a completed harness correction replay."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from socratic_tutor.benchmark.analysis_spec import (
    AnalysisSpecification,
    analysis_specification_hash,
    load_analysis_specification,
)
from socratic_tutor.benchmark.artifacts import write_immutable_json
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.harness_correction_replay import (
    HarnessCorrectionReplayPlan,
    HarnessCorrectionReplayReport,
    HarnessExecutionSnapshot,
)
from socratic_tutor.benchmark.hashing import file_sha256, model_content_hash
from socratic_tutor.benchmark.inferential_hierarchy import InferentialHierarchy
from socratic_tutor.benchmark.methodology_clarification import MethodologyClarification
from socratic_tutor.benchmark.primary_analysis import PrimaryAnalysisReport
from socratic_tutor.benchmark.public.controls import ControlFactory, ProbeExecutionSummary
from socratic_tutor.benchmark.public.models import (
    EXPECTED_CONDITIONS,
    BenchmarkCondition,
    PublicBenchmarkManifest,
)
from socratic_tutor.benchmark.public.offline import load_initial_tracker_state
from socratic_tutor.benchmark.public.runner import PairedConditionRunner
from socratic_tutor.benchmark.public_rating import PublicAnswerRatingReport
from socratic_tutor.benchmark.secondary_analysis import SecondaryAnalysisReport
from socratic_tutor.benchmark.statistics import (
    McNemarResult,
    PairedCaseInference,
    exact_mcnemar,
    paired_case_inference,
)
from socratic_tutor.contracts import ContractModel, Evidence
from socratic_tutor.policies import choose_action
from socratic_tutor.tracking import update_tracker


class HarnessCorrectionAnalysisError(ValueError):
    """Correction analysis inputs do not share one verified lineage."""


class CorrectedConditionScore(ContractModel):
    """One reconstructed tracker prediction and its corrected loss."""

    condition: BenchmarkCondition
    mastery_score: float = Field(ge=0.0, le=1.0)
    policy_threshold: float = Field(ge=0.0, le=1.0)
    binary_decision: bool
    correct: bool | None
    classification_error: float | None = Field(default=None, ge=0.0, le=1.0)
    input_hash: Sha256

    @model_validator(mode="after")
    def validate_score(self) -> CorrectedConditionScore:
        if self.binary_decision is not (self.mastery_score >= self.policy_threshold):
            raise ValueError("Corrected binary decision does not match its score")
        if self.correct is None:
            if self.classification_error is not None:
                raise ValueError("Missing criterion outcome cannot have a loss")
        elif self.classification_error != float(not self.correct):
            raise ValueError("Corrected loss does not match condition correctness")
        return self


class HarnessCorrectionCaseResult(ContractModel):
    """Hand-checkable corrected results for one authored case."""

    case_id: str = Field(min_length=1)
    public_rating_category: str = Field(min_length=1)
    original_evidence: HarnessExecutionSnapshot
    corrected_evidence: HarnessExecutionSnapshot
    original_criterion: HarnessExecutionSnapshot
    corrected_criterion: HarnessExecutionSnapshot
    conditions: tuple[CorrectedConditionScore, ...] = Field(min_length=4, max_length=4)
    primary_effect: float | None = Field(default=None, ge=-1.0, le=1.0)
    valid_vs_unrelated_effect: float | None = Field(default=None, ge=-1.0, le=1.0)
    valid_vs_corrupted_effect: float | None = Field(default=None, ge=-1.0, le=1.0)
    combined_specificity_effect: float | None = Field(default=None, ge=-1.0, le=1.0)
    evidence_attempt_hash: Sha256
    criterion_attempt_hash: Sha256
    result_hash: Sha256

    @model_validator(mode="after")
    def validate_case(self) -> HarnessCorrectionCaseResult:
        if tuple(item.condition for item in self.conditions) != EXPECTED_CONDITIONS:
            raise ValueError("Corrected case must preserve the four condition order")
        effects = (
            self.primary_effect,
            self.valid_vs_unrelated_effect,
            self.valid_vs_corrupted_effect,
            self.combined_specificity_effect,
        )
        if self.corrected_criterion.demonstrated_performance is None:
            if any(effect is not None for effect in effects):
                raise ValueError("Missing corrected criterion cannot produce case effects")
        else:
            losses = {item.condition: item.classification_error for item in self.conditions}
            dialogue = losses[BenchmarkCondition.DIALOGUE_ONLY]
            valid = losses[BenchmarkCondition.PROBE_INFORMED]
            unrelated = losses[BenchmarkCondition.UNRELATED_PROBE]
            corrupted = losses[BenchmarkCondition.CORRUPTED_PROBE]
            if any(value is None for value in (dialogue, valid, unrelated, corrupted)):
                raise ValueError("Observed corrected criterion requires all condition losses")
            assert dialogue is not None
            assert valid is not None
            assert unrelated is not None
            assert corrupted is not None
            expected = (
                dialogue - valid,
                unrelated - valid,
                corrupted - valid,
                ((unrelated - valid) + (corrupted - valid)) / 2,
            )
            if any(actual != value for actual, value in zip(effects, expected, strict=True)):
                raise ValueError("Corrected case effects do not match condition losses")
        if self.result_hash != model_content_hash(self, exclude={"result_hash"}):
            raise ValueError("Harness correction case hash does not match its content")
        return self


class HarnessCorrectionAnalysisPlan(ContractModel):
    """Verified sources and software identity for the post-hoc analysis."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.harness_correction_analysis_plan.v1"] = (
        "benchmark.harness_correction_analysis_plan.v1"
    )
    correction_replay_plan_hash: Sha256
    correction_replay_report_hash: Sha256
    source_manifest_hash: Sha256
    public_projection_hash: Sha256
    public_rating_report_hash: Sha256
    methodology_clarification_hash: Sha256
    inferential_hierarchy_hash: Sha256
    analysis_specification_hash: Sha256
    original_primary_report_hash: Sha256
    original_secondary_report_hash: Sha256
    tracker_version: Literal["simple-v1"] = "simple-v1"
    policy_version: Literal["heuristic-v1"] = "heuristic-v1"
    analysis_role: Literal["post_hoc_correction_not_confirmatory"] = (
        "post_hoc_correction_not_confirmatory"
    )
    analysis_code_revision: str = Field(min_length=1)
    pixi_lock_sha256: Sha256
    created_at_utc: datetime
    plan_hash: Sha256

    @model_validator(mode="after")
    def validate_plan(self) -> HarnessCorrectionAnalysisPlan:
        _require_utc(self.created_at_utc, "Harness correction analysis plan time")
        if self.plan_hash != model_content_hash(self, exclude={"plan_hash"}):
            raise ValueError("Harness correction analysis plan hash does not match its content")
        return self


class HarnessCorrectionAnalysisReport(ContractModel):
    """Corrected paired result with the original sealed result retained for comparison."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.harness_correction_analysis_report.v1"] = (
        "benchmark.harness_correction_analysis_report.v1"
    )
    analysis_plan_hash: Sha256
    total_case_count: Literal[24]
    eligible_case_count: int = Field(ge=1, le=24)
    missing_case_ids: tuple[str, ...]
    case_results: tuple[HarnessCorrectionCaseResult, ...] = Field(min_length=24, max_length=24)
    primary_inference: PairedCaseInference
    primary_mcnemar: McNemarResult
    improvement_count: int = Field(ge=0)
    regression_count: int = Field(ge=0)
    net_improvement_count: int
    valid_vs_unrelated_inference: PairedCaseInference
    valid_vs_corrupted_inference: PairedCaseInference
    combined_specificity_inference: PairedCaseInference
    original_sealed_primary_effect: float = Field(ge=-1.0, le=1.0)
    original_sealed_primary_interval: tuple[float, float]
    original_sealed_mcnemar_p_value: float = Field(ge=0.0, le=1.0)
    original_sealed_combined_specificity_effect: float = Field(ge=-1.0, le=1.0)
    corrected_primary_effect: float = Field(ge=-1.0, le=1.0)
    corrected_combined_specificity_effect: float = Field(ge=-1.0, le=1.0)
    valid_evidence_beats_unrelated_control: bool
    canonical_primary_replacement_allowed: Literal[False] = False
    human_learning_claim_supported: Literal[False] = False
    provider_calls_made: Literal[0] = 0
    sandbox_calls_made: Literal[0] = 0
    completed_at_utc: datetime
    report_hash: Sha256

    @model_validator(mode="after")
    def validate_report(self) -> HarnessCorrectionAnalysisReport:
        _require_utc(self.completed_at_utc, "Harness correction analysis completion time")
        if len(self.case_results) != self.total_case_count:
            raise ValueError("Correction analysis must retain all 24 cases")
        if self.eligible_case_count + len(self.missing_case_ids) != self.total_case_count:
            raise ValueError("Correction analysis case counts do not reconcile")
        if self.primary_inference.case_count != self.eligible_case_count:
            raise ValueError("Correction primary inference uses another denominator")
        if self.primary_mcnemar.pair_count != self.eligible_case_count:
            raise ValueError("Correction McNemar result uses another denominator")
        if self.improvement_count != self.primary_mcnemar.right_only_correct_count:
            raise ValueError("Correction improvement count differs from McNemar")
        if self.regression_count != self.primary_mcnemar.left_only_correct_count:
            raise ValueError("Correction regression count differs from McNemar")
        if self.net_improvement_count != self.improvement_count - self.regression_count:
            raise ValueError("Correction net improvement count does not reconcile")
        if self.corrected_primary_effect != self.primary_inference.mean_case_effect:
            raise ValueError("Correction primary summary differs from its inference")
        if (
            self.corrected_combined_specificity_effect
            != self.combined_specificity_inference.mean_case_effect
        ):
            raise ValueError("Correction specificity summary differs from its inference")
        if self.valid_evidence_beats_unrelated_control is not (
            self.valid_vs_unrelated_inference.mean_case_effect > 0.0
        ):
            raise ValueError("Correction specificity flag differs from unrelated control")
        if self.report_hash != model_content_hash(self, exclude={"report_hash"}):
            raise ValueError("Harness correction analysis hash does not match its content")
        return self


def run_harness_correction_analysis(
    *,
    correction_report_path: Path,
    public_manifest_path: Path,
    public_rating_report_path: Path,
    methodology_clarification_path: Path,
    inferential_hierarchy_path: Path,
    analysis_specification_path: Path,
    original_primary_report_path: Path,
    original_secondary_report_path: Path,
    benchmark_root: Path,
    pixi_lock_path: Path,
    output_root: Path,
    analysis_code_revision: str,
    created_at_utc: datetime | None = None,
) -> HarnessCorrectionAnalysisReport:
    """Reconstruct corrected conditions and calculate paired post-hoc statistics."""

    correction = _load_json(correction_report_path, HarnessCorrectionReplayReport)
    correction_plan = _load_json(
        correction_report_path.resolve().parent / "harness_correction_replay_plan.json",
        HarnessCorrectionReplayPlan,
    )
    public = _load_json(public_manifest_path, PublicBenchmarkManifest)
    ratings = _load_json(public_rating_report_path, PublicAnswerRatingReport)
    methodology = _load_json(methodology_clarification_path, MethodologyClarification)
    hierarchy = _load_json(inferential_hierarchy_path, InferentialHierarchy)
    specification = load_analysis_specification(analysis_specification_path)
    original_primary = _load_json(original_primary_report_path, PrimaryAnalysisReport)
    original_secondary = _load_json(original_secondary_report_path, SecondaryAnalysisReport)
    root = output_root.resolve()
    plan_path = root / "harness_correction_analysis_plan.json"
    existing_plan = (
        _load_json(plan_path, HarnessCorrectionAnalysisPlan) if plan_path.exists() else None
    )
    effective_created_at = (
        existing_plan.created_at_utc
        if existing_plan is not None
        else created_at_utc or datetime.now(UTC)
    )
    _validate_sources(
        correction=correction,
        correction_plan=correction_plan,
        public=public,
        ratings=ratings,
        methodology=methodology,
        hierarchy=hierarchy,
        specification=specification,
        original_primary=original_primary,
        original_secondary=original_secondary,
        created_at_utc=effective_created_at,
    )
    try:
        pixi_hash = file_sha256(pixi_lock_path.read_bytes())
    except OSError as error:
        raise HarnessCorrectionAnalysisError("Could not read Pixi lock") from error
    plan = _create_plan(
        correction=correction,
        correction_plan=correction_plan,
        public=public,
        ratings=ratings,
        methodology=methodology,
        hierarchy=hierarchy,
        specification=specification,
        original_primary=original_primary,
        original_secondary=original_secondary,
        pixi_hash=pixi_hash,
        analysis_code_revision=analysis_code_revision,
        created_at_utc=effective_created_at,
    )
    if existing_plan is not None and existing_plan != plan:
        raise HarnessCorrectionAnalysisError(
            "Existing correction analysis plan uses different sources or code"
        )
    write_immutable_json(plan_path, plan)
    report_path = root / "harness_correction_analysis_report.json"
    if report_path.exists():
        report = _load_json(report_path, HarnessCorrectionAnalysisReport)
        if report.analysis_plan_hash != plan.plan_hash:
            raise HarnessCorrectionAnalysisError("Existing correction analysis uses another plan")
        return report

    cases = _reconstruct_cases(
        correction=correction,
        public=public,
        ratings=ratings,
        methodology=methodology,
        benchmark_root=benchmark_root,
    )
    eligible = tuple(case for case in cases if case.primary_effect is not None)
    primary_effects = tuple(_present(case.primary_effect) for case in eligible)
    valid_unrelated = tuple(_present(case.valid_vs_unrelated_effect) for case in eligible)
    valid_corrupted = tuple(_present(case.valid_vs_corrupted_effect) for case in eligible)
    combined = tuple(_present(case.combined_specificity_effect) for case in eligible)
    primary_inference = _inference(primary_effects, specification)
    condition_maps = [{score.condition: score for score in case.conditions} for case in eligible]
    pairs = tuple(
        (
            _present_bool(scores[BenchmarkCondition.DIALOGUE_ONLY].correct),
            _present_bool(scores[BenchmarkCondition.PROBE_INFORMED].correct),
        )
        for scores in condition_maps
    )
    mcnemar = exact_mcnemar(pairs)
    unrelated_inference = _inference(valid_unrelated, specification)
    corrupted_inference = _inference(valid_corrupted, specification)
    combined_inference = _inference(combined, specification)
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.harness_correction_analysis_report.v1",
        "analysis_plan_hash": plan.plan_hash,
        "total_case_count": 24,
        "eligible_case_count": len(eligible),
        "missing_case_ids": tuple(case.case_id for case in cases if case.primary_effect is None),
        "case_results": cases,
        "primary_inference": primary_inference,
        "primary_mcnemar": mcnemar,
        "improvement_count": mcnemar.right_only_correct_count,
        "regression_count": mcnemar.left_only_correct_count,
        "net_improvement_count": (
            mcnemar.right_only_correct_count - mcnemar.left_only_correct_count
        ),
        "valid_vs_unrelated_inference": unrelated_inference,
        "valid_vs_corrupted_inference": corrupted_inference,
        "combined_specificity_inference": combined_inference,
        "original_sealed_primary_effect": original_primary.inference.mean_case_effect,
        "original_sealed_primary_interval": (
            original_primary.inference.interval_lower,
            original_primary.inference.interval_upper,
        ),
        "original_sealed_mcnemar_p_value": original_primary.mcnemar.two_sided_exact_p_value,
        "original_sealed_combined_specificity_effect": (
            original_secondary.evidence_specificity.combined_specificity_inference.mean_case_effect
        ),
        "corrected_primary_effect": primary_inference.mean_case_effect,
        "corrected_combined_specificity_effect": combined_inference.mean_case_effect,
        "valid_evidence_beats_unrelated_control": (unrelated_inference.mean_case_effect > 0.0),
        "canonical_primary_replacement_allowed": False,
        "human_learning_claim_supported": False,
        "provider_calls_made": 0,
        "sandbox_calls_made": 0,
        "completed_at_utc": effective_created_at,
    }
    draft = HarnessCorrectionAnalysisReport.model_construct(
        _fields_set=set(content), **content, report_hash="0" * 64
    )
    report = HarnessCorrectionAnalysisReport.model_validate(
        {**content, "report_hash": model_content_hash(draft, exclude={"report_hash"})}
    )
    write_immutable_json(report_path, report)
    return report


def _reconstruct_cases(
    *,
    correction: HarnessCorrectionReplayReport,
    public: PublicBenchmarkManifest,
    ratings: PublicAnswerRatingReport,
    methodology: MethodologyClarification,
    benchmark_root: Path,
) -> tuple[HarnessCorrectionCaseResult, ...]:
    attempts = {(item.case_id, item.channel.value): item for item in correction.attempts}
    rating_by_case = {item.case_id: item for item in ratings.final_ratings}
    controls = ControlFactory(benchmark_root, public)
    probe_by_case: dict[str, ProbeExecutionSummary] = {}
    for case in public.cases:
        attempt = attempts[(case.case_id, "evidence")]
        if attempt.corrected.status != "completed":
            raise HarnessCorrectionAnalysisError(
                f"Corrected evidence is unavailable: {case.case_id}"
            )
        probe_by_case[case.case_id] = controls.probe_summary(
            case_id=case.case_id,
            passed=attempt.corrected.passed,
            failed=attempt.corrected.failed,
        )
    runner = PairedConditionRunner(
        public,
        tracker_update=update_tracker,
        policy_decision=choose_action,
        tracker_version=methodology.tracker_version,
        policy_version="heuristic-v1",
    )
    results: list[HarnessCorrectionCaseResult] = []
    for case in public.cases:
        case_id = case.case_id
        evidence_attempt = attempts[(case_id, "evidence")]
        criterion_attempt = attempts[(case_id, "criterion")]
        rating = rating_by_case[case_id]
        probe = probe_by_case[case_id]
        paired = runner.run_case(
            case_id=case_id,
            initial_tracker_state=load_initial_tracker_state(
                benchmark_root,
                public,
                case_id=case_id,
            ),
            public_evidence=Evidence(
                category=rating.category,
                confidence=rating.confidence,
                rationale=f"Independent public-answer rating resolved by {rating.resolution}.",
                observed_signals=(f"rating_resolution:{rating.resolution}",),
            ),
            probe_evidence=probe,
            unrelated_evidence=controls.unrelated(
                case_id=case_id,
                evidence_by_case=probe_by_case,
            ),
            corrupted_evidence=controls.corrupted(case_id=case_id, evidence=probe),
        )
        criterion = criterion_attempt.corrected.demonstrated_performance
        scores = tuple(
            CorrectedConditionScore(
                condition=outcome.condition,
                mastery_score=outcome.tracker_after.mastery_probability,
                policy_threshold=methodology.policy_threshold,
                binary_decision=(
                    outcome.tracker_after.mastery_probability >= methodology.policy_threshold
                ),
                correct=(
                    None
                    if criterion is None
                    else (outcome.tracker_after.mastery_probability >= methodology.policy_threshold)
                    is criterion
                ),
                classification_error=(
                    None
                    if criterion is None
                    else float(
                        (outcome.tracker_after.mastery_probability >= methodology.policy_threshold)
                        is not criterion
                    )
                ),
                input_hash=outcome.input_hash,
            )
            for outcome in paired.outcomes
        )
        losses = {score.condition: score.classification_error for score in scores}
        dialogue = losses[BenchmarkCondition.DIALOGUE_ONLY]
        valid = losses[BenchmarkCondition.PROBE_INFORMED]
        unrelated = losses[BenchmarkCondition.UNRELATED_PROBE]
        corrupted = losses[BenchmarkCondition.CORRUPTED_PROBE]
        effects = (
            (None, None, None, None)
            if criterion is None
            else _case_effects(
                _present(dialogue),
                _present(valid),
                _present(unrelated),
                _present(corrupted),
            )
        )
        content = {
            "case_id": case_id,
            "public_rating_category": rating.category.value,
            "original_evidence": evidence_attempt.original,
            "corrected_evidence": evidence_attempt.corrected,
            "original_criterion": criterion_attempt.original,
            "corrected_criterion": criterion_attempt.corrected,
            "conditions": scores,
            "primary_effect": effects[0],
            "valid_vs_unrelated_effect": effects[1],
            "valid_vs_corrupted_effect": effects[2],
            "combined_specificity_effect": effects[3],
            "evidence_attempt_hash": evidence_attempt.attempt_hash,
            "criterion_attempt_hash": criterion_attempt.attempt_hash,
        }
        draft = HarnessCorrectionCaseResult.model_construct(
            _fields_set=set(content), **content, result_hash="0" * 64
        )
        results.append(
            HarnessCorrectionCaseResult.model_validate(
                {**content, "result_hash": model_content_hash(draft, exclude={"result_hash"})}
            )
        )
    return tuple(sorted(results, key=lambda item: item.case_id))


def _case_effects(
    dialogue: float,
    valid: float,
    unrelated: float,
    corrupted: float,
) -> tuple[float, float, float, float]:
    primary = dialogue - valid
    valid_unrelated = unrelated - valid
    valid_corrupted = corrupted - valid
    return primary, valid_unrelated, valid_corrupted, (valid_unrelated + valid_corrupted) / 2


def _inference(
    effects: tuple[float, ...], specification: AnalysisSpecification
) -> PairedCaseInference:
    return paired_case_inference(
        effects,
        bootstrap_resamples=specification.bootstrap_resamples,
        permutation_resamples=specification.permutation_resamples,
        minimum_interpretable_effect=specification.minimum_interpretable_effect,
        random_seed=specification.random_seed,
    )


def _create_plan(
    *,
    correction: HarnessCorrectionReplayReport,
    correction_plan: HarnessCorrectionReplayPlan,
    public: PublicBenchmarkManifest,
    ratings: PublicAnswerRatingReport,
    methodology: MethodologyClarification,
    hierarchy: InferentialHierarchy,
    specification: AnalysisSpecification,
    original_primary: PrimaryAnalysisReport,
    original_secondary: SecondaryAnalysisReport,
    pixi_hash: str,
    analysis_code_revision: str,
    created_at_utc: datetime,
) -> HarnessCorrectionAnalysisPlan:
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.harness_correction_analysis_plan.v1",
        "correction_replay_plan_hash": correction_plan.plan_hash,
        "correction_replay_report_hash": correction.report_hash,
        "source_manifest_hash": correction_plan.source_manifest_hash,
        "public_projection_hash": public.projection_hash,
        "public_rating_report_hash": ratings.report_hash,
        "methodology_clarification_hash": methodology.clarification_hash,
        "inferential_hierarchy_hash": hierarchy.hierarchy_hash,
        "analysis_specification_hash": analysis_specification_hash(specification),
        "original_primary_report_hash": original_primary.report_hash,
        "original_secondary_report_hash": original_secondary.report_hash,
        "tracker_version": methodology.tracker_version,
        "policy_version": "heuristic-v1",
        "analysis_role": "post_hoc_correction_not_confirmatory",
        "analysis_code_revision": analysis_code_revision,
        "pixi_lock_sha256": pixi_hash,
        "created_at_utc": created_at_utc,
    }
    draft = HarnessCorrectionAnalysisPlan.model_construct(
        _fields_set=set(content), **content, plan_hash="0" * 64
    )
    return HarnessCorrectionAnalysisPlan.model_validate(
        {**content, "plan_hash": model_content_hash(draft, exclude={"plan_hash"})}
    )


def _validate_sources(
    *,
    correction: HarnessCorrectionReplayReport,
    correction_plan: HarnessCorrectionReplayPlan,
    public: PublicBenchmarkManifest,
    ratings: PublicAnswerRatingReport,
    methodology: MethodologyClarification,
    hierarchy: InferentialHierarchy,
    specification: AnalysisSpecification,
    original_primary: PrimaryAnalysisReport,
    original_secondary: SecondaryAnalysisReport,
    created_at_utc: datetime,
) -> None:
    _require_utc(created_at_utc, "Harness correction analysis time")
    if not correction.gate_passed or correction.provider_call_count != 0:
        raise HarnessCorrectionAnalysisError("Correction replay did not pass cleanly")
    if correction.replay_plan_hash != correction_plan.plan_hash:
        raise HarnessCorrectionAnalysisError("Correction report belongs to another replay plan")
    if public.source_manifest_hash != correction_plan.source_manifest_hash:
        raise HarnessCorrectionAnalysisError("Public manifest belongs to another benchmark")
    if len(public.cases) != 24 or ratings.item_count != 24:
        raise HarnessCorrectionAnalysisError("Correction analysis requires all 24 authored cases")
    case_ids = {case.case_id for case in public.cases}
    rating_ids = {rating.case_id for rating in ratings.final_ratings}
    attempt_keys = {(attempt.case_id, attempt.channel.value) for attempt in correction.attempts}
    expected_attempt_keys = {
        (case_id, channel) for case_id in case_ids for channel in ("evidence", "criterion")
    }
    if case_ids != rating_ids or attempt_keys != expected_attempt_keys:
        raise HarnessCorrectionAnalysisError("Correction sources have different case coverage")
    specification_hash = analysis_specification_hash(specification)
    if methodology.public_rating_report_hash != ratings.report_hash:
        raise HarnessCorrectionAnalysisError("Methodology names another public rating report")
    if methodology.analysis_specification_hash != specification_hash:
        raise HarnessCorrectionAnalysisError("Methodology names another analysis specification")
    if hierarchy.analysis_specification_hash != specification_hash:
        raise HarnessCorrectionAnalysisError("Hierarchy names another analysis specification")
    if original_secondary.primary_analysis_report_hash != original_primary.report_hash:
        raise HarnessCorrectionAnalysisError("Original secondary report names another primary")
    if original_primary.run_id != original_secondary.run_id:
        raise HarnessCorrectionAnalysisError("Original reports belong to different runs")
    latest_source_time = max(
        correction.completed_at_utc,
        original_primary.completed_at_utc,
        original_secondary.completed_at_utc,
    )
    if created_at_utc <= latest_source_time:
        raise HarnessCorrectionAnalysisError("Correction analysis must follow all source reports")


def _load_json[ModelT: ContractModel](path: Path, model: type[ModelT]) -> ModelT:
    try:
        return model.model_validate_json(path.read_bytes())
    except (OSError, ValueError) as error:
        raise HarnessCorrectionAnalysisError(f"Could not verify analysis source: {path}") from error


def _present(value: float | None) -> float:
    if value is None:
        raise HarnessCorrectionAnalysisError("Expected an observed corrected loss")
    return value


def _present_bool(value: bool | None) -> bool:
    if value is None:
        raise HarnessCorrectionAnalysisError("Expected an observed corrected outcome")
    return value


def _require_utc(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError(f"{name} must use UTC")
