"""Network-free decision generation for development benchmark rehearsals."""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from socratic_tutor.benchmark.artifacts import (
    write_immutable_json,
    write_immutable_jsonl,
)
from socratic_tutor.benchmark.common import RelativePath, Sha256
from socratic_tutor.benchmark.generation import (
    GenerationRequestSpec,
    ModelRoute,
    SamplingConfig,
)
from socratic_tutor.benchmark.hashing import (
    canonical_sha256,
    file_sha256,
    model_content_hash,
)
from socratic_tutor.benchmark.parquet import AtomicParquetDatasetStore
from socratic_tutor.benchmark.public.commitments import (
    BenchmarkSampleKey,
    FilesystemConditionCommitStore,
)
from socratic_tutor.benchmark.public.controls import ControlFactory, ProbeExecutionSummary
from socratic_tutor.benchmark.public.datasets import publish_condition_predictions
from socratic_tutor.benchmark.public.global_seal import (
    DecisionRunProvenance,
    FilesystemGlobalDecisionSealStore,
    create_decision_run_plan,
)
from socratic_tutor.benchmark.public.models import (
    BenchmarkSplit,
    PublicArtifactClass,
    PublicBenchmarkManifest,
)
from socratic_tutor.benchmark.public.predictions import (
    DecisionPredictionRecord,
    PredictionRecordFactory,
)
from socratic_tutor.benchmark.public.requests import PublicChannelRequestBuilder
from socratic_tutor.benchmark.public.runner import PairedConditionRunner
from socratic_tutor.benchmark.replay import (
    OriginalGenerationSource,
    RecordedGenerationResponse,
    create_recorded_response,
    replay_recorded_response,
)
from socratic_tutor.contracts import ContractModel, Evidence, TrackerState, TutorAction
from socratic_tutor.policies import choose_action
from socratic_tutor.tracking import update_tracker


class OfflineDecisionError(ValueError):
    """A development-only decision rehearsal is invalid or unsafe."""


class OfflineDecisionCase(ContractModel):
    """Authored responses and normalized public evidence for one development case."""

    case_id: str = Field(min_length=1)
    public_response: str = Field(min_length=1)
    evidence_response: str = Field(min_length=1)
    public_evidence: Evidence
    probe_passed: int = Field(ge=0)
    probe_failed: int = Field(ge=0)

    @model_validator(mode="after")
    def require_probe_result(self) -> "OfflineDecisionCase":
        if self.probe_passed + self.probe_failed == 0:
            raise ValueError("Offline evidence probe requires at least one test result")
        return self


class OfflineDecisionPlan(ContractModel):
    """Typed, development-only input for a reproducible decision phase."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.offline_decision_plan.v1"] = "benchmark.offline_decision_plan.v1"
    mode: Literal["development_offline"] = "development_offline"
    run_id: str = Field(min_length=1)
    sample_ids: tuple[str, ...] = Field(min_length=1, max_length=100)
    model_route: ModelRoute
    sampling: SamplingConfig
    system_prompt_version: str = Field(min_length=1)
    system_prompt_ref: RelativePath
    system_prompt_sha256: Sha256
    tracker_version: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)
    code_revision: str = Field(min_length=1)
    dirty_worktree: bool
    environment_lock_hash: Sha256
    calibration_decision_hash: Sha256 | None = None
    root_seed: int = Field(ge=0, le=2**32 - 1)
    created_at_utc: datetime
    cases: tuple[OfflineDecisionCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_plan(self) -> "OfflineDecisionPlan":
        _require_utc(self.created_at_utc, "Offline decision-plan creation time")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("Offline decision sample IDs must be unique")
        case_ids = tuple(case.case_id for case in self.cases)
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("Offline decision case IDs must be unique")
        return self


class DecisionGenerationSummary(ContractModel):
    """Machine-readable result of a sealed offline decision phase."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.decision_generation_summary.v1"] = (
        "benchmark.decision_generation_summary.v1"
    )
    benchmark_version: str
    run_id: str
    case_count: int = Field(ge=1)
    sample_count: int = Field(ge=1)
    prediction_count: int = Field(ge=1)
    recorded_response_count: int = Field(ge=1)
    run_plan_hash: Sha256
    global_seal_hash: Sha256
    decision_publication_hash: Sha256


class ProbeSummarySet(ContractModel):
    """Content-addressed public probe summaries reused by scoring baselines."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.probe_summary_set.v1"] = "benchmark.probe_summary_set.v1"
    benchmark_version: str = Field(min_length=1)
    records: tuple[ProbeExecutionSummary, ...] = Field(min_length=1)
    set_hash: Sha256

    @model_validator(mode="after")
    def validate_set(self) -> "ProbeSummarySet":
        if len({record.case_id for record in self.records}) != len(self.records):
            raise ValueError("Probe summary set contains duplicate cases")
        if self.set_hash != model_content_hash(self, exclude={"set_hash"}):
            raise ValueError("Probe summary set hash does not match its content")
        return self


class _InitialTrackerArtifact(ContractModel):
    schema_id: Literal["benchmark.initial_tracker.v1"]
    concept_mastery_probability: float = Field(ge=0.0, le=1.0)
    observations: int = Field(ge=0)


def run_offline_decision_phase(
    *,
    manifest: PublicBenchmarkManifest,
    benchmark_root: Path,
    plan: OfflineDecisionPlan,
    output_root: Path,
) -> DecisionGenerationSummary:
    """Run, commit, globally seal, and publish development-only predictions."""

    _validate_decision_scope(manifest, plan)
    root = output_root.resolve()
    decision_root = root / "decision"
    datasets_root = root / "datasets"
    prompt = read_verified_public_file(
        benchmark_root,
        manifest,
        plan.system_prompt_ref,
        expected_class=PublicArtifactClass.PROMPT,
    )
    if file_sha256(prompt) != plan.system_prompt_sha256:
        raise OfflineDecisionError("Offline plan system-prompt hash does not match its content")

    route_id = f"{plan.model_route.provider}/{plan.model_route.model}"
    planned_keys = tuple(
        _sample_key(
            benchmark_version=manifest.benchmark_version,
            run_id=plan.run_id,
            case_id=case.case_id,
            sample_id=sample_id,
            model_route_id=route_id,
        )
        for case in manifest.cases
        for sample_id in plan.sample_ids
    )
    provenance = DecisionRunProvenance(
        code_revision=plan.code_revision,
        dirty_worktree=plan.dirty_worktree,
        pixi_lock_hash=plan.environment_lock_hash,
        resolved_config_hash=model_content_hash(plan),
        public_manifest_hash=manifest.projection_hash,
        split_hash=canonical_sha256(manifest.splits),
        prompt_version=plan.system_prompt_version,
        prompt_hash=plan.system_prompt_sha256,
        tracker_version=plan.tracker_version,
        policy_version=plan.policy_version,
        observation_schema="benchmark.condition_outcome.v1",
        action_mapping_hash=canonical_sha256(tuple(TutorAction)),
        calibration_decision_hash=plan.calibration_decision_hash,
        analysis_plan_hash=canonical_sha256(
            {"schema_id": "benchmark.offline_analysis_plan.v1", "run_id": plan.run_id}
        ),
        review_gate_hash=manifest.source_manifest_hash,
        root_seed=plan.root_seed,
    )
    run_plan = create_decision_run_plan(
        benchmark_version=manifest.benchmark_version,
        run_id=plan.run_id,
        provenance=provenance,
        planned_samples=planned_keys,
        created_at_utc=plan.created_at_utc,
    )
    write_immutable_json(root / "run_plan.json", run_plan)

    condition_sealed_at = plan.created_at_utc + timedelta(hours=1)
    global_sealed_at = condition_sealed_at + timedelta(minutes=1)
    conditions = FilesystemConditionCommitStore(
        decision_root,
        clock=lambda: condition_sealed_at,
    )
    controls = ControlFactory(benchmark_root, manifest)
    request_builder = PublicChannelRequestBuilder(benchmark_root, manifest)
    runner = PairedConditionRunner(
        manifest,
        tracker_update=update_tracker,
        policy_decision=choose_action,
        tracker_version=plan.tracker_version,
        policy_version=plan.policy_version,
    )
    prediction_factory = PredictionRecordFactory(manifest)
    case_inputs = {case.case_id: case for case in plan.cases}
    probe_by_case = {
        case.case_id: controls.probe_summary(
            case_id=case.case_id,
            passed=case.probe_passed,
            failed=case.probe_failed,
        )
        for case in plan.cases
    }
    probe_set_content = {
        "schema_version": 1,
        "schema_id": "benchmark.probe_summary_set.v1",
        "benchmark_version": manifest.benchmark_version,
        "records": tuple(probe_by_case[case.case_id] for case in manifest.cases),
    }
    probe_set_draft = ProbeSummarySet.model_construct(
        _fields_set=set(probe_set_content),
        **probe_set_content,
        set_hash="0" * 64,
    )
    probe_set = ProbeSummarySet.model_validate(
        {
            **probe_set_content,
            "set_hash": model_content_hash(probe_set_draft, exclude={"set_hash"}),
        }
    )
    write_immutable_json(root / "probe_summaries.json", probe_set)
    predictions: list[DecisionPredictionRecord] = []
    recordings: list[RecordedGenerationResponse] = []
    for ordinal, key in enumerate(run_plan.planned_samples, start=1):
        case = next(item for item in manifest.cases if item.case_id == key.case_id)
        case_input = case_inputs[key.case_id]
        committed_at = plan.created_at_utc + timedelta(seconds=ordinal)
        generation_spec = GenerationRequestSpec(
            run_id=plan.run_id,
            sample_id=key.sample_id,
            system_prompt_version=plan.system_prompt_version,
            system_prompt=prompt.decode("utf-8"),
            system_prompt_sha256=plan.system_prompt_sha256,
            model_route=plan.model_route,
            sampling=_sampling_for_ordinal(plan.sampling, ordinal),
        )
        public_request = request_builder.build_public(case_id=case.case_id, spec=generation_spec)
        evidence_request = request_builder.build_evidence(
            case_id=case.case_id,
            spec=generation_spec,
        )
        public_record = create_recorded_response(
            request=public_request,
            original_source=OriginalGenerationSource.AUTHORED,
            final_response=case_input.public_response,
            captured_at_utc=committed_at,
        )
        evidence_record = create_recorded_response(
            request=evidence_request,
            original_source=OriginalGenerationSource.AUTHORED,
            final_response=case_input.evidence_response,
            captured_at_utc=committed_at,
        )
        replay_recorded_response(public_request, public_record)
        replay_recorded_response(evidence_request, evidence_record)
        recordings.extend((public_record, evidence_record))

        probe = probe_by_case[case.case_id]
        paired = runner.run_case(
            case_id=case.case_id,
            initial_tracker_state=load_initial_tracker_state(
                benchmark_root,
                manifest,
                case_id=case.case_id,
            ),
            public_evidence=case_input.public_evidence,
            probe_evidence=probe,
            unrelated_evidence=controls.unrelated(
                case_id=case.case_id,
                evidence_by_case=probe_by_case,
            ),
            corrupted_evidence=controls.corrupted(case_id=case.case_id, evidence=probe),
        )
        records = prediction_factory.build(
            paired,
            run_id=plan.run_id,
            sample_id=key.sample_id,
            model_route_id=route_id,
            public_record_hash=public_record.response_hash,
            committed_at_utc=committed_at,
        )
        existing = conditions.get(key)
        if existing is None:
            for record in records:
                conditions.append(record)
            conditions.seal(key)
        elif conditions.get_records(key) != records:
            raise OfflineDecisionError("Existing condition set differs from offline replay")
        predictions.extend(records)

    write_immutable_jsonl(decision_root / "recorded_responses.jsonl", tuple(recordings))
    global_store = FilesystemGlobalDecisionSealStore(
        decision_root,
        conditions,
        clock=lambda: global_sealed_at,
    )
    global_seal = global_store.publish(run_plan)
    publication = publish_condition_predictions(
        AtomicParquetDatasetStore(datasets_root),
        tuple(predictions),
        global_seal=global_seal,
        published_at_utc=global_sealed_at + timedelta(minutes=1),
    )
    summary = DecisionGenerationSummary(
        benchmark_version=manifest.benchmark_version,
        run_id=plan.run_id,
        case_count=len(manifest.cases),
        sample_count=len(run_plan.planned_samples),
        prediction_count=len(predictions),
        recorded_response_count=len(recordings),
        run_plan_hash=run_plan.plan_hash,
        global_seal_hash=global_seal.seal_hash,
        decision_publication_hash=publication.publication_hash,
    )
    write_immutable_json(root / "decision_summary.json", summary)
    return summary


def _validate_decision_scope(
    manifest: PublicBenchmarkManifest,
    plan: OfflineDecisionPlan,
) -> None:
    if any(case.split is not BenchmarkSplit.DEVELOPMENT for case in manifest.cases):
        raise OfflineDecisionError(
            "Offline generation accepts development cases only; held-out generation is disabled"
        )
    if tuple(case.case_id for case in plan.cases) != tuple(case.case_id for case in manifest.cases):
        raise OfflineDecisionError("Offline decision plan must cover every case in manifest order")


def load_initial_tracker_state(
    benchmark_root: Path,
    manifest: PublicBenchmarkManifest,
    *,
    case_id: str,
) -> TrackerState:
    case = next(item for item in manifest.cases if item.case_id == case_id)
    content = read_verified_public_file(
        benchmark_root,
        manifest,
        case.initial_tracker_ref,
        expected_class=PublicArtifactClass.INITIAL_TRACKER,
    )
    if file_sha256(content) != case.initial_tracker_hash:
        raise OfflineDecisionError("Case initial-tracker hash differs from public inventory")
    artifact = _InitialTrackerArtifact.model_validate_json(content)
    return TrackerState(
        concept=case.target_concept,
        mastery_probability=artifact.concept_mastery_probability,
        observations=artifact.observations,
    )


def read_verified_public_file(
    benchmark_root: Path,
    manifest: PublicBenchmarkManifest,
    reference: str,
    *,
    expected_class: PublicArtifactClass,
) -> bytes:
    entries = {entry.path: entry for entry in manifest.files}
    try:
        entry = entries[reference]
    except KeyError as error:
        raise OfflineDecisionError(f"Public artifact is absent: {reference}") from error
    if entry.artifact_class is not expected_class:
        raise OfflineDecisionError(f"Public artifact has wrong class: {reference}")
    root = benchmark_root.resolve()
    path = (root / reference).resolve()
    if not path.is_relative_to(root):
        raise OfflineDecisionError(f"Public artifact escapes benchmark root: {reference}")
    try:
        content = path.read_bytes()
    except OSError as error:
        raise OfflineDecisionError(f"Could not read public artifact: {reference}") from error
    if len(content) != entry.byte_size or file_sha256(content) != entry.sha256:
        raise OfflineDecisionError(f"Public artifact identity differs: {reference}")
    return content


def _sampling_for_ordinal(sampling: SamplingConfig, ordinal: int) -> SamplingConfig:
    return sampling.model_copy(update={"seed": (sampling.seed + ordinal - 1) % (2**32)})


def _sample_key(
    *,
    benchmark_version: str,
    run_id: str,
    case_id: str,
    sample_id: str,
    model_route_id: str,
) -> BenchmarkSampleKey:
    return BenchmarkSampleKey(
        benchmark_version=benchmark_version,
        run_id=run_id,
        case_id=case_id,
        sample_id=sample_id,
        model_route_id=model_route_id,
    )


def _require_utc(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must use UTC")
