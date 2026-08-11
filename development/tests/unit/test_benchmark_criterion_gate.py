"""Criterion capability, private-loader, and gated request tests."""

from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

from socratic_tutor.benchmark.evaluator import (
    CriterionArtifactReadError,
    CriterionCapabilityError,
    CriterionChannelRequestBuilder,
    CriterionGate,
    CriterionIneligibleError,
    CriterionNotSealedError,
    EvaluatorManifestAccessError,
)
from socratic_tutor.benchmark.evaluator.loader import load_and_verify_manifest
from socratic_tutor.benchmark.evaluator.models import EvaluatorBenchmarkManifest
from socratic_tutor.benchmark.evaluator.projection import (
    project_evaluator_manifest,
    project_public_manifest,
)
from socratic_tutor.benchmark.generation import (
    CriterionTaskPayload,
    GenerationChannel,
    GenerationRequestSpec,
    ModelRoute,
    SamplingConfig,
)
from socratic_tutor.benchmark.hashing import file_sha256, model_content_hash
from socratic_tutor.benchmark.public import (
    BenchmarkSampleKey,
    FilesystemConditionCommitStore,
    FilesystemGlobalDecisionSealStore,
    SampleDecisionFailure,
    SampleDecisionReason,
    SampleDecisionStatus,
)
from socratic_tutor.benchmark.replay import (
    OriginalGenerationSource,
    RecordedResponseFileError,
    RecordedResponseGateway,
    create_recorded_response,
)
from tests.unit.test_benchmark_global_seal import (
    MODEL_ROUTE_ID,
    RUN_ID,
    benchmark_key,
    decision_plan,
    prediction_records_for_sample,
    seal_sample_records,
)
from tests.unit.test_benchmark_predictions import COMMITTED_AT

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = WORKSPACE_ROOT / "data" / "benchmarks" / "dev-v0"
MANIFEST_PATH = BENCHMARK_ROOT / "manifest.yaml"
SYSTEM_PROMPT_PATH = BENCHMARK_ROOT / "prompts" / "student-system-v1.md"
SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def _generation_spec() -> GenerationRequestSpec:
    provider, model = MODEL_ROUTE_ID.split("/", maxsplit=1)
    return GenerationRequestSpec(
        run_id=RUN_ID,
        sample_id="sample-001",
        system_prompt_version="student-simulator-v1",
        system_prompt=SYSTEM_PROMPT,
        system_prompt_sha256=file_sha256(SYSTEM_PROMPT.encode("utf-8")),
        model_route=ModelRoute(provider=provider, model=model),
        sampling=SamplingConfig(temperature=0.0, max_output_tokens=512, seed=20260811),
    )


def _evaluator_manifest() -> EvaluatorBenchmarkManifest:
    authored = load_and_verify_manifest(MANIFEST_PATH)
    public = project_public_manifest(authored)
    return project_evaluator_manifest(authored, public)


def _public_projection_hash() -> str:
    authored = load_and_verify_manifest(MANIFEST_PATH)
    return project_public_manifest(authored).projection_hash


def _complete_gate(
    tmp_path: Path,
    *,
    criterion_records: tuple[BenchmarkSampleKey, ...] | None = None,
) -> tuple[CriterionGate, FilesystemConditionCommitStore]:
    condition_store = FilesystemConditionCommitStore(tmp_path)
    condition_sealed_at = seal_sample_records(condition_store, "sample-001")
    plan = decision_plan(
        "sample-001",
        public_manifest_hash=_public_projection_hash(),
    )
    global_store = FilesystemGlobalDecisionSealStore(
        tmp_path,
        condition_store,
        clock=lambda: condition_sealed_at + timedelta(seconds=1),
    )
    global_store.publish(plan)
    existing = criterion_records if criterion_records is not None else ()
    return (
        CriterionGate(
            global_store,
            condition_store,
            plan,
            criterion_record_exists=lambda key: key in existing,
            nonce_factory=lambda: b"criterion-capability-nonce-001",
        ),
        condition_store,
    )


def test_private_loader_runs_only_after_complete_capability_is_consumed(
    tmp_path: Path,
) -> None:
    gate, _ = _complete_gate(tmp_path)
    loader_calls = 0

    def counted_loader() -> EvaluatorBenchmarkManifest:
        nonlocal loader_calls
        loader_calls += 1
        return _evaluator_manifest()

    capability = gate.issue(benchmark_key("sample-001"))
    assert loader_calls == 0
    assert capability.sample_key == benchmark_key("sample-001")
    assert len(capability.nonce) >= 16

    access = gate.open(capability, manifest_loader=counted_loader)
    assert loader_calls == 1
    assert access.global_seal_hash == capability.global_seal_hash
    assert access.condition_set_hash == capability.condition_set_hash

    request = CriterionChannelRequestBuilder(BENCHMARK_ROOT, access).build_criterion(
        case_id="dev-aliasing-001",
        spec=_generation_spec(),
    )
    assert isinstance(request.task_payload, CriterionTaskPayload)
    assert request.channel is GenerationChannel.CRITERION
    assert "Public interaction" not in request.task_payload.criterion_probe
    assert "Evidence probe" not in request.task_payload.criterion_probe


def test_gate_rejects_access_before_global_sealing(tmp_path: Path) -> None:
    condition_store = FilesystemConditionCommitStore(tmp_path)
    global_store = FilesystemGlobalDecisionSealStore(tmp_path, condition_store)
    plan = decision_plan(
        "sample-001",
        public_manifest_hash=_public_projection_hash(),
    )
    gate = CriterionGate(
        global_store,
        condition_store,
        plan,
        criterion_record_exists=lambda _key: False,
    )

    with pytest.raises(CriterionNotSealedError, match="not sealed"):
        gate.issue(benchmark_key("sample-001"))


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        (
            SampleDecisionStatus.PRECRITERION_MISSING,
            SampleDecisionReason.GENERATION_MISSING,
        ),
        (SampleDecisionStatus.INVALID, SampleDecisionReason.DECISION_FAILURE),
    ],
)
def test_only_complete_samples_receive_capabilities(
    tmp_path: Path,
    status: SampleDecisionStatus,
    reason: SampleDecisionReason,
) -> None:
    condition_store = FilesystemConditionCommitStore(tmp_path)
    plan = decision_plan(
        "sample-001",
        public_manifest_hash=_public_projection_hash(),
    )
    global_store = FilesystemGlobalDecisionSealStore(
        tmp_path,
        condition_store,
        clock=lambda: COMMITTED_AT + timedelta(days=1),
    )
    global_store.publish(
        plan,
        failures=(
            SampleDecisionFailure(
                key=benchmark_key("sample-001"),
                status=status,
                reason_code=reason,
            ),
        ),
    )
    gate = CriterionGate(
        global_store,
        condition_store,
        plan,
        criterion_record_exists=lambda _key: False,
    )

    with pytest.raises(CriterionIneligibleError, match="not eligible"):
        gate.issue(benchmark_key("sample-001"))


def test_capability_is_sample_bound_and_consumed_once(tmp_path: Path) -> None:
    gate, _ = _complete_gate(tmp_path)
    capability = gate.issue(benchmark_key("sample-001"))
    altered = replace(capability, nonce=b"altered-capability-nonce")

    with pytest.raises(CriterionCapabilityError, match="altered"):
        gate.open(altered, manifest_loader=_evaluator_manifest)

    access = gate.open(capability, manifest_loader=_evaluator_manifest)
    with pytest.raises(CriterionCapabilityError, match=r"unknown|altered"):
        gate.open(capability, manifest_loader=_evaluator_manifest)
    with pytest.raises(CriterionCapabilityError, match="already issued"):
        gate.issue(benchmark_key("sample-001"))

    builder = CriterionChannelRequestBuilder(BENCHMARK_ROOT, access)
    with pytest.raises(CriterionArtifactReadError, match="another case"):
        builder.build_criterion(case_id="dev-none-falsy-001", spec=_generation_spec())
    wrong_sample = _generation_spec().model_copy(update={"sample_id": "sample-002"})
    with pytest.raises(CriterionArtifactReadError, match="identity"):
        builder.build_criterion(case_id="dev-aliasing-001", spec=wrong_sample)


def test_gate_rejects_evaluator_projection_from_another_public_manifest(
    tmp_path: Path,
) -> None:
    gate, _ = _complete_gate(tmp_path)
    capability = gate.issue(benchmark_key("sample-001"))
    manifest = _evaluator_manifest().model_copy(update={"public_projection_hash": "0" * 64})
    mismatched = manifest.model_copy(
        update={
            "projection_hash": model_content_hash(
                manifest,
                exclude={"projection_hash", "projected_at_utc"},
            )
        }
    )

    with pytest.raises(EvaluatorManifestAccessError, match="public projection"):
        gate.open(capability, manifest_loader=lambda: mismatched)


def test_existing_criterion_record_blocks_capability(tmp_path: Path) -> None:
    existing = (benchmark_key("sample-001"),)
    gate, _ = _complete_gate(tmp_path, criterion_records=existing)

    with pytest.raises(CriterionIneligibleError, match="already exists"):
        gate.issue(benchmark_key("sample-001"))


def test_recorded_gateway_rejects_gated_criterion_file(tmp_path: Path) -> None:
    gate, _ = _complete_gate(tmp_path)
    access = gate.open(
        gate.issue(benchmark_key("sample-001")),
        manifest_loader=_evaluator_manifest,
    )
    request = CriterionChannelRequestBuilder(BENCHMARK_ROOT, access).build_criterion(
        case_id="dev-aliasing-001",
        spec=_generation_spec(),
    )
    record = create_recorded_response(
        request=request,
        original_source=OriginalGenerationSource.AUTHORED,
        final_response="Evaluator-only response",
    )
    response_file = tmp_path / "criterion.jsonl"
    response_file.write_text(record.model_dump_json() + "\n", encoding="utf-8")

    with pytest.raises(RecordedResponseFileError, match="not allowed"):
        RecordedResponseGateway.from_jsonl(
            response_file,
            allowed_channels=frozenset({GenerationChannel.PUBLIC, GenerationChannel.EVIDENCE}),
        )


def test_gate_rechecks_global_seal_before_opening_private_loader(tmp_path: Path) -> None:
    gate, condition_store = _complete_gate(tmp_path)
    capability = gate.issue(benchmark_key("sample-001"))
    loader_calls = 0

    def counted_loader() -> EvaluatorBenchmarkManifest:
        nonlocal loader_calls
        loader_calls += 1
        return _evaluator_manifest()

    records = prediction_records_for_sample("sample-002")
    for record in records:
        condition_store.append(record)
    condition_store.seal(benchmark_key("sample-002"))

    with pytest.raises(CriterionNotSealedError, match="failed verification"):
        gate.open(capability, manifest_loader=counted_loader)
    assert loader_calls == 0
