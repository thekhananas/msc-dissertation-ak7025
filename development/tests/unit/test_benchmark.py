"""Benchmark authoring, isolation, and integrity tests."""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from shutil import copytree

import pytest
from pydantic import ValidationError

from socratic_tutor.benchmark.evaluator.inventory import InventoryValidationError
from socratic_tutor.benchmark.evaluator.loader import (
    load_and_verify_manifest,
    load_authored_manifest,
)
from socratic_tutor.benchmark.evaluator.models import (
    ArtifactClass,
    ManifestStatus,
    ReviewDecision,
)
from socratic_tutor.benchmark.evaluator.projection import (
    project_evaluator_manifest,
    project_public_manifest,
)
from socratic_tutor.benchmark.evaluator.validation import (
    BenchmarkValidationError,
    validate_manifest_structure,
)
from socratic_tutor.benchmark.generation import (
    EvidenceTaskPayload,
    GenerationRequestSpec,
    ModelRoute,
    PublicTaskPayload,
    SamplingConfig,
    StudentGenerationRequest,
)
from socratic_tutor.benchmark.hashing import file_sha256
from socratic_tutor.benchmark.public import (
    EXPECTED_CONDITIONS,
    BenchmarkCaseView,
    PublicArtifactReadError,
    PublicChannelRequestBuilder,
    find_public_payload_violations,
)
from socratic_tutor.benchmark.replay import (
    OriginalGenerationSource,
    ProviderResponseMetadata,
    RecordedResponseFileError,
    RecordedResponseGateway,
    RecordedResponseNotFoundError,
    create_recorded_response,
    replay_recorded_response,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = WORKSPACE_ROOT / "data" / "benchmarks" / "dev-v0"
MANIFEST_PATH = BENCHMARK_ROOT / "manifest.yaml"
SYSTEM_PROMPT_PATH = BENCHMARK_ROOT / "prompts" / "student-system-v1.md"
SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def generation_spec(*, system_prompt: str = SYSTEM_PROMPT) -> GenerationRequestSpec:
    return GenerationRequestSpec(
        run_id="dev-run-001",
        sample_id="sample-001",
        system_prompt_version="student-simulator-v1",
        system_prompt=system_prompt,
        system_prompt_sha256=file_sha256(system_prompt.encode("utf-8")),
        model_route=ModelRoute(provider="recorded_fixture", model="authored-deterministic"),
        sampling=SamplingConfig(temperature=0.0, max_output_tokens=512, seed=20260725),
    )


def test_development_manifest_passes_structure_and_inventory_checks() -> None:
    manifest = load_and_verify_manifest(MANIFEST_PATH)

    assert manifest.status is ManifestStatus.DRAFT
    assert len(manifest.cases) == 2
    assert manifest.conditions == EXPECTED_CONDITIONS
    assert all(review.decision.value == "approved" for review in manifest.reviews)


def test_generation_prompt_is_inventoried_and_content_addressed() -> None:
    manifest = load_and_verify_manifest(MANIFEST_PATH)
    prompt = manifest.generation

    assert prompt.system_prompt_version == "student-system-v1"
    assert prompt.system_prompt_ref == "prompts/student-system-v1.md"
    assert prompt.system_prompt_sha256 == file_sha256(SYSTEM_PROMPT.encode("utf-8"))

    changed_generation = prompt.model_copy(update={"system_prompt_sha256": "0" * 64})
    changed_manifest = manifest.model_copy(update={"generation": changed_generation})
    with pytest.raises(BenchmarkValidationError, match="generation-prompt-hash"):
        validate_manifest_structure(changed_manifest)


def test_review_scope_covers_every_condition_defining_artifact() -> None:
    manifest = load_and_verify_manifest(MANIFEST_PATH)
    reviews = {review.case_id: review for review in manifest.reviews}

    for case in manifest.cases:
        public = case.public
        criterion = case.criterion
        reviewed_paths = {item.path for item in reviews[public.case_id].reviewed_files}
        required_paths = {
            public.public_fixture_ref,
            public.evidence_probe_ref,
            public.evidence_test_ref,
            public.initial_tracker_ref,
            public.unrelated_control_ref,
            public.corruption_spec_ref,
            manifest.generation.system_prompt_ref,
            criterion.criterion_prompt_ref,
            criterion.test_bundle_ref,
            criterion.rubric_ref,
            criterion.label_rationale_ref,
            criterion.structural_difference_record_ref,
        }
        assert required_paths <= reviewed_paths


def test_public_projection_contains_no_evaluator_material() -> None:
    manifest = load_and_verify_manifest(MANIFEST_PATH)
    public = project_public_manifest(
        manifest,
        projected_at_utc=datetime(2026, 7, 25, tzinfo=UTC),
    )
    payload = public.model_dump(mode="json")
    serialized = public.model_dump_json()

    assert find_public_payload_violations(payload) == ()
    assert "criterion_probe_id" not in serialized
    assert "evidence_pattern" not in serialized
    assert "label_rationale" not in serialized
    assert "misconception_id" not in serialized
    assert "rubric" not in serialized
    assert "transfer_case_id" not in serialized
    assert all("evaluator/" not in entry.path for entry in public.files)


def test_evaluator_projection_contains_only_private_files() -> None:
    manifest = load_and_verify_manifest(MANIFEST_PATH)
    public = project_public_manifest(manifest)
    evaluator = project_evaluator_manifest(manifest, public)
    private_classes = {
        ArtifactClass.CRITERION_PROBE,
        ArtifactClass.CRITERION_TEST,
        ArtifactClass.CRITERION_RUBRIC,
        ArtifactClass.LABEL_RATIONALE,
        ArtifactClass.REVIEW,
        ArtifactClass.STRUCTURAL_DIFFERENCE,
    }

    assert len(evaluator.criteria) == len(manifest.cases)
    assert evaluator.public_projection_hash == public.projection_hash
    assert all(entry.artifact_class in private_classes for entry in evaluator.files)
    assert evaluator.criteria[0].evidence_pattern.value == "false_public_mastery"


def test_public_case_contract_rejects_evaluator_fields() -> None:
    manifest = load_authored_manifest(MANIFEST_PATH)
    payload = manifest.cases[0].public.model_dump(mode="json")
    payload["criterion_probe_id"] = "must-not-cross-boundary"

    with pytest.raises(ValidationError, match="extra_forbidden"):
        BenchmarkCaseView.model_validate(payload)


def test_projection_hashes_ignore_projection_time() -> None:
    manifest = load_and_verify_manifest(MANIFEST_PATH)
    first_time = datetime(2026, 7, 25, tzinfo=UTC)
    second_time = first_time + timedelta(days=1)
    first_public = project_public_manifest(manifest, projected_at_utc=first_time)
    second_public = project_public_manifest(manifest, projected_at_utc=second_time)
    first_evaluator = project_evaluator_manifest(
        manifest,
        first_public,
        projected_at_utc=first_time,
    )
    second_evaluator = project_evaluator_manifest(
        manifest,
        second_public,
        projected_at_utc=second_time,
    )

    assert first_public.projection_hash == second_public.projection_hash
    assert first_evaluator.projection_hash == second_evaluator.projection_hash


def test_inventory_detects_fixture_mutation(tmp_path: Path) -> None:
    copied_root = tmp_path / "dev-v0"
    copytree(BENCHMARK_ROOT, copied_root)
    mutated_file = copied_root / "public" / "aliasing-dialogue.md"
    mutated_file.write_text(
        mutated_file.read_text(encoding="utf-8") + "\nchanged\n",
        encoding="utf-8",
    )

    with pytest.raises(InventoryValidationError, match="aliasing-dialogue"):
        load_and_verify_manifest(copied_root / "manifest.yaml")


def test_inventory_detects_undeclared_file(tmp_path: Path) -> None:
    copied_root = tmp_path / "dev-v0"
    copytree(BENCHMARK_ROOT, copied_root)
    (copied_root / "undeclared-answer.txt").write_text("not inventoried", encoding="utf-8")

    with pytest.raises(InventoryValidationError, match=r"untracked:undeclared-answer\.txt"):
        load_and_verify_manifest(copied_root / "manifest.yaml")


def test_case_hash_detects_public_case_mutation() -> None:
    manifest = load_authored_manifest(MANIFEST_PATH)
    first = manifest.cases[0]
    changed_public = first.public.model_copy(update={"task_family": "changed-family"})
    changed_case = first.model_copy(update={"public": changed_public})
    changed_manifest = manifest.model_copy(update={"cases": (changed_case, *manifest.cases[1:])})

    with pytest.raises(BenchmarkValidationError, match="case-hash"):
        validate_manifest_structure(changed_manifest)


def test_criterion_digest_must_match_inventory() -> None:
    manifest = load_authored_manifest(MANIFEST_PATH)
    first = manifest.cases[0]
    changed_criterion = first.criterion.model_copy(update={"test_bundle_sha256": "0" * 64})
    changed_case = first.model_copy(update={"criterion": changed_criterion})
    changed_manifest = manifest.model_copy(update={"cases": (changed_case, *manifest.cases[1:])})

    with pytest.raises(BenchmarkValidationError, match="criterion-test-hash"):
        validate_manifest_structure(changed_manifest)


def test_two_case_development_fixture_cannot_pass_as_frozen_benchmark() -> None:
    manifest = load_authored_manifest(MANIFEST_PATH)
    frozen = manifest.model_copy(
        update={
            "status": ManifestStatus.FROZEN,
            "frozen_at_utc": datetime(2026, 7, 25, tzinfo=UTC),
            "manifest_hash": "0" * 64,
            "reviews": tuple(
                review.model_copy(update={"decision": ReviewDecision.PENDING})
                for review in manifest.reviews
            ),
        }
    )

    with pytest.raises(BenchmarkValidationError) as error:
        validate_manifest_structure(frozen)

    assert "frozen-case-count" in error.value.violations
    assert any(item.startswith("frozen-review:") for item in error.value.violations)


def test_public_schema_has_no_criterion_definition() -> None:
    serialized_schema = str(BenchmarkCaseView.model_json_schema()).casefold()

    assert "criterion" not in serialized_schema
    assert "evidence_pattern" not in serialized_schema
    assert "misconception" not in serialized_schema
    assert "rubric" not in serialized_schema
    assert "transfer_case" not in serialized_schema


def test_channel_request_builders_isolate_context_and_cache_identity() -> None:
    authored = load_and_verify_manifest(MANIFEST_PATH)
    public_manifest = project_public_manifest(authored)
    public_builder = PublicChannelRequestBuilder(BENCHMARK_ROOT, public_manifest)
    spec = generation_spec()

    public = public_builder.build_public(case_id="dev-aliasing-001", spec=spec)
    evidence = public_builder.build_evidence(case_id="dev-aliasing-001", spec=spec)

    assert isinstance(public.task_payload, PublicTaskPayload)
    assert isinstance(evidence.task_payload, EvidenceTaskPayload)
    assert set(public.task_payload.model_dump()) == {
        "channel",
        "benchmark_version",
        "case_id",
        "task_family",
        "target_concept",
        "public_interaction",
    }
    assert set(evidence.task_payload.model_dump()) == {
        "channel",
        "benchmark_version",
        "case_id",
        "task_family",
        "target_concept",
        "evidence_probe",
    }
    assert "Evidence probe" not in public.task_payload.public_interaction
    assert "Public interaction" not in evidence.task_payload.evidence_probe
    assert public.cache_namespace != evidence.cache_namespace
    assert public.cache_key != evidence.cache_key
    assert public.request_hash != evidence.request_hash


def test_generation_request_hash_is_stable_and_validated() -> None:
    authored = load_and_verify_manifest(MANIFEST_PATH)
    public_manifest = project_public_manifest(authored)
    builder = PublicChannelRequestBuilder(BENCHMARK_ROOT, public_manifest)
    first = builder.build_public(case_id="dev-none-falsy-001", spec=generation_spec())
    second = builder.build_public(case_id="dev-none-falsy-001", spec=generation_spec())

    assert first == second
    assert first.request_hash == second.request_hash

    changed_prompt = builder.build_public(
        case_id="dev-none-falsy-001",
        spec=generation_spec(system_prompt=f"{SYSTEM_PROMPT} Do not guess."),
    )
    assert changed_prompt.system_prompt_version == first.system_prompt_version
    assert changed_prompt.request_hash != first.request_hash

    tampered_hash = first.model_dump(mode="json")
    tampered_hash["request_hash"] = "0" * 64
    with pytest.raises(ValidationError, match="hash does not match"):
        StudentGenerationRequest.model_validate(tampered_hash)

    wrong_namespace = first.model_dump(mode="json")
    wrong_namespace["cache_namespace"] = "benchmark-student-generation/evidence/v1"
    with pytest.raises(ValidationError, match="namespace"):
        StudentGenerationRequest.model_validate(wrong_namespace)

    wrong_prompt_hash = first.model_dump(mode="json")
    wrong_prompt_hash["system_prompt_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="System prompt hash"):
        StudentGenerationRequest.model_validate(wrong_prompt_hash)


def test_public_request_rejects_evaluator_field_injection() -> None:
    authored = load_and_verify_manifest(MANIFEST_PATH)
    public_manifest = project_public_manifest(authored)
    request = PublicChannelRequestBuilder(BENCHMARK_ROOT, public_manifest).build_public(
        case_id="dev-aliasing-001",
        spec=generation_spec(),
    )
    payload = request.model_dump(mode="json")
    payload["task_payload"]["criterion_probe"] = "hidden outcome"

    with pytest.raises(ValidationError, match="extra_forbidden"):
        StudentGenerationRequest.model_validate(payload)


def test_public_request_builder_rechecks_file_digest(tmp_path: Path) -> None:
    copied_root = tmp_path / "dev-v0"
    copytree(BENCHMARK_ROOT, copied_root)
    authored = load_and_verify_manifest(MANIFEST_PATH)
    public_manifest = project_public_manifest(authored)
    changed_fixture = copied_root / "public" / "aliasing-dialogue.md"
    changed_fixture.write_text("changed after projection", encoding="utf-8")
    builder = PublicChannelRequestBuilder(copied_root, public_manifest)

    with pytest.raises(PublicArtifactReadError, match="digest or size mismatch"):
        builder.build_public(case_id="dev-aliasing-001", spec=generation_spec())


def test_recorded_response_gateway_replays_exact_request_without_network(tmp_path: Path) -> None:
    authored = load_and_verify_manifest(MANIFEST_PATH)
    public_manifest = project_public_manifest(authored)
    request = PublicChannelRequestBuilder(BENCHMARK_ROOT, public_manifest).build_public(
        case_id="dev-aliasing-001",
        spec=generation_spec(),
    )
    record = create_recorded_response(
        request=request,
        original_source=OriginalGenerationSource.AUTHORED,
        final_response="The assignment makes both names refer to one list.",
        provider_metadata=ProviderResponseMetadata(
            provider_id="recorded_fixture",
            model_id="authored-deterministic",
            resolved_provider_id="local_fixture",
            resolved_model_id="authored-deterministic-v1",
            output_tokens=11,
        ),
        captured_at_utc=datetime(2026, 7, 25, tzinfo=UTC),
    )
    response_file = tmp_path / "responses.jsonl"
    response_file.write_text(record.model_dump_json() + "\n", encoding="utf-8")
    gateway = RecordedResponseGateway.from_jsonl(
        response_file,
        allowed_channels=frozenset({request.channel}),
    )

    result = asyncio.run(gateway.generate(request))

    assert result.source == "recorded_replay"
    assert result.original_source is OriginalGenerationSource.AUTHORED
    assert result.request_hash == request.request_hash
    assert result.response_hash == record.response_hash
    assert result.final_response == record.final_response


def test_recorded_response_rejects_request_hash_mismatch() -> None:
    authored = load_and_verify_manifest(MANIFEST_PATH)
    public_manifest = project_public_manifest(authored)
    builder = PublicChannelRequestBuilder(BENCHMARK_ROOT, public_manifest)
    public_request = builder.build_public(case_id="dev-aliasing-001", spec=generation_spec())
    evidence_request = builder.build_evidence(case_id="dev-aliasing-001", spec=generation_spec())
    record = create_recorded_response(
        request=public_request,
        original_source=OriginalGenerationSource.AUTHORED,
        final_response="Recorded public response",
    )

    with pytest.raises(RecordedResponseNotFoundError, match="request hash"):
        replay_recorded_response(evidence_request, record)


def test_recorded_response_file_rejects_tampering(tmp_path: Path) -> None:
    authored = load_and_verify_manifest(MANIFEST_PATH)
    public_manifest = project_public_manifest(authored)
    request = PublicChannelRequestBuilder(BENCHMARK_ROOT, public_manifest).build_public(
        case_id="dev-none-falsy-001",
        spec=generation_spec(),
    )
    record = create_recorded_response(
        request=request,
        original_source=OriginalGenerationSource.CEREBRAS,
        final_response="Original response",
        provider_metadata=ProviderResponseMetadata(
            provider_id=request.model_route.provider,
            model_id=request.model_route.model,
        ),
    )
    payload = record.model_dump(mode="json")
    payload["final_response"] = "Changed after recording"
    response_file = tmp_path / "tampered.jsonl"
    response_file.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(RecordedResponseFileError, match="hash does not match"):
        RecordedResponseGateway.from_jsonl(
            response_file,
            allowed_channels=frozenset({request.channel}),
        )


def test_recorded_response_rejects_provider_or_model_mismatch() -> None:
    authored = load_and_verify_manifest(MANIFEST_PATH)
    public_manifest = project_public_manifest(authored)
    request = PublicChannelRequestBuilder(BENCHMARK_ROOT, public_manifest).build_public(
        case_id="dev-aliasing-001",
        spec=generation_spec(),
    )

    with pytest.raises(ValidationError, match="provider does not match"):
        create_recorded_response(
            request=request,
            original_source=OriginalGenerationSource.CEREBRAS,
            final_response="Response from the wrong route",
            provider_metadata=ProviderResponseMetadata(
                provider_id="different-provider",
                model_id=request.model_route.model,
            ),
        )

    with pytest.raises(ValidationError, match="model does not match"):
        create_recorded_response(
            request=request,
            original_source=OriginalGenerationSource.CEREBRAS,
            final_response="Response from the wrong model",
            provider_metadata=ProviderResponseMetadata(
                provider_id=request.model_route.provider,
                model_id="different-model",
            ),
        )
