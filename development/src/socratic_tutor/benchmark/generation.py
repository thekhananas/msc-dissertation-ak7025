"""Content-addressed request contracts shared by benchmark generation channels."""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, model_validator

from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import canonical_sha256
from socratic_tutor.contracts.models import ContractModel


class GenerationChannel(StrEnum):
    """Isolated contexts used to generate benchmark student responses."""

    PUBLIC = "public"
    EVIDENCE = "evidence"
    CRITERION = "criterion"


class ModelRoute(ContractModel):
    """Pinned provider and model identity for one generation request."""

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)


class SamplingConfig(ContractModel):
    """Sampling controls included in the request hash."""

    temperature: float = Field(ge=0.0, le=2.0)
    max_output_tokens: int = Field(ge=1)
    seed: int = Field(ge=0, le=2**32 - 1)


class GenerationRequestSpec(ContractModel):
    """Run-level values supplied identically to every channel builder."""

    run_id: str = Field(min_length=1)
    sample_id: str = Field(min_length=1)
    system_prompt_version: str = Field(min_length=1)
    model_route: ModelRoute
    sampling: SamplingConfig


class PublicTaskPayload(ContractModel):
    """Only the authored interaction visible before an active probe."""

    channel: Literal[GenerationChannel.PUBLIC] = GenerationChannel.PUBLIC
    benchmark_version: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    task_family: str = Field(min_length=1)
    target_concept: str = Field(min_length=1)
    public_interaction: str = Field(min_length=1)


class EvidenceTaskPayload(ContractModel):
    """Fresh evidence-probe context with no public-dialogue history."""

    channel: Literal[GenerationChannel.EVIDENCE] = GenerationChannel.EVIDENCE
    benchmark_version: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    task_family: str = Field(min_length=1)
    target_concept: str = Field(min_length=1)
    evidence_probe: str = Field(min_length=1)


class CriterionTaskPayload(ContractModel):
    """Evaluator-owned transfer probe with no decision-phase history."""

    channel: Literal[GenerationChannel.CRITERION] = GenerationChannel.CRITERION
    benchmark_version: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    criterion_probe_id: str = Field(min_length=1)
    criterion_probe: str = Field(min_length=1)


type GenerationTaskPayload = Annotated[
    PublicTaskPayload | EvidenceTaskPayload | CriterionTaskPayload,
    Field(discriminator="channel"),
]


def generation_cache_namespace(channel: GenerationChannel) -> str:
    """Return a cache namespace that cannot collide across channels."""

    return f"benchmark-student-generation/{channel.value}/v1"


class StudentGenerationRequest(ContractModel):
    """Exact provider request whose hash binds identity, content, and sampling."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.student_generation_request.v1"] = (
        "benchmark.student_generation_request.v1"
    )
    run_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    sample_id: str = Field(min_length=1)
    channel: GenerationChannel
    system_prompt_version: str = Field(min_length=1)
    task_payload: GenerationTaskPayload
    model_route: ModelRoute
    sampling: SamplingConfig
    cache_namespace: str = Field(min_length=1)
    request_hash: Sha256

    @model_validator(mode="after")
    def validate_channel_identity_and_hash(self) -> "StudentGenerationRequest":
        if self.task_payload.channel is not self.channel:
            raise ValueError("Request channel must match task payload channel")
        if self.task_payload.case_id != self.case_id:
            raise ValueError("Request case ID must match task payload case ID")
        expected_namespace = generation_cache_namespace(self.channel)
        if self.cache_namespace != expected_namespace:
            raise ValueError("Cache namespace must match the isolated generation channel")
        expected_hash = request_content_hash(self)
        if self.request_hash != expected_hash:
            raise ValueError("Generation request hash does not match request content")
        return self

    @property
    def cache_key(self) -> str:
        """Return the channel-qualified immutable replay/cache key."""

        return f"{self.cache_namespace}:{self.request_hash}"


def request_content_hash(request: StudentGenerationRequest) -> Sha256:
    """Hash every request field except the self-referential digest."""

    return canonical_sha256(request.model_dump(mode="json", exclude={"request_hash"}))


def create_student_generation_request(
    *,
    spec: GenerationRequestSpec,
    payload: GenerationTaskPayload,
) -> StudentGenerationRequest:
    """Construct and validate a generation request with its canonical digest."""

    content = {
        "schema_version": 1,
        "schema_id": "benchmark.student_generation_request.v1",
        "run_id": spec.run_id,
        "case_id": payload.case_id,
        "sample_id": spec.sample_id,
        "channel": payload.channel,
        "system_prompt_version": spec.system_prompt_version,
        "task_payload": payload,
        "model_route": spec.model_route,
        "sampling": spec.sampling,
        "cache_namespace": generation_cache_namespace(payload.channel),
    }
    return StudentGenerationRequest.model_validate(
        {**content, "request_hash": canonical_sha256(content)}
    )
