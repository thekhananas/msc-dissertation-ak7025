"""Strict, network-free replay of content-addressed benchmark responses."""

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal, cast

from pydantic import Field, ValidationError, model_validator

from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.generation import (
    GenerationChannel,
    StudentGenerationRequest,
)
from socratic_tutor.benchmark.hashing import canonical_sha256
from socratic_tutor.contracts.models import ContractModel


class OriginalGenerationSource(StrEnum):
    """Source that produced the response before it was recorded."""

    AUTHORED = "authored"
    OPENROUTER = "openrouter"


class ProviderResponseMetadata(ContractModel):
    """Normalized operational metadata; hidden model reasoning is never accepted."""

    provider_request_id: str | None = None
    generation_id: str | None = None
    provider_id: str | None = None
    model_id: str | None = None
    resolved_provider_id: str | None = None
    resolved_model_id: str | None = None
    finish_reason: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cost_usd_micros: int | None = Field(default=None, ge=0)
    latency_ms: int | None = Field(default=None, ge=0)


class RecordedGenerationResponse(ContractModel):
    """Immutable response bound to the exact request it answers."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.recorded_generation_response.v1"] = (
        "benchmark.recorded_generation_response.v1"
    )
    request: StudentGenerationRequest
    original_source: OriginalGenerationSource
    final_response: str = Field(min_length=1)
    provider_metadata: ProviderResponseMetadata
    captured_at_utc: datetime
    response_hash: Sha256

    @model_validator(mode="after")
    def validate_response_hash(self) -> "RecordedGenerationResponse":
        if self.captured_at_utc.tzinfo is None:
            raise ValueError("Recorded response timestamp must include a timezone")
        route = self.request.model_route
        metadata = self.provider_metadata
        if metadata.provider_id is not None and metadata.provider_id != route.provider:
            raise ValueError("Recorded provider does not match requested provider")
        if metadata.model_id is not None and metadata.model_id != route.model:
            raise ValueError("Recorded model does not match requested model")
        if self.original_source is OriginalGenerationSource.OPENROUTER and (
            metadata.provider_id is None or metadata.model_id is None
        ):
            raise ValueError("OpenRouter recordings require requested provider and model metadata")
        expected_hash = recorded_response_content_hash(self)
        if self.response_hash != expected_hash:
            raise ValueError("Recorded response hash does not match response content")
        return self


class ReplayedGenerationResult(ContractModel):
    """Gateway output marking that no provider call occurred."""

    request_hash: Sha256
    response_hash: Sha256
    channel: GenerationChannel
    source: Literal["recorded_replay"] = "recorded_replay"
    original_source: OriginalGenerationSource
    final_response: str = Field(min_length=1)
    provider_metadata: ProviderResponseMetadata


class RecordedResponseError(ValueError):
    """Base class for fail-closed replay errors."""


class RecordedResponseNotFoundError(RecordedResponseError):
    """No exact response exists for a validated request."""


class RecordedResponseFileError(RecordedResponseError):
    """A response file is malformed, duplicated, or crosses a channel boundary."""


def _response_hash_payload(
    *,
    request: StudentGenerationRequest,
    original_source: OriginalGenerationSource,
    final_response: str,
    provider_metadata: ProviderResponseMetadata,
) -> dict[str, object]:
    return {
        "request_hash": request.request_hash,
        "channel": request.channel,
        "original_source": original_source,
        "final_response": final_response,
        "provider_metadata": provider_metadata,
    }


def recorded_response_content_hash(response: RecordedGenerationResponse) -> Sha256:
    """Hash semantic response content while excluding capture time and self hash."""

    return canonical_sha256(
        _response_hash_payload(
            request=response.request,
            original_source=response.original_source,
            final_response=response.final_response,
            provider_metadata=response.provider_metadata,
        )
    )


def create_recorded_response(
    *,
    request: StudentGenerationRequest,
    original_source: OriginalGenerationSource,
    final_response: str,
    provider_metadata: ProviderResponseMetadata | None = None,
    captured_at_utc: datetime | None = None,
) -> RecordedGenerationResponse:
    """Create one validated record suitable for JSONL persistence."""

    metadata = provider_metadata or ProviderResponseMetadata()
    response_hash = canonical_sha256(
        _response_hash_payload(
            request=request,
            original_source=original_source,
            final_response=final_response,
            provider_metadata=metadata,
        )
    )
    return RecordedGenerationResponse(
        request=request,
        original_source=original_source,
        final_response=final_response,
        provider_metadata=metadata,
        captured_at_utc=captured_at_utc or datetime.now(UTC),
        response_hash=response_hash,
    )


def replay_recorded_response(
    request: StudentGenerationRequest,
    record: RecordedGenerationResponse,
) -> ReplayedGenerationResult:
    """Replay only when the complete request and request hash match exactly."""

    if record.request.request_hash != request.request_hash:
        raise RecordedResponseNotFoundError("Recorded response request hash does not match request")
    if record.request != request:
        raise RecordedResponseError("Recorded response request content does not match request")
    return ReplayedGenerationResult(
        request_hash=request.request_hash,
        response_hash=record.response_hash,
        channel=request.channel,
        original_source=record.original_source,
        final_response=record.final_response,
        provider_metadata=record.provider_metadata,
    )


class RecordedResponseGateway:
    """In-memory exact-match gateway with no network implementation."""

    def __init__(
        self,
        records: Iterable[RecordedGenerationResponse],
        *,
        allowed_channels: frozenset[GenerationChannel],
    ) -> None:
        if not allowed_channels:
            raise RecordedResponseFileError("At least one replay channel must be allowed")
        self._allowed_channels = allowed_channels
        self._records: dict[str, RecordedGenerationResponse] = {}
        for record in records:
            channel = record.request.channel
            if channel not in allowed_channels:
                raise RecordedResponseFileError(
                    f"Recorded {channel.value} response is not allowed in this gateway"
                )
            cache_key = record.request.cache_key
            if cache_key in self._records:
                raise RecordedResponseFileError(
                    f"Duplicate recorded response for cache key: {cache_key}"
                )
            self._records[cache_key] = record

    @classmethod
    def from_jsonl(
        cls,
        path: Path,
        *,
        allowed_channels: frozenset[GenerationChannel],
    ) -> "RecordedResponseGateway":
        """Load strict JSONL records and report the exact malformed line."""

        records: list[RecordedGenerationResponse] = []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            raise RecordedResponseFileError(f"Could not read response file: {path}") from error
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                record = RecordedGenerationResponse.model_validate(cast(object, raw))
            except (json.JSONDecodeError, ValidationError) as error:
                raise RecordedResponseFileError(
                    f"Invalid recorded response at line {line_number}: {error}"
                ) from error
            records.append(record)
        return cls(records, allowed_channels=allowed_channels)

    async def generate(self, request: StudentGenerationRequest) -> ReplayedGenerationResult:
        """Return a recording for an exact request without performing I/O."""

        if request.channel not in self._allowed_channels:
            raise RecordedResponseError(
                f"Request channel {request.channel.value} is not allowed in this gateway"
            )
        try:
            record = self._records[request.cache_key]
        except KeyError as error:
            raise RecordedResponseNotFoundError(
                f"No recorded response for request hash: {request.request_hash}"
            ) from error
        return replay_recorded_response(request, record)
