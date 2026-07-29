"""Typed direct Cerebras adapter for development-only provider qualification."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from time import perf_counter
from typing import Protocol, cast

import httpx
from pydantic import Field, SecretStr, model_validator

from socratic_tutor.benchmark.generation import (
    EvidenceTaskPayload,
    PublicTaskPayload,
    StudentGenerationRequest,
)
from socratic_tutor.benchmark.hashing import model_content_hash
from socratic_tutor.benchmark.replay import (
    OriginalGenerationSource,
    ProviderResponseMetadata,
    RecordedGenerationResponse,
    create_recorded_response,
)
from socratic_tutor.contracts import ContractModel


class CerebrasGatewayError(RuntimeError):
    """A direct Cerebras generation cannot produce a safe recorded response."""


class CerebrasRequestError(CerebrasGatewayError):
    """Cerebras rejected a request that should not be retried."""


class CerebrasUnavailableError(CerebrasGatewayError):
    """A retryable or transport failure exhausted the configured attempts."""


class CerebrasGatewayConfig(ContractModel):
    """Explicit direct-provider controls for a pinned qualification route."""

    api_key: SecretStr
    api_base_url: str = "https://api.cerebras.ai/v1"
    timeout_seconds: float = Field(default=60.0, gt=0.0, le=120.0)
    max_attempts: int = Field(default=2, ge=1, le=3)
    retry_delay_seconds: float = Field(default=1.0, ge=0.0, le=10.0)


class CerebrasTransportResponse(ContractModel):
    """Minimal HTTP response supplied to the provider-specific parser."""

    status_code: int = Field(ge=100, le=599)
    body: dict[str, object]
    request_id: str | None = None


class CerebrasTransport(Protocol):
    """Small seam that keeps qualification tests network-free."""

    async def post_chat_completion(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, object],
        timeout_seconds: float,
    ) -> CerebrasTransportResponse: ...


class HttpxCerebrasTransport:
    """`httpx` transport implementation; client lifecycle remains caller-owned."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def post_chat_completion(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, object],
        timeout_seconds: float,
    ) -> CerebrasTransportResponse:
        try:
            response = await self._client.post(
                url,
                headers=dict(headers),
                json=dict(payload),
                timeout=timeout_seconds,
            )
            body = cast(object, response.json())
        except (httpx.HTTPError, ValueError) as error:
            raise CerebrasUnavailableError("Cerebras transport request failed") from error
        if not isinstance(body, Mapping):
            raise CerebrasUnavailableError("Cerebras response body must be a JSON object")
        return CerebrasTransportResponse(
            status_code=response.status_code,
            body=_mapping(cast(object, body)),
            request_id=response.headers.get("x-request-id"),
        )


class GatewayGenerationResult(ContractModel):
    """Recorded response plus retry facts needed for a qualification audit."""

    response: RecordedGenerationResponse
    attempt_count: int = Field(ge=1)
    retried_status_codes: tuple[int, ...]
    invocation_hash: str

    @model_validator(mode="after")
    def validate_invocation_hash(self) -> GatewayGenerationResult:
        expected = model_content_hash(self, exclude={"invocation_hash"})
        if self.invocation_hash != expected:
            raise ValueError("Gateway invocation hash does not match its content")
        return self


type Sleeper = Callable[[float], Awaitable[None]]
type Clock = Callable[[], datetime]


class CerebrasGateway:
    """Generate one direct response without retaining hidden reasoning content."""

    def __init__(
        self,
        *,
        config: CerebrasGatewayConfig,
        transport: CerebrasTransport,
        clock: Clock | None = None,
        sleeper: Sleeper = asyncio.sleep,
    ) -> None:
        self._config = config
        self._transport = transport
        self._clock = clock or (lambda: datetime.now(UTC))
        self._sleeper = sleeper

    async def generate(self, request: StudentGenerationRequest) -> GatewayGenerationResult:
        """Call a direct Cerebras route and create an immutable response record."""

        if request.model_route.provider != "cerebras":
            raise CerebrasGatewayError("Cerebras gateway requires a cerebras model route")
        payload = _request_payload(request)
        headers = {
            "Authorization": f"Bearer {self._config.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        retry_status_codes: list[int] = []
        for attempt in range(1, self._config.max_attempts + 1):
            started_at = perf_counter()
            try:
                provider_response = await self._transport.post_chat_completion(
                    url=f"{self._config.api_base_url.rstrip('/')}/chat/completions",
                    headers=headers,
                    payload=payload,
                    timeout_seconds=self._config.timeout_seconds,
                )
            except CerebrasUnavailableError:
                if attempt == self._config.max_attempts:
                    raise
                await self._sleeper(self._config.retry_delay_seconds)
                continue

            latency_ms = round((perf_counter() - started_at) * 1_000)
            if provider_response.status_code in {408, 409, 425, 429} or (
                500 <= provider_response.status_code <= 599
            ):
                retry_status_codes.append(provider_response.status_code)
                if attempt == self._config.max_attempts:
                    raise CerebrasUnavailableError(
                        "Cerebras retryable response exhausted configured attempts: "
                        f"{provider_response.status_code}"
                    )
                await self._sleeper(self._config.retry_delay_seconds)
                continue
            if not 200 <= provider_response.status_code <= 299:
                raise CerebrasRequestError(
                    f"Cerebras rejected the generation request: {provider_response.status_code}"
                )

            response = create_recorded_response(
                request=request,
                original_source=OriginalGenerationSource.CEREBRAS,
                final_response=_completion_text(provider_response.body),
                provider_metadata=_provider_metadata(
                    request=request,
                    response=provider_response,
                    latency_ms=latency_ms,
                ),
                captured_at_utc=_require_utc(self._clock()),
            )
            content = {
                "response": response,
                "attempt_count": attempt,
                "retried_status_codes": tuple(retry_status_codes),
            }
            draft = GatewayGenerationResult.model_construct(
                _fields_set=set(content), **content, invocation_hash="0" * 64
            )
            return GatewayGenerationResult.model_validate(
                {
                    **content,
                    "invocation_hash": model_content_hash(draft, exclude={"invocation_hash"}),
                }
            )
        raise AssertionError("Cerebras retry loop must return or raise")


def _request_payload(request: StudentGenerationRequest) -> dict[str, object]:
    """Translate an isolated benchmark request into Cerebras chat-completion shape."""

    return {
        "model": request.model_route.model,
        "messages": [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": _user_prompt(request)},
        ],
        "temperature": request.sampling.temperature,
        "max_completion_tokens": request.sampling.max_output_tokens,
        "seed": request.sampling.seed,
    }


def _user_prompt(request: StudentGenerationRequest) -> str:
    payload = request.task_payload
    if isinstance(payload, PublicTaskPayload):
        return payload.public_interaction
    if isinstance(payload, EvidenceTaskPayload):
        return payload.evidence_probe
    return payload.criterion_probe


def _completion_text(body: Mapping[str, object]) -> str:
    choice = _first_choice(body)
    message = _mapping(choice.get("message"))
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise CerebrasUnavailableError("Cerebras completion content must be non-empty text")
    return content


def _provider_metadata(
    *,
    request: StudentGenerationRequest,
    response: CerebrasTransportResponse,
    latency_ms: int,
) -> ProviderResponseMetadata:
    body = response.body
    usage = _mapping(body.get("usage"))
    return ProviderResponseMetadata(
        provider_request_id=response.request_id or _text(body.get("id")),
        generation_id=_text(body.get("id")),
        provider_id=request.model_route.provider,
        model_id=request.model_route.model,
        resolved_provider_id="cerebras",
        resolved_model_id=_text(body.get("model")),
        backend_fingerprint=_text(body.get("system_fingerprint")),
        finish_reason=_finish_reason(body),
        input_tokens=_integer(usage.get("prompt_tokens")),
        output_tokens=_integer(usage.get("completion_tokens")),
        latency_ms=latency_ms,
    )


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    mapping = cast(Mapping[object, object], value)
    return {key: item for key, item in mapping.items() if isinstance(key, str)}


def _first_choice(body: Mapping[str, object]) -> dict[str, object]:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise CerebrasUnavailableError("Cerebras response has no completion choice")
    choice = _mapping(cast(object, choices[0]))
    if not choice:
        raise CerebrasUnavailableError("Cerebras response has no completion choice")
    return choice


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _integer(value: object) -> int | None:
    return value if isinstance(value, int) and value >= 0 else None


def _finish_reason(body: Mapping[str, object]) -> str | None:
    try:
        return _text(_first_choice(body).get("finish_reason"))
    except CerebrasUnavailableError:
        return None


def _require_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise CerebrasGatewayError("Gateway clock must return a UTC-aware timestamp")
    return value
