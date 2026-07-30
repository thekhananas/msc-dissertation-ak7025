"""Network-free contract tests for the direct Cerebras qualification gateway."""

import asyncio
from collections.abc import Mapping
from datetime import UTC, datetime

import pytest
from pydantic import SecretStr

from socratic_tutor.benchmark.cerebras import (
    CerebrasGateway,
    CerebrasGatewayConfig,
    CerebrasGatewayError,
    CerebrasRequestError,
    CerebrasResponseFormatError,
    CerebrasTransportResponse,
)
from socratic_tutor.benchmark.generation import (
    GenerationRequestSpec,
    ModelRoute,
    PublicTaskPayload,
    SamplingConfig,
    StudentGenerationRequest,
    create_student_generation_request,
)
from socratic_tutor.benchmark.hashing import file_sha256
from socratic_tutor.benchmark.replay import OriginalGenerationSource, replay_recorded_response


class FakeTransport:
    def __init__(self, responses: list[CerebrasTransportResponse]) -> None:
        self._responses = iter(responses)
        self.calls: list[dict[str, object]] = []

    async def post_chat_completion(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, object],
        timeout_seconds: float,
    ) -> CerebrasTransportResponse:
        self.calls.append(
            {
                "url": url,
                "headers": headers,
                "payload": payload,
                "timeout_seconds": timeout_seconds,
            }
        )
        return next(self._responses)


def _request(provider: str = "cerebras") -> StudentGenerationRequest:
    system_prompt = "Return only the requested Python answer."
    return create_student_generation_request(
        spec=GenerationRequestSpec(
            run_id="qualification-test",
            sample_id="sample-001",
            system_prompt_version="student-system-v1",
            system_prompt=system_prompt,
            system_prompt_sha256=file_sha256(system_prompt.encode("utf-8")),
            model_route=ModelRoute(provider=provider, model="gpt-oss-120b"),
            sampling=SamplingConfig(temperature=0.0, max_output_tokens=64, seed=7),
        ),
        payload=PublicTaskPayload(
            benchmark_version="dev-v0",
            case_id="case-001",
            task_family="development",
            target_concept="assignment",
            public_interaction="What does this Python code print?",
        ),
    )


def _success() -> CerebrasTransportResponse:
    return CerebrasTransportResponse(
        status_code=200,
        request_id="request-123",
        body={
            "id": "generation-123",
            "model": "gpt-oss-120b",
            "system_fingerprint": "fp-test-001",
            "choices": [{"message": {"content": "It prints 3."}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4},
        },
    )


def _gateway(transport: FakeTransport) -> CerebrasGateway:
    async def no_sleep(_: float) -> None:
        return None

    return CerebrasGateway(
        config=CerebrasGatewayConfig(
            api_key=SecretStr("test-key"),
            timeout_seconds=12.0,
            max_attempts=2,
            retry_delay_seconds=0.0,
        ),
        transport=transport,
        clock=lambda: datetime(2026, 8, 22, 20, 0, tzinfo=UTC),
        sleeper=no_sleep,
    )


def test_gateway_records_a_direct_response_and_replayable_backend_fingerprint() -> None:
    transport = FakeTransport([_success()])
    request = _request()
    result = asyncio.run(_gateway(transport).generate(request))

    assert result.response.original_source is OriginalGenerationSource.CEREBRAS
    assert result.response.provider_metadata.resolved_provider_id == "cerebras"
    assert result.response.provider_metadata.backend_fingerprint == "fp-test-001"
    assert replay_recorded_response(request, result.response).final_response == "It prints 3."

    call = transport.calls[0]
    assert call["url"] == "https://api.cerebras.ai/v1/chat/completions"
    assert call["headers"] == {
        "Authorization": "Bearer test-key",
        "Content-Type": "application/json",
    }
    payload = call["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == "gpt-oss-120b"
    assert payload["temperature"] == 0.0
    assert payload["max_completion_tokens"] == 64
    assert payload["seed"] == 7
    assert payload["reasoning_effort"] == "medium"


def test_gateway_retries_only_retryable_provider_responses() -> None:
    delays: list[float] = []

    async def collect_delay(delay: float) -> None:
        delays.append(delay)

    transport = FakeTransport(
        [
            CerebrasTransportResponse(
                status_code=429,
                body={"error": "rate limited"},
                retry_after_seconds=3.0,
            ),
            _success(),
        ]
    )

    gateway = CerebrasGateway(
        config=CerebrasGatewayConfig(
            api_key=SecretStr("test-key"),
            timeout_seconds=12.0,
            max_attempts=2,
            retry_delay_seconds=1.0,
        ),
        transport=transport,
        sleeper=collect_delay,
    )
    result = asyncio.run(gateway.generate(_request()))

    assert result.attempt_count == 2
    assert result.retried_status_codes == (429,)
    assert len(transport.calls) == 2
    assert delays == [3.0]


def test_gateway_rejects_wrong_route_nonretryable_and_malformed_responses() -> None:
    with pytest.raises(CerebrasGatewayError, match="cerebras model route"):
        asyncio.run(_gateway(FakeTransport([_success()])).generate(_request("other-provider")))
    with pytest.raises(CerebrasRequestError, match="provider_code=invalid_request") as error:
        asyncio.run(
            _gateway(
                FakeTransport(
                    [
                        CerebrasTransportResponse(
                            status_code=400,
                            request_id="request-invalid",
                            body={
                                "error": {
                                    "code": "invalid_request",
                                    "message": "The requested parameter is invalid.",
                                }
                            },
                        )
                    ]
                )
            ).generate(_request())
        )
    assert error.value.status_code == 400
    assert error.value.request_id == "request-invalid"
    assert error.value.provider_error_code == "invalid_request"
    assert error.value.provider_error_message == "The requested parameter is invalid."

    with pytest.raises(CerebrasResponseFormatError, match="status=200") as malformed:
        asyncio.run(
            _gateway(
                FakeTransport([CerebrasTransportResponse(status_code=200, body={"choices": [{}]})])
            ).generate(_request())
        )
    assert malformed.value.latency_ms >= 0
    assert malformed.value.message_fields == ()
