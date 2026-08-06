# pyright: reportPrivateUsage=false
"""Tests for operational resource reconciliation."""

from datetime import UTC, datetime, timedelta

from socratic_tutor.benchmark.replay import (
    ProviderResponseMetadata,
    RecordedGenerationResponse,
)
from socratic_tutor.benchmark.resource_reconciliation import _provider_phase


def test_provider_phase_separates_latency_from_capture_span() -> None:
    started = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)
    responses = (
        _response(started, input_tokens=100, output_tokens=20, latency_ms=250),
        _response(
            started + timedelta(seconds=12),
            input_tokens=110,
            output_tokens=30,
            latency_ms=300,
        ),
    )

    summary = _provider_phase(
        "decision_generation",
        responses,
        provider_failure_count=0,
        retry_count=0,
    )

    assert summary.input_tokens == 210
    assert summary.output_tokens == 50
    assert summary.summed_provider_latency_ms == 550
    assert summary.response_capture_span_seconds == 12.0
    assert summary.responses_with_reported_cost == 0


def _response(
    captured_at_utc: datetime,
    *,
    input_tokens: int,
    output_tokens: int,
    latency_ms: int,
) -> RecordedGenerationResponse:
    metadata = ProviderResponseMetadata(
        provider_id="cerebras",
        model_id="gpt-oss-120b",
        resolved_provider_id="cerebras",
        resolved_model_id="gpt-oss-120b",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
    )
    return RecordedGenerationResponse.model_construct(
        provider_metadata=metadata,
        captured_at_utc=captured_at_utc,
    )
