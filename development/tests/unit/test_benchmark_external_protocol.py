"""Tests for the immutable v2 external-model execution protocol."""

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from socratic_tutor.benchmark.analysis_spec import (
    AnalysisSpecification,
    load_analysis_specification,
)
from socratic_tutor.benchmark.evaluator.loader import load_and_verify_manifest
from socratic_tutor.benchmark.evaluator.models import AuthoredBenchmarkManifest
from socratic_tutor.benchmark.evaluator.projection import project_public_manifest
from socratic_tutor.benchmark.external_protocol import (
    ExternalModelExecutionProtocol,
    create_external_model_execution_protocol,
    freeze_external_model_execution_protocol,
    load_external_model_protocol_freeze_plan,
    validate_external_model_execution_protocol,
)
from socratic_tutor.benchmark.public.models import PublicBenchmarkManifest

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = WORKSPACE_ROOT / "data" / "benchmarks" / "v1" / "manifest.yaml"
SPEC_PATH = WORKSPACE_ROOT / "configs" / "benchmark" / "v1-analysis-spec.yaml"
FREEZE_PLAN_PATH = WORKSPACE_ROOT / "configs" / "benchmark" / "v2-external-protocol-cerebras.yaml"
PROMPT_PATH = (
    WORKSPACE_ROOT
    / "data"
    / "benchmark-execution"
    / "v2"
    / "prompts"
    / "external-evaluation-system-v1.md"
)
PROMPT_HASH = "a" * 64


def _protocol() -> tuple[
    ExternalModelExecutionProtocol,
    AuthoredBenchmarkManifest,
    PublicBenchmarkManifest,
    AnalysisSpecification,
]:
    manifest = load_and_verify_manifest(MANIFEST_PATH)
    public_manifest = project_public_manifest(
        manifest,
        projected_at_utc=datetime(2026, 8, 23, tzinfo=UTC),
    )
    specification = load_analysis_specification(SPEC_PATH)
    protocol = create_external_model_execution_protocol(
        manifest=manifest,
        public_manifest=public_manifest,
        analysis_specification=specification,
        provider_id="cerebras",
        adapter_version="cerebras-chat-completions-v1",
        requested_model_id="gpt-oss-120b",
        system_prompt_version="external-evaluation-v2",
        system_prompt_sha256=PROMPT_HASH,
        temperature=0.0,
        max_output_tokens=1_024,
        run_seed=20260823,
        reasoning_effort="medium",
        timeout_seconds=60.0,
        max_attempts=2,
        request_spacing_seconds=1.0,
        repeats_per_case=1,
        request_budget=72,
        cost_budget_usd=Decimal("5.00"),
        created_at_utc=datetime(2026, 8, 23, tzinfo=UTC),
    )
    return protocol, manifest, public_manifest, specification


def test_protocol_binds_exact_frozen_inputs_and_execution_settings() -> None:
    protocol, manifest, public_manifest, specification = _protocol()

    assert protocol.source_benchmark_version == "v1"
    assert protocol.provider_id == "cerebras"
    assert protocol.repeats_per_case == 1
    assert len(protocol.protocol_hash) == 64

    validate_external_model_execution_protocol(
        protocol,
        manifest=manifest,
        public_manifest=public_manifest,
        analysis_specification=specification,
    )


def test_protocol_rejects_tampered_execution_setting() -> None:
    protocol, _, _, _ = _protocol()
    payload = protocol.model_dump(mode="python")
    payload["temperature"] = 1.0

    with pytest.raises(ValidationError, match="hash does not match"):
        ExternalModelExecutionProtocol.model_validate(payload)


def test_protocol_rejects_non_utc_timestamp() -> None:
    protocol, _, _, _ = _protocol()
    payload = protocol.model_dump(mode="python")
    payload["created_at_utc"] = datetime(2026, 8, 23)

    with pytest.raises(ValidationError, match="timestamp must be UTC"):
        ExternalModelExecutionProtocol.model_validate(payload)


def test_protocol_rejects_a_different_analysis_specification() -> None:
    protocol, manifest, public_manifest, specification = _protocol()
    changed_specification = specification.model_copy(update={"minimum_interpretable_effect": 0.2})

    with pytest.raises(ValueError, match="different analysis specification"):
        validate_external_model_execution_protocol(
            protocol,
            manifest=manifest,
            public_manifest=public_manifest,
            analysis_specification=changed_specification,
        )


def test_protocol_freeze_binds_the_external_prompt_and_full_request_budget(
    tmp_path: Path,
) -> None:
    protocol = freeze_external_model_execution_protocol(
        plan=load_external_model_protocol_freeze_plan(FREEZE_PLAN_PATH),
        manifest_path=MANIFEST_PATH,
        analysis_specification_path=SPEC_PATH,
        system_prompt_path=PROMPT_PATH,
        output_path=tmp_path / "external_model_protocol.json",
    )

    assert protocol.provider_id == "cerebras"
    assert protocol.request_budget == 72
    assert protocol.repeats_per_case == 1
    assert (tmp_path / "external_model_protocol.json").exists()
