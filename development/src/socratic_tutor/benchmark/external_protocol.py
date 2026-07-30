"""Immutable study settings for a v2 external-model benchmark execution."""

from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Literal, cast

from pydantic import Field, model_validator

from socratic_tutor.benchmark.analysis_spec import (
    AnalysisSpecification,
    analysis_specification_hash,
)
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.evaluator.models import (
    AuthoredBenchmarkManifest,
    ManifestStatus,
)
from socratic_tutor.benchmark.evaluator.projection import authored_manifest_hash
from socratic_tutor.benchmark.hashing import canonical_sha256, model_content_hash
from socratic_tutor.benchmark.public.models import PublicBenchmarkManifest
from socratic_tutor.contracts import ContractModel


class ExternalModelExecutionProtocol(ContractModel):
    """Settings that must be fixed before a held-out provider execution begins."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.external_model_protocol.v1"] = (
        "benchmark.external_model_protocol.v1"
    )
    protocol_version: Literal["v2"] = "v2"
    source_benchmark_version: Literal["v1"] = "v1"
    source_manifest_hash: Sha256
    source_public_projection_hash: Sha256
    analysis_specification_hash: Sha256
    provider_id: str = Field(min_length=1)
    adapter_version: str = Field(min_length=1)
    requested_model_id: str = Field(min_length=1)
    system_prompt_version: str = Field(min_length=1)
    system_prompt_sha256: Sha256
    temperature: float = Field(ge=0.0, le=2.0)
    max_output_tokens: int = Field(ge=1)
    run_seed: int = Field(ge=0, le=2**32 - 1)
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    timeout_seconds: float = Field(gt=0.0, le=120.0)
    max_attempts: int = Field(ge=1, le=5)
    request_spacing_seconds: float = Field(ge=0.0, le=120.0)
    repeats_per_case: int = Field(ge=1, le=10)
    request_budget: int = Field(ge=1)
    cost_budget_usd: Decimal = Field(gt=Decimal("0"))
    missingness_rule: Literal["exclude_missing_criterion_and_report_denominator"]
    created_at_utc: datetime
    protocol_hash: Sha256

    @model_validator(mode="after")
    def validate_protocol_hash_and_time(self) -> "ExternalModelExecutionProtocol":
        if self.created_at_utc.tzinfo is None or self.created_at_utc.utcoffset() != UTC.utcoffset(
            self.created_at_utc
        ):
            raise ValueError("External-model protocol timestamp must be UTC")
        if self.protocol_hash != external_model_protocol_hash(self):
            raise ValueError("External-model protocol hash does not match its content")
        return self


def external_model_protocol_hash(protocol: ExternalModelExecutionProtocol) -> Sha256:
    """Hash every protocol field except the self-referential digest."""

    return canonical_sha256(protocol.model_dump(mode="json", exclude={"protocol_hash"}))


def create_external_model_execution_protocol(
    *,
    manifest: AuthoredBenchmarkManifest,
    public_manifest: PublicBenchmarkManifest,
    analysis_specification: AnalysisSpecification,
    provider_id: str,
    adapter_version: str,
    requested_model_id: str,
    system_prompt_version: str,
    system_prompt_sha256: str,
    temperature: float,
    max_output_tokens: int,
    run_seed: int,
    reasoning_effort: Literal["low", "medium", "high"] | None,
    timeout_seconds: float,
    max_attempts: int,
    request_spacing_seconds: float,
    repeats_per_case: int,
    request_budget: int,
    cost_budget_usd: Decimal,
    created_at_utc: datetime,
) -> ExternalModelExecutionProtocol:
    """Create a v2 protocol only when it is bound to frozen v1 inputs."""

    _validate_bound_sources(manifest, public_manifest, analysis_specification)
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.external_model_protocol.v1",
        "protocol_version": "v2",
        "source_benchmark_version": "v1",
        "source_manifest_hash": authored_manifest_hash(manifest),
        "source_public_projection_hash": public_manifest.projection_hash,
        "analysis_specification_hash": analysis_specification_hash(analysis_specification),
        "provider_id": provider_id,
        "adapter_version": adapter_version,
        "requested_model_id": requested_model_id,
        "system_prompt_version": system_prompt_version,
        "system_prompt_sha256": system_prompt_sha256,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "run_seed": run_seed,
        "reasoning_effort": reasoning_effort,
        "timeout_seconds": timeout_seconds,
        "max_attempts": max_attempts,
        "request_spacing_seconds": request_spacing_seconds,
        "repeats_per_case": repeats_per_case,
        "request_budget": request_budget,
        "cost_budget_usd": cost_budget_usd,
        "missingness_rule": analysis_specification.missingness_rule,
        "created_at_utc": created_at_utc,
    }
    return ExternalModelExecutionProtocol.model_validate(
        {**content, "protocol_hash": _protocol_content_hash(content)}
    )


def validate_external_model_execution_protocol(
    protocol: ExternalModelExecutionProtocol,
    *,
    manifest: AuthoredBenchmarkManifest,
    public_manifest: PublicBenchmarkManifest,
    analysis_specification: AnalysisSpecification,
) -> None:
    """Verify that a stored protocol still describes these exact frozen inputs."""

    _validate_bound_sources(manifest, public_manifest, analysis_specification)
    if protocol.source_manifest_hash != authored_manifest_hash(manifest):
        raise ValueError("External-model protocol belongs to a different frozen manifest")
    if protocol.source_public_projection_hash != public_manifest.projection_hash:
        raise ValueError("External-model protocol belongs to a different public projection")
    if protocol.analysis_specification_hash != analysis_specification_hash(analysis_specification):
        raise ValueError("External-model protocol belongs to a different analysis specification")
    if protocol.missingness_rule != analysis_specification.missingness_rule:
        raise ValueError("External-model protocol has a different missingness rule")


def _validate_bound_sources(
    manifest: AuthoredBenchmarkManifest,
    public_manifest: PublicBenchmarkManifest,
    analysis_specification: AnalysisSpecification,
) -> None:
    if manifest.status is not ManifestStatus.FROZEN:
        raise ValueError("External-model execution requires a frozen source manifest")
    if manifest.benchmark_version != "v1":
        raise ValueError("External-model protocol requires source benchmark version v1")
    source_manifest_hash = authored_manifest_hash(manifest)
    if public_manifest.benchmark_version != manifest.benchmark_version:
        raise ValueError("Public projection belongs to a different benchmark version")
    if public_manifest.source_manifest_hash != source_manifest_hash:
        raise ValueError("Public projection belongs to a different frozen manifest")
    if analysis_specification.benchmark_version != manifest.benchmark_version:
        raise ValueError("Analysis specification belongs to a different benchmark version")


def _protocol_content_hash(content: Mapping[str, object]) -> Sha256:
    """Hash constructor input using Pydantic's canonical JSON representation."""

    draft = ExternalModelExecutionProtocol.model_construct(
        **cast(dict[str, Any], content),
        protocol_hash="0" * 64,
    )
    return model_content_hash(draft, exclude={"protocol_hash"})
