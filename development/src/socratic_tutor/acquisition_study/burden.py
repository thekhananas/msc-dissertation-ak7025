"""Separate probe burden from measured external execution workload."""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, model_validator

from socratic_tutor.acquisition_study.contracts import PolicyId
from socratic_tutor.contracts import ContractModel


class ExternalWorkloadSource(StrEnum):
    """How external execution measurements were obtained."""

    GLASS_BOX_NO_EXTERNAL_EXECUTION = "glass_box_no_external_execution"
    RECORDED_EXTERNAL_ARTIFACTS = "recorded_external_artifacts"


class CurrencyCostRecord(ContractModel):
    """Optional currency value with enough information to audit its origin."""

    usd_micros: int = Field(ge=0)
    basis: Literal["provider_reported", "dated_pricing_calculation"]
    source_reference: str = Field(min_length=1)
    source_date: date


class ExternalWorkload(ContractModel):
    """External work measured separately, with unavailable values left as None."""

    source: ExternalWorkloadSource
    model_request_count: int = Field(ge=0)
    model_failure_count: int = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    provider_latency_ms: int | None = Field(default=None, ge=0)
    sandbox_execution_count: int = Field(ge=0)
    sandbox_failure_count: int = Field(ge=0)
    sandbox_latency_ms: int | None = Field(default=None, ge=0)
    currency_cost: CurrencyCostRecord | None = None

    @model_validator(mode="after")
    def validate_workload(self) -> Self:
        if self.model_failure_count > self.model_request_count:
            raise ValueError("Model failures cannot exceed model requests")
        if self.sandbox_failure_count > self.sandbox_execution_count:
            raise ValueError("Sandbox failures cannot exceed sandbox executions")
        if self.source is ExternalWorkloadSource.GLASS_BOX_NO_EXTERNAL_EXECUTION:
            counts = (
                self.model_request_count,
                self.model_failure_count,
                self.sandbox_execution_count,
                self.sandbox_failure_count,
            )
            optional_measurements = (
                self.input_tokens,
                self.output_tokens,
                self.provider_latency_ms,
                self.sandbox_latency_ms,
                self.currency_cost,
            )
            if any(counts) or any(value is not None for value in optional_measurements):
                raise ValueError("Glass-box simulation cannot report external execution work")
        return self


class AcquisitionBurdenRecord(ContractModel):
    """Per-episode probe outcomes and external workload without a combined score."""

    schema_version: Literal[1] = 1
    schema_id: Literal["acquisition_study.probe_burden.v1"] = "acquisition_study.probe_burden.v1"
    study_id: Literal["reliability-aware-probing-v1"] = "reliability-aware-probing-v1"
    episode_id: str = Field(min_length=1)
    policy_id: PolicyId
    candidate_count: int = Field(ge=1)
    selected_probe_count: int = Field(ge=0)
    usable_probe_result_count: int = Field(ge=0)
    missing_probe_result_count: int = Field(ge=0)
    failed_probe_result_count: int = Field(ge=0)
    external_workload: ExternalWorkload
    interpretation_scope: Literal["experimental_workload_not_deployment_cost"] = (
        "experimental_workload_not_deployment_cost"
    )

    @model_validator(mode="after")
    def validate_record(self) -> Self:
        if self.selected_probe_count > self.candidate_count:
            raise ValueError("Selected probes cannot exceed candidates")
        accounted_results = (
            self.usable_probe_result_count
            + self.missing_probe_result_count
            + self.failed_probe_result_count
        )
        if accounted_results != self.selected_probe_count:
            raise ValueError("Every selected probe must have exactly one recorded outcome")
        return self


def glass_box_burden_record(
    *,
    episode_id: str,
    policy_id: PolicyId,
    candidate_count: int,
    selected_probe_count: int,
    missing_probe_result_count: int,
    failed_probe_result_count: int,
) -> AcquisitionBurdenRecord:
    """Build a simulator record without inventing external execution measurements."""

    usable_probe_result_count = (
        selected_probe_count - missing_probe_result_count - failed_probe_result_count
    )
    return AcquisitionBurdenRecord(
        episode_id=episode_id,
        policy_id=policy_id,
        candidate_count=candidate_count,
        selected_probe_count=selected_probe_count,
        usable_probe_result_count=usable_probe_result_count,
        missing_probe_result_count=missing_probe_result_count,
        failed_probe_result_count=failed_probe_result_count,
        external_workload=ExternalWorkload(
            source=ExternalWorkloadSource.GLASS_BOX_NO_EXTERNAL_EXECUTION,
            model_request_count=0,
            model_failure_count=0,
            sandbox_execution_count=0,
            sandbox_failure_count=0,
        ),
    )
