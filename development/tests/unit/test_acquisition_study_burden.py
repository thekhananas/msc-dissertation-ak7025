"""Checks that simulated probes are not reported as real external work."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from socratic_tutor.acquisition_study import (
    ExternalWorkload,
    ExternalWorkloadSource,
    PolicyId,
    glass_box_burden_record,
)


def test_glass_box_record_separates_simulated_outcomes_from_external_work() -> None:
    record = glass_box_burden_record(
        episode_id="matched:0001",
        policy_id=PolicyId.RELIABILITY_AWARE_BOUNDED,
        candidate_count=40,
        selected_probe_count=20,
        missing_probe_result_count=2,
        failed_probe_result_count=1,
    )

    assert record.usable_probe_result_count == 17
    assert record.external_workload.model_request_count == 0
    assert record.external_workload.sandbox_execution_count == 0
    assert record.external_workload.input_tokens is None
    assert record.external_workload.provider_latency_ms is None
    assert record.external_workload.currency_cost is None


def test_glass_box_source_rejects_real_execution_measurements() -> None:
    with pytest.raises(ValidationError, match="cannot report external execution work"):
        ExternalWorkload(
            source=ExternalWorkloadSource.GLASS_BOX_NO_EXTERNAL_EXECUTION,
            model_request_count=1,
            model_failure_count=0,
            input_tokens=100,
            output_tokens=20,
            provider_latency_ms=250,
            sandbox_execution_count=1,
            sandbox_failure_count=0,
            sandbox_latency_ms=40,
        )
