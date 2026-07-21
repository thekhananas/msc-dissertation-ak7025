"""Strict case-level aggregate records consumed by later statistical analysis."""

from datetime import datetime, timedelta
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.evaluator.scoring import RepeatEstimator
from socratic_tutor.benchmark.hashing import model_content_hash
from socratic_tutor.contracts import ContractModel


class CaseComparison(StrEnum):
    """Frozen within-case loss comparisons in favourable-direction order."""

    PRIMARY_VALID = "primary_valid"
    UNRELATED_CONTROL = "unrelated_control"
    CORRUPTED_CONTROL = "corrupted_control"
    BASELINE_CONSTANT = "baseline_constant"
    BASELINE_PROBE_ONLY = "baseline_probe_only"


_ESTIMATORS_BY_COMPARISON: dict[
    CaseComparison,
    tuple[RepeatEstimator, RepeatEstimator],
] = {
    CaseComparison.PRIMARY_VALID: (
        RepeatEstimator.DIALOGUE_ONLY,
        RepeatEstimator.PROBE_INFORMED,
    ),
    CaseComparison.UNRELATED_CONTROL: (
        RepeatEstimator.DIALOGUE_ONLY,
        RepeatEstimator.UNRELATED_PROBE,
    ),
    CaseComparison.CORRUPTED_CONTROL: (
        RepeatEstimator.DIALOGUE_ONLY,
        RepeatEstimator.CORRUPTED_PROBE,
    ),
    CaseComparison.BASELINE_CONSTANT: (
        RepeatEstimator.DIALOGUE_ONLY,
        RepeatEstimator.CONSTANT_PREVALENCE,
    ),
    CaseComparison.BASELINE_PROBE_ONLY: (
        RepeatEstimator.DIALOGUE_ONLY,
        RepeatEstimator.PROBE_ONLY,
    ),
}


class CaseAggregate(ContractModel):
    """One nested-repeat comparison for one independent benchmark case."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.case_aggregate.v1"] = "benchmark.case_aggregate.v1"
    benchmark_version: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    model_route_id: str = Field(min_length=1)
    comparison: CaseComparison
    left_estimator: RepeatEstimator
    right_estimator: RepeatEstimator
    eligible_repeat_count: int = Field(ge=0)
    planned_repeat_count: int = Field(ge=1)
    mean_left_loss: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    mean_right_loss: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    case_effect: float | None = Field(default=None, ge=-1.0, le=1.0, allow_inf_nan=False)
    missing_repeat_count: int = Field(ge=0)
    missingness_rule: str = Field(min_length=1)
    aggregation_version: str = Field(min_length=1)
    source_repeat_hashes: tuple[Sha256, ...]
    created_at_utc: datetime
    record_hash: Sha256

    @model_validator(mode="after")
    def validate_aggregate(self) -> "CaseAggregate":
        _require_utc(self.created_at_utc, "Case-aggregate creation time")
        expected_left, expected_right = _ESTIMATORS_BY_COMPARISON[self.comparison]
        if (self.left_estimator, self.right_estimator) != (
            expected_left,
            expected_right,
        ):
            raise ValueError("Case comparison has the wrong estimator pair")
        if self.eligible_repeat_count + self.missing_repeat_count != self.planned_repeat_count:
            raise ValueError("Eligible and missing repeats must equal planned repeats")
        if len(set(self.source_repeat_hashes)) != len(self.source_repeat_hashes):
            raise ValueError("Case aggregate source hashes must be unique")
        if self.eligible_repeat_count == 0:
            if any(
                value is not None
                for value in (
                    self.mean_left_loss,
                    self.mean_right_loss,
                    self.case_effect,
                )
            ):
                raise ValueError("Aggregate without eligible repeats cannot contain losses")
        else:
            if (
                self.mean_left_loss is None
                or self.mean_right_loss is None
                or self.case_effect is None
            ):
                raise ValueError("Eligible aggregate requires both means and a case effect")
            expected_effect = self.mean_left_loss - self.mean_right_loss
            if abs(self.case_effect - expected_effect) > 1e-12:
                raise ValueError("Case effect must equal mean left loss minus mean right loss")
            if len(self.source_repeat_hashes) < 2 * self.eligible_repeat_count:
                raise ValueError("Eligible aggregate lacks paired repeat-level sources")
        if self.record_hash != case_aggregate_hash(self):
            raise ValueError("Case aggregate hash does not match its content")
        return self


def case_aggregate_hash(aggregate: CaseAggregate) -> Sha256:
    """Hash all case-aggregate fields except the self digest."""

    return model_content_hash(aggregate, exclude={"record_hash"})


def create_case_aggregate(
    *,
    benchmark_version: str,
    run_id: str,
    case_id: str,
    model_route_id: str,
    comparison: CaseComparison,
    eligible_repeat_count: int,
    planned_repeat_count: int,
    mean_left_loss: float | None,
    mean_right_loss: float | None,
    missingness_rule: str,
    aggregation_version: str,
    source_repeat_hashes: tuple[Sha256, ...],
    created_at_utc: datetime,
) -> CaseAggregate:
    """Create a record while deriving estimator identities, missingness, and effect."""

    left, right = _ESTIMATORS_BY_COMPARISON[comparison]
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.case_aggregate.v1",
        "benchmark_version": benchmark_version,
        "run_id": run_id,
        "case_id": case_id,
        "model_route_id": model_route_id,
        "comparison": comparison,
        "left_estimator": left,
        "right_estimator": right,
        "eligible_repeat_count": eligible_repeat_count,
        "planned_repeat_count": planned_repeat_count,
        "mean_left_loss": mean_left_loss,
        "mean_right_loss": mean_right_loss,
        "case_effect": (
            None
            if mean_left_loss is None or mean_right_loss is None
            else mean_left_loss - mean_right_loss
        ),
        "missing_repeat_count": planned_repeat_count - eligible_repeat_count,
        "missingness_rule": missingness_rule,
        "aggregation_version": aggregation_version,
        "source_repeat_hashes": source_repeat_hashes,
        "created_at_utc": created_at_utc,
    }
    draft = CaseAggregate.model_construct(
        _fields_set=set(content),
        **content,
        record_hash="0" * 64,
    )
    return CaseAggregate.model_validate({**content, "record_hash": case_aggregate_hash(draft)})


def _require_utc(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must use UTC")
