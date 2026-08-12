"""Policy-safe contracts for selecting executable-evidence probes."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol, Self, runtime_checkable

from pydantic import Field, model_validator

from socratic_tutor.contracts import ContractModel


class AcquisitionContractError(ValueError):
    """A policy decision does not satisfy its visible request."""


class ProbeClassId(StrEnum):
    HIGH_RELIABILITY_DENSE = "high_reliability_dense"
    HIGH_RELIABILITY_SPARSE = "high_reliability_sparse"
    MODERATE_RELIABILITY_DENSE = "moderate_reliability_dense"
    ASYMMETRIC_RELIABILITY_DENSE = "asymmetric_reliability_dense"


class PolicyId(StrEnum):
    RELIABILITY_AWARE_BOUNDED = "reliability_aware_quantile_bounded"
    SEEDED_RANDOM_BOUNDED = "seeded_random_bounded"
    UNCERTAINTY_ONLY_BOUNDED = "uncertainty_only_bounded"
    PLUG_IN_EVSI_BOUNDED = "plug_in_evsi_bounded"
    NEVER_PROBE = "never_probe"
    ALWAYS_PROBE_BOUNDED = "always_probe_bounded"
    ORACLE_TRUE_RELIABILITY_BOUNDED = "oracle_true_reliability_bounded"


class BetaPosterior(ContractModel):
    """One calibration-derived beta distribution."""

    alpha: float = Field(gt=0.0, allow_inf_nan=False)
    beta: float = Field(gt=0.0, allow_inf_nan=False)

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)


class CalibrationBetaPosterior(ContractModel):
    """Uncertain sensitivity and specificity estimated from calibration data."""

    sensitivity: BetaPosterior
    specificity: BetaPosterior


class AcquisitionCandidate(ContractModel):
    """All information an ordinary policy may see about one candidate probe."""

    case_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
    prior_success_probability: float = Field(gt=0.0, lt=1.0, allow_inf_nan=False)
    probe_class: ProbeClassId
    calibration_beta_posterior: CalibrationBetaPosterior


class AcquisitionRequest(ContractModel):
    """A fixed-budget selection request containing no simulator-owned truth."""

    candidates: tuple[AcquisitionCandidate, ...] = Field(min_length=1)
    remaining_probe_budget: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        case_ids = tuple(candidate.case_id for candidate in self.candidates)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Candidate case IDs must be unique")
        if self.remaining_probe_budget > len(self.candidates):
            raise ValueError("Probe budget cannot exceed the number of candidates")
        return self


class AcquisitionDecision(ContractModel):
    """The ordered cases selected by one policy."""

    policy_id: PolicyId
    selected_case_ids: tuple[str, ...]

    @model_validator(mode="after")
    def validate_unique_case_ids(self) -> Self:
        if len(self.selected_case_ids) != len(set(self.selected_case_ids)):
            raise ValueError("A policy cannot select the same case more than once")
        return self

    def checked_against(self, request: AcquisitionRequest) -> Self:
        """Reject an over-budget, under-budget, or unknown selection."""

        if len(self.selected_case_ids) != request.remaining_probe_budget:
            raise AcquisitionContractError(
                "Selected case count must equal the request's remaining probe budget"
            )
        available_case_ids = {candidate.case_id for candidate in request.candidates}
        unknown_case_ids = set(self.selected_case_ids) - available_case_ids
        if unknown_case_ids:
            joined = ", ".join(sorted(unknown_case_ids))
            raise AcquisitionContractError(f"Policy selected unknown case IDs: {joined}")
        return self


@runtime_checkable
class AcquisitionPolicy(Protocol):
    """Common interface for non-oracle acquisition policies."""

    @property
    def policy_id(self) -> PolicyId: ...

    def select(self, request: AcquisitionRequest) -> AcquisitionDecision: ...
