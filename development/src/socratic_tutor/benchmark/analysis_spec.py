"""Pre-registered analysis rules for the v1 benchmark."""

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from socratic_tutor.benchmark.hashing import model_content_hash
from socratic_tutor.contracts import ContractModel


class AnalysisSpecification(ContractModel):
    """Analysis choices fixed before held-out outcomes are available."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.analysis_specification.v1"] = (
        "benchmark.analysis_specification.v1"
    )
    benchmark_version: str = Field(min_length=1)
    primary_metric: Literal["paired_brier_difference"] = "paired_brier_difference"
    uncalibrated_fallback_metric: Literal["paired_classification_error_difference"] = (
        "paired_classification_error_difference"
    )
    aggregation_unit: Literal["case"] = "case"
    aggregation_version: str = Field(min_length=1)
    missingness_rule: Literal["exclude_missing_criterion_and_report_denominator"] = (
        "exclude_missing_criterion_and_report_denominator"
    )
    minimum_interpretable_effect: float = Field(gt=0.0, le=1.0)
    policy_threshold: float = Field(ge=0.0, le=1.0)
    bootstrap_method: Literal["paired_case_percentile"] = "paired_case_percentile"
    bootstrap_resamples: int = Field(ge=1000, le=100000)
    permutation_method: Literal["within_case_sign_swap"] = "within_case_sign_swap"
    permutation_resamples: int = Field(ge=1000, le=100000)
    random_seed: int = Field(ge=0, le=2**32 - 1)
    confirmatory_exclusions: tuple[str, ...] = Field(min_length=1)
    exploratory_metrics: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_rules(self) -> "AnalysisSpecification":
        if self.bootstrap_resamples != self.permutation_resamples:
            raise ValueError("Bootstrap and permutation resamples must use one frozen count")
        entries = (*self.confirmatory_exclusions, *self.exploratory_metrics)
        if any(not item.strip() for item in entries):
            raise ValueError("Analysis specification entries cannot be blank")
        if not any("case" in item.casefold() for item in self.confirmatory_exclusions):
            raise ValueError("Analysis exclusions must state the case-level independence rule")
        return self


def load_analysis_specification(path: Path) -> AnalysisSpecification:
    """Load one analysis specification from YAML."""

    import yaml

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"Could not read analysis specification: {path}") from error
    return AnalysisSpecification.model_validate(raw)


def analysis_specification_hash(specification: AnalysisSpecification) -> str:
    """Return the content hash used to bind an analysis run to its rules."""

    return model_content_hash(specification)
