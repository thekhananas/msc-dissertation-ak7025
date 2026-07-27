"""Versioned observable failure taxonomy for benchmark review."""

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from socratic_tutor.benchmark.hashing import model_content_hash
from socratic_tutor.contracts import ContractModel


class FailureCategory(StrEnum):
    """Failure labels that can be assigned from observable records."""

    ITEM_AMBIGUITY_OR_TEST_DEFECT = "item_ambiguity_or_test_defect"
    SYNTHETIC_STUDENT_INCONSISTENCY = "synthetic_student_inconsistency"
    EVIDENCE_EXTRACTION_OR_EXECUTION_ERROR = "evidence_extraction_or_execution_error"
    TRACKER_UPDATE_OR_CALIBRATION_ERROR = "tracker_update_or_calibration_error"
    POLICY_SELECTION_ERROR = "policy_selection_error"
    TUTOR_RESPONSE_FAILURE = "tutor_response_failure"
    PROVIDER_OR_SANDBOX_FAILURE = "provider_or_sandbox_failure"
    UNCLASSIFIED = "unclassified"


class FailureCategorySpec(ContractModel):
    """One category with evidence that a reviewer can actually observe."""

    category: FailureCategory
    meaning: str = Field(min_length=1)
    observable_signals: tuple[str, ...] = Field(min_length=1)
    examples: tuple[str, ...] = Field(min_length=1)


class FailureTaxonomy(ContractModel):
    """Frozen review instructions for primary failure analysis."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.failure_taxonomy.v1"] = "benchmark.failure_taxonomy.v1"
    taxonomy_version: str = Field(min_length=1)
    categories: tuple[FailureCategorySpec, ...] = Field(min_length=1)
    reviewer_instructions: tuple[str, ...] = Field(min_length=1)
    adjudication_rules: tuple[str, ...] = Field(min_length=1)
    forbidden_labels: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_taxonomy(self) -> "FailureTaxonomy":
        category_names = tuple(spec.category for spec in self.categories)
        if len(set(category_names)) != len(category_names):
            raise ValueError("Failure taxonomy categories must be unique")
        if set(category_names) != set(FailureCategory):
            raise ValueError("Failure taxonomy must define every required category exactly once")
        forbidden = " ".join(self.forbidden_labels).casefold()
        if "hallucinated reasoning" in forbidden:
            raise ValueError("Hidden reasoning cannot be used as a failure label")
        instructions = (*self.reviewer_instructions, *self.adjudication_rules)
        if any(not rule.strip() for rule in instructions):
            raise ValueError("Failure taxonomy instructions cannot be blank")
        return self


def load_failure_taxonomy(path: Path) -> FailureTaxonomy:
    """Load and validate a YAML taxonomy without changing its source file."""

    import yaml

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"Could not read failure taxonomy: {path}") from error
    return FailureTaxonomy.model_validate(raw)


def failure_taxonomy_hash(taxonomy: FailureTaxonomy) -> str:
    """Return the content hash used to identify reviewer instructions."""

    return model_content_hash(taxonomy)
