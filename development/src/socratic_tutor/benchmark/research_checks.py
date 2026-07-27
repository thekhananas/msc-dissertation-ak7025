"""Pre-run shortcut and sample-sensitivity checks with typed inputs."""

import math
import random
import re
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from statistics import fmean, stdev
from typing import Literal

from pydantic import Field, model_validator

from socratic_tutor.benchmark.artifacts import write_immutable_json
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import model_content_hash
from socratic_tutor.contracts import ContractModel

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NORMAL_975 = 1.959963984540054


class ShortcutVariantKind(StrEnum):
    REFERENCE = "reference"
    IDENTIFIER_RENAME = "identifier_rename"
    PARAPHRASE = "paraphrase"
    CLUE_REMOVAL = "clue_removal"
    SEMANTIC_CHANGE = "semantic_change"


class ShortcutTrainingExample(ContractModel):
    example_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    label: bool


class ShortcutAuditExample(ContractModel):
    example_id: str = Field(min_length=1)
    reference_id: str = Field(min_length=1)
    variant_kind: ShortcutVariantKind
    text: str = Field(min_length=1)
    expected_label: bool


class ShortcutAuditPlan(ContractModel):
    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.shortcut_audit_plan.v1"] = "benchmark.shortcut_audit_plan.v1"
    training_examples: tuple[ShortcutTrainingExample, ...] = Field(min_length=2)
    audit_examples: tuple[ShortcutAuditExample, ...] = Field(min_length=1)
    maximum_surface_accuracy: float = Field(ge=0.0, le=1.0)
    created_at_utc: datetime

    @model_validator(mode="after")
    def validate_audit(self) -> "ShortcutAuditPlan":
        _require_utc(self.created_at_utc, "Shortcut plan creation time")
        training = {example.example_id: example for example in self.training_examples}
        if len(training) != len(self.training_examples):
            raise ValueError("Shortcut training IDs must be unique")
        if len({example.example_id for example in self.audit_examples}) != len(self.audit_examples):
            raise ValueError("Shortcut audit IDs must be unique")
        if len({example.label for example in self.training_examples}) != 2:
            raise ValueError("Shortcut training examples must include both labels")
        for example in self.audit_examples:
            try:
                reference = training[example.reference_id]
            except KeyError as error:
                raise ValueError("Shortcut variant references unknown training example") from error
            should_change = example.variant_kind is ShortcutVariantKind.SEMANTIC_CHANGE
            if should_change is (example.expected_label == reference.label):
                raise ValueError("Shortcut variant label does not match its declared semantics")
        return self


class ShortcutPrediction(ContractModel):
    example_id: str
    variant_kind: ShortcutVariantKind
    expected_label: bool
    predicted_label: bool
    nearest_training_id: str
    lexical_similarity: float = Field(ge=0.0, le=1.0)
    correct: bool


class ShortcutVariantSummary(ContractModel):
    """Accuracy for one controlled wording transformation."""

    variant_kind: ShortcutVariantKind
    audit_count: int = Field(ge=1) #audit_count >= 1
    accuracy: float = Field(ge=0.0, le=1.0) #0.0 <= accuracy <= 1.0


class ShortcutAuditSummary(ContractModel):
    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.shortcut_audit_summary.v1"] = (
        "benchmark.shortcut_audit_summary.v1"
    )
    method: Literal["token_jaccard_1nn_v1"] = "token_jaccard_1nn_v1"
    training_count: int = Field(ge=2)
    audit_count: int = Field(ge=1)
    surface_accuracy: float = Field(ge=0.0, le=1.0)
    maximum_surface_accuracy: float = Field(ge=0.0, le=1.0)
    gate_passed: bool
    variant_summaries: tuple[ShortcutVariantSummary, ...] = Field(min_length=1)
    predictions: tuple[ShortcutPrediction, ...]
    plan_hash: Sha256
    summary_hash: Sha256

    @model_validator(mode="after")
    def validate_summary(self) -> "ShortcutAuditSummary":
        if self.gate_passed is not (self.surface_accuracy <= self.maximum_surface_accuracy):
            raise ValueError("Shortcut gate does not match the frozen threshold")
        if self.summary_hash != model_content_hash(self, exclude={"summary_hash"}):
            raise ValueError("Shortcut summary hash does not match its content")
        return self


class SensitivityPlan(ContractModel):
    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.sensitivity_plan.v1"] = "benchmark.sensitivity_plan.v1"
    case_count: int = Field(ge=4, le=200)
    repeats_per_case: int = Field(ge=1, le=20)
    simulation_count: int = Field(ge=20, le=10000)
    true_effects: tuple[float, ...] = Field(min_length=2)
    between_case_sd: float = Field(gt=0.0, le=1.0)
    within_case_sd: float = Field(ge=0.0, le=1.0)
    repeat_missing_probability: float = Field(ge=0.0, lt=1.0)
    minimum_interpretable_effect: float = Field(gt=0.0, le=1.0)
    minimum_power: float = Field(ge=0.0, le=1.0)
    maximum_false_positive_rate: float = Field(ge=0.0, le=1.0)
    minimum_mean_eligible_cases: float = Field(ge=2.0)
    seed: int = Field(ge=0, le=2**32 - 1)
    created_at_utc: datetime

    @model_validator(mode="after")
    def validate_plan(self) -> "SensitivityPlan":
        _require_utc(self.created_at_utc, "Sensitivity plan creation time")
        if len(set(self.true_effects)) != len(self.true_effects):
            raise ValueError("Sensitivity effect grid must be unique")
        if 0.0 not in self.true_effects:
            raise ValueError("Sensitivity effect grid must include the null effect")
        if self.minimum_interpretable_effect not in self.true_effects:
            raise ValueError("Sensitivity grid must include the interpretable effect")
        if any(not -1.0 <= effect <= 1.0 for effect in self.true_effects):
            raise ValueError("Sensitivity effects must lie within [-1, 1]")
        if self.minimum_mean_eligible_cases > self.case_count:
            raise ValueError("Required eligible cases cannot exceed planned cases")
        return self


class SensitivityEffectSummary(ContractModel):
    true_effect: float = Field(ge=-1.0, le=1.0)
    successful_simulations: int = Field(ge=0)
    mean_eligible_cases: float = Field(ge=0.0)
    mean_interval_width: float | None = Field(default=None, ge=0.0)
    rejection_rate: float = Field(ge=0.0, le=1.0)
    coverage_rate: float = Field(ge=0.0, le=1.0)


class SensitivitySummary(ContractModel):
    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.sensitivity_summary.v1"] = "benchmark.sensitivity_summary.v1"
    method: Literal["nested_case_normal_interval_v1"] = "nested_case_normal_interval_v1"
    case_count: int
    repeats_per_case: int
    simulation_count: int
    effects: tuple[SensitivityEffectSummary, ...]
    false_positive_rate: float = Field(ge=0.0, le=1.0)
    interpretable_effect_power: float = Field(ge=0.0, le=1.0)
    mean_eligible_cases_at_interpretable_effect: float = Field(ge=0.0)
    gate_passed: bool
    plan_hash: Sha256
    summary_hash: Sha256

    @model_validator(mode="after")
    def validate_summary_hash(self) -> "SensitivitySummary":
        if self.summary_hash != model_content_hash(self, exclude={"summary_hash"}):
            raise ValueError("Sensitivity summary hash does not match its content")
        return self


def run_shortcut_audit(
    plan: ShortcutAuditPlan,
    *,
    output_path: Path,
) -> ShortcutAuditSummary:
    """Evaluate a deterministic lexical 1-NN shortcut baseline."""

    training = tuple((example, _tokens(example.text)) for example in plan.training_examples)
    predictions: list[ShortcutPrediction] = []
    for audit in plan.audit_examples:
        audit_tokens = _tokens(audit.text)
        nearest, similarity = max(
            ((example, _jaccard(audit_tokens, tokens)) for example, tokens in training),
            key=lambda item: (item[1], item[0].example_id),
        )
        predictions.append(
            ShortcutPrediction(
                example_id=audit.example_id,
                variant_kind=audit.variant_kind,
                expected_label=audit.expected_label,
                predicted_label=nearest.label,
                nearest_training_id=nearest.example_id,
                lexical_similarity=similarity,
                correct=nearest.label == audit.expected_label,
            )
        )
    accuracy = sum(item.correct for item in predictions) / len(predictions)
    variant_summaries = tuple(
        ShortcutVariantSummary(
            variant_kind=variant_kind,
            audit_count=sum(item.variant_kind is variant_kind for item in predictions),
            accuracy=(
                sum(item.correct for item in predictions if item.variant_kind is variant_kind)
                / sum(item.variant_kind is variant_kind for item in predictions)
            ),
        )
        for variant_kind in ShortcutVariantKind
        if any(item.variant_kind is variant_kind for item in predictions)
    )
    content: dict[str, object] = {
        "schema_version": 1,
        "schema_id": "benchmark.shortcut_audit_summary.v1",
        "method": "token_jaccard_1nn_v1",
        "training_count": len(training),
        "audit_count": len(predictions),
        "surface_accuracy": accuracy,
        "maximum_surface_accuracy": plan.maximum_surface_accuracy,
        "gate_passed": accuracy <= plan.maximum_surface_accuracy,
        "variant_summaries": variant_summaries,
        "predictions": tuple(predictions),
        "plan_hash": model_content_hash(plan),
    }
    draft = ShortcutAuditSummary.model_construct(
        _fields_set=set(content),
        **content,
        summary_hash="0" * 64,
    )
    summary = ShortcutAuditSummary.model_validate(
        {**content, "summary_hash": model_content_hash(draft, exclude={"summary_hash"})}
    )
    write_immutable_json(output_path, summary)
    return summary


def run_sensitivity_analysis(
    plan: SensitivityPlan,
    *,
    output_path: Path,
) -> SensitivitySummary:
    """Simulate nested repeats while treating cases as the independent units."""

    effect_summaries = tuple(
        _simulate_effect(
            plan,
            effect=true_effect,
            randomizer=random.Random(plan.seed),
        )
        for true_effect in plan.true_effects
    )
    by_effect = {summary.true_effect: summary for summary in effect_summaries}
    null = by_effect[0.0]
    target = by_effect[plan.minimum_interpretable_effect]
    gate_passed = (
        null.rejection_rate <= plan.maximum_false_positive_rate
        and target.rejection_rate >= plan.minimum_power
        and target.mean_eligible_cases >= plan.minimum_mean_eligible_cases
    )
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.sensitivity_summary.v1",
        "method": "nested_case_normal_interval_v1",
        "case_count": plan.case_count,
        "repeats_per_case": plan.repeats_per_case,
        "simulation_count": plan.simulation_count,
        "effects": effect_summaries,
        "false_positive_rate": null.rejection_rate,
        "interpretable_effect_power": target.rejection_rate,
        "mean_eligible_cases_at_interpretable_effect": target.mean_eligible_cases,
        "gate_passed": gate_passed,
        "plan_hash": model_content_hash(plan),
    }
    draft = SensitivitySummary.model_construct(
        _fields_set=set(content),
        **content,
        summary_hash="0" * 64,
    )
    summary = SensitivitySummary.model_validate(
        {**content, "summary_hash": model_content_hash(draft, exclude={"summary_hash"})}
    )
    write_immutable_json(output_path, summary)
    return summary


def _simulate_effect(
    plan: SensitivityPlan,
    *,
    effect: float,
    randomizer: random.Random,
) -> SensitivityEffectSummary:
    eligible_counts: list[int] = []
    interval_widths: list[float] = []
    rejections: int = 0
    covered: int = 0
    successful: int = 0
    for _ in range(plan.simulation_count):
        case_effects: list[float] = []
        for _case in range(plan.case_count):
            latent = randomizer.gauss(effect, plan.between_case_sd)
            repeats = [
                randomizer.gauss(latent, plan.within_case_sd)
                for _repeat in range(plan.repeats_per_case)
                if randomizer.random() >= plan.repeat_missing_probability
            ]
            if repeats:
                case_effects.append(fmean(repeats))
        eligible_counts.append(len(case_effects))
        if len(case_effects) < 2:
            continue
        mean_effect = fmean(case_effects)
        standard_error = stdev(case_effects) / math.sqrt(len(case_effects))
        half_width = _NORMAL_975 * standard_error
        lower = mean_effect - half_width
        upper = mean_effect + half_width
        interval_widths.append(2 * half_width)
        successful += 1
        rejections += lower > 0.0 or upper < 0.0
        covered += lower <= effect <= upper
    return SensitivityEffectSummary(
        true_effect=effect,
        successful_simulations=successful,
        mean_eligible_cases=fmean(eligible_counts),
        mean_interval_width=fmean(interval_widths) if interval_widths else None,
        rejection_rate=rejections / successful if successful else 0.0,
        coverage_rate=covered / successful if successful else 0.0,
    )


def _tokens(text: str) -> frozenset[str]:
    return frozenset(token.casefold() for token in _TOKEN.findall(text))


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _require_utc(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must use UTC")
