"""Deterministic, decision-safe construction of benchmark negative controls."""

from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Literal, cast

import yaml
from pydantic import Field, model_validator

from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import canonical_sha256, file_sha256, model_content_hash
from socratic_tutor.benchmark.public.models import (
    BenchmarkCaseView,
    BenchmarkCondition,
    PublicArtifactClass,
    PublicBenchmarkManifest,
    PublicFileEntry,
)
from socratic_tutor.benchmark.public.safety import assert_public_payload_safe
from socratic_tutor.contracts.models import ContractModel


class ControlConstructionError(ValueError):
    """A control specification or source execution is invalid."""


class ControlTransformation(StrEnum):
    """Frozen transformations supported by the first benchmark version."""

    FROZEN_UNRELATED_ASSIGNMENT = "frozen_unrelated_assignment"
    INVERT_PASS_FAIL_SUMMARY = "invert_pass_fail_summary"


class UnrelatedControlSpec(ContractModel):
    """Frozen mapping from each case to a different-concept evidence source."""

    schema_id: Literal["benchmark.unrelated_control.v1"]
    assignments: dict[str, str] = Field(min_length=1)
    rule: Literal["use_the_other_cases_evidence_summary_without_criterion_access"]


class CorruptionControlSpec(ContractModel):
    """Frozen deterministic transformation over an evidence execution summary."""

    schema_id: Literal["benchmark.corruption_control.v1"]
    transformation: Literal["invert_pass_fail_summary"]
    input_fields: tuple[Literal["passed"], Literal["failed"]]
    reject_undeclared_fields: Literal[True]
    seed: int = Field(ge=0, le=2**32 - 1)


class ProbeExecutionSummary(ContractModel):
    """Successful evidence-probe result containing no evaluator-owned outcome."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.probe_execution_summary.v1"] = (
        "benchmark.probe_execution_summary.v1"
    )
    case_id: str = Field(min_length=1)
    target_concept: str = Field(min_length=1)
    evidence_test_sha256: Sha256
    passed: int = Field(ge=0)
    failed: int = Field(ge=0)
    summary_hash: Sha256

    @model_validator(mode="after")
    def validate_successful_summary(self) -> "ProbeExecutionSummary":
        if self.passed + self.failed == 0:
            raise ValueError("A successful probe summary requires at least one test result")
        expected_hash = model_content_hash(self, exclude={"summary_hash"})
        if self.summary_hash != expected_hash:
            raise ValueError("Probe execution summary hash does not match its content")
        return self


class ControlEvidenceRecord(ContractModel):
    """Content-addressed negative-control evidence for one benchmark condition."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.control_evidence.v1"] = "benchmark.control_evidence.v1"
    target_case_id: str = Field(min_length=1)
    condition: Literal[
        BenchmarkCondition.UNRELATED_PROBE,
        BenchmarkCondition.CORRUPTED_PROBE,
    ]
    source_case_id: str = Field(min_length=1)
    source_target_concept: str = Field(min_length=1)
    passed: int = Field(ge=0)
    failed: int = Field(ge=0)
    source_summary_hash: Sha256
    control_spec_hash: Sha256
    transformation: ControlTransformation
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)
    record_hash: Sha256

    @model_validator(mode="after")
    def validate_condition_and_hash(self) -> "ControlEvidenceRecord":
        if self.condition is BenchmarkCondition.UNRELATED_PROBE:
            if self.target_case_id == self.source_case_id:
                raise ValueError("Unrelated control must use a different source case")
            if self.transformation is not ControlTransformation.FROZEN_UNRELATED_ASSIGNMENT:
                raise ValueError("Unrelated control has the wrong transformation")
            if self.seed is not None:
                raise ValueError("Frozen unrelated assignments do not use a runtime seed")
        else:
            if self.target_case_id != self.source_case_id:
                raise ValueError("Corrupted control must transform its target case evidence")
            if self.transformation is not ControlTransformation.INVERT_PASS_FAIL_SUMMARY:
                raise ValueError("Corrupted control has the wrong transformation")
            if self.seed is None:
                raise ValueError("Corrupted control must record its frozen seed")

        expected_hash = model_content_hash(self, exclude={"record_hash"})
        if self.record_hash != expected_hash:
            raise ValueError("Control evidence hash does not match its content")
        assert_public_payload_safe(self.model_dump(mode="json"))
        return self


def _create_probe_summary(
    *,
    case: BenchmarkCaseView,
    evidence_test_sha256: Sha256,
    passed: int,
    failed: int,
) -> ProbeExecutionSummary:
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.probe_execution_summary.v1",
        "case_id": case.case_id,
        "target_concept": case.target_concept,
        "evidence_test_sha256": evidence_test_sha256,
        "passed": passed,
        "failed": failed,
    }
    return ProbeExecutionSummary.model_validate(
        {**content, "summary_hash": canonical_sha256(content)}
    )


def _create_control_record(
    *,
    target_case_id: str,
    condition: BenchmarkCondition,
    source: ProbeExecutionSummary,
    control_spec_hash: Sha256,
    transformation: ControlTransformation,
    passed: int,
    failed: int,
    seed: int | None,
) -> ControlEvidenceRecord:
    content = {
        "schema_version": 1,
        "schema_id": "benchmark.control_evidence.v1",
        "target_case_id": target_case_id,
        "condition": condition,
        "source_case_id": source.case_id,
        "source_target_concept": source.target_concept,
        "passed": passed,
        "failed": failed,
        "source_summary_hash": source.summary_hash,
        "control_spec_hash": control_spec_hash,
        "transformation": transformation,
        "seed": seed,
    }
    return ControlEvidenceRecord.model_validate(
        {**content, "record_hash": canonical_sha256(content)}
    )


class ControlFactory:
    """Construct controls solely from a verified public manifest and probe summaries."""

    def __init__(self, benchmark_root: Path, manifest: PublicBenchmarkManifest) -> None:
        self._root = benchmark_root.resolve()
        self._cases = {case.case_id: case for case in manifest.cases}
        self._files = {entry.path: entry for entry in manifest.files}
        self._unrelated_specs = {
            reference: UnrelatedControlSpec.model_validate(self._read_control(reference))
            for reference in {case.unrelated_control_ref for case in manifest.cases}
        }
        self._corruption_specs = {
            reference: CorruptionControlSpec.model_validate(self._read_control(reference))
            for reference in {case.corruption_spec_ref for case in manifest.cases}
        }
        self._validate_assignments()

    def probe_summary(
        self,
        *,
        case_id: str,
        passed: int,
        failed: int,
    ) -> ProbeExecutionSummary:
        """Bind normalized pass/fail counts to the case's evidence-test identity."""

        case = self._case(case_id)
        evidence_test = self._file_entry(case.evidence_test_ref)
        if evidence_test.artifact_class is not PublicArtifactClass.EVIDENCE_TEST:
            raise ControlConstructionError(
                f"Evidence test has the wrong artifact class: {case.evidence_test_ref}"
            )
        return _create_probe_summary(
            case=case,
            evidence_test_sha256=evidence_test.sha256,
            passed=passed,
            failed=failed,
        )

    def unrelated(
        self,
        *,
        case_id: str,
        evidence_by_case: Mapping[str, ProbeExecutionSummary],
    ) -> ControlEvidenceRecord:
        """Use the preassigned different-concept evidence summary unchanged."""

        target = self._case(case_id)
        spec = self._unrelated_specs[target.unrelated_control_ref]
        source_case_id = spec.assignments[target.case_id]
        try:
            source = evidence_by_case[source_case_id]
        except KeyError as error:
            raise ControlConstructionError(
                f"Missing unrelated source evidence for case: {source_case_id}"
            ) from error
        self._validate_source_summary(self._case(source_case_id), source)
        return _create_control_record(
            target_case_id=target.case_id,
            condition=BenchmarkCondition.UNRELATED_PROBE,
            source=source,
            control_spec_hash=model_content_hash(spec),
            transformation=ControlTransformation.FROZEN_UNRELATED_ASSIGNMENT,
            passed=source.passed,
            failed=source.failed,
            seed=None,
        )

    def corrupted(
        self,
        *,
        case_id: str,
        evidence: ProbeExecutionSummary,
    ) -> ControlEvidenceRecord:
        """Invert pass/fail counts using the predeclared transformation and seed."""

        target = self._case(case_id)
        self._validate_source_summary(target, evidence)
        spec = self._corruption_specs[target.corruption_spec_ref]
        return _create_control_record(
            target_case_id=target.case_id,
            condition=BenchmarkCondition.CORRUPTED_PROBE,
            source=evidence,
            control_spec_hash=model_content_hash(spec),
            transformation=ControlTransformation.INVERT_PASS_FAIL_SUMMARY,
            passed=evidence.failed,
            failed=evidence.passed,
            seed=spec.seed,
        )

    def _validate_assignments(self) -> None:
        for reference, spec in self._unrelated_specs.items():
            expected_targets = {
                case.case_id
                for case in self._cases.values()
                if case.unrelated_control_ref == reference
            }
            if set(spec.assignments) != expected_targets:
                raise ControlConstructionError(
                    f"Unrelated assignments do not match referencing cases: {reference}"
                )
        for target in self._cases.values():
            spec = self._unrelated_specs[target.unrelated_control_ref]
            try:
                source = self._cases[spec.assignments[target.case_id]]
            except KeyError as error:
                raise ControlConstructionError(
                    f"Missing or unknown unrelated assignment for case: {target.case_id}"
                ) from error
            if source.case_id == target.case_id:
                raise ControlConstructionError(
                    f"Unrelated assignment points to the same case: {target.case_id}"
                )
            if source.target_concept == target.target_concept:
                raise ControlConstructionError(
                    f"Unrelated assignment uses the same target concept: {target.case_id}"
                )

    def _validate_source_summary(
        self,
        source_case: BenchmarkCaseView,
        summary: ProbeExecutionSummary,
    ) -> None:
        evidence_test = self._file_entry(source_case.evidence_test_ref)
        if evidence_test.artifact_class is not PublicArtifactClass.EVIDENCE_TEST:
            raise ControlConstructionError(
                f"Evidence test has the wrong artifact class: {source_case.evidence_test_ref}"
            )
        if summary.case_id != source_case.case_id:
            raise ControlConstructionError("Probe summary case does not match its source case")
        if summary.target_concept != source_case.target_concept:
            raise ControlConstructionError("Probe summary concept does not match its source case")
        if summary.evidence_test_sha256 != evidence_test.sha256:
            raise ControlConstructionError("Probe summary test hash does not match its source case")

    def _case(self, case_id: str) -> BenchmarkCaseView:
        try:
            return self._cases[case_id]
        except KeyError as error:
            raise ControlConstructionError(f"Unknown benchmark case: {case_id}") from error

    def _read_control(self, reference: str) -> dict[str, object]:
        entry = self._file_entry(reference)
        if entry.artifact_class is not PublicArtifactClass.CONTROL:
            raise ControlConstructionError(f"Artifact {reference} is not a control specification")
        path = (self._root / reference).resolve()
        if not path.is_relative_to(self._root):
            raise ControlConstructionError(f"Control path escapes benchmark root: {reference}")
        try:
            content = path.read_bytes()
        except OSError as error:
            raise ControlConstructionError(f"Could not read control: {reference}") from error
        if len(content) != entry.byte_size or file_sha256(content) != entry.sha256:
            raise ControlConstructionError(f"Control digest or size mismatch: {reference}")
        try:
            raw = yaml.safe_load(content)
        except yaml.YAMLError as error:
            raise ControlConstructionError(f"Control is not valid YAML: {reference}") from error
        if not isinstance(raw, dict):
            raise ControlConstructionError(f"Control must be a mapping: {reference}")
        return cast(dict[str, object], raw)

    def _file_entry(self, reference: str) -> PublicFileEntry:
        try:
            return self._files[reference]
        except KeyError as error:
            raise ControlConstructionError(
                f"Control dependency is absent from public inventory: {reference}"
            ) from error
