"""Evaluator-only builder for isolated criterion generation requests."""

from pathlib import Path

from socratic_tutor.benchmark.evaluator.models import (
    ArtifactClass,
    CriterionSpec,
    EvaluatorBenchmarkManifest,
    FileInventoryEntry,
)
from socratic_tutor.benchmark.generation import (
    CriterionTaskPayload,
    GenerationRequestSpec,
    StudentGenerationRequest,
    create_student_generation_request,
)
from socratic_tutor.benchmark.hashing import file_sha256


class CriterionArtifactReadError(ValueError):
    """An evaluator artifact could not be read with its declared identity."""


class CriterionChannelRequestBuilder:
    """Build criterion requests from evaluator capabilities only."""

    def __init__(self, benchmark_root: Path, manifest: EvaluatorBenchmarkManifest) -> None:
        self._root = benchmark_root.resolve()
        self._manifest = manifest
        self._criteria = {criterion.case_id: criterion for criterion in manifest.criteria}
        self._files = {entry.path: entry for entry in manifest.files}

    def build_criterion(
        self,
        *,
        case_id: str,
        spec: GenerationRequestSpec,
    ) -> StudentGenerationRequest:
        """Build a fresh request containing only the held-out transfer probe."""

        criterion = self._criterion(case_id)
        probe = self._read_text(
            criterion.criterion_prompt_ref,
            expected_class=ArtifactClass.CRITERION_PROBE,
        )
        payload = CriterionTaskPayload(
            benchmark_version=self._manifest.benchmark_version,
            case_id=criterion.case_id,
            criterion_probe_id=criterion.criterion_probe_id,
            criterion_probe=probe,
        )
        return create_student_generation_request(spec=spec, payload=payload)

    def _criterion(self, case_id: str) -> CriterionSpec:
        try:
            return self._criteria[case_id]
        except KeyError as error:
            raise CriterionArtifactReadError(
                f"Unknown evaluator benchmark case: {case_id}"
            ) from error

    def _read_text(self, reference: str, *, expected_class: ArtifactClass) -> str:
        entry = self._file_entry(reference)
        if entry.artifact_class is not expected_class:
            actual_class = entry.artifact_class.value
            raise CriterionArtifactReadError(
                f"Artifact {reference} is {actual_class}, expected {expected_class.value}"
            )
        path = (self._root / reference).resolve()
        if not path.is_relative_to(self._root):
            raise CriterionArtifactReadError(
                f"Artifact resolves outside benchmark root: {reference}"
            )
        try:
            content = path.read_bytes()
        except OSError as error:
            raise CriterionArtifactReadError(
                f"Could not read criterion artifact: {reference}"
            ) from error
        if len(content) != entry.byte_size or file_sha256(content) != entry.sha256:
            raise CriterionArtifactReadError(
                f"Criterion artifact digest or size mismatch: {reference}"
            )
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as error:
            raise CriterionArtifactReadError(
                f"Criterion artifact is not UTF-8: {reference}"
            ) from error
        if not text.strip():
            raise CriterionArtifactReadError(f"Criterion artifact is empty: {reference}")
        return text

    def _file_entry(self, reference: str) -> FileInventoryEntry:
        try:
            return self._files[reference]
        except KeyError as error:
            raise CriterionArtifactReadError(
                f"Criterion artifact is absent from evaluator inventory: {reference}"
            ) from error
