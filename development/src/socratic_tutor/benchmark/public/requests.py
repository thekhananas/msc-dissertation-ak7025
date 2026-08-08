"""Decision-phase builders for isolated public and evidence requests."""

from pathlib import Path

from socratic_tutor.benchmark.generation import (
    EvidenceTaskPayload,
    GenerationRequestSpec,
    PublicTaskPayload,
    StudentGenerationRequest,
    create_student_generation_request,
)
from socratic_tutor.benchmark.hashing import file_sha256
from socratic_tutor.benchmark.public.models import (
    BenchmarkCaseView,
    PublicArtifactClass,
    PublicBenchmarkManifest,
    PublicFileEntry,
)
from socratic_tutor.benchmark.public.safety import assert_public_payload_safe


class PublicArtifactReadError(ValueError):
    """A public artifact could not be read with its declared identity."""


class PublicChannelRequestBuilder:
    """Build public/evidence requests without an evaluator dependency."""

    def __init__(self, benchmark_root: Path, manifest: PublicBenchmarkManifest) -> None:
        self._root = benchmark_root.resolve()
        self._manifest = manifest
        self._cases = {case.case_id: case for case in manifest.cases}
        self._files = {entry.path: entry for entry in manifest.files}

    def build_public(
        self,
        *,
        case_id: str,
        spec: GenerationRequestSpec,
    ) -> StudentGenerationRequest:
        """Build a request containing only the authored public interaction."""

        case = self._case(case_id)
        interaction = self._read_text(
            case.public_fixture_ref,
            expected_class=PublicArtifactClass.PUBLIC_FIXTURE,
        )
        payload = PublicTaskPayload(
            benchmark_version=self._manifest.benchmark_version,
            case_id=case.case_id,
            task_family=case.task_family,
            target_concept=case.target_concept,
            public_interaction=interaction,
        )
        request = create_student_generation_request(spec=spec, payload=payload)
        assert_public_payload_safe(request.model_dump(mode="json"))
        return request

    def build_evidence(
        self,
        *,
        case_id: str,
        spec: GenerationRequestSpec,
    ) -> StudentGenerationRequest:
        """Build a fresh request containing only the active evidence probe."""

        case = self._case(case_id)
        probe = self._read_text(
            case.evidence_probe_ref,
            expected_class=PublicArtifactClass.EVIDENCE_PROBE,
        )
        payload = EvidenceTaskPayload(
            benchmark_version=self._manifest.benchmark_version,
            case_id=case.case_id,
            task_family=case.task_family,
            target_concept=case.target_concept,
            evidence_probe=probe,
        )
        request = create_student_generation_request(spec=spec, payload=payload)
        assert_public_payload_safe(request.model_dump(mode="json"))
        return request

    def _case(self, case_id: str) -> BenchmarkCaseView:
        try:
            return self._cases[case_id]
        except KeyError as error:
            raise PublicArtifactReadError(f"Unknown public benchmark case: {case_id}") from error

    def _read_text(self, reference: str, *, expected_class: PublicArtifactClass) -> str:
        entry = self._file_entry(reference)
        if entry.artifact_class is not expected_class:
            actual_class = entry.artifact_class.value
            raise PublicArtifactReadError(
                f"Artifact {reference} is {actual_class}, expected {expected_class.value}"
            )
        path = (self._root / reference).resolve()
        if not path.is_relative_to(self._root):
            raise PublicArtifactReadError(f"Artifact resolves outside benchmark root: {reference}")
        try:
            content = path.read_bytes()
        except OSError as error:
            raise PublicArtifactReadError(f"Could not read public artifact: {reference}") from error
        if len(content) != entry.byte_size or file_sha256(content) != entry.sha256:
            raise PublicArtifactReadError(f"Public artifact digest or size mismatch: {reference}")
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as error:
            raise PublicArtifactReadError(f"Public artifact is not UTF-8: {reference}") from error
        if not text.strip():
            raise PublicArtifactReadError(f"Public artifact is empty: {reference}")
        return text

    def _file_entry(self, reference: str) -> PublicFileEntry:
        try:
            return self._files[reference]
        except KeyError as error:
            raise PublicArtifactReadError(
                f"Public artifact is absent from the projected inventory: {reference}"
            ) from error
