"""Content-addressed projections separating decision and evaluator data."""

from datetime import UTC, datetime

from socratic_tutor.benchmark.evaluator.models import (
    ArtifactClass,
    AuthoredBenchmarkManifest,
    EvaluatorBenchmarkManifest,
)
from socratic_tutor.benchmark.hashing import model_content_hash
from socratic_tutor.benchmark.public import (
    PublicArtifactClass,
    PublicBenchmarkManifest,
    PublicFileEntry,
    assert_public_payload_safe,
)

_PUBLIC_ARTIFACTS: dict[ArtifactClass, PublicArtifactClass] = {
    ArtifactClass.PUBLIC_FIXTURE: PublicArtifactClass.PUBLIC_FIXTURE,
    ArtifactClass.EVIDENCE_PROBE: PublicArtifactClass.EVIDENCE_PROBE,
    ArtifactClass.EVIDENCE_TEST: PublicArtifactClass.EVIDENCE_TEST,
    ArtifactClass.INITIAL_TRACKER: PublicArtifactClass.INITIAL_TRACKER,
    ArtifactClass.CONTROL: PublicArtifactClass.CONTROL,
    ArtifactClass.SPLIT: PublicArtifactClass.SPLIT,
    ArtifactClass.LIMITATIONS: PublicArtifactClass.LIMITATIONS,
    ArtifactClass.PROMPT: PublicArtifactClass.PROMPT,
}


def authored_manifest_hash(manifest: AuthoredBenchmarkManifest) -> str:
    """Hash authored content without the self-referential manifest hash field."""

    return model_content_hash(manifest, exclude={"manifest_hash"})


def _source_hash(manifest: AuthoredBenchmarkManifest) -> str:
    calculated = authored_manifest_hash(manifest)
    if manifest.manifest_hash is not None and manifest.manifest_hash != calculated:
        raise ValueError("Authored manifest hash does not match its content")
    return calculated


def project_public_manifest(
    manifest: AuthoredBenchmarkManifest,
    *,
    projected_at_utc: datetime | None = None,
) -> PublicBenchmarkManifest:
    """Produce and recursively inspect the decision-time manifest."""

    source_hash = _source_hash(manifest)
    public_files = tuple(
        PublicFileEntry(
            path=entry.path,
            artifact_class=_PUBLIC_ARTIFACTS[entry.artifact_class],
            schema_id=entry.schema_id,
            sha256=entry.sha256,
            byte_size=entry.byte_size,
        )
        for entry in manifest.files
        if entry.artifact_class in _PUBLIC_ARTIFACTS
    )
    draft = PublicBenchmarkManifest(
        benchmark_version=manifest.benchmark_version,
        source_manifest_hash=source_hash,
        projection_hash="0" * 64,
        projected_at_utc=projected_at_utc or datetime.now(UTC),
        conditions=manifest.conditions,
        cases=tuple(item.public for item in manifest.cases),
        splits=manifest.splits,
        files=public_files,
    )
    projection_hash = model_content_hash(
        draft,
        exclude={"projection_hash", "projected_at_utc"},
    )
    result = draft.model_copy(update={"projection_hash": projection_hash})
    assert_public_payload_safe(result.model_dump(mode="json"))
    return result


def project_evaluator_manifest(
    manifest: AuthoredBenchmarkManifest,
    public_manifest: PublicBenchmarkManifest,
    *,
    projected_at_utc: datetime | None = None,
) -> EvaluatorBenchmarkManifest:
    """Produce the private projection linked to the exact public projection."""

    source_hash = _source_hash(manifest)
    if public_manifest.source_manifest_hash != source_hash:
        raise ValueError("Public projection belongs to a different authored manifest")
    private_files = tuple(
        entry for entry in manifest.files if entry.artifact_class not in _PUBLIC_ARTIFACTS
    )
    draft = EvaluatorBenchmarkManifest(
        benchmark_version=manifest.benchmark_version,
        source_manifest_hash=source_hash,
        public_projection_hash=public_manifest.projection_hash,
        projection_hash="0" * 64,
        projected_at_utc=projected_at_utc or datetime.now(UTC),
        criteria=tuple(item.criterion for item in manifest.cases),
        reviews=manifest.reviews,
        files=private_files,
    )
    projection_hash = model_content_hash(
        draft,
        exclude={"projection_hash", "projected_at_utc"},
    )
    return draft.model_copy(update={"projection_hash": projection_hash})
