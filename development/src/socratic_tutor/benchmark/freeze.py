"""Prepare a content-addressed benchmark freeze without revealing outcomes."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from socratic_tutor.benchmark.analysis_spec import (
    analysis_specification_hash,
    load_analysis_specification,
)
from socratic_tutor.benchmark.artifacts import write_immutable_json
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.evaluator.loader import load_and_verify_manifest
from socratic_tutor.benchmark.evaluator.models import (
    ManifestStatus,
    authored_manifest_content_hash,
)
from socratic_tutor.benchmark.hashing import model_content_hash
from socratic_tutor.benchmark.readiness import BenchmarkReadinessReport
from socratic_tutor.contracts import ContractModel


class BenchmarkFreezePreparation(ContractModel):
    """Evidence that one draft manifest is ready for a separate freeze edit."""

    schema_version: Literal[1] = 1
    schema_id: Literal["benchmark.freeze_preparation.v1"] = "benchmark.freeze_preparation.v1"
    benchmark_version: str = Field(min_length=1)
    source_draft_manifest_hash: Sha256
    frozen_at_utc: datetime
    frozen_manifest_hash: Sha256
    readiness_report_hash: Sha256
    analysis_specification_hash: Sha256
    case_count: int = Field(ge=1)
    preparation_hash: Sha256

    @model_validator(mode="after")
    def validate_preparation(self) -> BenchmarkFreezePreparation:
        if self.frozen_at_utc.tzinfo is None or self.frozen_at_utc.utcoffset() != UTC.utcoffset(
            self.frozen_at_utc
        ):
            raise ValueError("Freeze preparation time must be UTC-aware")
        if self.preparation_hash != model_content_hash(self, exclude={"preparation_hash"}):
            raise ValueError("Freeze preparation hash does not match its content")
        return self


def prepare_benchmark_freeze(
    *,
    manifest_path: Path,
    readiness_report_path: Path,
    analysis_specification_path: Path,
    frozen_at_utc: datetime,
    output_path: Path,
) -> BenchmarkFreezePreparation:
    """Validate freeze inputs and publish the required manifest digest.

    The caller must then make the manifest's status, timestamp, and digest match this
    preparation. Keeping that final edit explicit prevents an accidental freeze.
    """

    if frozen_at_utc.tzinfo is None or frozen_at_utc.utcoffset() != UTC.utcoffset(frozen_at_utc):
        raise ValueError("Freeze time must be an explicit UTC timestamp")

    manifest = load_and_verify_manifest(manifest_path)
    if manifest.status is not ManifestStatus.DRAFT:
        raise ValueError("Only a draft benchmark manifest can be prepared for freezing")

    readiness = _load_readiness_report(readiness_report_path)
    # Readiness records the complete draft object, while the frozen digest omits only
    # its self-referential manifest_hash field. They intentionally serve different roles.
    source_draft_manifest_hash = model_content_hash(manifest)
    if not readiness.gate_passed:
        raise ValueError("A failed readiness report cannot prepare a benchmark freeze")
    if readiness.benchmark_version != manifest.benchmark_version:
        raise ValueError("Readiness report belongs to another benchmark version")
    if readiness.source_manifest_hash != source_draft_manifest_hash:
        raise ValueError("Readiness report belongs to another manifest revision")

    analysis_specification = load_analysis_specification(analysis_specification_path)
    if analysis_specification.benchmark_version != manifest.benchmark_version:
        raise ValueError("Analysis specification belongs to another benchmark version")

    frozen_manifest = manifest.model_copy(
        update={
            "status": ManifestStatus.FROZEN,
            "frozen_at_utc": frozen_at_utc,
            "manifest_hash": None,
        }
    )
    frozen_manifest_hash = authored_manifest_content_hash(frozen_manifest)
    content: dict[str, object] = {
        "schema_version": 1,
        "schema_id": "benchmark.freeze_preparation.v1",
        "benchmark_version": manifest.benchmark_version,
        "source_draft_manifest_hash": source_draft_manifest_hash,
        "frozen_at_utc": frozen_at_utc,
        "frozen_manifest_hash": frozen_manifest_hash,
        "readiness_report_hash": readiness.report_hash,
        "analysis_specification_hash": analysis_specification_hash(analysis_specification),
        "case_count": len(manifest.cases),
    }
    draft = BenchmarkFreezePreparation.model_construct(
        _fields_set=set(content), **content, preparation_hash="0" * 64
    )
    preparation = BenchmarkFreezePreparation.model_validate(
        {**content, "preparation_hash": model_content_hash(draft, exclude={"preparation_hash"})}
    )
    write_immutable_json(output_path, preparation)
    return preparation


def _load_readiness_report(path: Path) -> BenchmarkReadinessReport:
    try:
        return BenchmarkReadinessReport.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"Could not read readiness report: {path}") from error
