"""One-command offline reproduction of the complete CSEDM companion study."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from socratic_tutor.benchmark.artifacts import artifact_locations, write_immutable_json
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import canonical_sha256, file_sha256
from socratic_tutor.contracts import ContractModel
from socratic_tutor.csedm_study.adapter import build_csedm_adapter
from socratic_tutor.csedm_study.analysis import publish_uncertainty_analysis
from socratic_tutor.csedm_study.analysis_plan import load_analysis_plan
from socratic_tutor.csedm_study.features import CSEDMAdapterError
from socratic_tutor.csedm_study.inventory import create_inventory, load_inventory_specification
from socratic_tutor.csedm_study.prediction import publish_out_of_fold_predictions
from socratic_tutor.csedm_study.publication import publish_csedm_results
from socratic_tutor.repository_state import current_clean_revision

_STAGES = ("inventory", "adapter", "prediction", "analysis", "publication")


class CSEDMPipelineManifest(ContractModel):
    """Lineage for one complete local reproduction."""

    schema_id: Literal["csedm.pipeline_manifest.v1"] = "csedm.pipeline_manifest.v1"
    schema_version: Literal[1] = 1
    study_id: Literal["csedm-uncertainty-triage-v1"] = "csedm-uncertainty-triage-v1"
    code_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    pixi_lock_sha256: Sha256
    archive_sha256: Sha256
    inventory_specification_sha256: Sha256
    analysis_plan_hash: Sha256
    inventory_report_hash: Sha256
    adapter_manifest_hash: Sha256
    prediction_manifest_hash: Sha256
    analysis_manifest_hash: Sha256
    publication_manifest_hash: Sha256
    completed_stages: tuple[str, ...]
    network_calls: Literal[0]
    sandbox_calls: Literal[0]
    manifest_hash: Sha256

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        if self.completed_stages != _STAGES:
            raise ValueError("CSEDM pipeline stages are missing or out of order")
        expected = canonical_sha256(self.model_dump(mode="json", exclude={"manifest_hash"}))
        if self.manifest_hash != expected:
            raise ValueError("CSEDM pipeline manifest hash does not match its content")
        return self


def reproduce_csedm_pipeline(
    *,
    project_root: Path,
    inventory_specification_path: Path,
    analysis_plan_path: Path,
    pixi_lock_path: Path,
    output_root: Path,
    code_revision: str,
) -> CSEDMPipelineManifest:
    """Rebuild every M7C stage from the untouched local archive without network calls."""

    inventory_root = output_root / "inventory"
    adapter_root = output_root / "adapter"
    prediction_root = output_root / "predictions"
    analysis_root = output_root / "analysis"
    publication_root = output_root / "publication"

    inventory = create_inventory(
        inventory_specification_path,
        project_root=project_root,
        output_root=inventory_root,
    )
    adapter = build_csedm_adapter(
        inventory_specification_path,
        analysis_plan_path,
        inventory_root / "inventory_report.json",
        project_root=project_root,
        output_root=adapter_root,
    )
    predictions = publish_out_of_fold_predictions(
        adapter_root=adapter_root,
        analysis_plan_path=analysis_plan_path,
        inventory_specification_path=inventory_specification_path,
        project_root=project_root,
        pixi_lock_path=pixi_lock_path,
        output_root=prediction_root,
        code_revision=code_revision,
    )
    analysis = publish_uncertainty_analysis(
        prediction_root=prediction_root,
        analysis_plan_path=analysis_plan_path,
        pixi_lock_path=pixi_lock_path,
        output_root=analysis_root,
        code_revision=code_revision,
    )
    publication = publish_csedm_results(
        analysis_root=analysis_root,
        pixi_lock_path=pixi_lock_path,
        output_root=publication_root,
        code_revision=code_revision,
    )

    specification = load_inventory_specification(inventory_specification_path)
    plan = load_analysis_plan(analysis_plan_path)
    payload = {
        "schema_id": "csedm.pipeline_manifest.v1",
        "schema_version": 1,
        "study_id": plan.study_id,
        "code_revision": code_revision,
        "pixi_lock_sha256": file_sha256(pixi_lock_path.read_bytes()),
        "archive_sha256": specification.archive_sha256,
        "inventory_specification_sha256": file_sha256(inventory_specification_path.read_bytes()),
        "analysis_plan_hash": plan.plan_hash,
        "inventory_report_hash": inventory.report_hash,
        "adapter_manifest_hash": adapter.manifest_hash,
        "prediction_manifest_hash": predictions.manifest_hash,
        "analysis_manifest_hash": analysis.manifest_hash,
        "publication_manifest_hash": publication.manifest_hash,
        "completed_stages": _STAGES,
        "network_calls": 0,
        "sandbox_calls": 0,
    }
    payload["manifest_hash"] = canonical_sha256(payload)
    manifest = CSEDMPipelineManifest.model_validate(payload)
    write_immutable_json(output_root / "pipeline_manifest.json", manifest)
    return manifest


def build_parser(project_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduce the complete CSEDM study offline")
    parser.add_argument(
        "--inventory-specification",
        type=Path,
        default=project_root / "configs" / "csedm-study" / "v1-inventory.yaml",
    )
    parser.add_argument(
        "--analysis-plan",
        type=Path,
        default=project_root / "configs" / "csedm-study" / "v1-analysis.yaml",
    )
    parser.add_argument("--pixi-lock", type=Path, default=project_root / "pixi.lock")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--code-revision")
    return parser


def main(argv: list[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parents[3]
    args = build_parser(project_root).parse_args(argv)
    try:
        revision = args.code_revision or current_clean_revision(
            project_root,
            dirty_message="Commit tracked changes before reproducing the CSEDM study",
            untracked_files="no",
            operation_name="CSEDM study reproduction",
            error_factory=CSEDMAdapterError,
        )
        output_root = args.output_root or (
            project_root / "artifacts" / "csedm-study" / f"reproduction-{revision[:7]}"
        )
        manifest = reproduce_csedm_pipeline(
            project_root=project_root,
            inventory_specification_path=args.inventory_specification,
            analysis_plan_path=args.analysis_plan,
            pixi_lock_path=args.pixi_lock,
            output_root=output_root,
            code_revision=revision,
        )
    except (CSEDMAdapterError, OSError, ValueError) as error:
        print(
            json.dumps(
                {
                    "command": "csedm-reproduce",
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "status": "error",
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "artifact_locations": artifact_locations(output_root),
                "command": "csedm-reproduce",
                "result": manifest.model_dump(mode="json"),
                "status": "ok",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0
