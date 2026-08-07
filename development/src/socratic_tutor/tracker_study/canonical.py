"""One-shot orchestration for the canonical glass-box tracker experiment."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Literal

import yaml
from pydantic import ValidationError, model_validator

from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import model_content_hash
from socratic_tutor.contracts import ContractModel
from socratic_tutor.tracker_study.analysis import (
    TrackerCanonicalAnalysisReport,
    run_canonical_analysis,
)
from socratic_tutor.tracker_study.analysis_spec import (
    load_tracker_study_analysis_specification,
)
from socratic_tutor.tracker_study.config import load_tracker_study_configuration
from socratic_tutor.tracker_study.simulation import (
    publish_canonical_matrix,
    simulate_verified_canonical_matrix,
)


class CanonicalExecutionError(ValueError):
    """The canonical execution plan or environment violates the frozen design."""


class CanonicalExecutionPlan(ContractModel):
    schema_version: Literal[1] = 1
    schema_id: Literal["tracker_study.canonical_execution_plan.v1"] = (
        "tracker_study.canonical_execution_plan.v1"
    )
    study_id: Literal["bayesian-evidence-robustness-v1"] = "bayesian-evidence-robustness-v1"
    plan_status: Literal["frozen_before_canonical_test"] = "frozen_before_canonical_test"
    configuration_path: Literal["configs/tracker-study/v1.yaml"] = "configs/tracker-study/v1.yaml"
    configuration_hash: Sha256
    analysis_specification_path: Literal["configs/tracker-study/v1-analysis.yaml"] = (
        "configs/tracker-study/v1-analysis.yaml"
    )
    analysis_specification_hash: Sha256
    pixi_lock_path: Literal["pixi.lock"] = "pixi.lock"
    output_root: Literal["artifacts/tracker-study/canonical-v1"] = (
        "artifacts/tracker-study/canonical-v1"
    )
    run_id: Literal["bayesian-evidence-robustness-v1-canonical"] = (
        "bayesian-evidence-robustness-v1-canonical"
    )
    split: Literal["test"] = "test"
    episodes_per_condition: Literal[500] = 500
    turns_per_episode: Literal[40] = 40
    canonical_test_run_limit: Literal[1] = 1
    exact_identical_retry_allowed: Literal[True] = True
    parameter_selection_after_test_allowed: Literal[False] = False
    external_model_data_used: Literal[False] = False
    human_data_used: Literal[False] = False
    plan_hash: Sha256

    @model_validator(mode="after")
    def validate_plan(self) -> CanonicalExecutionPlan:
        if self.plan_hash != model_content_hash(self, exclude={"plan_hash"}):
            raise ValueError("Canonical execution plan hash does not match its content")
        return self


def load_canonical_execution_plan(path: Path) -> CanonicalExecutionPlan:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise CanonicalExecutionError("Canonical execution plan must be a mapping")
        return CanonicalExecutionPlan.model_validate(raw)
    except CanonicalExecutionError:
        raise
    except (OSError, yaml.YAMLError, ValidationError) as error:
        raise CanonicalExecutionError(f"Could not load canonical execution plan: {path}") from error


def run_canonical_study(
    *,
    execution_plan_path: Path,
    code_revision: str,
) -> TrackerCanonicalAnalysisReport:
    """Execute the exact test matrix and frozen analysis described by one plan."""

    plan = load_canonical_execution_plan(execution_plan_path)
    project_root = execution_plan_path.resolve().parents[2]
    configuration = load_tracker_study_configuration(project_root / plan.configuration_path)
    specification = load_tracker_study_analysis_specification(
        project_root / plan.analysis_specification_path,
        configuration,
    )
    if configuration.study_id != plan.study_id:
        raise CanonicalExecutionError("Execution plan and tracker configuration differ")
    if configuration.configuration_hash != plan.configuration_hash:
        raise CanonicalExecutionError("Execution plan has the wrong configuration hash")
    if specification.analysis_specification_hash != plan.analysis_specification_hash:
        raise CanonicalExecutionError("Execution plan has the wrong analysis specification hash")
    if configuration.experiment.test_episodes_per_condition != plan.episodes_per_condition:
        raise CanonicalExecutionError("Execution plan has the wrong test episode count")
    if configuration.experiment.turns_per_episode != plan.turns_per_episode:
        raise CanonicalExecutionError("Execution plan has the wrong turn count")
    if len(code_revision) != 40 or any(
        character not in "0123456789abcdef" for character in code_revision
    ):
        raise CanonicalExecutionError("Canonical code revision must be a full lowercase Git SHA")
    _verify_committed_revision(project_root, code_revision)

    output_root = project_root / plan.output_root
    matrix, replay_hash = simulate_verified_canonical_matrix(configuration)
    publish_canonical_matrix(
        matrix,
        replay_content_hash=replay_hash,
        output_root=output_root,
        run_id=plan.run_id,
        code_revision=code_revision,
        configuration=configuration,
        canonical_execution_plan_hash=plan.plan_hash,
    )
    return run_canonical_analysis(
        simulation_manifest_path=output_root / "canonical_stress_matrix_manifest.json",
        configuration=configuration,
        specification=specification,
        pixi_lock_path=project_root / plan.pixi_lock_path,
        output_root=output_root,
        run_id=plan.run_id,
        analysis_code_revision=code_revision,
        canonical_execution_plan_hash=plan.plan_hash,
    )


def _verify_committed_revision(project_root: Path, expected_revision: str) -> None:
    try:
        actual_revision = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        relevant_status = subprocess.run(
            [
                "git",
                "-C",
                str(project_root),
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--",
                "src/socratic_tutor/tracker_study",
                "configs/tracker-study",
                "scripts/tracker_study_cli.py",
                "pixi.lock",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise CanonicalExecutionError("Could not verify the canonical Git revision") from error
    if actual_revision != expected_revision:
        raise CanonicalExecutionError(
            f"Canonical code revision is {actual_revision}, not {expected_revision}"
        )
    if relevant_status:
        raise CanonicalExecutionError(
            "Canonical tracker code or frozen inputs contain uncommitted changes"
        )
