"""Paired development runner for the frozen calibration-seed sensitivity check."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, ValidationError, model_validator

from socratic_tutor.acquisition_study.budget_curve import BudgetPolicyResult
from socratic_tutor.acquisition_study.budget_curve_runner import (
    BudgetCurveEpisodeMetric,
    compact_budget_policy_result,
)
from socratic_tutor.acquisition_study.calibration import estimate_probe_reliability
from socratic_tutor.acquisition_study.contracts import PolicyId
from socratic_tutor.acquisition_study.evaluation import (
    build_primary_non_oracle_policies,
    evaluate_policy_episode,
)
from socratic_tutor.acquisition_study.plan import (
    AcquisitionAnalysisSpecification,
    AcquisitionEnvironmentSpecification,
    EvaluationEnvironmentId,
)
from socratic_tutor.acquisition_study.primary_analysis import (
    load_verified_development_policy_matrix,
)
from socratic_tutor.acquisition_study.simulation import generate_acquisition_episode
from socratic_tutor.benchmark.artifacts import write_immutable_bytes, write_immutable_json
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import (
    canonical_json_bytes,
    canonical_sha256,
    file_sha256,
    model_content_hash,
)
from socratic_tutor.contracts import ContractModel

_POLICY_ORDER = (
    PolicyId.RELIABILITY_AWARE_BOUNDED,
    PolicyId.SEEDED_RANDOM_BOUNDED,
    PolicyId.UNCERTAINTY_ONLY_BOUNDED,
    PolicyId.PLUG_IN_EVSI_BOUNDED,
)


class CalibrationSensitivityError(ValueError):
    """A calibration-sensitivity source or result violates the frozen design."""


class SensitivityCalibrationRecord(ContractModel):
    """One predeclared calibration seed and its resulting posterior hash."""

    calibration_seed: int = Field(ge=0)
    calibration_hash: Sha256


class CalibrationSensitivityEpisodeMetric(ContractModel):
    """Compact policy result under one alternative calibration sample."""

    calibration_seed: int = Field(ge=0)
    calibration_hash: Sha256
    metric: BudgetCurveEpisodeMetric


class DevelopmentCalibrationSensitivityMatrix(ContractModel):
    """Primary-policy metrics across all predeclared calibration seeds."""

    schema_version: Literal[1] = 1
    schema_id: Literal["acquisition_study.development_calibration_sensitivity_matrix.v1"] = (
        "acquisition_study.development_calibration_sensitivity_matrix.v1"
    )
    study_id: Literal["reliability-aware-probing-v1"] = "reliability-aware-probing-v1"
    split: Literal["development"] = "development"
    result_scope: Literal["development_diagnostic_not_canonical"] = (
        "development_diagnostic_not_canonical"
    )
    sensitivity_id: Literal[
        "repeat_complete_analysis_over_twenty_predeclared_calibration_seeds"
    ] = "repeat_complete_analysis_over_twenty_predeclared_calibration_seeds"
    source_manifest_hash: Sha256
    source_primary_calibration_hash: Sha256
    environment_specification_hash: Sha256
    analysis_plan_hash: Sha256
    calibrations: tuple[SensitivityCalibrationRecord, ...] = Field(
        min_length=20,
        max_length=20,
    )
    episodes_per_environment: int = Field(ge=1)
    candidates_per_episode: int = Field(ge=1)
    fixed_probe_budget: int = Field(ge=1)
    fixed_budget_fraction: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    environment_count: Literal[7] = 7
    policy_ids: tuple[PolicyId, ...] = Field(min_length=4, max_length=4)
    row_count: int = Field(ge=1)
    rows: tuple[CalibrationSensitivityEpisodeMetric, ...]
    privileged_source_reproduction_verified: Literal[True] = True
    external_model_call_count: Literal[0] = 0
    sandbox_call_count: Literal[0] = 0
    human_record_count: Literal[0] = 0
    canonical_claim_allowed: Literal[False] = False
    access_scope: Literal["simulator_restricted"] = "simulator_restricted"

    @model_validator(mode="after")
    def validate_matrix(self) -> Self:
        if self.environment_count != len(EvaluationEnvironmentId):
            raise ValueError("Calibration sensitivity does not contain every environment")
        if self.policy_ids != _POLICY_ORDER:
            raise ValueError("Calibration sensitivity differs from the primary policy order")
        if not math.isclose(self.fixed_budget_fraction, 0.5, abs_tol=1e-12):
            raise ValueError("Calibration sensitivity differs from the primary budget")
        if self.fixed_probe_budget != self.candidates_per_episode * self.fixed_budget_fraction:
            raise ValueError("Calibration-sensitivity probe count differs from its budget")
        calibration_seeds = tuple(item.calibration_seed for item in self.calibrations)
        if len(set(calibration_seeds)) != len(calibration_seeds):
            raise ValueError("Calibration-sensitivity seeds must be unique")
        calibration_hashes = {
            item.calibration_seed: item.calibration_hash for item in self.calibrations
        }
        expected_count = (
            len(self.calibrations)
            * self.environment_count
            * self.episodes_per_environment
            * len(self.policy_ids)
        )
        if self.row_count != expected_count or len(self.rows) != expected_count:
            raise ValueError("Calibration-sensitivity row count is inconsistent")
        expected_keys = tuple(
            (seed, environment_id, episode_index, policy_id)
            for seed in calibration_seeds
            for environment_id in EvaluationEnvironmentId
            for episode_index in range(self.episodes_per_environment)
            for policy_id in self.policy_ids
        )
        actual_keys = tuple(
            (
                row.calibration_seed,
                row.metric.environment_id,
                row.metric.episode_index,
                row.metric.policy_id,
            )
            for row in self.rows
        )
        if actual_keys != expected_keys:
            raise ValueError("Calibration-sensitivity order or coverage differs")
        if any(
            row.calibration_hash != calibration_hashes[row.calibration_seed]
            or row.metric.candidate_count != self.candidates_per_episode
            or row.metric.selected_probe_count != self.fixed_probe_budget
            or not math.isclose(
                row.metric.budget_fraction,
                self.fixed_budget_fraction,
                abs_tol=1e-12,
            )
            for row in self.rows
        ):
            raise ValueError("Calibration-sensitivity rows use inconsistent inputs")
        return self

    @property
    def content_hash(self) -> Sha256:
        return canonical_sha256(self)


class DevelopmentCalibrationSensitivityManifest(ContractModel):
    """Self-verifying record of one development calibration-sensitivity matrix."""

    schema_version: Literal[1] = 1
    schema_id: Literal["acquisition_study.development_calibration_sensitivity_manifest.v1"] = (
        "acquisition_study.development_calibration_sensitivity_manifest.v1"
    )
    study_id: Literal["reliability-aware-probing-v1"] = "reliability-aware-probing-v1"
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,99}$")
    code_revision: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    split: Literal["development"] = "development"
    result_scope: Literal["development_diagnostic_not_canonical"] = (
        "development_diagnostic_not_canonical"
    )
    access_scope: Literal["simulator_restricted"] = "simulator_restricted"
    sensitivity_id: Literal[
        "repeat_complete_analysis_over_twenty_predeclared_calibration_seeds"
    ] = "repeat_complete_analysis_over_twenty_predeclared_calibration_seeds"
    source_manifest_hash: Sha256
    source_primary_calibration_hash: Sha256
    environment_specification_hash: Sha256
    analysis_plan_hash: Sha256
    calibrations: tuple[SensitivityCalibrationRecord, ...] = Field(
        min_length=20,
        max_length=20,
    )
    pixi_lock_sha256: Sha256
    result_file: Literal["development_calibration_sensitivity.jsonl"] = (
        "development_calibration_sensitivity.jsonl"
    )
    result_file_sha256: Sha256
    matrix_content_hash: Sha256
    replay_content_hash: Sha256
    deterministic_replay_verified: Literal[True] = True
    privileged_source_reproduction_verified: Literal[True] = True
    episodes_per_environment: int = Field(ge=1)
    candidates_per_episode: int = Field(ge=1)
    fixed_probe_budget: int = Field(ge=1)
    environment_count: Literal[7] = 7
    policy_count: Literal[4] = 4
    row_count: int = Field(ge=1)
    case_prediction_count: int = Field(ge=1)
    selected_probe_count: int = Field(ge=1)
    external_model_call_count: Literal[0] = 0
    sandbox_call_count: Literal[0] = 0
    human_record_count: Literal[0] = 0
    canonical_claim_allowed: Literal[False] = False
    manifest_hash: Sha256

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        expected_rows = (
            len(self.calibrations)
            * self.environment_count
            * self.episodes_per_environment
            * self.policy_count
        )
        if self.row_count != expected_rows:
            raise ValueError("Calibration-sensitivity manifest row count is inconsistent")
        if self.case_prediction_count != self.row_count * self.candidates_per_episode:
            raise ValueError("Calibration-sensitivity manifest case count is inconsistent")
        if self.selected_probe_count != self.row_count * self.fixed_probe_budget:
            raise ValueError("Calibration-sensitivity manifest probe count is inconsistent")
        if self.matrix_content_hash != self.replay_content_hash:
            raise ValueError("Calibration-sensitivity replay hash differs")
        if self.manifest_hash != model_content_hash(self, exclude={"manifest_hash"}):
            raise ValueError("Calibration-sensitivity manifest hash does not match its content")
        return self


def run_verified_development_calibration_sensitivity(
    source_manifest_path: Path,
    *,
    specification: AcquisitionEnvironmentSpecification,
    analysis: AcquisitionAnalysisSpecification,
) -> tuple[DevelopmentCalibrationSensitivityMatrix, Sha256]:
    """Run the calibration sensitivity twice and require exact agreement."""

    first = run_development_calibration_sensitivity(
        source_manifest_path,
        specification=specification,
        analysis=analysis,
    )
    replay = run_development_calibration_sensitivity(
        source_manifest_path,
        specification=specification,
        analysis=analysis,
    )
    if first.content_hash != replay.content_hash:
        raise CalibrationSensitivityError("Development calibration-sensitivity replay differs")
    return first, replay.content_hash


def run_development_calibration_sensitivity(
    source_manifest_path: Path,
    *,
    specification: AcquisitionEnvironmentSpecification,
    analysis: AcquisitionAnalysisSpecification,
) -> DevelopmentCalibrationSensitivityMatrix:
    """Repeat primary development policy evaluation over all frozen calibration seeds."""

    source_manifest, source_matrix = load_verified_development_policy_matrix(
        source_manifest_path,
        specification=specification,
        analysis=analysis,
    )
    if (
        analysis.secondary.calibration_seed_sensitivity
        != "repeat_complete_analysis_over_twenty_predeclared_calibration_seeds"
    ):
        raise CalibrationSensitivityError("Analysis specifies another sensitivity check")

    rows: list[CalibrationSensitivityEpisodeMetric] = []
    calibrations: list[SensitivityCalibrationRecord] = []
    for seed in specification.calibration.sensitivity_seeds:
        calibration = estimate_probe_reliability(specification, seed=seed)
        calibrations.append(
            SensitivityCalibrationRecord(
                calibration_seed=seed,
                calibration_hash=calibration.calibration_hash,
            )
        )
        policies = build_primary_non_oracle_policies(analysis)
        for source in source_matrix.comparisons:
            policy_episode, privileged_episode = generate_acquisition_episode(
                specification,
                calibration,
                environment_id=source.environment_id,
                episode_index=source.episode_index,
                probe_budget=analysis.primary.exact_selected_probes_per_episode,
            )
            if model_content_hash(privileged_episode) != source.privileged_episode_hash:
                raise CalibrationSensitivityError(
                    "Calibration sensitivity changed the hidden evaluation episode"
                )
            for policy in policies:
                result = evaluate_policy_episode(
                    policy,
                    policy_episode,
                    privileged_episode,
                    kappa=analysis.policies.bounded_log_likelihood_kappa,
                    classification_threshold=analysis.primary.classification_threshold,
                    probability_floor=analysis.secondary.probability_floor,
                )
                rows.append(
                    CalibrationSensitivityEpisodeMetric(
                        calibration_seed=seed,
                        calibration_hash=calibration.calibration_hash,
                        metric=compact_budget_policy_result(
                            BudgetPolicyResult(
                                budget_fraction=analysis.primary.primary_budget_fraction,
                                selected_probe_count=result.burden.selected_probe_count,
                                result=result,
                            ),
                            analysis.secondary.ece_bin_count,
                        ),
                    )
                )

    return DevelopmentCalibrationSensitivityMatrix(
        source_manifest_hash=source_manifest.manifest_hash,
        source_primary_calibration_hash=source_manifest.calibration_hash,
        environment_specification_hash=specification.specification_hash,
        analysis_plan_hash=analysis.plan_hash,
        calibrations=tuple(calibrations),
        episodes_per_environment=source_matrix.episodes_per_environment,
        candidates_per_episode=specification.episodes.candidates_per_episode,
        fixed_probe_budget=analysis.primary.exact_selected_probes_per_episode,
        fixed_budget_fraction=analysis.primary.primary_budget_fraction,
        policy_ids=_POLICY_ORDER,
        row_count=len(rows),
        rows=tuple(rows),
    )


def publish_development_calibration_sensitivity(
    matrix: DevelopmentCalibrationSensitivityMatrix,
    *,
    replay_content_hash: Sha256,
    output_root: Path,
    run_id: str,
    code_revision: str,
    pixi_lock_path: Path,
) -> DevelopmentCalibrationSensitivityManifest:
    """Write compact calibration-sensitivity rows and their immutable manifest."""

    if matrix.content_hash != replay_content_hash:
        raise CalibrationSensitivityError("Sensitivity matrix and replay hashes differ")
    rows_bytes = b"".join(canonical_json_bytes(row) + b"\n" for row in matrix.rows)
    try:
        pixi_lock_hash = file_sha256(pixi_lock_path.read_bytes())
    except OSError as error:
        raise CalibrationSensitivityError(f"Could not read Pixi lock: {pixi_lock_path}") from error
    content = {
        "schema_version": 1,
        "schema_id": "acquisition_study.development_calibration_sensitivity_manifest.v1",
        "study_id": matrix.study_id,
        "run_id": run_id,
        "code_revision": code_revision,
        "split": matrix.split,
        "result_scope": matrix.result_scope,
        "access_scope": matrix.access_scope,
        "sensitivity_id": matrix.sensitivity_id,
        "source_manifest_hash": matrix.source_manifest_hash,
        "source_primary_calibration_hash": matrix.source_primary_calibration_hash,
        "environment_specification_hash": matrix.environment_specification_hash,
        "analysis_plan_hash": matrix.analysis_plan_hash,
        "calibrations": matrix.calibrations,
        "pixi_lock_sha256": pixi_lock_hash,
        "result_file": "development_calibration_sensitivity.jsonl",
        "result_file_sha256": file_sha256(rows_bytes),
        "matrix_content_hash": matrix.content_hash,
        "replay_content_hash": replay_content_hash,
        "deterministic_replay_verified": True,
        "privileged_source_reproduction_verified": (matrix.privileged_source_reproduction_verified),
        "episodes_per_environment": matrix.episodes_per_environment,
        "candidates_per_episode": matrix.candidates_per_episode,
        "fixed_probe_budget": matrix.fixed_probe_budget,
        "environment_count": matrix.environment_count,
        "policy_count": len(matrix.policy_ids),
        "row_count": matrix.row_count,
        "case_prediction_count": matrix.row_count * matrix.candidates_per_episode,
        "selected_probe_count": matrix.row_count * matrix.fixed_probe_budget,
        "external_model_call_count": matrix.external_model_call_count,
        "sandbox_call_count": matrix.sandbox_call_count,
        "human_record_count": matrix.human_record_count,
        "canonical_claim_allowed": matrix.canonical_claim_allowed,
    }
    manifest = DevelopmentCalibrationSensitivityManifest.model_validate(
        {**content, "manifest_hash": canonical_sha256(content)}
    )
    write_immutable_bytes(output_root / manifest.result_file, rows_bytes)
    write_immutable_json(
        output_root / "development_calibration_sensitivity_manifest.json",
        manifest,
    )
    return manifest


def load_verified_development_calibration_sensitivity(
    manifest_path: Path,
    *,
    specification: AcquisitionEnvironmentSpecification,
    analysis: AcquisitionAnalysisSpecification,
) -> tuple[
    DevelopmentCalibrationSensitivityManifest,
    DevelopmentCalibrationSensitivityMatrix,
]:
    """Load sensitivity rows only after checking their complete manifest."""

    try:
        manifest = DevelopmentCalibrationSensitivityManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        result_path = manifest_path.parent / manifest.result_file
        result_bytes = result_path.read_bytes()
        if file_sha256(result_bytes) != manifest.result_file_sha256:
            raise CalibrationSensitivityError("Calibration-sensitivity file hash differs")
        lines = result_bytes.decode("utf-8").splitlines()
        if len(lines) != manifest.row_count or any(not line for line in lines):
            raise CalibrationSensitivityError("Calibration-sensitivity row count differs")
        rows = tuple(
            CalibrationSensitivityEpisodeMetric.model_validate(json.loads(line)) for line in lines
        )
        matrix = DevelopmentCalibrationSensitivityMatrix(
            source_manifest_hash=manifest.source_manifest_hash,
            source_primary_calibration_hash=manifest.source_primary_calibration_hash,
            environment_specification_hash=manifest.environment_specification_hash,
            analysis_plan_hash=manifest.analysis_plan_hash,
            calibrations=manifest.calibrations,
            episodes_per_environment=manifest.episodes_per_environment,
            candidates_per_episode=manifest.candidates_per_episode,
            fixed_probe_budget=manifest.fixed_probe_budget,
            fixed_budget_fraction=analysis.primary.primary_budget_fraction,
            environment_count=manifest.environment_count,
            policy_ids=_POLICY_ORDER,
            row_count=manifest.row_count,
            rows=rows,
        )
    except CalibrationSensitivityError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError) as error:
        raise CalibrationSensitivityError(
            f"Could not verify development calibration sensitivity: {manifest_path}"
        ) from error
    if manifest.environment_specification_hash != specification.specification_hash:
        raise CalibrationSensitivityError("Sensitivity uses another environment specification")
    if manifest.analysis_plan_hash != analysis.plan_hash:
        raise CalibrationSensitivityError("Sensitivity uses another frozen analysis plan")
    if tuple(item.calibration_seed for item in matrix.calibrations) != (
        specification.calibration.sensitivity_seeds
    ):
        raise CalibrationSensitivityError("Sensitivity uses another calibration-seed set")
    if matrix.content_hash != manifest.matrix_content_hash:
        raise CalibrationSensitivityError("Calibration-sensitivity content hash differs")
    return manifest, matrix
