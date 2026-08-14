"""Paired component ablations for reliability-aware probe acquisition."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, model_validator

from socratic_tutor.acquisition_study.calibration import ReliabilityCalibrationRun
from socratic_tutor.acquisition_study.contracts import (
    AcquisitionPolicy,
    PolicyId,
    ProbeClassId,
)
from socratic_tutor.acquisition_study.evaluation import (
    PolicyEpisodeResult,
    evaluate_policy_episode,
)
from socratic_tutor.acquisition_study.plan import (
    AblationId,
    AcquisitionAnalysisSpecification,
    AcquisitionEnvironmentSpecification,
    EvaluationEnvironmentId,
)
from socratic_tutor.acquisition_study.policies import (
    PlugInEVSIPolicy,
    PlugInEVSIUnboundedPolicy,
    ReliabilityAwareEVSIPolicy,
    ReliabilityAwareUnboundedPolicy,
)
from socratic_tutor.acquisition_study.simulation import generate_acquisition_episode
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import canonical_sha256, model_content_hash
from socratic_tutor.contracts import ContractModel


class AblationPolicyCellId(StrEnum):
    FULL_CANDIDATE = "full_candidate"
    POSTERIOR_MEAN_BOUNDED = AblationId.REPLACE_LOWER_QUANTILE_WITH_POSTERIOR_MEAN
    LOWER_QUANTILE_UNBOUNDED = AblationId.REMOVE_BOUNDED_UPDATE_FROM_CANDIDATE
    POSTERIOR_MEAN_UNBOUNDED = AblationId.REMOVE_BOTH_RELIABILITY_CONSERVATISM_AND_BOUNDED_UPDATE


class ProbeClassSelectionCount(ContractModel):
    """Selected and available cases for one probe class."""

    probe_class: ProbeClassId
    available_case_count: int = Field(ge=1)
    selected_case_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_count(self) -> Self:
        if self.selected_case_count > self.available_case_count:
            raise ValueError("Selected probe-class cases exceed those available")
        return self


class AblationPolicyResult(ContractModel):
    """One cell in the prespecified two-by-two component comparison."""

    cell_id: AblationPolicyCellId
    reliability_summary: Literal["lower_quantile", "posterior_mean"]
    update_rule: Literal["bounded", "unbounded"]
    result: PolicyEpisodeResult
    selection_by_probe_class: tuple[ProbeClassSelectionCount, ...] = Field(
        min_length=4,
        max_length=4,
    )

    @model_validator(mode="after")
    def validate_cell(self) -> Self:
        expected = {
            AblationPolicyCellId.FULL_CANDIDATE: (
                PolicyId.RELIABILITY_AWARE_BOUNDED,
                "lower_quantile",
                "bounded",
            ),
            AblationPolicyCellId.POSTERIOR_MEAN_BOUNDED: (
                PolicyId.PLUG_IN_EVSI_BOUNDED,
                "posterior_mean",
                "bounded",
            ),
            AblationPolicyCellId.LOWER_QUANTILE_UNBOUNDED: (
                PolicyId.ABLATION_QUANTILE_UNBOUNDED,
                "lower_quantile",
                "unbounded",
            ),
            AblationPolicyCellId.POSTERIOR_MEAN_UNBOUNDED: (
                PolicyId.ABLATION_MEAN_UNBOUNDED,
                "posterior_mean",
                "unbounded",
            ),
        }[self.cell_id]
        if (self.result.policy_id, self.reliability_summary, self.update_rule) != expected:
            raise ValueError("Ablation cell does not match its declared components")
        if tuple(row.probe_class for row in self.selection_by_probe_class) != tuple(ProbeClassId):
            raise ValueError("Ablation selection counts do not cover every probe class")
        if sum(row.available_case_count for row in self.selection_by_probe_class) != len(
            self.result.case_results
        ):
            raise ValueError("Ablation probe-class availability does not cover every case")
        if sum(row.selected_case_count for row in self.selection_by_probe_class) != len(
            self.result.decision.selected_case_ids
        ):
            raise ValueError("Ablation probe-class selections do not match the decision")
        return self


class EpisodeAblationComparison(ContractModel):
    """Four component combinations scored against one hidden episode."""

    schema_version: Literal[1] = 1
    schema_id: Literal["acquisition_study.episode_ablation_comparison.v1"] = (
        "acquisition_study.episode_ablation_comparison.v1"
    )
    study_id: Literal["reliability-aware-probing-v1"] = "reliability-aware-probing-v1"
    environment_id: EvaluationEnvironmentId
    episode_index: int = Field(ge=0)
    episode_id: str = Field(min_length=1)
    shared_candidate_hash: Sha256
    privileged_episode_hash: Sha256
    candidate_count: int = Field(ge=1)
    fixed_probe_budget: int = Field(ge=1)
    cells: tuple[AblationPolicyResult, ...] = Field(min_length=4, max_length=4)
    external_model_call_count: Literal[0] = 0
    sandbox_call_count: Literal[0] = 0
    human_record_count: Literal[0] = 0
    access_scope: Literal["simulator_restricted"] = "simulator_restricted"

    @model_validator(mode="after")
    def validate_comparison(self) -> Self:
        if self.episode_id != f"episode-{self.episode_index:06d}":
            raise ValueError("Ablation episode identifier does not match its index")
        if tuple(cell.cell_id for cell in self.cells) != tuple(AblationPolicyCellId):
            raise ValueError("Ablation cells differ from the prespecified order")
        if any(
            cell.result.episode_id != self.episode_id
            or cell.result.environment_id is not self.environment_id
            or len(cell.result.case_results) != self.candidate_count
            or cell.result.burden.selected_probe_count != self.fixed_probe_budget
            for cell in self.cells
        ):
            raise ValueError("Ablation cells do not share one episode and budget")
        if any(
            cell.result.burden.external_workload.model_request_count != 0
            or cell.result.burden.external_workload.sandbox_execution_count != 0
            for cell in self.cells
        ):
            raise ValueError("Glass-box ablations cannot contain external work")
        reference_cases = tuple(
            (case.case_id, case.prior_success_probability, case.latent_success)
            for case in self.cells[0].result.case_results
        )
        if any(
            tuple(
                (case.case_id, case.prior_success_probability, case.latent_success)
                for case in cell.result.case_results
            )
            != reference_cases
            for cell in self.cells
        ):
            raise ValueError("Ablation cells were not scored on identical cases")
        observed_by_case: dict[str, object] = {}
        for cell in self.cells:
            for case in cell.result.case_results:
                if case.observed_probe_outcome is None:
                    continue
                previous = observed_by_case.setdefault(case.case_id, case.observed_probe_outcome)
                if previous != case.observed_probe_outcome:
                    raise ValueError("Ablation cells received different outcomes for one probe")
        return self

    @property
    def content_hash(self) -> Sha256:
        return canonical_sha256(self)


def run_episode_ablation_comparison(
    specification: AcquisitionEnvironmentSpecification,
    analysis: AcquisitionAnalysisSpecification,
    calibration: ReliabilityCalibrationRun,
    *,
    environment_id: EvaluationEnvironmentId,
    episode_index: int,
) -> EpisodeAblationComparison:
    """Evaluate the frozen component combinations on one paired episode."""

    if specification.specification_hash != analysis.environment_specification_hash:
        raise ValueError("Analysis and environment specifications do not match")
    if calibration.environment_specification_hash != specification.specification_hash:
        raise ValueError("Calibration belongs to another environment specification")
    budget = analysis.primary.exact_selected_probes_per_episode
    policy_episode, privileged_episode = generate_acquisition_episode(
        specification,
        calibration,
        environment_id=environment_id,
        episode_index=episode_index,
        probe_budget=budget,
    )
    policies: tuple[tuple[AblationPolicyCellId, AcquisitionPolicy, float | None], ...] = (
        (
            AblationPolicyCellId.FULL_CANDIDATE,
            ReliabilityAwareEVSIPolicy(
                lower_quantile=analysis.policies.conservative_reliability_quantile,
                kappa=analysis.policies.bounded_log_likelihood_kappa,
                draw_count=analysis.policies.reliability_draw_count,
                seed=analysis.policies.reliability_draw_seed,
            ),
            analysis.policies.bounded_log_likelihood_kappa,
        ),
        (
            AblationPolicyCellId.POSTERIOR_MEAN_BOUNDED,
            PlugInEVSIPolicy(kappa=analysis.policies.bounded_log_likelihood_kappa),
            analysis.policies.bounded_log_likelihood_kappa,
        ),
        (
            AblationPolicyCellId.LOWER_QUANTILE_UNBOUNDED,
            ReliabilityAwareUnboundedPolicy(
                lower_quantile=analysis.policies.conservative_reliability_quantile,
                draw_count=analysis.policies.reliability_draw_count,
                seed=analysis.policies.reliability_draw_seed,
            ),
            None,
        ),
        (
            AblationPolicyCellId.POSTERIOR_MEAN_UNBOUNDED,
            PlugInEVSIUnboundedPolicy(),
            None,
        ),
    )
    candidates_by_id = {
        candidate.case_id: candidate for candidate in policy_episode.request.candidates
    }
    cells: list[AblationPolicyResult] = []
    for cell_id, policy, kappa in policies:
        result = evaluate_policy_episode(
            policy,
            policy_episode,
            privileged_episode,
            kappa=kappa,
            classification_threshold=analysis.primary.classification_threshold,
            probability_floor=analysis.secondary.probability_floor,
        )
        selected_ids = set(result.decision.selected_case_ids)
        cells.append(
            AblationPolicyResult(
                cell_id=cell_id,
                reliability_summary=(
                    "lower_quantile"
                    if cell_id
                    in {
                        AblationPolicyCellId.FULL_CANDIDATE,
                        AblationPolicyCellId.LOWER_QUANTILE_UNBOUNDED,
                    }
                    else "posterior_mean"
                ),
                update_rule=(
                    "bounded"
                    if cell_id
                    in {
                        AblationPolicyCellId.FULL_CANDIDATE,
                        AblationPolicyCellId.POSTERIOR_MEAN_BOUNDED,
                    }
                    else "unbounded"
                ),
                result=result,
                selection_by_probe_class=tuple(
                    ProbeClassSelectionCount(
                        probe_class=probe_class,
                        available_case_count=sum(
                            candidate.probe_class is probe_class
                            for candidate in policy_episode.request.candidates
                        ),
                        selected_case_count=sum(
                            case_id in selected_ids
                            and candidates_by_id[case_id].probe_class is probe_class
                            for case_id in candidates_by_id
                        ),
                    )
                    for probe_class in ProbeClassId
                ),
            )
        )
    return EpisodeAblationComparison(
        environment_id=environment_id,
        episode_index=episode_index,
        episode_id=policy_episode.episode_id,
        shared_candidate_hash=canonical_sha256(policy_episode.request.candidates),
        privileged_episode_hash=model_content_hash(privileged_episode),
        candidate_count=len(policy_episode.request.candidates),
        fixed_probe_budget=budget,
        cells=tuple(cells),
    )
