from __future__ import annotations

from pathlib import Path

import pytest

from socratic_tutor.tracker_study.analysis_spec import (
    TrackerStudyAnalysisSpecificationError,
    load_tracker_study_analysis_specification,
)
from socratic_tutor.tracker_study.cli import main
from socratic_tutor.tracker_study.config import (
    StressCondition,
    TrackerId,
    load_tracker_study_configuration,
)

ROOT = Path(__file__).parents[2]
CONFIG_PATH = ROOT / "configs" / "tracker-study" / "v1.yaml"
ANALYSIS_PATH = ROOT / "configs" / "tracker-study" / "v1-analysis.yaml"
CONFIGURATION = load_tracker_study_configuration(CONFIG_PATH)


def test_analysis_specification_freezes_primary_comparison_before_results() -> None:
    specification = load_tracker_study_analysis_specification(
        ANALYSIS_PATH,
        CONFIGURATION,
    )

    assert specification.primary.candidate_tracker is TrackerId.BOUNDED_CHANNEL_AWARE
    assert specification.primary.reference_tracker is TrackerId.CHANNEL_AWARE
    assert specification.primary.adverse_conditions == tuple(StressCondition)[1:]
    assert specification.primary.metric == "brier_score"
    assert specification.primary.maximum_clean_brier_degradation == 0.01
    assert not specification.development_results_inspected_before_freeze
    assert not specification.held_out_benchmark_outcomes_used
    assert not specification.external_model_data_used
    assert not specification.human_data_used


def test_analysis_specification_hash_detects_changed_primary_design(
    tmp_path: Path,
) -> None:
    changed = ANALYSIS_PATH.read_text(encoding="utf-8").replace(
        "maximum_clean_brier_degradation: 0.01",
        "maximum_clean_brier_degradation: 0.02",
    )
    path = tmp_path / "changed-analysis.yaml"
    path.write_text(changed, encoding="utf-8")

    with pytest.raises(TrackerStudyAnalysisSpecificationError):
        load_tracker_study_analysis_specification(path, CONFIGURATION)


def test_analysis_validation_command_reports_claim_boundary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "validate-analysis",
            "--config",
            str(CONFIG_PATH),
            "--analysis",
            str(ANALYSIS_PATH),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert '"primary_metric": "brier_score"' in captured.out
    assert '"adverse_condition_count": 5' in captured.out
    assert '"development_results_inspected_before_freeze": false' in captured.out
    assert '"claim_scope": "glass_box_simulator_robustness_not_human_cognition"' in captured.out
    assert captured.err == ""
