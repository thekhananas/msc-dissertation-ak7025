from __future__ import annotations

import math
from pathlib import Path

import pytest

from socratic_tutor.tracker_study.cli import main
from socratic_tutor.tracker_study.config import (
    EvidenceChannel,
    StressCondition,
    TrackerStudyConfigurationError,
    load_hand_worked_trace,
    load_tracker_study_configuration,
)

ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "configs" / "tracker-study" / "v1.yaml"
TRACE = ROOT / "data" / "tracker-study" / "v1" / "hand-worked-trace.yaml"


def test_frozen_tracker_study_configuration_and_trace_are_consistent() -> None:
    configuration = load_tracker_study_configuration(CONFIG)
    trace = load_hand_worked_trace(TRACE, configuration)

    assert configuration.configuration_hash == (
        "8b1656f59265b04ac255c2d825807b3d8ed3fd8f43c2e454d07c58d7ef221c4b"
    )
    assert tuple(model.channel for model in configuration.observation_models) == tuple(
        EvidenceChannel
    )
    assert tuple(item.condition for item in configuration.stress_conditions) == tuple(
        StressCondition
    )
    assert math.isclose(trace.ordinary_posterior, 0.8823529411764706, abs_tol=1e-12)
    assert math.isclose(trace.bounded_posterior, 0.8013152260326584, abs_tol=1e-12)
    assert math.isclose(trace.bounded_log_odds_delta, 1.8, abs_tol=1e-12)


def test_configuration_hash_detects_a_changed_stress_parameter(tmp_path: Path) -> None:
    changed = CONFIG.read_text(encoding="utf-8").replace(
        "missing_probability: 0.50", "missing_probability: 0.40"
    )
    path = tmp_path / "changed.yaml"
    path.write_text(changed, encoding="utf-8")

    with pytest.raises(TrackerStudyConfigurationError):
        load_tracker_study_configuration(path)


def test_validation_command_reports_frozen_scope(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["validate", "--config", str(CONFIG), "--trace", str(TRACE)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert '"status": "ok"' in captured.out
    assert '"test_episodes_per_condition": 500' in captured.out
    assert '"tracker_count": 5' in captured.out
    assert '"claim_scope": "glass_box_simulator_robustness_not_human_cognition"' in (captured.out)
    assert captured.err == ""
