from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml
from matplotlib.figure import Figure
from matplotlib.text import Annotation

from socratic_tutor.publication.system_boundary import (
    SystemBoundaryFigureError,
    SystemEdge,
    _draw_edge,
    render_system_boundary_figure,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
SPECIFICATION = WORKSPACE_ROOT / "configs" / "publication" / "v1-system-boundary.yaml"
PIXI_LOCK = WORKSPACE_ROOT / "pixi.lock"


@pytest.mark.parametrize("source_x,target_x", [(0.1, 0.6), (0.6, 0.1)])
def test_horizontal_arrows_stay_between_boxes(source_x: float, target_x: float) -> None:
    axis = Figure().subplots()
    width = 0.2
    _draw_edge(
        axis,
        SystemEdge(source="source", target="target", label="response"),
        {"source": (source_x, 0.2, width, 0.1), "target": (target_x, 0.2, width, 0.1)},
        {},
    )
    arrow = next(item for item in axis.texts if isinstance(item, Annotation))
    left_edge = min(source_x, target_x) + width
    right_edge = max(source_x, target_x)
    assert left_edge < arrow.xy[0] < right_edge
    assert left_edge < arrow.get_position()[0] < right_edge
    assert (arrow.xy[0] > arrow.get_position()[0]) == (target_x > source_x)


def test_renders_system_boundaries_and_retries_exactly(tmp_path: Path) -> None:
    output_root = tmp_path / "publication"
    generated_at = datetime(2026, 9, 3, 20, 0, tzinfo=UTC)

    first = render_system_boundary_figure(
        specification_path=SPECIFICATION,
        source_root=WORKSPACE_ROOT,
        pixi_lock_path=PIXI_LOCK,
        output_root=output_root,
        publication_code_revision="a" * 40,
        generated_at_utc=generated_at,
    )
    retry = render_system_boundary_figure(
        specification_path=SPECIFICATION,
        source_root=WORKSPACE_ROOT,
        pixi_lock_path=PIXI_LOCK,
        output_root=output_root,
        publication_code_revision="a" * 40,
    )

    assert first == retry
    assert not first.live_llm_used_in_demo
    assert not first.evaluation_model_is_human_learner
    assert not first.evaluation_model_is_validated_student_simulator
    assert first.offline_analysis_external_calls == 0
    assert len(first.source_files) == 13
    assert (output_root / "system_boundaries.pdf").read_bytes().startswith(b"%PDF")
    svg = (output_root / "system_boundaries.svg").read_text(encoding="utf-8")
    assert "The local demo is deterministic" in svg
    assert "Five Bayesian trackers" in svg
    csv_text = (output_root / "system_boundary_components.csv").read_text(encoding="utf-8")
    assert "interactive_demo,0,3,langgraph_turn" in csv_text
    assert "external_benchmark,0,2,evaluation_model,Pinned evaluation model" in csv_text
    assert "offline_analysis,1,1,glass_box_simulator" in csv_text


def test_rejects_an_edge_to_an_unknown_component(tmp_path: Path) -> None:
    raw = yaml.safe_load(SPECIFICATION.read_text(encoding="utf-8"))
    raw["edges"][0]["target"] = "missing_component"
    changed = tmp_path / "invalid.yaml"
    changed.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(SystemBoundaryFigureError, match="unknown node"):
        render_system_boundary_figure(
            specification_path=changed,
            source_root=WORKSPACE_ROOT,
            pixi_lock_path=PIXI_LOCK,
            output_root=tmp_path / "publication",
            publication_code_revision="a" * 40,
        )
