from pathlib import Path

import yaml

from socratic_tutor.benchmark.artifacts import artifact_locations
from socratic_tutor.publication.figure_style import (
    BLUE,
    GREEN,
    ORANGE,
    PRESENTATION_STYLE,
    REPORT_STYLE,
    style_for,
)


def test_report_and_presentation_profiles_share_semantic_colours() -> None:
    assert style_for("report") == REPORT_STYLE
    assert style_for("presentation") == PRESENTATION_STYLE
    assert (BLUE, ORANGE, GREEN) == ("#0072B2", "#D55E00", "#009E73")
    assert REPORT_STYLE.font_size < PRESENTATION_STYLE.font_size


def test_artifact_locations_reports_files_relative_to_working_directory(tmp_path: Path) -> None:
    output_root = tmp_path / "publication"
    output_root.mkdir()
    (output_root / "figure.pdf").write_bytes(b"pdf")
    (output_root / "figure.svg").write_text("svg", encoding="utf-8")

    locations = artifact_locations(output_root)

    assert locations["output_root"] == str(output_root.resolve())
    assert locations["files"] == [str(output_root / "figure.pdf"), str(output_root / "figure.svg")]


def test_visual_register_has_unique_ids_and_both_output_profiles() -> None:
    register_path = Path(__file__).parents[2] / "configs/publication/v1-figure-register.yaml"
    register = yaml.safe_load(register_path.read_text(encoding="utf-8"))
    figures = register["figures"]

    assert len({figure["figure_id"] for figure in figures}) == len(figures)
    assert {figure["status"] for figure in figures} == {"primary", "supporting"}
    assert set(register["profiles"]) == {"report", "presentation"}
    assert all(figure["question"] and figure["claim_boundary"] for figure in figures)
