"""Validated architecture specification and reproducible system-boundary figure."""

# pyright: reportArgumentType=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import textwrap
from collections import defaultdict
from datetime import UTC, datetime
from enum import StrEnum
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Literal

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch
from pydantic import Field, ValidationError, model_validator

from socratic_tutor.benchmark.artifacts import write_immutable_bytes, write_immutable_json
from socratic_tutor.benchmark.command_io import load_command_model
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import file_sha256, model_content_hash
from socratic_tutor.contracts import ContractModel
from socratic_tutor.csv_output import csv_bytes

_BLUE = "#0072B2"
_ORANGE = "#D55E00"
_GREEN = "#009E73"
_PURPLE = "#7A5195"
_GREY = "#8A949E"
_LIGHT_GREY = "#D7DDE2"
_PALE_BLUE = "#EEF2F3"
_PALE_ORANGE = "#F3EEE8"
_PALE_GREY = "#F0F0EC"
_INK = "#17212B"
_MUTED = "#56616B"
_STYLE: dict[str, object] = {
    "figure.facecolor": "#FAFAF7",
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans", "sans-serif"],
    "font.size": 11,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "svg.fonttype": "none",
    "svg.hashsalt": "socratic-tutor-system-boundary-v1",
    "text.color": _INK,
}

# The component CSV keeps the full specification text. The figure uses short
# summaries so the report-sized export remains readable at normal print size.
_FIGURE_DETAILS: dict[str, str] = {
    "demo_user": "Sends an answer to a prepared task.",
    "web_api": "Shows the session; checks each request.",
    "langgraph_turn": "Updates the tracker; chooses an action.",
    "template_tutor": "Returns the next prompt; no live model.",
    "local_log": "Saves each turn for replay.",
    "frozen_cases": "Separates three task views.",
    "evaluation_model": "Supplies evaluation responses.",
    "evidence_path": "Runs probe code; records results.",
    "decision_seal": "Saves four predictions before outcome.",
    "criterion_gate": "Requests the later task after predictions.",
    "criterion_outcomes": "Runs outcome code; records completion.",
    "immutable_records": "Stores predictions and reviews.",
    "paired_analysis": "Compares paired results.",
    "report_outputs": "Creates report files.",
    "glass_box_simulator": "Creates known noisy states.",
    "bayesian_trackers": "Compares update rules.",
    "robustness_outputs": "Reports accuracy and cost.",
}

_FIGURE_TITLES: dict[str, str] = {
    "template_tutor": "Prompt template",
    "local_log": "Saved turn log",
    "frozen_cases": "Prepared cases",
    "evaluation_model": "Evaluation model",
    "evidence_path": "Code check + tracker",
    "decision_seal": "Saved predictions",
    "criterion_gate": "Outcome request",
    "criterion_outcomes": "Outcome code check",
    "immutable_records": "Saved benchmark",
    "paired_analysis": "Result comparison",
    "report_outputs": "Report figures",
    "glass_box_simulator": "Controlled simulator",
    "robustness_outputs": "Tracking results",
}


class SystemBoundaryFigureError(ValueError):
    """System-boundary sources are missing, invalid, or inconsistent."""


class BoundaryId(StrEnum):
    INTERACTIVE_DEMO = "interactive_demo"
    EXTERNAL_BENCHMARK = "external_benchmark"
    OFFLINE_ANALYSIS = "offline_analysis"


_FIGURE_STATEMENTS: dict[BoundaryId, str] = {
    BoundaryId.INTERACTIVE_DEMO: "One local tutoring turn.",
    BoundaryId.EXTERNAL_BENCHMARK: "Predictions saved before outcome request.",
    BoundaryId.OFFLINE_ANALYSIS: "Reads saved records; makes no new calls.",
}


class NodeKind(StrEnum):
    ACTOR = "actor"
    INTERFACE = "interface"
    ORCHESTRATION = "orchestration"
    GENERATION = "generation"
    STORAGE = "storage"
    DATA = "data"
    MODEL = "model"
    EXECUTION = "execution"
    CONTROL = "control"
    ANALYSIS = "analysis"
    PUBLICATION = "publication"
    SIMULATION = "simulation"


class SystemBoundary(ContractModel):
    boundary_id: BoundaryId
    title: str = Field(min_length=1)
    statement: str = Field(min_length=1)


class SystemNode(ContractModel):
    node_id: str = Field(pattern=r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
    boundary_id: BoundaryId
    lane: int = Field(ge=0, le=1)
    order: int = Field(ge=1)
    title: str = Field(min_length=1)
    detail: str = Field(min_length=1)
    kind: NodeKind


class SystemEdge(ContractModel):
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    label: str = Field(min_length=1)


class SystemBoundarySpecification(ContractModel):
    schema_version: Literal[1] = 1
    schema_id: Literal["publication.system_boundary_specification.v1"] = (
        "publication.system_boundary_specification.v1"
    )
    specification_version: Literal["v1"] = "v1"
    status: Literal["frozen"] = "frozen"
    frozen_at_utc: datetime
    title: str = Field(min_length=1)
    subtitle: str = Field(min_length=1)
    boundaries: tuple[SystemBoundary, SystemBoundary, SystemBoundary]
    nodes: tuple[SystemNode, ...] = Field(min_length=17, max_length=17)
    edges: tuple[SystemEdge, ...] = Field(min_length=13, max_length=13)
    source_files: tuple[str, ...] = Field(min_length=10)
    claim_boundaries: tuple[str, ...] = Field(min_length=5)

    @model_validator(mode="after")
    def validate_specification(self) -> SystemBoundarySpecification:
        _require_utc(self.frozen_at_utc)
        if tuple(boundary.boundary_id for boundary in self.boundaries) != tuple(BoundaryId):
            raise ValueError("System boundaries must appear in the declared reading order")
        node_ids = tuple(node.node_id for node in self.nodes)
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("System-boundary node IDs must be unique")
        required_nodes = {
            "demo_user",
            "web_api",
            "langgraph_turn",
            "template_tutor",
            "local_log",
            "frozen_cases",
            "evaluation_model",
            "evidence_path",
            "decision_seal",
            "criterion_gate",
            "criterion_outcomes",
            "immutable_records",
            "paired_analysis",
            "report_outputs",
            "glass_box_simulator",
            "bayesian_trackers",
            "robustness_outputs",
        }
        if set(node_ids) != required_nodes:
            raise ValueError("System-boundary specification has the wrong component set")
        node_lookup = {node.node_id: node for node in self.nodes}
        positions: set[tuple[BoundaryId, int, int]] = set()
        for node in self.nodes:
            position = (node.boundary_id, node.lane, node.order)
            if position in positions:
                raise ValueError("System-boundary nodes cannot share a layout position")
            positions.add(position)
            if node.boundary_id is not BoundaryId.OFFLINE_ANALYSIS and node.lane != 0:
                raise ValueError("Only offline analysis may use a second lane")
        edge_pairs: set[tuple[str, str]] = set()
        for edge in self.edges:
            if edge.source not in node_lookup or edge.target not in node_lookup:
                raise ValueError("System-boundary edge references an unknown node")
            if node_lookup[edge.source].boundary_id != node_lookup[edge.target].boundary_id:
                raise ValueError(
                    "Cross-boundary data flow must be described, not implied by an edge"
                )
            pair = (edge.source, edge.target)
            if pair in edge_pairs:
                raise ValueError("System-boundary edges must be unique")
            edge_pairs.add(pair)
        if len(set(self.source_files)) != len(self.source_files):
            raise ValueError("System-boundary source files must be unique")
        for source in self.source_files:
            path = PurePosixPath(source)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("System-boundary source paths must be repository-relative")
        if not any("no live LLM" in item for item in self.claim_boundaries):
            raise ValueError("System boundary must disclose the demo generation mode")
        if not any("not a human learner" in item for item in self.claim_boundaries):
            raise ValueError("System boundary must distinguish the evaluation model")
        return self


class SourceFileDigest(ContractModel):
    path: str = Field(min_length=1)
    sha256: Sha256


class SystemBoundaryFigureManifest(ContractModel):
    schema_version: Literal[1] = 1
    schema_id: Literal["publication.system_boundary_figure_manifest.v1"] = (
        "publication.system_boundary_figure_manifest.v1"
    )
    specification_sha256: Sha256
    specification_content_hash: Sha256
    source_files: tuple[SourceFileDigest, ...]
    publication_pixi_lock_sha256: Sha256
    publication_code_revision: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    figure_formats: tuple[Literal["pdf", "svg"], Literal["pdf", "svg"]]
    pdf_sha256: Sha256
    svg_sha256: Sha256
    component_table_csv_sha256: Sha256
    generated_at_utc: datetime
    live_llm_used_in_demo: Literal[False] = False
    evaluation_model_is_human_learner: Literal[False] = False
    evaluation_model_is_validated_student_simulator: Literal[False] = False
    offline_analysis_external_calls: Literal[0] = 0
    human_learning_claim_supported: Literal[False] = False
    manifest_hash: Sha256

    @model_validator(mode="after")
    def validate_manifest(self) -> SystemBoundaryFigureManifest:
        _require_utc(self.generated_at_utc)
        if self.figure_formats != ("pdf", "svg"):
            raise ValueError("System-boundary figure must publish PDF and SVG")
        if self.manifest_hash != model_content_hash(self, exclude={"manifest_hash"}):
            raise ValueError("System-boundary manifest hash does not match its content")
        return self


def render_system_boundary_figure(
    *,
    specification_path: Path,
    source_root: Path,
    pixi_lock_path: Path,
    output_root: Path,
    publication_code_revision: str,
    generated_at_utc: datetime | None = None,
) -> SystemBoundaryFigureManifest:
    """Render the three implementation boundaries from a frozen design source."""

    specification = _load_specification(specification_path)
    try:
        specification_bytes = specification_path.read_bytes()
        pixi_lock_hash = file_sha256(pixi_lock_path.read_bytes())
    except OSError as error:
        raise SystemBoundaryFigureError("Could not read a system-boundary source") from error
    source_digests = _source_digests(source_root, specification.source_files)
    generated_at = _resolve_generated_at(
        generated_at_utc,
        output_root / "system_boundary_figure_manifest.json",
    )
    component_table = _component_table_csv(specification)
    pdf, svg = _render_vector_files(specification, generated_at)
    content = {
        "schema_version": 1,
        "schema_id": "publication.system_boundary_figure_manifest.v1",
        "specification_sha256": file_sha256(specification_bytes),
        "specification_content_hash": model_content_hash(specification),
        "source_files": source_digests,
        "publication_pixi_lock_sha256": pixi_lock_hash,
        "publication_code_revision": publication_code_revision,
        "figure_formats": ("pdf", "svg"),
        "pdf_sha256": file_sha256(pdf),
        "svg_sha256": file_sha256(svg),
        "component_table_csv_sha256": file_sha256(component_table),
        "generated_at_utc": generated_at,
        "live_llm_used_in_demo": False,
        "evaluation_model_is_human_learner": False,
        "evaluation_model_is_validated_student_simulator": False,
        "offline_analysis_external_calls": 0,
        "human_learning_claim_supported": False,
    }
    draft = SystemBoundaryFigureManifest.model_construct(
        _fields_set=set(content),
        **content,
        manifest_hash="0" * 64,
    )
    manifest = SystemBoundaryFigureManifest.model_validate(
        {**content, "manifest_hash": model_content_hash(draft, exclude={"manifest_hash"})}
    )
    write_immutable_bytes(output_root / "system_boundaries.pdf", pdf)
    write_immutable_bytes(output_root / "system_boundaries.svg", svg)
    write_immutable_bytes(output_root / "system_boundary_components.csv", component_table)
    write_immutable_json(output_root / "system_boundary_figure_manifest.json", manifest)
    return manifest


def _source_digests(source_root: Path, sources: tuple[str, ...]) -> tuple[SourceFileDigest, ...]:
    records: list[SourceFileDigest] = []
    for relative in sources:
        path = source_root / relative
        try:
            digest = file_sha256(path.read_bytes())
        except OSError as error:
            raise SystemBoundaryFigureError(
                f"Could not verify implementation source: {path}"
            ) from error
        records.append(SourceFileDigest(path=relative, sha256=digest))
    return tuple(records)


def _component_table_csv(specification: SystemBoundarySpecification) -> bytes:
    rows = [
        (
            node.boundary_id.value,
            node.lane,
            node.order,
            node.node_id,
            node.title,
            node.kind.value,
            node.detail,
        )
        for node in specification.nodes
    ]
    return csv_bytes(
        ("boundary", "lane", "order", "node_id", "title", "kind", "detail"),
        rows,
    )


def _render_vector_files(
    specification: SystemBoundarySpecification,
    generated_at_utc: datetime,
) -> tuple[bytes, bytes]:
    with mpl.rc_context(_STYLE):
        figure = _build_figure(specification)
        pdf_buffer = BytesIO()
        svg_buffer = BytesIO()
        figure.savefig(
            pdf_buffer,
            format="pdf",
            metadata={
                "Title": specification.title,
                "Author": "Anas Khan",
                "Subject": "Socratic tutoring proof-of-concept system boundaries",
                "Creator": "Socratic Tutor reproducible figure pipeline",
                "CreationDate": generated_at_utc,
                "ModDate": generated_at_utc,
            },
        )
        figure.savefig(
            svg_buffer,
            format="svg",
            metadata={
                "Title": specification.title,
                "Description": specification.subtitle,
                "Creator": "Socratic Tutor reproducible figure pipeline",
                "Date": generated_at_utc.isoformat(),
            },
        )
        figure.clear()
    return pdf_buffer.getvalue(), svg_buffer.getvalue()


def _build_figure(specification: SystemBoundarySpecification) -> Figure:
    figure = Figure(figsize=(6.0, 8.2), facecolor="white")
    axis = figure.add_axes((0, 0, 1, 1))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    figure.text(0.035, 0.970, specification.title, fontsize=13, fontweight="bold", color=_INK)
    figure.text(
        0.035,
        0.921,
        "The local demo is deterministic in the path shown here.\n"
        "The benchmark calls the model; analysis reads its saved responses.",
        fontsize=9.5,
        color=_MUTED,
    )

    layouts = {
        # Keep a uniform gap between the three sections.
        BoundaryId.INTERACTIVE_DEMO: (0.630, 0.265, _PALE_BLUE, _BLUE),
        BoundaryId.EXTERNAL_BENCHMARK: (0.345, 0.265, _PALE_ORANGE, _ORANGE),
        BoundaryId.OFFLINE_ANALYSIS: (0.060, 0.265, _PALE_GREY, _PURPLE),
    }
    coordinates: dict[str, tuple[float, float, float, float]] = {}
    for boundary in specification.boundaries:
        y_position, height, background, accent = layouts[boundary.boundary_id]
        _draw_boundary(axis, boundary, y_position, height, background, accent)
        boundary_nodes = tuple(
            node for node in specification.nodes if node.boundary_id is boundary.boundary_id
        )
        grouped: dict[int, list[SystemNode]] = defaultdict(list)
        for node in boundary_nodes:
            grouped[node.lane].append(node)
        lane_count = len(grouped)
        for lane, nodes in sorted(grouped.items()):
            nodes.sort(key=lambda item: item.order)
            lane_height = (height - 0.060) / lane_count
            lane_y = y_position + 0.012 + (lane_count - lane - 1) * lane_height
            coordinates.update(_draw_lane(axis, nodes, lane_y, lane_height - 0.018, accent))

    for edge in specification.edges:
        _draw_edge(axis, edge, coordinates)

    figure.text(
        0.035,
        0.006,
        "Results concern prepared cases and simulated states.\n"
        "Neither study measures human learning.",
        fontsize=9,
        color=_MUTED,
        va="bottom",
    )
    return figure


def _draw_boundary(
    axis: Axes,
    boundary: SystemBoundary,
    y_position: float,
    height: float,
    background: str,
    accent: str,
) -> None:
    axis.add_patch(
        FancyBboxPatch(
            (0.035, y_position),
            0.93,
            height,
            boxstyle="square,pad=0.006",
            facecolor=background,
            edgecolor=accent,
            linewidth=1.1,
        )
    )
    header_y = y_position + height - 0.020
    axis.text(
        0.050,
        header_y,
        boundary.title,
        fontsize=10.5,
        fontweight="bold",
        color=accent,
        va="baseline",
    )
    axis.text(
        0.050,
        header_y - 0.023,
        _FIGURE_STATEMENTS[boundary.boundary_id],
        fontsize=9,
        color=_MUTED,
        va="baseline",
    )


def _draw_lane(
    axis: Axes,
    nodes: list[SystemNode],
    y_position: float,
    height: float,
    accent: str,
) -> dict[str, tuple[float, float, float, float]]:
    left = 0.05
    right = 0.95
    columns = min(3, len(nodes))
    rows = (len(nodes) + columns - 1) // columns
    horizontal_gap = 0.035
    vertical_gap = 0.016 if rows > 1 else 0.0
    width = (right - left - horizontal_gap * (columns - 1)) / columns
    box_height = (height - vertical_gap * (rows - 1)) / rows
    coordinates: dict[str, tuple[float, float, float, float]] = {}
    for index, node in enumerate(nodes):
        row = index // columns
        column_index = index % columns
        # The second row reads back towards the left, so the arrow from the
        # last box on row one drops straight down into the next box.
        column = columns - column_index - 1 if row % 2 else column_index
        x_position = left + column * (width + horizontal_gap)
        box_y = y_position + (rows - row - 1) * (box_height + vertical_gap)
        wrapped_title = textwrap.fill(_FIGURE_TITLES.get(node.node_id, node.title), width=20)
        display_detail = _FIGURE_DETAILS.get(node.node_id, node.detail)
        wrapped_detail = textwrap.fill(display_detail, width=23)
        title_top = box_y + box_height - 0.012
        # Leave a visible gap below the heading and above the description;
        # this keeps the last line clear of the lower box edge.
        points_per_axis_height = axis.figure.get_figheight() * 72
        detail_top = (
            title_top - (11.4 * len(wrapped_title.splitlines()) + 4) / points_per_axis_height
        )
        box = FancyBboxPatch(
            (x_position, box_y),
            width,
            box_height,
            boxstyle="square,pad=0.004",
            facecolor="white",
            edgecolor=_LIGHT_GREY,
            linewidth=0.9,
        )
        axis.add_patch(box)
        axis.text(
            x_position + 0.009,
            title_top,
            wrapped_title,
            fontsize=9.5,
            fontweight="bold",
            color=_INK,
            va="top",
        )
        axis.text(
            x_position + 0.009,
            detail_top,
            wrapped_detail,
            fontsize=9,
            color=_MUTED,
            va="top",
            linespacing=1.10,
        )
        axis.plot(
            (x_position + 0.006, x_position + width - 0.006),
            (box_y + box_height - 0.006, box_y + box_height - 0.006),
            color=accent,
            linewidth=2.2,
        )
        coordinates[node.node_id] = (x_position, box_y, width, box_height)
    return coordinates


def _draw_edge(
    axis: Axes,
    edge: SystemEdge,
    coordinates: dict[str, tuple[float, float, float, float]],
) -> None:
    source_x, source_y, source_width, source_height = coordinates[edge.source]
    target_x, target_y, _target_width, target_height = coordinates[edge.target]
    source_mid_y = source_y + source_height / 2
    target_mid_y = target_y + target_height / 2
    if abs(source_mid_y - target_mid_y) < 0.01:
        if target_x > source_x:
            start = (source_x + source_width + 0.003, source_mid_y)
            end = (target_x - 0.003, target_mid_y)
        else:
            start = (source_x - 0.003, source_mid_y)
            end = (target_x + _target_width + 0.003, target_mid_y)
    else:
        if source_mid_y > target_mid_y:
            start = (source_x + source_width / 2, source_y)
            end = (target_x + _target_width / 2, target_y + target_height)
        else:
            start = (source_x + source_width / 2, source_y + source_height)
            end = (target_x + _target_width / 2, target_y)
    axis.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={"arrowstyle": "-|>", "color": _GREY, "linewidth": 0.9},
    )


def _load_specification(path: Path) -> SystemBoundarySpecification:
    try:
        return load_command_model(path, SystemBoundarySpecification)
    except (OSError, ValidationError, ValueError) as error:
        raise SystemBoundaryFigureError(
            f"Could not verify system-boundary specification: {path}: {error}"
        ) from error


def _resolve_generated_at(value: datetime | None, manifest_path: Path) -> datetime:
    if value is not None:
        _require_utc(value)
        return value
    if manifest_path.exists():
        try:
            return SystemBoundaryFigureManifest.model_validate_json(
                manifest_path.read_bytes()
            ).generated_at_utc
        except (OSError, ValidationError) as error:
            raise SystemBoundaryFigureError(
                f"Could not verify system-boundary figure manifest: {manifest_path}"
            ) from error
    return datetime.now(UTC)


def _require_utc(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("System-boundary timestamp must be timezone-aware UTC")
