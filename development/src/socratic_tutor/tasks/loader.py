"""Load versioned task definitions bundled with the package."""

from importlib.resources import files
from typing import Any, cast

import yaml

from socratic_tutor.contracts import TaskDefinition


def load_task(task_id: str = "mutable-list-aliasing") -> TaskDefinition:
    """Load and validate an authored YAML task by stable identifier."""

    if task_id != "mutable-list-aliasing":
        raise KeyError(f"Unknown task: {task_id}")

    task_path = files("socratic_tutor.tasks").joinpath("mutable_list_aliasing.yaml")
    raw = yaml.safe_load(task_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Task definition must be a mapping: {task_path}")
    return TaskDefinition.model_validate(cast(dict[str, Any], raw))
