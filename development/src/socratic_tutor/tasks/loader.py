"""Load versioned task definitions bundled with the package."""

from importlib.resources import files
from typing import Any, cast

import yaml

from socratic_tutor.contracts import TaskDefinition

_TASK_FILES = {
    "mutable-list-aliasing": "mutable_list_aliasing.yaml",
    "none-versus-falsy": "none_versus_falsy.yaml",
}


def load_task(task_id: str = "mutable-list-aliasing") -> TaskDefinition:
    """Load and validate an authored YAML task by stable identifier."""

    try:
        task_filename = _TASK_FILES[task_id]
    except KeyError as error:
        raise KeyError(f"Unknown task: {task_id}") from error

    task_path = files("socratic_tutor.tasks").joinpath(task_filename)
    raw = yaml.safe_load(task_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Task definition must be a mapping: {task_path}")
    task = TaskDefinition.model_validate(cast(dict[str, Any], raw))
    if task.task_id != task_id:
        raise ValueError(f"Task registry ID does not match definition: {task_path}")
    return task


def list_tasks() -> tuple[TaskDefinition, ...]:
    """Return all student-safe practice tasks in stable display order."""

    tasks = tuple(load_task(task_id) for task_id in _TASK_FILES)
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("Task registry contains duplicate task IDs")
    return tasks
