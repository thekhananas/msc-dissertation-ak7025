"""Shared typed-input and machine-readable-output helpers for benchmark commands."""

import json
import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


def load_command_model[ModelT: BaseModel](
    path: Path,
    model_type: type[ModelT],
) -> ModelT:
    """Load one explicit JSON or YAML path into a strict Pydantic model."""

    try:
        content = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"Could not read command input: {path}") from error
    try:
        if path.suffix.casefold() in {".yaml", ".yml"}:
            raw = yaml.safe_load(content)
        else:
            raw = json.loads(content)
    except (json.JSONDecodeError, yaml.YAMLError) as error:
        raise ValueError(f"Command input is not valid JSON or YAML: {path}") from error
    return model_type.model_validate(raw)


def print_command_result(
    *,
    command: str,
    result: BaseModel | dict[str, object],
    gate_passed: bool = True,
) -> None:
    """Print one stable success or scientific-gate result to stdout."""

    payload = result.model_dump(mode="json") if isinstance(result, BaseModel) else result
    _print_json(
        {
            "command": command,
            "result": payload,
            "status": "ok" if gate_passed else "gate_failed",
        },
        stream=sys.stdout,
    )


def print_command_error(*, command: str, error: Exception) -> None:
    """Print one machine-readable integrity failure to stderr."""

    _print_json(
        {
            "command": command,
            "error_type": type(error).__name__,
            "message": str(error),
            "status": "error",
        },
        stream=sys.stderr,
    )


def _print_json(value: dict[str, object], *, stream: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), file=stream)
