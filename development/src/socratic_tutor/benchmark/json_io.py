"""Shared JSON loading primitives for benchmark artefacts."""

from __future__ import annotations

import json
from pathlib import Path

from socratic_tutor.contracts import ContractModel


def load_json_model[ModelT: ContractModel](path: Path, model: type[ModelT]) -> ModelT:
    """Load and validate one JSON artefact as a contract model."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read JSON artifact: {path}") from error
    return model.model_validate(raw)
