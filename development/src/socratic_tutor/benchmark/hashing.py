"""Canonical content hashing shared by benchmark authoring and replay."""

import hashlib
import json
from datetime import date, datetime
from enum import Enum
from pathlib import PurePath
from typing import Any

from pydantic import BaseModel

from socratic_tutor.benchmark.common import Sha256


def _json_default(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, PurePath):
        return value.as_posix()
    raise TypeError(f"Unsupported canonical JSON value: {type(value).__name__}")


def canonical_json_bytes(value: object) -> bytes:
    """Serialize a JSON-compatible value with stable ordering and no NaN values."""

    return json.dumps(
        value,
        default=_json_default,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: object) -> Sha256:
    """Return the SHA-256 digest of canonical JSON bytes."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def model_content_hash(model: BaseModel, *, exclude: set[str] | None = None) -> Sha256:
    """Hash a model after excluding self-referential metadata fields."""

    payload: dict[str, Any] = model.model_dump(mode="json", exclude=exclude or set())
    return canonical_sha256(payload)


def file_sha256(content: bytes) -> Sha256:
    """Return a file-compatible SHA-256 digest."""

    return hashlib.sha256(content).hexdigest()
