"""Primitive benchmark types safe for both public and evaluator packages."""

from pathlib import PurePosixPath
from typing import Annotated

from pydantic import AfterValidator, StringConstraints

type Sha256 = Annotated[
    str,
    StringConstraints(pattern=r"^[0-9a-f]{64}$"),
]


def _validate_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value.startswith("./") or "\\" in value:
        raise ValueError("Path must be a normalized relative POSIX path")
    if not path.parts or value in {"", "."}:
        raise ValueError("Path cannot be empty")
    return value


type RelativePath = Annotated[str, AfterValidator(_validate_relative_path)]
