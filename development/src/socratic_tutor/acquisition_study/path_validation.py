"""Validation helpers for paths stored in acquisition-study artefacts."""

from pathlib import PurePosixPath


def validate_relative_path(value: str) -> None:
    """Require a normal relative POSIX path without parent traversal."""
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value != path.as_posix():
        raise ValueError(f"Path must be a normal relative POSIX path: {value}")
