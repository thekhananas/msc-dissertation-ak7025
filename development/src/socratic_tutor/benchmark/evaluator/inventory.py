"""Content-addressed file inventory construction and verification."""

import hmac
from pathlib import Path

from socratic_tutor.benchmark.evaluator.models import ArtifactClass, FileInventoryEntry
from socratic_tutor.benchmark.hashing import file_sha256


class InventoryValidationError(RuntimeError):
    """Raised when one or more authored files differ from their inventory."""

    def __init__(self, violations: tuple[str, ...]) -> None:
        self.violations = violations
        super().__init__("; ".join(violations))


def _resolve_inventory_path(root: Path, relative_path: str) -> Path:
    resolved_root = root.resolve()
    resolved = (root / relative_path).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError("Inventory path escapes the benchmark root")
    return resolved


def inventory_file(
    root: Path,
    relative_path: str,
    artifact_class: ArtifactClass,
    schema_id: str,
) -> FileInventoryEntry:
    """Build one inventory row from immutable file bytes."""

    path = _resolve_inventory_path(root, relative_path)
    content = path.read_bytes()
    return FileInventoryEntry(
        path=relative_path,
        artifact_class=artifact_class,
        schema_id=schema_id,
        sha256=file_sha256(content),
        byte_size=len(content),
    )


def verify_inventory(
    root: Path,
    entries: tuple[FileInventoryEntry, ...],
    *,
    ignored_paths: frozenset[str] = frozenset(),
) -> None:
    """Verify that every inventoried file exists with the exact bytes and size."""

    violations: list[str] = []
    expected_paths = {entry.path for entry in entries}
    actual_paths = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    for undeclared_path in sorted(actual_paths - expected_paths - ignored_paths):
        violations.append(f"untracked:{undeclared_path}")
    for entry in entries:
        path = _resolve_inventory_path(root, entry.path)
        if not path.is_file():
            violations.append(f"missing:{entry.path}")
            continue
        content = path.read_bytes()
        if len(content) != entry.byte_size:
            violations.append(f"size:{entry.path}")
        actual_hash = file_sha256(content)
        if not hmac.compare_digest(actual_hash, entry.sha256):
            violations.append(f"sha256:{entry.path}")
    if violations:
        raise InventoryValidationError(tuple(violations))
