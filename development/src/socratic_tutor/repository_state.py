"""Shared Git repository checks for reproducible research outputs."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Literal

UntrackedFiles = Literal["default", "all", "no"]
ErrorFactory = Callable[[str], Exception]


def current_clean_revision(
    project_root: Path,
    *,
    dirty_message: str,
    untracked_files: UntrackedFiles = "default",
    required_branch: str | None = None,
    operation_name: str = "Repository operation",
    invalid_revision_message: str = "Git did not return a full source revision",
    error_factory: ErrorFactory = ValueError,
    invalid_revision_error_factory: ErrorFactory | None = None,
) -> str:
    """Return HEAD after enforcing the caller's branch and worktree rules."""
    if required_branch is not None:
        branch = _git(project_root, "branch", "--show-current")
        if branch != required_branch:
            actual_branch = branch or "detached HEAD"
            raise error_factory(
                f"{operation_name} requires branch {required_branch}, not {actual_branch}"
            )

    status_arguments = ["status", "--porcelain"]
    if untracked_files != "default":
        status_arguments.append(f"--untracked-files={untracked_files}")
    if _git(project_root, *status_arguments):
        raise error_factory(dirty_message)

    revision = _git(project_root, "rev-parse", "HEAD")
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        revision_error = invalid_revision_error_factory or error_factory
        raise revision_error(invalid_revision_message)
    return revision


def _git(project_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(project_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
