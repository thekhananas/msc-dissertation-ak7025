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

    if worktree_status(project_root, untracked_files=untracked_files):
        raise error_factory(dirty_message)

    revision = git_head_revision(project_root)
    return validate_git_revision(
        revision,
        invalid_message=invalid_revision_message,
        error_factory=invalid_revision_error_factory or error_factory,
    )


def validate_git_revision(
    value: str,
    *,
    invalid_message: str,
    error_factory: ErrorFactory = ValueError,
    allow_abbreviated: bool = False,
) -> str:
    """Validate a lowercase full SHA, or an explicitly permitted short SHA."""
    pattern = r"[0-9a-f]{7,40}" if allow_abbreviated else r"[0-9a-f]{40}"
    if re.fullmatch(pattern, value) is None:
        raise error_factory(invalid_message)
    return value


def git_head_revision(project_root: Path) -> str:
    """Return the repository's current HEAD without applying policy checks."""
    return _git(project_root, "rev-parse", "HEAD")


def worktree_status(
    project_root: Path,
    *,
    untracked_files: UntrackedFiles = "default",
    pathspecs: tuple[str, ...] = (),
) -> str:
    """Return porcelain status for the whole worktree or selected paths."""
    arguments = ["status", "--porcelain"]
    if untracked_files != "default":
        arguments.append(f"--untracked-files={untracked_files}")
    if pathspecs:
        arguments.extend(("--", *pathspecs))
    return _git(project_root, *arguments)


def _git(project_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(project_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
