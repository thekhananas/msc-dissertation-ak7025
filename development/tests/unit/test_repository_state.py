"""Behavioural tests for the shared research-output provenance check."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from socratic_tutor.repository_state import current_clean_revision, validate_git_revision


def test_returns_full_revision_for_clean_repository(tmp_path: Path) -> None:
    revision = _initialise_repository(tmp_path)

    actual = current_clean_revision(tmp_path, dirty_message="Repository is dirty")

    assert actual == revision


def test_untracked_file_policy_is_explicit(tmp_path: Path) -> None:
    revision = _initialise_repository(tmp_path)
    (tmp_path / "local-notes.txt").write_text("private notes\n", encoding="utf-8")

    ignored_revision = current_clean_revision(
        tmp_path,
        dirty_message="Tracked files changed",
        untracked_files="no",
    )

    assert ignored_revision == revision
    with pytest.raises(ValueError, match="All visible files must be clean"):
        current_clean_revision(
            tmp_path,
            dirty_message="All visible files must be clean",
            untracked_files="all",
        )


def test_rejects_tracked_changes_and_wrong_branch(tmp_path: Path) -> None:
    _initialise_repository(tmp_path)
    (tmp_path / "tracked.txt").write_text("changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Tracked files changed"):
        current_clean_revision(
            tmp_path,
            dirty_message="Tracked files changed",
            untracked_files="no",
        )

    _git(tmp_path, "restore", "tracked.txt")
    with pytest.raises(ValueError, match="Publication requires branch release, not main"):
        current_clean_revision(
            tmp_path,
            dirty_message="Repository is dirty",
            required_branch="release",
            operation_name="Publication",
        )


def test_rejects_non_full_revision(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_git(_project_root: Path, *arguments: str) -> str:
        return "abc1234" if arguments == ("rev-parse", "HEAD") else ""

    monkeypatch.setattr("socratic_tutor.repository_state._git", fake_git)

    with pytest.raises(ValueError, match="Git did not return a full source revision"):
        current_clean_revision(tmp_path, dirty_message="Repository is dirty")


def test_revision_validation_requires_explicit_abbreviation_permission() -> None:
    full_revision = "a" * 40
    short_revision = "b" * 7

    assert validate_git_revision(full_revision, invalid_message="Invalid revision") == full_revision
    with pytest.raises(ValueError, match="Invalid revision"):
        validate_git_revision(short_revision, invalid_message="Invalid revision")

    assert (
        validate_git_revision(
            short_revision,
            invalid_message="Invalid revision",
            allow_abbreviated=True,
        )
        == short_revision
    )
    with pytest.raises(ValueError, match="Invalid revision"):
        validate_git_revision(
            "c" * 6,
            invalid_message="Invalid revision",
            allow_abbreviated=True,
        )


def _initialise_repository(path: Path) -> str:
    _git(path, "init", "--initial-branch=main")
    _git(path, "config", "user.name", "Repository State Test")
    _git(path, "config", "user.email", "repository-state@example.invalid")
    (path / "tracked.txt").write_text("original\n", encoding="utf-8")
    _git(path, "add", "tracked.txt")
    _git(path, "commit", "-m", "Initial fixture")
    return _git(path, "rev-parse", "HEAD")


def _git(path: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(path), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
