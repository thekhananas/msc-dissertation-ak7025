"""Content-addressed implementation stop for the M7B acquisition study."""

# pyright: reportUnknownArgumentType=false

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Literal, Self

import yaml
from pydantic import Field, ValidationError, model_validator

from socratic_tutor.benchmark.artifacts import artifact_locations, write_immutable_json
from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import canonical_sha256, file_sha256, model_content_hash
from socratic_tutor.contracts import ContractModel
from socratic_tutor.repository_state import current_clean_revision, validate_git_revision


class ImplementationFreezeError(ValueError):
    """The M7B implementation cannot be frozen from the supplied sources."""


class FreezeSourceGroup(ContractModel):
    """Patterns selecting one distinct class of freeze evidence."""

    group_id: str = Field(pattern=r"^[a-z][a-z0-9_]{2,79}$")
    patterns: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_patterns(self) -> Self:
        if len(self.patterns) != len(set(self.patterns)):
            raise ValueError("Freeze patterns must be unique within a group")
        for pattern in self.patterns:
            _validate_relative_pattern(pattern)
        return self


class FreezeLimitation(ContractModel):
    """One limitation retained when implementation ends."""

    limitation_id: str = Field(pattern=r"^[a-z][a-z0-9_]{2,79}$")
    statement: str = Field(min_length=20)


class ImplementationFreezeSpecification(ContractModel):
    """Declared material included in the M7B implementation stop."""

    schema_version: Literal[1] = 1
    schema_id: Literal["acquisition_study.implementation_freeze_specification.v1"] = (
        "acquisition_study.implementation_freeze_specification.v1"
    )
    study_id: Literal["reliability-aware-probing-v1"] = "reliability-aware-probing-v1"
    closed_on: date
    output_path: str = Field(min_length=1)
    source_groups: tuple[FreezeSourceGroup, ...] = Field(min_length=7, max_length=7)
    unresolved_limitations: tuple[FreezeLimitation, ...] = Field(min_length=8, max_length=8)
    deferred_work: tuple[str, ...] = Field(min_length=6)

    @model_validator(mode="after")
    def validate_specification(self) -> Self:
        _validate_relative_pattern(self.output_path, allow_glob=False)
        group_ids = tuple(group.group_id for group in self.source_groups)
        limitation_ids = tuple(item.limitation_id for item in self.unresolved_limitations)
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("Freeze source group identifiers must be unique")
        if len(limitation_ids) != len(set(limitation_ids)):
            raise ValueError("Freeze limitation identifiers must be unique")
        if len(self.deferred_work) != len(set(self.deferred_work)):
            raise ValueError("Deferred work entries must be unique")
        return self


class FrozenFile(ContractModel):
    """Identity and size of one frozen file."""

    path: str = Field(min_length=1)
    sha256: Sha256
    size_bytes: int = Field(ge=0)


class FrozenSourceGroup(ContractModel):
    """Stable digest over one ordered group of files."""

    group_id: str = Field(min_length=1)
    files: tuple[FrozenFile, ...] = Field(min_length=1)
    group_hash: Sha256

    @model_validator(mode="after")
    def validate_group(self) -> Self:
        if tuple(item.path for item in self.files) != tuple(
            sorted(item.path for item in self.files)
        ):
            raise ValueError("Frozen files must use stable path order")
        if self.group_hash != canonical_sha256(
            [item.model_dump(mode="json") for item in self.files]
        ):
            raise ValueError("Frozen source group hash does not match its files")
        return self


class ImplementationFreezeManifest(ContractModel):
    """Final M7B code, evidence, visual, and limitation identity."""

    schema_version: Literal[1] = 1
    schema_id: Literal["acquisition_study.implementation_freeze_manifest.v1"] = (
        "acquisition_study.implementation_freeze_manifest.v1"
    )
    study_id: Literal["reliability-aware-probing-v1"] = "reliability-aware-probing-v1"
    closure_status: Literal["implementation_stopped_results_frozen"] = (
        "implementation_stopped_results_frozen"
    )
    result_classification: Literal[
        "matched_setting_useful_harmful_under_important_misspecification"
    ] = "matched_setting_useful_harmful_under_important_misspecification"
    code_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    closed_on: date
    specification_sha256: Sha256
    source_groups: tuple[FrozenSourceGroup, ...] = Field(min_length=7, max_length=7)
    source_group_count: Literal[7] = 7
    frozen_file_count: int = Field(ge=1)
    unresolved_limitations: tuple[FreezeLimitation, ...] = Field(min_length=8, max_length=8)
    limitations_hash: Sha256
    deferred_work: tuple[str, ...] = Field(min_length=6)
    test_execution_scope: Literal[
        "test_sources_hashed_execution_evidence_remains_in_focused_checks_and_ci"
    ] = "test_sources_hashed_execution_evidence_remains_in_focused_checks_and_ci"
    canonical_simulator_result_frozen: Literal[True] = True
    robust_all_comparator_advantage_supported: Literal[False] = False
    human_learning_claim_supported: Literal[False] = False
    tutoring_efficacy_claim_supported: Literal[False] = False
    cognitive_offloading_claim_supported: Literal[False] = False
    deployed_cost_saving_claim_supported: Literal[False] = False
    network_calls_made: Literal[0] = 0
    sandbox_calls_made: Literal[0] = 0
    manifest_hash: Sha256

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        group_ids = tuple(group.group_id for group in self.source_groups)
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("Frozen source group identifiers must be unique")
        paths = [item.path for group in self.source_groups for item in group.files]
        if len(paths) != len(set(paths)):
            raise ValueError("A frozen file cannot appear in more than one source group")
        if self.source_group_count != len(self.source_groups):
            raise ValueError("Frozen source group count does not reconcile")
        if self.frozen_file_count != len(paths):
            raise ValueError("Frozen file count does not reconcile")
        if self.limitations_hash != canonical_sha256(
            [item.model_dump(mode="json") for item in self.unresolved_limitations]
        ):
            raise ValueError("Freeze limitation hash does not match")
        if self.manifest_hash != model_content_hash(self, exclude={"manifest_hash"}):
            raise ValueError("Implementation freeze manifest hash does not match")
        return self


def load_implementation_freeze_specification(
    path: Path,
) -> ImplementationFreezeSpecification:
    """Load one strict freeze specification."""

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return ImplementationFreezeSpecification.model_validate(raw)
    except OSError as error:
        raise ImplementationFreezeError(f"Could not read freeze specification: {path}") from error
    except (TypeError, ValidationError, yaml.YAMLError) as error:
        raise ImplementationFreezeError(f"Invalid freeze specification: {path}") from error


def freeze_m7b_implementation(
    specification_path: Path,
    *,
    project_root: Path,
    code_revision: str,
) -> ImplementationFreezeManifest:
    """Hash the final M7B implementation and publish one immutable stop record."""

    root = project_root.resolve()
    specification_bytes = _read_file(specification_path, "freeze specification")
    specification = load_implementation_freeze_specification(specification_path)
    groups = _resolve_groups(specification.source_groups, root)
    limitations_hash = canonical_sha256(
        [item.model_dump(mode="json") for item in specification.unresolved_limitations]
    )
    content: dict[str, object] = {
        "schema_version": 1,
        "schema_id": "acquisition_study.implementation_freeze_manifest.v1",
        "study_id": specification.study_id,
        "closure_status": "implementation_stopped_results_frozen",
        "result_classification": (
            "matched_setting_useful_harmful_under_important_misspecification"
        ),
        "code_revision": validate_git_revision(
            code_revision,
            invalid_message="Implementation revision must be a full Git SHA",
            error_factory=ImplementationFreezeError,
        ),
        "closed_on": specification.closed_on,
        "specification_sha256": file_sha256(specification_bytes),
        "source_groups": groups,
        "source_group_count": len(groups),
        "frozen_file_count": sum(len(group.files) for group in groups),
        "unresolved_limitations": specification.unresolved_limitations,
        "limitations_hash": limitations_hash,
        "deferred_work": specification.deferred_work,
        "test_execution_scope": (
            "test_sources_hashed_execution_evidence_remains_in_focused_checks_and_ci"
        ),
        "canonical_simulator_result_frozen": True,
        "robust_all_comparator_advantage_supported": False,
        "human_learning_claim_supported": False,
        "tutoring_efficacy_claim_supported": False,
        "cognitive_offloading_claim_supported": False,
        "deployed_cost_saving_claim_supported": False,
        "network_calls_made": 0,
        "sandbox_calls_made": 0,
    }
    draft = ImplementationFreezeManifest.model_construct(
        _fields_set=set(content),
        **content,
        manifest_hash="0" * 64,
    )
    manifest = ImplementationFreezeManifest.model_validate(
        {**content, "manifest_hash": model_content_hash(draft, exclude={"manifest_hash"})}
    )
    output_path = (root / specification.output_path).resolve()
    _require_within_root(output_path, root)
    write_immutable_json(output_path, manifest)
    return manifest


def _resolve_groups(
    specifications: tuple[FreezeSourceGroup, ...],
    root: Path,
) -> tuple[FrozenSourceGroup, ...]:
    groups: list[FrozenSourceGroup] = []
    all_paths: set[str] = set()
    for specification in specifications:
        selected: dict[str, FrozenFile] = {}
        for pattern in specification.patterns:
            matches = sorted(path for path in root.glob(pattern) if path.is_file())
            if not matches:
                raise ImplementationFreezeError(
                    f"Freeze pattern matched no files: {specification.group_id}: {pattern}"
                )
            for path in matches:
                if path.is_symlink():
                    raise ImplementationFreezeError(f"Freeze sources cannot be symlinks: {path}")
                resolved = path.resolve()
                _require_within_root(resolved, root)
                relative = resolved.relative_to(root).as_posix()
                content = _read_file(resolved, "freeze source")
                selected[relative] = FrozenFile(
                    path=relative,
                    sha256=file_sha256(content),
                    size_bytes=len(content),
                )
        ordered_files = tuple(selected[path] for path in sorted(selected))
        overlaps = all_paths.intersection(selected)
        if overlaps:
            raise ImplementationFreezeError(
                f"Freeze source appears in multiple groups: {sorted(overlaps)[0]}"
            )
        all_paths.update(selected)
        group_hash = canonical_sha256([item.model_dump(mode="json") for item in ordered_files])
        groups.append(
            FrozenSourceGroup(
                group_id=specification.group_id,
                files=ordered_files,
                group_hash=group_hash,
            )
        )
    return tuple(groups)


def _validate_relative_pattern(value: str, *, allow_glob: bool = True) -> None:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value.strip() != value:
        raise ValueError(f"Freeze path must remain project-relative: {value}")
    if not allow_glob and any(character in value for character in "*?[]"):
        raise ValueError(f"Freeze output path cannot contain a glob: {value}")


def _require_within_root(path: Path, root: Path) -> None:
    if not path.is_relative_to(root):
        raise ImplementationFreezeError(f"Freeze path escapes the project root: {path}")


def _read_file(path: Path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise ImplementationFreezeError(f"Could not read {label}: {path}") from error


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPECIFICATION = (
    PROJECT_ROOT / "configs" / "acquisition-study" / "v1-implementation-freeze.yaml"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze the completed M7B implementation")
    parser.add_argument("--specification", type=Path, default=DEFAULT_SPECIFICATION)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        specification = load_implementation_freeze_specification(args.specification)
        manifest = freeze_m7b_implementation(
            args.specification,
            project_root=PROJECT_ROOT,
            code_revision=current_clean_revision(
                PROJECT_ROOT,
                dirty_message=(
                    "Commit or remove all visible changes before freezing the implementation"
                ),
                untracked_files="all",
                required_branch="main",
                operation_name="Implementation freeze",
                invalid_revision_message="Implementation revision must be a full Git SHA",
                error_factory=ImplementationFreezeError,
            ),
        )
        output_path = (PROJECT_ROOT / specification.output_path).resolve()
    except Exception as error:
        print(
            json.dumps(
                {
                    "command": "acquisition-implementation-freeze",
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "status": "error",
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "artifact_locations": artifact_locations(output_path.parent),
                "command": "acquisition-implementation-freeze",
                "result": manifest.model_dump(mode="json"),
                "status": "ok",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0
