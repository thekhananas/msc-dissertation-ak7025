"""Structural validation independent of tracker, policy, and generated outcomes."""

from collections import defaultdict

from socratic_tutor.benchmark.evaluator.models import (
    ArtifactClass,
    AuthoredBenchmarkManifest,
    ManifestStatus,
    ReviewDecision,
    authored_manifest_content_hash,
)
from socratic_tutor.benchmark.hashing import model_content_hash
from socratic_tutor.benchmark.public.models import BenchmarkCaseView, BenchmarkSplit


class BenchmarkValidationError(RuntimeError):
    """Raised when authored definitions violate benchmark structure."""

    def __init__(self, violations: tuple[str, ...]) -> None:
        self.violations = violations
        super().__init__("; ".join(violations))


def case_content_hash(case: BenchmarkCaseView) -> str:
    """Hash a public case without its self-referential content hash."""

    return model_content_hash(case, exclude={"case_content_hash"})


def validate_manifest_structure(manifest: AuthoredBenchmarkManifest) -> None:
    """Validate references, split isolation, reviews, and frozen matrix size."""

    violations: list[str] = []
    inventory = {entry.path: entry for entry in manifest.files}
    family_roles: dict[str, set[str]] = defaultdict(set)
    for split in manifest.splits:
        for family in split.task_families:
            family_roles[family].add(split.split.value)
    for family, roles in family_roles.items():
        if len(roles) > 1:
            violations.append(f"task-family-overlap:{family}")

    expected_public_classes = {
        "public_fixture_ref": ArtifactClass.PUBLIC_FIXTURE,
        "evidence_probe_ref": ArtifactClass.EVIDENCE_PROBE,
        "evidence_test_ref": ArtifactClass.EVIDENCE_TEST,
        "initial_tracker_ref": ArtifactClass.INITIAL_TRACKER,
        "unrelated_control_ref": ArtifactClass.CONTROL,
        "corruption_spec_ref": ArtifactClass.CONTROL,
    }
    expected_private_classes = {
        "criterion_prompt_ref": ArtifactClass.CRITERION_PROBE,
        "test_bundle_ref": ArtifactClass.CRITERION_TEST,
        "rubric_ref": ArtifactClass.CRITERION_RUBRIC,
        "label_rationale_ref": ArtifactClass.LABEL_RATIONALE,
        "review_record_ref": ArtifactClass.REVIEW,
        "structural_difference_record_ref": ArtifactClass.STRUCTURAL_DIFFERENCE,
    }
    split_lookup = {
        (split.split.value, family) for split in manifest.splits for family in split.task_families
    }
    reviews_by_case = {review.case_id: review for review in manifest.reviews}
    generation_prompt_entry = inventory.get(manifest.generation.system_prompt_ref)
    if (
        generation_prompt_entry is None
        or generation_prompt_entry.artifact_class is not ArtifactClass.PROMPT
    ):
        violations.append("generation-prompt-ref")
    elif generation_prompt_entry.sha256 != manifest.generation.system_prompt_sha256:
        violations.append("generation-prompt-hash")

    for item in manifest.cases:
        public = item.public
        private = item.criterion
        if public.benchmark_version != manifest.benchmark_version:
            violations.append(f"benchmark-version:{public.case_id}")
        if (public.split.value, public.task_family) not in split_lookup:
            violations.append(f"split-membership:{public.case_id}")
        if public.case_content_hash != case_content_hash(public):
            violations.append(f"case-hash:{public.case_id}")
        for field_name, artifact_class in expected_public_classes.items():
            path = getattr(public, field_name)
            entry = inventory.get(path)
            if entry is None or entry.artifact_class is not artifact_class:
                violations.append(f"public-ref:{public.case_id}:{field_name}")
        for field_name, artifact_class in expected_private_classes.items():
            path = getattr(private, field_name)
            entry = inventory.get(path)
            if entry is None or entry.artifact_class is not artifact_class:
                violations.append(f"private-ref:{public.case_id}:{field_name}")
        tracker_entry = inventory.get(public.initial_tracker_ref)
        if tracker_entry is not None and tracker_entry.sha256 != public.initial_tracker_hash:
            violations.append(f"initial-tracker-hash:{public.case_id}")
        test_entry = inventory.get(private.test_bundle_ref)
        if test_entry is not None and test_entry.sha256 != private.test_bundle_sha256:
            violations.append(f"criterion-test-hash:{public.case_id}")
        rubric_entry = inventory.get(private.rubric_ref)
        if rubric_entry is not None and rubric_entry.sha256 != private.rubric_sha256:
            violations.append(f"criterion-rubric-hash:{public.case_id}")

        review = reviews_by_case.get(public.case_id)
        if review is None:
            violations.append(f"missing-review:{public.case_id}")
            continue
        required_review_paths = {
            public.public_fixture_ref,
            public.evidence_probe_ref,
            public.evidence_test_ref,
            public.initial_tracker_ref,
            public.unrelated_control_ref,
            public.corruption_spec_ref,
            manifest.generation.system_prompt_ref,
            private.criterion_prompt_ref,
            private.test_bundle_ref,
            private.rubric_ref,
            private.label_rationale_ref,
            private.structural_difference_record_ref,
        }
        reviewed_paths = {reviewed.path for reviewed in review.reviewed_files}
        if not required_review_paths.issubset(reviewed_paths):
            violations.append(f"review-scope:{public.case_id}")

    for review in manifest.reviews:
        if review.case_id not in {item.public.case_id for item in manifest.cases}:
            violations.append(f"orphan-review:{review.review_id}")
        for reviewed in review.reviewed_files:
            entry = inventory.get(reviewed.path)
            if entry is None or entry.sha256 != reviewed.sha256:
                violations.append(f"review-hash:{review.review_id}:{reviewed.path}")

    limitations_entry = inventory.get(manifest.limitations_ref)
    if (
        limitations_entry is None
        or limitations_entry.artifact_class is not ArtifactClass.LIMITATIONS
    ):
        violations.append("limitations-ref")

    if manifest.status is ManifestStatus.FROZEN:
        held_out_cases = tuple(
            item.public for item in manifest.cases if item.public.split is BenchmarkSplit.HELD_OUT
        )
        if len(held_out_cases) < 24:
            violations.append("frozen-case-count")
        if manifest.manifest_hash is None:
            violations.append("frozen-manifest-hash")
        elif manifest.manifest_hash != authored_manifest_content_hash(manifest):
            violations.append("frozen-manifest-hash-mismatch")
        for review in manifest.reviews:
            if review.decision is not ReviewDecision.APPROVED:
                violations.append(f"frozen-review:{review.review_id}")
            if review.author.casefold() == review.reviewer.casefold():
                violations.append(f"frozen-reviewer-independence:{review.review_id}")
        concepts: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        held_out_case_ids = {case.case_id for case in held_out_cases}
        for item in manifest.cases:
            if item.public.case_id not in held_out_case_ids:
                continue
            concepts[item.public.target_concept][item.criterion.misconception_id].add(
                item.criterion.transfer_case_id
            )
        if len(concepts) < 4:
            violations.append("frozen-concept-count")
        for concept, misconceptions in concepts.items():
            if len(misconceptions) < 2:
                violations.append(f"frozen-misconceptions:{concept}")
            for misconception, transfers in misconceptions.items():
                if len(transfers) < 3:
                    violations.append(f"frozen-transfers:{concept}:{misconception}")

    if violations:
        raise BenchmarkValidationError(tuple(violations))
