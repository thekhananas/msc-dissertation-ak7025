"""Workspace package-boundary smoke tests."""

import ast
import importlib
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "socratic_tutor.contracts",
        "socratic_tutor.graph",
        "socratic_tutor.tracking",
        "socratic_tutor.policies",
        "socratic_tutor.simulator",
        "socratic_tutor.cognitive",
        "socratic_tutor.generation",
        "socratic_tutor.guardrails",
        "socratic_tutor.trajectories",
        "socratic_tutor.evaluation",
        "socratic_tutor.sandbox",
        "socratic_tutor.tasks",
        "socratic_tutor.turn",
        "socratic_tutor.benchmark",
        "socratic_tutor.benchmark.public",
        "socratic_tutor.benchmark.evaluator",
    ],
)
def test_package_boundary_imports(module_name: str) -> None:
    assert importlib.import_module(module_name) is not None


def test_policy_safe_packages_cannot_import_private_simulator_state() -> None:
    package_root = Path("src/socratic_tutor")
    policy_safe_packages = (
        "contracts",
        "generation",
        "graph",
        "guardrails",
        "policies",
        "tracking",
    )

    violations: list[str] = []
    for package_name in policy_safe_packages:
        for source_path in (package_root / package_name).rglob("*.py"):
            tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports = tuple(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imports = (node.module or "",)
                else:
                    continue
                if any(name.startswith("socratic_tutor.simulator") for name in imports):
                    violations.append(f"{source_path}:{node.lineno}")

    assert violations == []


def test_decision_time_packages_cannot_import_evaluator_benchmark_data() -> None:
    package_root = Path("src/socratic_tutor")
    decision_time_roots = (
        package_root / "benchmark" / "public",
        package_root / "contracts",
        package_root / "generation",
        package_root / "graph",
        package_root / "guardrails",
        package_root / "policies",
        package_root / "simulator",
        package_root / "tasks",
        package_root / "tracking",
        package_root / "turn",
        Path("apps/api"),
    )

    violations: list[str] = []
    for root in decision_time_roots:
        for source_path in root.rglob("*.py"):
            tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports = tuple(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imports = (node.module or "",)
                else:
                    continue
                if any(name.startswith("socratic_tutor.benchmark.evaluator") for name in imports):
                    violations.append(f"{source_path}:{node.lineno}")

    assert violations == []
