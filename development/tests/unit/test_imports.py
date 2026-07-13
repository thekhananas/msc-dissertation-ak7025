"""Workspace package-boundary smoke tests."""

import importlib

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
    ],
)
def test_package_boundary_imports(module_name: str) -> None:
    assert importlib.import_module(module_name) is not None
