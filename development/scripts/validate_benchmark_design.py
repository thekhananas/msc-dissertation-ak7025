"""Validate the benchmark concept, split, and 24-case allocation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from socratic_tutor.benchmark.design import load_design
from socratic_tutor.benchmark.hashing import model_content_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path, help="Path to the benchmark design YAML")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = load_design(args.plan)
    pattern_counts = Counter(case.evidence_pattern.value for case in plan.held_out_cases)
    difficulty_counts = Counter(case.difficulty.value for case in plan.held_out_cases)
    summary = {
        "benchmark_version": plan.benchmark_version,
        "concept_count": len(plan.concepts),
        "design_hash": model_content_hash(plan),
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "evidence_pattern_counts": dict(sorted(pattern_counts.items())),
        "held_out_case_count": len(plan.held_out_cases),
        "misconception_count": sum(len(item.misconception_ids) for item in plan.concepts),
        "status": plan.status.value,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
