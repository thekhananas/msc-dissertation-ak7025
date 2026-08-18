"""Fresh-process worker used by the acquisition runtime forecast."""

from __future__ import annotations

import argparse
import gc
import json
import resource
import sys
from pathlib import Path
from time import perf_counter_ns

from socratic_tutor.acquisition_study.plan import load_acquisition_study_plan
from socratic_tutor.acquisition_study.runner import run_development_policy_matrix
from socratic_tutor.acquisition_study.runtime_forecast import AcquisitionRuntimeSample


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure one acquisition development matrix")
    parser.add_argument("--environments", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--episodes-per-environment", type=int, required=True)
    parser.add_argument("--repetition-index", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        specification, analysis = load_acquisition_study_plan(
            args.environments,
            args.analysis,
        )
        gc.collect()
        baseline_peak = _peak_rss_bytes()
        started = perf_counter_ns()
        matrix = run_development_policy_matrix(
            specification,
            analysis,
            episodes_per_environment=args.episodes_per_environment,
        )
        elapsed = perf_counter_ns() - started
        observed_peak = _peak_rss_bytes()
        policy_result_count = sum(len(row.results) for row in matrix.comparisons)
        sample = AcquisitionRuntimeSample(
            repetition_index=args.repetition_index,
            episodes_per_environment=matrix.episodes_per_environment,
            comparison_count=matrix.comparison_count,
            policy_result_count=policy_result_count,
            case_prediction_count=sum(
                len(result.case_results)
                for comparison in matrix.comparisons
                for result in comparison.results
            ),
            elapsed_nanoseconds=elapsed,
            baseline_peak_rss_bytes=baseline_peak,
            observed_peak_rss_bytes=observed_peak,
            incremental_peak_rss_bytes=observed_peak - baseline_peak,
            matrix_content_hash=matrix.content_hash,
        )
    except Exception as error:
        print(
            json.dumps(
                {
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
    print(sample.model_dump_json(indent=2))
    return 0


def _peak_rss_bytes() -> int:
    raw_value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw_value if sys.platform == "darwin" else raw_value * 1024


if __name__ == "__main__":
    raise SystemExit(main())
