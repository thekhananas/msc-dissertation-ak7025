"""Fresh-process worker for compact acquisition-stream resource measurements."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from time import perf_counter_ns

from socratic_tutor.acquisition_study.compact_runtime_forecast import CompactRuntimeSample
from socratic_tutor.acquisition_study.compact_stream import (
    AcquisitionStudyPartition,
    iter_compact_policy_records,
)
from socratic_tutor.acquisition_study.plan import load_acquisition_study_plan
from socratic_tutor.acquisition_study.runtime_metrics import peak_rss_bytes
from socratic_tutor.benchmark.hashing import canonical_json_bytes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure one complete compact development stream")
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
        baseline_peak = peak_rss_bytes()
        digest = hashlib.sha256()
        row_count = 0
        case_prediction_count = 0
        output_bytes = 0
        started = perf_counter_ns()
        with tempfile.TemporaryFile(mode="w+b") as handle:
            for record in iter_compact_policy_records(
                specification,
                analysis,
                partition=AcquisitionStudyPartition.DEVELOPMENT,
                development_episode_limit=args.episodes_per_environment,
            ):
                content = canonical_json_bytes(record) + b"\n"
                handle.write(content)
                digest.update(content)
                row_count += 1
                case_prediction_count += record.metric.candidate_count
                output_bytes += len(content)
            handle.flush()
            os.fsync(handle.fileno())
        elapsed = perf_counter_ns() - started
        observed_peak = peak_rss_bytes()
        environment_count = 7
        policy_count = 7
        sample = CompactRuntimeSample(
            repetition_index=args.repetition_index,
            episodes_per_environment=args.episodes_per_environment,
            environment_count=environment_count,
            policy_count=policy_count,
            episode_count=environment_count * args.episodes_per_environment,
            row_count=row_count,
            case_prediction_count=case_prediction_count,
            output_bytes=output_bytes,
            elapsed_nanoseconds=elapsed,
            baseline_peak_rss_bytes=baseline_peak,
            observed_peak_rss_bytes=observed_peak,
            incremental_peak_rss_bytes=observed_peak - baseline_peak,
            stream_file_sha256=digest.hexdigest(),
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


if __name__ == "__main__":
    raise SystemExit(main())
