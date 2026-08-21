from __future__ import annotations

from pathlib import Path

import pytest

from socratic_tutor.acquisition_study.calibration import estimate_probe_reliability
from socratic_tutor.acquisition_study.canonical_budget_curve import (
    CanonicalBudgetCurveError,
    CanonicalBudgetCurveRecord,
    write_verified_canonical_budget_curve_stream,
)
from socratic_tutor.acquisition_study.canonical_stream import (
    CanonicalPublicPolicyMetric,
    project_canonical_episode,
)
from socratic_tutor.acquisition_study.plan import (
    EvaluationEnvironmentId,
    load_acquisition_study_plan,
)
from socratic_tutor.acquisition_study.runner import run_episode_policy_bundle
from socratic_tutor.benchmark.hashing import canonical_json_bytes, file_sha256

ROOT = Path(__file__).parents[2]
SPECIFICATION, ANALYSIS = load_acquisition_study_plan(
    ROOT / "configs" / "acquisition-study" / "v1-environments.yaml",
    ROOT / "configs" / "acquisition-study" / "v1-analysis.yaml",
)


def _write_public_fixture(path: Path) -> bytes:
    calibration = estimate_probe_reliability(SPECIFICATION)
    content = bytearray()
    for environment_id in EvaluationEnvironmentId:
        run = run_episode_policy_bundle(
            SPECIFICATION,
            ANALYSIS,
            calibration,
            environment_id=environment_id,
            episode_index=1,
        )
        _, rows = project_canonical_episode(run, ANALYSIS)
        for row in rows:
            content.extend(canonical_json_bytes(row) + b"\n")
    frozen = bytes(content)
    path.write_bytes(frozen)
    return frozen


def test_budget_stream_reproduces_sealed_primary_rows_without_truth(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "canonical_public.jsonl"
    source = _write_public_fixture(source_path)

    first, replay = write_verified_canonical_budget_curve_stream(
        SPECIFICATION,
        ANALYSIS,
        source_public_path=source_path,
        expected_source_hash=file_sha256(source),
        episode_indices=(1,),
        output_path=tmp_path / "budget.jsonl",
    )

    output = (tmp_path / "budget.jsonl").read_bytes()
    first_record = CanonicalBudgetCurveRecord.model_validate_json(output.splitlines()[0])
    assert first == replay
    assert first.source_rows == 49
    assert first.budget_rows == 175
    assert first.primary_parity_checks == 35
    assert first.selected_probe_count == 3_500
    assert first_record.metric.episode_index == 1
    assert b"latent_success" not in output
    assert b"privileged_episode" not in output


def test_budget_stream_rejects_a_changed_primary_result(tmp_path: Path) -> None:
    source_path = tmp_path / "canonical_public.jsonl"
    source = _write_public_fixture(source_path)
    lines = source.splitlines()
    first = CanonicalPublicPolicyMetric.model_validate_json(lines[0]).model_copy(
        update={"policy_result_hash": "0" * 64}
    )
    lines[0] = canonical_json_bytes(first)
    changed = b"\n".join(lines) + b"\n"
    source_path.write_bytes(changed)

    with pytest.raises(CanonicalBudgetCurveError, match="50% result differs"):
        write_verified_canonical_budget_curve_stream(
            SPECIFICATION,
            ANALYSIS,
            source_public_path=source_path,
            expected_source_hash=file_sha256(changed),
            episode_indices=(1,),
            output_path=tmp_path / "budget.jsonl",
        )
