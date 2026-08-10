# pyright: reportPrivateUsage=false
"""Tests for the final claim boundary after the harness correction."""

import csv
from io import StringIO

from socratic_tutor.benchmark.harness_correction_analysis import (
    HarnessCorrectionAnalysisReport,
)
from socratic_tutor.benchmark.harness_correction_closure import (
    CLAIM_ORDER,
    OUTPUT_ORDER,
    _build_claims,
    _build_prior_outputs,
    _claim_table_bytes,
)
from socratic_tutor.benchmark.statistics import PairedCaseInference


def test_claims_keep_corrected_benchmark_and_human_claims_separate() -> None:
    analysis = HarnessCorrectionAnalysisReport.model_construct(
        corrected_primary_effect=1 / 23,
        valid_vs_unrelated_inference=_inference(0.0),
        valid_vs_corrupted_inference=_inference(15 / 23),
    )

    claims = _build_claims(analysis)

    assert tuple(claim.claim_id for claim in claims) == CLAIM_ORDER
    assert claims[0].status == "not_supported_after_complete_correction"
    assert claims[0].estimate == 1 / 23
    assert claims[1].status == "not_supported_after_complete_correction"
    assert claims[1].estimate == 0.0
    assert claims[2].status == "descriptive_post_hoc_only"
    assert all(claim.status == "outside_study_scope" for claim in claims[3:])


def test_prior_outputs_do_not_discard_the_original_sealed_result() -> None:
    outputs = _build_prior_outputs()

    assert tuple(output.output_id for output in outputs) == OUTPUT_ORDER
    statuses = {output.output_id: output.status for output in outputs}
    assert statuses["sealed_primary_result"] == "historical_context_only"
    assert statuses["three_case_probe_correction"] == ("superseded_for_final_interpretation")
    assert statuses["resource_figure_v1"] == "superseded_for_final_interpretation"


def test_claim_table_is_complete_and_plainly_labelled() -> None:
    analysis = HarnessCorrectionAnalysisReport.model_construct(
        corrected_primary_effect=1 / 23,
        valid_vs_unrelated_inference=_inference(0.0),
        valid_vs_corrupted_inference=_inference(15 / 23),
    )

    rows = tuple(
        csv.DictReader(StringIO(_claim_table_bytes(_build_claims(analysis)).decode("utf-8")))
    )

    assert len(rows) == 6
    assert tuple(row["claim_id"] for row in rows) == CLAIM_ORDER
    assert rows[0]["estimate"] == "0.043478"
    assert rows[3]["estimate"] == ""
    assert "human students" in rows[3]["plain_language"]


def _inference(effect: float) -> PairedCaseInference:
    return PairedCaseInference.model_construct(mean_case_effect=effect)
