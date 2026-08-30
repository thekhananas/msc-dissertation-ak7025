# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
"""Compatibility check for the example output supplied with CSEDM v1.1."""

from __future__ import annotations

import csv
import io
import zipfile
from typing import Literal

from pydantic import Field
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

from socratic_tutor.benchmark.common import Sha256
from socratic_tutor.benchmark.hashing import file_sha256
from socratic_tutor.contracts import ContractModel
from socratic_tutor.csedm_study.analysis_plan import ReferenceBaselineSpecification
from socratic_tutor.csedm_study.features import CSEDMAdapterError

PREDICTIONS_MEMBER = "Example/cv_predict.csv"


class ReferenceMetrics(ContractModel):
    """Three classification metrics published by the dataset authors."""

    accuracy: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)
    cohen_kappa: float = Field(ge=-1.0, le=1.0)


class ReferenceBaselineReport(ContractModel):
    """Public evidence that the supplied example metrics can be reproduced."""

    schema_id: Literal["csedm.reference_baseline_report.v1"] = "csedm.reference_baseline_report.v1"
    schema_version: Literal[1] = 1
    predictions_member: Literal["Example/cv_predict.csv"]
    published_metrics_member: Literal["Example/evaluation_overall.csv"]
    prediction_row_count: int = Field(ge=1)
    predictions_sha256: Sha256
    published_metrics_sha256: Sha256
    expected: ReferenceMetrics
    published: ReferenceMetrics
    recalculated: ReferenceMetrics
    absolute_tolerance: float = Field(gt=0.0)
    passed: Literal[True]
    interpretation: Literal["source_compatibility_check_not_primary_model_reproduction"]


def check_reference_baseline(
    archive: zipfile.ZipFile,
    specification: ReferenceBaselineSpecification,
) -> ReferenceBaselineReport:
    """Recalculate the supplied example metrics without exposing its row data."""

    prediction_bytes = _read_member(archive, PREDICTIONS_MEMBER)
    published_bytes = _read_member(archive, specification.source_member)
    prediction_rows = tuple(
        csv.DictReader(io.StringIO(prediction_bytes.decode("utf-8-sig"), newline=""))
    )
    if not prediction_rows:
        raise CSEDMAdapterError("The supplied CSEDM example has no prediction rows")
    expected_columns = {"FirstCorrect", "prediction"}
    if not expected_columns.issubset(prediction_rows[0]):
        raise CSEDMAdapterError("The supplied CSEDM predictions lack required columns")

    labels = [_boolean(row["FirstCorrect"]) for row in prediction_rows]
    predictions = [_boolean(row["prediction"]) for row in prediction_rows]
    recalculated = ReferenceMetrics(
        accuracy=float(accuracy_score(labels, predictions)),
        f1=float(f1_score(labels, predictions)),
        cohen_kappa=float(cohen_kappa_score(labels, predictions)),
    )
    published = _published_metrics(published_bytes)
    expected = ReferenceMetrics(
        accuracy=specification.expected_accuracy,
        f1=specification.expected_f1,
        cohen_kappa=specification.expected_cohen_kappa,
    )
    for source_name, values in (("published", published), ("recalculated", recalculated)):
        for metric in ("accuracy", "f1", "cohen_kappa"):
            difference = abs(getattr(values, metric) - getattr(expected, metric))
            if difference > specification.absolute_tolerance:
                raise CSEDMAdapterError(
                    f"CSEDM {source_name} {metric} differs from the frozen reference value"
                )

    return ReferenceBaselineReport(
        predictions_member=PREDICTIONS_MEMBER,
        published_metrics_member=specification.source_member,
        prediction_row_count=len(prediction_rows),
        predictions_sha256=file_sha256(prediction_bytes),
        published_metrics_sha256=file_sha256(published_bytes),
        expected=expected,
        published=published,
        recalculated=recalculated,
        absolute_tolerance=specification.absolute_tolerance,
        passed=True,
        interpretation="source_compatibility_check_not_primary_model_reproduction",
    )


def _published_metrics(content: bytes) -> ReferenceMetrics:
    rows = tuple(csv.DictReader(io.StringIO(content.decode("utf-8-sig"), newline="")))
    if len(rows) != 1:
        raise CSEDMAdapterError("The supplied CSEDM metric file must contain one row")
    try:
        return ReferenceMetrics(
            accuracy=float(rows[0]["accuracy"]),
            f1=float(rows[0]["f1"]),
            cohen_kappa=float(rows[0]["kappa"]),
        )
    except (KeyError, ValueError) as error:
        raise CSEDMAdapterError("The supplied CSEDM metric row is invalid") from error


def _read_member(archive: zipfile.ZipFile, member: str) -> bytes:
    try:
        return archive.read(member)
    except KeyError as error:
        raise CSEDMAdapterError(f"CSEDM archive is missing required member: {member}") from error


def _boolean(value: str) -> bool:
    normalised = value.strip().upper()
    if normalised not in {"TRUE", "FALSE"}:
        raise CSEDMAdapterError("CSEDM example labels must be TRUE or FALSE")
    return normalised == "TRUE"
