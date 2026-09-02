"""Display names and values for acquisition publications; source values stay unchanged."""

from decimal import ROUND_HALF_UP, Decimal

from socratic_tutor.acquisition_study.contracts import PolicyId

COMPARATOR_LABELS = {
    PolicyId.SEEDED_RANDOM_BOUNDED: "Random selection",
    PolicyId.UNCERTAINTY_ONLY_BOUNDED: "Highest uncertainty",
    PolicyId.PLUG_IN_EVSI_BOUNDED: "Plug-in clipped-belief rule",
}


def signed_percentage_points(value: float) -> str:
    points = (Decimal(str(value)) * 100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return f"{points:+.2f}"
