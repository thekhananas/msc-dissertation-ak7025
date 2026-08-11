"""Shared, semantic styling for figures used in reports and presentations."""

from dataclasses import dataclass
from typing import Final, Literal

FigureProfile = Literal["report", "presentation"]

# These colours are chosen for contrast and are reused by meaning, not by figure.
BLUE: Final = "#0072B2"
ORANGE: Final = "#D55E00"
GREEN: Final = "#009E73"
PURPLE: Final = "#7A5195"
GREY: Final = "#7A858F"
LIGHT_GREY: Final = "#D7DDE2"
INK: Final = "#17212B"
MUTED: Final = "#56616B"


@dataclass(frozen=True)
class FigureStyle:
    """A format profile; colour semantics remain shared between profiles."""

    width_in: float
    height_in: float
    font_size: float
    title_size: float


REPORT_STYLE: Final = FigureStyle(width_in=6.5, height_in=4.5, font_size=8.5, title_size=10.5)
PRESENTATION_STYLE: Final = FigureStyle(
    width_in=12.8,
    height_in=7.2,
    font_size=14.0,
    title_size=18.0,
)


def style_for(profile: FigureProfile) -> FigureStyle:
    """Return the dimensions and type scale for one publication format."""

    return REPORT_STYLE if profile == "report" else PRESENTATION_STYLE
