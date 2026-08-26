"""Shared process metrics for acquisition-study runtime measurements."""

import resource
import sys

NANOSECONDS_PER_SECOND = 1_000_000_000


def peak_rss_bytes() -> int:
    """Return the process peak resident set size in bytes."""
    raw_value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw_value if sys.platform == "darwin" else raw_value * 1024


def runtime_seconds(nanoseconds: int) -> float:
    """Convert a stored integer duration to seconds."""
    return nanoseconds / NANOSECONDS_PER_SECOND
