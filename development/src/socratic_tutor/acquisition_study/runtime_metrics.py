"""Shared process metrics for acquisition-study runtime measurements."""

import resource
import sys


def peak_rss_bytes() -> int:
    """Return the process peak resident set size in bytes."""
    raw_value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw_value if sys.platform == "darwin" else raw_value * 1024
