"""Small filesystem primitives shared by research artifact writers."""

import os
from pathlib import Path


def fsync_directory(path: Path) -> None:
    """Persist directory-entry changes after an atomic file replacement."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
