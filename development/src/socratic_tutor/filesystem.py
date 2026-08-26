"""Small filesystem primitives shared by research artifact writers."""

import os
from pathlib import Path

_READ_CHUNK_SIZE = 1024 * 1024


def files_equal(left: Path, right: Path) -> bool:
    """Compare two files without loading either file fully into memory."""
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as left_handle, right.open("rb") as right_handle:
        while True:
            left_chunk = left_handle.read(_READ_CHUNK_SIZE)
            right_chunk = right_handle.read(_READ_CHUNK_SIZE)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def fsync_directory(path: Path) -> None:
    """Persist directory-entry changes after an atomic file replacement."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
