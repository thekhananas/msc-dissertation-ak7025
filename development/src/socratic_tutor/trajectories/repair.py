"""Non-destructive recovery for the disposable local demonstration log."""

from datetime import UTC, datetime
from pathlib import Path

from socratic_tutor.trajectories.events import EventLogCorruptionError, JsonlEventStore


def prepare_demo_event_log(path: Path, *, now: datetime | None = None) -> Path | None:
    """Quarantine a corrupt local log and return its preserved path."""

    try:
        JsonlEventStore(path)
    except EventLogCorruptionError:
        timestamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%S%fZ")
        quarantine_path = path.with_name(f"{path.stem}.corrupt-{timestamp}{path.suffix}")
        path.replace(quarantine_path)
        path.touch()
        JsonlEventStore(path)
        return quarantine_path
    return None
