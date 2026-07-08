"""Versioned JSONL and Parquet trajectory I/O."""

from socratic_tutor.trajectories.events import (
    EventLogCorruptionError,
    EventType,
    JsonlEventStore,
    SessionCreatedPayload,
    StoredEvent,
    TurnCompletedPayload,
)
from socratic_tutor.trajectories.repair import prepare_demo_event_log

__all__ = [
    "EventLogCorruptionError",
    "EventType",
    "JsonlEventStore",
    "SessionCreatedPayload",
    "StoredEvent",
    "TurnCompletedPayload",
    "prepare_demo_event_log",
]
