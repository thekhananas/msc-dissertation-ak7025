"""Validate the local demo log and preserve any corrupt copy before startup."""

from socratic_tutor.settings import Settings
from socratic_tutor.trajectories import prepare_demo_event_log


def main() -> None:
    path = Settings().event_log_path
    quarantine_path = prepare_demo_event_log(path)
    if quarantine_path is None:
        print(f"Demo event log is valid: {path}")
        return
    print(f"Preserved corrupt demo event log: {quarantine_path}")
    print(f"Started a new demo event log: {path}")


if __name__ == "__main__":
    main()
