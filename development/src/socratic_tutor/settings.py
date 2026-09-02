"""Typed process settings with offline-safe defaults."""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration loaded from environment variables or a local `.env` file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SOCRATIC_",
        # The shared ignored .env also holds benchmark-provider credentials.
        extra="ignore",
        frozen=True,
    )

    environment: Literal["development", "test", "demo"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    template_only: bool = True
    wandb_mode: Literal["disabled", "offline", "online"] = "disabled"
    event_log_path: Path = Path(".local/demo-events.jsonl")
    benchmark_replay_path: Path = Path("data/demo/benchmark-replay-v1.json")
