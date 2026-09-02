"""Typed process settings with offline-safe defaults."""

from pathlib import Path
from typing import Literal

from pydantic import SecretStr
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
    experiment_summary_path: Path = Path("data/demo/experiment-summary-v1.json")
    live_evaluation_enabled: bool = False
    live_evaluation_case_id: str = "dev-aliasing-001"
    live_evaluation_manifest_path: Path = Path("data/benchmarks/dev-v0/manifest.yaml")
    live_evaluation_system_prompt_path: Path = Path("data/demo/live-evaluation-system-v1.md")
    cerebras_api_key: SecretStr | None = None
