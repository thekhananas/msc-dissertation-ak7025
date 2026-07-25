"""Typed process settings with offline-safe defaults."""

from pathlib import Path
from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration loaded from environment variables or a local `.env` file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SOCRATIC_",
        extra="forbid",
        frozen=True,
    )

    environment: Literal["development", "test", "demo"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    template_only: bool = True
    wandb_mode: Literal["disabled", "offline", "online"] = "disabled"
    openrouter_api_key: SecretStr | None = None
    event_log_path: Path = Path(".local/demo-events.jsonl")

    @model_validator(mode="after")
    def require_template_or_model_access(self) -> "Settings":
        if not self.template_only and self.openrouter_api_key is None:
            raise ValueError(
                "SOCRATIC_OPENROUTER_API_KEY is required when template-only mode is disabled"
            )
        return self
