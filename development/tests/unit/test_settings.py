"""Safe-default and secret-handling tests for process settings."""

import pytest
from pydantic import SecretStr, ValidationError

from socratic_tutor.settings import Settings


def test_settings_default_to_offline_services() -> None:
    settings = Settings()

    assert settings.template_only is True
    assert settings.wandb_mode == "disabled"
    assert settings.openrouter_api_key is None


def test_model_access_requires_a_key() -> None:
    with pytest.raises(ValidationError, match="OPENROUTER_API_KEY"):
        Settings(template_only=False)


def test_openrouter_key_remains_redacted() -> None:
    settings = Settings(
        template_only=False,
        openrouter_api_key=SecretStr("not-a-real-key"),
    )

    assert "not-a-real-key" not in repr(settings)
