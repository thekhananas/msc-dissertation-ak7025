"""Safe-default and secret-handling tests for process settings."""

from socratic_tutor.settings import Settings


def test_settings_default_to_offline_services() -> None:
    settings = Settings()

    assert settings.template_only is True
    assert settings.wandb_mode == "disabled"
    assert Settings(template_only=False).template_only is False


def test_settings_ignore_provider_secret_owned_by_qualification_runner() -> None:
    settings = Settings.model_validate({"SOCRATIC_CEREBRAS_API_KEY": "local-test-key"})

    assert settings.template_only is True
