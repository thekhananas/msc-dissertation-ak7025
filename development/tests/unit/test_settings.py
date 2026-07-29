"""Safe-default and secret-handling tests for process settings."""

from socratic_tutor.settings import Settings


def test_settings_default_to_offline_services() -> None:
    settings = Settings()

    assert settings.template_only is True
    assert settings.wandb_mode == "disabled"
    assert Settings(template_only=False).template_only is False
