"""Safe-default tests for process settings."""

from pytest import MonkeyPatch

from socratic_tutor.settings import Settings


def test_settings_default_to_offline_services() -> None:
    settings = Settings()

    assert settings.template_only is True
    assert settings.wandb_mode == "disabled"
    assert Settings(template_only=False).template_only is False


def test_provider_secret_does_not_enable_live_evaluation(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("SOCRATIC_CEREBRAS_API_KEY", "local-test-key")
    settings = Settings()

    assert settings.cerebras_api_key is not None
    assert settings.live_evaluation_enabled is False
    assert settings.template_only is True
