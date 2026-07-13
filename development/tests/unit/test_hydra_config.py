"""Hydra composition smoke tests."""

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def test_default_configuration_composes() -> None:
    config_directory = Path(__file__).parents[2] / "configs"

    with initialize_config_dir(version_base="1.3", config_dir=str(config_directory)):
        config = compose(config_name="config")

    resolved = OmegaConf.to_container(config, resolve=True)
    assert isinstance(resolved, dict)
    assert resolved["env"]["gymnasium_id"] == "SocraticTutor/POMDP-v0"
    assert resolved["tracker"]["name"] == "simple"
    assert resolved["policy"]["name"] == "heuristic"
    assert resolved["cbfm"]["enabled"] is False
    assert resolved["generation"]["network_enabled"] is False
