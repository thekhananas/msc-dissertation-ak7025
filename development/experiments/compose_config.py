"""Compose and print the initial Hydra experiment configuration."""

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    """Print a resolved configuration without executing an experiment."""

    print(OmegaConf.to_yaml(config, resolve=True))


if __name__ == "__main__":
    main()
