"""Training script for Eigen3 trading system (Hydra entry point)."""

import hydra
from omegaconf import DictConfig

from eigen3.entrypoints.training import run_training


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Load config, run data setup and ``TradingERLWorkflow`` training."""
    run_training(cfg)


if __name__ == "__main__":
    main()
