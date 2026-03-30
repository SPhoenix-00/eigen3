"""Training script for Eigen3 trading system (Hydra entry point)."""

import sys
from pathlib import Path

# Repo root on path: script dir is ``scripts/``, not the package parent.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import hydra
from omegaconf import DictConfig

from eigen3.entrypoints.training import run_training


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Load config, run data setup and ``TradingERLWorkflow`` training."""
    run_training(cfg)


if __name__ == "__main__":
    main()
