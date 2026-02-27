"""Training script for Eigen3 trading system (Eigen2-aligned)."""

import hydra
from omegaconf import DictConfig, OmegaConf
import jax
import jax.numpy as jnp
from pathlib import Path
import logging

from eigen3.data import load_trading_data, create_synthetic_data
from eigen3.environment.trading_env import TradingEnv

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function: load data, create env, then run EvoRL workflow when integrated."""

    # Print configuration
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    # Set random seed
    logger.info(f"Setting random seed: {cfg.seed}")
    key = jax.random.PRNGKey(cfg.seed)

    # Load data (Eigen2-compatible schema: 117 cols, identity norm)
    data_path = OmegaConf.select(cfg, "env.data_path", default="data/raw")
    path = Path(data_path)
    if path.is_dir() and (path / "data_array.npy").exists():
        logger.info(f"Loading data from: {data_path}")
        data_array, data_array_full, norm_stats = load_trading_data(str(path))
    else:
        logger.info("Using synthetic data (Eigen2 dimensions: 117 columns)")
        num_cols = OmegaConf.select(cfg, "env.num_columns", default=117)
        data_array, data_array_full, norm_stats = create_synthetic_data(
            num_days=2000,
            num_columns=num_cols,
            seed=cfg.seed,
        )

    logger.info(f"Data shapes: obs {data_array.shape}, full {data_array_full.shape}")

    # Create environment (Eigen2-aligned defaults from config)
    logger.info("Creating trading environment...")
    def _env_cfg(key: str, default):
        return OmegaConf.select(cfg, f"env.{key}", default=default)

    env = TradingEnv(
        data_array=data_array,
        data_array_full=data_array_full,
        norm_stats=norm_stats,
        context_window_days=_env_cfg("context_window_days", 151),
        trading_period_days=_env_cfg("trading_period_days", 125),
        settlement_period_days=_env_cfg("settlement_period_days", 30),
        min_holding_period=_env_cfg("min_holding_period", 20),
        max_holding_days=_env_cfg("max_holding_period", 30),
        investable_start_col=_env_cfg("investable_start_col", 9),
        num_investable_stocks=_env_cfg("num_investable_stocks", 108),
    )

    # Workflow: instantiate TradingERLWorkflow with env, agent, evaluator, config when ready
    # For now we only run a quick env sanity check
    try:
        state = env.reset(key)
        logger.info(f"Env reset OK; obs shape: {state.obs.shape}")
    except Exception as e:
        logger.warning(f"Env reset failed (evorl may not be installed): {e}")

    logger.info(
        "Data and env wired. To run full training, instantiate TradingERLWorkflow "
        "(eigen3.workflows.trading_workflow) with env, agent, evaluator, and config."
    )


if __name__ == "__main__":
    main()
