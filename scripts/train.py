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


def _build_networks(cfg: DictConfig):
    """Actor / critic sized from env and agent.feature_extractor."""
    from eigen3.models import Actor, DoubleCritic

    def _env(k: str, default):
        return OmegaConf.select(cfg, f"env.{k}", default=default)

    def _agent_fe(k: str, default):
        return OmegaConf.select(cfg, f"agent.feature_extractor.{k}", default=default)

    nc = int(_env("num_columns", 117))
    nf = int(_env("num_features_obs", 5))
    ni = int(_env("num_investable_stocks", 108))
    istart = int(_env("investable_start_col", 9))
    chunk = int(_agent_fe("chunk_size", 64))
    use_remat_a = OmegaConf.select(cfg, "agent.actor_network.use_remat", default=True)
    use_remat_c = OmegaConf.select(cfg, "agent.critic_network.use_remat", default=True)

    actor = Actor(
        num_columns=nc,
        num_features=nf,
        num_investable_stocks=ni,
        investable_start_col=istart,
        column_chunk_size=chunk,
        use_remat=use_remat_a,
    )
    critic = DoubleCritic(
        num_columns=nc,
        num_features=nf,
        num_investable_stocks=ni,
        column_chunk_size=chunk,
        use_remat=use_remat_c,
    )
    return actor, critic


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function: load data, create env, then run EvoRL workflow when integrated."""

    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    logger.info(f"Setting random seed: {cfg.seed}")
    key = jax.random.PRNGKey(cfg.seed)

    data_path = OmegaConf.select(cfg, "env.data_path", default="data/raw")
    path = Path(data_path)
    column_index = OmegaConf.select(cfg, "env.column_index", default=None)

    if path.is_dir() and (path / "data_array.npy").exists():
        logger.info(f"Loading Eigen2 npy bundle from: {data_path}")
        load_kwargs = {}
        if column_index is not None:
            load_kwargs["column_index"] = int(column_index)
            logger.info(f"Using column_index={column_index}")
        data_array, data_array_full, norm_stats = load_trading_data(str(path), **load_kwargs)
    elif path.is_file() and path.suffix.lower() in (".pkl", ".pickle", ".csv"):
        logger.info(f"Loading mono table from: {data_path}")
        mono_ch = int(OmegaConf.select(cfg, "env.mono_num_channels", default=18))
        mono_hdr = OmegaConf.select(cfg, "env.mono_csv_header", default=0)
        data_array, data_array_full, norm_stats = load_trading_data(
            str(path),
            mono_num_channels=mono_ch,
            mono_csv_header=mono_hdr,
        )
    else:
        num_cols = int(OmegaConf.select(cfg, "env.num_columns", default=117))
        num_feat = int(OmegaConf.select(cfg, "env.num_features_obs", default=5))
        logger.info(f"Using synthetic data ({num_cols} columns, F={num_feat})")
        data_array, data_array_full, norm_stats = create_synthetic_data(
            num_days=2000,
            num_columns=num_cols,
            num_features_obs=num_feat,
            seed=cfg.seed,
        )

    logger.info(f"Data shapes: obs {data_array.shape}, full {data_array_full.shape}")

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
        max_positions=_env_cfg("max_positions", 10),
        inaction_penalty=_env_cfg("inaction_penalty", 0.0),
        coefficient_threshold=_env_cfg("coefficient_threshold", 1.0),
        min_sale_target=_env_cfg("min_sale_target", 10.0),
        max_sale_target=_env_cfg("max_sale_target", 50.0),
        investable_start_col=_env_cfg("investable_start_col", 9),
        num_investable_stocks=_env_cfg("num_investable_stocks", 108),
        loss_penalty_multiplier=_env_cfg("loss_penalty_multiplier", 1.0),
        hurdle_rate=_env_cfg("hurdle_rate", 0.006),
        conviction_scaling_power=_env_cfg("conviction_scaling_power", 1.25),
        observation_noise_std=_env_cfg("observation_noise_std", 0.01),
    )

    try:
        state = env.reset(key)
        logger.info(f"Env reset OK; obs shape: {state.obs.shape}")
    except Exception as e:
        logger.warning(f"Env reset failed (evorl may not be installed): {e}")

    try:
        from eigen3.agents import TradingAgent

        actor, critic = _build_networks(cfg)
        agent = TradingAgent(actor_network=actor, critic_network=critic)
        akey = jax.random.PRNGKey(cfg.seed + 1)
        agent_state = agent.init(env.obs_space, env.action_space, akey)
        logger.info(
            "TradingAgent init OK (actor/critic dims match env). "
            "Wire TradingERLWorkflow when ready."
        )
        del agent_state
    except Exception as e:
        logger.warning(f"TradingAgent init skipped or failed: {e}")

    logger.info(
        "Data and env wired. To run full training, instantiate TradingERLWorkflow "
        "(eigen3.workflows.trading_workflow) with env, agent, evaluator, and config."
    )


if __name__ == "__main__":
    main()
