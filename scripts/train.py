"""Training script for Eigen3 trading system (Eigen2-aligned)."""

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import logging
from typing import Optional

from eigen3.data import load_trading_data, create_synthetic_data
from eigen3.data.splits import compute_train_val_holdout_split, slice_trading_timeline
from eigen3.config import (
    DEFAULT_CONVICTION_SCALING_POWER,
    DEFAULT_EPISODE_REWARD_MULTIPLIER,
    DEFAULT_HURDLE_RATE,
    DEFAULT_LOSS_PENALTY_MULTIPLIER,
)
from eigen3.environment.trading_env import TradingEnv

logger = logging.getLogger(__name__)


def _short_class(path: Optional[str]) -> str:
    if not path:
        return "?"
    s = str(path)
    return s.rsplit(".", maxsplit=1)[-1]


def _run_config_summary(cfg: DictConfig) -> str:
    """Compact, human-readable startup summary (full YAML stays at DEBUG)."""

    def S(key: str, default=None):
        return OmegaConf.select(cfg, key, default=default)

    lines: list[str] = [
        "",
        "──────── Run ────────",
        f"  experiment: {S('experiment_name', '?')!s}   run: {S('run_name', '?')!s}",
        f"  seed: {S('seed', '?')!s}   device: {S('device', '?')!s}   "
        f"jit: {S('enable_jit', True)!s}   pmap: {S('enable_pmap', False)!s}",
    ]
    try:
        from hydra.core.hydra_config import HydraConfig

        out = HydraConfig.get().runtime.output_dir
        lines.append(f"  hydra output: {out}")
    except Exception:
        pass

    lines.extend(
        [
            "──────── Env ────────",
            f"  data_path: {S('env.data_path', '?')!s}",
            f"  context: {S('env.context_window_days', '?')!s} d   "
            f"trading: {S('env.trading_period_days', '?')!s} d   "
            f"settlement: {S('env.settlement_period_days', 0)!s} d",
            f"  val_reserve: ×{S('env.validation_reserve_multiplier', '?')!s}   "
            f"columns: {S('env.num_columns', '?')!s}   "
            f"F_obs: {S('env.num_features_obs', '?')!s}   "
            f"investable: {S('env.num_investable_stocks', '?')!s}",
            "──────── Agent ───────",
            f"  workflow: {_short_class(S('agent.workflow_cls'))}",
            f"  actor_lr: {S('agent.optimizer.actor_lr', '?')!s}   "
            f"critic_lr: {S('agent.optimizer.critic_lr', '?')!s}   "
            f"mixed_precision: {S('agent.use_mixed_precision', False)!s}",
            "──────── Population ─",
            f"  pop_size: {S('population.pop_size', '?')!s}   "
            f"generations: {S('population.total_generations', '?')!s}   "
            f"batch: {S('population.batch_size', '?')!s}",
            f"  replay: {S('population.replay_buffer_size', '?')!s}   "
            f"grad_steps/gen: {S('population.gradient_steps_per_gen', '?')!s}   "
            f"eval_episodes: {S('population.eval_episodes', '?')!s}",
            f"  gauntlet: {S('population.gauntlet_enabled', '?')!s}",
            "──────── Logging ─────",
            f"  console: {S('logging.console_log_level', 'INFO')!s}   "
            f"tensorboard: {S('logging.use_tensorboard', False)!s}   "
            f"wandb: {S('logging.wandb_project', '?')!s} "
            f"({S('logging.wandb_mode', '?')!s})",
            "──────────────────────",
            "  Full config: Hydra run dir → .hydra/config.yaml (+ overrides.yaml).",
            "",
        ]
    )
    return "\n".join(lines)


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

    min_st = float(_env("min_sale_target", 10.0))
    max_st = float(_env("max_sale_target", 50.0))

    actor = Actor(
        num_columns=nc,
        num_features=nf,
        num_investable_stocks=ni,
        investable_start_col=istart,
        column_chunk_size=chunk,
        use_remat=use_remat_a,
        min_sale_target=min_st,
        max_sale_target=max_st,
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

    logger.info(_run_config_summary(cfg))
    logger.debug("Full resolved configuration:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    logger.info("Setting random seed: %s", cfg.seed)
    key = jax.random.PRNGKey(cfg.seed)

    data_path = OmegaConf.select(cfg, "env.data_path", default="data/raw")
    path = Path(to_absolute_path(str(data_path)))
    column_index = OmegaConf.select(cfg, "env.column_index", default=None)

    if path.is_dir() and (path / "data_array.npy").exists():
        logger.info(f"Loading Eigen2 npy bundle from: {data_path}")
        load_kwargs = {}
        if column_index is not None:
            load_kwargs["column_index"] = int(column_index)
            logger.info(f"Using column_index={column_index}")
        data_array, data_array_full, norm_stats, dates_ordinal = load_trading_data(str(path), **load_kwargs)
    elif path.is_file() and path.suffix.lower() in (".pkl", ".pickle", ".csv"):
        logger.info(f"Loading mono table from: {data_path}")
        mono_ch = int(OmegaConf.select(cfg, "env.mono_num_channels", default=18))
        mono_hdr = OmegaConf.select(cfg, "env.mono_csv_header", default=0)
        data_array, data_array_full, norm_stats, dates_ordinal = load_trading_data(
            str(path),
            mono_num_channels=mono_ch,
            mono_csv_header=mono_hdr,
        )
    else:
        num_cols = int(OmegaConf.select(cfg, "env.num_columns", default=117))
        num_feat = int(OmegaConf.select(cfg, "env.num_features_obs", default=5))
        logger.info(f"Using synthetic data ({num_cols} columns, F={num_feat})")
        data_array, data_array_full, norm_stats, dates_ordinal = create_synthetic_data(
            num_days=2000,
            num_columns=num_cols,
            num_features_obs=num_feat,
            seed=cfg.seed,
        )

    logger.info(f"Data shapes: obs {data_array.shape}, full {data_array_full.shape}")

    def _env_cfg(key: str, default):
        return OmegaConf.select(cfg, f"env.{key}", default=default)

    logger.info("Computing train / validation / holdout timeline...")
    dates_np = np.asarray(dates_ordinal, dtype=np.int64).reshape(-1)
    num_days = int(data_array.shape[0])
    ctx = int(_env_cfg("context_window_days", 151))
    trading_days = int(_env_cfg("trading_period_days", 364))
    settlement = int(_env_cfg("settlement_period_days", 0))
    ep_cal = OmegaConf.select(cfg, "env.episode_calendar_days", default=None)
    ep_cal_i = int(ep_cal) if ep_cal is not None else trading_days
    val_mult = float(_env_cfg("validation_reserve_multiplier", 1.5))

    split = compute_train_val_holdout_split(
        num_days=num_days,
        dates_ordinal=dates_np,
        context_window_days=ctx,
        episode_calendar_days=ep_cal_i,
        settlement_period_days=settlement,
        validation_reserve_multiplier=val_mult,
    )
    logger.info(
        f"Split: train rows [0, {split.train_end}), "
        f"val trading band [{split.val_start}, {split.val_end}), "
        f"val env rows [{split.val_env_start}, {split.val_end}) "
        f"(context may use [{split.val_env_start}, {split.val_start})), "
        f"holdout trading [{split.holdout_start}, {split.holdout_end}), "
        f"holdout env [{split.holdout_env_start}, {split.holdout_end}) "
        f"(final episode start index: {split.last_episode_start})"
    )

    train_obs, train_full, dates_train = slice_trading_timeline(
        data_array, data_array_full, dates_np, 0, split.train_end
    )
    val_obs, val_full, dates_val = slice_trading_timeline(
        data_array, data_array_full, dates_np, split.val_env_start, split.val_end
    )

    logger.info("Creating trading environments (train + validation; holdout excluded)...")

    def _make_env(data_obs, data_f, dates_ord, *, is_training: bool):
        return TradingEnv(
            data_array=data_obs,
            data_array_full=data_f,
            norm_stats=norm_stats,
            context_window_days=ctx,
            trading_period_days=trading_days,
            settlement_period_days=settlement,
            episode_calendar_days=_env_cfg("episode_calendar_days", None),
            min_holding_period=_env_cfg("min_holding_period", 30),
            max_positions=_env_cfg("max_positions", 10),
            inaction_penalty=_env_cfg("inaction_penalty", 0.0),
            coefficient_threshold=_env_cfg("coefficient_threshold", 1.0),
            min_sale_target=_env_cfg("min_sale_target", 10.0),
            max_sale_target=_env_cfg("max_sale_target", 50.0),
            investable_start_col=_env_cfg("investable_start_col", 9),
            num_investable_stocks=_env_cfg("num_investable_stocks", 108),
            loss_penalty_multiplier=_env_cfg(
                "loss_penalty_multiplier", DEFAULT_LOSS_PENALTY_MULTIPLIER
            ),
            hurdle_rate=_env_cfg("hurdle_rate", DEFAULT_HURDLE_RATE),
            conviction_scaling_power=_env_cfg(
                "conviction_scaling_power", DEFAULT_CONVICTION_SCALING_POWER
            ),
            observation_noise_std=_env_cfg("observation_noise_std", 0.01),
            is_training=is_training,
            dates_ordinal=dates_ord,
            episode_reward_multiplier=_env_cfg(
                "episode_reward_multiplier", DEFAULT_EPISODE_REWARD_MULTIPLIER
            ),
        )

    env = _make_env(train_obs, train_full, dates_train, is_training=True)
    val_env = _make_env(val_obs, val_full, dates_val, is_training=False)

    try:
        state = env.reset(key)
        logger.info(f"Train env reset OK; obs shape: {state.obs.shape}")
        key, vkey = jax.random.split(key)
        vstate = val_env.reset(vkey)
        logger.info(f"Validation env reset OK; obs shape: {vstate.obs.shape}")
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

    # ---- Hall of Fame ----
    from eigen3.erl.cloud_sync import CloudSync
    from eigen3.erl.hall_of_fame import HallOfFame

    hof_capacity = int(OmegaConf.select(cfg, "population.hof_capacity", default=10))
    cloud_project = OmegaConf.select(cfg, "population.cloud_project_name", default="eigen3")
    cloud_sync = CloudSync.from_env(project_name=cloud_project)

    checkpoint_dir = Path(hydra.utils.to_absolute_path("checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    hof = HallOfFame(
        capacity=hof_capacity,
        checkpoint_dir=checkpoint_dir,
        cloud_sync=cloud_sync,
        cloud_prefix=f"{cloud_project}/hall_of_fame",
    )
    hof.load()
    logger.info(
        "HoF ready: capacity=%d, loaded=%d, cloud=%s",
        hof.capacity, len(hof), cloud_sync.provider,
    )

    logger.info(
        "Data and env wired. Instantiate TradingERLWorkflow with env=train env, "
        "eval_env=val_env, hall_of_fame=hof; reserve holdout for final offline evaluation only."
    )


if __name__ == "__main__":
    main()
