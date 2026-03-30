"""Hydra-driven training: data, envs, Hall of Fame, ``TradingERLWorkflow``."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, List, Optional
from unittest.mock import MagicMock

import hydra
import jax
import numpy as np
from omegaconf import DictConfig, OmegaConf

from eigen3.agents import TradingAgent
from eigen3.config import (
    DEFAULT_CONVICTION_SCALING_POWER,
    DEFAULT_EPISODE_REWARD_MULTIPLIER,
    DEFAULT_HURDLE_RATE,
    DEFAULT_LOSS_PENALTY_MULTIPLIER,
)
from eigen3.data import create_synthetic_data, load_trading_data
from eigen3.data.splits import compute_train_val_holdout_split, slice_trading_timeline
from eigen3.environment.trading_env import TradingEnv
from eigen3.models import Actor, DoubleCritic
from eigen3.workflows import TradingWorkflowConfig, create_trading_workflow

logger = logging.getLogger(__name__)

TEE_ENV_VAR = "EIGEN3_TRAINING_LOG"


def _is_tqdm_final_line(line: str) -> bool:
    return "100%" in line and ("|" in line or "it/s" in line or "it]" in line)


class TeeLogger:
    """Mirror stdout to a line-buffered file; collapse tqdm in-progress lines."""

    def __init__(self, filepath: Path, terminal):
        self.terminal = terminal
        self.log_file = open(filepath, "w", encoding="utf-8", buffering=1)

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.terminal.flush()
        if "\r" in message:
            last = message.split("\r")[-1].strip()
            if _is_tqdm_final_line(last):
                self.log_file.write(last + "\n")
                self.log_file.flush()
        else:
            self.log_file.write(message)
            self.log_file.flush()

    def flush(self) -> None:
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self) -> bool:
        return self.terminal.isatty()

    def fileno(self) -> int:
        return self.terminal.fileno()

    def close(self) -> None:
        self.log_file.close()


def _short_class(path: Optional[str]) -> str:
    if not path:
        return "?"
    s = str(path)
    return s.rsplit(".", maxsplit=1)[-1]


def run_config_summary(cfg: DictConfig) -> str:
    """Compact startup summary for the console."""

    def S(key: str, default=None):
        return OmegaConf.select(cfg, key, default=default)

    lines: list[str] = [
        "",
        "-------- Run --------",
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
            "-------- Env --------",
            f"  data_path: {S('env.data_path', '?')!s}",
            f"  context: {S('env.context_window_days', '?')!s} d   "
            f"trading: {S('env.trading_period_days', '?')!s} d   "
            f"settlement: {S('env.settlement_period_days', 0)!s} d",
            f"  val_reserve: x{S('env.validation_reserve_multiplier', '?')!s}   "
            f"columns: {S('env.num_columns', '?')!s}   "
            f"F_obs: {S('env.num_features_obs', '?')!s}   "
            f"investable: {S('env.num_investable_stocks', '?')!s}",
            "-------- Agent -------",
            f"  workflow: {_short_class(S('agent.workflow_cls'))}",
            f"  actor_lr: {S('agent.optimizer.actor_lr', '?')!s}   "
            f"critic_lr: {S('agent.optimizer.critic_lr', '?')!s}   "
            f"mixed_precision: {S('agent.use_mixed_precision', False)!s}",
            "-------- Population --",
            f"  pop_size: {S('population.pop_size', '?')!s}   "
            f"generations: {S('population.total_generations', '?')!s}   "
            f"batch: {S('population.batch_size', '?')!s}",
            f"  replay: {S('population.replay_buffer_size', '?')!s}   "
            f"grad_steps/gen: {S('population.gradient_steps_per_gen', '?')!s}   "
            f"eval_episodes: {S('population.eval_episodes', '?')!s}",
            f"  steps_per_agent/gen: {S('population.steps_per_agent', '?')!s}",
            f"  gauntlet: {S('population.gauntlet_enabled', '?')!s}",
            "-------- Logging -----",
            f"  console: {S('logging.console_log_level', 'INFO')!s}   "
            f"tensorboard: {S('logging.use_tensorboard', False)!s}   "
            f"wandb: {S('logging.wandb_project', '?')!s} "
            f"({S('logging.wandb_mode', '?')!s})",
            "----------------------",
            "  Full config: Hydra run dir -> .hydra/config.yaml (+ overrides.yaml).",
            "",
        ]
    )
    return "\n".join(lines)


def build_networks(cfg: DictConfig) -> tuple[Actor, DoubleCritic]:
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


def build_trading_workflow_config(cfg: DictConfig) -> TradingWorkflowConfig:
    """Map Hydra ``population`` / ``agent`` into ``TradingWorkflowConfig``."""

    pop_size = int(OmegaConf.select(cfg, "population.pop_size", default=96))
    elite_frac = float(OmegaConf.select(cfg, "population.elite_frac", default=0.4))
    elite_size = max(1, min(int(round(pop_size * elite_frac)), pop_size))

    tournament_size = int(OmegaConf.select(cfg, "population.tournament_size", default=3))
    tournament_size = max(1, min(tournament_size, pop_size))

    crossover_rate = float(
        OmegaConf.select(cfg, "population.genetic_crossover_rate", default=0.5)
    )
    target_update_period = int(
        OmegaConf.select(cfg, "agent.actor_update_interval", default=2)
    )
    steps_per_agent = int(OmegaConf.select(cfg, "population.steps_per_agent", default=100))

    return TradingWorkflowConfig(
        population_size=pop_size,
        elite_size=elite_size,
        tournament_size=tournament_size,
        mutation_rate=float(OmegaConf.select(cfg, "population.mutation_rate", default=0.2)),
        mutation_std=float(OmegaConf.select(cfg, "population.mutation_std", default=0.025)),
        crossover_rate=crossover_rate,
        gradient_steps_per_gen=int(
            OmegaConf.select(cfg, "population.gradient_steps_per_gen", default=16)
        ),
        batch_size=int(OmegaConf.select(cfg, "population.batch_size", default=160)),
        replay_buffer_size=int(
            OmegaConf.select(cfg, "population.replay_buffer_size", default=1_500_000)
        ),
        eval_episodes=int(OmegaConf.select(cfg, "population.eval_episodes", default=5)),
        target_update_period=target_update_period,
        steps_per_agent=steps_per_agent,
    )


def run_training(cfg: DictConfig) -> List[dict[str, Any]]:
    """Load data, build envs and workflow, run ``TradingERLWorkflow.train``.

    Returns:
        List of per-generation metric dicts from the workflow.
    """
    tee_path = os.environ.get(TEE_ENV_VAR)
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    tee: Optional[TeeLogger] = None
    if tee_path:
        tee = TeeLogger(Path(tee_path), sys.stdout)
        sys.stdout = tee
        sys.stderr = tee

    try:
        return _run_training_impl(cfg)
    finally:
        if tee is not None:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            tee.close()


def _run_training_impl(cfg: DictConfig) -> List[dict[str, Any]]:
    tee_active = os.environ.get(TEE_ENV_VAR)
    if tee_active:
        logger.info("Mirroring console to %s", tee_active)
    logger.info(run_config_summary(cfg))
    logger.debug("Full resolved configuration:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    logger.info("Setting random seed: %s", cfg.seed)
    key = jax.random.PRNGKey(int(cfg.seed))

    data_path = OmegaConf.select(cfg, "env.data_path", default="data/raw")
    path = Path(hydra.utils.to_absolute_path(str(data_path)))
    column_index = OmegaConf.select(cfg, "env.column_index", default=None)

    if path.is_dir() and (path / "data_array.npy").exists():
        logger.info("Loading Eigen2 npy bundle from: %s", data_path)
        load_kwargs: dict = {}
        if column_index is not None:
            load_kwargs["column_index"] = int(column_index)
            logger.info("Using column_index=%s", column_index)
        data_array, data_array_full, norm_stats, dates_ordinal = load_trading_data(
            str(path), **load_kwargs
        )
    elif path.is_file() and path.suffix.lower() in (".pkl", ".pickle", ".csv"):
        logger.info("Loading mono table from: %s", data_path)
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
        logger.info("Using synthetic data (%s columns, F=%s)", num_cols, num_feat)
        data_array, data_array_full, norm_stats, dates_ordinal = create_synthetic_data(
            num_days=2000,
            num_columns=num_cols,
            num_features_obs=num_feat,
            seed=int(cfg.seed),
        )

    logger.info("Data shapes: obs %s, full %s", data_array.shape, data_array_full.shape)

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
        "Split: train rows [0, %s), val trading band [%s, %s), "
        "val env rows [%s, %s) (context may use [%s, %s)), "
        "holdout trading [%s, %s), holdout env [%s, %s) "
        "(final episode start index: %s)",
        split.train_end,
        split.val_start,
        split.val_end,
        split.val_env_start,
        split.val_end,
        split.val_env_start,
        split.val_start,
        split.holdout_start,
        split.holdout_end,
        split.holdout_env_start,
        split.holdout_end,
        split.last_episode_start,
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

    state = env.reset(key)
    logger.info("Train env reset OK; obs shape: %s", state.obs.shape)
    key, vkey = jax.random.split(key)
    vstate = val_env.reset(vkey)
    logger.info("Validation env reset OK; obs shape: %s", vstate.obs.shape)

    actor, critic = build_networks(cfg)
    agent = TradingAgent(
        actor_network=actor,
        critic_network=critic,
        exploration_noise=float(OmegaConf.select(cfg, "agent.exploration_noise", default=0.1)),
        discount=float(OmegaConf.select(cfg, "agent.discount", default=0.99)),
        tau=float(OmegaConf.select(cfg, "agent.tau", default=0.005)),
        min_sale_target=float(_env_cfg("min_sale_target", 10.0)),
        max_sale_target=float(_env_cfg("max_sale_target", 50.0)),
    )

    akey = jax.random.PRNGKey(int(cfg.seed) + 1)
    _ = agent.init(env.obs_space, env.action_space, akey)
    logger.info("TradingAgent init OK; starting TradingERLWorkflow.")

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
        hof.capacity,
        len(hof),
        cloud_sync.provider,
    )

    wf_cfg = build_trading_workflow_config(cfg)
    evaluator = MagicMock()
    workflow = create_trading_workflow(
        env,
        agent,
        evaluator,
        config=wf_cfg,
        seed=int(cfg.seed),
        eval_env=val_env,
        hall_of_fame=hof,
    )

    num_gen = int(OmegaConf.select(cfg, "population.total_generations", default=100))
    logger.info("Running %d generations (TradingERLWorkflow)...", num_gen)
    all_metrics = workflow.train(num_gen)

    last = all_metrics[-1] if all_metrics else {}
    logger.info(
        "Training finished. Last generation: mean_fitness=%.4f max_fitness=%.4f "
        "total_env_steps=%s",
        last.get("mean_fitness", float("nan")),
        last.get("max_fitness", float("nan")),
        last.get("total_env_steps", "?"),
    )

    try:
        from hydra.core.hydra_config import HydraConfig

        out_dir = HydraConfig.get().runtime.output_dir
        logger.info("Hydra output directory: %s", out_dir)
    except Exception:
        pass

    use_tb = OmegaConf.select(cfg, "logging.use_tensorboard", default=False)
    if use_tb:
        tb_dir = OmegaConf.select(cfg, "logging.tensorboard_log_dir", default="")
        if tb_dir:
            logger.info("TensorBoard: tensorboard --logdir %s", tb_dir)

    return all_metrics
