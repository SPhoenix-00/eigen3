"""Full training-state checkpoints for ``TradingERLWorkflow`` (resume support).

Serialises host-side NumPy copies of JAX arrays via pickle (trusted local paths only).
These artifacts include the replay buffer and must remain local-only; they are never
uploaded via :class:`eigen3.erl.cloud_sync.CloudSync`.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from eigen3.agents.trading_agent import TradingNetworkParams
from eigen3.environment.trading_env import EnvState, TradingEnvState
from eigen3.workflows.trading_workflow import (
    ReplayBufferState,
    TradingERLWorkflow,
    create_replay_buffer,
)

logger = logging.getLogger(__name__)

CHECKPOINT_VERSION = 1
TRAINING_STATE_BASENAME = "training_state.pkl"
TRAINING_STATE_META_BASENAME = "training_state_meta.json"


def training_state_path(run_dir: Path) -> Path:
    return run_dir / TRAINING_STATE_BASENAME


def _np_tree(tree: Any) -> Any:
    return jax.tree.map(lambda x: np.asarray(jax.device_get(x)), tree)


def _jnp_tree(tree: Any) -> Any:
    return jax.tree.map(lambda x: jnp.asarray(x), tree)


def _stacked_params_to_checkpoint(p: TradingNetworkParams) -> dict[str, Any]:
    return {
        "actor_params": _np_tree(p.actor_params),
        "critic_params": _np_tree(p.critic_params),
        "target_actor_params": _np_tree(p.target_actor_params),
        "target_critic_params": _np_tree(p.target_critic_params),
    }


def stacked_params_from_checkpoint(d: Mapping[str, Any]) -> TradingNetworkParams:
    return TradingNetworkParams(
        actor_params=_jnp_tree(d["actor_params"]),
        critic_params=_jnp_tree(d["critic_params"]),
        target_actor_params=_jnp_tree(d["target_actor_params"]),
        target_critic_params=_jnp_tree(d["target_critic_params"]),
    )


def _replay_to_checkpoint(buf: ReplayBufferState) -> dict[str, Any]:
    return {
        "obs": np.asarray(jax.device_get(buf.obs)),
        "actions": np.asarray(jax.device_get(buf.actions)),
        "rewards": np.asarray(jax.device_get(buf.rewards)),
        "next_obs": np.asarray(jax.device_get(buf.next_obs)),
        "dones": np.asarray(jax.device_get(buf.dones)),
        "size": np.asarray(jax.device_get(buf.size)),
        "insert_idx": np.asarray(jax.device_get(buf.insert_idx)),
    }


def _replay_from_checkpoint(d: Mapping[str, Any]) -> ReplayBufferState:
    return ReplayBufferState(
        obs=jnp.asarray(d["obs"]),
        actions=jnp.asarray(d["actions"]),
        rewards=jnp.asarray(d["rewards"]),
        next_obs=jnp.asarray(d["next_obs"]),
        dones=jnp.asarray(d["dones"]),
        size=jnp.asarray(d["size"]),
        insert_idx=jnp.asarray(d["insert_idx"]),
    )


def _trading_env_state_from_checkpoint(d: Mapping[str, Any]) -> TradingEnvState:
    return TradingEnvState(
        current_step=jnp.asarray(d["current_step"]),
        start_step=jnp.asarray(d["start_step"]),
        end_step=jnp.asarray(d["end_step"]),
        trading_end_step=jnp.asarray(d["trading_end_step"]),
        positions=jnp.asarray(d["positions"]),
        num_active_positions=jnp.asarray(d["num_active_positions"]),
        cumulative_reward=jnp.asarray(d["cumulative_reward"]),
        num_trades=jnp.asarray(d["num_trades"]),
        num_wins=jnp.asarray(d["num_wins"]),
        num_losses=jnp.asarray(d["num_losses"]),
        total_gain_pct=jnp.asarray(d["total_gain_pct"]),
        days_with_positions=jnp.asarray(d["days_with_positions"]),
        days_without_positions=jnp.asarray(d["days_without_positions"]),
        peak_capital_employed=jnp.asarray(d["peak_capital_employed"]),
        total_pnl=jnp.asarray(d["total_pnl"]),
        episode_benchmark_excess=jnp.asarray(d["episode_benchmark_excess"]),
        rng_key=jnp.asarray(d["rng_key"]),
    )


def _env_states_to_checkpoint(es: EnvState) -> dict[str, Any]:
    s = es.env_state
    return {
        "obs": np.asarray(jax.device_get(es.obs)),
        "reward": np.asarray(jax.device_get(es.reward)),
        "done": np.asarray(jax.device_get(es.done)),
        "env_state": {
            "current_step": np.asarray(jax.device_get(s.current_step)),
            "start_step": np.asarray(jax.device_get(s.start_step)),
            "end_step": np.asarray(jax.device_get(s.end_step)),
            "trading_end_step": np.asarray(jax.device_get(s.trading_end_step)),
            "positions": np.asarray(jax.device_get(s.positions)),
            "num_active_positions": np.asarray(jax.device_get(s.num_active_positions)),
            "cumulative_reward": np.asarray(jax.device_get(s.cumulative_reward)),
            "num_trades": np.asarray(jax.device_get(s.num_trades)),
            "num_wins": np.asarray(jax.device_get(s.num_wins)),
            "num_losses": np.asarray(jax.device_get(s.num_losses)),
            "total_gain_pct": np.asarray(jax.device_get(s.total_gain_pct)),
            "days_with_positions": np.asarray(jax.device_get(s.days_with_positions)),
            "days_without_positions": np.asarray(jax.device_get(s.days_without_positions)),
            "peak_capital_employed": np.asarray(jax.device_get(s.peak_capital_employed)),
            "total_pnl": np.asarray(jax.device_get(s.total_pnl)),
            "episode_benchmark_excess": np.asarray(jax.device_get(s.episode_benchmark_excess)),
            "rng_key": np.asarray(jax.device_get(s.rng_key)),
        },
    }


def _env_states_from_checkpoint(d: Mapping[str, Any]) -> EnvState:
    return EnvState(
        env_state=_trading_env_state_from_checkpoint(d["env_state"]),
        obs=jnp.asarray(d["obs"]),
        reward=jnp.asarray(d["reward"]),
        done=jnp.asarray(d["done"]),
        info={},
    )


def fresh_replay_buffer_for_workflow(workflow: TradingERLWorkflow) -> ReplayBufferState:
    """Empty replay buffer matching *workflow* env shapes and configured capacity."""
    obs_shape = tuple(int(x) for x in workflow.env.obs_space.shape)
    action_shape = tuple(int(x) for x in workflow.env.action_space.shape)
    return create_replay_buffer(
        int(workflow.config.replay_buffer_size),
        obs_shape,
        action_shape,
    )


def _validate_meta(meta: Mapping[str, Any], workflow: TradingERLWorkflow) -> None:
    if int(meta.get("version", 0)) != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint version {meta.get('version')!r}; "
            f"expected {CHECKPOINT_VERSION}"
        )
    if int(meta["pop_size"]) != int(workflow._pop_size):
        raise ValueError(
            f"Checkpoint pop_size={meta['pop_size']} does not match "
            f"current workflow ({workflow._pop_size})."
        )
    if int(meta["replay_buffer_size"]) != int(workflow.config.replay_buffer_size):
        raise ValueError(
            f"Checkpoint replay_buffer_size={meta['replay_buffer_size']} does not match "
            f"current config ({workflow.config.replay_buffer_size})."
        )
    wobs = tuple(int(x) for x in workflow.env.obs_space.shape)
    wact = tuple(int(x) for x in workflow.env.action_space.shape)
    if tuple(meta["obs_shape"]) != wobs:
        raise ValueError(f"Checkpoint obs_shape {meta['obs_shape']} != current {wobs}")
    if tuple(meta["action_shape"]) != wact:
        raise ValueError(f"Checkpoint action_shape {meta['action_shape']} != current {wact}")


def save_training_checkpoint(
    run_dir: Path,
    workflow: TradingERLWorkflow,
    *,
    best_score: float,
    run_name: str,
    seed: int,
) -> Path:
    """Atomically write ``training_state.pkl`` (+ small JSON sidecar) under *run_dir*."""
    if workflow._stacked_params is None or workflow._replay_buffer is None:
        raise RuntimeError("Workflow has no population or replay buffer to save.")
    if workflow._env_states is None:
        raise RuntimeError("Workflow env states are missing.")

    run_dir.mkdir(parents=True, exist_ok=True)
    path = training_state_path(run_dir)
    obs_shape = tuple(int(x) for x in workflow.env.obs_space.shape)
    action_shape = tuple(int(x) for x in workflow.env.action_space.shape)
    meta = {
        "version": CHECKPOINT_VERSION,
        "generation": int(workflow.generation),
        "total_env_steps": int(workflow.total_env_steps),
        "pop_size": int(workflow._pop_size),
        "replay_buffer_size": int(workflow.config.replay_buffer_size),
        "obs_shape": obs_shape,
        "action_shape": action_shape,
        "run_name": run_name,
        "seed": int(seed),
        "best_score": float(best_score),
    }
    blob = {
        "meta": meta,
        "rng_key": np.asarray(jax.device_get(workflow.key)),
        "stacked_params": _stacked_params_to_checkpoint(workflow._stacked_params),
        "replay_buffer": _replay_to_checkpoint(workflow._replay_buffer),
        "env_states": _env_states_to_checkpoint(workflow._env_states),
        "last_best_idx": workflow._last_best_idx,
        "printed_train_compile_hint": bool(workflow._printed_train_compile_hint),
    }
    raw = pickle.dumps(blob, protocol=4)
    tmp = path.with_suffix(".pkl.tmp")
    tmp.write_bytes(raw)
    tmp.replace(path)

    meta_path = run_dir / TRAINING_STATE_META_BASENAME
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Wrote training checkpoint: %s", path)
    return path


def load_training_checkpoint(
    path: Path,
    workflow: TradingERLWorkflow,
    *,
    restore_replay_buffer: bool = False,
) -> dict[str, Any]:
    """Load checkpoint into *workflow* (RNG, params, env states; replay optional).

    For ``--resume``, pass ``restore_replay_buffer=False`` (default): the replay
    buffer is **not** restored from disk; an empty buffer is allocated and refilled
    from on-device rollouts. Checkpoint files may still contain a saved buffer for
    local crash recovery if you call with ``restore_replay_buffer=True``.
    """
    blob = pickle.loads(path.read_bytes())
    meta = blob["meta"]
    _validate_meta(meta, workflow)

    workflow.generation = int(meta["generation"])
    workflow.total_env_steps = int(meta["total_env_steps"])
    workflow.key = jnp.asarray(blob["rng_key"])
    workflow._stacked_params = stacked_params_from_checkpoint(blob["stacked_params"])
    if restore_replay_buffer:
        workflow._replay_buffer = _replay_from_checkpoint(blob["replay_buffer"])
        logger.info("Resumed replay buffer from checkpoint.")
    else:
        workflow._replay_buffer = fresh_replay_buffer_for_workflow(workflow)
        logger.info(
            "Resume: replay buffer not loaded from checkpoint — starting empty "
            "(capacity=%s); refilling from training rollouts.",
            workflow.config.replay_buffer_size,
        )
    workflow._env_states = _env_states_from_checkpoint(blob["env_states"])
    workflow._last_best_idx = blob.get("last_best_idx")
    workflow._printed_train_compile_hint = bool(blob.get("printed_train_compile_hint", True))

    logger.info(
        "Resumed training state from %s (generation=%s, total_env_steps=%s)",
        path,
        workflow.generation,
        workflow.total_env_steps,
    )
    return {"meta": meta, "best_score": float(meta.get("best_score", float("-inf")))}
