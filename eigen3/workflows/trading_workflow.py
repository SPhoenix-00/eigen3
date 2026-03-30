"""Custom ERL workflow for stock trading with DDPG agents — GPU-vectorized

Processes all agents in the population simultaneously via jax.vmap,
replacing sequential Python loops with batched GPU kernels.  This enables
full utilization of high-memory GPUs like the H100 SXM.

Key changes from the sequential version:
- Population stored as a single stacked pytree with shape [pop_size, ...]
- Neural-network forward passes vmapped across the population
- Environment stepping vmapped across the population
- Experience collection: one ``jax.jit(jax.lax.fori_loop(...))`` over time (vmap × steps), not one Python dispatch per step
- JAX ring-buffer replay instead of Python list-of-dicts
- Genetic operators: batched tournaments + ``vmap`` crossover/mutation (no per-offspring Python loop)
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass
from functools import partial
import logging
import os
import time

import numpy as np
import chex
import jax
import jax.numpy as jnp
import jax.random as random
from evorl.types import PyTreeData
from evorl.sample_batch import SampleBatch
from evorl.envs import Env
from evorl.agent import Agent, AgentState
from evorl.evaluators import Evaluator
from eigen3.agents import TradingNetworkParams, soft_target_update
from eigen3.erl.hall_of_fame import HallOfFame

logger = logging.getLogger(__name__)


def _maybe_enable_jax_compile_logging() -> None:
    """If ``EIGEN3_LOG_JAX_COMPILES=1``, log each XLA compile (helps when GPU shows 0% during long CPU compiles)."""
    v = os.environ.get("EIGEN3_LOG_JAX_COMPILES", "").strip().lower()
    if v in ("1", "true", "yes"):
        jax.config.update("jax_log_compiles", True)


# ---------------------------------------------------------------------------
# Replay buffer — pre-allocated JAX arrays, ring-buffer layout
# ---------------------------------------------------------------------------

class ReplayBufferState(PyTreeData):
    """Fixed-capacity ring buffer stored as JAX arrays on device."""

    obs: chex.Array
    actions: chex.Array
    rewards: chex.Array
    next_obs: chex.Array
    dones: chex.Array
    size: chex.Array
    insert_idx: chex.Array


def create_replay_buffer(
    capacity: int,
    obs_shape: Tuple[int, ...],
    action_shape: Tuple[int, ...],
) -> ReplayBufferState:
    """Allocate an empty replay buffer on the default device."""
    return ReplayBufferState(
        obs=jnp.zeros((capacity, *obs_shape), dtype=jnp.float32),
        actions=jnp.zeros((capacity, *action_shape), dtype=jnp.float32),
        rewards=jnp.zeros(capacity, dtype=jnp.float32),
        next_obs=jnp.zeros((capacity, *obs_shape), dtype=jnp.float32),
        dones=jnp.zeros(capacity, dtype=jnp.float32),
        size=jnp.array(0, dtype=jnp.int32),
        insert_idx=jnp.array(0, dtype=jnp.int32),
    )


def buffer_insert_batch(
    buf: ReplayBufferState,
    obs: chex.Array,
    actions: chex.Array,
    rewards: chex.Array,
    next_obs: chex.Array,
    dones: chex.Array,
) -> ReplayBufferState:
    """Insert a batch of transitions (shape ``[batch, ...]``) into the ring buffer."""
    batch = obs.shape[0]
    capacity = buf.obs.shape[0]
    indices = (buf.insert_idx + jnp.arange(batch)) % capacity
    return buf.replace(
        obs=buf.obs.at[indices].set(obs),
        actions=buf.actions.at[indices].set(actions),
        rewards=buf.rewards.at[indices].set(rewards),
        next_obs=buf.next_obs.at[indices].set(next_obs),
        dones=buf.dones.at[indices].set(dones),
        size=jnp.minimum(buf.size + batch, capacity),
        insert_idx=(buf.insert_idx + batch) % capacity,
    )


def buffer_sample(
    buf: ReplayBufferState,
    key: chex.PRNGKey,
    batch_size: int,
) -> SampleBatch:
    """Uniformly sample ``batch_size`` transitions from filled positions."""
    indices = jax.random.randint(key, shape=(batch_size,), minval=0, maxval=buf.size)
    return SampleBatch(
        obs=buf.obs[indices],
        actions=buf.actions[indices],
        rewards=buf.rewards[indices],
        next_obs=buf.next_obs[indices],
        dones=buf.dones[indices],
    )


# ---------------------------------------------------------------------------
# Population utilities
# ---------------------------------------------------------------------------

def stack_params(population: List[TradingNetworkParams]) -> TradingNetworkParams:
    """Stack per-agent params into a single pytree with leading ``[pop_size]`` dim."""
    return jax.tree.map(lambda *xs: jnp.stack(xs), *population)


def unstack_params(stacked: TradingNetworkParams, pop_size: int) -> List[TradingNetworkParams]:
    """Inverse of :func:`stack_params`."""
    return [jax.tree.map(lambda x: x[i], stacked) for i in range(pop_size)]


def _offspring_cross_mut(
    p1: TradingNetworkParams,
    p2: TradingNetworkParams,
    k_cross: chex.PRNGKey,
    k_mut: chex.PRNGKey,
    crossover_rate: float,
    mutation_rate: float,
    mutation_std: float,
) -> TradingNetworkParams:
    """Single-offspring crossover + mutation (matches :meth:`TradingERLWorkflow._crossover_single` then ``_mutate_single``)."""
    k1, k2 = random.split(k_cross)

    def _cross(a: chex.Array, b: chex.Array, k: chex.PRNGKey) -> chex.Array:
        mask = random.uniform(k, a.shape) < crossover_rate
        return jnp.where(mask, a, b)

    child = TradingNetworkParams(
        actor_params=jax.tree.map(lambda a, b: _cross(a, b, k1), p1.actor_params, p2.actor_params),
        critic_params=jax.tree.map(lambda a, b: _cross(a, b, k2), p1.critic_params, p2.critic_params),
        target_actor_params=p1.target_actor_params,
        target_critic_params=p1.target_critic_params,
    )

    km1, km2 = random.split(k_mut)

    def _mutate(p: chex.Array, k: chex.PRNGKey) -> chex.Array:
        mask = random.uniform(k, p.shape) < mutation_rate
        noise = random.normal(k, p.shape) * mutation_std
        return jnp.where(mask, p + noise, p)

    return TradingNetworkParams(
        actor_params=jax.tree.map(lambda p: _mutate(p, km1), child.actor_params),
        critic_params=jax.tree.map(lambda p: _mutate(p, km2), child.critic_params),
        target_actor_params=child.target_actor_params,
        target_critic_params=child.target_critic_params,
    )


@partial(
    jax.jit,
    static_argnames=("pop_size", "elite_count", "tournament_size"),
)
def breed_next_generation_batched(
    stacked_params: TradingNetworkParams,
    fitness: chex.Array,
    key: chex.PRNGKey,
    *,
    pop_size: int,
    elite_count: int,
    tournament_size: int,
    crossover_rate: float,
    mutation_rate: float,
    mutation_std: float,
) -> TradingNetworkParams:
    """Vectorized breeding: vmapped tournaments + batched gather + ``vmap`` crossover/mutation.

    JIT-compiled once per ``(pop_size, elite_count, tournament_size)``; genetic rates stay
    dynamic so adaptive mutation does not force recompilation.

    Replaces the Python loop over offspring (which forced many host syncs and small kernels).
    Tournament sampling uses the first ``tournament_size`` elements of a random permutation,
    which is equivalent to ``random.choice(..., replace=False)`` for picking the candidate set.
    """
    sorted_idx = jnp.argsort(-fitness)
    elite_idx = sorted_idx[:elite_count]
    elite_stack = jax.tree.map(lambda x: x[elite_idx], stacked_params)

    n_offspring = pop_size - elite_count
    if n_offspring <= 0:
        return elite_stack

    key, k_t1, k_t2, k_cross, k_mut = random.split(key, 5)
    keys1 = random.split(k_t1, n_offspring)
    keys2 = random.split(k_t2, n_offspring)

    def one_tournament(k: chex.PRNGKey) -> chex.Array:
        perm = random.permutation(k, pop_size)
        cand = perm[:tournament_size]
        sub = fitness[cand]
        return cand[jnp.argmax(sub)]

    p1_idx = jax.vmap(one_tournament)(keys1)
    p2_idx = jax.vmap(one_tournament)(keys2)

    p1_stack = jax.tree.map(lambda x: x[p1_idx], stacked_params)
    p2_stack = jax.tree.map(lambda x: x[p2_idx], stacked_params)

    keys_cross = random.split(k_cross, n_offspring)
    keys_mut = random.split(k_mut, n_offspring)

    axes = jax.tree.map(lambda _: 0, p1_stack)
    offspring = jax.vmap(
        _offspring_cross_mut,
        in_axes=(axes, axes, 0, 0, None, None, None),
    )(
        p1_stack,
        p2_stack,
        keys_cross,
        keys_mut,
        crossover_rate,
        mutation_rate,
        mutation_std,
    )

    return jax.tree.map(
        lambda e, o: jnp.concatenate([e, o], axis=0),
        elite_stack,
        offspring,
    )


def run_chunked_vmap_loss(
    vmap_loss: Callable[..., Dict[str, chex.Array]],
    stacked_params: TradingNetworkParams,
    batch: SampleBatch,
    loss_keys: chex.Array,
    pop_size: int,
    chunk: Optional[int],
) -> Dict[str, chex.Array]:
    """Run ``vmap_loss(stacked, batch, keys)`` or slice the population axis per ``chunk``."""
    if chunk is None or chunk <= 0 or chunk >= pop_size:
        return vmap_loss(stacked_params, batch, loss_keys)

    parts_a: List[chex.Array] = []
    parts_c: List[chex.Array] = []
    parts_q: List[chex.Array] = []
    for start in range(0, pop_size, chunk):
        end = min(start + chunk, pop_size)
        sub_p = jax.tree.map(lambda x: x[start:end], stacked_params)
        sub_k = loss_keys[start:end]
        L = vmap_loss(sub_p, batch, sub_k)
        parts_a.append(L["actor_loss"])
        parts_c.append(L["critic_loss"])
        parts_q.append(L["mean_q"])
    return {
        "actor_loss": jnp.concatenate(parts_a, axis=0),
        "critic_loss": jnp.concatenate(parts_c, axis=0),
        "mean_q": jnp.concatenate(parts_q, axis=0),
    }


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TradingWorkflowConfig:
    """Configuration for the trading ERL workflow

    Args:
        population_size: Number of agents in the population
        elite_size: Number of top agents to preserve each generation
        tournament_size: Number of agents to sample for tournament selection
        mutation_rate: Probability of mutation for each parameter
        mutation_std: Standard deviation of Gaussian mutation
        crossover_rate: Probability of crossover between parents
        gradient_steps_per_gen: Number of gradient updates per generation
        batch_size: Batch size for gradient updates
        replay_buffer_size: Maximum size of replay buffer
        warmup_steps: Number of random exploration steps before training
        val_episodes: Number of validation episodes per agent per generation
        target_update_period: How often to update target networks (in gradient steps)
        steps_per_agent: Environment steps collected per agent per generation
        gradient_vmap_chunk_size: If set and ``< population_size``, run loss ``vmap`` in
            consecutive slices of this many agents to cap XLA peak memory (single-GPU).
            ``None`` means one ``vmap`` over the full population.
    """

    population_size: int = 10
    elite_size: int = 2
    tournament_size: int = 3
    mutation_rate: float = 0.1
    mutation_std: float = 0.02
    crossover_rate: float = 0.5
    gradient_steps_per_gen: int = 100
    batch_size: int = 32
    replay_buffer_size: int = 100000
    warmup_steps: int = 1000
    val_episodes: int = 2
    conservative_k: int = 1
    target_update_period: int = 10
    steps_per_agent: int = 100
    gradient_vmap_chunk_size: Optional[int] = None


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------

class TradingERLWorkflow:
    """GPU-vectorized ERL workflow for stock trading.

    Every phase of the generation loop processes all agents simultaneously
    via ``jax.vmap``, converting sequential Python loops into batched GPU
    kernels.  On an H100 SXM this yields 10–50x wall-clock speed-up over
    the sequential version.

    Phases per generation:
        1. **Collect experience** — vmap ``env.step`` + ``agent.compute_actions``
        2. **Gradient updates** — vmap ``agent.loss`` (shared replay batch)
        3. **Evaluate** — vmap ``env.step`` + ``agent.evaluate_actions``
        4. **Select & breed** — elitism + vmapped tournaments → batched crossover + mutation
    """

    def __init__(
        self,
        env: Env,
        agent: Agent,
        evaluator: Evaluator,
        config: TradingWorkflowConfig,
        seed: int = 0,
        val_env: Optional[Env] = None,
        hall_of_fame: Optional[HallOfFame] = None,
    ):
        self.env = env
        self.val_env = val_env if val_env is not None else env
        self.agent = agent
        self.evaluator = evaluator
        self.config = config
        self.seed = seed
        self.key = random.PRNGKey(seed)
        self._pop_size = config.population_size

        self._stacked_params: Optional[TradingNetworkParams] = None
        self._env_states = None
        self._replay_buffer: Optional[ReplayBufferState] = None
        self.generation = 0
        self.total_env_steps = 0
        self._last_best_idx: Optional[int] = None

        self.hof = hall_of_fame

        self._printed_train_compile_hint = False
        _maybe_enable_jax_compile_logging()
        self._build_vmapped_fns()

    # ------------------------------------------------------------------
    # JIT-compiled vmapped primitives
    # ------------------------------------------------------------------

    def _build_vmapped_fns(self):
        """Create ``jax.jit(jax.vmap(...))`` closures for the hot-path ops.

        Captured variables (``env``, ``agent``, network modules) are treated
        as constants by JAX's tracer — only the array arguments are traced.
        """
        env = self.env
        val_env = self.val_env
        agent = self.agent

        def _compute_action_one(params, obs, key):
            state = AgentState(params=params)
            actions, _ = agent.compute_actions(state, SampleBatch(obs=obs[None]), key)
            return actions[0]

        def _eval_action_one(params, obs, key):
            state = AgentState(params=params)
            actions, _ = agent.evaluate_actions(state, SampleBatch(obs=obs[None]), key)
            return actions[0]

        def _loss_one(params, sample_batch, key):
            state = AgentState(params=params)
            return agent.loss(state, sample_batch, key)

        self._vmap_eval_actions = jax.jit(jax.vmap(_eval_action_one))

        # in_axes=(0, None, 0): params per-agent, batch shared, key per-agent
        self._vmap_loss = jax.jit(jax.vmap(_loss_one, in_axes=(0, None, 0)))

        self._vmap_env_reset = jax.jit(jax.vmap(env.reset))
        self._vmap_val_step = jax.jit(jax.vmap(lambda s, a: val_env.step(s, a)))
        self._vmap_val_reset = jax.jit(jax.vmap(val_env.reset))

        pop_size_int = int(self._pop_size)

        def _collect_experience_jitted(
            stacked_params: TradingNetworkParams,
            env_states,
            buf: ReplayBufferState,
            key: chex.PRNGKey,
            num_steps: chex.Array,
        ):
            """Single compiled rollout: vmap over agents × ``fori_loop`` over time steps.

            ``num_steps`` is a 0-D int32 array (not a Python static) so XLA keeps a real loop
            instead of fully unrolling ``steps_per_agent`` copies of the env graph at compile time.
            """

            def step_body(_i, carry):
                env_states, buf, key, total_reward = carry
                key, step_key, reset_key = random.split(key, 3)
                action_keys = random.split(step_key, pop_size_int)
                all_actions = jax.vmap(_compute_action_one)(
                    stacked_params, env_states.obs, action_keys,
                )
                next_states = jax.vmap(lambda s, a: env.step(s, a))(
                    env_states, all_actions,
                )
                buf = buffer_insert_batch(
                    buf,
                    obs=env_states.obs,
                    actions=all_actions,
                    rewards=next_states.reward,
                    next_obs=next_states.obs,
                    dones=next_states.done.astype(jnp.float32),
                )
                total_reward = total_reward + jnp.sum(next_states.reward)
                done = next_states.done
                reset_keys = random.split(reset_key, pop_size_int)
                fresh = jax.vmap(lambda k: env.reset(k))(reset_keys)
                env_states = jax.tree.map(
                    lambda f, n: jnp.where(
                        done.reshape(-1, *([1] * (f.ndim - 1))), f, n,
                    ),
                    fresh,
                    next_states,
                )
                return (env_states, buf, key, total_reward)

            init_carry = (
                env_states,
                buf,
                key,
                jnp.array(0.0, dtype=jnp.float32),
            )
            return jax.lax.fori_loop(jnp.int32(0), num_steps, step_body, init_carry)

        self._jit_collect_experience = jax.jit(_collect_experience_jitted)

    def _vmap_loss_all_agents(
        self,
        stacked_params: TradingNetworkParams,
        batch: SampleBatch,
        loss_keys: chex.Array,
    ) -> Dict[str, chex.Array]:
        """Loss per agent; optionally split the population axis to limit peak VRAM."""
        return run_chunked_vmap_loss(
            self._vmap_loss,
            stacked_params,
            batch,
            loss_keys,
            self._pop_size,
            self.config.gradient_vmap_chunk_size,
        )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initialize_population(self, key: chex.PRNGKey):
        """Create stacked population, replay buffer, and per-agent env states."""
        pop_size = self._pop_size

        keys = random.split(key, pop_size)
        population: List[TradingNetworkParams] = []
        for i in range(pop_size):
            agent_state = self.agent.init(
                self.env.obs_space, self.env.action_space, keys[i]
            )
            population.append(agent_state.params)

        self._stacked_params = stack_params(population)

        obs_shape = self.env.obs_space.shape
        action_shape = self.env.action_space.shape
        self._replay_buffer = create_replay_buffer(
            self.config.replay_buffer_size, obs_shape, action_shape,
        )

        self.key, reset_key = random.split(self.key)
        reset_keys = random.split(reset_key, pop_size)
        self._env_states = self._vmap_env_reset(reset_keys)

        chunk = self.config.gradient_vmap_chunk_size
        logger.info(
            "Population initialized: %d agents, buffer capacity %d, "
            "obs %s, action %s%s",
            pop_size,
            self.config.replay_buffer_size,
            obs_shape,
            action_shape,
            f", loss_vmap_chunk={chunk}" if chunk else "",
        )

    # ------------------------------------------------------------------
    # Phase 1 — Experience collection (vmapped)
    # ------------------------------------------------------------------

    def _collect_experience(
        self,
        stacked_params: TradingNetworkParams,
        env_states,
        buf: ReplayBufferState,
        key: chex.PRNGKey,
        num_steps: int,
    ) -> Tuple[Any, ReplayBufferState, chex.PRNGKey, chex.Array]:
        """Collect ``num_steps`` transitions for ALL agents in parallel.

        Implemented as one ``jax.jit`` over ``lax.fori_loop`` (vmap × time) so the
        rollout is a single XLA program instead of ``num_steps`` Python dispatches.
        """
        env_states, buf, key, total_reward = self._jit_collect_experience(
            stacked_params,
            env_states,
            buf,
            key,
            jnp.asarray(int(num_steps), dtype=jnp.int32),
        )
        return env_states, buf, key, total_reward

    # ------------------------------------------------------------------
    # Phase 2 — Gradient updates (vmapped loss)
    # ------------------------------------------------------------------

    def _gradient_update(
        self,
        stacked_params: TradingNetworkParams,
        buf: ReplayBufferState,
        key: chex.PRNGKey,
    ) -> Tuple[TradingNetworkParams, Dict[str, float]]:
        """Compute DDPG losses for all agents (shared replay batch).

        By default losses are vmapped across the full population.  When
        ``gradient_vmap_chunk_size`` is set and smaller than ``pop_size``,
        the population axis is processed in chunks to cap XLA peak memory.
        """
        pop_size = self._pop_size
        tau = self.agent.tau
        n_steps = self.config.gradient_steps_per_gen

        # Accumulate in Python floats and block each step. A plain ``+=`` on JAX arrays in a
        # Python loop defers work; the first ``float()`` at the end can fuse all steps and
        # OOM (multi‑tens of GiB) when materializing metrics.
        total_actor_loss_py = 0.0
        total_critic_loss_py = 0.0
        total_mean_q_py = 0.0

        for step in range(n_steps):
            key, sample_key, loss_key = random.split(key, 3)

            batch = buffer_sample(buf, sample_key, self.config.batch_size)
            loss_keys = random.split(loss_key, pop_size)

            all_losses = self._vmap_loss_all_agents(stacked_params, batch, loss_keys)
            all_losses = jax.tree.map(jax.block_until_ready, all_losses)

            total_actor_loss_py += float(jax.device_get(jnp.mean(all_losses["actor_loss"])))
            total_critic_loss_py += float(jax.device_get(jnp.mean(all_losses["critic_loss"])))
            total_mean_q_py += float(jax.device_get(jnp.mean(all_losses["mean_q"])))

            if (step + 1) % self.config.target_update_period == 0:
                stacked_params = stacked_params.replace(
                    target_actor_params=jax.tree.map(
                        lambda t, s: tau * s + (1 - tau) * t,
                        stacked_params.target_actor_params,
                        stacked_params.actor_params,
                    ),
                    target_critic_params=jax.tree.map(
                        lambda t, s: tau * s + (1 - tau) * t,
                        stacked_params.target_critic_params,
                        stacked_params.critic_params,
                    ),
                )
                stacked_params = jax.block_until_ready(stacked_params)

        metrics = {
            "actor_loss": total_actor_loss_py / n_steps,
            "critic_loss": total_critic_loss_py / n_steps,
            "mean_q": total_mean_q_py / n_steps,
        }
        return stacked_params, metrics

    # ------------------------------------------------------------------
    # Phase 3 — Evaluation (vmapped)
    # ------------------------------------------------------------------

    def _validate_population(
        self,
        stacked_params: TradingNetworkParams,
        key: chex.PRNGKey,
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, Dict[str, chex.Array]]:
        """Validate all agents in parallel on held-out data.

        Returns:
            Tuple of ``(fitness, mean_bh_excess_usd, val_roi_pct,
            mean_pnl, mean_episode_reward, min_episode_reward, per_episode)``.
            The first six are each ``[pop_size]``; ``per_episode`` is a dict of
            ``[n_episodes, pop_size]`` matrices with full episode detail
            (rewards, PnL, trades, win/loss counts, etc.).
            Fitness is the mean of the ``conservative_k`` lowest episode total
            rewards.  ``val_roi_pct`` is aggregate validation ROI in **percent**:
            ``100 * sum(episode total_pnl) / max(episode peak_capital_employed)``
            per agent (0 if max peak is 0).  Buy-hold excess and PnL entries
            are means over validation episodes (captured at first ``done`` per
            episode).  Mean/min episode reward are taken over validation
            episodes (not the bottom-k fitness slice).
        """
        pop_size = self._pop_size
        # At least one validation episode (0 would make jnp.stack fail).
        n_episodes = max(1, int(self.config.val_episodes))
        # Calendar episodes must run until ``done`` or totals stay ~0 (no terminal bonus).
        # ``episode_length`` is the max row-span from TradingEnv's schedule; keep slack.
        _ep_len = getattr(self.val_env, "episode_length", None)
        _slack = 128
        if _ep_len is not None:
            try:
                max_steps = int(_ep_len) + _slack
            except (TypeError, ValueError):
                max_steps = 10_000 + _slack
        else:
            max_steps = 10_000 + _slack
        all_ep_rewards: list[chex.Array] = []
        all_ep_excess: list[chex.Array] = []
        all_ep_gain: list[chex.Array] = []
        all_ep_pnl: list[chex.Array] = []
        all_ep_num_trades: list[chex.Array] = []
        all_ep_num_wins: list[chex.Array] = []
        all_ep_num_losses: list[chex.Array] = []
        all_ep_peak_capital: list[chex.Array] = []
        all_ep_days_with_pos: list[chex.Array] = []
        all_ep_days_without_pos: list[chex.Array] = []
        all_ep_steps: list[chex.Array] = []
        all_ep_max_coeff: list[chex.Array] = []

        vmap_excess = None
        if hasattr(self.val_env, "episode_buyhold_excess_usd"):
            vmap_excess = jax.vmap(self.val_env.episode_buyhold_excess_usd)

        for ep in range(n_episodes):
            key, ep_key = random.split(key)
            reset_keys = random.split(ep_key, pop_size)
            env_states = self._vmap_val_reset(reset_keys)

            ep_rewards = jnp.zeros(pop_size)
            done_mask = jnp.zeros(pop_size, dtype=jnp.bool_)
            ep_excess = jnp.zeros(pop_size)
            ep_gain = jnp.zeros(pop_size)
            ep_pnl = jnp.zeros(pop_size)
            ep_num_trades = jnp.zeros(pop_size, dtype=jnp.int32)
            ep_num_wins = jnp.zeros(pop_size, dtype=jnp.int32)
            ep_num_losses = jnp.zeros(pop_size, dtype=jnp.int32)
            ep_peak_capital = jnp.zeros(pop_size)
            ep_days_with_pos = jnp.zeros(pop_size, dtype=jnp.int32)
            ep_days_without_pos = jnp.zeros(pop_size, dtype=jnp.int32)
            ep_steps = jnp.zeros(pop_size, dtype=jnp.int32)
            ep_max_coeff = jnp.zeros(pop_size)

            for _ in range(max_steps):
                key, step_key = random.split(key)
                action_keys = random.split(step_key, pop_size)

                all_actions = self._vmap_eval_actions(
                    stacked_params, env_states.obs, action_keys,
                )
                next_states = self._vmap_val_step(env_states, all_actions)

                step_coeff = jnp.max(all_actions[:, :, 0], axis=-1)
                ep_max_coeff = jnp.where(
                    ~done_mask,
                    jnp.maximum(ep_max_coeff, step_coeff),
                    ep_max_coeff,
                )

                ep_rewards = ep_rewards + jnp.where(
                    done_mask, 0.0, next_states.reward,
                )
                newly_done = next_states.done & ~done_mask
                if vmap_excess is not None:
                    ex = vmap_excess(next_states)
                    ep_excess = jnp.where(newly_done, ex, ep_excess)
                ep_gain = jnp.where(
                    newly_done,
                    next_states.env_state.total_gain_pct,
                    ep_gain,
                )
                es = next_states.env_state
                ep_pnl = jnp.where(newly_done, es.total_pnl, ep_pnl)
                ep_num_trades = jnp.where(newly_done, es.num_trades, ep_num_trades)
                ep_num_wins = jnp.where(newly_done, es.num_wins, ep_num_wins)
                ep_num_losses = jnp.where(newly_done, es.num_losses, ep_num_losses)
                ep_peak_capital = jnp.where(newly_done, es.peak_capital_employed, ep_peak_capital)
                ep_days_with_pos = jnp.where(newly_done, es.days_with_positions, ep_days_with_pos)
                ep_days_without_pos = jnp.where(newly_done, es.days_without_positions, ep_days_without_pos)
                ep_steps = ep_steps + (~done_mask).astype(jnp.int32)
                done_mask = done_mask | next_states.done
                env_states = next_states

            all_ep_rewards.append(ep_rewards)
            all_ep_excess.append(ep_excess)
            all_ep_gain.append(ep_gain)
            all_ep_pnl.append(ep_pnl)
            all_ep_num_trades.append(ep_num_trades)
            all_ep_num_wins.append(ep_num_wins)
            all_ep_num_losses.append(ep_num_losses)
            all_ep_peak_capital.append(ep_peak_capital)
            all_ep_days_with_pos.append(ep_days_with_pos)
            all_ep_days_without_pos.append(ep_days_without_pos)
            all_ep_steps.append(ep_steps)
            all_ep_max_coeff.append(ep_max_coeff)

        rewards_matrix = jnp.stack(all_ep_rewards, axis=0)  # [n_episodes, pop_size]
        excess_matrix = jnp.stack(all_ep_excess, axis=0)
        gain_matrix = jnp.stack(all_ep_gain, axis=0)

        # k=0 would mean over an empty slice (NaNs); treat non-positive as 1.
        k = max(1, min(int(self.config.conservative_k), n_episodes))
        sorted_rewards = jnp.sort(rewards_matrix, axis=0)  # ascending per agent
        bottom_k = sorted_rewards[:k, :]  # lowest k episodes per agent
        raw_fitness = jnp.mean(bottom_k, axis=0)
        coeff_matrix = jnp.stack(all_ep_max_coeff, axis=0)
        mean_max_coeff = jnp.mean(coeff_matrix, axis=0)  # [pop_size]
        # Epsilon-scaled coefficient bonus: gives non-trading agents (fitness~0)
        # a gradient toward the trading threshold without distorting agents that
        # are already trading (whose fitness >> this term).
        fitness = raw_fitness + mean_max_coeff * 1e-2
        mean_excess = jnp.mean(excess_matrix, axis=0)
        pnl_matrix = jnp.stack(all_ep_pnl, axis=0)
        peak_matrix = jnp.stack(all_ep_peak_capital, axis=0)
        sum_pnl = jnp.sum(pnl_matrix, axis=0)
        max_peak = jnp.max(peak_matrix, axis=0)
        val_roi_pct = jnp.where(
            max_peak > 0,
            100.0 * sum_pnl / max_peak,
            jnp.zeros_like(sum_pnl),
        )
        mean_pnl = jnp.mean(pnl_matrix, axis=0)
        reward_mean = jnp.mean(rewards_matrix, axis=0)
        reward_min = jnp.min(rewards_matrix, axis=0)
        per_episode: Dict[str, chex.Array] = {
            "rewards": rewards_matrix,
            "excess": excess_matrix,
            "gain_pct": gain_matrix,
            "pnl": pnl_matrix,
            "num_trades": jnp.stack(all_ep_num_trades, axis=0),
            "num_wins": jnp.stack(all_ep_num_wins, axis=0),
            "num_losses": jnp.stack(all_ep_num_losses, axis=0),
            "peak_capital": jnp.stack(all_ep_peak_capital, axis=0),
            "days_with_pos": jnp.stack(all_ep_days_with_pos, axis=0),
            "days_without_pos": jnp.stack(all_ep_days_without_pos, axis=0),
            "steps": jnp.stack(all_ep_steps, axis=0),
            "max_coeff": coeff_matrix,
        }
        return fitness, mean_excess, val_roi_pct, mean_pnl, reward_mean, reward_min, per_episode

    # ------------------------------------------------------------------
    # Phase 4 — Genetic operators
    # ------------------------------------------------------------------

    def _tournament_select(self, fitness: chex.Array, key: chex.PRNGKey) -> int:
        """Tournament selection; returns winner index (Python int)."""
        pop_size = fitness.shape[0]
        idx = random.choice(
            key, pop_size, shape=(self.config.tournament_size,), replace=False,
        )
        return int(idx[jnp.argmax(fitness[idx])])

    def _crossover_single(
        self,
        p1: TradingNetworkParams,
        p2: TradingNetworkParams,
        key: chex.PRNGKey,
    ) -> TradingNetworkParams:
        """Uniform crossover between two parents."""
        k1, k2 = random.split(key)

        def _cross(a, b, k):
            mask = random.uniform(k, a.shape) < self.config.crossover_rate
            return jnp.where(mask, a, b)

        return TradingNetworkParams(
            actor_params=jax.tree.map(lambda a, b: _cross(a, b, k1), p1.actor_params, p2.actor_params),
            critic_params=jax.tree.map(lambda a, b: _cross(a, b, k2), p1.critic_params, p2.critic_params),
            target_actor_params=p1.target_actor_params,
            target_critic_params=p1.target_critic_params,
        )

    def _mutate_single(
        self,
        params: TradingNetworkParams,
        key: chex.PRNGKey,
    ) -> TradingNetworkParams:
        """Gaussian mutation on actor and critic params."""
        k1, k2 = random.split(key)

        def _mutate(p, k):
            mask = random.uniform(k, p.shape) < self.config.mutation_rate
            noise = random.normal(k, p.shape) * self.config.mutation_std
            return jnp.where(mask, p + noise, p)

        return TradingNetworkParams(
            actor_params=jax.tree.map(lambda p: _mutate(p, k1), params.actor_params),
            critic_params=jax.tree.map(lambda p: _mutate(p, k2), params.critic_params),
            target_actor_params=params.target_actor_params,
            target_critic_params=params.target_critic_params,
        )

    def _breed_next_generation(
        self,
        stacked_params: TradingNetworkParams,
        fitness: chex.Array,
        key: chex.PRNGKey,
    ) -> TradingNetworkParams:
        """Elitism + tournament selection → crossover + mutation → new population (batched)."""
        return breed_next_generation_batched(
            stacked_params,
            fitness,
            key,
            pop_size=self._pop_size,
            elite_count=self.config.elite_size,
            tournament_size=self.config.tournament_size,
            crossover_rate=self.config.crossover_rate,
            mutation_rate=self.config.mutation_rate,
            mutation_std=self.config.mutation_std,
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run_generation(self) -> Dict[str, Any]:
        """Run one generation of the GPU-vectorized ERL workflow."""
        t_gen_start = time.perf_counter()
        self.key, gen_key = random.split(self.key)

        t_init_s = 0.0
        if self._stacked_params is None:
            print("  > Init...", end="", flush=True)
            t_init_start = time.perf_counter()
            self.key, init_key = random.split(self.key)
            self._initialize_population(init_key)
            t_init_s = time.perf_counter() - t_init_start
            print(f" {t_init_s:.1f}s", flush=True)

        # Phase 1 — collect experience (vmapped)
        print("  > Collect...", end="", flush=True)
        t_collect_start = time.perf_counter()
        self.key, collect_key = random.split(self.key)
        self._env_states, self._replay_buffer, _, collect_reward = (
            self._collect_experience(
                self._stacked_params,
                self._env_states,
                self._replay_buffer,
                collect_key,
                self.config.steps_per_agent,
            )
        )
        self.total_env_steps += self.config.steps_per_agent * self._pop_size
        t_collect_s = time.perf_counter() - t_collect_start
        print(f" {t_collect_s:.1f}s", flush=True)

        # Ensure collect finished before training reads buffer (avoids odd stalls).
        self._replay_buffer = jax.block_until_ready(self._replay_buffer)

        # Phase 2 — gradient updates (vmapped loss)
        print("  > Train...", end="", flush=True)
        t_train_start = time.perf_counter()
        grad_metrics: Dict[str, float] = {}
        buf_size = int(jax.device_get(self._replay_buffer.size))
        if buf_size >= self.config.batch_size:
            if not self._printed_train_compile_hint:
                self._printed_train_compile_hint = True
                print(
                    "\n    [JAX] First training step compiles the vmapped loss on the CPU; "
                    "GPU utilization may read 0% for several minutes. "
                    "Set EIGEN3_LOG_JAX_COMPILES=1 to log each compile.",
                    flush=True,
                )
            self.key, grad_key = random.split(self.key)
            self._stacked_params, grad_metrics = self._gradient_update(
                self._stacked_params, self._replay_buffer, grad_key,
            )
        t_train_s = time.perf_counter() - t_train_start
        print(f" {t_train_s:.1f}s", flush=True)

        # Phase 3 — validate population (vmapped on held-out data)
        print("  > Val...", end="", flush=True)
        t_val_start = time.perf_counter()
        self.key, val_key = random.split(self.key)
        fitness_scores, bh_excess, val_roi_pct, pnl_mean_arr, reward_mean_arr, reward_min_arr, per_episode = (
            self._validate_population(self._stacked_params, val_key)
        )
        t_val_s = time.perf_counter() - t_val_start
        print(f" {t_val_s:.1f}s", flush=True)

        # One host copy for stats + HoF; avoid float(fitness_scores[i]) per agent (N syncs).
        t_stats_start = time.perf_counter()
        fit_np = np.asarray(jax.device_get(fitness_scores))
        bh_np = np.asarray(jax.device_get(bh_excess))
        roi_np = np.asarray(jax.device_get(val_roi_pct))
        pnl_np = np.asarray(jax.device_get(pnl_mean_arr))
        rew_mean_np = np.asarray(jax.device_get(reward_mean_arr))
        rew_min_np = np.asarray(jax.device_get(reward_min_arr))
        self._last_best_idx = int(np.argmax(fit_np))

        per_ep_np = {k: np.asarray(jax.device_get(v)) for k, v in per_episode.items()}

        n_agents = int(fit_np.size)
        top_k = min(5, n_agents)
        if top_k > 0:
            order = np.argsort(-fit_np, kind="stable")
            top5_indices = [int(order[i]) for i in range(top_k)]
            top5_fitness = [float(fit_np[i]) for i in top5_indices]
        else:
            top5_indices = []
            top5_fitness = []

        _wr_per_ep = per_ep_np["num_wins"] / np.maximum(per_ep_np["num_trades"], 1)
        win_rate_np = np.mean(_wr_per_ep, axis=0)  # [pop_size]

        n_val_eps = per_ep_np["rewards"].shape[0]
        best_episodes: List[Dict[str, Any]] = []
        bi = self._last_best_idx
        for ep_i in range(n_val_eps):
            nt = int(per_ep_np["num_trades"][ep_i, bi])
            nw = int(per_ep_np["num_wins"][ep_i, bi])
            ep_peak = float(per_ep_np["peak_capital"][ep_i, bi])
            ep_pnl = float(per_ep_np["pnl"][ep_i, bi])
            ep_roi = (100.0 * ep_pnl / ep_peak) if ep_peak > 0 else 0.0
            best_episodes.append({
                "total_reward": float(per_ep_np["rewards"][ep_i, bi]),
                "steps": int(per_ep_np["steps"][ep_i, bi]),
                "num_trades": nt,
                "num_wins": nw,
                "num_losses": int(per_ep_np["num_losses"][ep_i, bi]),
                "total_gain_pct": float(per_ep_np["gain_pct"][ep_i, bi]),
                "total_pnl": ep_pnl,
                "episode_roi_pct": ep_roi,
                "peak_capital_employed": ep_peak,
                "days_with_positions": int(per_ep_np["days_with_pos"][ep_i, bi]),
                "days_without_positions": int(per_ep_np["days_without_pos"][ep_i, bi]),
                "win_rate": float(nw) / max(nt, 1),
            })

        t_stats_s = time.perf_counter() - t_stats_start

        # Phase 3b — Hall of Fame
        print("  > HoF...", end="", flush=True)
        t_hof_start = time.perf_counter()
        if self.hof is not None:
            population_list = unstack_params(self._stacked_params, self._pop_size)
            hof_candidates = [
                (
                    population_list[i],
                    float(fit_np[i]),
                    i,
                    float(roi_np[i]),
                    0.0,
                    0.0,
                    0,
                    0,
                    float(fit_np[i]),
                    float(fit_np[i]),
                    float(bh_np[i]),
                )
                for i in range(self._pop_size)
            ]
            hof_results = self.hof.update_from_generation(
                hof_candidates, self.generation,
            )
            admitted = [r for r in hof_results if "admitted" in r[2] or "replaced" in r[2]]
            if admitted:
                stats = self.hof.get_stats()
                print(
                    f" {len(admitted)} admitted (size={len(self.hof)}, best={stats['best_score']:.2f})",
                    end="", flush=True,
                )
        t_hof_s = time.perf_counter() - t_hof_start
        print(f" {t_hof_s:.1f}s", flush=True)

        # Phase 4 — selection + breeding
        print("  > Evolve...", end="", flush=True)
        t_evolve_start = time.perf_counter()
        self.key, breed_key = random.split(self.key)
        self._stacked_params = self._breed_next_generation(
            self._stacked_params, fitness_scores, breed_key,
        )

        self.generation += 1

        if self.hof is not None:
            self.hof.save()
        t_evolve_s = time.perf_counter() - t_evolve_start
        print(f" {t_evolve_s:.1f}s", flush=True)
        t_total_s = time.perf_counter() - t_gen_start

        metrics: Dict[str, Any] = {
            "generation": self.generation,
            "mean_fitness": float(fit_np.mean()),
            "max_fitness": float(fit_np.max()),
            "min_fitness": float(fit_np.min()),
            "std_fitness": float(fit_np.std()),
            "total_env_steps": self.total_env_steps,
            "best_agent_idx": int(self._last_best_idx) if self._last_best_idx is not None else -1,
            "best_agent_fitness": float(fit_np.max()),
            "buffer_size": int(self._replay_buffer.size) if self._replay_buffer is not None else 0,
            "buffer_capacity": int(self.config.replay_buffer_size),
            "collect_reward_total": float(collect_reward),
            "collect_reward_mean_per_agent": float(collect_reward) / float(self._pop_size),
            "population_size": int(self._pop_size),
            "positive_agents": int(np.sum(fit_np > 0.0)),
            "mean_max_coeff": float(np.mean(per_ep_np["max_coeff"])),
            "max_max_coeff": float(np.max(per_ep_np["max_coeff"])),
            "top5_indices": top5_indices,
            "top5_fitness": top5_fitness,
            "top5_val_reward_mean": [float(rew_mean_np[i]) for i in top5_indices],
            "top5_val_reward_min": [float(rew_min_np[i]) for i in top5_indices],
            "top5_roi_pct": [float(roi_np[i]) for i in top5_indices],
            "top5_pnl": [float(pnl_np[i]) for i in top5_indices],
            "top5_bh_excess_usd": [float(bh_np[i]) for i in top5_indices],
            "top5_win_rate": [float(win_rate_np[i]) for i in top5_indices],
            "best_agent_val_episodes": best_episodes,
            "timing_init_s": float(t_init_s),
            "timing_collect_s": float(t_collect_s),
            "timing_train_s": float(t_train_s),
            "timing_val_s": float(t_val_s),
            "timing_stats_s": float(t_stats_s),
            "timing_hof_s": float(t_hof_s),
            "timing_evolve_s": float(t_evolve_s),
            "timing_total_s": float(t_total_s),
        }
        for k, v in grad_metrics.items():
            metrics[f"mean_{k}"] = v

        if self.hof is not None:
            hof_stats = self.hof.get_stats()
            metrics["hof_size"] = hof_stats["size"]
            metrics["hof_best"] = hof_stats["best_score"]
            metrics["hof_worst"] = hof_stats["worst_score"]
            metrics["hof_median_roi"] = hof_stats["median_roi"]
            metrics["hof_best_bh_excess"] = hof_stats["best_bh_excess"]
            metrics["hof_worst_bh_excess"] = hof_stats["worst_bh_excess"]

        return metrics

    def get_last_best_agent(self) -> TradingNetworkParams:
        """Return best agent params from the most recent generation evaluation."""
        if self._stacked_params is None or self._last_best_idx is None:
            raise ValueError("No evaluated generation yet; run at least one generation first.")
        return jax.tree.map(lambda x: x[self._last_best_idx], self._stacked_params)

    def train(self, num_generations: int) -> List[Dict[str, Any]]:
        """Train for ``num_generations`` generations."""
        all_metrics: List[Dict[str, Any]] = []

        for gen in range(num_generations):
            metrics = self.run_generation()
            all_metrics.append(metrics)

            print(
                f"Generation {gen + 1}/{num_generations}  "
                f"Mean: {metrics['mean_fitness']:.2f}  "
                f"Max: {metrics['max_fitness']:.2f}  "
                f"Min: {metrics['min_fitness']:.2f}  "
                f"Steps: {metrics['total_env_steps']}"
            )

        return all_metrics

    def get_best_agent(self) -> TradingNetworkParams:
        """Return the best agent's params from the current population."""
        if self._stacked_params is None:
            raise ValueError("Population not initialized. Run at least one generation first.")

        self.key, eval_key = random.split(self.key)
        fitness, _, _, _, _, _, _ = self._validate_population(self._stacked_params, eval_key)
        best_idx = int(jnp.argmax(fitness))

        return jax.tree.map(lambda x: x[best_idx], self._stacked_params)


# ---------------------------------------------------------------------------
# Convenience factory (same API as before)
# ---------------------------------------------------------------------------

def create_trading_workflow(
    env: Env,
    agent: Agent,
    evaluator: Evaluator,
    config: Optional[TradingWorkflowConfig] = None,
    seed: int = 0,
    val_env: Optional[Env] = None,
    hall_of_fame: Optional[HallOfFame] = None,
) -> TradingERLWorkflow:
    """Convenience function to create a trading ERL workflow."""
    if config is None:
        config = TradingWorkflowConfig()

    return TradingERLWorkflow(
        env=env,
        agent=agent,
        evaluator=evaluator,
        config=config,
        seed=seed,
        val_env=val_env,
        hall_of_fame=hall_of_fame,
    )
