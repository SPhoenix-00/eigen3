"""Trading Agent with DDPG algorithm (JAX/Flax implementation)

Implements the EvoRL Agent interface with:
- Actor-Critic networks with target networks
- DDPG-style updates
- Exploration noise
- Soft target updates
"""

from typing import Any, Tuple, Optional
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.tree as jtree
import chex
from evorl.agent import Agent, AgentState
from evorl.types import PyTreeData
from evorl.sample_batch import SampleBatch

from eigen3.models.actor import Actor
from eigen3.models.critic import DoubleCritic


class TradingNetworkParams(PyTreeData):
    """Network parameters for trading agent

    Stores all network parameters including targets for DDPG.
    """
    actor_params: chex.ArrayTree
    critic_params: chex.ArrayTree
    target_actor_params: chex.ArrayTree
    target_critic_params: chex.ArrayTree


def params_for_flax_msgpack(params: Any) -> Any:
    """Plain dict for ``flax.serialization.to_bytes`` (msgpack cannot encode PyTreeData)."""
    if isinstance(params, TradingNetworkParams):
        return {
            "actor_params": params.actor_params,
            "critic_params": params.critic_params,
            "target_actor_params": params.target_actor_params,
            "target_critic_params": params.target_critic_params,
        }
    return params


def trading_params_from_msgpack_bytes(raw: bytes, template: TradingNetworkParams) -> TradingNetworkParams:
    """Load weights saved via :func:`params_for_flax_msgpack` and ``flax.serialization.to_bytes``."""
    from flax.serialization import from_bytes

    loaded = from_bytes(params_for_flax_msgpack(template), raw)
    return TradingNetworkParams(
        actor_params=loaded["actor_params"],
        critic_params=loaded["critic_params"],
        target_actor_params=loaded["target_actor_params"],
        target_critic_params=loaded["target_critic_params"],
    )


class TradingAgent(Agent):
    """DDPG-based trading agent for stock market simulation

    Implements the EvoRL Agent interface with DDPG algorithm.
    Synced with Eigen2: coefficient clip [0, 100], exploration noise, soft target updates.
    """

    # Network architecture
    actor_network: nn.Module
    critic_network: nn.Module

    # DDPG hyperparameters
    exploration_noise: float = 0.1
    discount: float = 0.99
    tau: float = 0.005  # Soft target update rate

    # Action space bounds
    min_sale_target: float = 10.0
    max_sale_target: float = 50.0

    def init(
        self,
        obs_space: 'Space',
        action_space: 'Space',
        key: chex.PRNGKey
    ) -> AgentState:
        """Initialize agent parameters

        Args:
            obs_space: Observation space
            action_space: Action space
            key: JAX random key

        Returns:
            Initial AgentState with network parameters
        """
        actor_key, critic_key = jax.random.split(key, 2)

        # Model input includes the portfolio tail (obs_space is market-only).
        pdim = getattr(self.actor_network, 'portfolio_dim', 0)
        model_features = obs_space.shape[-1] + pdim
        model_shape = (*obs_space.shape[:-1], model_features)
        dummy_obs = jnp.zeros((1, *model_shape))
        dummy_action = jnp.zeros((1, *action_space.shape))

        # Initialize actor network
        actor_params = self.actor_network.init(
            actor_key,
            dummy_obs,
            train=False,
            return_attention_weights=False
        )

        # Initialize critic network
        critic_params = self.critic_network.init(
            critic_key,
            dummy_obs,
            dummy_action,
            train=False
        )

        # Targets must be separate array buffers or soft updates are no-ops (τs+(1-τ)t=t when s is t).
        params = TradingNetworkParams(
            actor_params=actor_params,
            critic_params=critic_params,
            target_actor_params=jtree.map(jnp.copy, actor_params),
            target_critic_params=jtree.map(jnp.copy, critic_params),
        )

        return AgentState(params=params)

    def compute_actions(
        self,
        agent_state: AgentState,
        sample_batch: SampleBatch,
        key: chex.PRNGKey
    ) -> Tuple[chex.Array, dict]:
        """Get actions with exploration noise (for training)

        Args:
            agent_state: Current agent state
            sample_batch: Batch of observations
            key: JAX random key for exploration noise

        Returns:
            Tuple of (actions, policy_info)
        """
        params = agent_state.params
        obs = sample_batch.obs

        # Get deterministic actions from actor
        actions, _ = self.actor_network.apply(
            params.actor_params,
            obs,
            train=False,
            return_attention_weights=False
        )

        # Add exploration noise (Eigen2: NOISE_SCALE)
        noise = jax.random.normal(key, actions.shape) * self.exploration_noise
        noisy_actions = actions + noise

        # Clip to valid ranges (Eigen2: coefficient [0, 100], sale target [10, 50])
        noisy_actions = noisy_actions.at[:, :, 0].set(
            jnp.clip(noisy_actions[:, :, 0], 0.0, 100.0)
        )
        noisy_actions = noisy_actions.at[:, :, 1].set(
            jnp.clip(noisy_actions[:, :, 1], self.min_sale_target, self.max_sale_target)
        )
        noisy_actions = noisy_actions.at[:, :, 2].set(
            jnp.clip(noisy_actions[:, :, 2], 0.0, 1.0)
        )

        policy_info = {}

        return noisy_actions, policy_info

    def evaluate_actions(
        self,
        agent_state: AgentState,
        sample_batch: SampleBatch,
        key: chex.PRNGKey
    ) -> Tuple[chex.Array, dict]:
        """Get deterministic actions (for evaluation, no noise)

        Args:
            agent_state: Current agent state
            sample_batch: Batch of observations
            key: JAX random key (unused for deterministic actions)

        Returns:
            Tuple of (actions, policy_info)
        """
        params = agent_state.params
        obs = sample_batch.obs

        # Get deterministic actions (no noise)
        actions, _ = self.actor_network.apply(
            params.actor_params,
            obs,
            train=False,
            return_attention_weights=False
        )

        policy_info = {}

        return actions, policy_info

    def loss(
        self,
        agent_state: AgentState,
        sample_batch: SampleBatch,
        key: chex.PRNGKey
    ) -> dict:
        """Compute DDPG losses

        Args:
            agent_state: Current agent state
            sample_batch: Batch of transitions
            key: JAX random key

        Returns:
            Dictionary of losses
        """
        params = agent_state.params

        # Extract batch data
        obs = sample_batch.obs  # [batch, context_days, num_columns, 5]
        actions = sample_batch.actions  # [batch, num_stocks, 3]
        rewards = sample_batch.rewards  # [batch]
        next_obs = sample_batch.next_obs  # [batch, context_days, num_columns, 5]
        dones = sample_batch.dones  # [batch]

        # ============ Critic Loss ============
        # Compute target Q-values (no gradient through targets)
        next_actions, _ = self.actor_network.apply(
            params.target_actor_params,
            next_obs,
            train=False,
            return_attention_weights=False
        )

        target_q = self.critic_network.apply(
            params.target_critic_params,
            next_obs,
            next_actions,
            train=False
        )

        # For twin critics (TD3-style), take minimum
        if target_q.shape[-1] == 2:
            target_q = jnp.min(target_q, axis=-1, keepdims=True)

        # Compute TD target: r + gamma * (1 - done) * Q_target(s', a')
        rewards = rewards.reshape(-1, 1)
        dones = dones.reshape(-1, 1)

        td_target = jax.lax.stop_gradient(
            rewards + self.discount * (1.0 - dones) * target_q
        )

        # Compute current Q-values
        # train=False: avoid updating BatchNorm batch_stats inside a frozen param pytree.
        current_q = self.critic_network.apply(
            params.critic_params,
            obs,
            actions,
            train=False
        )

        # Critic loss (MSE for each critic if using twin critics)
        if current_q.shape[-1] == 2:
            # Twin critics: compute loss for each
            q1_loss = jnp.mean((current_q[:, 0:1] - td_target) ** 2)
            q2_loss = jnp.mean((current_q[:, 1:2] - td_target) ** 2)
            critic_loss = q1_loss + q2_loss
        else:
            # Single critic
            critic_loss = jnp.mean((current_q - td_target) ** 2)

        # ============ Actor Loss ============
        # Get actions from current actor
        actor_actions, _ = self.actor_network.apply(
            params.actor_params,
            obs,
            train=False,
            return_attention_weights=False
        )

        # Compute Q-values for actor actions
        actor_q = self.critic_network.apply(
            params.critic_params,
            obs,
            actor_actions,
            train=False
        )

        # For twin critics, use mean or first critic
        if actor_q.shape[-1] == 2:
            actor_q = jnp.mean(actor_q, axis=-1, keepdims=True)

        # Actor loss: maximize Q (minimize -Q)
        actor_loss = -jnp.mean(actor_q)

        return {
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'mean_q': jnp.mean(current_q),
            'mean_reward': jnp.mean(rewards),
        }


def soft_target_update(
    params: TradingNetworkParams,
    tau: float
) -> TradingNetworkParams:
    """Soft update of target networks

    θ_target = tau * θ_source + (1 - tau) * θ_target

    Args:
        params: Current network parameters
        tau: Soft update coefficient

    Returns:
        Updated parameters with soft-updated targets
    """

    def update_tree(target, source):
        """Update target tree with soft update from source"""
        return jtree.map(
            lambda t, s: tau * s + (1 - tau) * t,
            target,
            source,
        )

    return params.replace(
        target_actor_params=update_tree(
            params.target_actor_params,
            params.actor_params
        ),
        target_critic_params=update_tree(
            params.target_critic_params,
            params.critic_params
        ),
    )


def test_trading_agent():
    """Test the TradingAgent implementation"""
    import jax.random as random

    print("Testing TradingAgent...")

    # Create agent
    actor = Actor()
    critic = DoubleCritic()  # Twin critics

    agent = TradingAgent(
        actor_network=actor,
        critic_network=critic,
        exploration_noise=0.1,
        discount=0.99,
        tau=0.005,
    )

    # Create dummy spaces
    from evorl.envs import Box

    oshp = (151, 117, 5)
    obs_space = Box(low=jnp.full(oshp, -jnp.inf), high=jnp.full(oshp, jnp.inf))
    ash = (108, 3)
    action_space = Box(
        low=jnp.broadcast_to(jnp.array([0.0, 10.0, 0.0], dtype=jnp.float32), ash),
        high=jnp.broadcast_to(jnp.array([jnp.inf, 50.0, 1.0], dtype=jnp.float32), ash),
    )

    # Initialize agent
    key = random.PRNGKey(0)
    agent_state = agent.init(obs_space, action_space, key)

    print(f"✓ Agent initialized")

    # Test compute_actions (with noise)
    batch_size = 4
    obs = random.normal(key, (batch_size, 151, 117, 5))
    sample_batch = SampleBatch(obs=obs)

    key, action_key = random.split(key)
    actions, policy_info = agent.compute_actions(agent_state, sample_batch, action_key)

    print(f"✓ compute_actions: actions shape = {actions.shape}")
    assert actions.shape == (batch_size, 108, 3)

    # Test evaluate_actions (no noise)
    eval_actions, _ = agent.evaluate_actions(agent_state, sample_batch, key)
    print(f"✓ evaluate_actions: actions shape = {eval_actions.shape}")
    assert eval_actions.shape == (batch_size, 108, 3)

    # Test loss computation
    next_obs = random.normal(key, (batch_size, 151, 117, 5))
    rewards = random.normal(key, (batch_size,))
    dones = jnp.zeros((batch_size,))

    sample_batch_full = SampleBatch(
        obs=obs,
        actions=actions,
        rewards=rewards,
        next_obs=next_obs,
        dones=dones,
    )

    losses = agent.loss(agent_state, sample_batch_full, key)

    print(f"✓ Loss computation:")
    print(f"  Actor loss: {losses['actor_loss']:.4f}")
    print(f"  Critic loss: {losses['critic_loss']:.4f}")
    print(f"  Mean Q: {losses['mean_q']:.4f}")

    assert 'actor_loss' in losses
    assert 'critic_loss' in losses

    # Test soft target update
    new_params = soft_target_update(agent_state.params, tau=0.005)
    print(f"✓ Soft target update completed")

    # Verify targets changed slightly
    actor_param = jax.tree_util.tree_leaves(agent_state.params.actor_params)[0]
    target_param_old = jax.tree_util.tree_leaves(agent_state.params.target_actor_params)[0]
    target_param_new = jax.tree_util.tree_leaves(new_params.target_actor_params)[0]

    assert not jnp.array_equal(target_param_old, target_param_new)
    print(f"✓ Target networks updated (verified parameter change)")

    print("\n✓ All TradingAgent tests passed!")

    return agent, agent_state


if __name__ == "__main__":
    test_trading_agent()
