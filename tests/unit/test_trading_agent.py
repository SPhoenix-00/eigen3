"""Unit tests for TradingAgent"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as random
from evorl.envs import Box
from evorl.sample_batch import SampleBatch

from eigen3.agents.trading_agent import TradingAgent, TradingNetworkParams, soft_target_update
from eigen3.models.actor import Actor
from eigen3.models.critic import Critic, DoubleCritic


def create_test_agent():
    """Create a test agent"""
    actor = Actor(use_remat=False)
    critic = DoubleCritic(use_remat=False)

    agent = TradingAgent(
        actor_network=actor,
        critic_network=critic,
        exploration_noise=0.1,
        discount=0.99,
        tau=0.005,
    )

    return agent


def create_test_spaces():
    """Create test observation and action spaces"""
    obs_space = Box(low=-jnp.inf, high=jnp.inf, shape=(504, 669, 5))
    action_space = Box(low=0.0, high=jnp.inf, shape=(108, 2))
    return obs_space, action_space


class TestTradingAgent:
    """Test TradingAgent basic functionality"""

    def test_agent_initialization(self):
        """Test agent initialization"""
        agent = create_test_agent()
        obs_space, action_space = create_test_spaces()

        key = random.PRNGKey(0)
        agent_state = agent.init(obs_space, action_space, key)

        # Check that params were created
        assert hasattr(agent_state.params, 'actor_params')
        assert hasattr(agent_state.params, 'critic_params')
        assert hasattr(agent_state.params, 'target_actor_params')
        assert hasattr(agent_state.params, 'target_critic_params')

    def test_compute_actions(self):
        """Test action computation with exploration noise"""
        agent = create_test_agent()
        obs_space, action_space = create_test_spaces()

        key = random.PRNGKey(0)
        agent_state = agent.init(obs_space, action_space, key)

        # Create batch
        batch_size = 4
        obs = random.normal(key, (batch_size, 504, 669, 5))
        sample_batch = SampleBatch(obs=obs)

        # Get actions
        key, action_key = random.split(key)
        actions, policy_info = agent.compute_actions(agent_state, sample_batch, action_key)

        # Check shape
        assert actions.shape == (batch_size, 108, 2)

        # Check ranges
        coefficients = actions[:, :, 0]
        sale_targets = actions[:, :, 1]

        assert jnp.all(coefficients >= 0)
        assert jnp.all(sale_targets >= 10.0)
        assert jnp.all(sale_targets <= 50.0)

    def test_evaluate_actions(self):
        """Test deterministic action computation"""
        agent = create_test_agent()
        obs_space, action_space = create_test_spaces()

        key = random.PRNGKey(0)
        agent_state = agent.init(obs_space, action_space, key)

        # Create batch
        obs = random.normal(key, (2, 504, 669, 5))
        sample_batch = SampleBatch(obs=obs)

        # Get actions twice
        actions1, _ = agent.evaluate_actions(agent_state, sample_batch, key)
        actions2, _ = agent.evaluate_actions(agent_state, sample_batch, key)

        # Should be identical (deterministic)
        assert jnp.array_equal(actions1, actions2)

    def test_exploration_noise(self):
        """Test that exploration noise is added"""
        agent = create_test_agent()
        obs_space, action_space = create_test_spaces()

        key = random.PRNGKey(0)
        agent_state = agent.init(obs_space, action_space, key)

        obs = random.normal(key, (2, 504, 669, 5))
        sample_batch = SampleBatch(obs=obs)

        # Get deterministic actions
        eval_actions, _ = agent.evaluate_actions(agent_state, sample_batch, key)

        # Get noisy actions
        key, noise_key1, noise_key2 = random.split(key, 3)
        noisy_actions1, _ = agent.compute_actions(agent_state, sample_batch, noise_key1)
        noisy_actions2, _ = agent.compute_actions(agent_state, sample_batch, noise_key2)

        # Noisy actions should differ from deterministic
        assert not jnp.array_equal(noisy_actions1, eval_actions)

        # Different noise keys should produce different actions
        assert not jnp.array_equal(noisy_actions1, noisy_actions2)

    def test_loss_computation(self):
        """Test loss computation"""
        agent = create_test_agent()
        obs_space, action_space = create_test_spaces()

        key = random.PRNGKey(0)
        agent_state = agent.init(obs_space, action_space, key)

        # Create batch
        batch_size = 8
        obs = random.normal(key, (batch_size, 504, 669, 5))
        actions = random.normal(key, (batch_size, 108, 2))
        rewards = random.normal(key, (batch_size,))
        next_obs = random.normal(key, (batch_size, 504, 669, 5))
        dones = jnp.zeros((batch_size,))

        sample_batch = SampleBatch(
            obs=obs,
            actions=actions,
            rewards=rewards,
            next_obs=next_obs,
            dones=dones,
        )

        # Compute losses
        losses = agent.loss(agent_state, sample_batch, key)

        # Check that losses exist
        assert 'actor_loss' in losses
        assert 'critic_loss' in losses
        assert 'mean_q' in losses

        # Check that losses are finite
        assert jnp.isfinite(losses['actor_loss'])
        assert jnp.isfinite(losses['critic_loss'])

    def test_gradient_flow(self):
        """Test that gradients flow through loss"""
        agent = create_test_agent()
        obs_space, action_space = create_test_spaces()

        key = random.PRNGKey(0)
        agent_state = agent.init(obs_space, action_space, key)

        # Create batch
        obs = random.normal(key, (4, 504, 669, 5))
        actions = random.normal(key, (4, 108, 2))
        rewards = random.normal(key, (4,))
        next_obs = random.normal(key, (4, 504, 669, 5))
        dones = jnp.zeros((4,))

        sample_batch = SampleBatch(
            obs=obs,
            actions=actions,
            rewards=rewards,
            next_obs=next_obs,
            dones=dones,
        )

        # Compute gradients
        def loss_fn(params):
            temp_state = agent_state.replace(params=params)
            losses = agent.loss(temp_state, sample_batch, key)
            return losses['actor_loss'] + losses['critic_loss']

        loss, grads = jax.value_and_grad(loss_fn)(agent_state.params)

        # Check that gradients exist and are finite
        assert jnp.isfinite(loss)
        for leaf in jax.tree_util.tree_leaves(grads):
            assert jnp.all(jnp.isfinite(leaf))


class TestSoftTargetUpdate:
    """Test soft target update functionality"""

    def test_soft_update_basic(self):
        """Test basic soft target update"""
        agent = create_test_agent()
        obs_space, action_space = create_test_spaces()

        key = random.PRNGKey(0)
        agent_state = agent.init(obs_space, action_space, key)

        # Perform soft update
        new_params = soft_target_update(agent_state.params, tau=0.1)

        # Check that targets changed
        old_target = jax.tree_util.tree_leaves(agent_state.params.target_actor_params)[0]
        new_target = jax.tree_util.tree_leaves(new_params.target_actor_params)[0]

        assert not jnp.array_equal(old_target, new_target)

    def test_soft_update_tau_effect(self):
        """Test that tau affects update magnitude"""
        agent = create_test_agent()
        obs_space, action_space = create_test_spaces()

        key = random.PRNGKey(0)
        agent_state = agent.init(obs_space, action_space, key)

        # Get a leaf from source and target
        source_leaf = jax.tree_util.tree_leaves(agent_state.params.actor_params)[0]
        target_leaf = jax.tree_util.tree_leaves(agent_state.params.target_actor_params)[0]

        # Update with small tau
        small_tau_params = soft_target_update(agent_state.params, tau=0.001)
        small_tau_leaf = jax.tree_util.tree_leaves(small_tau_params.target_actor_params)[0]

        # Update with large tau
        large_tau_params = soft_target_update(agent_state.params, tau=0.1)
        large_tau_leaf = jax.tree_util.tree_leaves(large_tau_params.target_actor_params)[0]

        # Large tau should move target closer to source
        small_diff = jnp.abs(small_tau_leaf - source_leaf).mean()
        large_diff = jnp.abs(large_tau_leaf - source_leaf).mean()

        assert large_diff < small_diff

    def test_soft_update_tau_one(self):
        """Test that tau=1.0 copies source to target"""
        agent = create_test_agent()
        obs_space, action_space = create_test_spaces()

        key = random.PRNGKey(0)
        agent_state = agent.init(obs_space, action_space, key)

        # Update with tau=1.0 (full copy)
        new_params = soft_target_update(agent_state.params, tau=1.0)

        # Target should equal source
        source_leaf = jax.tree_util.tree_leaves(agent_state.params.actor_params)[0]
        target_leaf = jax.tree_util.tree_leaves(new_params.target_actor_params)[0]

        assert jnp.allclose(source_leaf, target_leaf)


class TestNetworkParams:
    """Test TradingNetworkParams structure"""

    def test_params_structure(self):
        """Test that params have correct structure"""
        agent = create_test_agent()
        obs_space, action_space = create_test_spaces()

        key = random.PRNGKey(0)
        agent_state = agent.init(obs_space, action_space, key)

        params = agent_state.params

        # Check structure
        assert isinstance(params, TradingNetworkParams)
        assert hasattr(params, 'actor_params')
        assert hasattr(params, 'critic_params')
        assert hasattr(params, 'target_actor_params')
        assert hasattr(params, 'target_critic_params')

    def test_params_are_pytrees(self):
        """Test that params are valid PyTrees"""
        agent = create_test_agent()
        obs_space, action_space = create_test_spaces()

        key = random.PRNGKey(0)
        agent_state = agent.init(obs_space, action_space, key)

        # Should be able to tree_map
        doubled_params = jax.tree_map(lambda x: x * 2, agent_state.params)

        # Should have same structure
        assert isinstance(doubled_params, TradingNetworkParams)


class TestDifferentCriticTypes:
    """Test with different critic architectures"""

    def test_single_critic(self):
        """Test with single critic network"""
        actor = Actor(use_remat=False)
        critic = Critic(use_remat=False)  # Single critic

        agent = TradingAgent(
            actor_network=actor,
            critic_network=critic,
        )

        obs_space, action_space = create_test_spaces()

        key = random.PRNGKey(0)
        agent_state = agent.init(obs_space, action_space, key)

        # Create batch
        obs = random.normal(key, (4, 504, 669, 5))
        actions = random.normal(key, (4, 108, 2))
        rewards = random.normal(key, (4,))
        next_obs = random.normal(key, (4, 504, 669, 5))
        dones = jnp.zeros((4,))

        sample_batch = SampleBatch(
            obs=obs,
            actions=actions,
            rewards=rewards,
            next_obs=next_obs,
            dones=dones,
        )

        # Should work with single critic
        losses = agent.loss(agent_state, sample_batch, key)
        assert 'actor_loss' in losses
        assert 'critic_loss' in losses

    def test_double_critic(self):
        """Test with double critic network"""
        actor = Actor(use_remat=False)
        critic = DoubleCritic(use_remat=False)  # Twin critics

        agent = TradingAgent(
            actor_network=actor,
            critic_network=critic,
        )

        obs_space, action_space = create_test_spaces()

        key = random.PRNGKey(0)
        agent_state = agent.init(obs_space, action_space, key)

        # Create batch
        obs = random.normal(key, (4, 504, 669, 5))
        actions = random.normal(key, (4, 108, 2))
        rewards = random.normal(key, (4,))
        next_obs = random.normal(key, (4, 504, 669, 5))
        dones = jnp.zeros((4,))

        sample_batch = SampleBatch(
            obs=obs,
            actions=actions,
            rewards=rewards,
            next_obs=next_obs,
            dones=dones,
        )

        # Should work with twin critics
        losses = agent.loss(agent_state, sample_batch, key)
        assert 'actor_loss' in losses
        assert 'critic_loss' in losses


class TestJAXFeatures:
    """Test JAX-specific features"""

    def test_agent_is_jittable(self):
        """Test that agent methods can be JIT compiled"""
        agent = create_test_agent()
        obs_space, action_space = create_test_spaces()

        key = random.PRNGKey(0)
        agent_state = agent.init(obs_space, action_space, key)

        obs = random.normal(key, (2, 504, 669, 5))
        sample_batch = SampleBatch(obs=obs)

        @jax.jit
        def jitted_compute(state, batch, key):
            return agent.compute_actions(state, batch, key)

        actions, _ = jitted_compute(agent_state, sample_batch, key)
        assert actions.shape == (2, 108, 2)

    def test_loss_is_jittable(self):
        """Test that loss computation can be JIT compiled"""
        agent = create_test_agent()
        obs_space, action_space = create_test_spaces()

        key = random.PRNGKey(0)
        agent_state = agent.init(obs_space, action_space, key)

        obs = random.normal(key, (4, 504, 669, 5))
        actions = random.normal(key, (4, 108, 2))
        rewards = random.normal(key, (4,))
        next_obs = random.normal(key, (4, 504, 669, 5))
        dones = jnp.zeros((4,))

        sample_batch = SampleBatch(
            obs=obs,
            actions=actions,
            rewards=rewards,
            next_obs=next_obs,
            dones=dones,
        )

        @jax.jit
        def jitted_loss(state, batch, key):
            return agent.loss(state, batch, key)

        losses = jitted_loss(agent_state, sample_batch, key)
        assert 'actor_loss' in losses

    def test_agent_is_vmappable(self):
        """Test that agent can be vectorized"""
        agent = create_test_agent()
        obs_space, action_space = create_test_spaces()

        # Create batch of agent states
        batch_size = 4
        keys = jax.random.split(random.PRNGKey(0), batch_size)

        # Note: init is not directly vmappable due to network initialization
        # but we can vmap compute_actions and evaluate_actions

        key = random.PRNGKey(0)
        agent_state = agent.init(obs_space, action_space, key)

        # Create batch of observations
        obs = random.normal(key, (batch_size, 504, 669, 5))

        # Create sample batches for each
        def get_actions_single(obs_single, key_single):
            batch = SampleBatch(obs=obs_single[None, ...])
            actions, _ = agent.evaluate_actions(agent_state, batch, key_single)
            return actions[0]

        # Vectorize
        actions = jax.vmap(get_actions_single)(obs, keys)
        assert actions.shape == (batch_size, 108, 2)


@pytest.mark.slow
class TestFullScale:
    """Test with full-scale dimensions"""

    def test_full_scale_agent(self):
        """Test agent with full dimensions"""
        actor = Actor(use_remat=True)
        critic = DoubleCritic(use_remat=True)

        agent = TradingAgent(
            actor_network=actor,
            critic_network=critic,
        )

        obs_space, action_space = create_test_spaces()

        key = random.PRNGKey(0)
        agent_state = agent.init(obs_space, action_space, key)

        # Full batch
        batch_size = 64
        obs = random.normal(key, (batch_size, 504, 669, 5))
        actions = random.normal(key, (batch_size, 108, 2))
        rewards = random.normal(key, (batch_size,))
        next_obs = random.normal(key, (batch_size, 504, 669, 5))
        dones = jnp.zeros((batch_size,))

        sample_batch = SampleBatch(
            obs=obs,
            actions=actions,
            rewards=rewards,
            next_obs=next_obs,
            dones=dones,
        )

        # Compute losses
        losses = agent.loss(agent_state, sample_batch, key)

        assert jnp.isfinite(losses['actor_loss'])
        assert jnp.isfinite(losses['critic_loss'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
