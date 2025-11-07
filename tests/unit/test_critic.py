"""Unit tests for Critic network"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as random
from eigen3.models.critic import Critic, DoubleCritic


class TestCritic:
    """Test Critic network"""

    def test_critic_forward(self):
        """Test forward pass"""
        critic = Critic(use_attention=False, use_remat=False)

        key = random.PRNGKey(0)
        batch_size = 2
        context_days = 504

        state = random.normal(key, (batch_size, context_days, 669, 5))
        action = random.normal(key, (batch_size, 108, 2))

        params = critic.init(key, state, action, train=False)
        q_values = critic.apply(params, state, action, train=False)

        # Check output shape
        assert q_values.shape == (batch_size, 1)

    def test_critic_with_attention(self):
        """Test Critic with self-attention"""
        critic = Critic(use_attention=True, use_remat=False)

        key = random.PRNGKey(0)
        state = random.normal(key, (2, 100, 669, 5))
        action = random.normal(key, (2, 108, 2))

        params = critic.init(key, state, action, train=False)
        q_values = critic.apply(params, state, action, train=False)

        assert q_values.shape == (2, 1)

    def test_critic_without_attention(self):
        """Test Critic without attention"""
        critic = Critic(use_attention=False, use_remat=False)

        key = random.PRNGKey(0)
        state = random.normal(key, (2, 100, 669, 5))
        action = random.normal(key, (2, 108, 2))

        params = critic.init(key, state, action, train=False)
        q_values = critic.apply(params, state, action, train=False)

        assert q_values.shape == (2, 1)

    def test_critic_with_remat(self):
        """Test forward pass with gradient checkpointing"""
        critic = Critic(use_remat=True)

        key = random.PRNGKey(0)
        state = random.normal(key, (2, 100, 669, 5))
        action = random.normal(key, (2, 108, 2))

        params = critic.init(key, state, action, train=False)
        q_values_train = critic.apply(params, state, action, train=True)
        q_values_eval = critic.apply(params, state, action, train=False)

        assert q_values_train.shape == (2, 1)
        assert q_values_eval.shape == (2, 1)

    def test_critic_gradient(self):
        """Test that gradients can be computed"""
        critic = Critic(use_remat=False)

        key = random.PRNGKey(0)
        state = random.normal(key, (2, 50, 669, 5))
        action = random.normal(key, (2, 108, 2))

        def loss_fn(params):
            q_values = critic.apply(params, state, action, train=True)
            return jnp.mean(q_values ** 2)

        params = critic.init(key, state, action, train=True)
        loss, grads = jax.value_and_grad(loss_fn)(params)

        assert jnp.isfinite(loss)
        # Check that all gradients are finite
        for leaf in jax.tree_util.tree_leaves(grads):
            assert jnp.all(jnp.isfinite(leaf))

    def test_critic_deterministic(self):
        """Test that eval mode is deterministic"""
        critic = Critic(use_remat=False)

        key = random.PRNGKey(0)
        state = random.normal(key, (2, 100, 669, 5))
        action = random.normal(key, (2, 108, 2))

        params = critic.init(key, state, action, train=False)

        # Run multiple times
        q_values1 = critic.apply(params, state, action, train=False)
        q_values2 = critic.apply(params, state, action, train=False)

        # Should be identical in eval mode
        assert jnp.array_equal(q_values1, q_values2)

    def test_critic_jit(self):
        """Test that Critic can be JIT compiled"""
        critic = Critic(use_remat=False)

        key = random.PRNGKey(0)
        state = random.normal(key, (2, 100, 669, 5))
        action = random.normal(key, (2, 108, 2))
        params = critic.init(key, state, action, train=False)

        @jax.jit
        def forward(params, state, action):
            return critic.apply(params, state, action, train=False)

        q_values = forward(params, state, action)
        assert q_values.shape == (2, 1)

    def test_critic_different_batch_sizes(self):
        """Test with different batch sizes"""
        critic = Critic(use_remat=False)

        key = random.PRNGKey(0)

        for batch_size in [1, 4, 8, 16]:
            state = random.normal(key, (batch_size, 100, 669, 5))
            action = random.normal(key, (batch_size, 108, 2))

            params = critic.init(key, state, action, train=False)
            q_values = critic.apply(params, state, action, train=False)

            assert q_values.shape == (batch_size, 1)

    def test_critic_different_context_lengths(self):
        """Test with different context window sizes"""
        critic = Critic(use_remat=False)

        key = random.PRNGKey(0)

        for context_days in [100, 200, 504]:
            state = random.normal(key, (2, context_days, 669, 5))
            action = random.normal(key, (2, 108, 2))

            params = critic.init(key, state, action, train=False)
            q_values = critic.apply(params, state, action, train=False)

            assert q_values.shape == (2, 1)


class TestDoubleCritic:
    """Test DoubleCritic (twin critics for TD3)"""

    def test_double_critic_forward(self):
        """Test forward pass"""
        double_critic = DoubleCritic(use_attention=False, use_remat=False)

        key = random.PRNGKey(0)
        batch_size = 2
        context_days = 504

        state = random.normal(key, (batch_size, context_days, 669, 5))
        action = random.normal(key, (batch_size, 108, 2))

        params = double_critic.init(key, state, action, train=False)
        q_values = double_critic.apply(params, state, action, train=False)

        # Should return Q-values from both critics
        assert q_values.shape == (batch_size, 2)

    def test_double_critic_independence(self):
        """Test that the two critics are independent"""
        double_critic = DoubleCritic(use_remat=False)

        key = random.PRNGKey(0)
        state = random.normal(key, (4, 100, 669, 5))
        action = random.normal(key, (4, 108, 2))

        params = double_critic.init(key, state, action, train=False)
        q_values = double_critic.apply(params, state, action, train=False)

        # Extract Q-values from each critic
        q1 = q_values[:, 0]
        q2 = q_values[:, 1]

        # They should not be identical (different parameters)
        assert not jnp.array_equal(q1, q2)

    def test_double_critic_with_remat(self):
        """Test DoubleCritic with gradient checkpointing"""
        double_critic = DoubleCritic(use_remat=True)

        key = random.PRNGKey(0)
        state = random.normal(key, (2, 100, 669, 5))
        action = random.normal(key, (2, 108, 2))

        params = double_critic.init(key, state, action, train=False)
        q_values_train = double_critic.apply(params, state, action, train=True)
        q_values_eval = double_critic.apply(params, state, action, train=False)

        assert q_values_train.shape == (2, 2)
        assert q_values_eval.shape == (2, 2)

    def test_double_critic_gradient(self):
        """Test that gradients can be computed"""
        double_critic = DoubleCritic(use_remat=False)

        key = random.PRNGKey(0)
        state = random.normal(key, (2, 50, 669, 5))
        action = random.normal(key, (2, 108, 2))

        def loss_fn(params):
            q_values = double_critic.apply(params, state, action, train=True)
            # Use both critics in loss
            return jnp.mean(q_values ** 2)

        params = double_critic.init(key, state, action, train=True)
        loss, grads = jax.value_and_grad(loss_fn)(params)

        assert jnp.isfinite(loss)
        for leaf in jax.tree_util.tree_leaves(grads):
            assert jnp.all(jnp.isfinite(leaf))

    def test_double_critic_min_q(self):
        """Test taking minimum Q-value (common in TD3)"""
        double_critic = DoubleCritic(use_remat=False)

        key = random.PRNGKey(0)
        state = random.normal(key, (4, 100, 669, 5))
        action = random.normal(key, (4, 108, 2))

        params = double_critic.init(key, state, action, train=False)
        q_values = double_critic.apply(params, state, action, train=False)

        # Take minimum Q-value across critics (TD3 strategy)
        min_q = jnp.min(q_values, axis=-1, keepdims=True)

        assert min_q.shape == (4, 1)

    def test_double_critic_jit(self):
        """Test that DoubleCritic can be JIT compiled"""
        double_critic = DoubleCritic(use_remat=False)

        key = random.PRNGKey(0)
        state = random.normal(key, (2, 100, 669, 5))
        action = random.normal(key, (2, 108, 2))
        params = double_critic.init(key, state, action, train=False)

        @jax.jit
        def forward(params, state, action):
            return double_critic.apply(params, state, action, train=False)

        q_values = forward(params, state, action)
        assert q_values.shape == (2, 2)


class TestCriticTrainingMode:
    """Test Critic in training vs eval mode"""

    def test_dropout_difference(self):
        """Test that dropout causes differences between runs"""
        critic = Critic(dropout_rate=0.5, use_remat=False)  # High dropout

        key = random.PRNGKey(42)
        state = random.normal(key, (2, 100, 669, 5))
        action = random.normal(key, (2, 108, 2))

        params = critic.init(key, state, action, train=True)

        # Training mode with different dropout keys
        q1 = critic.apply(params, state, action, train=True, rngs={'dropout': random.PRNGKey(0)})
        q2 = critic.apply(params, state, action, train=True, rngs={'dropout': random.PRNGKey(1)})

        # Should be different due to dropout
        assert not jnp.array_equal(q1, q2)

    def test_eval_mode_no_dropout(self):
        """Test that eval mode disables dropout"""
        critic = Critic(dropout_rate=0.5, use_remat=False)

        key = random.PRNGKey(42)
        state = random.normal(key, (2, 100, 669, 5))
        action = random.normal(key, (2, 108, 2))

        params = critic.init(key, state, action, train=False)

        # Eval mode should be deterministic
        q1 = critic.apply(params, state, action, train=False)
        q2 = critic.apply(params, state, action, train=False)

        assert jnp.array_equal(q1, q2)


@pytest.mark.slow
class TestCriticFullScale:
    """Test with full eigen2 dimensions"""

    def test_critic_full_size(self):
        """Test Critic with full dimensions"""
        critic = Critic(use_attention=False, use_remat=True)

        key = random.PRNGKey(0)
        batch_size = 8
        context_days = 504

        state = random.normal(key, (batch_size, context_days, 669, 5))
        action = random.normal(key, (batch_size, 108, 2))

        params = critic.init(key, state, action, train=False)
        q_values = critic.apply(params, state, action, train=True)

        assert q_values.shape == (batch_size, 1)
        assert jnp.all(jnp.isfinite(q_values))

    def test_double_critic_full_size(self):
        """Test DoubleCritic with full dimensions"""
        double_critic = DoubleCritic(use_attention=False, use_remat=True)

        key = random.PRNGKey(0)
        batch_size = 8
        context_days = 504

        state = random.normal(key, (batch_size, context_days, 669, 5))
        action = random.normal(key, (batch_size, 108, 2))

        params = double_critic.init(key, state, action, train=False)
        q_values = double_critic.apply(params, state, action, train=True)

        assert q_values.shape == (batch_size, 2)
        assert jnp.all(jnp.isfinite(q_values))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
