"""Unit tests for Actor network"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as random
from eigen3.models.actor import Actor


class TestActor:
    """Test Actor network"""

    def test_actor_forward(self):
        """Test forward pass"""
        actor = Actor(use_remat=False)

        key = random.PRNGKey(0)
        batch_size = 2
        context_days = 504

        state = random.normal(key, (batch_size, context_days, 669, 5))
        params = actor.init(key, state, train=False, return_attention_weights=False)

        actions, attn_weights = actor.apply(
            params, state, train=False, return_attention_weights=False
        )

        # Check output shape
        assert actions.shape == (batch_size, 108, 2)
        assert attn_weights is None

    def test_actor_action_ranges(self):
        """Test that actions are within valid ranges"""
        actor = Actor(use_remat=False)

        key = random.PRNGKey(42)
        state = random.normal(key, (4, 100, 669, 5))  # Smaller context for speed

        params = actor.init(key, state, train=False)
        actions, _ = actor.apply(params, state, train=False)

        # Extract coefficients and sale targets
        coefficients = actions[:, :, 0]
        sale_targets = actions[:, :, 1]

        # Coefficient should be >= 0
        assert jnp.all(coefficients >= 0), f"Negative coefficients: {jnp.min(coefficients)}"

        # Sale targets should be in [10, 50]
        assert jnp.all(sale_targets >= 10.0), f"Sale target too low: {jnp.min(sale_targets)}"
        assert jnp.all(sale_targets <= 50.0), f"Sale target too high: {jnp.max(sale_targets)}"

    def test_actor_with_attention_weights(self):
        """Test forward pass with attention weights"""
        actor = Actor(use_attention=True, use_remat=False)

        key = random.PRNGKey(0)
        state = random.normal(key, (2, 100, 669, 5))

        params = actor.init(key, state, train=False, return_attention_weights=True)
        actions, attn_weights = actor.apply(
            params, state, train=False, return_attention_weights=True
        )

        assert actions.shape == (2, 108, 2)
        assert attn_weights is not None
        assert attn_weights.shape == (2, 669)

    def test_actor_without_attention(self):
        """Test Actor without attention module"""
        actor = Actor(use_attention=False, use_remat=False)

        key = random.PRNGKey(0)
        state = random.normal(key, (2, 100, 669, 5))

        params = actor.init(key, state, train=False)
        actions, attn_weights = actor.apply(params, state, train=False)

        assert actions.shape == (2, 108, 2)
        # Attention weights should still be None
        assert attn_weights is None

    def test_actor_with_remat(self):
        """Test forward pass with gradient checkpointing"""
        actor = Actor(use_remat=True)

        key = random.PRNGKey(0)
        state = random.normal(key, (2, 100, 669, 5))

        params = actor.init(key, state, train=False)
        actions_train, _ = actor.apply(params, state, train=True)
        actions_eval, _ = actor.apply(params, state, train=False)

        assert actions_train.shape == (2, 108, 2)
        assert actions_eval.shape == (2, 108, 2)

    def test_actor_gradient(self):
        """Test that gradients can be computed"""
        actor = Actor(use_remat=False)

        key = random.PRNGKey(0)
        state = random.normal(key, (2, 50, 669, 5))  # Small for speed

        def loss_fn(params):
            actions, _ = actor.apply(params, state, train=True)
            return jnp.mean(actions ** 2)

        params = actor.init(key, state, train=True)
        loss, grads = jax.value_and_grad(loss_fn)(params)

        assert jnp.isfinite(loss)
        # Check that all gradients are finite
        for leaf in jax.tree_util.tree_leaves(grads):
            assert jnp.all(jnp.isfinite(leaf))

    def test_actor_coefficient_activation(self):
        """Test coefficient activation function"""
        actor = Actor()

        # Test positive values (should be >= 1)
        positive_raw = jnp.array([1.0, 2.0, 5.0])
        positive_activated = actor._apply_coefficient_activation(positive_raw)
        assert jnp.all(positive_activated >= 1.0)

        # Test negative values (should be ~0)
        negative_raw = jnp.array([-1.0, -2.0, -5.0])
        negative_activated = actor._apply_coefficient_activation(negative_raw)
        assert jnp.all(negative_activated < 0.2)

        # Test zero
        zero_raw = jnp.array([0.0])
        zero_activated = actor._apply_coefficient_activation(zero_raw)
        assert zero_activated[0] < 1.0  # Should be near 0.5 + exp(0)*0.5 = 1.0

    def test_actor_deterministic(self):
        """Test that eval mode is deterministic"""
        actor = Actor(use_remat=False)

        key = random.PRNGKey(0)
        state = random.normal(key, (2, 100, 669, 5))

        params = actor.init(key, state, train=False)

        # Run multiple times
        actions1, _ = actor.apply(params, state, train=False)
        actions2, _ = actor.apply(params, state, train=False)

        # Should be identical in eval mode
        assert jnp.array_equal(actions1, actions2)

    def test_actor_jit(self):
        """Test that Actor can be JIT compiled"""
        actor = Actor(use_remat=False)

        key = random.PRNGKey(0)
        state = random.normal(key, (2, 100, 669, 5))
        params = actor.init(key, state, train=False)

        @jax.jit
        def forward(params, state):
            return actor.apply(params, state, train=False)

        actions, _ = forward(params, state)
        assert actions.shape == (2, 108, 2)

    def test_actor_small_batch(self):
        """Test with batch size of 1"""
        actor = Actor(use_remat=False)

        key = random.PRNGKey(0)
        state = random.normal(key, (1, 100, 669, 5))

        params = actor.init(key, state, train=False)
        actions, _ = actor.apply(params, state, train=False)

        assert actions.shape == (1, 108, 2)

    def test_actor_large_batch(self):
        """Test with larger batch size"""
        actor = Actor(use_remat=False)

        key = random.PRNGKey(0)
        batch_size = 16
        state = random.normal(key, (batch_size, 100, 669, 5))

        params = actor.init(key, state, train=False)
        actions, _ = actor.apply(params, state, train=False)

        assert actions.shape == (batch_size, 108, 2)

    def test_actor_different_context_lengths(self):
        """Test with different context window sizes"""
        actor = Actor(use_remat=False)

        key = random.PRNGKey(0)

        for context_days in [100, 200, 504]:
            state = random.normal(key, (2, context_days, 669, 5))
            params = actor.init(key, state, train=False)
            actions, _ = actor.apply(params, state, train=False)

            assert actions.shape == (2, 108, 2)

    @pytest.mark.slow
    def test_actor_full_size(self):
        """Test with full eigen2 dimensions"""
        actor = Actor(
            use_attention=True,
            use_remat=True
        )

        key = random.PRNGKey(0)
        batch_size = 8
        context_days = 504

        state = random.normal(key, (batch_size, context_days, 669, 5))

        params = actor.init(key, state, train=False)
        actions, attn_weights = actor.apply(
            params, state, train=True, return_attention_weights=True
        )

        assert actions.shape == (batch_size, 108, 2)
        assert attn_weights.shape == (batch_size, 669)
        assert jnp.all(jnp.isfinite(actions))
        assert jnp.all(jnp.isfinite(attn_weights))

        # Check ranges
        coefficients = actions[:, :, 0]
        sale_targets = actions[:, :, 1]

        assert jnp.all(coefficients >= 0)
        assert jnp.all(sale_targets >= 10.0)
        assert jnp.all(sale_targets <= 50.0)


class TestActorTrainingMode:
    """Test Actor in training vs eval mode"""

    def test_dropout_difference(self):
        """Test that dropout causes differences between runs"""
        actor = Actor(dropout_rate=0.5, use_remat=False)  # High dropout for testing

        key = random.PRNGKey(42)
        state = random.normal(key, (2, 100, 669, 5))

        params = actor.init(key, state, train=True)

        # Training mode with different dropout keys
        actions1, _ = actor.apply(params, state, train=True, rngs={'dropout': random.PRNGKey(0)})
        actions2, _ = actor.apply(params, state, train=True, rngs={'dropout': random.PRNGKey(1)})

        # Should be different due to dropout
        assert not jnp.array_equal(actions1, actions2)

    def test_eval_mode_no_dropout(self):
        """Test that eval mode disables dropout"""
        actor = Actor(dropout_rate=0.5, use_remat=False)

        key = random.PRNGKey(42)
        state = random.normal(key, (2, 100, 669, 5))

        params = actor.init(key, state, train=False)

        # Eval mode should be deterministic
        actions1, _ = actor.apply(params, state, train=False)
        actions2, _ = actor.apply(params, state, train=False)

        assert jnp.array_equal(actions1, actions2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
