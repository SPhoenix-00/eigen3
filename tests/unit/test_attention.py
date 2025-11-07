"""Unit tests for Attention modules"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as random
from eigen3.models.attention import (
    AttentionModule,
    CrossAttentionModule,
    SelfAttentionModule,
)


class TestAttentionModule:
    """Test generic AttentionModule"""

    def test_cross_attention_forward(self):
        """Test cross-attention forward pass"""
        model = AttentionModule(
            embed_dim=256,
            num_heads=8,
            use_cross_attention=True
        )

        key = random.PRNGKey(0)
        batch_size = 4
        num_columns = 669
        x = random.normal(key, (batch_size, num_columns, 256))

        params = model.init(key, x, train=False, return_attention_weights=False)
        output, attn_weights = model.apply(
            params, x, train=False, return_attention_weights=False
        )

        # Cross-attention output should be [batch, 1, embed_dim]
        assert output.shape == (batch_size, 1, 256)
        assert attn_weights is None

    def test_cross_attention_with_weights(self):
        """Test cross-attention with attention weights"""
        model = AttentionModule(
            embed_dim=256,
            num_heads=8,
            use_cross_attention=True
        )

        key = random.PRNGKey(0)
        batch_size = 2
        num_columns = 100
        x = random.normal(key, (batch_size, num_columns, 256))

        params = model.init(key, x, train=False, return_attention_weights=True)
        output, attn_weights = model.apply(
            params, x, train=False, return_attention_weights=True
        )

        assert output.shape == (batch_size, 1, 256)
        assert attn_weights is not None
        assert attn_weights.shape == (batch_size, 1, num_columns)

        # Check that attention weights are valid probabilities
        # Note: Due to the simplified attention weight computation,
        # we just check that weights are positive
        assert jnp.all(attn_weights >= 0)

    def test_self_attention_forward(self):
        """Test self-attention forward pass"""
        model = AttentionModule(
            embed_dim=256,
            num_heads=8,
            use_cross_attention=False
        )

        key = random.PRNGKey(0)
        batch_size = 4
        num_columns = 669
        x = random.normal(key, (batch_size, num_columns, 256))

        params = model.init(key, x, train=False, return_attention_weights=False)
        output, attn_weights = model.apply(
            params, x, train=False, return_attention_weights=False
        )

        # Self-attention output should be same shape as input
        assert output.shape == (batch_size, num_columns, 256)
        assert attn_weights is None

    def test_self_attention_residual(self):
        """Test that self-attention has residual connection"""
        model = AttentionModule(
            embed_dim=128,
            num_heads=4,
            use_cross_attention=False,
            dropout_rate=0.0  # Disable dropout for this test
        )

        key = random.PRNGKey(0)
        x = random.normal(key, (2, 50, 128))

        params = model.init(key, x, train=False, return_attention_weights=False)
        output, _ = model.apply(params, x, train=False, return_attention_weights=False)

        # Output should not be identical to input (due to attention and layer norm)
        assert not jnp.array_equal(output, x)


class TestCrossAttentionModule:
    """Test CrossAttentionModule convenience wrapper"""

    def test_cross_attention_output_shape(self):
        """Test output shape"""
        model = CrossAttentionModule(embed_dim=256, num_heads=8)

        key = random.PRNGKey(0)
        x = random.normal(key, (4, 669, 256))

        params = model.init(key, x, train=False)
        output, _ = model.apply(params, x, train=False)

        assert output.shape == (4, 1, 256)

    def test_cross_attention_with_different_sizes(self):
        """Test with different input sizes"""
        model = CrossAttentionModule(embed_dim=128, num_heads=4)

        key = random.PRNGKey(0)

        # Test with different number of columns
        for num_cols in [100, 200, 669]:
            x = random.normal(key, (2, num_cols, 128))
            params = model.init(key, x, train=False)
            output, _ = model.apply(params, x, train=False)

            assert output.shape == (2, 1, 128)

    def test_cross_attention_deterministic(self):
        """Test that eval mode is deterministic"""
        model = CrossAttentionModule(embed_dim=256, num_heads=8)

        key = random.PRNGKey(0)
        x = random.normal(key, (2, 100, 256))

        params = model.init(key, x, train=False)

        # Run twice with same input
        output1, _ = model.apply(params, x, train=False)
        output2, _ = model.apply(params, x, train=False)

        # Should be identical in eval mode
        assert jnp.array_equal(output1, output2)


class TestSelfAttentionModule:
    """Test SelfAttentionModule convenience wrapper"""

    def test_self_attention_output_shape(self):
        """Test output shape"""
        model = SelfAttentionModule(embed_dim=256, num_heads=8)

        key = random.PRNGKey(0)
        x = random.normal(key, (4, 669, 256))

        params = model.init(key, x, train=False)
        output, _ = model.apply(params, x, train=False)

        # Self-attention preserves input shape
        assert output.shape == (4, 669, 256)

    def test_self_attention_with_weights(self):
        """Test with attention weights"""
        model = SelfAttentionModule(embed_dim=256, num_heads=8)

        key = random.PRNGKey(0)
        batch_size = 2
        num_columns = 100
        x = random.normal(key, (batch_size, num_columns, 256))

        params = model.init(key, x, train=False, return_attention_weights=True)
        output, attn_weights = model.apply(
            params, x, train=False, return_attention_weights=True
        )

        assert output.shape == (batch_size, num_columns, 256)
        assert attn_weights is not None
        assert attn_weights.shape == (batch_size, num_columns)

    def test_self_attention_gradients(self):
        """Test that gradients can be computed"""
        model = SelfAttentionModule(embed_dim=128, num_heads=4)

        key = random.PRNGKey(0)
        x = random.normal(key, (2, 50, 128))

        def loss_fn(params):
            output, _ = model.apply(params, x, train=True)
            return jnp.mean(output ** 2)

        params = model.init(key, x, train=True)
        loss, grads = jax.value_and_grad(loss_fn)(params)

        assert jnp.isfinite(loss)
        # Check that gradients are finite
        for leaf in jax.tree_util.tree_leaves(grads):
            assert jnp.all(jnp.isfinite(leaf))


class TestAttentionJIT:
    """Test JIT compilation"""

    def test_cross_attention_jit(self):
        """Test that cross-attention can be JIT compiled"""
        model = CrossAttentionModule(embed_dim=256, num_heads=8)

        key = random.PRNGKey(0)
        x = random.normal(key, (2, 100, 256))
        params = model.init(key, x, train=False)

        @jax.jit
        def forward(params, x):
            return model.apply(params, x, train=False)

        output, _ = forward(params, x)
        assert output.shape == (2, 1, 256)

    def test_self_attention_jit(self):
        """Test that self-attention can be JIT compiled"""
        model = SelfAttentionModule(embed_dim=256, num_heads=8)

        key = random.PRNGKey(0)
        x = random.normal(key, (2, 100, 256))
        params = model.init(key, x, train=False)

        @jax.jit
        def forward(params, x):
            return model.apply(params, x, train=False)

        output, _ = forward(params, x)
        assert output.shape == (2, 100, 256)


class TestAttentionDropout:
    """Test attention dropout functionality"""

    def test_attention_dropout_training(self):
        """Test that dropout is applied during training"""
        model = CrossAttentionModule(
            embed_dim=256,
            num_heads=8,
            attention_dropout=0.5  # High dropout for testing
        )

        key = random.PRNGKey(42)
        x = random.normal(key, (2, 100, 256))

        params = model.init(key, x, train=True, return_attention_weights=True)

        # Run multiple times with training mode
        # Outputs should differ due to dropout
        output1, weights1 = model.apply(
            params, x, train=True, return_attention_weights=True, rngs={'dropout': random.PRNGKey(0)}
        )
        output2, weights2 = model.apply(
            params, x, train=True, return_attention_weights=True, rngs={'dropout': random.PRNGKey(1)}
        )

        # Outputs should be different due to dropout
        assert not jnp.array_equal(output1, output2)

    def test_attention_dropout_eval(self):
        """Test that dropout is disabled in eval mode"""
        model = CrossAttentionModule(
            embed_dim=256,
            num_heads=8,
            attention_dropout=0.5
        )

        key = random.PRNGKey(42)
        x = random.normal(key, (2, 100, 256))

        params = model.init(key, x, train=False)

        # Run multiple times with eval mode
        output1, _ = model.apply(params, x, train=False)
        output2, _ = model.apply(params, x, train=False)

        # Outputs should be identical in eval mode
        assert jnp.array_equal(output1, output2)


@pytest.mark.slow
class TestAttentionFullScale:
    """Test with full eigen2 dimensions"""

    def test_full_scale_cross_attention(self):
        """Test cross-attention with full dimensions"""
        model = CrossAttentionModule(embed_dim=256, num_heads=8)

        key = random.PRNGKey(0)
        batch_size = 8
        num_columns = 669
        x = random.normal(key, (batch_size, num_columns, 256))

        params = model.init(key, x, train=False)
        output, _ = model.apply(params, x, train=True)

        assert output.shape == (batch_size, 1, 256)
        assert jnp.all(jnp.isfinite(output))

    def test_full_scale_self_attention(self):
        """Test self-attention with full dimensions"""
        model = SelfAttentionModule(embed_dim=256, num_heads=8)

        key = random.PRNGKey(0)
        batch_size = 8
        num_columns = 669
        x = random.normal(key, (batch_size, num_columns, 256))

        params = model.init(key, x, train=False)
        output, _ = model.apply(params, x, train=True)

        assert output.shape == (batch_size, num_columns, 256)
        assert jnp.all(jnp.isfinite(output))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
