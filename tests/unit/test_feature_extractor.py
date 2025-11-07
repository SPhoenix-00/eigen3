"""Unit tests for FeatureExtractor (CNN-LSTM)"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as random
from eigen3.models.feature_extractor import FeatureExtractor, BiLSTM


class TestBiLSTM:
    """Test bidirectional LSTM"""

    def test_bilstm_forward(self):
        """Test forward pass"""
        model = BiLSTM(hidden_size=128, num_layers=2)
        key = random.PRNGKey(0)
        batch_size = 4
        time_steps = 100
        input_size = 32

        x = random.normal(key, (batch_size, time_steps, input_size))
        params = model.init(key, x, train=False)
        output = model.apply(params, x, train=False)

        # Output should be [batch, time_steps, 2*hidden_size]
        assert output.shape == (batch_size, time_steps, 256)

    def test_bilstm_single_layer(self):
        """Test with single layer"""
        model = BiLSTM(hidden_size=64, num_layers=1, dropout_rate=0.0)
        key = random.PRNGKey(42)
        x = random.normal(key, (2, 50, 16))

        params = model.init(key, x, train=False)
        output = model.apply(params, x, train=False)

        assert output.shape == (2, 50, 128)  # 2*64


class TestFeatureExtractor:
    """Test FeatureExtractor"""

    def test_feature_extractor_forward(self):
        """Test forward pass with standard parameters"""
        model = FeatureExtractor(
            num_columns=669,
            num_features=5,
            cnn_filters=32,
            lstm_hidden_size=128,
            lstm_num_layers=2,
            column_chunk_size=64,
            use_remat=False
        )

        key = random.PRNGKey(0)
        batch_size = 2
        context_days = 504

        x = random.normal(key, (batch_size, context_days, 669, 5))
        params = model.init(key, x, train=False)
        output = model.apply(params, x, train=False)

        # Expected output: [batch, 669, 256]
        assert output.shape == (batch_size, 669, 256)

    def test_feature_extractor_with_remat(self):
        """Test forward pass with gradient checkpointing"""
        model = FeatureExtractor(use_remat=True)

        key = random.PRNGKey(0)
        x = random.normal(key, (2, 504, 669, 5))

        params = model.init(key, x, train=False)
        output_train = model.apply(params, x, train=True)
        output_eval = model.apply(params, x, train=False)

        assert output_train.shape == (2, 669, 256)
        assert output_eval.shape == (2, 669, 256)

    def test_feature_extractor_small_batch(self):
        """Test with small batch and context window"""
        model = FeatureExtractor(
            num_columns=100,  # Smaller for testing
            column_chunk_size=32,
            use_remat=False
        )

        key = random.PRNGKey(42)
        x = random.normal(key, (1, 100, 100, 5))  # Small context window

        params = model.init(key, x, train=False)
        output = model.apply(params, x, train=False)

        assert output.shape == (1, 100, 256)

    def test_feature_extractor_gradient(self):
        """Test that gradients can be computed"""
        model = FeatureExtractor(
            num_columns=100,
            column_chunk_size=50,
            use_remat=False
        )

        key = random.PRNGKey(0)
        x = random.normal(key, (2, 50, 100, 5))

        def loss_fn(params):
            output = model.apply(params, x, train=True)
            return jnp.mean(output ** 2)

        params = model.init(key, x, train=True)
        loss, grads = jax.value_and_grad(loss_fn)(params)

        assert jnp.isfinite(loss)
        # Check that all gradients are finite
        for leaf in jax.tree_util.tree_leaves(grads):
            assert jnp.all(jnp.isfinite(leaf))

    def test_feature_extractor_chunking(self):
        """Test that column chunking produces consistent results"""
        key = random.PRNGKey(0)
        x = random.normal(key, (1, 50, 128, 5))

        # Model with large chunk size (no chunking needed)
        model_no_chunk = FeatureExtractor(
            num_columns=128,
            column_chunk_size=128,
            use_remat=False
        )

        # Model with small chunk size (chunking required)
        model_chunk = FeatureExtractor(
            num_columns=128,
            column_chunk_size=32,
            use_remat=False
        )

        # Use same parameters
        params = model_no_chunk.init(key, x, train=False)

        output_no_chunk = model_no_chunk.apply(params, x, train=False)
        output_chunk = model_chunk.apply(params, x, train=False)

        # Outputs should be identical (or very close)
        assert jnp.allclose(output_no_chunk, output_chunk, atol=1e-5)

    def test_feature_extractor_jit(self):
        """Test that the model can be JIT compiled"""
        model = FeatureExtractor(
            num_columns=100,
            column_chunk_size=50,
            use_remat=False
        )

        key = random.PRNGKey(0)
        x = random.normal(key, (2, 50, 100, 5))
        params = model.init(key, x, train=False)

        @jax.jit
        def forward(params, x):
            return model.apply(params, x, train=False)

        output = forward(params, x)
        assert output.shape == (2, 100, 256)

    def test_feature_extractor_nan_handling(self):
        """Test that NaN values are properly handled"""
        model = FeatureExtractor(
            num_columns=50,
            column_chunk_size=50,
            use_remat=False
        )

        key = random.PRNGKey(0)
        x = random.normal(key, (1, 50, 50, 5))

        # Insert some NaN values
        x = x.at[0, 10:20, :, :].set(jnp.nan)

        params = model.init(key, x, train=False)
        output = model.apply(params, x, train=False)

        # Output should not contain NaN
        assert jnp.all(jnp.isfinite(output))

    def test_feature_extractor_deterministic(self):
        """Test that eval mode is deterministic"""
        model = FeatureExtractor(use_remat=False)

        key = random.PRNGKey(0)
        x = random.normal(key, (2, 100, 669, 5))

        params = model.init(key, x, train=False)

        # Run multiple times with same input
        output1 = model.apply(params, x, train=False)
        output2 = model.apply(params, x, train=False)

        # Outputs should be identical
        assert jnp.array_equal(output1, output2)

    @pytest.mark.slow
    def test_feature_extractor_full_size(self):
        """Test with full eigen2 dimensions"""
        model = FeatureExtractor(
            num_columns=669,
            num_features=5,
            cnn_filters=32,
            lstm_hidden_size=128,
            lstm_num_layers=2,
            column_chunk_size=64,
            use_remat=True
        )

        key = random.PRNGKey(0)
        batch_size = 8  # Typical batch size
        context_days = 504  # Full context window

        x = random.normal(key, (batch_size, context_days, 669, 5))

        params = model.init(key, x, train=False)
        output = model.apply(params, x, train=True)

        assert output.shape == (batch_size, 669, 256)
        assert jnp.all(jnp.isfinite(output))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
