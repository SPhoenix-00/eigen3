"""Actor network for stock trading (JAX/Flax implementation)

Converts PyTorch Actor to JAX/Flax with:
- FeatureExtractor for temporal features
- Cross-attention for global context
- Dual processing paths (investable stocks + global context)
- Two output heads (coefficient and sale_target)
- Gradient checkpointing support
"""

from typing import Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import checkpoint as remat
import chex

from eigen3.models.feature_extractor import FeatureExtractor
from eigen3.models.attention import CrossAttentionModule


class Actor(nn.Module):
    """Actor network: outputs actions [coefficient, sale_target] for each stock (Eigen2-aligned).

    Uses cross-attention (or self-attention) for feature importance. Eigen2: self-attention,
    investable slice 9:117, coefficient ReLU then clamp [0, 100], sale target clip [10, 50].
    """
    # Architecture parameters (Eigen2: INVESTABLE_START_COL=9, 108 stocks)
    num_investable_stocks: int = 108
    investable_start_col: int = 9
    investable_end_col: int = 116  # Inclusive (slice 9:117 = 108 columns)
    actor_hidden_dims: Tuple[int, int, int] = (256, 128, 64)

    # Sale target range (Eigen2: MIN/MAX_SALE_TARGET)
    min_sale_target: float = 10.0
    max_sale_target: float = 50.0

    # Feature extraction parameters (Eigen2: 117 columns)
    num_columns: int = 117
    num_features: int = 5

    # Attention parameters
    use_attention: bool = True
    attention_heads: int = 8

    # Regularization
    dropout_rate: float = 0.2
    use_remat: bool = True  # Gradient checkpointing

    def setup(self):
        """Initialize submodules"""
        # Feature extraction
        self.feature_extractor = FeatureExtractor(
            num_columns=self.num_columns,
            num_features=self.num_features,
            use_remat=self.use_remat,
        )

        lstm_output_size = self.feature_extractor.lstm_output_size

        # Cross-attention for global context
        if self.use_attention:
            self.attention = CrossAttentionModule(
                embed_dim=lstm_output_size,
                num_heads=self.attention_heads,
                dropout_rate=0.1,
                attention_dropout=0.1,
            )
        else:
            self.attention = None

        # Investable stocks processing
        self.investable_fc1 = nn.Dense(self.actor_hidden_dims[0], name='investable_fc1')
        self.investable_fc2 = nn.Dense(self.actor_hidden_dims[1], name='investable_fc2')

        # Context processing
        self.context_fc = nn.Dense(self.actor_hidden_dims[1], name='context_fc')

        # Combined dimension
        combined_dim = self.actor_hidden_dims[1] * 2  # Investable + context

        # Coefficient head (Eigen2: init bias 0.5 to avoid dead ReLU)
        self.coeff_fc1 = nn.Dense(self.actor_hidden_dims[2], name='coeff_fc1')
        self.coeff_fc2 = nn.Dense(
            1,
            name='coeff_fc2',
            kernel_init=nn.initializers.lecun_normal(),
            bias_init=nn.initializers.constant(0.5),
        )

        # Sale target head (Eigen2: no sigmoid, init bias at center of [10, 50])
        center_sale = (self.min_sale_target + self.max_sale_target) / 2.0
        self.sale_fc1 = nn.Dense(self.actor_hidden_dims[2], name='sale_fc1')
        self.sale_fc2 = nn.Dense(
            1,
            name='sale_fc2',
            kernel_init=nn.initializers.lecun_normal(),
            bias_init=nn.initializers.constant(center_sale),
        )

    def _process_investable(self, investable_features: chex.Array, train: bool) -> chex.Array:
        """Process investable stocks (for gradient checkpointing)

        Args:
            investable_features: [batch, 108, lstm_output_size]
            train: Whether in training mode

        Returns:
            Processed features [batch, 108, actor_hidden_dims[1]]
        """
        x = self.investable_fc1(investable_features)
        x = nn.relu(x)
        x = nn.Dropout(self.dropout_rate, deterministic=not train)(x)
        x = self.investable_fc2(x)
        x = nn.relu(x)
        return x

    def _process_context(self, global_context: chex.Array, train: bool) -> chex.Array:
        """Process context features (for gradient checkpointing)

        Args:
            global_context: [batch, lstm_output_size]
            train: Whether in training mode

        Returns:
            Processed context [batch, actor_hidden_dims[1]]
        """
        x = self.context_fc(global_context)
        x = nn.relu(x)
        return x

    def _process_coefficient_head(self, combined: chex.Array, train: bool) -> chex.Array:
        """Process coefficient head (for gradient checkpointing)

        Args:
            combined: [batch, 108, combined_dim]
            train: Whether in training mode

        Returns:
            Raw coefficients [batch, 108]
        """
        x = self.coeff_fc1(combined)
        x = nn.relu(x)
        x = nn.Dropout(self.dropout_rate, deterministic=not train)(x)
        x = self.coeff_fc2(x)
        x = jnp.squeeze(x, axis=-1)  # [batch, 108]
        return x

    def _process_sale_target_head(self, combined: chex.Array, train: bool) -> chex.Array:
        """Process sale target head (Eigen2: no sigmoid, clip in __call__).

        Args:
            combined: [batch, 108, combined_dim]
            train: Whether in training mode

        Returns:
            Raw sale targets [batch, 108] (clipped to [min, max] in __call__)
        """
        x = self.sale_fc1(combined)
        x = nn.relu(x)
        x = nn.Dropout(self.dropout_rate, deterministic=not train)(x)
        x = self.sale_fc2(x)
        x = jnp.squeeze(x, axis=-1)  # [batch, 108]
        return x

    def _apply_coefficient_activation(self, raw: chex.Array) -> chex.Array:
        """Apply activation: ReLU then clamp [0, 100] (Eigen2: align training with inference)."""
        coefficients = nn.relu(raw)
        coefficients = jnp.clip(coefficients, 0.0, 100.0)
        return coefficients

    @nn.compact
    def __call__(
        self,
        state: chex.Array,
        train: bool = True,
        return_attention_weights: bool = False
    ) -> Tuple[chex.Array, Optional[chex.Array]]:
        """Forward pass with gradient checkpointing

        Args:
            state: Input tensor [batch, context_days, num_columns, num_features]
            train: Whether in training mode (for dropout and BatchNorm)
            return_attention_weights: Whether to return attention weights for logging

        Returns:
            actions: [batch, 108, 2] with [coefficient, sale_target] per stock
            attention_weights: [batch, num_columns] or None
        """
        batch_size = state.shape[0]

        # Extract features from all columns
        features = self.feature_extractor(state, train=train)
        # [batch, num_columns, lstm_output_size]

        # Apply cross-attention if enabled
        attention_weights = None
        if self.attention is not None:
            # Cross-attention: single query attends to all column features
            if train and self.use_remat:
                global_context, attention_weights = remat(
                    lambda x: self.attention(x, train=train, return_attention_weights=True)
                )(features)
            else:
                global_context, attention_weights = self.attention(
                    features, train=train, return_attention_weights=True
                )
            # global_context: [batch, 1, lstm_output_size]
            # attention_weights: [batch, 1, num_columns]

            # Squeeze attention weights: [batch, num_columns]
            if attention_weights is not None:
                attention_weights = jnp.squeeze(attention_weights, axis=1)

            # Process global context through context FC (with checkpointing)
            global_context = jnp.squeeze(global_context, axis=1)  # [batch, lstm_output_size]

            if train and self.use_remat:
                context_processed = remat(lambda x: self._process_context(x, train))(global_context)
            else:
                context_processed = self._process_context(global_context, train)
            # [batch, actor_hidden_dims[1]]

            context_processed = jnp.expand_dims(context_processed, axis=1)  # [batch, 1, actor_hidden_dims[1]]

        else:
            # Fallback: pool all features if no attention
            context_features = jnp.mean(features, axis=1)  # [batch, lstm_output_size]

            if train and self.use_remat:
                context_processed = remat(lambda x: self._process_context(x, train))(context_features)
            else:
                context_processed = self._process_context(context_features, train)

            context_processed = jnp.expand_dims(context_processed, axis=1)  # [batch, 1, actor_hidden_dims[1]]

        # Extract investable stock features (Eigen2: columns 9-116 inclusive = 108 stocks)
        investable_features = features[:, self.investable_start_col:self.investable_end_col+1, :]
        # [batch, 108, lstm_output_size]

        # Process investable stocks (with checkpointing)
        if train and self.use_remat:
            investable_processed = remat(lambda x: self._process_investable(x, train))(investable_features)
        else:
            investable_processed = self._process_investable(investable_features, train)
        # [batch, 108, actor_hidden_dims[1]]

        # Expand context to all stocks
        context_expanded = jnp.tile(context_processed, (1, self.num_investable_stocks, 1))
        # [batch, 108, actor_hidden_dims[1]]

        # Combine
        combined = jnp.concatenate([investable_processed, context_expanded], axis=-1)
        # [batch, 108, combined_dim]

        # Output heads (with checkpointing)
        if train and self.use_remat:
            raw_coefficients = remat(lambda x: self._process_coefficient_head(x, train))(combined)
            raw_sale_targets = remat(lambda x: self._process_sale_target_head(x, train))(combined)
        else:
            raw_coefficients = self._process_coefficient_head(combined, train)
            raw_sale_targets = self._process_sale_target_head(combined, train)

        # Apply activations (Eigen2: coefficient ReLU+clamp [0,100]; sale target clip [10,50])
        coefficients = self._apply_coefficient_activation(raw_coefficients)
        sale_targets = jnp.clip(raw_sale_targets, self.min_sale_target, self.max_sale_target)

        # Stack into action tensor
        actions = jnp.stack([coefficients, sale_targets], axis=-1)
        # [batch, 108, 2]

        if return_attention_weights:
            return actions, attention_weights
        else:
            return actions, None


def test_actor():
    """Test the Actor network"""
    import jax.random as random

    key = random.PRNGKey(0)
    batch_size = 2
    context_days = 151

    # Create input (Eigen2: 117 columns, 151 context days, 5 features)
    state = random.normal(key, (batch_size, context_days, 117, 5))

    # Create Actor
    actor = Actor(use_remat=False)  # Disable remat for testing

    # Initialize
    params = actor.init(key, state, train=False, return_attention_weights=False)

    # Forward pass without attention weights
    actions, attn_weights = actor.apply(
        params, state, train=False, return_attention_weights=False
    )

    print(f"Input shape: {state.shape}")
    print(f"Actions shape: {actions.shape}")
    print(f"Expected shape: ({batch_size}, 108, 2)")
    assert actions.shape == (batch_size, 108, 2)
    assert attn_weights is None

    # Check coefficient range (should be >= 0)
    coefficients = actions[:, :, 0]
    assert jnp.all(coefficients >= 0), f"Negative coefficients found: {jnp.min(coefficients)}"
    print(f"✓ Coefficient range: [{jnp.min(coefficients):.4f}, {jnp.max(coefficients):.4f}]")

    # Check sale target range (should be [10, 50])
    sale_targets = actions[:, :, 1]
    assert jnp.all(sale_targets >= 10.0) and jnp.all(sale_targets <= 50.0), \
        f"Sale targets out of range: [{jnp.min(sale_targets)}, {jnp.max(sale_targets)}]"
    print(f"✓ Sale target range: [{jnp.min(sale_targets):.4f}, {jnp.max(sale_targets):.4f}]")

    # Forward pass with attention weights
    actions, attn_weights = actor.apply(
        params, state, train=False, return_attention_weights=True
    )
    assert attn_weights is not None
    assert attn_weights.shape == (batch_size, 117)
    print(f"✓ Attention weights shape: {attn_weights.shape}")

    # Test with gradient checkpointing
    actor_remat = Actor(use_remat=True)
    params_remat = actor_remat.init(key, state, train=False)
    actions_remat, _ = actor_remat.apply(params_remat, state, train=True)
    print(f"✓ Gradient checkpointing test passed! Actions shape: {actions_remat.shape}")

    print("\n✓ All Actor tests passed!")

    return actor, params, actions


if __name__ == "__main__":
    test_actor()
