"""Actor network for stock trading (JAX/Flax implementation)

Converts PyTorch Actor to JAX/Flax with:
- FeatureExtractor for temporal features
- Cross-attention for global context
- Dual processing paths (investable stocks + global context)
- Three output heads (coefficient, sale_target, close_fraction)
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


def split_market_portfolio(
    state: chex.Array,
    portfolio_dim: int,
) -> Tuple[chex.Array, Optional[chex.Array]]:
    """Separate market observations from the broadcast portfolio tail.

    The portfolio vector is identical at every ``(time, column)`` cell, so we
    read it once from ``[:, 0, 0, ...]``.

    Returns:
        ``(state_market, port_raw)`` where *port_raw* is ``None`` when
        ``portfolio_dim == 0``.
    """
    if portfolio_dim > 0:
        state_market = state[..., :-portfolio_dim]
        port_raw = state[:, 0, 0, -portfolio_dim:]
        return state_market, port_raw
    return state, None


def combine_market_portfolio(
    market_obs: chex.Array,
    portfolio_obs: chex.Array,
) -> chex.Array:
    """Broadcast compact portfolio vector into market obs for model input.

    Inverse of :func:`split_market_portfolio`.  Only used transiently before
    a forward pass — never stored in the replay buffer.

    Args:
        market_obs: ``[..., T, C, F]`` market-only observations.
        portfolio_obs: ``[..., P]`` compact portfolio vector (``P == 0``
            means portfolio is disabled; returns *market_obs* unchanged).
    """
    p = portfolio_obs.shape[-1]
    if p == 0:
        return market_obs
    # Broadcast (P,) to (T, C, P) — works for both batched and unbatched obs.
    extra_dims = market_obs.shape[:-1]  # (..., T, C)
    port_broadcast = jnp.broadcast_to(
        portfolio_obs[..., None, None, :],
        (*extra_dims, p),
    )
    return jnp.concatenate([market_obs, port_broadcast], axis=-1)


class Actor(nn.Module):
    """Actor network: outputs [coefficient, sale_target, close_fraction] per stock.

    Uses cross-attention (or self-attention) for feature importance. Eigen2: self-attention,
    investable slice 9:117, coefficient ReLU then clamp [0, 100], sale target clip [10, 50],
    close_fraction sigmoid in [0, 1].
    """
    # Architecture parameters (Eigen2: INVESTABLE_START_COL=9, 108 stocks)
    num_investable_stocks: int = 108
    investable_start_col: int = 9
    actor_hidden_dims: Tuple[int, int, int] = (256, 128, 64)

    # Sale target range (Eigen2: MIN/MAX_SALE_TARGET)
    min_sale_target: float = 10.0
    max_sale_target: float = 50.0

    # Feature extraction parameters (Eigen2: 117 columns)
    num_columns: int = 117
    num_features: int = 5
    # Suffix on state[..., -portfolio_dim:] (broadcast); concatenated into context MLP input.
    portfolio_dim: int = 0
    column_chunk_size: int = 64

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
            column_chunk_size=self.column_chunk_size,
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

        if self.portfolio_dim > 0:
            self.portfolio_ln = nn.LayerNorm(name='portfolio_ln')

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

        self.close_fc1 = nn.Dense(self.actor_hidden_dims[2], name='close_fc1')
        self.close_fc2 = nn.Dense(
            1,
            name='close_fc2',
            kernel_init=nn.initializers.lecun_normal(),
            bias_init=nn.initializers.constant(-2.0),
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

    def _process_close_head(self, combined: chex.Array, train: bool) -> chex.Array:
        """Raw logits for discretionary close fraction (sigmoid applied in __call__)."""
        x = self.close_fc1(combined)
        x = nn.relu(x)
        x = nn.Dropout(self.dropout_rate, deterministic=not train)(x)
        x = self.close_fc2(x)
        x = jnp.squeeze(x, axis=-1)
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
            state: Input tensor [batch, context_days, num_columns, num_features + portfolio_dim]
                (market stats match ``num_features``; optional portfolio tail).
            train: Whether in training mode (for dropout and BatchNorm)
            return_attention_weights: Whether to return attention weights for logging

        Returns:
            actions: [batch, num_investable_stocks, 3] with [coefficient, sale_target, close_fraction]
            attention_weights: [batch, num_columns] or None
        """
        batch_size = state.shape[0]

        state_mkt, port_raw = split_market_portfolio(state, self.portfolio_dim)
        if port_raw is not None:
            port_raw = self.portfolio_ln(port_raw)

        # Extract features from all columns (market slice only)
        features = self.feature_extractor(state_mkt, train=train)
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
            if port_raw is not None:
                global_context = jnp.concatenate([global_context, port_raw], axis=-1)

            if train and self.use_remat:
                context_processed = remat(lambda x: self._process_context(x, train))(global_context)
            else:
                context_processed = self._process_context(global_context, train)
            # [batch, actor_hidden_dims[1]]

            context_processed = jnp.expand_dims(context_processed, axis=1)  # [batch, 1, actor_hidden_dims[1]]

        else:
            # Fallback: pool all features if no attention
            context_features = jnp.mean(features, axis=1)  # [batch, lstm_output_size]
            if port_raw is not None:
                context_features = jnp.concatenate([context_features, port_raw], axis=-1)

            if train and self.use_remat:
                context_processed = remat(lambda x: self._process_context(x, train))(context_features)
            else:
                context_processed = self._process_context(context_features, train)

            context_processed = jnp.expand_dims(context_processed, axis=1)  # [batch, 1, actor_hidden_dims[1]]

        # Investable slice: contiguous columns [start, start + num_investable - 1]
        inv_end = self.investable_start_col + self.num_investable_stocks - 1
        investable_features = features[:, self.investable_start_col : inv_end + 1, :]
        # [batch, num_investable_stocks, lstm_output_size]

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
            raw_close = remat(lambda x: self._process_close_head(x, train))(combined)
        else:
            raw_coefficients = self._process_coefficient_head(combined, train)
            raw_sale_targets = self._process_sale_target_head(combined, train)
            raw_close = self._process_close_head(combined, train)

        # Apply activations (Eigen2: coefficient ReLU+clamp [0,100]; sale target clip [10,50])
        coefficients = self._apply_coefficient_activation(raw_coefficients)
        sale_targets = jnp.clip(raw_sale_targets, self.min_sale_target, self.max_sale_target)
        close_fractions = jax.nn.sigmoid(raw_close)

        # Stack into action tensor
        actions = jnp.stack([coefficients, sale_targets, close_fractions], axis=-1)

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
    print(f"Expected shape: ({batch_size}, 108, 3)")
    assert actions.shape == (batch_size, 108, 3)
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

    close_fr = actions[:, :, 2]
    assert jnp.all(close_fr >= 0.0) and jnp.all(close_fr <= 1.0)
    print(f"✓ Close fraction range: [{jnp.min(close_fr):.4f}, {jnp.max(close_fr):.4f}]")

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

    # Test with portfolio_dim > 0
    pdim = 32
    state_pf = random.normal(key, (batch_size, context_days, 117, 5 + pdim))
    actor_pf = Actor(portfolio_dim=pdim, use_remat=False)
    params_pf = actor_pf.init(key, state_pf, train=False, return_attention_weights=False)
    actions_pf, _ = actor_pf.apply(params_pf, state_pf, train=False, return_attention_weights=False)
    assert actions_pf.shape == (batch_size, 108, 3)
    print(f"✓ Portfolio dim={pdim} test passed! Actions shape: {actions_pf.shape}")

    print("\n✓ All Actor tests passed!")

    return actor, params, actions


if __name__ == "__main__":
    test_actor()
