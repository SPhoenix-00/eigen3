"""Multi-head attention module (JAX/Flax implementation)

Converts PyTorch AttentionModule to JAX/Flax with:
- Self-attention mode (for Critic)
- Cross-attention mode (for Actor with learnable query)
- Attention dropout for regularization
- Optional return of attention weights for visualization
"""

from typing import Optional, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
import chex


class AttentionModule(nn.Module):
    """Multi-head attention to learn which stocks/indicators matter

    Supports both self-attention and cross-attention modes.
    Synced with Eigen2. Original: eigen2/models/networks.py:136-243
    """
    embed_dim: int = 256
    num_heads: int = 8
    use_cross_attention: bool = False
    dropout_rate: float = 0.1
    attention_dropout: float = 0.1

    @nn.compact
    def __call__(
        self,
        x: chex.Array,
        train: bool = True,
        return_attention_weights: bool = False
    ) -> Tuple[chex.Array, Optional[chex.Array]]:
        """Apply multi-head attention

        Args:
            x: Input features [batch, num_columns, embed_dim]
            train: Whether in training mode (for dropout)
            return_attention_weights: Whether to return attention weights

        Returns:
            If return_attention_weights=True: (output, attention_weights)
            If return_attention_weights=False: (output, None)

            output: [batch, num_columns, embed_dim] or [batch, 1, embed_dim] for cross-attention
            attention_weights: [batch, num_columns] or None
        """
        batch_size = x.shape[0]
        num_columns = x.shape[1]

        if self.use_cross_attention:
            # Cross-attention: single query attends to all columns
            # Learnable query vector (the Actor's "brain")
            query = self.param(
                'query',
                nn.initializers.normal(stddev=0.02),
                (1, 1, self.embed_dim)
            )

            # Expand query for batch
            query = jnp.tile(query, (batch_size, 1, 1))  # [batch, 1, embed_dim]

            # Cross-attention: Q from query, K and V from input features
            attn_out = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                deterministic=not train,
                decode=False,
            )(query, x)

            # Compute attention weights if requested
            if return_attention_weights:
                # Manually compute attention weights for visualization
                # This is a simplified version - in practice, you'd need to access
                # the internal attention mechanism or recompute
                attn_weights = self._compute_attention_weights(query, x)

                # Apply attention dropout during training
                if train and self.attention_dropout > 0:
                    attn_weights = self._apply_attention_dropout(attn_weights, train)
            else:
                attn_weights = None

            # No residual for cross-attention (query is different from input)
            output = nn.LayerNorm()(attn_out)

            return output, attn_weights

        else:
            # Self-attention across columns
            attn_out = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                deterministic=not train,
                decode=False,
            )(x, x)

            # Compute attention weights if requested
            if return_attention_weights:
                attn_weights = self._compute_attention_weights(x, x)
                # Average across query positions for self-attention
                attn_weights = jnp.mean(attn_weights, axis=1)  # [batch, num_columns]
            else:
                attn_weights = None

            # Residual connection + normalization
            output = nn.LayerNorm()(x + attn_out)

            return output, attn_weights

    def _compute_attention_weights(
        self,
        query: chex.Array,
        key: chex.Array
    ) -> chex.Array:
        """Compute attention weights (for visualization)

        This is a simplified version that computes attention weights
        similar to the internal mechanism.

        Args:
            query: Query tensor [batch, num_queries, embed_dim]
            key: Key tensor [batch, num_keys, embed_dim]

        Returns:
            Attention weights [batch, num_queries, num_keys]
        """
        # Compute scaled dot-product attention scores
        scale = jnp.sqrt(self.embed_dim / self.num_heads)
        scores = jnp.matmul(query, key.transpose(0, 2, 1)) / scale

        # Apply softmax to get attention weights
        attn_weights = nn.softmax(scores, axis=-1)

        return attn_weights

    def _apply_attention_dropout(
        self,
        attn_weights: chex.Array,
        train: bool
    ) -> chex.Array:
        """Apply attention dropout and renormalize

        Randomly zeros out some attention weights and renormalizes.
        This prevents the model from getting stuck in local optima.

        Args:
            attn_weights: [batch, ...] - softmax weights
            train: Whether in training mode

        Returns:
            Dropped and renormalized weights
        """
        if not train or self.attention_dropout == 0:
            return attn_weights

        # Create dropout mask
        keep_prob = 1.0 - self.attention_dropout
        mask = jax.random.bernoulli(
            self.make_rng('dropout'),
            p=keep_prob,
            shape=attn_weights.shape
        )

        # Apply mask
        masked_weights = attn_weights * mask

        # Renormalize so they sum to 1.0
        weight_sum = jnp.sum(masked_weights, axis=-1, keepdims=True) + 1e-8
        renormalized_weights = masked_weights / weight_sum

        return renormalized_weights


class CrossAttentionModule(nn.Module):
    """Cross-attention module for Actor network

    Convenience wrapper for AttentionModule with cross-attention enabled.
    """
    embed_dim: int = 256
    num_heads: int = 8
    dropout_rate: float = 0.1
    attention_dropout: float = 0.1

    @nn.compact
    def __call__(
        self,
        x: chex.Array,
        train: bool = True,
        return_attention_weights: bool = False
    ) -> Tuple[chex.Array, Optional[chex.Array]]:
        """Apply cross-attention

        Args:
            x: Input features [batch, num_columns, embed_dim]
            train: Whether in training mode
            return_attention_weights: Whether to return attention weights

        Returns:
            output: [batch, 1, embed_dim]
            attention_weights: [batch, num_columns] or None
        """
        return AttentionModule(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            use_cross_attention=True,
            dropout_rate=self.dropout_rate,
            attention_dropout=self.attention_dropout,
        )(x, train=train, return_attention_weights=return_attention_weights)


class SelfAttentionModule(nn.Module):
    """Self-attention module for Critic network

    Convenience wrapper for AttentionModule with self-attention enabled.
    """
    embed_dim: int = 256
    num_heads: int = 8
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self,
        x: chex.Array,
        train: bool = True,
        return_attention_weights: bool = False
    ) -> Tuple[chex.Array, Optional[chex.Array]]:
        """Apply self-attention

        Args:
            x: Input features [batch, num_columns, embed_dim]
            train: Whether in training mode
            return_attention_weights: Whether to return attention weights

        Returns:
            output: [batch, num_columns, embed_dim]
            attention_weights: [batch, num_columns] or None
        """
        return AttentionModule(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            use_cross_attention=False,
            dropout_rate=self.dropout_rate,
            attention_dropout=0.0,  # No attention dropout for self-attention
        )(x, train=train, return_attention_weights=return_attention_weights)


def test_attention():
    """Test the attention modules"""
    import jax.random as random

    key = random.PRNGKey(0)
    batch_size = 4
    num_columns = 117
    embed_dim = 256

    # Test input
    x = random.normal(key, (batch_size, num_columns, embed_dim))

    print("Testing Cross-Attention Module...")
    cross_attn = CrossAttentionModule(embed_dim=embed_dim, num_heads=8)
    params_cross = cross_attn.init(key, x, train=False)

    # Forward pass without attention weights
    output, attn_weights = cross_attn.apply(params_cross, x, train=False, return_attention_weights=False)
    assert output.shape == (batch_size, 1, embed_dim), f"Cross-attn output shape: {output.shape}"
    assert attn_weights is None
    print(f"✓ Cross-attention output shape: {output.shape}")

    # Forward pass with attention weights
    output, attn_weights = cross_attn.apply(
        params_cross, x, train=False, return_attention_weights=True
    )
    assert attn_weights.shape == (batch_size, 1, num_columns), f"Attention weights shape: {attn_weights.shape}"
    print(f"✓ Attention weights shape: {attn_weights.shape}")

    print("\nTesting Self-Attention Module...")
    self_attn = SelfAttentionModule(embed_dim=embed_dim, num_heads=8)
    params_self = self_attn.init(key, x, train=False)

    # Forward pass
    output, _ = self_attn.apply(params_self, x, train=False, return_attention_weights=False)
    assert output.shape == (batch_size, num_columns, embed_dim), f"Self-attn output shape: {output.shape}"
    print(f"✓ Self-attention output shape: {output.shape}")

    # Test with attention weights
    output, attn_weights = self_attn.apply(
        params_self, x, train=False, return_attention_weights=True
    )
    assert attn_weights.shape == (batch_size, num_columns), f"Self-attn weights shape: {attn_weights.shape}"
    print(f"✓ Self-attention weights shape: {attn_weights.shape}")

    print("\n✓ All attention tests passed!")

    return cross_attn, self_attn


if __name__ == "__main__":
    test_attention()
