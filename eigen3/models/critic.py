"""Critic network for Q-value estimation (JAX/Flax implementation)

Converts PyTorch Critic to JAX/Flax with:
- FeatureExtractor for temporal features
- Optional self-attention
- Q-value estimation from state-action pairs
- Twin critics for TD3-style training
- Gradient checkpointing support
"""

from typing import Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import checkpoint as remat
import chex

from eigen3.models.feature_extractor import FeatureExtractor
from eigen3.models.attention import SelfAttentionModule


class Critic(nn.Module):
    """Critic network: estimates Q-value for state-action pair

    Synced with Eigen2 (mean pool, 117 cols). Original: eigen2/models/networks.py:453-532
    """
    # Architecture parameters
    num_investable_stocks: int = 108
    action_dim: int = 2
    critic_hidden_dims: Tuple[int, int] = (256, 128)

    # Feature extraction parameters
    num_columns: int = 669
    num_features: int = 5

    # Attention parameters
    use_attention: bool = False  # Self-attention (not commonly used)
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

        # Optional self-attention
        if self.use_attention:
            self.attention = SelfAttentionModule(
                embed_dim=lstm_output_size,
                num_heads=self.attention_heads,
                dropout_rate=0.1,
            )
        else:
            self.attention = None

        # Action dimension (flattened)
        self.action_flat_dim = self.num_investable_stocks * self.action_dim  # 108 * 2 = 216

        # Critic FC layers
        # Input: state_features (256) + action_flat (216) = 472
        self.fc1 = nn.Dense(self.critic_hidden_dims[0], name='critic_fc1')
        self.fc2 = nn.Dense(self.critic_hidden_dims[1], name='critic_fc2')
        self.fc3 = nn.Dense(1, name='critic_fc3')

    def _process_critic_fc(self, x: chex.Array, train: bool) -> chex.Array:
        """Process critic FC layers (for gradient checkpointing)

        Args:
            x: Combined state-action features [batch, state_dim + action_dim]
            train: Whether in training mode

        Returns:
            Q-values [batch, 1]
        """
        x = self.fc1(x)
        x = nn.relu(x)
        x = nn.Dropout(self.dropout_rate, deterministic=not train)(x)

        x = self.fc2(x)
        x = nn.relu(x)
        x = nn.Dropout(self.dropout_rate, deterministic=not train)(x)

        x = self.fc3(x)
        return x

    @nn.compact
    def __call__(
        self,
        state: chex.Array,
        action: chex.Array,
        train: bool = True
    ) -> chex.Array:
        """Forward pass with gradient checkpointing

        Args:
            state: Input tensor [batch, context_days, num_columns, num_features]
            action: Action tensor [batch, 108, 2]
            train: Whether in training mode (for dropout and BatchNorm)

        Returns:
            Q-values: [batch, 1]
        """
        # Extract features
        features = self.feature_extractor(state, train=train)
        # [batch, num_columns, lstm_output_size]

        # Apply attention if enabled
        if self.attention is not None:
            features, _ = self.attention(features, train=train, return_attention_weights=False)
            # [batch, num_columns, lstm_output_size]

        # Pool features across all columns
        state_features = jnp.mean(features, axis=1)  # [batch, lstm_output_size]

        # Flatten action
        action_flat = action.reshape(action.shape[0], -1)  # [batch, 216]

        # Concatenate state and action
        x = jnp.concatenate([state_features, action_flat], axis=-1)
        # [batch, lstm_output_size + 216]

        # Critic network (with checkpointing)
        if train and self.use_remat:
            q_value = remat(lambda x: self._process_critic_fc(x, train))(x)
        else:
            q_value = self._process_critic_fc(x, train)

        return q_value


class DoubleCritic(nn.Module):
    """Twin Q-networks for TD3-style training (Eigen2-aligned)."""
    # Architecture parameters
    num_investable_stocks: int = 108
    action_dim: int = 2
    critic_hidden_dims: Tuple[int, int] = (256, 128)

    # Feature extraction parameters (Eigen2: 117 columns)
    num_columns: int = 117
    num_features: int = 5

    # Attention parameters
    use_attention: bool = False
    attention_heads: int = 8

    # Regularization
    dropout_rate: float = 0.2
    use_remat: bool = True

    @nn.compact
    def __call__(
        self,
        state: chex.Array,
        action: chex.Array,
        train: bool = True
    ) -> chex.Array:
        """Forward pass through both critics

        Args:
            state: Input tensor [batch, context_days, num_columns, num_features]
            action: Action tensor [batch, 108, 2]
            train: Whether in training mode

        Returns:
            Q-values: [batch, 2] with Q-values from both critics
        """
        # Critic 1
        q1 = Critic(
            num_investable_stocks=self.num_investable_stocks,
            action_dim=self.action_dim,
            critic_hidden_dims=self.critic_hidden_dims,
            num_columns=self.num_columns,
            num_features=self.num_features,
            use_attention=self.use_attention,
            attention_heads=self.attention_heads,
            dropout_rate=self.dropout_rate,
            use_remat=self.use_remat,
            name='critic_1'
        )(state, action, train=train)

        # Critic 2
        q2 = Critic(
            num_investable_stocks=self.num_investable_stocks,
            action_dim=self.action_dim,
            critic_hidden_dims=self.critic_hidden_dims,
            num_columns=self.num_columns,
            num_features=self.num_features,
            use_attention=self.use_attention,
            attention_heads=self.attention_heads,
            dropout_rate=self.dropout_rate,
            use_remat=self.use_remat,
            name='critic_2'
        )(state, action, train=train)

        # Concatenate Q-values from both critics
        q_values = jnp.concatenate([q1, q2], axis=-1)  # [batch, 2]

        return q_values


def test_critic():
    """Test the Critic network"""
    import jax.random as random

    key = random.PRNGKey(0)
    batch_size = 2
    context_days = 504

    # Create inputs
    state = random.normal(key, (batch_size, context_days, 669, 5))
    action = random.normal(key, (batch_size, 108, 2))

    # Test single Critic
    print("Testing Critic...")
    critic = Critic(use_attention=False, use_remat=False)
    params = critic.init(key, state, action, train=False)
    q_values = critic.apply(params, state, action, train=False)

    print(f"State shape: {state.shape}")
    print(f"Action shape: {action.shape}")
    print(f"Q-values shape: {q_values.shape}")
    print(f"Expected shape: ({batch_size}, 1)")
    assert q_values.shape == (batch_size, 1)
    print(f"✓ Single Critic test passed! Q-values: {q_values.ravel()}")

    # Test with attention
    print("\nTesting Critic with attention...")
    critic_attn = Critic(use_attention=True, use_remat=False)
    params_attn = critic_attn.init(key, state, action, train=False)
    q_values_attn = critic_attn.apply(params_attn, state, action, train=False)
    assert q_values_attn.shape == (batch_size, 1)
    print(f"✓ Critic with attention test passed!")

    # Test DoubleCritic
    print("\nTesting DoubleCritic...")
    double_critic = DoubleCritic(use_attention=False, use_remat=False)
    params_double = double_critic.init(key, state, action, train=False)
    q_values_double = double_critic.apply(params_double, state, action, train=False)

    print(f"DoubleCritic Q-values shape: {q_values_double.shape}")
    print(f"Expected shape: ({batch_size}, 2)")
    assert q_values_double.shape == (batch_size, 2)
    print(f"✓ DoubleCritic test passed! Q-values: {q_values_double}")

    # Test with gradient checkpointing
    print("\nTesting with gradient checkpointing...")
    critic_remat = Critic(use_remat=True)
    params_remat = critic_remat.init(key, state, action, train=False)
    q_values_remat = critic_remat.apply(params_remat, state, action, train=True)
    print(f"✓ Gradient checkpointing test passed! Q-values shape: {q_values_remat.shape}")

    print("\n✓ All Critic tests passed!")

    return critic, double_critic


if __name__ == "__main__":
    test_critic()
