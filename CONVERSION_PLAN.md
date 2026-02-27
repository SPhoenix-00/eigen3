# Eigen2 → Eigen3: JAX Conversion & EvoRL Migration Plan

## Executive Summary

This document outlines the complete strategy for converting Eigen2 (PyTorch-based ERL stock trading system) to Eigen3 (JAX-based system using the EvoRL framework).

**Source Project:** `C:\Users\antoi\Documents\GitHub\eigen2`
**Target Project:** `C:\Users\antoi\Documents\GitHub\eigen3`
**EvoRL Framework:** `C:\Users\antoi\Documents\GitHub\eigen3\evorl`

**Sync status (2025-02-27):** Eigen3 has been brought up to speed with Eigen2 changes. See [EIGEN2_DELTA.md](EIGEN2_DELTA.md) for the delta (117-column skinny dataset, 151-day context, instance norm, holding periods, hurdle/conviction rewards, coefficient clamp 100, etc.) and which Eigen3 files were updated.

---

## Phase 1: The Great JAX Conversion

### Overview

Convert all PyTorch components to JAX-native implementations that integrate with the EvoRL framework.

---

## 1. Neural Network Architecture Conversion

### 1.1 FeatureExtractor (CNN-LSTM)

**Source:** `eigen2/models/networks.py:16-134`

**Target:** `eigen3/models/feature_extractor.py`

**Key Challenges:**
- PyTorch LSTM → Flax RNN cells
- Gradient checkpointing → JAX remat
- Column chunking strategy (memory management)
- Bidirectional LSTM with 2 layers

**Conversion Strategy:**

```python
# Current PyTorch structure:
# Input: [batch, num_columns, context_days, num_features]
# CNN: Conv1d(5, 32, kernel_size=3) + BatchNorm1d + ReLU
# LSTM: Bidirectional, hidden=128, layers=2
# Output: [batch, 669, 256]

# JAX/Flax implementation:
import flax.linen as nn
import jax.numpy as jnp
from flax.linen import scan

class FeatureExtractor(nn.Module):
    """JAX-native feature extractor with CNN and Bidirectional LSTM"""
    num_columns: int = 669
    cnn_filters: int = 32
    lstm_hidden_size: int = 128
    num_lstm_layers: int = 2
    chunk_size: int = 64  # Process columns in chunks

    @nn.compact
    def __call__(self, x, train: bool = True):
        # x: [batch, num_columns, context_days, num_features]
        batch_size = x.shape[0]

        # Process in chunks to manage memory
        chunk_outputs = []
        for i in range(0, self.num_columns, self.chunk_size):
            chunk = x[:, i:i+self.chunk_size]  # [batch, chunk_size, T, F]

            # CNN block
            chunk = jnp.transpose(chunk, (0, 1, 3, 2))  # [B, C, F, T]
            chunk = nn.Conv(
                features=self.cnn_filters,
                kernel_size=(3,),
                padding='SAME',
                name=f'cnn_chunk_{i}'
            )(chunk)
            chunk = nn.BatchNorm(use_running_average=not train)(chunk)
            chunk = nn.relu(chunk)

            # Prepare for LSTM: [batch, chunk_size, T, features]
            chunk = jnp.transpose(chunk, (0, 1, 3, 2))

            # Bidirectional LSTM
            chunk = BiLSTM(
                hidden_size=self.lstm_hidden_size,
                num_layers=self.num_lstm_layers
            )(chunk)

            chunk_outputs.append(chunk)

        # Concatenate all chunks: [batch, num_columns, 256]
        output = jnp.concatenate(chunk_outputs, axis=1)
        return output

class BiLSTM(nn.Module):
    """Bidirectional LSTM using Flax"""
    hidden_size: int
    num_layers: int

    @nn.compact
    def __call__(self, x):
        # x: [batch, num_columns, time_steps, features]
        # Process each column independently

        def process_column(carry, col_data):
            # col_data: [batch, time_steps, features]
            # Forward LSTM
            lstm_fwd = nn.scan(
                nn.LSTMCell,
                variable_broadcast="params",
                split_rngs={"params": False}
            )(self.hidden_size)

            # Backward LSTM
            lstm_bwd = nn.scan(
                nn.LSTMCell,
                variable_broadcast="params",
                split_rngs={"params": False},
                reverse=True
            )(self.hidden_size)

            # Apply
            carry_fwd, hidden_fwd = lstm_fwd(carry, col_data)
            carry_bwd, hidden_bwd = lstm_bwd(carry, col_data[:, ::-1])

            # Concatenate: [batch, time_steps, 256]
            hidden = jnp.concatenate([hidden_fwd, hidden_bwd], axis=-1)

            # Average last 3 timesteps
            output = jnp.mean(hidden[:, -3:], axis=1)  # [batch, 256]
            return carry, output

        # Use vmap over columns
        outputs = jax.vmap(process_column, in_axes=1, out_axes=1)(None, x)
        return outputs
```

**Gradient Checkpointing:**
Replace `torch.utils.checkpoint.checkpoint` with `jax.checkpoint` (remat):

```python
from jax import checkpoint as remat

class FeatureExtractor(nn.Module):
    use_remat: bool = True

    @nn.compact
    def __call__(self, x, train: bool = True):
        if self.use_remat:
            return remat(self._forward)(x, train)
        return self._forward(x, train)

    def _forward(self, x, train):
        # Actual computation here
        ...
```

---

### 1.2 AttentionModule

**Source:** `eigen2/models/networks.py:136-243`

**Target:** `eigen3/models/attention.py`

**Conversion Strategy:**

```python
class AttentionModule(nn.Module):
    """Cross-attention or self-attention using Flax"""
    embed_dim: int = 256
    num_heads: int = 8
    dropout_rate: float = 0.1
    mode: str = "cross"  # "cross" or "self"

    @nn.compact
    def __call__(self, features, train: bool = True):
        # features: [batch, 669, 256]

        if self.mode == "cross":
            # Learnable query vector for actor
            query = self.param(
                'query',
                nn.initializers.normal(0.02),
                (1, 1, self.embed_dim)
            )
            query = jnp.tile(query, (features.shape[0], 1, 1))  # [batch, 1, 256]

            attn_output = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                deterministic=not train
            )(query, features)

        else:
            # Self-attention for critic
            attn_output = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                deterministic=not train
            )(features, features)

        # Layer normalization
        output = nn.LayerNorm()(attn_output + features)  # Residual

        return output
```

---

### 1.3 Actor Network

**Source:** `eigen2/models/networks.py:245-451`

**Target:** `eigen3/models/actor.py`

**Architecture:**
1. FeatureExtractor → `[batch, 669, 256]`
2. Cross-Attention → `[batch, 1, 256]` (global context)
3. Separate paths for investable stocks (108) and global context
4. Two heads: coefficient and sale_target

**Conversion Strategy:**

```python
class Actor(nn.Module):
    """Actor network for stock trading"""
    num_stocks: int = 108  # Columns 8-115
    stock_start_idx: int = 8
    use_remat: bool = True

    @nn.compact
    def __call__(self, obs, train: bool = True):
        # obs: [batch, 504, 669, 5]

        # Feature extraction
        features = FeatureExtractor(use_remat=self.use_remat)(obs, train)
        # features: [batch, 669, 256]

        # Cross-attention for global context
        global_context = AttentionModule(mode="cross")(features, train)
        # global_context: [batch, 1, 256]

        # Extract investable stock features
        stock_features = features[:, self.stock_start_idx:self.stock_start_idx+self.num_stocks]
        # stock_features: [batch, 108, 256]

        # Process investable stocks
        stock_hidden = nn.Dense(256)(stock_features)
        stock_hidden = nn.relu(stock_hidden)
        stock_hidden = nn.Dropout(0.2, deterministic=not train)(stock_hidden)
        stock_hidden = nn.Dense(128)(stock_hidden)
        stock_hidden = nn.relu(stock_hidden)
        # stock_hidden: [batch, 108, 128]

        # Process global context
        global_hidden = nn.Dense(128)(global_context)
        global_hidden = nn.relu(global_hidden)
        global_hidden = jnp.tile(global_hidden, (1, self.num_stocks, 1))
        # global_hidden: [batch, 108, 128]

        # Concatenate
        combined = jnp.concatenate([stock_hidden, global_hidden], axis=-1)
        # combined: [batch, 108, 256]

        # Coefficient head
        coeff = nn.Dense(64)(combined)
        coeff = nn.relu(coeff)
        coeff = nn.Dense(1)(coeff)
        coeff = custom_activation(coeff)  # Custom exponential activation

        # Sale target head
        sale_target = nn.Dense(64)(combined)
        sale_target = nn.relu(sale_target)
        sale_target = nn.Dense(1)(sale_target)
        sale_target = nn.sigmoid(sale_target)
        sale_target = 10.0 + sale_target * 40.0  # Scale to [10, 50]

        # Concatenate outputs: [batch, 108, 2]
        actions = jnp.concatenate([coeff, sale_target], axis=-1)

        return actions

def custom_activation(x):
    """Custom activation for coefficient: exp(tanh(x)) - 1"""
    return jnp.exp(jnp.tanh(x)) - 1.0
```

---

### 1.4 Critic Network

**Source:** `eigen2/models/networks.py:453-533`

**Target:** `eigen3/models/critic.py`

**Conversion Strategy:**

```python
class Critic(nn.Module):
    """Critic (Q-function) network"""
    use_attention: bool = False
    use_remat: bool = True

    @nn.compact
    def __call__(self, obs, action, train: bool = True):
        # obs: [batch, 504, 669, 5]
        # action: [batch, 108, 2]

        # Feature extraction
        features = FeatureExtractor(use_remat=self.use_remat)(obs, train)
        # features: [batch, 669, 256]

        # Optional self-attention
        if self.use_attention:
            features = AttentionModule(mode="self")(features, train)

        # Pool across columns
        pooled = jnp.mean(features, axis=1)  # [batch, 256]

        # Flatten action
        action_flat = action.reshape(action.shape[0], -1)  # [batch, 216]

        # Concatenate
        combined = jnp.concatenate([pooled, action_flat], axis=-1)
        # combined: [batch, 472]

        # MLP
        hidden = nn.Dense(256)(combined)
        hidden = nn.relu(hidden)
        hidden = nn.Dropout(0.2, deterministic=not train)(hidden)

        hidden = nn.Dense(128)(hidden)
        hidden = nn.relu(hidden)
        hidden = nn.Dropout(0.2, deterministic=not train)(hidden)

        q_value = nn.Dense(1)(hidden)

        return q_value  # [batch, 1]

class DoubleCritic(nn.Module):
    """Twin Q-networks for TD3-style training"""

    @nn.compact
    def __call__(self, obs, action, train: bool = True):
        q1 = Critic(name="critic_1")(obs, action, train)
        q2 = Critic(name="critic_2")(obs, action, train)
        return jnp.concatenate([q1, q2], axis=-1)  # [batch, 2]
```

---

## 2. Environment Conversion

### 2.1 TradingEnvironment

**Source:** `eigen2/environment/trading_env.py`

**Target:** `eigen3/environment/trading_env.py`

**Key Challenges:**
- Convert from Gymnasium API to EvoRL Env API
- Make all operations JAX-native (no Pandas, no NumPy)
- Support vectorization with `jax.vmap`
- Immutable state management

**EvoRL Env Interface:**

```python
from evorl.envs import Env, EnvState
from evorl.types import PyTreeData
import chex

class TradingEnvState(PyTreeData):
    """Trading environment internal state"""
    # Time tracking
    current_step: chex.Array  # scalar int
    trading_end_step: chex.Array  # scalar int
    episode_end_step: chex.Array  # scalar int

    # Position tracking
    positions: chex.Array  # [max_positions, position_data]
    num_positions: chex.Array  # scalar int

    # Episode statistics
    cumulative_reward: chex.Array  # scalar float
    num_wins: chex.Array  # scalar int
    num_losses: chex.Array  # scalar int
    total_gain_pct: chex.Array  # scalar float

class Position(PyTreeData):
    """Single position state"""
    stock_idx: chex.Array  # int
    entry_step: chex.Array  # int
    entry_price: chex.Array  # float
    sale_target_price: chex.Array  # float
    coefficient: chex.Array  # float
    is_active: chex.Array  # bool

class TradingEnv(Env):
    """JAX-native trading environment"""

    def __init__(
        self,
        data_array: chex.Array,  # [num_days, 669, 5]
        data_array_full: chex.Array,  # [num_days, 669, 9]
        norm_stats: dict,
        context_window_days: int = 504,
        trading_period_days: int = 125,
        settlement_period_days: int = 20,
        max_holding_days: int = 20,
        max_positions: int = 10,
        inaction_penalty: float = 0.001,
        coefficient_threshold: float = 0.01,
    ):
        self.data_array = jnp.array(data_array)
        self.data_array_full = jnp.array(data_array_full)
        self.norm_stats = norm_stats
        self.context_window_days = context_window_days
        self.trading_period_days = trading_period_days
        self.settlement_period_days = settlement_period_days
        self.max_holding_days = max_holding_days
        self.max_positions = max_positions
        self.inaction_penalty = inaction_penalty
        self.coefficient_threshold = coefficient_threshold

        # Precompute valid episode ranges
        self.min_start_idx = context_window_days
        self.max_start_idx = len(data_array) - trading_period_days - settlement_period_days

    @property
    def obs_space(self):
        from evorl.envs import Box
        return Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self.context_window_days, 669, 5)
        )

    @property
    def action_space(self):
        from evorl.envs import Box
        return Box(
            low=jnp.array([[0.0, 10.0]]),
            high=jnp.array([[jnp.inf, 50.0]]),
            shape=(108, 2)
        )

    def reset(self, key: chex.PRNGKey) -> EnvState:
        """Reset to random episode window"""
        # Sample random start index
        start_idx = jax.random.randint(
            key,
            (),
            minval=self.min_start_idx,
            maxval=self.max_start_idx
        )

        trading_end_step = start_idx + self.trading_period_days
        episode_end_step = trading_end_step + self.settlement_period_days

        # Initialize empty positions
        positions = jnp.zeros((self.max_positions, 6))  # [stock_idx, entry_step, entry_price, target_price, coeff, is_active]

        # Initial state
        env_state = TradingEnvState(
            current_step=start_idx,
            trading_end_step=trading_end_step,
            episode_end_step=episode_end_step,
            positions=positions,
            num_positions=jnp.array(0),
            cumulative_reward=jnp.array(0.0),
            num_wins=jnp.array(0),
            num_losses=jnp.array(0),
            total_gain_pct=jnp.array(0.0),
        )

        # Get initial observation
        obs = self._get_observation(env_state)

        return EnvState(
            env_state=env_state,
            obs=obs,
            reward=jnp.array(0.0),
            done=jnp.array(False),
        )

    def step(self, state: EnvState, action: chex.Array) -> EnvState:
        """Take a step in the environment"""
        env_state = state.env_state

        # Update existing positions
        positions, reward, num_wins, num_losses, total_gain_pct = self._update_positions(
            env_state, env_state.current_step
        )

        # Process new action (only during trading period)
        positions, action_reward = jax.lax.cond(
            env_state.current_step < env_state.trading_end_step,
            lambda: self._process_action(env_state, action, positions),
            lambda: (positions, 0.0)
        )

        # Apply inaction penalty if no positions
        inaction_pen = jax.lax.cond(
            jnp.sum(positions[:, 5]) == 0,  # No active positions
            lambda: -self.inaction_penalty,
            lambda: 0.0
        )

        total_reward = reward + action_reward + inaction_pen

        # Update state
        new_step = env_state.current_step + 1
        done = new_step >= env_state.episode_end_step

        new_env_state = env_state.replace(
            current_step=new_step,
            positions=positions,
            cumulative_reward=env_state.cumulative_reward + total_reward,
            num_wins=num_wins,
            num_losses=num_losses,
            total_gain_pct=total_gain_pct,
        )

        obs = self._get_observation(new_env_state)

        return EnvState(
            env_state=new_env_state,
            obs=obs,
            reward=total_reward,
            done=done,
        )

    def _get_observation(self, env_state: TradingEnvState) -> chex.Array:
        """Get normalized observation window"""
        start = env_state.current_step - self.context_window_days
        end = env_state.current_step

        window = self.data_array[start:end]  # [context_days, 669, 5]

        # Normalize
        mean = jnp.array(self.norm_stats['mean'])
        std = jnp.array(self.norm_stats['std'])
        normalized = (window - mean) / (std + 1e-8)

        return normalized

    def _update_positions(self, env_state, current_step):
        """Update all positions and calculate rewards"""
        positions = env_state.positions

        def update_single_position(position):
            # Extract position data
            stock_idx = position[0].astype(int)
            entry_step = position[1].astype(int)
            entry_price = position[2]
            target_price = position[3]
            coefficient = position[4]
            is_active = position[5]

            # Skip inactive positions
            def inactive_branch():
                return position, 0.0, 0, 0, 0.0

            def active_branch():
                # Check exit conditions
                days_held = current_step - entry_step
                current_data = self.data_array_full[current_step, stock_idx]

                # Exit if: (1) target hit, (2) max holding, (3) delisted
                high_price = current_data[3]  # High price
                close_price = current_data[0]  # Close price
                is_valid = jnp.isfinite(close_price)

                target_hit = high_price >= target_price
                max_holding_reached = days_held >= self.max_holding_days
                should_exit = target_hit | max_holding_reached | ~is_valid

                # Determine exit price
                exit_price = jax.lax.select(
                    target_hit,
                    target_price,
                    jax.lax.select(is_valid, close_price, entry_price)
                )

                # Calculate reward
                gain_pct = ((exit_price - entry_price) / entry_price) * 100.0
                pos_reward = coefficient * gain_pct

                # Update position
                new_position = jax.lax.select(
                    should_exit,
                    jnp.zeros_like(position),  # Close position
                    position  # Keep position
                )

                # Win/loss tracking
                win = jax.lax.select(should_exit & (gain_pct > 0), 1, 0)
                loss = jax.lax.select(should_exit & (gain_pct <= 0), 1, 0)

                return (
                    new_position,
                    jax.lax.select(should_exit, pos_reward, 0.0),
                    win,
                    loss,
                    jax.lax.select(should_exit, gain_pct, 0.0)
                )

            return jax.lax.cond(is_active > 0.5, active_branch, inactive_branch)

        # Vectorized update
        results = jax.vmap(update_single_position)(positions)
        new_positions, rewards, wins, losses, gains = results

        return (
            new_positions,
            jnp.sum(rewards),
            env_state.num_wins + jnp.sum(wins),
            env_state.num_losses + jnp.sum(losses),
            env_state.total_gain_pct + jnp.sum(gains)
        )

    def _process_action(self, env_state, action, positions):
        """Process new action and potentially open a position"""
        # Find stock with highest coefficient
        coefficients = action[:, 0]
        best_stock_idx = jnp.argmax(coefficients)
        best_coeff = coefficients[best_stock_idx]
        sale_target_pct = action[best_stock_idx, 1]

        # Validate action
        valid_coeff = best_coeff > self.coefficient_threshold
        position_available = jnp.sum(positions[:, 5]) < self.max_positions

        # Check if stock already has a position
        stock_offset = 8  # Investable stocks start at column 8
        actual_stock_idx = best_stock_idx + stock_offset
        has_position = jnp.any(
            (positions[:, 0] == actual_stock_idx) & (positions[:, 5] > 0.5)
        )

        # Check if data is valid
        current_data = self.data_array_full[env_state.current_step, actual_stock_idx]
        entry_price = current_data[0]  # Close price
        data_valid = jnp.isfinite(entry_price)

        can_open = valid_coeff & position_available & ~has_position & data_valid

        def open_position_branch():
            # Calculate target price
            target_price = entry_price * (1.0 + sale_target_pct / 100.0)

            # Create new position
            new_position = jnp.array([
                actual_stock_idx,
                env_state.current_step,
                entry_price,
                target_price,
                best_coeff,
                1.0  # is_active
            ])

            # Find first inactive slot
            inactive_mask = positions[:, 5] < 0.5
            slot_idx = jnp.argmax(inactive_mask)

            # Insert position
            new_positions = positions.at[slot_idx].set(new_position)
            return new_positions, 0.0

        def no_action_branch():
            return positions, 0.0

        return jax.lax.cond(can_open, open_position_branch, no_action_branch)
```

**Vectorization Support:**

The environment can be vectorized using `jax.vmap`:

```python
# Create vectorized environment
vectorized_env = TradingEnv(data_array, data_array_full, norm_stats)

# Vectorize reset and step
batch_size = 16
keys = jax.random.split(key, batch_size)
states = jax.vmap(vectorized_env.reset)(keys)
actions = jax.random.uniform(key, (batch_size, 108, 2))
new_states = jax.vmap(vectorized_env.step)(states, actions)
```

---

## 3. DDPG Agent Conversion

**Source:** `eigen2/models/ddpg_agent.py`

**Target:** `eigen3/agents/trading_agent.py`

**Conversion Strategy:**

```python
from evorl.agent import Agent, AgentState
from evorl.types import PyTreeData, PyTreeNode
import optax

class TradingNetworkParams(PyTreeData):
    """Network parameters for trading agent"""
    actor_params: chex.ArrayTree
    critic_params: chex.ArrayTree
    target_actor_params: chex.ArrayTree
    target_critic_params: chex.ArrayTree

class TradingAgent(Agent, PyTreeNode):
    """DDPG-based trading agent"""

    actor_network: nn.Module
    critic_network: nn.Module
    exploration_noise: float = 0.1
    tau: float = 0.005  # Soft target update rate
    discount: float = 0.99

    def init(
        self,
        obs_space: Space,
        action_space: Space,
        key: chex.PRNGKey
    ) -> AgentState:
        """Initialize agent parameters"""
        actor_key, critic_key = jax.random.split(key)

        # Create dummy inputs
        dummy_obs = jnp.zeros((1, *obs_space.shape))
        dummy_action = jnp.zeros((1, *action_space.shape))

        # Initialize networks
        actor_params = self.actor_network.init(
            actor_key, dummy_obs, train=False
        )
        critic_params = self.critic_network.init(
            critic_key, dummy_obs, dummy_action, train=False
        )

        # Initialize targets (copies)
        params = TradingNetworkParams(
            actor_params=actor_params,
            critic_params=critic_params,
            target_actor_params=actor_params,
            target_critic_params=critic_params,
        )

        return AgentState(params=params)

    def compute_actions(
        self,
        agent_state: AgentState,
        sample_batch: SampleBatch,
        key: chex.PRNGKey
    ) -> tuple[chex.Array, dict]:
        """Get actions with exploration noise"""
        params = agent_state.params
        obs = sample_batch.obs

        # Get deterministic action
        actions = self.actor_network.apply(
            params.actor_params,
            obs,
            train=False
        )

        # Add exploration noise
        noise = jax.random.normal(key, actions.shape) * self.exploration_noise
        noisy_actions = actions + noise

        # Clip to valid ranges
        noisy_actions = noisy_actions.at[:, :, 0].set(
            jnp.maximum(noisy_actions[:, :, 0], 0.0)  # Coefficient >= 0
        )
        noisy_actions = noisy_actions.at[:, :, 1].set(
            jnp.clip(noisy_actions[:, :, 1], 10.0, 50.0)  # Sale target [10, 50]
        )

        return noisy_actions, {}

    def evaluate_actions(
        self,
        agent_state: AgentState,
        sample_batch: SampleBatch,
        key: chex.PRNGKey
    ) -> tuple[chex.Array, dict]:
        """Get deterministic actions (no noise)"""
        params = agent_state.params
        obs = sample_batch.obs

        actions = self.actor_network.apply(
            params.actor_params,
            obs,
            train=False
        )

        return actions, {}

    def loss(
        self,
        agent_state: AgentState,
        sample_batch: SampleBatch,
        key: chex.PRNGKey
    ) -> dict[str, chex.Array]:
        """Compute DDPG losses"""
        params = agent_state.params

        obs = sample_batch.obs
        actions = sample_batch.actions
        rewards = sample_batch.rewards
        next_obs = sample_batch.next_obs
        dones = sample_batch.dones

        # Critic loss (TD error)
        # Q(s, a)
        current_q = self.critic_network.apply(
            params.critic_params,
            obs,
            actions,
            train=True
        )

        # Target: r + gamma * Q_target(s', a'(s'))
        next_actions = self.actor_network.apply(
            params.target_actor_params,
            next_obs,
            train=False
        )
        next_q = self.critic_network.apply(
            params.target_critic_params,
            next_obs,
            next_actions,
            train=False
        )

        # Handle twin critics (TD3-style)
        if next_q.shape[-1] == 2:
            next_q = jnp.min(next_q, axis=-1, keepdims=True)

        target_q = rewards + self.discount * (1.0 - dones) * next_q
        target_q = jax.lax.stop_gradient(target_q)

        critic_loss = jnp.mean((current_q - target_q) ** 2)

        # Actor loss (maximize Q)
        actor_actions = self.actor_network.apply(
            params.actor_params,
            obs,
            train=True
        )
        actor_q = self.critic_network.apply(
            params.critic_params,
            obs,
            actor_actions,
            train=True
        )

        if actor_q.shape[-1] == 2:
            actor_q = jnp.mean(actor_q, axis=-1, keepdims=True)

        actor_loss = -jnp.mean(actor_q)

        return {
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'mean_q': jnp.mean(current_q),
        }

def soft_target_update(params: TradingNetworkParams, tau: float) -> TradingNetworkParams:
    """Soft update of target networks"""

    def update_tree(target, source):
        return jax.tree_map(
            lambda t, s: tau * s + (1 - tau) * t,
            target,
            source
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
```

---

## 4. ERL Workflow Creation

**Target:** `eigen3/workflows/trading_erl_workflow.py`

**Strategy:** Customize EvoRL's ERLWorkflow for trading

```python
from evorl.workflows import ERLWorkflowBase
from evorl.algorithms.erl import create_next_generation
from evorl.replay_buffers import UniformReplayBuffer
import jax

class TradingERLWorkflow(ERLWorkflowBase):
    """Custom ERL workflow for stock trading"""

    # Hyperparameters
    pop_size: int = 16
    num_elites: int = 6
    num_rl_agents: int = 1
    mutation_strength: float = 0.025
    mutation_rate: float = 0.2

    rl_updates_per_gen: int = 32
    batch_size: int = 64

    episodes_per_agent: int = 5  # Multi-slice evaluation

    def setup(self, key: chex.PRNGKey):
        """Initialize workflow"""
        # Create agent
        agent_key, env_key, key = jax.random.split(key, 3)

        agent = TradingAgent(
            actor_network=Actor(),
            critic_network=DoubleCritic(),
        )

        # Initialize agent population
        pop_keys = jax.random.split(agent_key, self.pop_size)
        pop_agent_states = jax.vmap(
            lambda k: agent.init(self.env.obs_space, self.env.action_space, k)
        )(pop_keys)

        # Initialize RL optimizer
        optimizer = optax.adam(learning_rate=3e-4)
        rl_opt_state = optimizer.init(pop_agent_states[0].params)

        # Initialize replay buffer
        replay_buffer = UniformReplayBuffer(
            max_size=100000,
            obs_space=self.env.obs_space,
            action_space=self.env.action_space,
        )
        replay_buffer_state = replay_buffer.init(key)

        # Initialize environment
        env_state = self.env.reset(env_key)

        return State(
            agent_states=pop_agent_states,
            rl_opt_state=rl_opt_state,
            replay_buffer_state=replay_buffer_state,
            env_state=env_state,
            generation=0,
            best_fitness=-jnp.inf,
            best_agent_state=pop_agent_states[0],
        )

    def step(self, state: State) -> tuple[Metrics, State]:
        """Execute one generation"""

        # 1. Evaluate population (conservative multi-slice)
        fitnesses, eval_metrics = self._evaluate_population(state)

        # 2. Train RL agent with DDPG
        rl_agent_state, rl_metrics = self._train_rl_agent(state)

        # 3. Evolve population
        new_pop = self._evolve_population(state.agent_states, fitnesses)

        # 4. RL injection (replace worst with RL agent)
        new_pop = new_pop.at[-1].set(rl_agent_state)

        # 5. Update best agent
        best_idx = jnp.argmax(fitnesses)
        best_fitness = fitnesses[best_idx]
        best_agent = new_pop[best_idx]

        new_state = state.replace(
            agent_states=new_pop,
            generation=state.generation + 1,
            best_fitness=jnp.maximum(state.best_fitness, best_fitness),
            best_agent_state=jax.lax.select(
                best_fitness > state.best_fitness,
                best_agent,
                state.best_agent_state
            ),
        )

        metrics = {
            **eval_metrics,
            **rl_metrics,
            'best_fitness': best_fitness,
        }

        return metrics, new_state

    def _evaluate_population(self, state):
        """Conservative multi-slice evaluation"""

        def evaluate_single_agent(agent_state, key):
            # Run 5 episodes on random slices
            keys = jax.random.split(key, self.episodes_per_agent)

            def run_episode(k):
                env_state = self.env.reset(k)
                trajectory, env_state = rollout(
                    self.env.step,
                    lambda s, sb, k: self.agent.evaluate_actions(s, sb, k)[0],
                    env_state,
                    agent_state,
                    k,
                    rollout_length=145,  # Trading + settlement
                )
                return trajectory.rewards.sum()

            episode_returns = jax.vmap(run_episode)(keys)

            # Conservative: average of 2 worst episodes
            sorted_returns = jnp.sort(episode_returns)
            fitness = jnp.mean(sorted_returns[:2])

            return fitness

        # Evaluate all agents
        eval_keys = jax.random.split(state.key, self.pop_size)
        fitnesses = jax.vmap(evaluate_single_agent)(
            state.agent_states,
            eval_keys
        )

        metrics = {
            'mean_fitness': jnp.mean(fitnesses),
            'max_fitness': jnp.max(fitnesses),
            'min_fitness': jnp.min(fitnesses),
        }

        return fitnesses, metrics

    def _train_rl_agent(self, state):
        """Train RL agent with DDPG"""
        rl_agent_state = state.agent_states[0]  # First agent is RL

        # Rollout and collect experience
        trajectory, env_state = rollout(
            self.env.step,
            self.agent.compute_actions,
            state.env_state,
            rl_agent_state,
            state.key,
            rollout_length=145,
        )

        # Add to replay buffer
        replay_buffer_state = self.replay_buffer.add(
            state.replay_buffer_state,
            trajectory
        )

        # Multiple gradient updates
        def update_step(carry, _):
            agent_state, opt_state, key = carry

            # Sample batch
            key, sample_key = jax.random.split(key)
            batch = self.replay_buffer.sample(
                replay_buffer_state,
                sample_key,
                self.batch_size
            )

            # Update
            loss_dict, agent_state, opt_state = agent_gradient_update(
                self.agent.loss,
                self.optimizer,
            )(opt_state, agent_state, batch, key)

            # Soft target update
            agent_state = agent_state.replace(
                params=soft_target_update(agent_state.params, self.tau)
            )

            return (agent_state, opt_state, key), loss_dict

        (rl_agent_state, rl_opt_state, _), losses = jax.lax.scan(
            update_step,
            (rl_agent_state, state.rl_opt_state, state.key),
            None,
            length=self.rl_updates_per_gen
        )

        metrics = {
            'rl_actor_loss': jnp.mean(losses['actor_loss']),
            'rl_critic_loss': jnp.mean(losses['critic_loss']),
        }

        return rl_agent_state, metrics

    def _evolve_population(self, population, fitnesses):
        """Genetic algorithm evolution"""
        # Use EvoRL's built-in genetic operators or custom ones
        return create_next_generation(
            population,
            fitnesses,
            num_elites=self.num_elites,
            mutation_strength=self.mutation_strength,
            mutation_rate=self.mutation_rate,
        )
```

---

## 5. Data Pipeline Conversion

**Source:** `eigen2/data/loader.py`

**Target:** `eigen3/data/loader.py`

**Strategy:** Minimal changes - load data as NumPy, convert to JAX arrays

```python
import jax.numpy as jnp
import numpy as np
import pandas as pd

def load_trading_data(filepath: str) -> tuple[jnp.ndarray, jnp.ndarray, dict]:
    """Load and preprocess trading data"""

    # Load CSV/pickle (NumPy is fine here - one-time operation)
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    else:
        df = pd.read_pickle(filepath)

    # Process data (same as eigen2)
    # ...

    # Convert to JAX arrays
    data_array = jnp.array(data_array_np)  # [T, 669, 5]
    data_array_full = jnp.array(data_array_full_np)  # [T, 669, 9]

    # Compute normalization stats
    mean = jnp.mean(data_array, axis=(0, 1))
    std = jnp.std(data_array, axis=(0, 1))

    norm_stats = {
        'mean': mean,
        'std': std,
    }

    return data_array, data_array_full, norm_stats
```

---

## 6. Configuration Files

**Target:** `eigen3/configs/`

### Agent Config: `configs/agent/trading_erl.yaml`

```yaml
workflow_cls: eigen3.workflows.trading_erl_workflow.TradingERLWorkflow

# Environment
env: trading  # Custom env registered in eigen3.environment
context_window_days: 504
trading_period_days: 125
settlement_period_days: 20

# Population
pop_size: 16
num_elites: 6

# RL Training
num_rl_agents: 1
rl_updates_per_gen: 32
batch_size: 64
discount: 0.99
tau: 0.005
exploration_noise: 0.1

# Genetic Algorithm
mutation_strength: 0.025
mutation_rate: 0.2
tournament_size: 3

# Network Architecture
actor_network:
  num_stocks: 108
  use_remat: true

critic_network:
  use_attention: false
  use_remat: true

# Optimizer
optimizer:
  lr: 0.0003
  grad_clip_norm: 1.0

# Training
total_generations: 10000
episodes_per_agent: 5  # Multi-slice evaluation
eval_interval: 50

# Logging
logging:
  wandb: true
  tensorboard: true
  log_interval: 1
```

---

## 7. Testing Strategy

### 7.1 Component Tests

Create unit tests for each converted component:

```python
# tests/test_networks.py
def test_feature_extractor():
    batch_size = 2
    x = jnp.ones((batch_size, 504, 669, 5))

    model = FeatureExtractor()
    params = model.init(jax.random.PRNGKey(0), x, train=False)

    output = model.apply(params, x, train=False)

    assert output.shape == (batch_size, 669, 256)

def test_actor():
    batch_size = 2
    obs = jnp.ones((batch_size, 504, 669, 5))

    model = Actor()
    params = model.init(jax.random.PRNGKey(0), obs, train=False)

    actions = model.apply(params, obs, train=False)

    assert actions.shape == (batch_size, 108, 2)
    assert jnp.all(actions[:, :, 0] >= 0)  # Coefficients
    assert jnp.all(actions[:, :, 1] >= 10) & jnp.all(actions[:, :, 1] <= 50)  # Sale targets

# tests/test_environment.py
def test_trading_env():
    # Create dummy data
    data_array = jnp.ones((1000, 669, 5))
    data_array_full = jnp.ones((1000, 669, 9))
    norm_stats = {'mean': jnp.zeros(5), 'std': jnp.ones(5)}

    env = TradingEnv(data_array, data_array_full, norm_stats)

    # Test reset
    state = env.reset(jax.random.PRNGKey(0))
    assert state.obs.shape == (504, 669, 5)

    # Test step
    action = jnp.ones((108, 2))
    new_state = env.step(state, action)
    assert new_state.obs.shape == (504, 669, 5)
    assert isinstance(new_state.reward, jnp.ndarray)

# tests/test_agent.py
def test_trading_agent():
    agent = TradingAgent(
        actor_network=Actor(),
        critic_network=DoubleCritic(),
    )

    obs_space = Box(low=-jnp.inf, high=jnp.inf, shape=(504, 669, 5))
    action_space = Box(low=0, high=jnp.inf, shape=(108, 2))

    agent_state = agent.init(obs_space, action_space, jax.random.PRNGKey(0))

    # Test action computation
    sample_batch = SampleBatch(obs=jnp.ones((1, 504, 669, 5)))
    actions, _ = agent.compute_actions(agent_state, sample_batch, jax.random.PRNGKey(1))

    assert actions.shape == (1, 108, 2)
```

### 7.2 Integration Tests

```python
# tests/test_integration.py
def test_full_training_step():
    """Test one full generation"""

    # Load data
    data_array, data_array_full, norm_stats = load_trading_data("data/test.pkl")

    # Create environment
    env = TradingEnv(data_array, data_array_full, norm_stats)

    # Create workflow
    config = load_config("configs/agent/trading_erl.yaml")
    workflow = TradingERLWorkflow.build_from_config(config, env)

    # Initialize
    state = workflow.init(jax.random.PRNGKey(0))

    # Run one step
    metrics, new_state = workflow.step(state)

    assert 'mean_fitness' in metrics
    assert new_state.generation == 1
```

---

## Phase 2: Integration with EvoRL (Brief Overview)

Once Phase 1 is complete, Phase 2 involves:

1. **Workflow Registration:** Register TradingERLWorkflow with EvoRL
2. **Environment Registration:** Register TradingEnv with EvoRL's env factory
3. **Training Script:** Create `scripts/train_trading.py` using EvoRL's training utilities
4. **Monitoring:** Set up WandB/TensorBoard logging
5. **Checkpointing:** Implement model saving/loading
6. **Hyperparameter Tuning:** Use Hydra for sweeps

---

## Timeline & Milestones

### Week 1-2: Core Networks
- [ ] FeatureExtractor (CNN-LSTM)
- [ ] AttentionModule
- [ ] Actor network
- [ ] Critic network
- [ ] Unit tests for all networks

### Week 3: Environment
- [ ] TradingEnv with EvoRL interface
- [ ] Vectorization support
- [ ] Position management
- [ ] Environment tests

### Week 4: Agent & Replay
- [ ] TradingAgent (DDPG)
- [ ] Replay buffer integration
- [ ] Agent tests

### Week 5: Workflow
- [ ] TradingERLWorkflow
- [ ] Population evaluation
- [ ] RL training loop
- [ ] Genetic operators

### Week 6: Integration & Testing
- [ ] End-to-end training
- [ ] Config files
- [ ] Logging and monitoring
- [ ] Performance validation

---

## Key Differences: PyTorch vs JAX

| Aspect | PyTorch (Eigen2) | JAX (Eigen3) |
|--------|------------------|--------------|
| Paradigm | OOP, mutable state | Functional, immutable state |
| State | `model.parameters()` | External `AgentState` |
| Gradient | `loss.backward()` | `jax.grad(loss_fn)` |
| Optimization | `optimizer.step()` | `optax.update()` |
| LSTM | `nn.LSTM()` | `nn.LSTMCell` + `scan` |
| Checkpointing | `torch.utils.checkpoint` | `jax.checkpoint` (remat) |
| Target update | In-place `.copy_()` | Tree map with tau |
| Vectorization | Manual loops | `jax.vmap` |
| Random | `torch.randn()` | `jax.random.normal(key)` |
| Control flow | Python if/for | `jax.lax.cond/scan` |

---

## Critical Success Factors

1. **Functional Purity:** All functions must be pure (no side effects)
2. **Immutable State:** Never mutate state - always create new copies
3. **JAX Transformations:** Design for `jit`, `vmap`, `grad`
4. **PRNGKey Management:** Thread keys explicitly through all random operations
5. **PyTree Structure:** Properly structure all state objects as PyTrees
6. **Memory Efficiency:** Use remat for large models, chunking for large data

---

## Next Steps

1. ✅ Complete analysis of eigen2 (DONE)
2. ✅ Complete analysis of EvoRL (DONE)
3. ⏭️ Set up eigen3 project structure
4. ⏭️ Start with FeatureExtractor conversion
5. ⏭️ Proceed through network conversions
6. ⏭️ Environment conversion
7. ⏭️ Agent and workflow integration
8. ⏭️ End-to-end testing

Let's begin! 🚀
