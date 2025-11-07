"""JAX-native Trading Environment for stock market simulation

Implements the EvoRL Env interface with:
- Immutable state (PyTree-based)
- Pure functional operations
- Position management with fixed-size arrays
- Fully JIT-compilable and vmap-able
"""

from typing import Tuple
import jax
import jax.numpy as jnp
import chex
from evorl.types import PyTreeData, pytree_field
from evorl.envs import Env


class TradingEnvState(PyTreeData):
    """Trading environment internal state (immutable)"""
    # Time tracking
    current_step: chex.Array  # scalar int
    start_step: chex.Array  # scalar int
    end_step: chex.Array  # scalar int
    trading_end_step: chex.Array  # scalar int

    # Position tracking (fixed-size arrays for JIT)
    # positions shape: [max_positions, 6]
    # Each position: [stock_idx, entry_step, entry_price, target_price, coefficient, is_active]
    positions: chex.Array
    num_active_positions: chex.Array  # scalar int

    # Episode statistics
    cumulative_reward: chex.Array  # scalar float
    num_trades: chex.Array  # scalar int
    num_wins: chex.Array  # scalar int
    num_losses: chex.Array  # scalar int
    total_gain_pct: chex.Array  # scalar float
    days_with_positions: chex.Array  # scalar int
    days_without_positions: chex.Array  # scalar int


class EnvState(PyTreeData):
    """EvoRL-compatible environment state"""
    env_state: TradingEnvState
    obs: chex.Array
    reward: chex.Array
    done: chex.Array
    info: dict = pytree_field(default_factory=dict)


class TradingEnv(Env):
    """JAX-native trading environment for stock market simulation

    Matches PyTorch implementation from eigen2/environment/trading_env.py
    but with pure functional design for JAX.

    Observation: [context_days, num_columns, num_features] normalized window
    Action: [num_investable_stocks, 2] with [coefficient, sale_target] per stock
    Reward: Gains/losses from closed positions minus penalties
    """

    def __init__(
        self,
        data_array: chex.Array,  # [num_days, 669, 5] for observations
        data_array_full: chex.Array,  # [num_days, 669, 9] for rewards
        norm_stats: dict,  # {'mean': array, 'std': array}
        context_window_days: int = 504,
        trading_period_days: int = 125,
        settlement_period_days: int = 20,
        max_holding_days: int = 20,
        max_positions: int = 10,
        inaction_penalty: float = 5.0,
        coefficient_threshold: float = 0.5,
        min_coefficient: float = 1.0,
        min_sale_target: float = 10.0,
        max_sale_target: float = 50.0,
        investable_start_col: int = 8,
        num_investable_stocks: int = 108,
        loss_penalty_multiplier: float = 1.0,
    ):
        """Initialize trading environment

        Args:
            data_array: Observation data [num_days, num_columns, num_features]
            data_array_full: Full data for rewards [num_days, num_columns, 9]
            norm_stats: Normalization statistics {'mean', 'std'}
            context_window_days: Lookback window size
            trading_period_days: Days to open new positions
            settlement_period_days: Days to close existing positions
            max_holding_days: Force exit after N days
            max_positions: Maximum concurrent positions
            inaction_penalty: Penalty per day with no positions
            coefficient_threshold: Minimum coefficient to open position
            min_coefficient: Absolute minimum coefficient
            min_sale_target: Minimum sale target percentage
            max_sale_target: Maximum sale target percentage
            investable_start_col: Starting column for investable stocks
            num_investable_stocks: Number of tradeable stocks
            loss_penalty_multiplier: Multiplier for loss penalties
        """
        self.data_array = jnp.array(data_array, dtype=jnp.float32)
        self.data_array_full = jnp.array(data_array_full, dtype=jnp.float32)
        self.norm_mean = jnp.array(norm_stats['mean'], dtype=jnp.float32)
        self.norm_std = jnp.array(norm_stats['std'], dtype=jnp.float32)

        self.context_window_days = context_window_days
        self.trading_period_days = trading_period_days
        self.settlement_period_days = settlement_period_days
        self.max_holding_days = max_holding_days
        self.max_positions = max_positions
        self.inaction_penalty = inaction_penalty
        self.coefficient_threshold = coefficient_threshold
        self.min_coefficient = min_coefficient
        self.min_sale_target = min_sale_target
        self.max_sale_target = max_sale_target
        self.investable_start_col = investable_start_col
        self.num_investable_stocks = num_investable_stocks
        self.loss_penalty_multiplier = loss_penalty_multiplier

        # Compute valid episode ranges
        self.min_start_idx = context_window_days
        self.max_start_idx = len(data_array) - trading_period_days - settlement_period_days

        # Total episode length
        self.episode_length = trading_period_days + settlement_period_days

    @property
    def obs_space(self):
        """Observation space"""
        from evorl.envs import Box
        return Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self.context_window_days, self.data_array.shape[1], self.data_array.shape[2])
        )

    @property
    def action_space(self):
        """Action space"""
        from evorl.envs import Box
        return Box(
            low=jnp.array([[0.0, self.min_sale_target]]),
            high=jnp.array([[jnp.inf, self.max_sale_target]]),
            shape=(self.num_investable_stocks, 2)
        )

    def reset(self, key: chex.PRNGKey) -> EnvState:
        """Reset to random episode window

        Args:
            key: JAX random key

        Returns:
            Initial EnvState
        """
        # Sample random start index
        start_idx = jax.random.randint(
            key,
            shape=(),
            minval=self.min_start_idx,
            maxval=self.max_start_idx + 1
        )

        trading_end_step = start_idx + self.trading_period_days
        end_step = trading_end_step + self.settlement_period_days

        # Initialize empty positions
        # Position format: [stock_idx, entry_step, entry_price, target_price, coefficient, is_active]
        positions = jnp.zeros((self.max_positions, 6), dtype=jnp.float32)

        # Create initial trading state
        trading_state = TradingEnvState(
            current_step=start_idx,
            start_step=start_idx,
            end_step=end_step,
            trading_end_step=trading_end_step,
            positions=positions,
            num_active_positions=jnp.array(0, dtype=jnp.int32),
            cumulative_reward=jnp.array(0.0, dtype=jnp.float32),
            num_trades=jnp.array(0, dtype=jnp.int32),
            num_wins=jnp.array(0, dtype=jnp.int32),
            num_losses=jnp.array(0, dtype=jnp.int32),
            total_gain_pct=jnp.array(0.0, dtype=jnp.float32),
            days_with_positions=jnp.array(0, dtype=jnp.int32),
            days_without_positions=jnp.array(0, dtype=jnp.int32),
        )

        # Get initial observation
        obs = self._get_observation(trading_state)

        return EnvState(
            env_state=trading_state,
            obs=obs,
            reward=jnp.array(0.0, dtype=jnp.float32),
            done=jnp.array(False, dtype=jnp.bool_),
        )

    def step(self, state: EnvState, action: chex.Array) -> EnvState:
        """Take a step in the environment

        Args:
            state: Current EnvState
            action: Action array [num_investable_stocks, 2]

        Returns:
            New EnvState
        """
        env_state = state.env_state

        # Count active positions before updates
        had_positions = env_state.num_active_positions > 0

        # 1. Update existing positions and collect rewards
        positions, close_reward, num_wins, num_losses, total_gain, num_trades = self._update_positions(
            env_state
        )

        # 2. Process new action (only during trading period)
        positions, action_reward, position_opened = jax.lax.cond(
            env_state.current_step < env_state.trading_end_step,
            lambda: self._process_action(env_state, action, positions),
            lambda: (positions, 0.0, False)
        )

        # Count active positions after updates
        num_active = jnp.sum(positions[:, 5])  # Count is_active flags

        # 3. Apply inaction penalty if no positions
        has_positions_now = num_active > 0
        inaction_pen = jax.lax.select(
            ~had_positions & ~has_positions_now,
            -self.inaction_penalty,
            0.0
        )

        # Track days with/without positions
        days_with = env_state.days_with_positions + jax.lax.select(has_positions_now, 1, 0)
        days_without = env_state.days_without_positions + jax.lax.select(~has_positions_now, 1, 0)

        # Total reward for this step
        step_reward = close_reward + action_reward + inaction_pen

        # 4. Move to next step
        new_step = env_state.current_step + 1
        done = new_step >= env_state.end_step

        # 5. Update state
        new_env_state = env_state.replace(
            current_step=new_step,
            positions=positions,
            num_active_positions=jnp.array(num_active, dtype=jnp.int32),
            cumulative_reward=env_state.cumulative_reward + step_reward,
            num_trades=num_trades,
            num_wins=num_wins,
            num_losses=num_losses,
            total_gain_pct=total_gain,
            days_with_positions=days_with,
            days_without_positions=days_without,
        )

        # Get new observation
        obs = self._get_observation(new_env_state)

        return EnvState(
            env_state=new_env_state,
            obs=obs,
            reward=step_reward,
            done=done,
        )

    def _get_observation(self, env_state: TradingEnvState) -> chex.Array:
        """Get normalized observation window

        Args:
            env_state: Current trading state

        Returns:
            Normalized window [context_days, num_columns, num_features]
        """
        # Extract window ending at current step
        start = env_state.current_step - self.context_window_days + 1
        end = env_state.current_step + 1

        window = jax.lax.dynamic_slice(
            self.data_array,
            (start, 0, 0),
            (self.context_window_days, self.data_array.shape[1], self.data_array.shape[2])
        )

        # Normalize
        normalized = (window - self.norm_mean) / (self.norm_std + 1e-8)

        return normalized

    def _update_positions(
        self,
        env_state: TradingEnvState
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
        """Update all positions and close if necessary

        Args:
            env_state: Current trading state

        Returns:
            Tuple of (new_positions, total_reward, num_wins, num_losses, total_gain_pct, num_trades)
        """
        positions = env_state.positions

        def update_single_position(position):
            """Update single position and check exit conditions"""
            stock_idx = position[0].astype(jnp.int32)
            entry_step = position[1].astype(jnp.int32)
            entry_price = position[2]
            target_price = position[3]
            coefficient = position[4]
            is_active = position[5]

            # Skip inactive positions
            def inactive_branch():
                return position, 0.0, 0, 0, 0.0

            def active_branch():
                # Increment days held
                days_held = env_state.current_step - entry_step

                # Get current stock data
                actual_col_idx = self.investable_start_col + stock_idx
                current_data = self.data_array_full[env_state.current_step, actual_col_idx]

                # Extract prices (full data: [Open, Close, High, Low, ...])
                day_high = current_data[2]  # High price
                close_price = current_data[1]  # Close price
                is_valid = jnp.isfinite(close_price)

                # Check exit conditions
                target_hit = day_high >= target_price
                max_holding_reached = days_held >= self.max_holding_days
                should_exit = target_hit | max_holding_reached | ~is_valid

                # Determine exit price
                exit_price = jax.lax.select(
                    target_hit,
                    target_price,
                    jax.lax.select(is_valid, close_price, entry_price)
                )

                # Calculate gain/loss
                gain_pct = ((exit_price - entry_price) / entry_price) * 100.0

                # Calculate reward (apply loss penalty multiplier for losses)
                pos_reward = jax.lax.select(
                    gain_pct >= 0,
                    coefficient * gain_pct,
                    -self.loss_penalty_multiplier * coefficient * jnp.abs(gain_pct)
                )

                # Update position (close if should_exit)
                new_position = jax.lax.select(
                    should_exit,
                    jnp.zeros_like(position),  # Close position
                    position  # Keep position
                )

                # Win/loss tracking
                win = jax.lax.select(should_exit & (gain_pct > 0), 1, 0)
                loss = jax.lax.select(should_exit & (gain_pct <= 0), 1, 0)
                trade = jax.lax.select(should_exit, 1, 0)

                return (
                    new_position,
                    jax.lax.select(should_exit, pos_reward, 0.0),
                    win,
                    loss,
                    jax.lax.select(should_exit, gain_pct, 0.0),
                )

            return jax.lax.cond(
                is_active > 0.5,
                active_branch,
                inactive_branch
            )

        # Vectorized update across all positions
        results = jax.vmap(update_single_position)(positions)
        new_positions, rewards, wins, losses, gains = results

        return (
            new_positions,
            jnp.sum(rewards),
            env_state.num_wins + jnp.sum(wins),
            env_state.num_losses + jnp.sum(losses),
            env_state.total_gain_pct + jnp.sum(gains),
            env_state.num_trades + jnp.sum(jax.lax.select(rewards != 0, 1, 0))
        )

    def _process_action(
        self,
        env_state: TradingEnvState,
        action: chex.Array,
        positions: chex.Array
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Process action and potentially open new position

        Args:
            env_state: Current trading state
            action: Action array [num_investable_stocks, 2]
            positions: Current positions array

        Returns:
            Tuple of (new_positions, reward, position_opened)
        """
        # Extract coefficients and sale targets
        coefficients = action[:, 0]
        sale_targets = action[:, 1]

        # Find stock with highest coefficient
        best_stock_idx = jnp.argmax(coefficients)
        best_coeff = coefficients[best_stock_idx]
        sale_target_pct = sale_targets[best_stock_idx]

        # Clip sale target to valid range
        sale_target_pct = jnp.clip(sale_target_pct, self.min_sale_target, self.max_sale_target)

        # Check if action is valid
        valid_coeff = (best_coeff >= self.coefficient_threshold) & (best_coeff >= self.min_coefficient)

        # Check if position available
        position_available = env_state.num_active_positions < self.max_positions

        # Check if stock already has a position
        has_position = jnp.any(
            (positions[:, 0] == best_stock_idx) & (positions[:, 5] > 0.5)
        )

        # Get stock data
        actual_col_idx = self.investable_start_col + best_stock_idx
        current_data = self.data_array_full[env_state.current_step, actual_col_idx]
        entry_price = current_data[1]  # Close price
        data_valid = jnp.isfinite(entry_price) & (entry_price > 0)

        # Can open position?
        can_open = valid_coeff & position_available & ~has_position & data_valid

        def open_position_branch():
            """Open a new position"""
            # Calculate target price
            target_price = entry_price * (1.0 + sale_target_pct / 100.0)

            # Create new position
            new_position = jnp.array([
                best_stock_idx,
                env_state.current_step,
                entry_price,
                target_price,
                best_coeff,
                1.0  # is_active
            ])

            # Find first inactive slot
            inactive_mask = positions[:, 5] < 0.5
            slot_idx = jnp.argmax(inactive_mask.astype(jnp.float32))

            # Insert position
            new_positions = positions.at[slot_idx].set(new_position)

            return new_positions, 0.0, True

        def no_action_branch():
            """No position opened"""
            return positions, 0.0, False

        return jax.lax.cond(can_open, open_position_branch, no_action_branch)


def test_trading_env():
    """Test the TradingEnv implementation"""
    import jax.random as random

    print("Testing TradingEnv...")

    # Create dummy data
    num_days = 1000
    num_columns = 669
    data_array = jnp.ones((num_days, num_columns, 5))
    data_array_full = jnp.ones((num_days, num_columns, 9))

    # Add some price variation
    key = random.PRNGKey(0)
    prices = 100.0 + random.normal(key, (num_days, num_columns)) * 10
    data_array = data_array.at[:, :, 0].set(prices)  # Close prices
    data_array_full = data_array_full.at[:, :, 1].set(prices)  # Close in full data
    data_array_full = data_array_full.at[:, :, 2].set(prices * 1.02)  # High prices

    norm_stats = {
        'mean': jnp.zeros(5),
        'std': jnp.ones(5)
    }

    # Create environment
    env = TradingEnv(data_array, data_array_full, norm_stats)

    # Test reset
    key = random.PRNGKey(0)
    state = env.reset(key)

    print(f"✓ Reset successful")
    print(f"  Observation shape: {state.obs.shape}")
    print(f"  Expected: (504, 669, 5)")
    assert state.obs.shape == (504, 669, 5)

    # Test step
    action = jnp.ones((108, 2))
    action = action.at[:, 0].set(2.0)  # Coefficients
    action = action.at[:, 1].set(20.0)  # Sale targets

    new_state = env.step(state, action)

    print(f"✓ Step successful")
    print(f"  Observation shape: {new_state.obs.shape}")
    print(f"  Reward: {new_state.reward}")
    print(f"  Done: {new_state.done}")

    # Run multiple steps
    print("\nRunning 10 steps...")
    for i in range(10):
        action = jnp.ones((108, 2)) * (1.0 + i * 0.1)
        action = action.at[:, 1].set(20.0)
        new_state = env.step(new_state, action)
        print(f"  Step {i+1}: reward={new_state.reward:.4f}, "
              f"positions={new_state.env_state.num_active_positions}, "
              f"cum_reward={new_state.env_state.cumulative_reward:.4f}")

    print("\n✓ All TradingEnv tests passed!")

    return env, state


if __name__ == "__main__":
    test_trading_env()
