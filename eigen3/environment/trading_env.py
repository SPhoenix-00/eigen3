"""JAX-native Trading Environment for stock market simulation

Implements the EvoRL Env interface with:
- Immutable state (PyTree-based)
- Pure functional operations
- Position management with fixed-size arrays
- Fully JIT-compilable and vmap-able
"""

from typing import Tuple
import numpy as np
import jax
import jax.numpy as jnp
import chex
from evorl.types import PyTreeData, pytree_field
from evorl.envs import Env

from eigen3.config import (
    DEFAULT_CONVICTION_SCALING_POWER,
    DEFAULT_EPISODE_REWARD_MULTIPLIER,
    DEFAULT_HURDLE_RATE,
    DEFAULT_LOSS_PENALTY_MULTIPLIER,
)


class TradingEnvState(PyTreeData):
    """Trading environment internal state (immutable). Synced with Eigen2."""
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

    # Episode-level value-add tracking
    peak_capital_employed: chex.Array  # scalar float – max sum of entry prices across steps
    total_pnl: chex.Array  # scalar float – cumulative raw dollar PnL (exit − entry)
    # One-shot terminal term: agent vs equal-weight buy-hold (scaled $), same as episode bonus in reward
    episode_benchmark_excess: chex.Array  # scalar float; 0 until the terminal step

    # RNG for observation noise (Eigen2: multiplicative noise when is_training)
    rng_key: chex.Array = pytree_field(default_factory=lambda: jnp.zeros((2,), dtype=jnp.uint32))


class EnvState(PyTreeData):
    """EvoRL-compatible environment state"""
    env_state: TradingEnvState
    obs: chex.Array
    reward: chex.Array
    done: chex.Array
    info: dict = pytree_field(default_factory=dict)


class TradingEnv(Env):
    """JAX-native trading environment for stock market simulation.

    Supports single-stock (mono) and multi-stock modes:
    - Multiple buys allowed on the same stock
    - No sell until min_holding_period **calendar days** since the last buy.
      Calendar gaps are resolved via ``dates_ordinal``, a 1-D array mapping each
      row index to its ordinal day number (``datetime.date.toordinal()``).
    - All remaining positions liquidated at episode end

    Observation: [context_days, num_columns, num_features] normalized window
    Action: [num_investable_stocks, 3] with [coefficient, sale_target, close_fraction] per stock.
        close_fraction in [0, 1]: fraction of open lots on that stock to close at market when
        the min-holding rule is satisfied (FIFO by entry step).
    Reward: Gains/losses from closed positions minus penalties
    """

    @staticmethod
    def _build_calendar_episode_schedule(
        num_days: int,
        dates_ordinal: np.ndarray,
        episode_calendar_days: int,
        settlement_period_days: int,
        context_window_days: int,
        *,
        allow_empty: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
        """Per start row i: first exclusive row index after the calendar window.

        Inclusive calendar-day count from row i through last row j of the window
        equals ``episode_calendar_days`` (i.e. ``ord[j] - ord[i] == episode_calendar_days - 1``).
        """
        d = np.asarray(dates_ordinal, dtype=np.int64)
        if d.shape[0] != num_days:
            raise ValueError("dates_ordinal length must match num_days")
        span_target = int(episode_calendar_days) - 1
        if span_target < 0:
            raise ValueError("episode_calendar_days must be >= 1")

        cal_end_excl = np.full(num_days, -1, dtype=np.int32)
        for i in range(num_days):
            if i < int(context_window_days):
                continue
            target_ord = d[i] + span_target
            j = int(np.searchsorted(d, target_ord, side="left"))
            if j >= num_days:
                continue
            cal_excl = j + 1
            if cal_excl + int(settlement_period_days) > num_days:
                continue
            cal_end_excl[i] = cal_excl

        valid = np.nonzero(cal_end_excl >= 0)[0].astype(np.int32)
        if valid.size == 0:
            if allow_empty:
                return (
                    jnp.asarray(cal_end_excl, dtype=jnp.int32),
                    jnp.asarray(valid, dtype=jnp.int32),
                    0,
                )
            raise ValueError(
                "No valid episode starts: need enough rows and calendar span so that "
                f"{episode_calendar_days} inclusive calendar days fit after "
                f"context_window_days={context_window_days} (and settlement rows)."
            )

        spans = cal_end_excl[valid].astype(np.int64) + int(settlement_period_days) - valid.astype(np.int64)
        max_episode_rows = int(np.max(spans))
        return (
            jnp.asarray(cal_end_excl, dtype=jnp.int32),
            jnp.asarray(valid, dtype=jnp.int32),
            max_episode_rows,
        )

    def __init__(
        self,
        data_array: chex.Array,  # [num_days, num_columns, F] observations
        data_array_full: chex.Array,  # [num_days, num_columns, 9] price at index 1
        norm_stats: dict,  # {'mean': array, 'std': array}
        context_window_days: int = 151,
        trading_period_days: int = 364,
        settlement_period_days: int = 0,
        episode_calendar_days: int | None = None,
        min_holding_period: int = 30,
        max_positions: int = 10,
        inaction_penalty: float = 0.0,
        coefficient_threshold: float = 1.0,
        min_coefficient: float = 1.0,
        min_sale_target: float = 10.0,
        max_sale_target: float = 50.0,
        investable_start_col: int = 9,
        num_investable_stocks: int = 108,
        loss_penalty_multiplier: float = DEFAULT_LOSS_PENALTY_MULTIPLIER,
        hurdle_rate: float = DEFAULT_HURDLE_RATE,
        conviction_scaling_power: float = DEFAULT_CONVICTION_SCALING_POWER,
        observation_noise_std: float = 0.01,
        is_training: bool = True,
        dates_ordinal=None,
        episode_reward_multiplier: float = DEFAULT_EPISODE_REWARD_MULTIPLIER,
    ):
        """Initialize trading environment.

        Args:
            data_array: Observation data [num_days, num_columns, num_features]
            data_array_full: Price data for rewards [num_days, num_columns, 9]; price at index 1
            norm_stats: Normalization statistics {'mean', 'std'}
            context_window_days: Lookback window (default 151)
            trading_period_days: Target **calendar-day** span of the primary episode window
                (inclusive count from the date at the start row through the date at the
                last row of that window). Defaults to 364. Row count per episode follows
                from ``dates_ordinal`` (weekends/holidays may yield fewer than 364 rows).
            settlement_period_days: Extra **rows** after the calendar window (no new opens);
                use 0 for a single continuous window.
            episode_calendar_days: If set, overrides ``trading_period_days`` for the calendar
                span only (keeps ``trading_period_days`` attribute for logging/config parity).
            min_holding_period: Minimum **calendar days** since the last buy on a stock
                before any sell (target hit, discretionary, or liquidation). Calendar
                gaps between rows are handled by ``dates_ordinal``.
            max_positions: Maximum concurrent positions
            inaction_penalty: Penalty per day with no positions
            coefficient_threshold: Minimum coefficient to open position
            min_coefficient: Absolute minimum coefficient
            min_sale_target / max_sale_target: Sale target % bounds
            investable_start_col: Starting column for investable stocks
            num_investable_stocks: Number of tradeable stocks
            loss_penalty_multiplier: Multiplier for loss penalties
            hurdle_rate: Hurdle rate for reward
            conviction_scaling_power: Power for conviction scaling
            observation_noise_std: Multiplicative obs noise when is_training
            is_training: If True, apply observation noise
            dates_ordinal: 1-D int array [num_days] of ``date.toordinal()`` values.
                When ``None``, defaults to ``arange(num_days)`` (1 row = 1 calendar day).
            episode_reward_multiplier: Scale factor for the per-episode bonus/penalty
                (agent PnL vs buy-and-hold benchmark).
        """
        self.data_array = jnp.array(data_array, dtype=jnp.float32)
        self.data_array_full = jnp.array(data_array_full, dtype=jnp.float32)
        self.norm_mean = jnp.array(norm_stats['mean'], dtype=jnp.float32)
        self.norm_std = jnp.array(norm_stats['std'], dtype=jnp.float32)

        num_days = len(data_array)
        if dates_ordinal is None:
            dates_ordinal = jnp.arange(num_days, dtype=jnp.int32)
        self.dates_ordinal = jnp.array(dates_ordinal, dtype=jnp.int32)

        self.context_window_days = context_window_days
        self.trading_period_days = trading_period_days
        self.settlement_period_days = settlement_period_days
        self.episode_calendar_days = (
            int(episode_calendar_days)
            if episode_calendar_days is not None
            else int(trading_period_days)
        )
        self.min_holding_period = min_holding_period
        self.max_positions = max_positions
        self.inaction_penalty = inaction_penalty
        self.coefficient_threshold = coefficient_threshold
        self.min_coefficient = min_coefficient
        self.min_sale_target = min_sale_target
        self.max_sale_target = max_sale_target
        self.investable_start_col = investable_start_col
        self.num_investable_stocks = num_investable_stocks
        self.loss_penalty_multiplier = loss_penalty_multiplier
        self.hurdle_rate = hurdle_rate
        self.conviction_scaling_power = conviction_scaling_power
        self.observation_noise_std = observation_noise_std
        self.is_training = is_training
        self.episode_reward_multiplier = episode_reward_multiplier

        # Calendar-based episode: inclusive span episode_calendar_days on dates_ordinal.
        (
            self._calendar_end_exclusive,
            self._valid_start_indices,
            self.episode_length,
        ) = self._build_calendar_episode_schedule(
            num_days=num_days,
            dates_ordinal=np.asarray(self.dates_ordinal, dtype=np.int64),
            episode_calendar_days=self.episode_calendar_days,
            settlement_period_days=self.settlement_period_days,
            context_window_days=self.context_window_days,
        )
        self.min_start_idx = int(self._valid_start_indices.min()) if self._valid_start_indices.size > 0 else context_window_days
        self.max_start_idx = int(self._valid_start_indices.max()) if self._valid_start_indices.size > 0 else num_days - 1

    @property
    def obs_space(self):
        """Observation space"""
        from evorl.envs import Box
        shp = (self.context_window_days, self.data_array.shape[1], self.data_array.shape[2])
        return Box(low=jnp.full(shp, -jnp.inf), high=jnp.full(shp, jnp.inf))

    @property
    def action_space(self):
        """Action space"""
        from evorl.envs import Box
        shp = (self.num_investable_stocks, 3)
        return Box(
            low=jnp.broadcast_to(
                jnp.array([0.0, self.min_sale_target, 0.0], dtype=jnp.float32), shp
            ),
            high=jnp.broadcast_to(
                jnp.array([jnp.inf, self.max_sale_target, 1.0], dtype=jnp.float32), shp
            ),
        )

    def reset(self, key: chex.PRNGKey) -> EnvState:
        """Reset to random episode window

        Args:
            key: JAX random key

        Returns:
            Initial EnvState
        """
        # Random valid start (enough history + full calendar episode fits in data)
        pick_key, rng_key = jax.random.split(key)
        n_starts = self._valid_start_indices.shape[0]
        pick = jax.random.randint(pick_key, shape=(), minval=0, maxval=n_starts)
        start_idx = self._valid_start_indices[pick]

        cal_excl = self._calendar_end_exclusive[start_idx]
        # Last calendar row index is cal_excl - 1; no new opens on that row (liquidation only).
        trading_end_step = cal_excl - 1
        end_step = cal_excl + self.settlement_period_days

        # Initialize empty positions
        # Position format: [stock_idx, entry_step, entry_price, target_price, coefficient, is_active]
        positions = jnp.zeros((self.max_positions, 6), dtype=jnp.float32)

        # Create initial trading state (rng_key for observation noise)
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
            peak_capital_employed=jnp.array(0.0, dtype=jnp.float32),
            total_pnl=jnp.array(0.0, dtype=jnp.float32),
            episode_benchmark_excess=jnp.array(0.0, dtype=jnp.float32),
            rng_key=rng_key,
        )

        # Get initial observation (with optional noise when is_training)
        obs = self._get_observation(trading_state, key)

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
            action: Action array [num_investable_stocks, 3]

        Returns:
            New EnvState
        """
        env_state = state.env_state

        # Capital employed at start of step (before any closures count toward peak)
        active_mask = (env_state.positions[:, 5] > 0.5).astype(jnp.float32)
        capital_employed = jnp.sum(env_state.positions[:, 2] * active_mask)
        new_peak = jnp.maximum(env_state.peak_capital_employed, capital_employed)

        # Count active positions before updates
        had_positions = env_state.num_active_positions > 0

        # 1. Update existing positions and collect rewards
        positions, close_reward, num_wins, num_losses, total_gain, num_trades, close_pnl = (
            self._update_positions(env_state)
        )

        # 2. Discretionary market sells (same min-hold rule as target exits)
        positions, disc_r, dw, dl, dg, dt, disc_pnl = self._process_discretionary_sells(
            env_state, positions, action[:, 2]
        )
        close_reward = close_reward + disc_r
        num_wins = num_wins + dw
        num_losses = num_losses + dl
        total_gain = total_gain + dg
        num_trades = num_trades + dt

        # Accumulate raw dollar PnL
        new_total_pnl = env_state.total_pnl + close_pnl + disc_pnl

        # 3. Process new action (only during trading period)
        positions, action_reward, position_opened = jax.lax.cond(
            env_state.current_step < env_state.trading_end_step,
            lambda: self._process_action(env_state, action, positions),
            lambda: (positions, 0.0, False)
        )

        # Count active positions after updates
        num_active = jnp.sum(positions[:, 5])  # Count is_active flags

        # 4. Apply inaction penalty if no positions
        has_positions_now = num_active > 0
        inaction_pen = jax.lax.select(
            ~had_positions & ~has_positions_now,
            -self.inaction_penalty,
            0.0
        )

        # Track days with/without positions
        days_with = env_state.days_with_positions + jax.lax.select(has_positions_now, 1, 0)
        days_without = env_state.days_without_positions + jax.lax.select(~has_positions_now, 1, 0)

        # Total per-step reward (trade-level)
        step_reward = close_reward + action_reward + inaction_pen

        # 5. Move to next step
        new_step = env_state.current_step + 1
        done = new_step >= env_state.end_step

        # 6. Episode-level bonus/penalty (agent PnL vs buy-and-hold benchmark)
        episode_bonus = self._compute_episode_bonus(env_state, new_total_pnl, new_peak)
        step_reward = step_reward + jnp.where(done, episode_bonus, 0.0)
        # Persist for metrics / HoF: same episode-wide scalar folded into reward on ``done``
        new_benchmark_excess = jnp.where(done, episode_bonus, jnp.zeros_like(episode_bonus))

        # Split RNG for next step observation noise
        step_key, next_key = jax.random.split(env_state.rng_key)

        # 7. Update state
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
            peak_capital_employed=new_peak,
            total_pnl=new_total_pnl,
            episode_benchmark_excess=new_benchmark_excess,
            rng_key=next_key,
        )

        # Get new observation (with optional noise when is_training)
        obs = self._get_observation(new_env_state, step_key)

        return EnvState(
            env_state=new_env_state,
            obs=obs,
            reward=step_reward,
            done=done,
        )

    def _get_observation(self, env_state: TradingEnvState, key: chex.PRNGKey = None) -> chex.Array:
        """Get normalized observation window (Eigen2: optional multiplicative noise when is_training).

        Args:
            env_state: Current trading state
            key: JAX PRNG key for observation noise (used when is_training and observation_noise_std > 0)

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

        # Normalize (support (C,F) or (F,) norm_stats)
        norm_mean = self.norm_mean
        norm_std = self.norm_std
        if norm_mean.ndim == 1:
            norm_mean = jnp.reshape(norm_mean, (1, 1, -1))
            norm_std = jnp.reshape(norm_std, (1, 1, -1))
        else:
            norm_mean = jnp.reshape(norm_mean, (1, norm_mean.shape[0], norm_mean.shape[1]))
            norm_std = jnp.reshape(norm_std, (1, norm_std.shape[0], norm_std.shape[1]))
        normalized = (window - norm_mean) / (norm_std + 1e-8)

        # Eigen2: multiplicative observation noise for regularization when is_training
        if self.is_training and self.observation_noise_std > 0 and key is not None:
            noise = jax.random.normal(key, normalized.shape) * self.observation_noise_std
            normalized = normalized * (1.0 + noise)

        return normalized

    def _calendar_gap(self, step_a: chex.Array, step_b: chex.Array) -> chex.Array:
        """Calendar-day difference ``dates_ordinal[step_a] - dates_ordinal[step_b]``.

        Both indices are clamped to valid bounds so that out-of-range sentinels
        (e.g. ``-1`` or ``-inf`` cast to int) do not cause OOB reads.
        """
        n = self.dates_ordinal.shape[0]
        a = jnp.clip(step_a, 0, n - 1).astype(jnp.int32)
        b = jnp.clip(step_b, 0, n - 1).astype(jnp.int32)
        return self.dates_ordinal[a] - self.dates_ordinal[b]

    def _gain_reward_scalar(
        self,
        entry_price: chex.Array,
        exit_price: chex.Array,
        coefficient: chex.Array,
    ) -> chex.Array:
        """Reward for closing one lot (hurdle + conviction scaling)."""
        gain_pct = ((exit_price - entry_price) / entry_price) * 100.0
        hurdle_pct = self.hurdle_rate * 100.0
        net_gain_pct = gain_pct - hurdle_pct
        scaled_coef = jnp.power(coefficient, self.conviction_scaling_power)
        return jnp.where(
            net_gain_pct >= 0,
            scaled_coef * net_gain_pct,
            scaled_coef * net_gain_pct * self.loss_penalty_multiplier,
        )

    def _compute_episode_bonus(
        self,
        env_state: TradingEnvState,
        total_pnl: chex.Array,
        peak_capital: chex.Array,
    ) -> chex.Array:
        """Per-episode reward: agent PnL vs buy-and-hold benchmark.

        When the agent deployed capital, the benchmark is peak capital employed
        times the equal-weight average return from episode start to the
        terminal row. When peak capital is zero (idle episode), coefficient 1
        means one share per investable name bought at the episode start price:
        benchmark dollar PnL is the sum of ``end - start`` per stock (mono:
        one share of that stock).
        """
        start_step = env_state.start_step
        current_step = env_state.current_step

        start_prices = jax.lax.dynamic_slice(
            self.data_array_full[:, :, 1],
            (start_step, self.investable_start_col),
            (1, self.num_investable_stocks),
        ).squeeze(0)
        end_prices = jax.lax.dynamic_slice(
            self.data_array_full[:, :, 1],
            (current_step, self.investable_start_col),
            (1, self.num_investable_stocks),
        ).squeeze(0)

        valid = (
            jnp.isfinite(start_prices)
            & jnp.isfinite(end_prices)
            & (start_prices > 0)
        )
        returns = jnp.where(
            valid, (end_prices - start_prices) / start_prices, 0.0
        )
        n_valid = jnp.maximum(jnp.sum(valid.astype(jnp.float32)), 1.0)
        avg_return = jnp.sum(returns) / n_valid

        active_bnh = peak_capital * avg_return
        idle_bnh = jnp.sum(jnp.where(valid, end_prices - start_prices, 0.0))
        benchmark_pnl = jnp.where(peak_capital > 0, active_bnh, idle_bnh)
        return (total_pnl - benchmark_pnl) * self.episode_reward_multiplier

    def episode_buyhold_excess_usd(self, state: EnvState) -> jnp.ndarray:
        """Episode-wide buy-hold excess (scaled $): the terminal bonus term from :meth:`step`.

        This is not per-trade; it is computed once when the episode ends and added to
        ``reward`` on that step. Stored on :class:`TradingEnvState` so readers use the
        exact value from the reward decomposition.
        """
        return state.env_state.episode_benchmark_excess

    def _update_positions(
        self,
        env_state: TradingEnvState
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
        """Update all positions and close if necessary

        Args:
            env_state: Current trading state

        Returns:
            Tuple of (new_positions, total_reward, num_wins, num_losses,
                       total_gain_pct, num_trades, dollar_pnl)
        """
        positions = env_state.positions

        def update_single_position(position):
            """Update single position and check exit conditions"""
            stock_idx = position[0].astype(jnp.int32)
            entry_price = position[2]
            target_price = position[3]
            coefficient = position[4]
            is_active = position[5]

            def inactive_branch():
                return position, 0.0, 0, 0, 0.0, 0.0

            def active_branch():
                same_stock_active = (positions[:, 0] == stock_idx) & (positions[:, 5] > 0.5)
                last_buy_step = jnp.max(jnp.where(same_stock_active, positions[:, 1], -1))
                cal_days = self._calendar_gap(env_state.current_step, last_buy_step)
                can_sell_window = cal_days >= self.min_holding_period

                actual_col_idx = self.investable_start_col + stock_idx
                current_data = self.data_array_full[env_state.current_step, actual_col_idx]

                price = current_data[1]
                is_valid = jnp.isfinite(price)

                target_hit = price >= target_price
                is_last_step = env_state.current_step >= env_state.end_step - 1

                exit_on_target = can_sell_window & target_hit
                should_exit = (~is_valid) | is_last_step | exit_on_target

                exit_price = jax.lax.select(
                    exit_on_target,
                    target_price,
                    jax.lax.select(is_valid, price, entry_price)
                )

                gain_pct = ((exit_price - entry_price) / entry_price) * 100.0

                pos_reward = jax.lax.select(
                    should_exit,
                    self._gain_reward_scalar(entry_price, exit_price, coefficient),
                    0.0,
                )

                new_position = jax.lax.select(
                    should_exit,
                    jnp.zeros_like(position),
                    position
                )

                win = jax.lax.select(should_exit & (gain_pct > 0), 1, 0)
                loss = jax.lax.select(should_exit & (gain_pct <= 0), 1, 0)
                dollar_pnl = jax.lax.select(should_exit, exit_price - entry_price, 0.0)

                return (
                    new_position,
                    pos_reward,
                    win,
                    loss,
                    jax.lax.select(should_exit, gain_pct, 0.0),
                    dollar_pnl,
                )

            return jax.lax.cond(
                is_active > 0.5,
                active_branch,
                inactive_branch
            )

        results = jax.vmap(update_single_position)(positions)
        new_positions, rewards, wins, losses, gains, pnls = results

        return (
            new_positions,
            jnp.sum(rewards),
            env_state.num_wins + jnp.sum(wins),
            env_state.num_losses + jnp.sum(losses),
            env_state.total_gain_pct + jnp.sum(gains),
            env_state.num_trades + jnp.sum(jnp.where(rewards != 0, 1, 0)),
            jnp.sum(pnls),
        )

    def _discretionary_close_for_stock(
        self,
        env_state: TradingEnvState,
        positions: chex.Array,
        close_frac: chex.Array,
        stock_idx: chex.Array,
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
        """Close up to round(close_frac * n_eligible) lots (FIFO) for one stock at market."""
        same_s = (positions[:, 0].astype(jnp.int32) == stock_idx) & (positions[:, 5] > 0.5)
        has = jnp.any(same_s)
        last_buy = jnp.max(jnp.where(same_s, positions[:, 1], -jnp.inf))
        cal_days = self._calendar_gap(env_state.current_step, last_buy)
        can_sell = has & (cal_days >= self.min_holding_period)
        eligible = same_s & can_sell

        n_int = jnp.sum(eligible.astype(jnp.int32))
        k = jnp.rint(close_frac * n_int.astype(jnp.float32)).astype(jnp.int32)
        k = jnp.clip(k, jnp.array(0, dtype=jnp.int32), n_int)

        sort_key = jnp.where(eligible, positions[:, 1], jnp.inf)
        order = jnp.argsort(sort_key)

        def close_body(carry, i):
            pos, r_acc, closed, w_acc, l_acc, g_acc, t_acc, pnl_acc = carry
            idx = order[i].astype(jnp.int32)
            was_elig = eligible[idx]
            still = pos[idx, 5] > 0.5
            do = (closed < k) & was_elig & still

            actual_col_idx = self.investable_start_col + stock_idx
            current_data = self.data_array_full[env_state.current_step, actual_col_idx]
            price = current_data[1]

            entry_price = pos[idx, 2]
            coefficient = pos[idx, 4]
            valid = jnp.isfinite(price)
            exit_price = jnp.where(valid, price, entry_price)

            r = jnp.where(do, self._gain_reward_scalar(entry_price, exit_price, coefficient), 0.0)
            gain_pct = jnp.where(do, ((exit_price - entry_price) / entry_price) * 100.0, 0.0)
            w_add = jnp.where(do & (gain_pct > 0), jnp.int32(1), jnp.int32(0))
            l_add = jnp.where(do & (gain_pct <= 0), jnp.int32(1), jnp.int32(0))
            g_add = jnp.where(do, gain_pct, 0.0)
            t_add = jnp.where(do, jnp.int32(1), jnp.int32(0))
            pnl_add = jnp.where(do, exit_price - entry_price, 0.0)

            new_row = jnp.zeros(6, dtype=jnp.float32)
            pos_new = pos.at[idx].set(jnp.where(do, new_row, pos[idx]))

            closed_new = closed + jnp.where(do, jnp.int32(1), jnp.int32(0))
            return (
                pos_new,
                r_acc + r,
                closed_new,
                w_acc + w_add,
                l_acc + l_add,
                g_acc + g_add,
                t_acc + t_add,
                pnl_acc + pnl_add,
            ), None

        init_inner = (
            positions,
            jnp.array(0.0, dtype=jnp.float32),
            jnp.array(0, dtype=jnp.int32),
            jnp.array(0, dtype=jnp.int32),
            jnp.array(0, dtype=jnp.int32),
            jnp.array(0.0, dtype=jnp.float32),
            jnp.array(0, dtype=jnp.int32),
            jnp.array(0.0, dtype=jnp.float32),
        )
        idx_range = jnp.arange(self.max_positions, dtype=jnp.int32)
        (pos_out, r, _closed, w, l, g, t, pnl), _ = jax.lax.scan(close_body, init_inner, idx_range)
        return pos_out, r, w, l, g, t, pnl

    def _process_discretionary_sells(
        self,
        env_state: TradingEnvState,
        positions: chex.Array,
        close_fractions: chex.Array,
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
        """Apply per-stock discretionary market sells (after automatic position updates)."""

        def for_one_stock(carry, s):
            pos, r_tot, w_tot, l_tot, g_tot, t_tot, pnl_tot = carry
            s_int = s.astype(jnp.int32)
            frac = jnp.clip(close_fractions[s_int], 0.0, 1.0)
            npos, rr, dw, dl, dg, dt, dpnl = self._discretionary_close_for_stock(
                env_state, pos, frac, s_int
            )
            return (
                npos,
                r_tot + rr,
                w_tot + dw,
                l_tot + dl,
                g_tot + dg,
                t_tot + dt,
                pnl_tot + dpnl,
            ), None

        init = (
            positions,
            jnp.array(0.0, dtype=jnp.float32),
            jnp.array(0, dtype=jnp.int32),
            jnp.array(0, dtype=jnp.int32),
            jnp.array(0.0, dtype=jnp.float32),
            jnp.array(0, dtype=jnp.int32),
            jnp.array(0.0, dtype=jnp.float32),
        )
        stock_indices = jnp.arange(self.num_investable_stocks, dtype=jnp.int32)
        (positions_out, r, w, l, g, t, pnl), _ = jax.lax.scan(for_one_stock, init, stock_indices)
        return positions_out, r, w, l, g, t, pnl

    def _process_action(
        self,
        env_state: TradingEnvState,
        action: chex.Array,
        positions: chex.Array
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Process action and potentially open new position

        Args:
            env_state: Current trading state
            action: Action array [num_investable_stocks, 3]
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

        # Mono: Allow buying more even when holding (removed has_position check)

        actual_col_idx = self.investable_start_col + best_stock_idx
        current_data = self.data_array_full[env_state.current_step, actual_col_idx]
        entry_price = current_data[1]
        data_valid = jnp.isfinite(entry_price) & (entry_price > 0)

        # Can open position?
        can_open = valid_coeff & position_available & data_valid

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
    """Test the TradingEnv implementation (Eigen2-aligned defaults)."""
    import jax.random as random

    print("Testing TradingEnv...")

    # Create dummy data (Eigen2 skinny: 117 columns, 151 context)
    num_days = 1000
    num_columns = 117
    data_array = jnp.ones((num_days, num_columns, 5))
    data_array_full = jnp.ones((num_days, num_columns, 9))

    # Add some price variation
    key = random.PRNGKey(0)
    prices = 100.0 + random.normal(key, (num_days, num_columns)) * 10
    data_array = data_array.at[:, :, 0].set(prices)
    data_array_full = data_array_full.at[:, :, 1].set(prices)

    norm_stats = {
        'mean': jnp.zeros((num_columns, 5)),
        'std': jnp.ones((num_columns, 5))
    }

    env = TradingEnv(data_array, data_array_full, norm_stats)

    # Test reset
    key = random.PRNGKey(0)
    state = env.reset(key)

    print(f"✓ Reset successful")
    print(f"  Observation shape: {state.obs.shape}")
    print(f"  Expected: (151, 117, 5)")
    assert state.obs.shape == (151, 117, 5)

    # Test step
    action = jnp.ones((108, 3))
    action = action.at[:, 0].set(2.0)  # Coefficients
    action = action.at[:, 1].set(20.0)  # Sale targets
    action = action.at[:, 2].set(0.0)  # Discretionary close fraction

    new_state = env.step(state, action)

    print(f"✓ Step successful")
    print(f"  Observation shape: {new_state.obs.shape}")
    print(f"  Reward: {new_state.reward}")
    print(f"  Done: {new_state.done}")
    assert new_state.obs.shape == (151, 117, 5)

    # Run multiple steps
    print("\nRunning 10 steps...")
    for i in range(10):
        action = jnp.ones((108, 3)) * (1.0 + i * 0.1)
        action = action.at[:, 1].set(20.0)
        action = action.at[:, 2].set(0.0)
        new_state = env.step(new_state, action)
        print(f"  Step {i+1}: reward={new_state.reward:.4f}, "
              f"positions={new_state.env_state.num_active_positions}, "
              f"cum_reward={new_state.env_state.cumulative_reward:.4f}")

    print("\n✓ All TradingEnv tests passed!")

    return env, state


if __name__ == "__main__":
    test_trading_env()
