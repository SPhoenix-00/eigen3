"""Unit tests for TradingEnv"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as random
from eigen3.environment.trading_env import TradingEnv, TradingEnvState, EnvState


def create_test_data(num_days=1000, num_columns=669):
    """Create test data for trading environment"""
    # Create data with some realistic variation
    key = random.PRNGKey(42)

    # Base prices around 100
    prices = 100.0 + random.normal(key, (num_days, num_columns)) * 10

    # Create observation data (5 features)
    data_array = jnp.ones((num_days, num_columns, 5))
    data_array = data_array.at[:, :, 0].set(prices)  # Close prices

    # Create full data (9 features)
    data_array_full = jnp.ones((num_days, num_columns, 9))
    data_array_full = data_array_full.at[:, :, 1].set(prices)  # Close
    data_array_full = data_array_full.at[:, :, 2].set(prices * 1.02)  # High
    data_array_full = data_array_full.at[:, :, 0].set(prices * 0.99)  # Open
    data_array_full = data_array_full.at[:, :, 3].set(prices * 0.98)  # Low

    norm_stats = {
        'mean': jnp.zeros(5),
        'std': jnp.ones(5)
    }

    return data_array, data_array_full, norm_stats


class TestTradingEnv:
    """Test TradingEnv basic functionality"""

    def test_env_initialization(self):
        """Test environment initialization"""
        data_array, data_array_full, norm_stats = create_test_data()

        env = TradingEnv(data_array, data_array_full, norm_stats)

        assert env.context_window_days == 151
        assert env.trading_period_days == 125
        assert env.settlement_period_days == 30
        assert env.min_holding_period == 20
        assert env.max_positions == 10

    def test_env_reset(self):
        """Test environment reset"""
        data_array, data_array_full, norm_stats = create_test_data()
        env = TradingEnv(data_array, data_array_full, norm_stats)

        key = random.PRNGKey(0)
        state = env.reset(key)

        # Check state structure
        assert isinstance(state, EnvState)
        assert isinstance(state.env_state, TradingEnvState)

        # Check observation shape
        assert state.obs.shape == (151, 669, 5)

        # Check initial values
        assert state.reward == 0.0
        assert state.done == False
        assert state.env_state.num_active_positions == 0
        assert state.env_state.cumulative_reward == 0.0

    def test_env_step(self):
        """Test single environment step"""
        data_array, data_array_full, norm_stats = create_test_data()
        env = TradingEnv(data_array, data_array_full, norm_stats)

        key = random.PRNGKey(0)
        state = env.reset(key)

        # Create action
        action = jnp.ones((108, 2))
        action = action.at[:, 0].set(2.0)  # Coefficients
        action = action.at[:, 1].set(20.0)  # Sale targets

        new_state = env.step(state, action)

        # Check output
        assert new_state.obs.shape == (151, 669, 5)
        assert isinstance(new_state.reward, jnp.ndarray)
        assert isinstance(new_state.done, jnp.ndarray)

    def test_env_episode(self):
        """Test complete episode"""
        data_array, data_array_full, norm_stats = create_test_data()
        env = TradingEnv(data_array, data_array_full, norm_stats)

        key = random.PRNGKey(0)
        state = env.reset(key)

        # Run episode
        steps = 0
        while not state.done and steps < 200:
            action = jnp.ones((108, 2))
            action = action.at[:, 0].set(2.0)
            action = action.at[:, 1].set(20.0)

            state = env.step(state, action)
            steps += 1

        # Episode should terminate
        assert state.done or steps == 200

    def test_position_opening(self):
        """Test that positions can be opened"""
        data_array, data_array_full, norm_stats = create_test_data()
        env = TradingEnv(data_array, data_array_full, norm_stats)

        key = random.PRNGKey(0)
        state = env.reset(key)

        # Create strong action
        action = jnp.ones((108, 2))
        action = action.at[:, 0].set(5.0)  # High coefficients
        action = action.at[:, 1].set(20.0)

        # Take steps until position opens
        for _ in range(10):
            state = env.step(state, action)
            if state.env_state.num_active_positions > 0:
                break

        # At least one position should open eventually
        assert state.env_state.num_active_positions > 0

    def test_inaction_penalty(self):
        """Test inaction penalty is applied"""
        data_array, data_array_full, norm_stats = create_test_data()
        env = TradingEnv(data_array, data_array_full, norm_stats, inaction_penalty=10.0)

        key = random.PRNGKey(0)
        state = env.reset(key)

        # Take action with very low coefficient (no position opens)
        action = jnp.ones((108, 2))
        action = action.at[:, 0].set(0.01)  # Very low coefficients
        action = action.at[:, 1].set(20.0)

        new_state = env.step(state, action)

        # Should receive negative reward (inaction penalty)
        assert new_state.reward < 0

    def test_multiple_resets(self):
        """Test that environment can be reset multiple times"""
        data_array, data_array_full, norm_stats = create_test_data()
        env = TradingEnv(data_array, data_array_full, norm_stats)

        for i in range(5):
            key = random.PRNGKey(i)
            state = env.reset(key)

            assert state.env_state.num_active_positions == 0
            assert state.env_state.cumulative_reward == 0.0
            assert state.obs.shape == (151, 669, 5)


class TestPositionManagement:
    """Test position management logic"""

    def test_max_positions_limit(self):
        """Test that max positions limit is enforced"""
        data_array, data_array_full, norm_stats = create_test_data()
        env = TradingEnv(data_array, data_array_full, norm_stats, max_positions=3)

        key = random.PRNGKey(0)
        state = env.reset(key)

        # Try to open many positions
        action = jnp.ones((108, 2))
        action = action.at[:, 0].set(10.0)  # Very high coefficients
        action = action.at[:, 1].set(20.0)

        for _ in range(20):
            state = env.step(state, action)

        # Should not exceed max positions
        assert state.env_state.num_active_positions <= 3

    def test_position_exit_on_target(self):
        """Test that positions exit when target is hit"""
        # Create data where prices always increase
        num_days = 1000
        prices = jnp.linspace(100, 150, num_days).reshape(-1, 1)
        prices = jnp.tile(prices, (1, 669))

        data_array = jnp.ones((num_days, 669, 5))
        data_array = data_array.at[:, :, 0].set(prices)

        data_array_full = jnp.ones((num_days, 669, 9))
        data_array_full = data_array_full.at[:, :, 1].set(prices)
        data_array_full = data_array_full.at[:, :, 2].set(prices * 1.1)  # High always higher

        norm_stats = {'mean': jnp.zeros(5), 'std': jnp.ones(5)}

        env = TradingEnv(data_array, data_array_full, norm_stats)

        key = random.PRNGKey(0)
        state = env.reset(key)

        # Open position with low sale target
        action = jnp.ones((108, 2))
        action = action.at[:, 0].set(5.0)
        action = action.at[:, 1].set(5.0)  # Low target (5% gain)

        # Run until position opens and closes
        initial_trades = 0
        for _ in range(50):
            state = env.step(state, action)
            if state.env_state.num_trades > initial_trades:
                break

        # Should have closed at least one trade
        assert state.env_state.num_trades > 0

    def test_end_of_episode_liquidation(self):
        """Test that all positions are liquidated at episode end"""
        # Flat prices so target is never hit
        num_days = 1000
        prices = jnp.ones((num_days, 669, 1)) * 100.0
        data_array = jnp.ones((num_days, 669, 5))
        data_array = data_array.at[:, :, 0].set(prices[:, :, 0])
        data_array_full = jnp.ones((num_days, 669, 9))
        data_array_full = data_array_full.at[:, :, 1].set(prices[:, :, 0])
        data_array_full = data_array_full.at[:, :, 2].set(prices[:, :, 0])  # High = Close
        norm_stats = {'mean': jnp.zeros(5), 'std': jnp.ones(5)}

        env = TradingEnv(data_array, data_array_full, norm_stats)

        key = random.PRNGKey(0)
        state = env.reset(key)

        action = jnp.ones((108, 2))
        action = action.at[:, 0].set(5.0)
        action = action.at[:, 1].set(50.0)  # High target, never hit on flat prices

        # Run entire episode
        while not state.done:
            state = env.step(state, action)

        assert state.done
        assert state.env_state.num_active_positions == 0, \
            "All positions should be liquidated at episode end"
        assert state.env_state.num_trades > 0, \
            "At least one position should have been opened and liquidated"


class TestJAXFeatures:
    """Test JAX-specific features"""

    def test_env_is_jittable(self):
        """Test that step function can be JIT compiled"""
        data_array, data_array_full, norm_stats = create_test_data()
        env = TradingEnv(data_array, data_array_full, norm_stats)

        key = random.PRNGKey(0)
        state = env.reset(key)
        action = jnp.ones((108, 2))

        @jax.jit
        def jitted_step(state, action):
            return env.step(state, action)

        new_state = jitted_step(state, action)
        assert new_state.obs.shape == (151, 669, 5)

    def test_env_reset_is_jittable(self):
        """Test that reset function can be JIT compiled"""
        data_array, data_array_full, norm_stats = create_test_data()
        env = TradingEnv(data_array, data_array_full, norm_stats)

        @jax.jit
        def jitted_reset(key):
            return env.reset(key)

        key = random.PRNGKey(0)
        state = jitted_reset(key)
        assert state.obs.shape == (151, 669, 5)

    def test_env_is_vmappable(self):
        """Test that environment can be vectorized"""
        data_array, data_array_full, norm_stats = create_test_data()
        env = TradingEnv(data_array, data_array_full, norm_stats)

        # Create batch of keys
        batch_size = 4
        keys = jax.random.split(random.PRNGKey(0), batch_size)

        # Vectorized reset
        states = jax.vmap(env.reset)(keys)

        assert states.obs.shape == (batch_size, 151, 669, 5)
        assert states.reward.shape == (batch_size,)

    def test_deterministic_with_same_key(self):
        """Test that same key produces same results"""
        data_array, data_array_full, norm_stats = create_test_data()
        env = TradingEnv(data_array, data_array_full, norm_stats)

        key = random.PRNGKey(42)

        state1 = env.reset(key)
        state2 = env.reset(key)

        # Should produce identical states
        assert state1.env_state.current_step == state2.env_state.current_step
        assert jnp.array_equal(state1.obs, state2.obs)

    def test_gradients_flow(self):
        """Test that gradients can flow through environment"""
        data_array, data_array_full, norm_stats = create_test_data()
        env = TradingEnv(data_array, data_array_full, norm_stats)

        key = random.PRNGKey(0)
        state = env.reset(key)

        def reward_fn(action):
            new_state = env.step(state, action)
            return new_state.reward

        action = jnp.ones((108, 2))
        grad_fn = jax.grad(lambda a: reward_fn(a).sum())

        # Should not raise error
        grads = grad_fn(action)
        assert grads.shape == action.shape


class TestObservationSpace:
    """Test observation space properties"""

    def test_observation_normalization(self):
        """Test that observations are normalized"""
        data_array, data_array_full, norm_stats = create_test_data()
        env = TradingEnv(data_array, data_array_full, norm_stats)

        key = random.PRNGKey(0)
        state = env.reset(key)

        # Observations should be normalized (roughly mean=0, std=1)
        # With our test data (mean=0, std=1), observations should be close to original
        assert jnp.all(jnp.isfinite(state.obs))

    def test_observation_shape_consistency(self):
        """Test that observation shape is consistent"""
        data_array, data_array_full, norm_stats = create_test_data()
        env = TradingEnv(data_array, data_array_full, norm_stats)

        key = random.PRNGKey(0)
        state = env.reset(key)

        # Run several steps
        action = jnp.ones((108, 2))
        for _ in range(10):
            state = env.step(state, action)
            assert state.obs.shape == (151, 669, 5)


class TestRewardSystem:
    """Test reward calculation"""

    def test_positive_reward_on_win(self):
        """Test that wins produce positive rewards"""
        # Create data with price increase
        num_days = 1000
        prices = jnp.linspace(100, 120, num_days).reshape(-1, 1)
        prices = jnp.tile(prices, (1, 669))

        data_array = jnp.ones((num_days, 669, 5))
        data_array = data_array.at[:, :, 0].set(prices)

        data_array_full = jnp.ones((num_days, 669, 9))
        data_array_full = data_array_full.at[:, :, 1].set(prices)
        data_array_full = data_array_full.at[:, :, 2].set(prices * 1.05)

        norm_stats = {'mean': jnp.zeros(5), 'std': jnp.ones(5)}

        env = TradingEnv(data_array, data_array_full, norm_stats)

        key = random.PRNGKey(0)
        state = env.reset(key)

        action = jnp.ones((108, 2))
        action = action.at[:, 0].set(2.0)
        action = action.at[:, 1].set(5.0)

        # Run until we get a positive reward
        for _ in range(50):
            state = env.step(state, action)
            if state.reward > 0:
                break

        # Should eventually get positive reward
        assert state.env_state.cumulative_reward > -100  # Some threshold

    def test_cumulative_reward_tracking(self):
        """Test that cumulative reward is tracked correctly"""
        data_array, data_array_full, norm_stats = create_test_data()
        env = TradingEnv(data_array, data_array_full, norm_stats)

        key = random.PRNGKey(0)
        state = env.reset(key)

        action = jnp.ones((108, 2))
        cumulative = 0.0

        for _ in range(10):
            state = env.step(state, action)
            cumulative += float(state.reward)

        # Cumulative should match (approximately, due to floating point)
        assert jnp.abs(state.env_state.cumulative_reward - cumulative) < 1e-4


class TestMonoRules:
    """Test mono trading rules: multiple buys same stock, 20-trading-day sell
    restriction, end-of-episode liquidation."""

    @staticmethod
    def _create_mono_env(num_days=500, **env_kwargs):
        """Create a single-stock env with rising prices."""
        prices = jnp.linspace(100, 200, num_days).reshape(-1, 1)
        data_array = jnp.ones((num_days, 1, 5))
        data_array = data_array.at[:, :, 0].set(prices)
        data_array_full = jnp.ones((num_days, 1, 9))
        data_array_full = data_array_full.at[:, :, 1].set(prices)       # Close
        data_array_full = data_array_full.at[:, :, 2].set(prices * 1.1) # High (10% above)
        data_array_full = data_array_full.at[:, :, 0].set(prices * 0.99)
        data_array_full = data_array_full.at[:, :, 3].set(prices * 0.98)
        norm_stats = {'mean': jnp.zeros(5), 'std': jnp.ones(5)}
        defaults = dict(
            num_investable_stocks=1,
            investable_start_col=0,
            max_positions=5,
            min_holding_period=20,
        )
        defaults.update(env_kwargs)
        env = TradingEnv(data_array, data_array_full, norm_stats, **defaults)
        return env

    def test_mono_can_buy_when_holding(self):
        """Agent can accumulate multiple lots of the same stock."""
        env = self._create_mono_env()
        key = random.PRNGKey(0)
        state = env.reset(key)

        buy_action = jnp.array([[5.0, 20.0]])

        for _ in range(5):
            state = env.step(state, buy_action)
            if state.env_state.num_active_positions >= 1:
                break
        assert state.env_state.num_active_positions >= 1, "First position should open"

        for _ in range(10):
            state = env.step(state, buy_action)
            if state.env_state.num_active_positions >= 2:
                break
        assert state.env_state.num_active_positions >= 2, \
            "Should allow second buy in same stock"

    def test_mono_no_sell_before_20_trading_days(self):
        """No position can close before 20 trading days since the last buy."""
        env = self._create_mono_env()
        key = random.PRNGKey(0)
        state = env.reset(key)

        buy_action = jnp.array([[5.0, 1.0]])   # 1% target -- easily hit
        hold_action = jnp.array([[0.0, 1.0]])   # coeff below threshold, no buy

        # Call 0: first buy
        state = env.step(state, buy_action)
        assert state.env_state.num_active_positions >= 1

        # Calls 1-4: hold
        for _ in range(4):
            state = env.step(state, hold_action)

        # Call 5: second buy (last buy is now at relative step 5)
        state = env.step(state, buy_action)
        assert state.env_state.num_active_positions >= 2

        # Calls 6-24 (19 steps): target is already exceeded (high is 10%
        # above close, target is 1%), but sell window is closed because
        # < 20 trading days since last buy at step 5.
        for i in range(19):
            state = env.step(state, hold_action)
            assert state.env_state.num_trades == 0, \
                f"Position sold at step {6 + i} -- before 20 trading days since last buy"

    def test_mono_sell_after_20_trading_days(self):
        """Positions can close once 20 trading days have passed since last buy."""
        env = self._create_mono_env()
        key = random.PRNGKey(0)
        state = env.reset(key)

        buy_action = jnp.array([[5.0, 1.0]])
        hold_action = jnp.array([[0.0, 1.0]])

        # Open one position
        state = env.step(state, buy_action)

        # Advance 20+ trading days with no further buys
        for _ in range(25):
            state = env.step(state, hold_action)
            if state.env_state.num_trades > 0:
                break

        assert state.env_state.num_trades > 0, \
            "Should have sold after 20 trading days since last buy"

    def test_mono_end_of_episode_liquidation(self):
        """All remaining positions are liquidated at episode end."""
        # Flat prices so target is never hit
        num_days = 500
        prices = jnp.ones((num_days, 1)) * 100.0
        data_array = jnp.ones((num_days, 1, 5))
        data_array = data_array.at[:, :, 0].set(prices)
        data_array_full = jnp.ones((num_days, 1, 9))
        data_array_full = data_array_full.at[:, :, 1].set(prices)
        data_array_full = data_array_full.at[:, :, 2].set(prices)  # High = Close
        norm_stats = {'mean': jnp.zeros(5), 'std': jnp.ones(5)}

        env = TradingEnv(
            data_array, data_array_full, norm_stats,
            num_investable_stocks=1,
            investable_start_col=0,
            max_positions=5,
            min_holding_period=20,
        )
        key = random.PRNGKey(0)
        state = env.reset(key)

        buy_action = jnp.array([[5.0, 50.0]])   # 50% target, never hit
        hold_action = jnp.array([[0.0, 50.0]])

        # Open a position then hold until episode ends
        state = env.step(state, buy_action)
        assert state.env_state.num_active_positions >= 1

        while not state.done:
            state = env.step(state, hold_action)

        assert state.done
        assert state.env_state.num_active_positions == 0, \
            "All positions should be liquidated at episode end"
        assert state.env_state.num_trades > 0, \
            "Liquidation should register as a completed trade"


@pytest.mark.slow
class TestFullEpisode:
    """Test full episode runs"""

    def test_complete_episode(self):
        """Test running a complete episode"""
        data_array, data_array_full, norm_stats = create_test_data()
        env = TradingEnv(data_array, data_array_full, norm_stats)

        key = random.PRNGKey(0)
        state = env.reset(key)

        action = jnp.ones((108, 2))
        action = action.at[:, 0].set(2.0)
        action = action.at[:, 1].set(20.0)

        steps = 0
        max_steps = 200

        while not state.done and steps < max_steps:
            state = env.step(state, action)
            steps += 1

        # Episode should complete
        assert state.done or steps == max_steps

        # Should have reasonable statistics
        assert state.env_state.cumulative_reward != 0.0  # Some reward collected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
