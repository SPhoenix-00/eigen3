"""Unit tests for data loading and preprocessing"""

import pytest
import numpy as np
import jax.numpy as jnp
import tempfile
from dataclasses import replace
from pathlib import Path
from eigen3.data import StockDataLoader, DataConfig, create_synthetic_data, load_eigen2_data


class TestSyntheticData:
    """Test synthetic data generation"""

    def test_create_synthetic_data(self):
        """Test creating synthetic data"""
        data_obs, data_full, norm_stats, dates_ord = create_synthetic_data(
            num_days=500,
            num_columns=100,
            seed=42,
        )

        # Check shapes
        assert data_obs.shape == (500, 100, 5)
        assert data_full.shape == (500, 100, 9)

        # Check types
        assert isinstance(data_obs, jnp.ndarray)
        assert isinstance(data_full, jnp.ndarray)

        # Check normalization stats (per-column, per-feature; matches env / npy identity norm)
        assert 'mean' in norm_stats
        assert 'std' in norm_stats
        assert norm_stats['mean'].shape == (100, 5)
        assert norm_stats['std'].shape == (100, 5)

    def test_synthetic_data_deterministic(self):
        """Test that same seed produces same data"""
        data1_obs, data1_full, *_ = create_synthetic_data(seed=42)
        data2_obs, data2_full, *_ = create_synthetic_data(seed=42)

        # Should be identical
        assert jnp.allclose(data1_obs, data2_obs)
        assert jnp.allclose(data1_full, data2_full)

    def test_synthetic_data_different_seeds(self):
        """Test that different seeds produce different data"""
        data1_obs, *_ = create_synthetic_data(seed=0)
        data2_obs, *_ = create_synthetic_data(seed=999)

        # Should be different
        assert not jnp.allclose(data1_obs, data2_obs)

    def test_synthetic_data_reasonable_values(self):
        """Test that synthetic data has reasonable values"""
        data_obs, data_full, *_ = create_synthetic_data()

        # Prices should be positive
        prices_obs = data_obs[:, :, 0]  # Close prices
        # Synthetic full tensor only sets price channel on column 0 (see create_synthetic_data)
        prices_full_c0 = data_full[:, 0, 1]
        assert jnp.all(prices_obs > 0)
        assert jnp.all(prices_full_c0 > 0)

        # Volume should be positive
        volumes = data_obs[:, :, 1]
        assert jnp.all(volumes > 0)

        # Returns should be finite
        returns = data_obs[:, :, 2]
        assert jnp.all(jnp.isfinite(returns))


class TestDataLoaderInitialization:
    """Test data loader initialization"""

    def test_create_data_loader(self):
        """Test creating data loader"""
        config = DataConfig(
            data_path="data/stocks.csv",
            num_features_obs=5,
            num_features_full=9,
        )

        loader = StockDataLoader(config)

        assert loader.config == config
        assert loader.data_array_obs is None
        assert loader.data_array_full is None
        assert loader.norm_stats is None

    def test_data_config_defaults(self):
        """Test DataConfig default values"""
        config = DataConfig(data_path="test.csv")

        assert config.num_features_obs == 5
        assert config.num_features_full == 9
        assert config.num_columns == 117
        assert config.normalize is False
        assert config.context_window_days == 151
        assert config.trading_period_days == 364
        assert config.validation_reserve_multiplier == 1.5
        assert config.min_days == 1000


class TestLoadFromNumpy:
    """Test loading from numpy arrays"""

    def test_load_from_numpy(self):
        """Test loading data from numpy arrays"""
        # Create test data (long enough for default episode + validation + holdout)
        data_obs = np.random.randn(400, 100, 5)
        data_full = np.random.randn(400, 100, 9)

        config = DataConfig(
            data_path="dummy",
            min_days=300,
            context_window_days=20,
            trading_period_days=50,
            episode_calendar_days=50,
        )
        loader = StockDataLoader(config)

        loader.load_from_numpy(data_obs, data_full)

        # Check data is loaded
        assert loader.data_array_obs is not None
        assert loader.data_array_full is not None
        assert loader.norm_stats is not None

        # Check shapes
        assert loader.data_array_obs.shape == (400, 100, 5)
        assert loader.data_array_full.shape == (400, 100, 9)

    def test_load_from_numpy_validation(self):
        """Test validation of input shapes"""
        config = DataConfig(data_path="dummy")
        loader = StockDataLoader(config)

        # Wrong dimensions
        with pytest.raises(AssertionError):
            loader.load_from_numpy(
                np.random.randn(1000, 5),  # 2D instead of 3D
                np.random.randn(1000, 100, 9),
            )

        # Mismatched days
        with pytest.raises(AssertionError):
            loader.load_from_numpy(
                np.random.randn(1000, 100, 5),
                np.random.randn(500, 100, 9),  # Different number of days
            )

    def test_insufficient_data(self):
        """Test error when data is too short"""
        config = DataConfig(data_path="dummy", min_days=1000)
        loader = StockDataLoader(config)

        # Only 500 days, need 1000
        with pytest.raises(ValueError, match="Insufficient data"):
            loader.load_from_numpy(
                np.random.randn(500, 100, 5),
                np.random.randn(500, 100, 9),
            )


class TestNormalization:
    """Test normalization computation"""

    def test_compute_normalization(self):
        """Global normalization is disabled (identity norm + Instance Norm in network)."""
        data_obs = np.ones((400, 100, 5)) * 10.0
        data_full = np.random.randn(400, 100, 9)

        config = DataConfig(
            data_path="dummy",
            normalize=True,
            min_days=300,
            context_window_days=20,
            trading_period_days=50,
            episode_calendar_days=50,
        )
        loader = StockDataLoader(config)
        with pytest.raises(NotImplementedError):
            loader.load_from_numpy(data_obs, data_full)

    def test_normalization_no_division_by_zero(self):
        """normalize=True still hits disabled global path."""
        data_obs = np.ones((400, 100, 5))
        data_full = np.random.randn(400, 100, 9)

        config = DataConfig(
            data_path="dummy",
            normalize=True,
            min_days=300,
            context_window_days=20,
            trading_period_days=50,
            episode_calendar_days=50,
        )
        loader = StockDataLoader(config)
        with pytest.raises(NotImplementedError):
            loader.load_from_numpy(data_obs, data_full)

    def test_disable_normalization(self):
        """normalize=False uses identity per-(column, feature) stats."""
        data_obs = np.random.randn(400, 100, 5)
        data_full = np.random.randn(400, 100, 9)

        config = DataConfig(
            data_path="dummy",
            normalize=False,
            min_days=300,
            context_window_days=20,
            trading_period_days=50,
            episode_calendar_days=50,
        )
        loader = StockDataLoader(config)
        loader.load_from_numpy(data_obs, data_full)

        assert loader.norm_stats is not None
        assert loader.norm_stats["mean"].shape == (100, 5)
        assert loader.norm_stats["std"].shape == (100, 5)


class TestTrainValSplit:
    """Test train / validation / holdout splitting (calendar-episode tail holdout)."""

    @staticmethod
    def _compact_config(min_days: int = 250) -> DataConfig:
        return DataConfig(
            data_path="dummy",
            min_days=min_days,
            context_window_days=20,
            trading_period_days=50,
            episode_calendar_days=50,
            validation_reserve_multiplier=1.5,
        )

    def test_train_val_holdout_split(self):
        """Train, validation band, and holdout partition the timeline."""
        n = 250
        data_obs = np.random.randn(n, 100, 5)
        data_full = np.random.randn(n, 100, 9)

        loader = StockDataLoader(self._compact_config(min_days=n))
        loader.load_from_numpy(data_obs, data_full)

        sp = loader.split_info
        assert sp is not None
        assert sp.train_end + sp.val_rows + sp.holdout_rows == n
        assert loader.train_data_obs.shape[0] == sp.train_end
        assert loader.val_data_obs.shape[0] == sp.val_rows
        assert loader.holdout_data_obs.shape[0] == sp.holdout_rows

    def test_validation_reserve_multiplier(self):
        """Larger multiplier widens the validation band."""
        n = 300
        data_obs = np.random.randn(n, 100, 5)
        data_full = np.random.randn(n, 100, 9)

        base = self._compact_config(min_days=n)
        cfg_narrow = replace(base, validation_reserve_multiplier=1.5)
        cfg_wide = replace(base, validation_reserve_multiplier=2.5)

        loader_n = StockDataLoader(cfg_narrow)
        loader_n.load_from_numpy(np.array(data_obs), np.array(data_full))
        loader_w = StockDataLoader(cfg_wide)
        loader_w.load_from_numpy(data_obs, data_full)

        assert loader_w.val_data_obs.shape[0] > loader_n.val_data_obs.shape[0]

    def test_sequential_prefix_suffix(self):
        """Training is an initial prefix; val and holdout are disjoint tails."""
        n = 250
        data_obs = np.arange(n * 100 * 5).reshape(n, 100, 5).astype(float)
        data_full = np.random.randn(n, 100, 9)

        loader = StockDataLoader(self._compact_config(min_days=n))
        loader.load_from_numpy(data_obs, data_full)

        sp = loader.split_info
        assert np.array_equal(loader.train_data_obs, data_obs[: sp.train_end])
        assert np.array_equal(
            loader.val_data_obs, data_obs[sp.val_start : sp.val_end]
        )
        assert np.array_equal(
            loader.holdout_data_obs, data_obs[sp.holdout_start : sp.holdout_end]
        )


class TestJAXConversion:
    """Test conversion to JAX arrays"""

    def test_get_train_data(self):
        """Test getting training data as JAX arrays"""
        n = 250
        data_obs = np.random.randn(n, 100, 5)
        data_full = np.random.randn(n, 100, 9)

        config = DataConfig(
            data_path="dummy",
            min_days=n,
            context_window_days=20,
            trading_period_days=50,
            episode_calendar_days=50,
        )
        loader = StockDataLoader(config)
        loader.load_from_numpy(data_obs, data_full)

        # Get JAX arrays
        jax_obs, jax_full, norm_stats = loader.get_train_data()

        # Check types
        assert isinstance(jax_obs, jnp.ndarray)
        assert isinstance(jax_full, jnp.ndarray)
        assert isinstance(norm_stats['mean'], jnp.ndarray)
        assert isinstance(norm_stats['std'], jnp.ndarray)

        te = loader.split_info.train_end
        assert jax_obs.shape == (te, 100, 5)
        assert jax_full.shape == (te, 100, 9)

    def test_get_val_data(self):
        """Test getting validation data as JAX arrays"""
        n = 250
        data_obs = np.random.randn(n, 100, 5)
        data_full = np.random.randn(n, 100, 9)

        config = DataConfig(
            data_path="dummy",
            min_days=n,
            context_window_days=20,
            trading_period_days=50,
            episode_calendar_days=50,
        )
        loader = StockDataLoader(config)
        loader.load_from_numpy(data_obs, data_full)

        # Get JAX arrays
        jax_obs, jax_full, norm_stats = loader.get_val_data()

        vr = loader.split_info.val_rows
        assert jax_obs.shape == (vr, 100, 5)
        assert jax_full.shape == (vr, 100, 9)

    def test_get_holdout_data(self):
        """Holdout tail is exposed for final evaluation only."""
        n = 250
        data_obs = np.random.randn(n, 100, 5)
        data_full = np.random.randn(n, 100, 9)
        config = DataConfig(
            data_path="dummy",
            min_days=n,
            context_window_days=20,
            trading_period_days=50,
            episode_calendar_days=50,
        )
        loader = StockDataLoader(config)
        loader.load_from_numpy(data_obs, data_full)
        ho_obs, ho_full, _ = loader.get_holdout_data()
        assert ho_obs.shape[0] == loader.split_info.holdout_rows

    def test_get_all_data(self):
        """Test getting all data as JAX arrays"""
        data_obs = np.random.randn(400, 100, 5)
        data_full = np.random.randn(400, 100, 9)

        config = DataConfig(
            data_path="dummy",
            min_days=300,
            context_window_days=20,
            trading_period_days=50,
            episode_calendar_days=50,
        )
        loader = StockDataLoader(config)
        loader.load_from_numpy(data_obs, data_full)

        # Get all data
        jax_obs, jax_full, norm_stats = loader.get_all_data()

        assert jax_obs.shape == (400, 100, 5)
        assert jax_full.shape == (400, 100, 9)

    def test_get_data_before_loading(self):
        """Test error when accessing data before loading"""
        config = DataConfig(data_path="dummy")
        loader = StockDataLoader(config)

        # Should raise error
        with pytest.raises(ValueError, match="Data not loaded"):
            loader.get_train_data()

        with pytest.raises(ValueError, match="Data not loaded"):
            loader.get_val_data()

        with pytest.raises(ValueError, match="Data not loaded"):
            loader.get_all_data()


class TestSaveLoad:
    """Test saving and loading processed data"""

    def test_save_and_load_processed_data(self):
        """Test saving and loading processed data"""
        # Create data
        data_obs = np.random.randn(400, 100, 5)
        data_full = np.random.randn(400, 100, 9)

        config = DataConfig(
            data_path="dummy",
            min_days=300,
            context_window_days=20,
            trading_period_days=50,
            episode_calendar_days=50,
        )
        loader1 = StockDataLoader(config)
        loader1.load_from_numpy(data_obs, data_full)

        # Save to temporary file
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "processed_data.npz"

            loader1.save_processed_data(str(save_path))

            # Load in new loader
            loader2 = StockDataLoader(config)
            loader2.load_processed_data(str(save_path))

            # Should match
            assert np.allclose(loader1.data_array_obs, loader2.data_array_obs)
            assert np.allclose(loader1.data_array_full, loader2.data_array_full)
            assert np.allclose(loader1.norm_stats['mean'], loader2.norm_stats['mean'])
            assert np.allclose(loader1.norm_stats['std'], loader2.norm_stats['std'])

    def test_load_nonexistent_file(self):
        """Test error when loading nonexistent file"""
        config = DataConfig(data_path="dummy")
        loader = StockDataLoader(config)

        with pytest.raises(FileNotFoundError):
            loader.load_processed_data("nonexistent.npz")


class TestDataPipeline:
    """Test complete data pipeline"""

    def test_complete_pipeline(self):
        """Test complete data loading pipeline"""
        # Create synthetic data
        data_obs, data_full, *_ = create_synthetic_data(
            num_days=500,
            num_columns=100,
            seed=42,
        )

        # Convert to numpy
        data_obs_np = np.array(data_obs)
        data_full_np = np.array(data_full)

        # Create loader and load
        config = DataConfig(
            data_path="dummy",
            normalize=False,
            min_days=400,
            context_window_days=20,
            trading_period_days=50,
            episode_calendar_days=50,
        )
        loader = StockDataLoader(config)
        loader.load_from_numpy(data_obs_np, data_full_np)

        # Get training data
        train_obs, train_full, train_stats = loader.get_train_data()

        te = loader.split_info.train_end
        assert train_obs.shape == (te, 100, 5)
        assert train_full.shape == (te, 100, 9)
        assert isinstance(train_obs, jnp.ndarray)
        assert 'mean' in train_stats
        assert 'std' in train_stats

        # Get validation data
        val_obs, val_full, val_stats = loader.get_val_data()

        # Val should use same normalization stats as train
        assert jnp.allclose(train_stats['mean'], val_stats['mean'])
        assert jnp.allclose(train_stats['std'], val_stats['std'])

    def test_pipeline_with_save_load(self):
        """Test pipeline with save and reload"""
        # Create and process data
        data_obs, data_full, *_ = create_synthetic_data(num_days=400, seed=42)

        data_obs_np = np.array(data_obs)
        data_full_np = np.array(data_full)

        config = DataConfig(
            data_path="dummy",
            min_days=300,
            context_window_days=20,
            trading_period_days=50,
            episode_calendar_days=50,
        )
        loader1 = StockDataLoader(config)
        loader1.load_from_numpy(data_obs_np, data_full_np)

        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "data.npz"
            loader1.save_processed_data(str(save_path))

            # New loader, load from disk
            loader2 = StockDataLoader(config)
            loader2.load_processed_data(str(save_path))

            # Get data from both
            train1_obs, _, _ = loader1.get_train_data()
            train2_obs, _, _ = loader2.get_train_data()

            # Should match
            assert jnp.allclose(train1_obs, train2_obs)


class TestIntegrationWithEnvironment:
    """Test integration with TradingEnv"""

    def test_data_for_environment(self):
        """Test that loaded data works with TradingEnv"""
        from eigen3.environment import TradingEnv

        # Create data
        data_obs, data_full, norm_stats, _ = create_synthetic_data(
            num_days=1000,
            num_columns=669,  # Full size
            seed=42,
        )

        # Create environment
        env = TradingEnv(data_obs, data_full, norm_stats)

        # Reset environment
        import jax.random as random
        key = random.PRNGKey(0)
        state = env.reset(key)

        # Check observation shape
        assert state.obs.shape == (151, 669, 5)

        # Step environment
        action = jnp.concatenate([jnp.ones((108, 2)), jnp.zeros((108, 1))], axis=-1)
        new_state = env.step(state, action)

        # Should work without errors
        assert new_state.obs.shape == (151, 669, 5)


@pytest.mark.slow
class TestLargeScale:
    """Test with realistic large-scale data"""

    def test_full_scale_data(self):
        """Test loading full-scale data"""
        # Create full-scale synthetic data
        data_obs, data_full, norm_stats, _ = create_synthetic_data(
            num_days=2000,
            num_columns=669,
            seed=42,
        )

        # Convert to numpy
        data_obs_np = np.array(data_obs)
        data_full_np = np.array(data_full)

        # Load (default 151/364 episode params need ~2000+ days)
        config = DataConfig(data_path="dummy", min_days=1000)
        loader = StockDataLoader(config)
        loader.load_from_numpy(data_obs_np, data_full_np)

        # Get train data
        train_obs, train_full, train_stats = loader.get_train_data()
        sp = loader.split_info

        assert train_obs.shape == (sp.train_end, 669, 5)
        assert train_full.shape == (sp.train_end, 669, 9)

        val_obs, val_full, _ = loader.get_val_data()
        assert val_obs.shape == (sp.val_rows, 669, 5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
