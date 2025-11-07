"""Unit tests for data loading and preprocessing"""

import pytest
import numpy as np
import jax.numpy as jnp
import tempfile
from pathlib import Path
from eigen3.data import StockDataLoader, DataConfig, create_synthetic_data, load_eigen2_data


class TestSyntheticData:
    """Test synthetic data generation"""

    def test_create_synthetic_data(self):
        """Test creating synthetic data"""
        data_obs, data_full, norm_stats = create_synthetic_data(
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

        # Check normalization stats
        assert 'mean' in norm_stats
        assert 'std' in norm_stats
        assert norm_stats['mean'].shape == (5,)
        assert norm_stats['std'].shape == (5,)

    def test_synthetic_data_deterministic(self):
        """Test that same seed produces same data"""
        data1_obs, data1_full, _ = create_synthetic_data(seed=42)
        data2_obs, data2_full, _ = create_synthetic_data(seed=42)

        # Should be identical
        assert jnp.allclose(data1_obs, data2_obs)
        assert jnp.allclose(data1_full, data2_full)

    def test_synthetic_data_different_seeds(self):
        """Test that different seeds produce different data"""
        data1_obs, _, _ = create_synthetic_data(seed=0)
        data2_obs, _, _ = create_synthetic_data(seed=999)

        # Should be different
        assert not jnp.allclose(data1_obs, data2_obs)

    def test_synthetic_data_reasonable_values(self):
        """Test that synthetic data has reasonable values"""
        data_obs, data_full, _ = create_synthetic_data()

        # Prices should be positive
        prices_obs = data_obs[:, :, 0]  # Close prices
        prices_full = data_full[:, :, 1]  # Close prices
        assert jnp.all(prices_obs > 0)
        assert jnp.all(prices_full > 0)

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
        assert config.num_columns == 669
        assert config.normalize == True
        assert config.train_split == 0.8
        assert config.min_days == 1000


class TestLoadFromNumpy:
    """Test loading from numpy arrays"""

    def test_load_from_numpy(self):
        """Test loading data from numpy arrays"""
        # Create test data
        data_obs = np.random.randn(1000, 100, 5)
        data_full = np.random.randn(1000, 100, 9)

        config = DataConfig(data_path="dummy", min_days=500)
        loader = StockDataLoader(config)

        loader.load_from_numpy(data_obs, data_full)

        # Check data is loaded
        assert loader.data_array_obs is not None
        assert loader.data_array_full is not None
        assert loader.norm_stats is not None

        # Check shapes
        assert loader.data_array_obs.shape == (1000, 100, 5)
        assert loader.data_array_full.shape == (1000, 100, 9)

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
        """Test normalization statistics computation"""
        # Create data with known statistics
        data_obs = np.ones((1000, 100, 5)) * 10.0
        data_obs[:, :, 0] = 100.0  # Close price = 100
        data_obs[:, :, 1] = 50.0   # Volume = 50

        data_full = np.random.randn(1000, 100, 9)

        config = DataConfig(data_path="dummy", normalize=True, min_days=500)
        loader = StockDataLoader(config)
        loader.load_from_numpy(data_obs, data_full)

        # Check stats computed
        assert loader.norm_stats is not None
        assert 'mean' in loader.norm_stats
        assert 'std' in loader.norm_stats

        # Mean should be close to [100, 50, 10, 10, 10]
        mean = loader.norm_stats['mean']
        assert np.isclose(mean[0], 100.0, atol=0.1)
        assert np.isclose(mean[1], 50.0, atol=0.1)

    def test_normalization_no_division_by_zero(self):
        """Test that normalization handles zero std"""
        # Create data with zero variation in one feature
        data_obs = np.ones((1000, 100, 5))
        data_obs[:, :, 0] = 100.0  # Constant value
        data_full = np.random.randn(1000, 100, 9)

        config = DataConfig(data_path="dummy", normalize=True, min_days=500)
        loader = StockDataLoader(config)
        loader.load_from_numpy(data_obs, data_full)

        # Std should be 1.0 where it would be zero
        std = loader.norm_stats['std']
        assert np.all(std > 0)  # No zeros

    def test_disable_normalization(self):
        """Test disabling normalization"""
        data_obs = np.random.randn(1000, 100, 5)
        data_full = np.random.randn(1000, 100, 9)

        config = DataConfig(data_path="dummy", normalize=False, min_days=500)
        loader = StockDataLoader(config)
        loader.load_from_numpy(data_obs, data_full)

        # Norm stats should still be None
        assert loader.norm_stats is None


class TestTrainValSplit:
    """Test train/validation splitting"""

    def test_train_val_split(self):
        """Test splitting data into train and validation"""
        data_obs = np.random.randn(1000, 100, 5)
        data_full = np.random.randn(1000, 100, 9)

        config = DataConfig(data_path="dummy", train_split=0.8, min_days=500)
        loader = StockDataLoader(config)
        loader.load_from_numpy(data_obs, data_full)

        # Check split sizes
        assert loader.train_data_obs.shape[0] == 800
        assert loader.val_data_obs.shape[0] == 200
        assert loader.train_data_full.shape[0] == 800
        assert loader.val_data_full.shape[0] == 200

    def test_different_split_ratios(self):
        """Test different split ratios"""
        data_obs = np.random.randn(1000, 100, 5)
        data_full = np.random.randn(1000, 100, 9)

        # 70/30 split
        config = DataConfig(data_path="dummy", train_split=0.7, min_days=500)
        loader = StockDataLoader(config)
        loader.load_from_numpy(data_obs, data_full)

        assert loader.train_data_obs.shape[0] == 700
        assert loader.val_data_obs.shape[0] == 300

    def test_sequential_split(self):
        """Test that split is sequential (not random)"""
        # Create data with increasing values
        data_obs = np.arange(1000 * 100 * 5).reshape(1000, 100, 5).astype(float)
        data_full = np.random.randn(1000, 100, 9)

        config = DataConfig(data_path="dummy", train_split=0.8, min_days=500)
        loader = StockDataLoader(config)
        loader.load_from_numpy(data_obs, data_full)

        # Train should be first 800 days
        assert np.array_equal(loader.train_data_obs, data_obs[:800])

        # Val should be last 200 days
        assert np.array_equal(loader.val_data_obs, data_obs[800:])


class TestJAXConversion:
    """Test conversion to JAX arrays"""

    def test_get_train_data(self):
        """Test getting training data as JAX arrays"""
        data_obs = np.random.randn(1000, 100, 5)
        data_full = np.random.randn(1000, 100, 9)

        config = DataConfig(data_path="dummy", min_days=500)
        loader = StockDataLoader(config)
        loader.load_from_numpy(data_obs, data_full)

        # Get JAX arrays
        jax_obs, jax_full, norm_stats = loader.get_train_data()

        # Check types
        assert isinstance(jax_obs, jnp.ndarray)
        assert isinstance(jax_full, jnp.ndarray)
        assert isinstance(norm_stats['mean'], jnp.ndarray)
        assert isinstance(norm_stats['std'], jnp.ndarray)

        # Check shapes
        assert jax_obs.shape == (800, 100, 5)
        assert jax_full.shape == (800, 100, 9)

    def test_get_val_data(self):
        """Test getting validation data as JAX arrays"""
        data_obs = np.random.randn(1000, 100, 5)
        data_full = np.random.randn(1000, 100, 9)

        config = DataConfig(data_path="dummy", min_days=500)
        loader = StockDataLoader(config)
        loader.load_from_numpy(data_obs, data_full)

        # Get JAX arrays
        jax_obs, jax_full, norm_stats = loader.get_val_data()

        # Check shapes
        assert jax_obs.shape == (200, 100, 5)
        assert jax_full.shape == (200, 100, 9)

    def test_get_all_data(self):
        """Test getting all data as JAX arrays"""
        data_obs = np.random.randn(1000, 100, 5)
        data_full = np.random.randn(1000, 100, 9)

        config = DataConfig(data_path="dummy", min_days=500)
        loader = StockDataLoader(config)
        loader.load_from_numpy(data_obs, data_full)

        # Get all data
        jax_obs, jax_full, norm_stats = loader.get_all_data()

        # Should have all 1000 days
        assert jax_obs.shape == (1000, 100, 5)
        assert jax_full.shape == (1000, 100, 9)

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
        data_obs = np.random.randn(1000, 100, 5)
        data_full = np.random.randn(1000, 100, 9)

        config = DataConfig(data_path="dummy", min_days=500)
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
        data_obs, data_full, _ = create_synthetic_data(
            num_days=1200,
            num_columns=100,
            seed=42,
        )

        # Convert to numpy
        data_obs_np = np.array(data_obs)
        data_full_np = np.array(data_full)

        # Create loader and load
        config = DataConfig(
            data_path="dummy",
            train_split=0.8,
            normalize=True,
            min_days=1000,
        )
        loader = StockDataLoader(config)
        loader.load_from_numpy(data_obs_np, data_full_np)

        # Get training data
        train_obs, train_full, train_stats = loader.get_train_data()

        # Check everything works
        assert train_obs.shape == (960, 100, 5)
        assert train_full.shape == (960, 100, 9)
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
        data_obs, data_full, _ = create_synthetic_data(seed=42)

        data_obs_np = np.array(data_obs)
        data_full_np = np.array(data_full)

        config = DataConfig(data_path="dummy", min_days=500)
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
        data_obs, data_full, norm_stats = create_synthetic_data(
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
        assert state.obs.shape == (504, 669, 5)

        # Step environment
        action = jnp.ones((108, 2))
        new_state = env.step(state, action)

        # Should work without errors
        assert new_state.obs.shape == (504, 669, 5)


@pytest.mark.slow
class TestLargeScale:
    """Test with realistic large-scale data"""

    def test_full_scale_data(self):
        """Test loading full-scale data"""
        # Create full-scale synthetic data
        data_obs, data_full, norm_stats = create_synthetic_data(
            num_days=2000,
            num_columns=669,
            seed=42,
        )

        # Convert to numpy
        data_obs_np = np.array(data_obs)
        data_full_np = np.array(data_full)

        # Load
        config = DataConfig(data_path="dummy", min_days=1000)
        loader = StockDataLoader(config)
        loader.load_from_numpy(data_obs_np, data_full_np)

        # Get train data
        train_obs, train_full, train_stats = loader.get_train_data()

        # Check shapes
        assert train_obs.shape == (1600, 669, 5)
        assert train_full.shape == (1600, 669, 9)

        # Check memory usage is reasonable (data should fit in memory)
        # This is mainly to ensure JAX conversion doesn't explode memory
        val_obs, val_full, _ = loader.get_val_data()
        assert val_obs.shape == (400, 669, 5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
