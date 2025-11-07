"""Data loading and preprocessing for stock trading data

This module handles loading stock market data from various sources and
converting it into JAX arrays suitable for the TradingEnv.
"""

from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import jax.numpy as jnp
import pandas as pd


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing

    Args:
        data_path: Path to data directory or file
        num_features_obs: Number of features for observations (default 5)
        num_features_full: Number of features for full data (default 9)
        num_columns: Number of stock columns (default 669)
        normalize: Whether to normalize the data
        train_split: Fraction of data for training (rest is validation)
        min_days: Minimum number of days required
    """
    data_path: str
    num_features_obs: int = 5
    num_features_full: int = 9
    num_columns: int = 669
    normalize: bool = True
    train_split: float = 0.8
    min_days: int = 1000


class StockDataLoader:
    """Loader for stock market data

    This class handles loading, preprocessing, and normalization of stock
    market data for the trading environment.
    """

    def __init__(self, config: DataConfig):
        """Initialize data loader

        Args:
            config: Data configuration
        """
        self.config = config
        self.data_path = Path(config.data_path)

        # Data arrays
        self.data_array_obs = None      # [days, columns, 5] - observation features
        self.data_array_full = None     # [days, columns, 9] - full features
        self.norm_stats = None          # Normalization statistics

        # Train/validation split
        self.train_data_obs = None
        self.train_data_full = None
        self.val_data_obs = None
        self.val_data_full = None

    def load_from_csv(self, csv_path: Optional[str] = None) -> None:
        """Load data from CSV file(s)

        Expected CSV format:
        - Columns: date, ticker, open, high, low, close, volume, ...
        - One row per ticker per day

        Args:
            csv_path: Path to CSV file (uses config.data_path if None)
        """
        if csv_path is None:
            csv_path = self.config.data_path

        csv_path = Path(csv_path)

        if not csv_path.exists():
            raise FileNotFoundError(f"Data file not found: {csv_path}")

        # Load CSV
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)

        # Convert to arrays
        self._process_dataframe(df)

    def load_from_parquet(self, parquet_path: Optional[str] = None) -> None:
        """Load data from Parquet file

        Args:
            parquet_path: Path to Parquet file (uses config.data_path if None)
        """
        if parquet_path is None:
            parquet_path = self.config.data_path

        parquet_path = Path(parquet_path)

        if not parquet_path.exists():
            raise FileNotFoundError(f"Data file not found: {parquet_path}")

        # Load Parquet
        print(f"Loading data from {parquet_path}...")
        df = pd.read_parquet(parquet_path)

        # Convert to arrays
        self._process_dataframe(df)

    def load_from_numpy(
        self,
        data_obs: np.ndarray,
        data_full: np.ndarray,
    ) -> None:
        """Load data from numpy arrays

        Args:
            data_obs: Observation data [days, columns, 5]
            data_full: Full data [days, columns, 9]
        """
        print("Loading data from numpy arrays...")

        # Validate shapes
        assert data_obs.ndim == 3, f"data_obs must be 3D, got shape {data_obs.shape}"
        assert data_full.ndim == 3, f"data_full must be 3D, got shape {data_full.shape}"
        assert data_obs.shape[0] == data_full.shape[0], "Number of days must match"
        assert data_obs.shape[1] == data_full.shape[1], "Number of columns must match"

        num_days, num_columns, _ = data_obs.shape

        if num_days < self.config.min_days:
            raise ValueError(
                f"Insufficient data: {num_days} days, need at least {self.config.min_days}"
            )

        # Store as numpy arrays (will convert to JAX later)
        self.data_array_obs = data_obs
        self.data_array_full = data_full

        # Compute normalization statistics
        if self.config.normalize:
            self._compute_normalization_stats()

        # Split train/val
        self._split_train_val()

        print(f"Loaded {num_days} days, {num_columns} columns")

    def _process_dataframe(self, df: pd.DataFrame) -> None:
        """Process pandas DataFrame into arrays

        Args:
            df: DataFrame with stock data
        """
        # Expected columns
        required_cols = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']

        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Sort by date and ticker
        df = df.sort_values(['date', 'ticker'])

        # Get unique dates and tickers
        dates = df['date'].unique()
        tickers = df['ticker'].unique()

        num_days = len(dates)
        num_tickers = len(tickers)

        if num_days < self.config.min_days:
            raise ValueError(
                f"Insufficient data: {num_days} days, need at least {self.config.min_days}"
            )

        print(f"Processing {num_days} days, {num_tickers} tickers...")

        # Create ticker to index mapping
        ticker_to_idx = {ticker: idx for idx, ticker in enumerate(tickers)}

        # Initialize arrays
        # Observation features: close, volume, returns, volatility, etc.
        data_obs = np.zeros((num_days, num_tickers, self.config.num_features_obs))

        # Full features: open, close, high, low, volume, etc.
        data_full = np.zeros((num_days, num_tickers, self.config.num_features_full))

        # Fill arrays
        for day_idx, date in enumerate(dates):
            day_data = df[df['date'] == date]

            for _, row in day_data.iterrows():
                ticker = row['ticker']
                ticker_idx = ticker_to_idx[ticker]

                # Extract price and volume data
                open_price = row['open']
                high_price = row['high']
                low_price = row['low']
                close_price = row['close']
                volume = row['volume']

                # Compute additional features
                # Returns (will be NaN for first day)
                if day_idx > 0:
                    prev_close = data_full[day_idx - 1, ticker_idx, 1]  # Previous close
                    returns = (close_price - prev_close) / prev_close if prev_close > 0 else 0.0
                else:
                    returns = 0.0

                # Volatility (high - low) / close
                volatility = (high_price - low_price) / close_price if close_price > 0 else 0.0

                # Observation features (5 features used during trading)
                data_obs[day_idx, ticker_idx, 0] = close_price
                data_obs[day_idx, ticker_idx, 1] = volume
                data_obs[day_idx, ticker_idx, 2] = returns
                data_obs[day_idx, ticker_idx, 3] = volatility
                data_obs[day_idx, ticker_idx, 4] = (close_price - open_price) / open_price if open_price > 0 else 0.0

                # Full features (9 features including OHLC for reward calculation)
                data_full[day_idx, ticker_idx, 0] = open_price
                data_full[day_idx, ticker_idx, 1] = close_price
                data_full[day_idx, ticker_idx, 2] = high_price
                data_full[day_idx, ticker_idx, 3] = low_price
                data_full[day_idx, ticker_idx, 4] = volume
                data_full[day_idx, ticker_idx, 5] = returns
                data_full[day_idx, ticker_idx, 6] = volatility
                data_full[day_idx, ticker_idx, 7] = (high_price - open_price) / open_price if open_price > 0 else 0.0
                data_full[day_idx, ticker_idx, 8] = (open_price - low_price) / open_price if open_price > 0 else 0.0

        # Replace NaNs and Infs
        data_obs = np.nan_to_num(data_obs, nan=0.0, posinf=0.0, neginf=0.0)
        data_full = np.nan_to_num(data_full, nan=0.0, posinf=0.0, neginf=0.0)

        self.data_array_obs = data_obs
        self.data_array_full = data_full

        # Compute normalization
        if self.config.normalize:
            self._compute_normalization_stats()

        # Split train/val
        self._split_train_val()

        print("Data processing complete.")

    def _compute_normalization_stats(self) -> None:
        """Compute mean and std for normalization"""
        # Compute statistics over all days and columns
        # Shape: [num_features]
        mean = np.mean(self.data_array_obs, axis=(0, 1))
        std = np.std(self.data_array_obs, axis=(0, 1))

        # Avoid division by zero
        std = np.where(std < 1e-8, 1.0, std)

        self.norm_stats = {
            'mean': mean,
            'std': std,
        }

        print(f"Normalization stats computed:")
        print(f"  Mean: {mean}")
        print(f"  Std:  {std}")

    def _split_train_val(self) -> None:
        """Split data into training and validation sets"""
        num_days = self.data_array_obs.shape[0]
        split_idx = int(num_days * self.config.train_split)

        # Sequential split (train first, then val)
        self.train_data_obs = self.data_array_obs[:split_idx]
        self.train_data_full = self.data_array_full[:split_idx]
        self.val_data_obs = self.data_array_obs[split_idx:]
        self.val_data_full = self.data_array_full[split_idx:]

        print(f"Data split: {split_idx} train days, {num_days - split_idx} val days")

    def get_train_data(self) -> Tuple[jnp.ndarray, jnp.ndarray, Dict]:
        """Get training data as JAX arrays

        Returns:
            Tuple of (data_obs, data_full, norm_stats)
        """
        if self.train_data_obs is None:
            raise ValueError("Data not loaded. Call load_from_* first.")

        # Convert to JAX arrays
        data_obs = jnp.array(self.train_data_obs)
        data_full = jnp.array(self.train_data_full)

        # Convert normalization stats
        norm_stats = {
            'mean': jnp.array(self.norm_stats['mean']),
            'std': jnp.array(self.norm_stats['std']),
        }

        return data_obs, data_full, norm_stats

    def get_val_data(self) -> Tuple[jnp.ndarray, jnp.ndarray, Dict]:
        """Get validation data as JAX arrays

        Returns:
            Tuple of (data_obs, data_full, norm_stats)
        """
        if self.val_data_obs is None:
            raise ValueError("Data not loaded. Call load_from_* first.")

        # Convert to JAX arrays
        data_obs = jnp.array(self.val_data_obs)
        data_full = jnp.array(self.val_data_full)

        # Convert normalization stats (same as training)
        norm_stats = {
            'mean': jnp.array(self.norm_stats['mean']),
            'std': jnp.array(self.norm_stats['std']),
        }

        return data_obs, data_full, norm_stats

    def get_all_data(self) -> Tuple[jnp.ndarray, jnp.ndarray, Dict]:
        """Get all data (train + val) as JAX arrays

        Returns:
            Tuple of (data_obs, data_full, norm_stats)
        """
        if self.data_array_obs is None:
            raise ValueError("Data not loaded. Call load_from_* first.")

        # Convert to JAX arrays
        data_obs = jnp.array(self.data_array_obs)
        data_full = jnp.array(self.data_array_full)

        # Convert normalization stats
        norm_stats = {
            'mean': jnp.array(self.norm_stats['mean']),
            'std': jnp.array(self.norm_stats['std']),
        }

        return data_obs, data_full, norm_stats

    def save_processed_data(self, output_path: str) -> None:
        """Save processed data to disk

        Args:
            output_path: Path to save processed data
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Saving processed data to {output_path}...")

        # Save as npz
        np.savez(
            output_path,
            data_obs=self.data_array_obs,
            data_full=self.data_array_full,
            norm_mean=self.norm_stats['mean'],
            norm_std=self.norm_stats['std'],
        )

        print("Data saved successfully.")

    def load_processed_data(self, input_path: str) -> None:
        """Load previously processed data from disk

        Args:
            input_path: Path to processed data file
        """
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Processed data file not found: {input_path}")

        print(f"Loading processed data from {input_path}...")

        # Load npz
        data = np.load(input_path)

        self.data_array_obs = data['data_obs']
        self.data_array_full = data['data_full']
        self.norm_stats = {
            'mean': data['norm_mean'],
            'std': data['norm_std'],
        }

        # Split train/val
        self._split_train_val()

        print(f"Loaded {self.data_array_obs.shape[0]} days, {self.data_array_obs.shape[1]} columns")


def create_synthetic_data(
    num_days: int = 1000,
    num_columns: int = 669,
    seed: int = 42,
) -> Tuple[jnp.ndarray, jnp.ndarray, Dict]:
    """Create synthetic stock data for testing

    Args:
        num_days: Number of trading days
        num_columns: Number of stock columns
        seed: Random seed

    Returns:
        Tuple of (data_obs, data_full, norm_stats)
    """
    np.random.seed(seed)

    # Generate random walk prices
    initial_prices = 100.0 + np.random.randn(num_columns) * 10
    prices = np.zeros((num_days, num_columns))
    prices[0] = initial_prices

    # Random walk with drift
    for day in range(1, num_days):
        returns = np.random.randn(num_columns) * 0.02 + 0.0001  # 2% daily vol, small positive drift
        prices[day] = prices[day - 1] * (1 + returns)

    # Create observation features
    data_obs = np.zeros((num_days, num_columns, 5))
    data_obs[:, :, 0] = prices  # Close prices

    # Volume (random)
    data_obs[:, :, 1] = np.random.lognormal(10, 1, (num_days, num_columns))

    # Returns
    returns = np.zeros((num_days, num_columns))
    returns[1:] = (prices[1:] - prices[:-1]) / prices[:-1]
    data_obs[:, :, 2] = returns

    # Volatility (random)
    data_obs[:, :, 3] = np.abs(np.random.randn(num_days, num_columns) * 0.01)

    # Intraday return (random)
    data_obs[:, :, 4] = np.random.randn(num_days, num_columns) * 0.005

    # Create full features (OHLC + volume + derived)
    data_full = np.zeros((num_days, num_columns, 9))
    data_full[:, :, 0] = prices * (1 + np.random.randn(num_days, num_columns) * 0.001)  # Open
    data_full[:, :, 1] = prices  # Close
    data_full[:, :, 2] = prices * (1 + np.abs(np.random.randn(num_days, num_columns)) * 0.01)  # High
    data_full[:, :, 3] = prices * (1 - np.abs(np.random.randn(num_days, num_columns)) * 0.01)  # Low
    data_full[:, :, 4] = data_obs[:, :, 1]  # Volume
    data_full[:, :, 5] = data_obs[:, :, 2]  # Returns
    data_full[:, :, 6] = data_obs[:, :, 3]  # Volatility
    data_full[:, :, 7] = np.random.randn(num_days, num_columns) * 0.005  # High-Open
    data_full[:, :, 8] = np.random.randn(num_days, num_columns) * 0.005  # Open-Low

    # Normalization stats
    norm_stats = {
        'mean': jnp.zeros(5),
        'std': jnp.ones(5),
    }

    # Convert to JAX
    data_obs = jnp.array(data_obs)
    data_full = jnp.array(data_full)

    return data_obs, data_full, norm_stats


def load_eigen2_data(eigen2_data_path: str) -> Tuple[jnp.ndarray, jnp.ndarray, Dict]:
    """Load data from eigen2 format

    Args:
        eigen2_data_path: Path to eigen2 data directory

    Returns:
        Tuple of (data_obs, data_full, norm_stats)
    """
    data_path = Path(eigen2_data_path)

    # Look for expected files
    obs_path = data_path / "data_array.npy"
    full_path = data_path / "data_array_full.npy"
    stats_path = data_path / "norm_stats.npz"

    if not obs_path.exists():
        raise FileNotFoundError(f"Observation data not found: {obs_path}")
    if not full_path.exists():
        raise FileNotFoundError(f"Full data not found: {full_path}")

    print(f"Loading eigen2 data from {data_path}...")

    # Load arrays
    data_obs = np.load(obs_path)
    data_full = np.load(full_path)

    # Load normalization stats if available
    if stats_path.exists():
        stats = np.load(stats_path)
        norm_stats = {
            'mean': jnp.array(stats['mean']),
            'std': jnp.array(stats['std']),
        }
    else:
        # Compute from data
        mean = np.mean(data_obs, axis=(0, 1))
        std = np.std(data_obs, axis=(0, 1))
        std = np.where(std < 1e-8, 1.0, std)

        norm_stats = {
            'mean': jnp.array(mean),
            'std': jnp.array(std),
        }

    # Convert to JAX
    data_obs = jnp.array(data_obs)
    data_full = jnp.array(data_full)

    print(f"Loaded {data_obs.shape[0]} days, {data_obs.shape[1]} columns")

    return data_obs, data_full, norm_stats
