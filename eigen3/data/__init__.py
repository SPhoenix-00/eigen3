"""Data loading and preprocessing utilities"""

from eigen3.data.data_loader import (
    StockDataLoader,
    DataConfig,
    create_synthetic_data,
    load_eigen2_data,
    load_trading_data,
)
from eigen3.data.mono_loader import MONO_DEFAULT_NUM_CHANNELS, load_mono_table
from eigen3.data.splits import (
    TrainValHoldoutSplit,
    build_train_val_holdout_arrays,
    compute_train_val_holdout_split,
    slice_trading_timeline,
)

__all__ = [
    "StockDataLoader",
    "DataConfig",
    "create_synthetic_data",
    "load_eigen2_data",
    "load_trading_data",
    "TrainValHoldoutSplit",
    "build_train_val_holdout_arrays",
    "compute_train_val_holdout_split",
    "slice_trading_timeline",
    "MONO_DEFAULT_NUM_CHANNELS",
    "load_mono_table",
]
