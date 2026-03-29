"""Mono spreadsheet loader: A=date, B=price (ch0), C-S=context -> [T,18,1] obs."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import jax.numpy as jnp
import pandas as pd

MONO_DEFAULT_NUM_CHANNELS = 18


def load_mono_table(
    filepath: str,
    num_channels: int = MONO_DEFAULT_NUM_CHANNELS,
    csv_header: Optional[int] = 0,
) -> Tuple[jnp.ndarray, jnp.ndarray, Dict]:
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Mono table file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in (".pkl", ".pickle"):
        df = pd.read_pickle(path)
    elif suffix == ".csv":
        df = pd.read_csv(path, header=csv_header)
    else:
        raise ValueError(f"load_mono_table: expected .pkl or .csv, got {suffix}")

    ncols = df.shape[1]
    if ncols >= num_channels + 1:
        block = df.iloc[:, 1 : 1 + num_channels]
    elif ncols == num_channels:
        block = df.iloc[:, :num_channels]
    else:
        raise ValueError(
            f"load_mono_table: need {num_channels} or {num_channels + 1} columns, got {ncols}"
        )

    values = np.asarray(block, dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if values.shape[1] != num_channels:
        raise ValueError(
            f"load_mono_table: expected {num_channels} data columns, got {values.shape[1]}"
        )

    t = values.shape[0]
    data_obs = values.astype(np.float32)[:, :, np.newaxis]

    # The environment reads price from data_full[step, col, 1].
    # Only channel 0 (the tradable price) is populated.
    data_full = np.zeros((t, num_channels, 9), dtype=np.float32)
    data_full[:, 0, 1] = data_obs[:, 0, 0]

    norm_stats = {
        "mean": jnp.zeros((num_channels, 1), dtype=jnp.float32),
        "std": jnp.ones((num_channels, 1), dtype=jnp.float32),
    }

    print(f"Loaded mono table from {path}: {t} rows, {num_channels} channels, F=1")

    return jnp.array(data_obs), jnp.array(data_full), norm_stats
