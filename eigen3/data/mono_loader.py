"""Mono spreadsheet loader: A=date, B=price (ch0), C-S=context -> [T,18,1] obs."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import jax.numpy as jnp
import pandas as pd

logger = logging.getLogger(__name__)

MONO_DEFAULT_NUM_CHANNELS = 18


def _parse_dates_to_ordinal(series: pd.Series) -> np.ndarray:
    """Parse mixed-format date strings to int32 ordinal days.

    Handles MM/DD/YYYY, MM-DD-YY, and other formats that pandas can infer.
    """
    parsed = pd.to_datetime(series, format="mixed", dayfirst=False)
    if parsed.isna().any():
        n_bad = int(parsed.isna().sum())
        raise ValueError(
            f"_parse_dates_to_ordinal: {n_bad} unparseable date(s) in series"
        )
    ordinals = np.array(
        [d.toordinal() for d in parsed.dt.date], dtype=np.int32
    )
    return ordinals


def load_mono_table(
    filepath: str,
    num_channels: int = MONO_DEFAULT_NUM_CHANNELS,
    csv_header: Optional[int] = 0,
) -> Tuple[jnp.ndarray, jnp.ndarray, Dict, np.ndarray]:
    """Load mono spreadsheet and return (obs, full, norm_stats, dates_ordinal).

    ``dates_ordinal`` is a 1-D int32 array of length T holding the ordinal day
    number for each row, parsed from column A or the DataFrame index.
    """
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
        # Column 0 is the date column; data columns follow.
        date_series = df.iloc[:, 0]
        block = df.iloc[:, 1 : 1 + num_channels]
    elif ncols == num_channels:
        # No date column; try parsing the DataFrame index.
        date_series = pd.Series(df.index.astype(str))
        block = df.iloc[:, :num_channels]
    else:
        raise ValueError(
            f"load_mono_table: need {num_channels} or {num_channels + 1} columns, got {ncols}"
        )

    try:
        dates_ordinal = _parse_dates_to_ordinal(date_series)
    except Exception as exc:
        logger.warning(
            "load_mono_table: date parsing failed (%s). Falling back to "
            "sequential ordinals (1 row = 1 calendar day). This may cause "
            "incorrect episode boundaries and min_holding_period enforcement "
            "if the data has weekend/holiday gaps.",
            exc,
        )
        dates_ordinal = np.arange(df.shape[0], dtype=np.int32)

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

    return jnp.array(data_obs), jnp.array(data_full), norm_stats, dates_ordinal
