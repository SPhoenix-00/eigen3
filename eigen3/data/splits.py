"""Train / validation / holdout time ranges for trading data.

Holdout is the contiguous tail of rows consumed by the **last** valid calendar
episode (same rules as :class:`eigen3.environment.trading_env.TradingEnv`).
Validation is a fixed-width row window of ``ceil(multiplier * episode_trading_rows)``
rows immediately before that holdout; training uses everything before the
validation window. Episode starts within the validation slice are chosen at
random by the environment ``reset``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import numpy as np

from eigen3.environment.trading_env import TradingEnv


@dataclass(frozen=True)
class TrainValHoldoutSplit:
    """Index bounds on the full timeline (0-based, end-exclusive)."""

    num_days: int
    train_end: int
    val_start: int
    val_end: int
    holdout_start: int
    holdout_end: int
    last_episode_start: int
    last_episode_end_step: int
    validation_reserve_rows: int
    episode_trading_rows: int

    @property
    def train_rows(self) -> int:
        return self.train_end

    @property
    def val_rows(self) -> int:
        return self.val_end - self.val_start

    @property
    def holdout_rows(self) -> int:
        return self.holdout_end - self.holdout_start


def compute_train_val_holdout_split(
    num_days: int,
    dates_ordinal: np.ndarray,
    context_window_days: int,
    episode_calendar_days: int,
    settlement_period_days: int = 0,
    validation_reserve_multiplier: float = 1.5,
) -> TrainValHoldoutSplit:
    """Compute train / validation / holdout row indices.

    * **Holdout** — rows ``[holdout_start, num_days)`` are only used for the
      final episode that ends at the end of the dataset (last valid start in
      :meth:`TradingEnv._build_calendar_episode_schedule`). They must not be
      used for training or validation.
    * **Validation** — rows ``[val_start, val_end)`` with
      ``val_end == holdout_start`` and width
      ``ceil(validation_reserve_multiplier * episode_trading_rows)``, where
      ``episode_trading_rows`` is the trading row span of that last episode
      (``last_episode_end_step - last_episode_start``).
    * **Training** — rows ``[0, train_end)`` with ``train_end == val_start``.

    Args:
        num_days: Number of rows (``T``) in the panel.
        dates_ordinal: Length-``T`` calendar ordinals (see ``TradingEnv``).
        context_window_days: Observation lookback (must match env).
        episode_calendar_days: Inclusive calendar-day span of an episode (must match env).
        settlement_period_days: Extra rows after the calendar window (must match env).
        validation_reserve_multiplier: Width of the validation band is
            ``ceil(multiplier * episode_trading_rows)``.

    Returns:
        Frozen :class:`TrainValHoldoutSplit`.

    Raises:
        ValueError: If the timeline is too short or ``val_start < 0``.
    """
    if num_days < 1:
        raise ValueError("num_days must be >= 1")
    d = np.asarray(dates_ordinal, dtype=np.int64)
    if d.shape[0] != num_days:
        raise ValueError("dates_ordinal length must match num_days")

    cal_end_excl, valid_starts, _ = TradingEnv._build_calendar_episode_schedule(
        num_days=num_days,
        dates_ordinal=d,
        episode_calendar_days=int(episode_calendar_days),
        settlement_period_days=int(settlement_period_days),
        context_window_days=int(context_window_days),
    )
    valid_np = np.asarray(valid_starts)
    if valid_np.size == 0:
        raise ValueError("No valid episode starts in data (series too short or calendar mismatch)")

    last_start = int(valid_np[-1])
    cal_excl = int(np.asarray(cal_end_excl)[last_start])
    end_step = cal_excl + int(settlement_period_days)

    holdout_start = last_start - int(context_window_days) + 1
    if holdout_start < 0:
        raise ValueError("holdout_start < 0; context_window_days too large for last episode")

    episode_trading_rows = end_step - last_start
    if episode_trading_rows < 1:
        raise ValueError("episode_trading_rows < 1 (invalid schedule)")

    validation_reserve_rows = int(
        math.ceil(float(validation_reserve_multiplier) * float(episode_trading_rows))
    )
    val_end = holdout_start
    val_start = val_end - validation_reserve_rows
    if val_start < 0:
        raise ValueError(
            f"Not enough history for validation band ({validation_reserve_rows} rows) "
            f"before holdout; need at least {validation_reserve_rows} rows prior to "
            f"holdout_start={holdout_start}, got val_start={val_start}"
        )

    train_end = val_start

    # Training segment must admit at least one full episode (scheduler on prefix).
    _, train_valid, _ = TradingEnv._build_calendar_episode_schedule(
        num_days=train_end,
        dates_ordinal=d[:train_end],
        episode_calendar_days=int(episode_calendar_days),
        settlement_period_days=int(settlement_period_days),
        context_window_days=int(context_window_days),
        allow_empty=True,
    )
    if np.asarray(train_valid).size == 0:
        raise ValueError(
            f"Training prefix [0, {train_end}) has no valid episode starts; "
            "increase data length or reduce validation/holdout footprint."
        )

    # Validation segment must admit at least one valid start on its own timeline.
    vlen = val_end - val_start
    _, val_valid, _ = TradingEnv._build_calendar_episode_schedule(
        num_days=vlen,
        dates_ordinal=d[val_start:val_end],
        episode_calendar_days=int(episode_calendar_days),
        settlement_period_days=int(settlement_period_days),
        context_window_days=int(context_window_days),
        allow_empty=True,
    )
    if np.asarray(val_valid).size == 0:
        d_val = d[val_start:val_end]
        cal_span = int(d_val[-1] - d_val[0]) if vlen > 0 else 0
        raise ValueError(
            f"Validation band [{val_start}, {val_end}) ({vlen} rows, ~{cal_span} calendar days "
            f"from first to last row) has no valid episode starts for "
            f"episode_calendar_days={episode_calendar_days} after "
            f"context_window_days={context_window_days}. "
            "Try: env.validation_reserve_multiplier=2.5 (or higher), "
            "or reduce env.trading_period_days / env.context_window_days, "
            "or fix duplicate/flat dates in column A if rows are not one calendar day each."
        )

    return TrainValHoldoutSplit(
        num_days=num_days,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        holdout_start=holdout_start,
        holdout_end=num_days,
        last_episode_start=last_start,
        last_episode_end_step=end_step,
        validation_reserve_rows=validation_reserve_rows,
        episode_trading_rows=episode_trading_rows,
    )


def slice_trading_timeline(
    data_array: Any,
    data_array_full: Any,
    dates_ordinal: np.ndarray,
    start: int,
    end: int,
) -> Tuple[Any, Any, np.ndarray]:
    """Slice ``[start:end)`` along time for obs, full, and dates."""
    return (
        data_array[start:end],
        data_array_full[start:end],
        np.asarray(dates_ordinal[start:end]),
    )


def build_train_val_holdout_arrays(
    data_array: Any,
    data_array_full: Any,
    dates_ordinal: Union[np.ndarray, Any],
    split: TrainValHoldoutSplit,
) -> Dict[str, Tuple[Any, Any, np.ndarray]]:
    """Return train / val / holdout slices (JAX or NumPy arrays accepted)."""
    d = np.asarray(dates_ordinal)
    t0, t1 = split.train_end, split.val_start
    v0, v1 = split.val_start, split.val_end
    h0, h1 = split.holdout_start, split.holdout_end
    return {
        "train": slice_trading_timeline(data_array, data_array_full, d, 0, t0),
        "val": slice_trading_timeline(data_array, data_array_full, d, v0, v1),
        "holdout": slice_trading_timeline(data_array, data_array_full, d, h0, h1),
    }
