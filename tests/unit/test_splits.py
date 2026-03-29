"""Tests for train / validation / holdout timeline split."""

import numpy as np
import pytest

from eigen3.data.splits import (
    TrainValHoldoutSplit,
    compute_train_val_holdout_split,
    slice_trading_timeline,
)


def _compact_dates(n: int) -> np.ndarray:
    return np.arange(n, dtype=np.int64)


def test_split_contiguous_1to1_calendar():
    """1 row per calendar day: train + val + holdout partitions full series."""
    n = 250
    s = compute_train_val_holdout_split(
        num_days=n,
        dates_ordinal=_compact_dates(n),
        context_window_days=20,
        episode_calendar_days=50,
        settlement_period_days=0,
        validation_reserve_multiplier=1.5,
    )
    assert isinstance(s, TrainValHoldoutSplit)
    assert s.train_end + s.val_rows + s.holdout_rows == n
    assert s.val_start == s.train_end
    assert s.val_end == s.holdout_start
    assert s.validation_reserve_rows == int(np.ceil(1.5 * float(s.episode_trading_rows)))


def test_slice_trading_timeline():
    obs = np.arange(60 * 2 * 3, dtype=np.float32).reshape(60, 2, 3)
    full = obs * 2
    d = _compact_dates(60)
    o2, f2, d2 = slice_trading_timeline(obs, full, d, 10, 40)
    assert o2.shape[0] == 30
    assert np.array_equal(d2, np.arange(10, 40, dtype=np.int64))


def test_too_short_raises():
    with pytest.raises(ValueError, match="Not enough history"):
        compute_train_val_holdout_split(
            num_days=120,
            dates_ordinal=_compact_dates(120),
            context_window_days=20,
            episode_calendar_days=50,
            settlement_period_days=0,
            validation_reserve_multiplier=1.5,
        )
