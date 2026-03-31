"""Tests for mono table loader (18 channels, F=1)."""

import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from eigen3.data import create_synthetic_data, load_mono_table, load_trading_data
from eigen3.environment import TradingEnv
from eigen3.models import Actor, DoubleCritic


def test_load_mono_table_with_date_column():
    n = 10
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "d": pd.date_range("2020-01-01", periods=n),
            **{f"c{i}": rng.standard_normal(n) for i in range(18)},
        }
    )
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        p = Path(f.name)
    try:
        df.to_csv(p, index=False)
        obs, full, stats, dates_ord = load_mono_table(str(p), num_channels=18, csv_header=0)
        assert obs.shape == (n, 18, 1)
        assert full.shape == (n, 18, 9)
        assert stats["mean"].shape == (18, 1)
        assert float(full[0, 0, 1]) == float(obs[0, 0, 0])
        assert dates_ord.shape == (n,)
        assert dates_ord.dtype == np.int32
    finally:
        p.unlink(missing_ok=True)


def test_load_trading_data_dispatches_csv():
    n = 5
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "d": range(n),
            **{f"c{i}": rng.standard_normal(n) for i in range(18)},
        }
    )
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.csv"
        df.to_csv(p, index=False)
        obs, full, _, dates_ord = load_trading_data(str(p))
        assert obs.shape == (n, 18, 1)
        assert dates_ord.shape == (n,)


def test_create_synthetic_f1():
    obs, full, st, dates_ord = create_synthetic_data(
        num_days=100, num_columns=18, num_features_obs=1, seed=0
    )
    assert obs.shape == (100, 18, 1)
    assert st["mean"].shape == (18, 1)
    assert dates_ord.shape == (100,)


def test_env_mono_shapes_match_config():
    obs, full, stats, _ = create_synthetic_data(
        num_days=700, num_columns=18, num_features_obs=1, seed=2
    )
    env = TradingEnv(
        obs,
        full,
        stats,
        investable_start_col=0,
        num_investable_stocks=1,
    )
    key = jax.random.PRNGKey(0)
    state = env.reset(key)
    nf = env.num_market_features + env.portfolio_obs_dim
    assert state.obs.shape == (151, 18, nf)

    actor = Actor(
        num_columns=18,
        num_features=1,
        portfolio_dim=env.portfolio_obs_dim,
        num_investable_stocks=1,
        investable_start_col=0,
        column_chunk_size=64,
        use_remat=False,
    )
    critic = DoubleCritic(
        num_columns=18,
        num_features=1,
        portfolio_dim=env.portfolio_obs_dim,
        num_investable_stocks=1,
        column_chunk_size=64,
        use_remat=False,
    )
    batch = 2
    k1, k2, k3 = jax.random.split(key, 3)
    x = jax.random.normal(k1, (batch, 151, 18, 1 + env.portfolio_obs_dim))
    a = jax.random.normal(k2, (batch, 1, 3))
    ap = actor.init(k3, x, train=False, return_attention_weights=False)
    out, _ = actor.apply(ap, x, train=False, return_attention_weights=False)
    assert out.shape == (batch, 1, 3)
    k4, _ = jax.random.split(k3)
    cp = critic.init(k4, x, a, train=False)
    q = critic.apply(cp, x, a, train=False)
    assert q.shape == (batch, 2)
