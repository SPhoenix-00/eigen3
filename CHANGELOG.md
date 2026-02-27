# Changelog

All notable changes to Eigen3 are documented here.

## [Unreleased]

### Added

- **EIGEN2_DELTA.md**: Delta document listing Eigen2 vs Eigen3 sync (constants, behavior, file mapping). Use it to align Eigen3 with Eigen2 changes.
- **configs/env/trading.yaml**: Environment configuration (Eigen2-aligned: 151-day context, 117 columns, holding periods, hurdle rate, observation noise).
- **load_trading_data()**: Convenience loader that dispatches to `load_eigen2_data` for directories; supports Eigen2-exported npy/npz.
- **load_eigen2_data(..., use_identity_norm=True)**: Identity normalization option for Eigen2 schema; `norm_stats` shape `(num_columns, num_features)`.

### Changed

- **Data (Eigen2 sync)**:
  - **DataConfig**: Defaults `num_columns=117`, `context_window_days=151`, `normalize=False`; added `validation_days`, `committee_holdout_days`.
  - **create_synthetic_data**: Default `num_columns=117`.
- **Environment (Eigen2 sync)**:
  - **TradingEnv**: Defaults `context_window_days=151`, `settlement_period_days=30`, `min_holding_period=20`, `max_holding_days=30`, `investable_start_col=9`, `inaction_penalty=0`; added `hurdle_rate`, `conviction_scaling_power`, `forced_exit_penalty_pct`, `observation_noise_std`, `is_training`; `rng_key` in state for observation noise.
  - Reward: min/max holding (no exit before 20 days; force exit at 30), hurdle + conviction scaling, forced-exit penalty.
  - Observation: optional multiplicative noise when `is_training`; support for 1D/2D `norm_stats`.
- **Models (Eigen2 sync)**:
  - **FeatureExtractor**: Instance normalization before CNN; default `num_columns=117`.
  - **Actor**: `investable_start_col=9`, `investable_end_col=116`, `num_columns=117`; coefficient ReLU + clamp [0, 100]; sale target clip [10, 50] (no sigmoid); output head inits (coeff bias 0.5, sale bias 30).
  - **Critic / DoubleCritic**: Default `num_columns=117`.
- **Agent**: Coefficient after exploration noise clipped to [0, 100].
- **Config**: `configs/agent/trading_erl.yaml` aligned with Eigen2 (exploration_noise 0.125, noise_decay, stock_start_idx 9, feature_extractor num_columns 117).
- **scripts/train.py**: Wired to load data (`load_trading_data` / `create_synthetic_data`), build `TradingEnv` from Hydra config, and run env reset; workflow integration documented as next step.
- **README.md**: Sync status (2025-02-27), link to EIGEN2_DELTA.md; Architecture and Quick Start updated for 117 columns and 151-day context.
- **CONVERSION_PLAN.md**: Sync status and pointer to EIGEN2_DELTA.md.
- Docstrings in `feature_extractor`, `attention`, `actor`, `critic`, `trading_env`, `trading_agent` updated to "Synced with Eigen2" and original Eigen2 file refs.

### Fixed

- Observation normalization in `TradingEnv._get_observation`: support for 2D `norm_stats` (shape `(num_columns, num_features)`).
- **Workflow (`trading_workflow.py`)**: `_crossover` referenced wrong attribute names (`actor_target_params` → `target_actor_params`, `critic_target_params` → `target_critic_params`).
- **Workflow (`trading_workflow.py`)**: `_initialize_population` used hardcoded stale dimensions `(504, 669)`; now initializes via `self.env.obs_space` / `self.env.action_space`.
- **Workflow (`trading_workflow.py`)**: `compute_actions`, `evaluate_actions`, and `loss` were called with raw `TradingNetworkParams` instead of `AgentState` wrapper; raw dicts replaced with `SampleBatch`.
- **Agent (`trading_agent.py`)**: `jax.lax.stop_gradient` was used as a context manager (invalid); replaced with `jax.lax.stop_gradient(td_target)` function call.
- **Critic (`critic.py`)**: `Critic.num_columns` default was still `669`; changed to `117` to match `DoubleCritic` and Actor.
- **Train (`scripts/train.py`)**: 10 environment parameters from YAML config were silently ignored; now all forwarded to `TradingEnv` constructor.
- **Data (`data_loader.py`)**: `StockDataLoader` methods crashed with `TypeError` when `normalize=False` because `norm_stats` was `None`; added identity fallback.
- **Tests**: All inline `test_*()` functions updated from stale `(504, 669)` to Eigen2-aligned `(151, 117)` dimensions.

---

## [0.1.0] – Initial Eigen3 port

- JAX/Flax/EvoRL port of Eigen2 (PyTorch): FeatureExtractor, Attention, Actor, Critic, TradingEnv, DDPG agent, ERL workflow.
- Hydra config, train/eval script stubs, unit tests.
