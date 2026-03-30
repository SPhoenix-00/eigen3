# Eigen3: JAX-based ERL Stock Trading System

Eigen3 is a complete rewrite of the Eigen2 stock trading system, migrating from PyTorch to JAX and leveraging the EvoRL framework for efficient evolutionary reinforcement learning.

## Overview

Eigen3 combines Deep Reinforcement Learning (DDPG) with Evolutionary Algorithms (Genetic Algorithm) to train agents for stock trading. The system uses:

- **JAX** for high-performance numerical computing with automatic differentiation
- **Flax** for neural network architectures
- **EvoRL** framework for evolutionary RL workflows
- **Custom trading environment** for stock market simulation

## Architecture

### Neural Networks
- **FeatureExtractor**: CNN-LSTM architecture for temporal feature extraction
- **AttentionModule**: Multi-head attention for global context
- **Actor**: Policy network outputting **[coefficient, sale_target, close_fraction]** per stock (`close_fraction` ∈ [0, 1] for discretionary market sells)
- **Critic**: Value network (twin Q-functions for TD3-style training)

### Environment
- **TradingEnv**: JAX-native stock trading simulator (multi-stock default; mono via `configs/env/trading_mono.yaml`)
- Default data layout: 117 columns, 108 investable from column 9; **151-row** observation context window along the time axis
- **Episode length (defaults)**: Episodes are defined in **calendar days**, not row counts. Each step is still one row of `data_array`, but the horizon is chosen so the **inclusive calendar span** from the episode start row’s date through the last row of the primary window is at least **`trading_period_days`** (default **364**), using `dates_ordinal` per row (`date.toordinal()`). With trading-day-only rows or gaps, the **number of rows per episode** can be far fewer than 364 (or differ arbitrarily). Optional **`episode_calendar_days`** in config overrides that calendar span while keeping the `trading_period_days` field for logging parity. **`settlement_period_days`** adds extra **rows** after the calendar window (no new positions). See `TradingEnv` docstring and `configs/env/trading.yaml`.
- **Actions** (per investable stock): **coefficient**, **sale_target** (% gain target for new lots), **close_fraction** — after automatic position updates, up to that fraction of open lots on a stock may be closed **at market** (FIFO by entry step), only when the min-hold rule allows.
- **Min hold (`min_holding_period`)**: Counts **calendar days** using a `dates_ordinal` array (one `date.toordinal()` per data row). `30` always means **30 calendar days**, regardless of whether rows represent trading days with weekend/holiday gaps. The data pipeline parses dates from column A of the CSV (mixed `MM/DD/YYYY` and `MM-DD-YY` supported); for Eigen2 npy bundles or synthetic data, sequential ordinals (1 row = 1 day) are used as a fallback.
- **Exits**: Target-based sells when price reaches the lot target and min-hold is satisfied (using **last buy** on that stock across lots). Discretionary sells use the same gate. New lots are allowed only while `current_step < trading_end_step` (the last calendar row of the primary window is **liquidation-only**, so nothing re-opens after forced exit that day). All open lots are **liquidated at the episode’s final row** at the current price (or entry if invalid), with min-hold bypassed on that step.
- Hurdle rate, conviction scaling, optional observation noise; fixed-size position table

### Training
- **Population-based ERL**: 16 agents with genetic operators
- **DDPG updates**: Gradient-based learning with replay buffer
- **Train / validation / holdout**: The timeline is split into three contiguous row ranges. **Holdout** is the tail of rows used only by the **last** valid calendar episode (same rules as `TradingEnv`; default primary window **364** inclusive calendar days via `dates_ordinal`). **Validation** is the `ceil(env.validation_reserve_multiplier × episode_trading_rows)` rows immediately before holdout, where `episode_trading_rows` is the trading row span of that final episode; `TradingEnv.reset` picks a **random** valid start inside that slice. **Training** is everything before the validation band. Configure `validation_reserve_multiplier` in `configs/env/trading.yaml` or `trading_mono.yaml`. See `eigen3.data.splits.compute_train_val_holdout_split` and `eigen3.entrypoints.training` (used by `scripts/train.py`). For `TradingERLWorkflow`, pass `eval_env` set to a `TradingEnv` built on the validation slice only (`is_training=False`).

## Project Structure

```
eigen3/
├── main.py              # Optional: same Hydra CLI as scripts/train.py + tee log under evaluation_results/
├── eigen3/               # Main package
│   ├── models/          # Neural network architectures
│   ├── environment/     # Trading environment
│   ├── agents/          # DDPG agent implementation
│   ├── workflows/       # Custom ERL workflows
│   ├── entrypoints/     # run_training(cfg) — data, envs, HoF, TradingERLWorkflow
│   ├── data/            # Data loading and preprocessing
│   └── utils/           # Utility functions
├── configs/             # Hydra configuration files
│   ├── agent/          # Algorithm configs
│   ├── env/            # Environment configs
│   └── logging/        # Logging configs
├── scripts/             # Training and evaluation scripts
├── tests/               # Unit and integration tests
├── data/                # Data storage
│   ├── raw/            # Raw data files
│   └── processed/      # Preprocessed data
├── logs/                # Training logs
├── checkpoints/         # Model checkpoints
└── evorl/               # EvoRL framework (submodule)
```

## Installation

### 1. Install JAX with CUDA support (if using GPU)

```bash
# For CUDA 12
pip install -U "jax[cuda12]"

# For CUDA 11
pip install -U "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### 2. Install EvoRL framework

```bash
cd evorl
pip install -e .
cd ..
```

### 3. Install Eigen3

```bash
pip install -e .
```

### 4. Development installation (with testing tools)

```bash
pip install -e ".[dev]"
```

## Quick Start

### 1. Prepare Data

#### Eigen2-style bundle (multi-stock)

Schema: 117 columns, 5 observation features, 9 full features. Load from a directory containing `data_array.npy` and `data_array_full.npy`, or use synthetic data:

```python
from eigen3.data import load_trading_data, create_synthetic_data

# From Eigen2-exported directory (includes dates_ordinal per row)
data_array, data_array_full, norm_stats, dates_ordinal = load_trading_data("data/raw")

# Or synthetic (117 columns, F=5, identity norm)
data_array, data_array_full, norm_stats, dates_ordinal = create_synthetic_data(
    num_days=2000, num_columns=117, num_features_obs=5, seed=42
)
```

Use Hydra **`env=trading`** (override the default) for this layout.

#### Mono spreadsheet (single tradable price + context)

Table layout: **column A = date**, **B = price** (sole tradable series; P&L uses this channel), **C–S = 17 context columns** → **18 numeric channels**, **one feature per channel (F=1)**.

- Save as **`.csv`** (date in first column) or **`.pkl`** (either 19 columns with date first, or 18 columns with date as the index).
- `load_trading_data` detects `.pkl` / `.csv` files and returns shapes **`[T, 18, 1]`** and **`[T, 18, 9]`** (only channel 0's price slot is populated for reward calculation).
- Implementation: [`eigen3/data/mono_loader.py`](eigen3/data/mono_loader.py).

Default Hydra env is **`trading_mono`** ([`configs/env/trading_mono.yaml`](configs/env/trading_mono.yaml)). Set `env.data_path` to your file (pickle files are gitignored by default).

### 2. Train Model

From the repository root, ensure the package is importable (`pip install -e .` or `PYTHONPATH=.`) then:

```bash
# Default: env=trading_mono (mono PKL/CSV or synthetic if path missing)
python scripts/train.py

# Same Hydra overrides, plus a mirrored console log under evaluation_results/
python main.py

# Legacy Eigen2 npy bundle
python scripts/train.py env=trading env.data_path=data/raw

# Overrides (Hydra)
python scripts/train.py env.data_path=path/to/table.pkl seed=42 population.pop_size=48
```

Training runs **`TradingERLWorkflow`** for `population.total_generations` after loading data and building train/validation `TradingEnv` instances and the Hall of Fame. Implementation: [`eigen3/entrypoints/training.py`](eigen3/entrypoints/training.py).

### 3. Evaluate Model

```bash
python scripts/evaluate.py checkpoint_path=checkpoints/best_model.pkl
```

## Configuration

The project uses [Hydra](https://hydra.cc/) for configuration management. Configuration files are in `configs/`:

- `configs/agent/trading_erl.yaml` - Main training configuration
- `configs/env/trading.yaml` - Eigen2-style environment (117 cols, F=5, 108 investable)
- `configs/env/trading_mono.yaml` - Mono table (18 channels, F=1, one investable)
- `configs/config.yaml` - Defaults include `env: trading_mono`
- `configs/logging.yaml` - Logging settings

Example configuration override:

```bash
python scripts/train.py \
    population.pop_size=48 \
    population.total_generations=50 \
    population.batch_size=40 \
    seed=123
```

## Key Features

### Functional JAX Programming
- Immutable state management with PyTrees
- Pure functional environment and agent implementations
- JIT compilation for performance
- Vectorization with `jax.vmap`

### Memory Efficiency
- Gradient checkpointing (remat) for large models
- Column chunking for CNN-LSTM processing
- Efficient replay buffer implementation

### Evolutionary RL
- Population-based training with genetic operators
- Conservative multi-slice evaluation
- RL agent injection into evolutionary population
- Elite preservation and tournament selection

## Development

### Running Tests

```bash
# All tests
pytest tests/

# Unit tests only
pytest tests/unit/

# Mono loader and F=1 shapes
pytest tests/unit/test_mono_loader.py -q

# Integration tests
pytest tests/integration/

# With coverage
pytest tests/ --cov=eigen3 --cov-report=html
```

### Code Formatting

```bash
# Format code
black eigen3/ tests/ scripts/

# Sort imports
isort eigen3/ tests/ scripts/

# Lint
flake8 eigen3/ tests/ scripts/
```

## Migration from Eigen2

This project is a complete rewrite of Eigen2 (PyTorch-based) to JAX. **Eigen3 has been synced with Eigen2** (as of 2025-02-27); see [EIGEN2_DELTA.md](EIGEN2_DELTA.md) for a summary of ported behavior and constants (117-column skinny dataset, 151-day context, holding periods, hurdle/conviction rewards, etc.). Key differences:

| Aspect | Eigen2 (PyTorch) | Eigen3 (JAX) |
|--------|------------------|--------------|
| Framework | PyTorch | JAX + Flax |
| Paradigm | OOP, mutable | Functional, immutable |
| State | Model parameters | External PyTree state |
| Training | Custom loop | EvoRL workflows |
| Optimization | torch.optim | Optax |
| Vectorization | Manual loops | jax.vmap |

See [CONVERSION_PLAN.md](CONVERSION_PLAN.md) for the original migration strategy and [EIGEN2_DELTA.md](EIGEN2_DELTA.md) for the sync checklist.

## Performance

Expected improvements over Eigen2:
- **10-100x faster** environment steps (JIT + vmap)
- **2-5x faster** training (XLA compilation)
- **Better memory efficiency** (functional programming, no computation graphs)
- **Multi-GPU support** (jax.pmap)

## License

[Add your license here]

## Citation

If you use this code in your research, please cite:

```bibtex
@software{eigen3,
  title={Eigen3: JAX-based Evolutionary Reinforcement Learning for Stock Trading},
  author={[Your Name]},
  year={2025},
  url={https://github.com/[your-repo]/eigen3}
}
```

## Acknowledgments

- Built on the [EvoRL](https://github.com/EMI-Group/evorl) framework
- Inspired by Eigen2 (PyTorch implementation)
- Uses JAX, Flax, and Optax from Google Research
