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
- **Episode length (defaults)**: `trading_period_days` + `settlement_period_days` env steps (one step = one row in `data_array`). Config defaults target a **364-step** fiscal-year-style horizon with **no separate settlement window** (`settlement_period_days: 0`); adjust in YAML if your data cadence differs.
- **Actions** (per investable stock): **coefficient**, **sale_target** (% gain target for new lots), **close_fraction** — after automatic position updates, up to that fraction of open lots on a stock may be closed **at market** (FIFO by entry step), only when the min-hold rule allows.
- **Min hold (`min_holding_period`)**: Counts **env steps** = **differences in row indices** in your series (`current_step - last_buy_step`). The env does **not** parse wall-clock dates. If each row is a **trading session only**, then e.g. `30` means **30 trading days** (typically **fewer than 30 calendar days**). For calendar-day semantics you need one row per calendar day in the data (or extend the env).
- **Exits**: Target-based sells when price reaches the lot target and min-hold is satisfied (using **last buy** on that stock across lots). Discretionary sells use the same gate. New lots allowed until `trading_end_step` (subject to `max_positions`). All open lots are **liquidated at the episode’s last step** at the current price (or entry if invalid).
- Hurdle rate, conviction scaling, optional observation noise; fixed-size position table

### Training
- **Population-based ERL**: 16 agents with genetic operators
- **DDPG updates**: Gradient-based learning with replay buffer
- **Conservative evaluation**: Multi-slice validation

## Project Structure

```
eigen3/
├── eigen3/               # Main package
│   ├── models/          # Neural network architectures
│   ├── environment/     # Trading environment
│   ├── agents/          # DDPG agent implementation
│   ├── workflows/       # Custom ERL workflows
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

# From Eigen2-exported directory
data_array, data_array_full, norm_stats = load_trading_data("data/raw")

# Or synthetic (117 columns, F=5, identity norm)
data_array, data_array_full, norm_stats = create_synthetic_data(
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

# Legacy Eigen2 npy bundle
python scripts/train.py env=trading env.data_path=data/raw

# Overrides
python scripts/train.py env.data_path=path/to/table.pkl agent=trading_erl seed=42
```

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
    agent=trading_erl \
    pop_size=32 \
    rl_updates_per_gen=64 \
    mutation_strength=0.05
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
