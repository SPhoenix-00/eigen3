# Eigen3 Quick Start Guide

## Installation (5 minutes)

```bash
# 1. Install JAX with CUDA support
pip install -U "jax[cuda12]"

# 2. Install EvoRL framework
cd evorl && pip install -e . && cd ..

# 3. Install Eigen3 in development mode
pip install -e ".[dev]"

# 4. Verify installation
pytest tests/unit/test_example.py -v
```

## Project Layout

```
eigen3/
├── eigen3/              # Source code (implement here)
│   ├── models/         # Neural networks (FeatureExtractor, Actor, Critic)
│   ├── environment/    # Trading environment (JAX-native)
│   ├── agents/         # DDPG agent
│   └── workflows/      # ERL workflow
├── configs/             # Hydra configs (already set up)
├── scripts/             # train.py, evaluate.py (templates ready)
├── tests/               # Unit and integration tests
└── data/                # Put your data here
```

## Development Commands

```bash
# Format code
make format

# Run tests
make test

# Clean build artifacts
make clean
```

## Training (Once Implemented)

```bash
# Basic training
python scripts/train.py

# With custom parameters
python scripts/train.py agent.pop_size=32 seed=42

# Multi-run with different seeds
python scripts/train.py -m seed=1,2,3,4,5
```

## What's Next?

We're currently at **Phase 1: JAX Conversion**. Next components to implement:

1. `eigen3/models/feature_extractor.py` - CNN-LSTM feature extraction
2. `eigen3/models/attention.py` - Multi-head attention
3. `eigen3/models/actor.py` - Policy network
4. `eigen3/models/critic.py` - Value network

See [CONVERSION_PLAN.md](CONVERSION_PLAN.md) for detailed implementation guides.

## Key Files to Know

- `CONVERSION_PLAN.md` - Detailed conversion strategy with code examples
- `PROJECT_SETUP.md` - Complete project structure documentation
- `configs/agent/trading_erl.yaml` - Algorithm hyperparameters
- `configs/env/trading.yaml` - Environment configuration

## Architecture Overview

```
Eigen2 (PyTorch)  →  Eigen3 (JAX + EvoRL)
──────────────────────────────────────────
PyTorch nn.Module  →  Flax nn.Module
torch.optim        →  optax
Custom training    →  EvoRL workflows
Mutable state      →  Immutable PyTrees
.backward()        →  jax.grad()
Manual loops       →  jax.vmap, jax.scan
```

## EvoRL Integration Pattern

```python
# 1. Define network with Flax
class YourNetwork(nn.Module):
    @nn.compact
    def __call__(self, x):
        ...

# 2. Create Agent with EvoRL interface
class YourAgent(Agent):
    def init(self, obs_space, action_space, key) -> AgentState:
        ...
    def compute_actions(self, agent_state, sample_batch, key):
        ...

# 3. Create Workflow
class YourWorkflow(ERLWorkflowBase):
    def step(self, state) -> tuple[Metrics, State]:
        ...
```

## Getting Help

- **Conversion questions**: See [CONVERSION_PLAN.md](CONVERSION_PLAN.md)
- **Project structure**: See [PROJECT_SETUP.md](PROJECT_SETUP.md)
- **EvoRL framework**: See `evorl/` documentation
- **JAX basics**: https://jax.readthedocs.io/
- **Flax guide**: https://flax.readthedocs.io/

## Common JAX Patterns

```python
# Immutable updates
state = state.replace(value=new_value)

# Random numbers (need explicit key)
key, subkey = jax.random.split(key)
x = jax.random.normal(subkey, (10,))

# JIT compilation
@jax.jit
def fast_function(x):
    return jnp.sum(x**2)

# Vectorization
batched_fn = jax.vmap(fn, in_axes=0)

# Gradients
grad_fn = jax.grad(loss_fn)
```

## Tips for Eigen2 → Eigen3 Conversion

1. **Replace in-place ops**: `x += 1` → `x = x + 1`
2. **Use jax.numpy**: `import jax.numpy as jnp`
3. **Thread keys**: Pass `key` explicitly everywhere
4. **Use jax.lax.cond**: Instead of Python `if` in JIT'd functions
5. **Use jax.lax.scan**: Instead of Python `for` loops
6. **PyTree state**: All state must be PyTree-compatible

Ready to start? Begin with `eigen3/models/feature_extractor.py`! 🚀
