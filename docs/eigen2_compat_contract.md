# Eigen2 Compatibility Contract (Eigen3 UX Layer)

This contract defines the minimum user-facing behavior Eigen3 must preserve to provide an Eigen2-like flow, while keeping the Eigen3 JAX/Flax training core.

## Scope

- Applies to the default `python main.py` path.
- Covers CLI flow, phase telemetry, artifact placement, and evaluation bundle outputs.
- Does not require binary/algorithm parity with Eigen2 internals.

## Entry Flow Contract

Default run entry (`python main.py`) must:

1. Create and use Eigen2-style top-level folders when needed:
   - `evaluation_results/`
   - `checkpoints/`
   - `logs/`
2. Create a session tee log:
   - `evaluation_results/training_log_<YYYYMMDD_HHMMSS>.txt`
3. Print a concise startup banner with:
   - run metadata (`run_name`, seed, data path)
   - major phase markers
   - target artifact root paths
4. Execute training without requiring users to inspect Hydra output paths for core artifacts.

## Artifact Contract

Training must write a stable, discoverable artifact set rooted at repo root:

- `checkpoints/<run_name>/`
  - `best_agent.msgpack` (best known checkpoint in current run)
  - `best_agent.meta.json` (generation, score, timestamp)
  - `metrics_history.jsonl` (one JSON object per generation)
  - `run_summary.json` (final summary + resolved config pointers)
  - `hall_of_fame/` (existing HoF files)
- `last_run.json`
  - `run_name`, `run_id`, `timestamp`, `checkpoint_dir`, `best_agent_path`

Hydra output directories may still be produced, but they are not the primary user-facing contract.

## Runtime Telemetry Contract

Console/log flow must show recognizable phases:

1. `Phase 1: Data Loading`
2. `Phase 2: Environment Setup`
3. `Phase 3: Workflow Initialization`
4. `Phase 4: Training Loop`
5. `Phase 5: Finalization`

Per generation, output one compact line:

- `Generation k/N  Mean: <...>  Max: <...>  Min: <...>  Steps: <...>`

When a new run-best is detected, output:

- generation index
- score improvement
- saved best checkpoint path
- evaluation artifact paths

## Evaluation Bundle Contract

When evaluation runs, write files under `evaluation_results/` with stable names:

- `evaluation_<run_name>_<timestamp>.txt` (human summary)
- `summary_<run_name>_<timestamp>.csv` (single-row metrics table)
- `trades_<run_name>_<timestamp>.csv` (episode-level/trade-level tabular output)
- `evaluation_<run_name>_<timestamp>.json` (machine-readable payload)

## Data Path Contract

- Default data path remains `configs/env/trading_mono.yaml -> data_path`.
- Relative paths resolve from repository root for user-facing commands.
- Missing path fallback to synthetic data remains allowed, but must be clearly logged.

## Non-Goals

- Reverting Eigen3 to PyTorch `.pth` format.
- Replacing `TradingERLWorkflow` vmapped training loop.
- Requiring exact numerical parity with historical Eigen2 runs.
