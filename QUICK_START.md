# Eigen3 Quick Start

Fast path for running training with the Eigen2-compatible flow.

## 1) Setup

```bash
cd /workspace
git clone https://github.com/SPhoenix-00/eigen3.git
cd eigen3
git checkout mono
git submodule update --init --recursive

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -U "jax[cuda12]"
pip install -r requirements-train.txt
pip install -U google-cloud-storage
apt-get update && apt-get install -y tmux
```

Quick GPU check:

```bash
python -c "import jax; print(jax.devices())"
```

## 2) GCS sync (required)

Set this before training:

```bash
export CLOUD_PROVIDER=gcs
export CLOUD_BUCKET=eigen3-checkpoints-ase0
export GOOGLE_APPLICATION_CREDENTIALS=/workspace/eigen3/gcs-credentials.json
```

Quick test (real read/write/delete check):

```bash
python -c "import os,uuid; from google.cloud import storage; b=os.environ['CLOUD_BUCKET']; c=storage.Client(); blob=c.bucket(b).blob(f'eigen3_healthcheck/{uuid.uuid4().hex}.txt'); blob.upload_from_string('ok'); _=blob.download_as_text(); blob.delete(); print('ok gcs rw/delete', b)"
```

## 3) Data

Default data path is `Eigen3_Processed_OUTPUT.pkl` in repo root.

Options:
- put your `.pkl` or `.csv` in repo root with that name, or
- override on run: `python main.py env.data_path=/path/to/file.pkl`

Training **requires** a valid file at `env.data_path` (supported Eigen2 bundle or mono table). If the path is missing or invalid, training raises an error (no synthetic fallback). The standalone `scripts/evaluate.py` can still use synthetic data when no file is found, for quick demos only.

## 4) Train (recommended)

```bash
cd /workspace/eigen3
source .venv/bin/activate
tmux new -s training
python main.py
```

Useful variants:

```bash
# Hydra override examples
python main.py population.pop_size=48 seed=123

# Bypass compatibility wrapper
python main.py --raw-hydra population.pop_size=48

# Direct Hydra entrypoint
python scripts/train.py population.pop_size=48
```

## 5) What gets written

With `python main.py` (Eigen2-style compat mode), you always get:

- `evaluation_results/training_log_<timestamp>.txt`
- `checkpoints/<run_name>/metrics_history.jsonl` (one JSON line per generation)
- `checkpoints/<run_name>/run_summary.json` (final)
- `checkpoints/<run_name>/hall_of_fame/` (Hall of Fame; GCS when configured)
- `last_run.json`

**Compat checkpoints** (`best_agent.msgpack`, `best_agent.meta.json`, and the `evaluation_results/evaluation_*` / `summary_*` / `trades_*` bundles on each new best) are **off by default** (`population.save_checkpoints: false` in `configs/population/default.yaml`). Enable them when you need artifacts for `scripts/evaluate.py` or offline inspection:

```bash
python main.py population.save_checkpoints=true
```

## 6) Quick verification

```bash
ls -lah evaluation_results/ | tail -n 5
cat last_run.json
```

## 7) Explicit evaluation command

If you trained with `population.save_checkpoints=true`, each new best also writes the Eigen2-style evaluation bundle under `evaluation_results/`.  
Run this only when you want an extra holdout pass (requires `best_agent.msgpack` from that setting):

```bash
python scripts/evaluate.py \
  --checkpoint_path checkpoints/<run_name>/best_agent.msgpack \
  --num_episodes 10
```

## 8) Common issues

- No `checkpoints/<run_name>/best_agent.msgpack` after training -> enable `population.save_checkpoints=true` (default is off for lighter runs)
- `ModuleNotFoundError: eigen3` -> run `pip install -e .`
- `ModuleNotFoundError: evorl` -> run `pip install -e ./evorl`
- `403 ... storage.buckets.get denied` -> use the GCS test above (no bucket metadata call), and ensure object permissions on the bucket
- CPU only in JAX -> reinstall `jax[cuda12]` and verify CUDA drivers
- GPU OOM -> lower `population.pop_size` or `population.batch_size`

## 9) Training metrics, validation, and Hall of Fame

**Population fitness** (the “Fitness: best / mean / std” block) comes from the vmapped **evaluation** phase each generation. It is the mean of the **K worst** episode total rewards per agent (`population.conservative_k` and `population.eval_episodes`), not from the collect phase. Rollouts must run until the episode hits **`done`**; the step budget is `episode_length + 128` on the eval env so calendar episodes are not truncated (truncation often shows as all zeros and no positive agents).

**Hall of Fame** entries store **validation score** (same fitness used for admission), **ROI** as **100 × (sum of validation-episode net realized $ PnL) / (max validation-episode peak capital employed)** per agent (same aggregate shown for Top 5 in the console), and **`benchmark_excess_usd`**: the episode-wide **sum of daily alpha** tracked by the environment (historical field name kept for compatibility). The console prints best/worst scores with **(±N vs equal-weight b&h)** for that value. Legacy `hall_of_fame.json` without the field loads with `0.0`.

**Validation summary (Top 5)** after each generation requires workflow metrics **`top5_indices`** and **`top5_fitness`**; the training loop then runs extra validation rollouts for those agents. If Top 5 is empty, the workflow is not emitting those keys.

**Implementation note:** `TradingEnvState.episode_alpha_sum` accumulates unscaled daily alpha across the episode; workflow logging reads this accumulator directly and does not apply an extra terminal reward rewrite.
