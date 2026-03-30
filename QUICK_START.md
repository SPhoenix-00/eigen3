# Eigen3 — RunPod Quick-Start Guide

End-to-end instructions for spinning up a GPU pod and running Eigen3 training
(`main.py` / `scripts/train.py` -> `TradingERLWorkflow`), with optional GCS-backed
Hall of Fame persistence. **Section 7** is the canonical description of the
Eigen2-compatible default flow.

---

## 1. Create the RunPod Instance

| Setting | Recommended value |
|---------|-------------------|
| **GPU** | 1x **H100 SXM 80 GB** (best throughput) or A100 80 GB (or A6000 48 GB for budget runs) |
| **Template** | RunPod PyTorch 2.x / CUDA 12.x (any Linux image with CUDA 12 drivers) |
| **Disk** | 50 GB container + 20 GB volume (`/workspace`) |
| **Python** | 3.10 or 3.11 (pre-installed in most templates) |

After the pod boots, open a **Web Terminal** or SSH in:

```bash
ssh root@<pod-ip> -p <port> -i ~/.ssh/id_ed25519
```

---

## 2. Clone the Repo

```bash
cd /workspace
git clone https://github.com/SPhoenix-00/eigen3.git
cd eigen3
git checkout mono          # active development branch
git submodule update --init --recursive   # pulls evorl/
```

---

## 3. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
```

---

## 4. Install JAX with CUDA 12

RunPod images ship CUDA 12.x drivers. Install the matching `jaxlib`:

```bash
pip install -U "jax[cuda12]"
```

Verify GPU visibility:

```bash
python -c "import jax; print(jax.devices())"
# Expected: [CudaDevice(id=0)]
```

---

## 5. Install Dependencies

```bash
# EvoRL (editable) + Eigen3 + all training extras in one shot:
pip install -r requirements-train.txt
```

This installs:
- **evorl** (editable from `./evorl`) — pulls `optax`, `chex`, `orbax`, `gymnasium`, `hydra-core`, etc.
- **eigen3** (editable from `.`) — the main package.
- **flax**, **jaxtyping**, **tensorboard**, **matplotlib**, **seaborn**, **tqdm**, etc.

> **Tip:** If you also want `pytest`, `black`, `flake8`, and `mypy` for
> development, use `pip install -r requirements-dev.txt` instead.

---

## 6. Upload Your Data

Eigen3 expects a processed data file (pickle or CSV). Upload it to the pod:

```bash
# From your local machine:
scp -P <port> Eigen3_Processed_OUTPUT.pkl root@<pod-ip>:/workspace/eigen3/

# Or use RunPod's file manager in the web UI.
```

Training reads **`env.data_path`** from `configs/env/trading_mono.yaml`
(default: `Eigen3_Processed_OUTPUT.pkl` in the **repo root**). Hydra resolves
paths from your real working directory (the run’s output folder does not break
relative paths).

- Put the pickle (or CSV) in the repo root next to `scripts/`, **or** edit
  `data_path` in `configs/env/trading_mono.yaml`, **or** override once:
  `python main.py env.data_path=/path/to/table.pkl` (same as `scripts/train.py`).

If you build the table on the pod with **`python process_eigen_data.py`**
(from the repo root), the script **rewrites** `data_path` in
`configs/env/trading_mono.yaml` after each successful PKL write (including
custom `--output-pkl` names).

If no data file is found, training falls back to synthetic data so you can
smoke-test the full stack without uploading anything.

---

## 7. Training flow (Eigen2-compatible default)

### Commands

| Command | What it does |
|---------|----------------|
| **`python main.py`** | Recommended default: runs the Eigen2-compatible orchestration path and mirrors the **full console** to **`evaluation_results/training_log_<timestamp>.txt`**. |
| **`python main.py env.data_path=… population.pop_size=48 …`** | Any [Hydra override](https://hydra.cc/docs/advanced/override_grammar/basic/) after the script name (same grammar as below). |
| **`python main.py --raw-hydra ...`** | Escape hatch: bypass compatibility orchestration and use raw Hydra-centric behavior. |
| **`python scripts/train.py …`** | Direct Hydra entrypoint. To mirror to a file yourself, set **`EIGEN3_TRAINING_LOG=/path/to/log.txt`** before running. |

### What actually runs

1. **`main.py`** sets compatibility environment variables and creates root-level artifact folders (`evaluation_results/`, `checkpoints/`, `logs/`), then dispatches to training.
2. **Hydra** still loads `configs/config.yaml` (defaults: `env=trading_mono`, `agent=trading_erl`, `population=default`, `logging=default`) and keeps its own run metadata under `outputs/<date>/<time>/`.
3. **`scripts/train.py`** calls **`eigen3.entrypoints.training.run_training(cfg)`**, which executes phase-style training (`Data Loading` -> `Environment Setup` -> `Workflow Initialization` -> `Training Loop` -> `Finalization`) and writes Eigen2-style run artifacts under `checkpoints/<run_name>/`.

### What you should see

- Startup run summary and explicit phase banners.
- Each generation: `Generation k/N  Mean: ...  Max: ...  Min: ...  Steps: ...` plus optional HoF logs.
- On each new run-best: best checkpoint save + evaluation bundle in `evaluation_results/`.
- Finish: last-generation fitness, run summary metadata, and Hydra output directory reference.

### Where artifacts go

| Location | Contents |
|----------|-----------|
| **`evaluation_results/training_log_*.txt`** | Full console copy when using `main.py` |
| **`evaluation_results/evaluation_<run>_<timestamp>.(txt/json)`** | Evaluation-on-improvement outputs |
| **`evaluation_results/summary_<run>_<timestamp>.csv`** | Aggregate evaluation metrics |
| **`evaluation_results/trades_<run>_<timestamp>.csv`** | Episode/trade-level evaluation rows |
| **`checkpoints/<run_name>/`** | `best_agent.msgpack`, `best_agent.meta.json`, `metrics_history.jsonl`, `run_summary.json`, `hall_of_fame/` |
| **`last_run.json`** | Resume/discovery pointer to the latest run artifacts |
| **`outputs/<date>/<time>/`** | Hydra-only metadata (`.hydra/config.yaml`, overrides, Hydra logs) |

### Weights & Biases and TensorBoard

- **W&B is off by default** (`logging.use_wandb: false`, `wandb_mode: disabled` in **`configs/logging/default.yaml`**). You do **not** need `wandb login` for a normal run. To opt in later: `logging.use_wandb=true logging.wandb_mode=online` once the training loop writes to W&B.
- **TensorBoard** is enabled in config (`logging.use_tensorboard: true`); the run end prints a suggested `--logdir`. Per-generation metrics are currently **console / log file**; extend the workflow if you want scalars in TensorBoard.

### Eigen2 habits -> Eigen3 equivalents

| Eigen2 habit | Eigen3 equivalent |
|--------------|-------------------|
| `python main.py` as primary command | `python main.py` (default compatibility mode) |
| look in `evaluation_results/training_log_*.txt` | same path and naming |
| expect run-scoped checkpoints | `checkpoints/<run_name>/...` |
| track latest run pointer | `last_run.json` |
| expect CSV/text eval outputs for best improvements | `evaluation_results/summary_*.csv`, `trades_*.csv`, `evaluation_*.txt` |

---

## 8. Configure GCS for Hall of Fame (Optional but Recommended)

The Hall of Fame persists the best agents to a Google Cloud Storage bucket so
they survive across pods and runs.

### a. Upload your service-account key

Place the key in the **repo root** on the pod (same folder as `scripts/`):

```bash
# scp or paste your key file:
scp -P <port> gcs-credentials.json root@<pod-ip>:/workspace/eigen3/
```

### b. Set environment variables

Add these to your shell (or append to `~/.bashrc` for persistence):

```bash
export CLOUD_PROVIDER=gcs
export CLOUD_BUCKET=eigen3-checkpoints-ase0
export GOOGLE_APPLICATION_CREDENTIALS=/workspace/eigen3/gcs-credentials.json
```

If `gcs-credentials.json` sits at `/workspace/eigen3/gcs-credentials.json`, Eigen3’s
`CloudSync.from_env()` will pick it up automatically when `GOOGLE_APPLICATION_CREDENTIALS`
is unset — but setting the variable explicitly (as above) is still recommended.

> Without `CLOUD_PROVIDER=gcs` and a bucket, the HoF still works — it just saves locally under
> `checkpoints/hall_of_fame/` and won't sync across machines.

### c. Verify connectivity

```bash
python -c "
from eigen3.erl.cloud_sync import CloudSync
cs = CloudSync.from_env()
print('Provider:', cs.provider)
print('Bucket:', cs.bucket_name)
"
```

---

## 9. Run Training (copy-paste)

```bash
# Activate env if not already:
source /workspace/eigen3/.venv/bin/activate
cd /workspace/eigen3

# Default: loads Eigen3_Processed_OUTPUT.pkl from repo root (trading_mono.yaml).
# Missing file → synthetic data. See §7 for the full pipeline.
python main.py

# Same run, no evaluation_results/ tee:
python scripts/train.py

# Different data file:
python main.py env.data_path=/path/to/other.pkl

# Smaller population / reproducibility:
python main.py \
    population.pop_size=48 \
    population.hof_capacity=10 \
    seed=123
```

(Section **7** describes `main.py` vs `scripts/train.py`, artifacts, and compatibility defaults.)

---

## 10. Run Evaluation (Holdout)

After training produces checkpoints:

```bash
cd /workspace/eigen3
# Defaults: --data_path Eigen3_Processed_OUTPUT.pkl (repo root)
python scripts/evaluate.py \
    --checkpoint_path checkpoints/<run_name>/best_agent.msgpack \
    --num_episodes 10
```

Use `--data_path /path/to/file.pkl` only if your table is not the default name or location.

---

## 11. Monitor Training

### Console and files

- **Terminal**: phase banners, per-generation fitness, and HoF messages.
- **`evaluation_results/training_log_*.txt`**: full copy when started with `main.py`.
- **`checkpoints/<run_name>/metrics_history.jsonl`**: per-generation machine-readable metrics.
- **Hydra**: `outputs/<date>/<time>/` for resolved config and Hydra logs.

### TensorBoard

If `logging.use_tensorboard` is true, the training footer suggests a logdir (often under **`logs/tensorboard/<run_name>`**). From the repo root:

```bash
tensorboard --logdir logs/ --bind_all --port 6006 &
```

Expose port **6006** in RunPod’s **HTTP Service Ports** to open the UI. (Scalars from the ERL loop are not written automatically yet unless you add them.)

---

## 12. Detach Long Runs

Use `tmux` or `screen` so training survives SSH disconnects:

```bash
tmux new -s eigen3
source /workspace/eigen3/.venv/bin/activate
cd /workspace/eigen3
python main.py
# Ctrl-b d  to detach
# tmux attach -t eigen3  to reconnect
```

---

## 13. File Layout Reference

```
eigen3/
├── main.py                      # Default Eigen2-compatible orchestrator (raw mode: --raw-hydra)
├── gcs-credentials.json         # GCS service-account key (gitignored locally; upload on the pod)
├── evaluation_results/          # training logs + evaluation bundles
├── configs/
│   ├── config.yaml              # top-level Hydra config
│   ├── agent/trading_erl.yaml   # DDPG + network architecture
│   ├── env/trading_mono.yaml    # mono-stock environment
│   ├── env/trading.yaml         # multi-stock (Eigen2 layout)
│   ├── population/default.yaml  # population, ERL sizing, HoF, gauntlet
│   └── logging/default.yaml
├── eigen3/
│   ├── models/                  # Actor, DoubleCritic, FeatureExtractor
│   ├── environment/             # TradingEnv (JAX-native)
│   ├── agents/                  # TradingAgent (DDPG)
│   ├── workflows/               # TradingERLWorkflow
│   ├── erl/                     # Hall of Fame + CloudSync (NEW)
│   ├── data/                    # data_loader, mono_loader, splits
│   ├── entrypoints/             # run_training (Hydra → workflow)
│   └── utils/
├── evorl/                       # EvoRL framework (git submodule)
├── scripts/
│   ├── train.py                 # Hydra training entry point
│   └── evaluate.py              # Holdout evaluation
├── process_eigen_data.py        # CSV → indicators; updates env `data_path` in trading_mono.yaml
├── requirements-data.txt        # numpy + pandas only (data prep)
├── requirements-train.txt       # full training stack
├── requirements-dev.txt         # training + test/lint tooling
├── checkpoints/
│   └── <run_name>/              # run-scoped artifacts + hall_of_fame/
└── last_run.json                # latest run pointer
```

---

## 14. GPU-Vectorized Workflow (H100 / high-VRAM GPUs)

`TradingERLWorkflow` processes **all agents in the population simultaneously**
via `jax.vmap`, replacing sequential Python loops with batched GPU kernels.
On an H100 SXM this yields 10–50x wall-clock speed-up over a naive
one-agent-at-a-time loop.

### How it works

| Phase | What happens on GPU |
|-------|---------------------|
| **Collect experience** | `vmap(env.step)` + `vmap(agent.compute_actions)` — one kernel per time-step for the entire population |
| **Gradient updates** | `vmap(agent.loss)` — 5 forward passes × pop_size agents fused into 5 kernels (shared replay batch) |
| **Evaluate** | `vmap(eval_env.step)` + `vmap(agent.evaluate_actions)` — same pattern |
| **Replay buffer** | Pre-allocated JAX ring buffer on device (no Python list overhead) |

### Tuning batch size for your GPU

The default `population.local_batch_size: 40` was sized for ~18 GB VRAM.
For 80 GB GPUs, increase batch size to keep the GPU saturated:

```bash
# H100 SXM 80 GB — start here and watch nvidia-smi:
python main.py \
    population.batch_size=512 \
    population.local_batch_size=512 \
    population.pop_size=48 \
    population.gradient_steps_per_gen=32

# A100 80 GB:
python main.py \
    population.batch_size=256 \
    population.local_batch_size=256 \
    population.pop_size=48
```

### XLA flags (optional, H100)

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export XLA_FLAGS="--xla_gpu_enable_triton_softmax_fusion=true --xla_gpu_triton_gemm_any=true"
```

The Triton flags are experimental; disable them if you see NaN or incorrect
results.

---

## 15. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'eigen3'` | Run `pip install -e .` from the repo root, or `export PYTHONPATH=/workspace/eigen3` |
| `ModuleNotFoundError: No module named 'evorl'` | Run `pip install -e ./evorl` or `git submodule update --init --recursive` |
| `jax.devices()` returns `[CpuDevice]` | Reinstall JAX: `pip install -U "jax[cuda12]"` — make sure CUDA 12 drivers are present |
| `uvloop` install fails on Windows | Use WSL or Linux; `requirements-data.txt` is the only Windows-safe option |
| HoF says "falling back to local" | Check `CLOUD_PROVIDER`, `CLOUD_BUCKET`, and `GOOGLE_APPLICATION_CREDENTIALS` are set |
| OOM on GPU | Lower `population.pop_size` or `population.batch_size`; A100/H100 80 GB handles the defaults comfortably |
| Low GPU utilisation (`nvidia-smi` <30%) | Increase `population.batch_size` (try 256, 512, 1024); mono model is small and needs large batches to saturate |
| `orbax-checkpoint` build error | Ensure you're on Python 3.10–3.11; some 3.12 wheels may lag |
| W&B prompts or login errors | Defaults disable W&B (`logging.use_wandb=false`). Ignore or `pip uninstall wandb` if unused; to enable later, set `logging.use_wandb=true` when metrics are logged to W&B |
