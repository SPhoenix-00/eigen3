# Eigen3 — RunPod Quick-Start Guide

End-to-end instructions for spinning up a GPU pod and running Eigen3 training
with GCS-backed Hall of Fame persistence.

---

## 1. Create the RunPod Instance

| Setting | Recommended value |
|---------|-------------------|
| **GPU** | 1x A100 80 GB (or A6000 48 GB for budget runs) |
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

The default Hydra config (`configs/env/trading_mono.yaml`) looks for
`env.data_path`. You can either:

- Place the file where the config expects it, **or**
- Override on the command line (see Step 8).

If no data file is found, `train.py` falls back to synthetic data so you can
smoke-test the full stack without uploading anything.

---

## 7. Configure GCS for Hall of Fame (Optional but Recommended)

The Hall of Fame persists the best agents to a Google Cloud Storage bucket so
they survive across pods and runs.

### a. Upload your service-account key

```bash
mkdir -p /workspace/credentials
# scp or paste your key file:
scp -P <port> eigen3-sa-key.json root@<pod-ip>:/workspace/credentials/
```

### b. Set environment variables

Add these to your shell (or append to `~/.bashrc` for persistence):

```bash
export CLOUD_PROVIDER=gcs
export CLOUD_BUCKET=<your-gcs-bucket-name>
export GOOGLE_APPLICATION_CREDENTIALS=/workspace/credentials/eigen3-sa-key.json
```

> Without these variables the HoF still works — it just saves locally under
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

## 8. Run Training

```bash
# Activate env if not already:
source /workspace/eigen3/.venv/bin/activate
cd /workspace/eigen3

# Default run (mono data, synthetic fallback):
python scripts/train.py

# Point to your data file:
python scripts/train.py env.data_path=/workspace/eigen3/Eigen3_Processed_OUTPUT.pkl

# Override population sizing:
python scripts/train.py \
    env.data_path=/workspace/eigen3/Eigen3_Processed_OUTPUT.pkl \
    population.pop_size=48 \
    population.hof_capacity=10 \
    seed=123
```

### What happens on launch

1. Data is loaded and split into **train / validation / holdout** timelines.
2. Train and validation `TradingEnv` instances are created.
3. Actor and Critic networks are initialised and shapes are verified.
4. The **Hall of Fame** connects to GCS (or falls back to local) and restores
   any previously saved agents.
5. Population config is logged so you can sanity-check sizing.

---

## 9. Run Evaluation (Holdout)

After training produces checkpoints:

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pkl \
    --data /workspace/eigen3/Eigen3_Processed_OUTPUT.pkl \
    --episodes 10
```

---

## 10. Monitor Training

### TensorBoard

```bash
tensorboard --logdir logs/ --bind_all --port 6006 &
```

Expose port 6006 in RunPod's **HTTP Service Ports** setting to access the UI
from your browser.

### Logs

Hydra writes structured logs under `outputs/<date>/<time>/`. The console
output includes per-generation fitness stats and HoF admission events.

---

## 11. Detach Long Runs

Use `tmux` or `screen` so training survives SSH disconnects:

```bash
tmux new -s eigen3
source /workspace/eigen3/.venv/bin/activate
cd /workspace/eigen3
python scripts/train.py env.data_path=Eigen3_Processed_OUTPUT.pkl
# Ctrl-b d  to detach
# tmux attach -t eigen3  to reconnect
```

---

## File Layout Reference

```
eigen3/
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
│   └── utils/
├── evorl/                       # EvoRL framework (git submodule)
├── scripts/
│   ├── train.py                 # Hydra training entry point
│   └── evaluate.py              # Holdout evaluation
├── requirements-data.txt        # numpy + pandas only (data prep)
├── requirements-train.txt       # full training stack
├── requirements-dev.txt         # training + test/lint tooling
└── checkpoints/
    └── hall_of_fame/            # local HoF cache (synced to GCS)
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'eigen3'` | Run `pip install -e .` from the repo root, or `export PYTHONPATH=/workspace/eigen3` |
| `ModuleNotFoundError: No module named 'evorl'` | Run `pip install -e ./evorl` or `git submodule update --init --recursive` |
| `jax.devices()` returns `[CpuDevice]` | Reinstall JAX: `pip install -U "jax[cuda12]"` — make sure CUDA 12 drivers are present |
| `uvloop` install fails on Windows | Use WSL or Linux; `requirements-data.txt` is the only Windows-safe option |
| HoF says "falling back to local" | Check `CLOUD_PROVIDER`, `CLOUD_BUCKET`, and `GOOGLE_APPLICATION_CREDENTIALS` are set |
| OOM on GPU | Lower `population.pop_size` or `population.batch_size`; A100 80 GB handles the defaults comfortably |
| `orbax-checkpoint` build error | Ensure you're on Python 3.10–3.11; some 3.12 wheels may lag |
