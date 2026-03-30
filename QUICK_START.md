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

If data is missing, training falls back to synthetic data.

## 4) Train (recommended)

```bash
cd /workspace/eigen3
source .venv/bin/activate
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

- `evaluation_results/training_log_<timestamp>.txt`
- `evaluation_results/evaluation_<run>_<timestamp>.txt`
- `evaluation_results/evaluation_<run>_<timestamp>.json`
- `evaluation_results/summary_<run>_<timestamp>.csv`
- `evaluation_results/trades_<run>_<timestamp>.csv`
- `checkpoints/<run_name>/best_agent.msgpack`
- `checkpoints/<run_name>/best_agent.meta.json`
- `checkpoints/<run_name>/metrics_history.jsonl`
- `checkpoints/<run_name>/run_summary.json`
- `checkpoints/<run_name>/hall_of_fame/`
- `last_run.json`

## 6) Quick verification

```bash
ls -lah evaluation_results/ | tail -n 5
cat last_run.json
```

## 7) Explicit evaluation command

Training already emits evaluation bundles on new run-best.  
Run this only when you want an extra holdout pass:

```bash
python scripts/evaluate.py \
  --checkpoint_path checkpoints/<run_name>/best_agent.msgpack \
  --num_episodes 10
```

## 8) Common issues

- `ModuleNotFoundError: eigen3` -> run `pip install -e .`
- `ModuleNotFoundError: evorl` -> run `pip install -e ./evorl`
- `403 ... storage.buckets.get denied` -> use the GCS test above (no bucket metadata call), and ensure object permissions on the bucket
- CPU only in JAX -> reinstall `jax[cuda12]` and verify CUDA drivers
- GPU OOM -> lower `population.pop_size` or `population.batch_size`
