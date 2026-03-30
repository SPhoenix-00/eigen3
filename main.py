"""
Eigen3 convenience entry point.

Same Hydra CLI as ``scripts/train.py``, plus a mirrored log under
``evaluation_results/training_log_<timestamp>.txt`` (set via ``EIGEN3_TRAINING_LOG``).

Examples::

    python main.py
    python main.py env.data_path=/path/to/data.pkl population.pop_size=48
    python scripts/train.py   # identical overrides, no tee unless env is set
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
_TRAIN_SCRIPT = _REPO_ROOT / "scripts" / "train.py"


def main() -> None:
    if not _TRAIN_SCRIPT.is_file():
        print(f"Missing {_TRAIN_SCRIPT}", file=sys.stderr)
        sys.exit(1)

    out_dir = _REPO_ROOT / "evaluation_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = out_dir / f"training_log_{stamp}.txt"
    os.environ["EIGEN3_TRAINING_LOG"] = str(log_path)

    try:
        import jax

        devs = jax.devices()
    except Exception as exc:  # pragma: no cover - jax optional failure
        devs = f"(jax not ready: {exc})"

    print("Eigen3 — config via Hydra; see QUICK_START.md", file=sys.stderr)
    print(f"JAX devices: {devs}", file=sys.stderr)
    print(f"Console tee: {log_path}", file=sys.stderr)

    cmd = [sys.executable, str(_TRAIN_SCRIPT), *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd, cwd=str(_REPO_ROOT)))


if __name__ == "__main__":
    main()
