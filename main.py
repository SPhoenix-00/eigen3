"""Eigen3 top-level entrypoint with Eigen2-style orchestration defaults.

Default mode mirrors the legacy Eigen2 run experience:
- phase-style startup banner
- deterministic root-level artifact directories
- tee log under ``evaluation_results/training_log_<timestamp>.txt``

Use ``--raw-hydra`` to bypass compatibility orchestration and run the prior
Hydra wrapper behavior.
"""

from __future__ import annotations

import os
import subprocess
import sys
import argparse
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
_TRAIN_SCRIPT = _REPO_ROOT / "scripts" / "train.py"


def _parse_cli(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--raw-hydra", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="",
        default=None,
        metavar="CHECKPOINT_DIR",
        help="Resume from training_state.pkl under CHECKPOINT_DIR, or last_run.json if omitted.",
    )
    return parser.parse_known_args(argv)


def _set_compat_env(stamp: str, verbose: bool, quiet: bool) -> Path:
    # Suppress low-level backend compiler spam in default user-facing runs.
    # Keep raw backend logs available in verbose mode for debugging.
    if not verbose:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
        os.environ.setdefault("GLOG_minloglevel", "3")
    eval_dir = _REPO_ROOT / "evaluation_results"
    ckpt_dir = _REPO_ROOT / "checkpoints"
    log_dir = _REPO_ROOT / "logs"
    eval_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = eval_dir / f"training_log_{stamp}.txt"
    os.environ["EIGEN3_TRAINING_LOG"] = str(log_path)
    os.environ["EIGEN3_COMPAT_MODE"] = "1"
    os.environ["EIGEN3_ARTIFACT_ROOT"] = str(_REPO_ROOT)
    os.environ["EIGEN3_EVAL_DIR"] = str(eval_dir)
    os.environ["EIGEN3_CHECKPOINT_ROOT"] = str(ckpt_dir)
    os.environ["EIGEN3_LOG_ROOT"] = str(log_dir)
    os.environ["EIGEN3_RUN_STAMP"] = stamp
    if verbose and not quiet:
        os.environ["EIGEN3_VERBOSITY"] = "verbose"
    elif quiet and not verbose:
        os.environ["EIGEN3_VERBOSITY"] = "quiet"
    else:
        os.environ["EIGEN3_VERBOSITY"] = "normal"
    return log_path


def main() -> None:
    if not _TRAIN_SCRIPT.is_file():
        print(f"Missing {_TRAIN_SCRIPT}", file=sys.stderr)
        sys.exit(1)

    opts, passthrough = _parse_cli(sys.argv[1:])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = _set_compat_env(
        stamp=stamp,
        verbose=bool(opts.verbose),
        quiet=bool(opts.quiet),
    )
    if opts.raw_hydra:
        os.environ["EIGEN3_COMPAT_MODE"] = "0"

    if opts.resume is not None:
        os.environ["EIGEN3_RESUME"] = "1"
        os.environ["EIGEN3_RESUME_DIR"] = str(opts.resume)

    try:
        import jax

        devs = jax.devices()
    except Exception as exc:  # pragma: no cover - jax optional failure
        devs = f"(jax not ready: {exc})"

    print("Eigen3 -> Eigen2 compatibility mode", file=sys.stderr)
    print(
        "Mode: raw-hydra" if opts.raw_hydra else "Mode: eigen2-compatible default",
        file=sys.stderr,
    )
    print(f"Phase 0: Bootstrap ({stamp})", file=sys.stderr)
    print(f"JAX devices: {devs}", file=sys.stderr)
    print(f"Console tee: {log_path}", file=sys.stderr)
    print(f"Artifact root: {_REPO_ROOT}", file=sys.stderr)

    cmd = [sys.executable, str(_TRAIN_SCRIPT), *passthrough]
    raise SystemExit(subprocess.call(cmd, cwd=str(_REPO_ROOT)))


if __name__ == "__main__":
    main()
