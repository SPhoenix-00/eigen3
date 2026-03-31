"""Hydra-driven training: data, envs, Hall of Fame, ``TradingERLWorkflow``."""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional
from unittest.mock import MagicMock

import hydra
import jax
import numpy as np
from omegaconf import DictConfig, OmegaConf

from eigen3.agents import TradingAgent, params_for_flax_msgpack
from eigen3.config import (
    DEFAULT_CONVICTION_SCALING_POWER,
    DEFAULT_EPISODE_REWARD_MULTIPLIER,
    DEFAULT_HURDLE_RATE,
    DEFAULT_LOSS_PENALTY_MULTIPLIER,
)
from eigen3.data import load_trading_data
from eigen3.data.splits import (
    TrainValHoldoutSplit,
    compute_train_val_holdout_split,
    slice_trading_timeline,
)
from eigen3.environment.trading_env import TradingEnv
from eigen3.models import Actor, DoubleCritic
from eigen3.utils.gpu_memory import log_gpu_memory_report, should_log_gpu_memory_this_generation
from eigen3.utils.training_checkpoint import (
    load_training_checkpoint,
    save_training_checkpoint,
    training_state_path,
)
from eigen3.erl.global_fifteen import GlobalFifteen
from eigen3.workflows import TradingWorkflowConfig, create_trading_workflow
from eigen3.workflows.trading_workflow import stack_params

logger = logging.getLogger(__name__)

TEE_ENV_VAR = "EIGEN3_TRAINING_LOG"


def _is_tqdm_final_line(line: str) -> bool:
    return "100%" in line and ("|" in line or "it/s" in line or "it]" in line)


class TeeLogger:
    """Mirror stdout to a line-buffered file; collapse tqdm in-progress lines."""

    def __init__(self, filepath: Path, terminal):
        self.terminal = terminal
        self.log_file = open(filepath, "w", encoding="utf-8", buffering=1)
        self.verbosity = os.environ.get("EIGEN3_VERBOSITY", "normal").lower()
        self._suppress_xla_block = False

    def _is_noise_line(self, line: str) -> bool:
        if self.verbosity == "verbose":
            return False
        s = line.strip()
        if self._suppress_xla_block:
            if s == "}":
                self._suppress_xla_block = False
            return True
        if "xtile_compiler.cc:" in s:
            if "Computation:" in s:
                self._suppress_xla_block = True
            return True
        return False

    def write(self, message: str) -> None:
        if "\r" in message:
            self.terminal.write(message)
            self.terminal.flush()
            last = message.split("\r")[-1].strip()
            if _is_tqdm_final_line(last):
                self.log_file.write(last + "\n")
                self.log_file.flush()
            return
        for line in message.splitlines(keepends=True):
            if self._is_noise_line(line):
                continue
            self.terminal.write(line)
            self.log_file.write(line)
        self.terminal.flush()
        self.log_file.flush()

    def flush(self) -> None:
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self) -> bool:
        return self.terminal.isatty()

    def fileno(self) -> int:
        return self.terminal.fileno()

    def close(self) -> None:
        self.log_file.close()


class CompatArtifactManager:
    """Eigen2-style artifact writer rooted at repo-level directories."""

    def __init__(
        self,
        *,
        artifact_root: Path,
        eval_dir: Path,
        checkpoint_root: Path,
        run_name: str,
        run_id: str,
    ):
        self.artifact_root = artifact_root
        self.eval_dir = eval_dir
        self.checkpoint_root = checkpoint_root
        self.run_name = run_name
        self.run_id = run_id
        self.run_dir = checkpoint_root / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        self.best_agent_path = self.run_dir / "best_agent.msgpack"
        self.best_meta_path = self.run_dir / "best_agent.meta.json"
        self.metrics_path = self.run_dir / "metrics_history.jsonl"
        self.summary_path = self.run_dir / "run_summary.json"
        self.last_run_path = self.artifact_root / "last_run.json"

    def append_metric(self, metrics: dict[str, Any]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")

    def save_best_agent(self, params: Any, generation: int, score: float) -> Path:
        from flax.serialization import to_bytes

        self.best_agent_path.write_bytes(to_bytes(params_for_flax_msgpack(params)))
        with self.best_meta_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_name": self.run_name,
                    "run_id": self.run_id,
                    "generation": generation,
                    "score": float(score),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "best_agent_path": str(self.best_agent_path),
                },
                f,
                indent=2,
            )
        return self.best_agent_path

    def write_last_run(self) -> None:
        payload = {
            "run_name": self.run_name,
            "run_id": self.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checkpoint_dir": str(self.run_dir),
            "best_agent_path": str(self.best_agent_path),
        }
        with self.last_run_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def write_run_summary(
        self,
        *,
        last_metrics: dict[str, Any],
        hydra_output_dir: str | None,
        config_yaml: str,
    ) -> None:
        payload = {
            "run_name": self.run_name,
            "run_id": self.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "hydra_output_dir": hydra_output_dir,
            "checkpoint_dir": str(self.run_dir),
            "metrics_file": str(self.metrics_path),
            "best_meta_file": str(self.best_meta_path),
            "last_metrics": last_metrics,
            "resolved_config_yaml": config_yaml,
        }
        with self.summary_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def write_evaluation_bundle(self, payload: dict[str, Any]) -> dict[str, str]:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"{self.run_name}_{stamp}"
        txt_path = self.eval_dir / f"evaluation_{base}.txt"
        json_path = self.eval_dir / f"evaluation_{base}.json"
        summary_csv_path = self.eval_dir / f"summary_{base}.csv"
        trades_csv_path = self.eval_dir / f"trades_{base}.csv"

        with json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        summary_fields = [
            "run_name",
            "num_episodes",
            "reward_mean",
            "reward_std",
            "pnl_mean",
            "pnl_std",
            "win_rate_mean",
            "num_trades_mean",
            "roi_pct_mean",
        ]
        with summary_csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_fields)
            writer.writeheader()
            writer.writerow({k: payload.get(k, "") for k in summary_fields})

        episode_fields = [
            "episode_index",
            "total_reward",
            "steps",
            "num_trades",
            "num_wins",
            "num_losses",
            "total_gain_pct",
            "episode_roi_pct",
            "total_pnl",
            "win_rate",
            "peak_capital_employed",
            "days_with_positions",
            "days_without_positions",
        ]
        with trades_csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=episode_fields)
            writer.writeheader()
            for idx, row in enumerate(payload.get("episodes", []), start=1):
                out = dict(row)
                out["episode_index"] = idx
                writer.writerow({k: out.get(k, "") for k in episode_fields})

        lines = [
            f"Run: {self.run_name}",
            f"Episodes: {payload.get('num_episodes', 0)}",
            f"Reward mean/std: {payload.get('reward_mean', 0.0):.6f} / {payload.get('reward_std', 0.0):.6f}",
            f"PnL mean/std: {payload.get('pnl_mean', 0.0):.6f} / {payload.get('pnl_std', 0.0):.6f}",
            f"Win rate mean: {100.0 * payload.get('win_rate_mean', 0.0):.2f}%",
            f"Summary CSV: {summary_csv_path}",
            f"Trades CSV: {trades_csv_path}",
            f"JSON: {json_path}",
        ]
        txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        return {
            "txt": str(txt_path),
            "json": str(json_path),
            "summary_csv": str(summary_csv_path),
            "trades_csv": str(trades_csv_path),
        }

    def write_gauntlet_bundle(
        self, generation: int, payload: dict[str, Any]
    ) -> dict[str, str]:
        """Write ``gauntlet_gen{NNNN}.json`` and matching CSV under ``eval_dir``."""
        return _write_gauntlet_reports(
            self.eval_dir,
            generation=generation,
            payload=payload,
        )


def _write_gauntlet_reports(
    eval_dir: Path,
    *,
    generation: int,
    payload: dict[str, Any],
) -> dict[str, str]:
    eval_dir.mkdir(parents=True, exist_ok=True)
    gen_tag = f"{int(generation):04d}"
    json_path = eval_dir / f"gauntlet_gen{gen_tag}.json"
    csv_path = eval_dir / f"gauntlet_gen{gen_tag}.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    rows = payload.get("per_agent_rows", [])
    if rows:
        fields = list(rows[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    return {"json": str(json_path), "csv": str(csv_path)}


def _maybe_run_hof_gauntlet(
    *,
    workflow: Any,
    hof: Any,
    global_fifteen: GlobalFifteen,
    cfg: DictConfig,
    run_name: str,
    eval_dir: Path,
    split: TrainValHoldoutSplit,
) -> None:
    """Every ``population.gauntlet_interval`` generations: HoF gauntlet + Global 15 updates."""
    interval = int(OmegaConf.select(cfg, "population.gauntlet_interval", default=20))
    if interval <= 0 or workflow.generation % interval != 0 or len(hof) == 0:
        return
    if workflow._stacked_params is None:
        return

    # Use print (not only logger) so TeeLogger / EIGEN3_TRAINING_LOG captures output.
    print("\n" + "=" * 60, flush=True)
    print(
        f"HoF GAUNTLET | workflow gen {workflow.generation} | run {run_name}",
        flush=True,
    )
    print("=" * 60, flush=True)

    template = jax.tree.map(lambda x: x[0], workflow._stacked_params)
    entries_snapshot = list(hof.entries)
    n_hof = len(entries_snapshot)
    print(
        f"  Evaluating {n_hof} HoF agent(s): one shared random validation episode + holdout...",
        flush=True,
    )
    t_g_start = time.perf_counter()
    loaded = [hof.load_params(e.agent_id, template) for e in entries_snapshot]
    stacked = stack_params(loaded)

    seed = int(OmegaConf.select(cfg, "seed", default=0))
    gauntlet_key = jax.random.fold_in(
        jax.random.fold_in(jax.random.PRNGKey(seed), workflow.generation),
        0x47314E54,  # "GANT" — gauntlet stream salt
    )
    out = workflow.gauntlet_evaluate(stacked, gauntlet_key)
    print(f"  Gauntlet rollouts done ({time.perf_counter() - t_g_start:.1f}s).", flush=True)

    val_bn = out["val_bn_excess"]
    hold_bn = out["hold_bn_excess"]
    gscore = out["gauntlet_score"]

    purge_ids: list[int] = []
    per_agent_rows: list[dict[str, Any]] = []
    candidates: list[tuple[Any, float, float, float, Any]] = []

    for i, entry in enumerate(entries_snapshot):
        v = float(val_bn[i])
        h = float(hold_bn[i])
        g = float(gscore[i])
        if v <= 0.0:
            purge_ids.append(entry.agent_id)
        row: dict[str, Any] = {
            "hof_agent_id": entry.agent_id,
            "hof_generation": entry.generation,
            "val_bn_excess": v,
            "hold_bn_excess": h,
            "gauntlet_score": g,
            "purged_val_bn_fail": v <= 0.0,
            "promoted_global15": False,
            "global15_action": "",
        }
        per_agent_rows.append(row)
        if v > 0.0 and h > 0.0:
            candidates.append((loaded[i], g, v, h, entry))

    removed = hof.remove_entries_by_agent_ids(purge_ids)
    if removed:
        print(
            f"  Purged from HoF (val BNH <= 0): {len(removed)} — ids {removed}",
            flush=True,
        )
    else:
        print("  Purged from HoF: 0 (all agents positive val BNH).", flush=True)

    candidates.sort(key=lambda t: -t[1])
    for params_i, g, v, h, ent in candidates:
        action = global_fifteen.add_or_replace(
            params_i,
            gauntlet_score=g,
            val_bn_excess=v,
            hold_bn_excess=h,
            promoted_at_generation=int(workflow.generation),
            source_run_name=run_name,
            source_hof_agent_id=int(ent.agent_id),
        )
        for row in per_agent_rows:
            if row["hof_agent_id"] == ent.agent_id:
                if action.startswith("admitted") or action.startswith("replaced"):
                    row["promoted_global15"] = True
                    row["global15_action"] = action
                else:
                    row["global15_action"] = action
                break

    hof.save()
    global_fifteen.save()

    split_info = {
        "val_env_start": split.val_env_start,
        "val_end": split.val_end,
        "holdout_env_start": split.holdout_env_start,
        "holdout_end": split.holdout_end,
    }
    payload: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name,
        "workflow_generation": int(workflow.generation),
        "gauntlet_interval": interval,
        "split": split_info,
        "val_reset_key_uint32": out["val_reset_key_uint32"],
        "hold_reset_key_uint32": out["hold_reset_key_uint32"],
        "purged_hof_agent_ids": removed,
        "per_agent_rows": per_agent_rows,
        "global_15_snapshot": global_fifteen.to_snapshot_dicts(),
    }
    paths = _write_gauntlet_reports(eval_dir, generation=int(workflow.generation), payload=payload)

    promoted_n = sum(1 for r in per_agent_rows if r.get("promoted_global15"))
    attempted_g15 = len(candidates)
    print(
        f"  Global 15: {attempted_g15} double-pass candidate(s); "
        f"{promoted_n} admitted or replaced (size now {len(global_fifteen)}/{global_fifteen.capacity}).",
        flush=True,
    )
    print(
        "  Per-agent (val/hold BNH excess = terminal vs equal-weight B&H, scaled $):",
        flush=True,
    )
    hdr = (
        f"  {'hof_id':>7} {'val_BNH':>12} {'hold_BNH':>12} {'gauntlet':>12} "
        f"{'purged':>8} {'G15':>5} {'G15_action':<22}"
    )
    print(hdr, flush=True)
    print("  " + "-" * 86, flush=True)
    for r in sorted(per_agent_rows, key=lambda x: x["hof_agent_id"]):
        purged = "yes" if r["purged_val_bn_fail"] else "no"
        g15 = "yes" if r.get("promoted_global15") else "no"
        act = str(r.get("global15_action") or "")[:22]
        print(
            f"  {r['hof_agent_id']:7d} "
            f"{r['val_bn_excess']:12.4f} {r['hold_bn_excess']:12.4f} {r['gauntlet_score']:12.4f} "
            f"{purged:>8} {g15:>5} {act:<22}",
            flush=True,
        )
    print(f"  Val reset key (uint32): {out['val_reset_key_uint32']}", flush=True)
    print(f"  Hold reset key (uint32): {out['hold_reset_key_uint32']}", flush=True)
    print(f"  Split rows: val [{split.val_env_start}, {split.val_end}) | ", end="", flush=True)
    print(
        f"holdout env [{split.holdout_env_start}, {split.holdout_end})",
        flush=True,
    )
    print(f"  Report JSON: {paths['json']}", flush=True)
    print(f"  Report CSV:  {paths['csv']}", flush=True)
    print("=" * 60 + "\n", flush=True)

    logger.info(
        "HoF gauntlet gen=%s: %d agents, purged=%d, G15 promoted=%d | %s",
        workflow.generation,
        n_hof,
        len(removed),
        promoted_n,
        paths["json"],
    )


def _compat_mode_enabled() -> bool:
    return os.environ.get("EIGEN3_COMPAT_MODE", "1") != "0"


def _resolve_resume_run_dir(cfg: DictConfig, artifact_root: Path) -> Optional[Path]:
    """Directory containing ``training_state.pkl`` (from CLI, Hydra, or ``last_run.json``)."""
    env_resume = os.environ.get("EIGEN3_RESUME", "").strip().lower()
    env_on = env_resume in ("1", "true", "yes")
    resume_flag = bool(OmegaConf.select(cfg, "population.resume", default=False))
    rp_cfg = OmegaConf.select(cfg, "population.resume_path", default=None)
    path_hint = (str(rp_cfg).strip() if rp_cfg is not None else "") or os.environ.get(
        "EIGEN3_RESUME_DIR", ""
    ).strip()

    use_resume = env_on or resume_flag or bool(path_hint)
    if not use_resume:
        return None

    if path_hint:
        p = Path(path_hint)
        if not p.is_absolute():
            p = Path(hydra.utils.to_absolute_path(path_hint))
        if p.is_file():
            p = p.parent
        return p

    last_run = artifact_root / "last_run.json"
    if not last_run.is_file():
        raise FileNotFoundError(
            "Resume requested but no directory was given (``population.resume_path`` / "
            "``--resume DIR``) and "
            f"{last_run} is missing. Pass an explicit run directory."
        )
    data = json.loads(last_run.read_text(encoding="utf-8"))
    ck = data.get("checkpoint_dir")
    if not ck:
        raise ValueError(f"{last_run} has no checkpoint_dir field")
    return Path(ck)


def _phase_log(label: str) -> None:
    logger.info("========== %s ==========", label)


def _ascii_best_bnh_chart_lines(
    history: list[tuple[int, float]],
    *,
    height: int = 4,
    plot_width: int = 44,
) -> list[str]:
    """Best-agent validation BNH excess ($): density sparkline + small connected grid chart."""
    if not history:
        return []

    # 8-step density row (ASCII so cp1252 / piped logs still render).
    _SPK = " .:-=+*#"

    values = [v for _, v in history]
    gen_first, gen_last = history[0][0], history[-1][0]
    n = len(values)
    last_v = float(values[-1])
    if n >= 6:
        delta_lbl = f"d5={last_v - float(values[-6]):+.2f}"
    elif n >= 2:
        delta_lbl = f"d1={last_v - float(values[-2]):+.2f}"
    else:
        delta_lbl = ""

    if n > plot_width:
        cols: list[float] = []
        for i in range(plot_width):
            j0 = int(i * n / plot_width)
            j1 = max(int((i + 1) * n / plot_width), j0 + 1)
            chunk = values[j0:j1]
            cols.append(sum(chunk) / len(chunk))
    else:
        cols = list(values)

    raw_min, raw_max = min(values), max(values)
    span = raw_max - raw_min
    pad = 0.03 * span if span > 1e-12 else 1.0
    vmin, vmax = raw_min - pad, raw_max + pad
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1.0

    w = len(cols)
    rows_idx = []
    for val in cols:
        rb = int(round((val - vmin) / (vmax - vmin) * (height - 1)))
        rows_idx.append(height - 1 - rb)

    grid = [[" " for _ in range(w)] for _ in range(height)]

    def _bres(x0: int, y0: int, x1: int, y1: int) -> None:
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        err, x, y = dx - dy, x0, y0
        while True:
            if 0 <= y < height and 0 <= x < w and grid[y][x] == " ":
                grid[y][x] = "."
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    for j in range(w - 1):
        _bres(j, rows_idx[j], j + 1, rows_idx[j + 1])

    if vmin < 0.0 < vmax:
        z_row = height - 1 - int(round((0.0 - vmin) / (vmax - vmin) * (height - 1)))
        z_row = max(0, min(height - 1, z_row))
        for j in range(w):
            c = grid[z_row][j]
            if c == " ":
                grid[z_row][j] = "-"
            elif c == ".":
                grid[z_row][j] = "+"
            elif c == "*":
                grid[z_row][j] = "#"

    for j, r in enumerate(rows_idx):
        grid[r][j] = "*"

    # Block row uses same vmin/vmax as the grid for consistency.
    block_row_parts: list[str] = []
    for val in cols:
        t = (val - vmin) / (vmax - vmin)
        bi = int(max(0, min(7, round(t * 7.0))))
        block_row_parts.append(_SPK[bi])
    block_row = "".join(block_row_parts)

    lines: list[str] = []
    extra = f"  {delta_lbl}" if delta_lbl else ""
    lines.append(
        f"  Best BNH (val $ vs B&H)  gens {gen_first}-{gen_last}  n={n}  "
        f"last={last_v:+.2f}{extra}  range=[{raw_min:+.2f},{raw_max:+.2f}]"
    )
    lines.append(f"  {'sparkline':>10}|{block_row}|")
    for r in range(height):
        y = vmax - (vmax - vmin) * (r / max(height - 1, 1))
        row_s = "".join(grid[r])
        lines.append(f"  {y:10.2f}|{row_s}|")
    lines.append(f"  {' ' * 10}+{'-' * w}+  gen {gen_first} .. {gen_last}")
    return lines


def _print_generation_summary(
    *,
    gen: int,
    num_gen: int,
    metrics: dict[str, Any],
    progress_eval: dict[str, Any],
    prev_metrics: Optional[dict[str, Any]] = None,
    prev_eval: Optional[dict[str, Any]] = None,
    avg_gen_seconds: Optional[float] = None,
    top5_evals: Optional[list] = None,
    wall_clock_s: Optional[float] = None,
    best_bnh_history: Optional[list[tuple[int, float]]] = None,
) -> None:
    def _delta(curr: float, prev: Optional[float], suffix: str = "") -> str:
        if prev is None:
            return ""
        d = curr - prev
        sign = "+" if d >= 0 else ""
        return f"  ({sign}{d:.2f}{suffix})"

    def _fmt_time(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        return f"{seconds / 60.0:.1f}m"

    def _pct(x: float, total: float) -> float:
        return 0.0 if total <= 0 else (100.0 * x / total)

    bsz = int(metrics.get("buffer_size", 0))
    bcap = max(1, int(metrics.get("buffer_capacity", 1)))
    phases = [
        ("Init", float(metrics.get("timing_init_s", 0.0))),
        ("Collect", float(metrics.get("timing_collect_s", 0.0))),
        ("Train", float(metrics.get("timing_train_s", 0.0))),
        ("Val", float(metrics.get("timing_val_s", 0.0))),
        ("Stats", float(metrics.get("timing_stats_s", 0.0))),
        ("HoF", float(metrics.get("timing_hof_s", 0.0))),
        ("Evolve", float(metrics.get("timing_evolve_s", 0.0))),
    ]
    t_total = float(metrics.get("timing_total_s", 0.0))
    t_wall = wall_clock_s or t_total
    t_outer = max(0.0, t_wall - t_total)
    remaining = max(0, num_gen - gen)
    eta_s = (avg_gen_seconds or t_wall) * remaining

    print("\n" + "=" * 70)
    print(f"  Gen {gen}/{num_gen} | {_fmt_time(t_wall)} | ETA {_fmt_time(eta_s)}")
    print("=" * 70)
    print("\n  POPULATION")
    print("  -------------------------------------------------------")
    positive = int(metrics.get("positive_agents", 0))
    print(
        f"  Fitness: best={metrics.get('max_fitness', 0.0):8.2f}  "
        f"mean={metrics.get('mean_fitness', 0.0):8.2f}  "
        f"std={metrics.get('std_fitness', 0.0):8.2f}"
    )
    print(f"  Positive: {positive}/{int(metrics.get('population_size', 0) or 0)} agents")
    print(
        f"  Coeff: mean_max={metrics.get('mean_max_coeff', 0.0):.3f}  "
        f"best_max={metrics.get('max_max_coeff', 0.0):.3f}  "
        f"(threshold=1.0)"
    )
    print(
        f"  Train: collect/agent={metrics.get('collect_reward_mean_per_agent', 0.0):.2f}  "
        f"actor={metrics.get('mean_actor_loss', float('nan')):.4f}  "
        f"critic={metrics.get('mean_critic_loss', float('nan')):.4f}  "
        f"q={metrics.get('mean_mean_q', float('nan')):.4f}"
    )
    print("\n--- Validation Summary ---")
    print("Top 5 by Fitness:")
    if top5_evals:
        for rank, (idx, fitness, ev) in enumerate(top5_evals, 1):
            roi = ev.get("roi_pct_mean", ev.get("gain_pct_mean", 0.0))
            pnl = ev.get('pnl_mean', 0.0)
            bh_excess = ev.get('bh_excess', 0.0)
            wr_raw = ev.get("win_rate_mean", None)
            wr_str = "  N/A" if wr_raw is None else f"{float(wr_raw) * 100.0:5.1f}%"
            r_mean = ev.get('reward_mean', 0.0)
            print(
                f"  {rank}. Agent {idx:2d}: Fitness={fitness:8.2f}, "
                f"Val={r_mean:7.2f}, ROI={roi:6.2f}%, "
                f"PnL=${pnl:8.2f} ({bh_excess:+.2f}), WR={wr_str}"
            )
    else:
        print("  (No Top 5 stats available)")

    if best_bnh_history and gen % 5 == 0:
        print("\n  BEST AGENT BNH (over time)")
        print("  -------------------------------------------------------")
        for line in _ascii_best_bnh_chart_lines(best_bnh_history):
            print(line)

    if "hof_size" in metrics:
        print("\n  HALL OF FAME")
        print("  -------------------------------------------------------")
        print(
            f"  Size: {metrics.get('hof_size', 0)}  |  "
            f"Best: {metrics.get('hof_best', 0.0):.2f}  "
            f"({metrics.get('hof_best_bh_excess', 0.0):+.2f} vs equal-weight b&h)  |  "
            f"Worst: {metrics.get('hof_worst', 0.0):.2f}  "
            f"({metrics.get('hof_worst_bh_excess', 0.0):+.2f} vs equal-weight b&h)"
        )
        print(f"  Median ROI: {metrics.get('hof_median_roi', 0.0):.2f}%")
    print("\n  TIMING")
    print("  -------------------------------------------------------")
    visible = [(name, t) for name, t in phases if t >= 1.0]
    hidden = sum(t for _, t in phases if t < 1.0) + t_outer
    parts = []
    for name, t in visible:
        parts.append(f"{name}: {_fmt_time(t)} ({_pct(t, t_wall):.0f}%)")
    if hidden >= 0.01:
        parts.append(f"Other: {_fmt_time(hidden)} ({_pct(hidden, t_wall):.0f}%)")
    print(f"  {' | '.join(parts)}")
    print(f"  Wall clock: {_fmt_time(t_wall)}")

    print(f"\n  BUFFER")
    print("  -------------------------------------------------------")
    print(
        f"  {bsz:,}/{bcap:,} ({100.0 * bsz / bcap:.1f}%)  |  "
        f"Env steps: {metrics.get('total_env_steps', 0):,}"
    )
    print("\n" + "=" * 70 + "\n")


def _top5_eval_rows_from_metrics(metrics: dict[str, Any]) -> list[tuple[int, float, dict[str, Any]]]:
    """Build Top-5 validation table rows from :meth:`TradingERLWorkflow.run_generation` metrics.

    Uses the same vmapped eval as fitness (no extra env rollouts).
    """
    indices = metrics.get("top5_indices") or []
    fitness_list = metrics.get("top5_fitness") or []
    rm = metrics.get("top5_val_reward_mean") or []
    rp = metrics.get("top5_roi_pct") or metrics.get("top5_gain_pct") or []
    pnl = metrics.get("top5_pnl") or []
    bh = metrics.get("top5_bh_excess_usd") or []
    wr = metrics.get("top5_win_rate") or []
    rows: list[tuple[int, float, dict[str, Any]]] = []
    for rank, idx in enumerate(indices):
        ev = {
            "reward_mean": rm[rank] if rank < len(rm) else 0.0,
            "roi_pct_mean": rp[rank] if rank < len(rp) else 0.0,
            "gain_pct_mean": rp[rank] if rank < len(rp) else 0.0,
            "pnl_mean": pnl[rank] if rank < len(pnl) else 0.0,
            "bh_excess": bh[rank] if rank < len(bh) else 0.0,
            "win_rate_mean": wr[rank] if rank < len(wr) else None,
        }
        fit = float(fitness_list[rank]) if rank < len(fitness_list) else float("nan")
        rows.append((int(idx), fit, ev))
    return rows


def _build_eval_payload_from_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Construct an evaluation bundle payload from vmapped eval data in metrics.

    Replaces the sequential ``_evaluate_agent_on_env`` call with data already
    captured during the batched population evaluation.
    """
    episodes = metrics.get("best_agent_val_episodes", [])
    if not episodes:
        return {"num_episodes": 0, "episodes": []}
    rewards = np.asarray([e["total_reward"] for e in episodes], dtype=np.float64)
    pnls = np.asarray([e["total_pnl"] for e in episodes], dtype=np.float64)
    peaks = np.asarray([e["peak_capital_employed"] for e in episodes], dtype=np.float64)
    wins = np.asarray([e["win_rate"] for e in episodes], dtype=np.float64)
    trades = np.asarray([e["num_trades"] for e in episodes], dtype=np.float64)
    gains = np.asarray([e["total_gain_pct"] for e in episodes], dtype=np.float64)
    peak_max = float(np.max(peaks)) if peaks.size else 0.0
    roi_agg = (100.0 * float(np.sum(pnls)) / peak_max) if peak_max > 0 else 0.0
    return {
        "num_episodes": len(episodes),
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "reward_min": float(rewards.min()),
        "reward_max": float(rewards.max()),
        "pnl_mean": float(pnls.mean()),
        "pnl_std": float(pnls.std()),
        "win_rate_mean": float(wins.mean()),
        "num_trades_mean": float(trades.mean()),
        "roi_pct_mean": roi_agg,
        "gain_pct_mean": roi_agg,
        "sum_gain_pct_mean": float(gains.mean()),
        "episodes": episodes,
    }


def _short_class(path: Optional[str]) -> str:
    if not path:
        return "?"
    s = str(path)
    return s.rsplit(".", maxsplit=1)[-1]


def _single_gpu_workflow_mode(cfg: DictConfig) -> bool:
    """True when we run one device without pmap (use local batch / replay sizes)."""
    pmap = bool(OmegaConf.select(cfg, "enable_pmap", default=False))
    n_dev = int(OmegaConf.select(cfg, "num_devices", default=1))
    return (not pmap) and n_dev == 1


def _effective_workflow_batch_size(cfg: DictConfig) -> int:
    """Gradient batch: ``local_batch_size`` on single-GPU, else ``batch_size``."""
    default = int(OmegaConf.select(cfg, "population.batch_size", default=160))
    if not _single_gpu_workflow_mode(cfg):
        return default
    local = OmegaConf.select(cfg, "population.local_batch_size", default=None)
    return int(local) if local is not None else default


def _effective_gradient_vmap_chunk_size(cfg: DictConfig) -> Optional[int]:
    """Cap loss ``vmap`` width on single-GPU (``local_training_agent_batch_size``)."""
    if not _single_gpu_workflow_mode(cfg):
        return None
    c = OmegaConf.select(cfg, "population.local_training_agent_batch_size", default=8)
    if c is None:
        return None
    ci = int(c)
    return ci if ci > 0 else None


def _effective_replay_buffer_size(cfg: DictConfig) -> int:
    """Replay capacity: ``local_replay_buffer_size`` on single-GPU, else full size."""
    default = int(OmegaConf.select(cfg, "population.replay_buffer_size", default=1_500_000))
    if not _single_gpu_workflow_mode(cfg):
        return default
    local = OmegaConf.select(cfg, "population.local_replay_buffer_size", default=None)
    return int(local) if local is not None else default


def run_config_summary(cfg: DictConfig) -> str:
    """Detailed Eigen2-style startup summary for the console."""
    lines: list[str] = [
        "",
        "============================================================",
        "Project Eigen 3 Configuration (Full)",
        "============================================================",
    ]
    
    def _flatten_dict(d: dict, parent_key: str = "") -> dict:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict) and v:
                items.extend(_flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)

    try:
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        cfg_dict = OmegaConf.to_container(cfg, resolve=False)
        
    flat_cfg = _flatten_dict(cfg_dict)
    
    # Custom additions that were calculated in the original logic
    try:
        from hydra.core.hydra_config import HydraConfig
        flat_cfg["hydra_output_dir"] = HydraConfig.get().runtime.output_dir
    except Exception:
        pass

    flat_cfg["effective_batch_size"] = _effective_workflow_batch_size(cfg)
    flat_cfg["effective_replay_buffer_size"] = _effective_replay_buffer_size(cfg)
    flat_cfg["loss_vmap_chunk"] = _effective_gradient_vmap_chunk_size(cfg) or "full"

    for k, v in flat_cfg.items():
        k_str = str(k).upper()
        v_str = str(v)
        lines.append(f"{k_str:.<40} {v_str}")
        
    lines.append("============================================================")
    lines.append("")
    return "\n".join(lines)


def build_networks(cfg: DictConfig) -> tuple[Actor, DoubleCritic]:
    def _env(k: str, default):
        return OmegaConf.select(cfg, f"env.{k}", default=default)

    def _agent_fe(k: str, default):
        return OmegaConf.select(cfg, f"agent.feature_extractor.{k}", default=default)

    nc = int(_env("num_columns", 117))
    nf = int(_env("num_features_obs", 5))
    ni = int(_env("num_investable_stocks", 108))
    istart = int(_env("investable_start_col", 9))
    chunk = int(_agent_fe("chunk_size", 64))
    use_remat_a = OmegaConf.select(cfg, "agent.actor_network.use_remat", default=True)
    use_remat_c = OmegaConf.select(cfg, "agent.critic_network.use_remat", default=True)

    min_st = float(_env("min_sale_target", 10.0))
    max_st = float(_env("max_sale_target", 50.0))

    actor = Actor(
        num_columns=nc,
        num_features=nf,
        num_investable_stocks=ni,
        investable_start_col=istart,
        column_chunk_size=chunk,
        use_remat=use_remat_a,
        min_sale_target=min_st,
        max_sale_target=max_st,
    )
    critic = DoubleCritic(
        num_columns=nc,
        num_features=nf,
        num_investable_stocks=ni,
        column_chunk_size=chunk,
        use_remat=use_remat_c,
    )
    return actor, critic


def build_trading_workflow_config(cfg: DictConfig) -> TradingWorkflowConfig:
    """Map Hydra ``population`` / ``agent`` into ``TradingWorkflowConfig``."""

    pop_size = int(OmegaConf.select(cfg, "population.pop_size", default=96))
    elite_frac = float(OmegaConf.select(cfg, "population.elite_frac", default=0.4))
    elite_size = max(1, min(int(round(pop_size * elite_frac)), pop_size))

    tournament_size = int(OmegaConf.select(cfg, "population.tournament_size", default=3))
    tournament_size = max(1, min(tournament_size, pop_size))

    crossover_rate = float(
        OmegaConf.select(cfg, "population.genetic_crossover_rate", default=0.5)
    )
    target_update_period = int(
        OmegaConf.select(cfg, "agent.actor_update_interval", default=2)
    )
    steps_per_agent = int(OmegaConf.select(cfg, "population.steps_per_agent", default=100))

    return TradingWorkflowConfig(
        population_size=pop_size,
        elite_size=elite_size,
        tournament_size=tournament_size,
        mutation_rate=float(OmegaConf.select(cfg, "population.mutation_rate", default=0.2)),
        mutation_std=float(OmegaConf.select(cfg, "population.mutation_std", default=0.025)),
        crossover_rate=crossover_rate,
        gradient_steps_per_gen=int(
            OmegaConf.select(cfg, "population.gradient_steps_per_gen", default=16)
        ),
        batch_size=_effective_workflow_batch_size(cfg),
        replay_buffer_size=_effective_replay_buffer_size(cfg),
        val_episodes=int(OmegaConf.select(cfg, "population.val_episodes", default=5)),
        conservative_k=int(OmegaConf.select(cfg, "population.conservative_k", default=3)),
        target_update_period=target_update_period,
        steps_per_agent=steps_per_agent,
        gradient_vmap_chunk_size=_effective_gradient_vmap_chunk_size(cfg),
        forced_exploration_buffer_pct=float(
            OmegaConf.select(cfg, "population.forced_exploration_buffer_pct", default=0.9)
        ),
        bnh_penalty_warmup_gens=int(
            OmegaConf.select(cfg, "population.bnh_penalty_warmup_gens", default=5)
        ),
    )


def run_training(cfg: DictConfig) -> List[dict[str, Any]]:
    """Load data, build envs and workflow, run ``TradingERLWorkflow.train``.

    Returns:
        List of per-generation metric dicts from the workflow.
    """
    tee_path = os.environ.get(TEE_ENV_VAR)
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    tee: Optional[TeeLogger] = None
    if tee_path:
        tee = TeeLogger(Path(tee_path), sys.stdout)
        sys.stdout = tee
        sys.stderr = tee

    try:
        return _run_training_impl(cfg)
    finally:
        if tee is not None:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            tee.close()


def _run_training_impl(cfg: DictConfig) -> List[dict[str, Any]]:
    compat_mode = _compat_mode_enabled()
    if compat_mode:
        # Reconfigure root logger to remove prefix formatting for a clean Eigen2-style output
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            handler.setFormatter(logging.Formatter("%(message)s"))

    tee_active = os.environ.get(TEE_ENV_VAR)
    if tee_active:
        logger.info("Mirroring console to %s", tee_active)
    if compat_mode:
        _phase_log("Phase 0: Startup")

    artifact_root = Path(
        os.environ.get(
            "EIGEN3_ARTIFACT_ROOT",
            str(Path(hydra.utils.to_absolute_path("."))),
        )
    ).resolve()

    from eigen3.utils.run_naming import resolved_checkpoint_root

    default_checkpoint_root = resolved_checkpoint_root(artifact_root)
    cloud_project = OmegaConf.select(cfg, "population.cloud_project_name", default="eigen3")

    from eigen3.erl.cloud_sync import CloudSync

    cloud_sync = CloudSync.from_env(project_name=cloud_project)
    resume_dir: Optional[Path] = None
    _rdir = _resolve_resume_run_dir(cfg, artifact_root)
    if _rdir is not None:
        resume_dir = _rdir.resolve()

    if resume_dir is not None:
        run_name = resume_dir.name
    else:
        wandb_style = bool(
            OmegaConf.select(cfg, "population.wandb_style_run_name", default=True)
        )
        if wandb_style:
            from eigen3.utils.run_naming import generate_wandb_style_run_name

            run_name = generate_wandb_style_run_name(default_checkpoint_root)
        else:
            run_name = str(OmegaConf.select(cfg, "run_name", default="run"))

    try:
        from omegaconf import open_dict

        with open_dict(cfg):
            cfg.run_name = run_name
    except Exception:
        pass

    # Claim the run directory immediately so a second launch sees it for suffix
    # allocation (otherwise names stay at *-1 until late mkdir after env build).
    if resume_dir is None:
        try:
            (default_checkpoint_root / run_name).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("Early run directory create failed (will retry later): %s", exc)

    hof_cloud_prefix = f"{cloud_project}/{run_name}/hall_of_fame"
    ck_banner_path = str(resume_dir) if resume_dir is not None else str(
        default_checkpoint_root / run_name
    )

    logger.info(run_config_summary(cfg))
    logger.debug("Full resolved configuration:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    logger.info("Setting random seed: %s", cfg.seed)
    key = jax.random.PRNGKey(int(cfg.seed))

    # If this shows backend=cpu, training uses no GPU (wrong jax[cuda] wheel or JAX_PLATFORMS=cpu).
    logger.info(
        "JAX: default_backend=%s devices=%s",
        jax.default_backend(),
        [str(d) for d in jax.devices()],
    )

    from eigen3.utils.terminal_banner import print_training_identity_banner

    print_training_identity_banner(
        run_name=run_name,
        cloud_sync=cloud_sync,
        cloud_prefix=hof_cloud_prefix,
        checkpoint_dir=ck_banner_path,
        resume=resume_dir is not None,
    )
    logger.info(
        "Training identity: run_name=%s cloud_provider=%s resume=%s",
        run_name,
        cloud_sync.provider,
        resume_dir is not None,
    )

    if compat_mode:
        _phase_log("Phase 1: Data Loading")
    data_path = OmegaConf.select(cfg, "env.data_path", default="data/raw")
    path = Path(hydra.utils.to_absolute_path(str(data_path)))
    column_index = OmegaConf.select(cfg, "env.column_index", default=None)

    if path.is_dir() and (path / "data_array.npy").exists():
        logger.info("Loading Eigen2 npy bundle from: %s", data_path)
        load_kwargs: dict = {}
        if column_index is not None:
            load_kwargs["column_index"] = int(column_index)
            logger.info("Using column_index=%s", column_index)
        data_array, data_array_full, norm_stats, dates_ordinal = load_trading_data(
            str(path), **load_kwargs
        )
    elif path.is_file() and path.suffix.lower() in (".pkl", ".pickle", ".csv"):
        logger.info("Loading mono table from: %s", data_path)
        mono_ch = int(OmegaConf.select(cfg, "env.mono_num_channels", default=18))
        mono_hdr = OmegaConf.select(cfg, "env.mono_csv_header", default=0)
        data_array, data_array_full, norm_stats, dates_ordinal = load_trading_data(
            str(path),
            mono_num_channels=mono_ch,
            mono_csv_header=mono_hdr,
        )
    else:
        if not path.exists():
            raise FileNotFoundError(
                f"Training data not found: {path} (env.data_path={data_path!r}). "
                "Provide an Eigen2 npy bundle directory containing data_array.npy, "
                "or a mono table file (.pkl, .pickle, or .csv)."
            )
        if path.is_dir():
            raise FileNotFoundError(
                f"Training data directory is not a valid Eigen2 bundle (missing data_array.npy): "
                f"{path} (env.data_path={data_path!r})."
            )
        raise ValueError(
            f"Unsupported training data file type for {path} (env.data_path={data_path!r}). "
            "Expected a mono table with extension .pkl, .pickle, or .csv."
        )

    logger.info("Data shapes: obs %s, full %s", data_array.shape, data_array_full.shape)

    def _env_cfg(key: str, default):
        return OmegaConf.select(cfg, f"env.{key}", default=default)

    logger.info("Computing train / validation / holdout timeline...")
    dates_np = np.asarray(dates_ordinal, dtype=np.int64).reshape(-1)
    num_days = int(data_array.shape[0])
    ctx = int(_env_cfg("context_window_days", 151))
    trading_days = int(_env_cfg("trading_period_days", 364))
    settlement = int(_env_cfg("settlement_period_days", 0))
    ep_cal = OmegaConf.select(cfg, "env.episode_calendar_days", default=None)
    ep_cal_i = int(ep_cal) if ep_cal is not None else trading_days
    val_mult = float(_env_cfg("validation_reserve_multiplier", 2.5))

    split = compute_train_val_holdout_split(
        num_days=num_days,
        dates_ordinal=dates_np,
        context_window_days=ctx,
        episode_calendar_days=ep_cal_i,
        settlement_period_days=settlement,
        validation_reserve_multiplier=val_mult,
    )
    
    split_info = [
        "",
        "Data Split (Three-Tier Strategy):",
        f"  Total days: {num_days}",
        f"  Training indices:   [0, {split.train_end})",
        f"  Validation indices: [{split.val_env_start}, {split.val_end})",
        f"  Holdout indices:    [{split.holdout_env_start}, {split.holdout_end})",
        "",
        "  [!] CRITICAL: Holdout data is NEVER used during training!",
        "             It is reserved EXCLUSIVELY for committee testing.",
        ""
    ]
    logger.info("\n".join(split_info))

    train_obs, train_full, dates_train = slice_trading_timeline(
        data_array, data_array_full, dates_np, 0, split.train_end
    )
    val_obs, val_full, dates_val = slice_trading_timeline(
        data_array, data_array_full, dates_np, split.val_env_start, split.val_end
    )
    holdout_obs, holdout_full, dates_holdout = slice_trading_timeline(
        data_array, data_array_full, dates_np, split.holdout_env_start, split.holdout_end
    )

    if compat_mode:
        _phase_log("Phase 2: Environment Setup")
    logger.info("Creating trading environments (train + validation; holdout excluded)...")

    def _make_env(data_obs, data_f, dates_ord, *, is_training: bool):
        return TradingEnv(
            data_array=data_obs,
            data_array_full=data_f,
            norm_stats=norm_stats,
            context_window_days=ctx,
            trading_period_days=trading_days,
            settlement_period_days=settlement,
            episode_calendar_days=_env_cfg("episode_calendar_days", None),
            min_holding_period=_env_cfg("min_holding_period", 30),
            max_positions=_env_cfg("max_positions", 10),
            inaction_penalty=_env_cfg("inaction_penalty", 0.0),
            coefficient_threshold=_env_cfg("coefficient_threshold", 1.0),
            min_sale_target=_env_cfg("min_sale_target", 10.0),
            max_sale_target=_env_cfg("max_sale_target", 50.0),
            investable_start_col=_env_cfg("investable_start_col", 9),
            num_investable_stocks=_env_cfg("num_investable_stocks", 108),
            loss_penalty_multiplier=_env_cfg(
                "loss_penalty_multiplier", DEFAULT_LOSS_PENALTY_MULTIPLIER
            ),
            hurdle_rate=_env_cfg("hurdle_rate", DEFAULT_HURDLE_RATE),
            conviction_scaling_power=_env_cfg(
                "conviction_scaling_power", DEFAULT_CONVICTION_SCALING_POWER
            ),
            observation_noise_std=_env_cfg("observation_noise_std", 0.01),
            is_training=is_training,
            dates_ordinal=dates_ord,
            episode_reward_multiplier=_env_cfg(
                "episode_reward_multiplier", DEFAULT_EPISODE_REWARD_MULTIPLIER
            ),
        )

    env = _make_env(train_obs, train_full, dates_train, is_training=True)
    val_env = _make_env(val_obs, val_full, dates_val, is_training=False)
    holdout_env = _make_env(holdout_obs, holdout_full, dates_holdout, is_training=False)

    state = env.reset(key)
    logger.info("Train env reset OK; obs shape: %s", state.obs.shape)
    key, vkey = jax.random.split(key)
    vstate = val_env.reset(vkey)
    logger.info("Validation env reset OK; obs shape: %s", vstate.obs.shape)

    actor, critic = build_networks(cfg)
    agent = TradingAgent(
        actor_network=actor,
        critic_network=critic,
        exploration_noise=float(OmegaConf.select(cfg, "agent.exploration_noise", default=0.1)),
        discount=float(OmegaConf.select(cfg, "agent.discount", default=0.99)),
        tau=float(OmegaConf.select(cfg, "agent.tau", default=0.005)),
        min_sale_target=float(_env_cfg("min_sale_target", 10.0)),
        max_sale_target=float(_env_cfg("max_sale_target", 50.0)),
    )

    akey = jax.random.PRNGKey(int(cfg.seed) + 1)
    _ = agent.init(env.obs_space, env.action_space, akey)
    if compat_mode:
        _phase_log("Phase 3: Workflow Initialization")
    logger.info("TradingAgent init OK; starting TradingERLWorkflow.")

    from eigen3.erl.hall_of_fame import HallOfFame

    hof_capacity = int(OmegaConf.select(cfg, "population.hof_capacity", default=10))

    run_id = os.environ.get("EIGEN3_RUN_STAMP", datetime.now().strftime("%Y%m%d_%H%M%S"))
    eval_dir = Path(os.environ.get("EIGEN3_EVAL_DIR", str(artifact_root / "evaluation_results")))
    if resume_dir is not None:
        checkpoint_root = resume_dir.parent
        run_checkpoint_dir = resume_dir
    else:
        checkpoint_root = default_checkpoint_root
        run_checkpoint_dir = checkpoint_root / run_name if compat_mode else Path(
            hydra.utils.to_absolute_path("checkpoints")
        )
    checkpoint_dir = run_checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    artifact_mgr: CompatArtifactManager | None = None
    if compat_mode:
        artifact_mgr = CompatArtifactManager(
            artifact_root=artifact_root,
            eval_dir=eval_dir,
            checkpoint_root=checkpoint_root,
            run_name=run_name,
            run_id=run_id,
        )
        artifact_mgr.write_last_run()

    hof = HallOfFame(
        capacity=hof_capacity,
        checkpoint_dir=checkpoint_dir,
        cloud_sync=cloud_sync,
        cloud_prefix=f"{cloud_project}/{run_name}/hall_of_fame",
    )
    hof.load()
    if cloud_sync.provider == "local":
        logger.info(
            "Cloud sync: DISABLED (local-only). HoF data stays under %s; nothing is uploaded. "
            "To enable GCS: set CLOUD_PROVIDER=gcs, CLOUD_BUCKET, and credentials "
            "(or repo-root gcs-credentials.json).",
            checkpoint_dir,
        )
    else:
        logger.info(
            "Cloud sync: ENABLED (Google Cloud Storage). bucket=%r HoF prefix=%r",
            cloud_sync.bucket_name,
            hof.cloud_prefix,
        )
    logger.info(
        "HoF ready: capacity=%d, loaded=%d entries",
        hof.capacity,
        len(hof),
    )

    g15_capacity = int(OmegaConf.select(cfg, "population.global_15_capacity", default=15))
    g15_dir = artifact_root / "global_15"
    global_fifteen = GlobalFifteen(
        capacity=g15_capacity,
        checkpoint_dir=g15_dir,
        cloud_sync=cloud_sync,
        cloud_prefix=f"{cloud_project}/global_15",
    )
    global_fifteen.load()
    logger.info(
        "Global 15 ready: capacity=%d, loaded=%d entries (local=%s, cloud_prefix=%s)",
        global_fifteen.capacity,
        len(global_fifteen),
        g15_dir,
        global_fifteen.cloud_prefix,
    )

    wf_cfg = build_trading_workflow_config(cfg)
    evaluator = MagicMock()
    workflow = create_trading_workflow(
        env,
        agent,
        evaluator,
        config=wf_cfg,
        seed=int(cfg.seed),
        val_env=val_env,
        holdout_env=holdout_env,
        hall_of_fame=hof,
    )

    log_vram = bool(OmegaConf.select(cfg, "logging.log_gpu_memory", default=True))
    vram_interval = int(OmegaConf.select(cfg, "logging.log_gpu_memory_every_n_generations", default=0))
    if log_vram:
        log_gpu_memory_report(logger, "workflow initialized (replay + compiled graphs warm)")

    num_gen = int(OmegaConf.select(cfg, "population.total_generations", default=100))
    save_checkpoints = bool(OmegaConf.select(cfg, "population.save_checkpoints", default=True))
    save_training_state = bool(
        OmegaConf.select(cfg, "population.save_training_state", default=True)
    )
    save_interval = max(1, int(OmegaConf.select(cfg, "population.save_interval", default=1)))

    start_gen = 0
    best_score = float("-inf")
    if resume_dir is not None:
        ckpt_file = training_state_path(checkpoint_dir)
        if not ckpt_file.is_file():
            raise FileNotFoundError(
                f"Resume directory has no training checkpoint ({ckpt_file}). "
                "Train with population.save_training_state=true (default) so "
                "training_state.pkl is written every save_interval generations."
            )
        if compat_mode:
            _phase_log("Phase 3b: Resume")
        ck_info = load_training_checkpoint(
            ckpt_file, workflow, restore_replay_buffer=False
        )
        best_score = float(ck_info["best_score"])
        meta_seed = int(ck_info["meta"].get("seed", int(cfg.seed)))
        if meta_seed != int(cfg.seed):
            logger.warning(
                "Checkpoint seed %s != config seed %s (continuing PRNG from checkpoint).",
                meta_seed,
                cfg.seed,
            )
        start_gen = int(workflow.generation)
        if start_gen >= num_gen:
            logger.info(
                "Checkpoint generation %s >= population.total_generations=%s; nothing to run.",
                start_gen,
                num_gen,
            )

    logger.info(
        "Running generations %s..%s of %s (TradingERLWorkflow)...",
        start_gen + 1,
        num_gen,
        num_gen,
    )
    if compat_mode:
        _phase_log("Phase 4: Training Loop")
    all_metrics: list[dict[str, Any]] = []
    last: dict[str, Any] = {}
    prev_metrics: Optional[dict[str, Any]] = None
    prev_eval: Optional[dict[str, Any]] = None
    best_bnh_history: list[tuple[int, float]] = []
    running_gen_seconds = 0.0
    try:
        for gen in range(start_gen, num_gen):
            t_gen_wall_start = time.perf_counter()
            buffer_size = 0 if workflow._replay_buffer is None else int(workflow._replay_buffer.size)
            buffer_cap = workflow.config.replay_buffer_size
            pct = 100.0 * buffer_size / buffer_cap if buffer_cap > 0 else 0.0
            print("\n============================================================")
            print(f"Generation {gen + 1} | Runway: {num_gen - gen}")
            print(f"Buffer: {buffer_size:,} / {buffer_cap:,} ({pct:.1f}%)")
            print("============================================================")
            
            metrics = workflow.run_generation()
            if log_vram and should_log_gpu_memory_this_generation(vram_interval, gen):
                log_gpu_memory_report(logger, f"after generation {gen + 1}/{num_gen}")
            all_metrics.append(metrics)

            top5_evals = _top5_eval_rows_from_metrics(metrics)
            bh_usd = metrics.get("top5_bh_excess_usd") or []
            if bh_usd:
                best_bnh_history.append((int(metrics.get("generation", gen + 1)), float(bh_usd[0])))
            progress_eval = top5_evals[0][2] if top5_evals else {}
            best_params_for_progress = jax.tree.map(lambda x: x[metrics["top5_indices"][0]], workflow._stacked_params) if top5_evals else workflow.get_last_best_agent()
            
            t_gen_wall_s = time.perf_counter() - t_gen_wall_start
            gens_this_session = gen - start_gen + 1
            _print_generation_summary(
                gen=gen + 1,
                num_gen=num_gen,
                metrics=metrics,
                progress_eval=progress_eval,
                prev_metrics=prev_metrics,
                prev_eval=prev_eval,
                avg_gen_seconds=(
                    (running_gen_seconds / (gens_this_session - 1))
                    if gens_this_session > 1
                    else None
                ),
                top5_evals=top5_evals,
                wall_clock_s=t_gen_wall_s,
                best_bnh_history=best_bnh_history,
            )
            running_gen_seconds += t_gen_wall_s
            prev_metrics = metrics
            prev_eval = progress_eval
            if artifact_mgr is not None:
                artifact_mgr.append_metric(metrics)
                score = float(metrics.get("max_fitness", float("-inf")))
                if score > best_score:
                    best_score = score
                    best_params = best_params_for_progress
                    if save_checkpoints:
                        logger.info(
                            "✓ New best! Agent %d with Fitness: %.2f",
                            metrics.get("best_agent_idx", -1), score
                        )
                        logger.info("  Saving checkpoint with new best agent...")
                        best_path = artifact_mgr.save_best_agent(
                            best_params,
                            generation=int(metrics.get("generation", gen + 1)),
                            score=score,
                        )
                        eval_payload = _build_eval_payload_from_metrics(metrics)
                        eval_payload["run_name"] = run_name
                        eval_payload["generation"] = int(metrics.get("generation", gen + 1))
                        eval_paths = artifact_mgr.write_evaluation_bundle(eval_payload)
                        logger.info(
                            "✓ Saved locally | eval: %s (HoF cloud upload best-effort, non-blocking)",
                            eval_paths["json"],
                        )
                    else:
                        logger.info(
                            "New best (gen=%s score=%.6f); checkpoint save skipped "
                            "(population.save_checkpoints=false)",
                            metrics.get("generation", gen + 1),
                            score,
                        )

            if save_training_state and workflow.generation > 0:
                if workflow.generation % save_interval == 0:
                    save_training_checkpoint(
                        checkpoint_dir,
                        workflow,
                        best_score=best_score,
                        run_name=run_name,
                        seed=int(cfg.seed),
                    )

            _maybe_run_hof_gauntlet(
                workflow=workflow,
                hof=hof,
                global_fifteen=global_fifteen,
                cfg=cfg,
                run_name=run_name,
                eval_dir=eval_dir,
                split=split,
            )

        last = all_metrics[-1] if all_metrics else {}
        logger.info(
            "Training finished. Last generation: mean_fitness=%.4f max_fitness=%.4f "
            "total_env_steps=%s",
            last.get("mean_fitness", float("nan")),
            last.get("max_fitness", float("nan")),
            last.get("total_env_steps", "?"),
        )

        out_dir: str | None = None
        try:
            from hydra.core.hydra_config import HydraConfig

            out_dir = HydraConfig.get().runtime.output_dir
            logger.info("Hydra output directory: %s", out_dir)
        except Exception:
            pass

        use_tb = OmegaConf.select(cfg, "logging.use_tensorboard", default=False)
        if use_tb:
            tb_dir = OmegaConf.select(cfg, "logging.tensorboard_log_dir", default="")
            if tb_dir:
                logger.info("TensorBoard: tensorboard --logdir %s", tb_dir)

        if artifact_mgr is not None:
            if compat_mode:
                _phase_log("Phase 5: Finalization")
            artifact_mgr.write_last_run()
            artifact_mgr.write_run_summary(
                last_metrics=last,
                hydra_output_dir=out_dir,
                config_yaml=OmegaConf.to_yaml(cfg, resolve=True),
            )

        return all_metrics
    finally:
        if hof.cloud_sync is not None:
            hof.cloud_sync.shutdown(wait=False)
