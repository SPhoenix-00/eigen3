"""Hydra-driven training: data, envs, Hall of Fame, ``TradingERLWorkflow``."""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
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
from eigen3.data.splits import compute_train_val_holdout_split, slice_trading_timeline
from eigen3.environment.trading_env import TradingEnv
from eigen3.models import Actor, DoubleCritic
from eigen3.utils.gpu_memory import log_gpu_memory_report, should_log_gpu_memory_this_generation
from eigen3.workflows import TradingWorkflowConfig, create_trading_workflow
from evorl.agent import AgentState
from evorl.sample_batch import SampleBatch

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


def _compat_mode_enabled() -> bool:
    return os.environ.get("EIGEN3_COMPAT_MODE", "1") != "0"


def _phase_log(label: str) -> None:
    logger.info("========== %s ==========", label)


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
    t_eval = float(metrics.get("timing_eval_s", 0.0))
    t_train = float(metrics.get("timing_train_s", 0.0))
    t_collect = float(metrics.get("timing_collect_s", 0.0))
    t_evolve = float(metrics.get("timing_evolve_s", 0.0))
    t_hof = float(metrics.get("timing_hof_s", 0.0))
    t_total = float(metrics.get("timing_total_s", t_eval + t_train + t_collect + t_evolve + t_hof))
    remaining = max(0, num_gen - gen)
    eta_s = (avg_gen_seconds or t_total) * remaining

    print("\n--- Generation Timing Breakdown ---")
    print(f"  Collect:    {t_collect:.2f}s ({_pct(t_collect, t_total):.1f}%)")
    print(f"  Evaluation: {t_eval:.2f}s ({_pct(t_eval, t_total):.1f}%)")
    print(f"  Training:   {t_train:.2f}s ({_pct(t_train, t_total):.1f}%)")
    print(f"  HoF:        {t_hof:.2f}s ({_pct(t_hof, t_total):.1f}%)")
    print(f"  Evolution:  {t_evolve:.2f}s ({_pct(t_evolve, t_total):.1f}%)")
    print(f"  Total:      {t_total:.2f}s")
    print("-" * 40)
    print("\n" + "=" * 70)
    print(f"  Gen {gen}/{num_gen} | {_fmt_time(t_total)} | ETA {_fmt_time(eta_s)}")
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
        f"  Train: collect/agent={metrics.get('collect_reward_mean_per_agent', 0.0):.2f}  "
        f"actor={metrics.get('mean_actor_loss', float('nan')):.4f}  "
        f"critic={metrics.get('mean_critic_loss', float('nan')):.4f}  "
        f"q={metrics.get('mean_mean_q', float('nan')):.4f}"
    )
    print("\n--- Validation Summary ---")
    print("Top 5 by Fitness:")
    if top5_evals:
        for rank, (idx, fitness, ev) in enumerate(top5_evals, 1):
            roi = ev.get('gain_pct_mean', 0.0)
            pnl = ev.get('pnl_mean', 0.0)
            wr = ev.get('win_rate_mean', 0.0) * 100.0
            r_mean = ev.get('reward_mean', 0.0)
            r_min = ev.get('reward_min', 0.0)
            print(f"  {rank}. Agent {idx:2d}: Fitness={fitness:8.2f}, Val=[mean:{r_mean:6.2f}, min:{r_min:6.2f}], ROI={roi:6.2f}%, PnL=${pnl:8.2f}, WR={wr:5.1f}%")
    else:
        print("  (No Top 5 stats available)")

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
    print("\n  LOOP PERFORMANCE")
    print("  -------------------------------------------------------")
    print(
        f"  Eval: {_fmt_time(t_eval)} ({_pct(t_eval, t_total):.0f}%)  |  "
        f"Train: {_fmt_time(t_train)} ({_pct(t_train, t_total):.0f}%)  |  "
        f"Collect: {_fmt_time(t_collect)} ({_pct(t_collect, t_total):.0f}%)  |  "
        f"Evolve: {_fmt_time(t_evolve)} ({_pct(t_evolve, t_total):.0f}%)"
    )
    print(
        f"  Buffer: {bsz:,}/{bcap:,} ({100.0 * bsz / bcap:.1f}%)  |  "
        f"Env steps: {metrics.get('total_env_steps', 0):,}"
    )
    print("\n" + "=" * 70 + "\n")


def _evaluate_agent_on_env(
    *,
    env: TradingEnv,
    agent: TradingAgent,
    params: Any,
    seed: int,
    num_episodes: int = 3,
) -> dict[str, Any]:
    key = jax.random.PRNGKey(seed)
    agent_state = AgentState(params=params)
    episodes: list[dict[str, Any]] = []

    for _ in range(num_episodes):
        key, reset_key = jax.random.split(key)
        state = env.reset(reset_key)
        total_reward = 0.0
        steps = 0
        # Batched host read: separate bool(done)/float(reward) sync twice per step and idle the GPU.
        while True:
            obs_batch = state.obs[None, ...]
            sample = SampleBatch(obs=obs_batch)
            key, act_key = jax.random.split(key)
            actions, _ = agent.evaluate_actions(agent_state, sample, act_key)
            action = actions[0]
            state = env.step(state, action)
            done_h, rew_h = jax.device_get((state.done, state.reward))
            total_reward += float(rew_h)
            steps += 1
            if bool(done_h):
                break

        es = state.env_state
        nt, nw, nl, tgp, tpnl, pce, dwp, dwo = jax.device_get(
            (
                es.num_trades,
                es.num_wins,
                es.num_losses,
                es.total_gain_pct,
                es.total_pnl,
                es.peak_capital_employed,
                es.days_with_positions,
                es.days_without_positions,
            )
        )
        episodes.append(
            {
                "total_reward": total_reward,
                "steps": steps,
                "num_trades": int(nt),
                "num_wins": int(nw),
                "num_losses": int(nl),
                "total_gain_pct": float(tgp),
                "total_pnl": float(tpnl),
                "peak_capital_employed": float(pce),
                "days_with_positions": int(dwp),
                "days_without_positions": int(dwo),
                "win_rate": float(nw) / max(int(nt), 1),
            }
        )

    rewards = np.asarray([e["total_reward"] for e in episodes], dtype=np.float64)
    pnls = np.asarray([e["total_pnl"] for e in episodes], dtype=np.float64)
    wins = np.asarray([e["win_rate"] for e in episodes], dtype=np.float64)
    trades = np.asarray([e["num_trades"] for e in episodes], dtype=np.float64)
    gains = np.asarray([e["total_gain_pct"] for e in episodes], dtype=np.float64)
    return {
        "num_episodes": int(num_episodes),
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "reward_min": float(rewards.min()),
        "reward_max": float(rewards.max()),
        "pnl_mean": float(pnls.mean()),
        "pnl_std": float(pnls.std()),
        "win_rate_mean": float(wins.mean()),
        "num_trades_mean": float(trades.mean()),
        "gain_pct_mean": float(gains.mean()),
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
        eval_episodes=int(OmegaConf.select(cfg, "population.eval_episodes", default=5)),
        conservative_k=int(OmegaConf.select(cfg, "population.conservative_k", default=3)),
        target_update_period=target_update_period,
        steps_per_agent=steps_per_agent,
        gradient_vmap_chunk_size=_effective_gradient_vmap_chunk_size(cfg),
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
    logger.info(run_config_summary(cfg))
    logger.debug("Full resolved configuration:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    logger.info("Setting random seed: %s", cfg.seed)
    key = jax.random.PRNGKey(int(cfg.seed))

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
    val_mult = float(_env_cfg("validation_reserve_multiplier", 1.5))

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

    from eigen3.erl.cloud_sync import CloudSync
    from eigen3.erl.hall_of_fame import HallOfFame

    hof_capacity = int(OmegaConf.select(cfg, "population.hof_capacity", default=10))
    cloud_project = OmegaConf.select(cfg, "population.cloud_project_name", default="eigen3")
    cloud_sync = CloudSync.from_env(project_name=cloud_project)

    run_name = str(OmegaConf.select(cfg, "run_name", default="run"))
    run_id = os.environ.get("EIGEN3_RUN_STAMP", datetime.now().strftime("%Y%m%d_%H%M%S"))
    artifact_root = Path(
        os.environ.get(
            "EIGEN3_ARTIFACT_ROOT",
            str(Path(hydra.utils.to_absolute_path("."))),
        )
    )
    eval_dir = Path(os.environ.get("EIGEN3_EVAL_DIR", str(artifact_root / "evaluation_results")))
    checkpoint_root = Path(
        os.environ.get("EIGEN3_CHECKPOINT_ROOT", str(artifact_root / "checkpoints"))
    )
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
        cloud_prefix=f"{cloud_project}/hall_of_fame",
    )
    hof.load()
    logger.info(
        "HoF ready: capacity=%d, loaded=%d, cloud=%s",
        hof.capacity,
        len(hof),
        cloud_sync.provider,
    )

    wf_cfg = build_trading_workflow_config(cfg)
    evaluator = MagicMock()
    workflow = create_trading_workflow(
        env,
        agent,
        evaluator,
        config=wf_cfg,
        seed=int(cfg.seed),
        eval_env=val_env,
        hall_of_fame=hof,
    )

    log_vram = bool(OmegaConf.select(cfg, "logging.log_gpu_memory", default=True))
    vram_interval = int(OmegaConf.select(cfg, "logging.log_gpu_memory_every_n_generations", default=0))
    progress_eval_episodes = int(OmegaConf.select(cfg, "logging.progress_eval_episodes", default=1))
    if log_vram:
        log_gpu_memory_report(logger, "workflow initialized (replay + compiled graphs warm)")

    num_gen = int(OmegaConf.select(cfg, "population.total_generations", default=100))
    save_checkpoints = bool(OmegaConf.select(cfg, "population.save_checkpoints", default=True))
    logger.info("Running %d generations (TradingERLWorkflow)...", num_gen)
    if compat_mode:
        _phase_log("Phase 4: Training Loop")
    all_metrics: list[dict[str, Any]] = []
    best_score = float("-inf")
    last: dict[str, Any] = {}
    prev_metrics: Optional[dict[str, Any]] = None
    prev_eval: Optional[dict[str, Any]] = None
    running_gen_seconds = 0.0
    try:
        for gen in range(num_gen):
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
            
            top5_evals = []
            logger.info(
                "\nValidation rollout for Top 5 agents (%d episode(s))...",
                max(1, progress_eval_episodes),
            )
            import sys
            sys.stdout.flush()
            
            for rank, idx in enumerate(metrics.get("top5_indices", [])):
                agent_params = jax.tree.map(lambda x: x[idx], workflow._stacked_params)
                ev = _evaluate_agent_on_env(
                    env=val_env,
                    agent=agent,
                    params=agent_params,
                    seed=int(cfg.seed) + gen + 500_000 + rank,
                    num_episodes=max(1, progress_eval_episodes),
                )
                top5_evals.append((idx, metrics["top5_fitness"][rank], ev))
                
            progress_eval = top5_evals[0][2] if top5_evals else {}
            best_params_for_progress = jax.tree.map(lambda x: x[metrics["top5_indices"][0]], workflow._stacked_params) if top5_evals else workflow.get_last_best_agent()
            
            _print_generation_summary(
                gen=gen + 1,
                num_gen=num_gen,
                metrics=metrics,
                progress_eval=progress_eval,
                prev_metrics=prev_metrics,
                prev_eval=prev_eval,
                avg_gen_seconds=(running_gen_seconds / gen) if gen > 0 else None,
                top5_evals=top5_evals,
            )
            running_gen_seconds += float(metrics.get("timing_total_s", 0.0))
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
                        _art_eps = min(
                            3, int(OmegaConf.select(cfg, "population.eval_episodes", default=5))
                        )
                        logger.info("  Artifact validation rollouts (%d episode(s))...", _art_eps)
                        eval_payload = _evaluate_agent_on_env(
                            env=val_env,
                            agent=agent,
                            params=best_params,
                            seed=int(cfg.seed) + gen + 1000,
                            num_episodes=_art_eps,
                        )
                        eval_payload["run_name"] = run_name
                        eval_payload["generation"] = int(metrics.get("generation", gen + 1))
                        eval_paths = artifact_mgr.write_evaluation_bundle(eval_payload)
                        logger.info(
                            "✓ Saved & syncing to cloud | eval: %s",
                            eval_paths["json"],
                        )
                    else:
                        logger.info(
                            "New best (gen=%s score=%.6f); checkpoint save skipped "
                            "(population.save_checkpoints=false)",
                            metrics.get("generation", gen + 1),
                            score,
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
            hof.cloud_sync.shutdown()
