"""Evaluation script for trained Eigen3 models.

Loads a checkpoint (flax msgpack), builds the holdout environment, and runs
deterministic episodes to produce per-episode and aggregate metrics.  Results
are saved as JSON and optionally printed to the console.
"""

import argparse
import csv
import json
import logging
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from eigen3.data import load_trading_data, create_synthetic_data
from eigen3.data.splits import compute_train_val_holdout_split, slice_trading_timeline
from eigen3.environment.trading_env import TradingEnv
from eigen3.agents import TradingAgent
from eigen3.models.actor import Actor
from eigen3.models.critic import DoubleCritic
from evorl.sample_batch import SampleBatch

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Eigen3 trading agent on holdout data",
    )
    parser.add_argument(
        "--checkpoint_path", type=str, required=True,
        help="Path to agent checkpoint (.msgpack or .npz)",
    )
    parser.add_argument(
        "--data_path", type=str, default="Eigen3_Processed_OUTPUT.pkl",
        help="Path to trading data (directory for Eigen2, .pkl/.csv for mono)",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=10,
        help="Number of evaluation episodes to run",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output_dir", type=str, default="evaluation_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="",
        help="Optional run name for output filenames (defaults to checkpoint stem)",
    )

    # Environment overrides (should match training config)
    parser.add_argument("--context_window_days", type=int, default=151)
    parser.add_argument("--trading_period_days", type=int, default=364)
    parser.add_argument("--settlement_period_days", type=int, default=0)
    parser.add_argument("--min_holding_period", type=int, default=30)
    parser.add_argument("--max_positions", type=int, default=10)
    parser.add_argument("--num_columns", type=int, default=18)
    parser.add_argument("--num_features_obs", type=int, default=1)
    parser.add_argument("--num_investable_stocks", type=int, default=1)
    parser.add_argument("--investable_start_col", type=int, default=0)
    parser.add_argument("--mono_num_channels", type=int, default=18)
    parser.add_argument("--mono_csv_header", type=int, default=0)
    parser.add_argument("--validation_reserve_multiplier", type=float, default=2.5)
    parser.add_argument("--min_sale_target", type=float, default=10.0)
    parser.add_argument("--max_sale_target", type=float, default=50.0)
    parser.add_argument("--hurdle_rate", type=float, default=0.005)
    parser.add_argument("--loss_penalty_multiplier", type=float, default=1.25)
    parser.add_argument("--conviction_scaling_power", type=float, default=1.0)
    parser.add_argument("--episode_reward_multiplier", type=float, default=1.0)

    # Network architecture (must match training / checkpoint)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument(
        "--no-remat",
        action="store_true",
        help="Set use_remat=False on Actor/DoubleCritic (default matches train: remat on).",
    )

    return parser.parse_args()


def _load_checkpoint(path: str, agent: TradingAgent, agent_state):
    """Load agent parameters from checkpoint on disk."""
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if ckpt_path.suffix == ".npz":
        data = np.load(str(ckpt_path), allow_pickle=True)
        params = data["params"].item()
        from eigen3.agents import TradingNetworkParams
        tm = getattr(jax.tree, "map", None) or jtu.tree_map
        loaded = TradingNetworkParams(
            actor_params=tm(jnp.array, params["actor_params"]),
            critic_params=tm(jnp.array, params["critic_params"]),
            target_actor_params=tm(jnp.array, params["target_actor_params"]),
            target_critic_params=tm(jnp.array, params["target_critic_params"]),
        )
    else:
        from eigen3.agents.trading_agent import trading_params_from_msgpack_bytes

        raw = ckpt_path.read_bytes()
        loaded = trading_params_from_msgpack_bytes(raw, agent_state.params)

    from evorl.agent import AgentState
    return AgentState(params=loaded)


def run_episode(env: TradingEnv, agent: TradingAgent, agent_state, key):
    """Run a single deterministic evaluation episode, returning metrics dict."""
    key, reset_key = jax.random.split(key)
    state = env.reset(reset_key)

    total_reward = 0.0
    steps = 0

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
    return {
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


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    logger.info("Evaluating checkpoint: %s", args.checkpoint_path)
    key = jax.random.PRNGKey(args.seed)

    # --- Load data ---
    data_path = Path(args.data_path)
    if data_path.is_file() and data_path.suffix.lower() in (".pkl", ".pickle", ".csv"):
        logger.info("Loading mono table: %s", data_path)
        data_obs, data_full, norm_stats, dates_ordinal = load_trading_data(
            str(data_path),
            mono_num_channels=args.mono_num_channels,
            mono_csv_header=args.mono_csv_header,
        )
    elif data_path.is_dir():
        logger.info("Loading Eigen2 bundle: %s", data_path)
        data_obs, data_full, norm_stats, dates_ordinal = load_trading_data(str(data_path))
    else:
        logger.warning("Data path not found; using synthetic data for demo")
        data_obs, data_full, norm_stats, dates_ordinal = create_synthetic_data(
            num_days=2000,
            num_columns=args.num_columns,
            num_features_obs=args.num_features_obs,
            seed=args.seed,
        )

    dates_np = np.asarray(dates_ordinal, dtype=np.int64).reshape(-1)
    num_days = int(data_obs.shape[0])
    ep_cal = args.trading_period_days

    split = compute_train_val_holdout_split(
        num_days=num_days,
        dates_ordinal=dates_np,
        context_window_days=args.context_window_days,
        episode_calendar_days=ep_cal,
        settlement_period_days=args.settlement_period_days,
        validation_reserve_multiplier=args.validation_reserve_multiplier,
    )
    logger.info(
        "Split: holdout trading [%d, %d) (%d rows), holdout env [%d, %d) (%d rows)",
        split.holdout_start,
        split.holdout_end,
        split.holdout_rows,
        split.holdout_env_start,
        split.holdout_end,
        split.holdout_env_rows,
    )

    holdout_obs, holdout_full, dates_holdout = slice_trading_timeline(
        data_obs, data_full, dates_np, split.holdout_env_start, split.holdout_end,
    )

    # --- Build environment ---
    env = TradingEnv(
        data_array=holdout_obs,
        data_array_full=holdout_full,
        norm_stats=norm_stats,
        context_window_days=args.context_window_days,
        trading_period_days=args.trading_period_days,
        settlement_period_days=args.settlement_period_days,
        min_holding_period=args.min_holding_period,
        max_positions=args.max_positions,
        inaction_penalty=0.0,
        coefficient_threshold=1.0,
        min_sale_target=args.min_sale_target,
        max_sale_target=args.max_sale_target,
        investable_start_col=args.investable_start_col,
        num_investable_stocks=args.num_investable_stocks,
        loss_penalty_multiplier=args.loss_penalty_multiplier,
        hurdle_rate=args.hurdle_rate,
        conviction_scaling_power=args.conviction_scaling_power,
        observation_noise_std=0.0,
        is_training=False,
        dates_ordinal=dates_holdout,
        episode_reward_multiplier=args.episode_reward_multiplier,
    )

    # --- Build agent & load checkpoint ---
    nc = int(data_obs.shape[1])
    nf = int(data_obs.shape[2])
    use_remat = not args.no_remat
    actor = Actor(
        num_columns=nc,
        num_features=nf,
        num_investable_stocks=args.num_investable_stocks,
        investable_start_col=args.investable_start_col,
        column_chunk_size=args.chunk_size,
        use_remat=use_remat,
        min_sale_target=args.min_sale_target,
        max_sale_target=args.max_sale_target,
    )
    critic = DoubleCritic(
        num_columns=nc,
        num_features=nf,
        num_investable_stocks=args.num_investable_stocks,
        column_chunk_size=args.chunk_size,
        use_remat=use_remat,
    )
    agent = TradingAgent(actor_network=actor, critic_network=critic)

    key, init_key = jax.random.split(key)
    agent_state = agent.init(env.obs_space, env.action_space, init_key)
    agent_state = _load_checkpoint(args.checkpoint_path, agent, agent_state)
    logger.info("Checkpoint loaded")

    # --- Run episodes ---
    episode_results = []
    for i in range(args.num_episodes):
        key, ep_key = jax.random.split(key)
        metrics = run_episode(env, agent, agent_state, ep_key)
        episode_results.append(metrics)
        logger.info(
            "Episode %d/%d: reward=%.4f  trades=%d  win_rate=%.1f%%  pnl=%.4f",
            i + 1, args.num_episodes,
            metrics["total_reward"],
            metrics["num_trades"],
            metrics["win_rate"] * 100,
            metrics["total_pnl"],
        )

    # --- Aggregate ---
    rewards = [e["total_reward"] for e in episode_results]
    pnls = [e["total_pnl"] for e in episode_results]
    win_rates = [e["win_rate"] for e in episode_results]

    summary = {
        "checkpoint": args.checkpoint_path,
        "data_path": args.data_path,
        "num_episodes": args.num_episodes,
        "seed": args.seed,
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "reward_min": float(np.min(rewards)),
        "reward_max": float(np.max(rewards)),
        "pnl_mean": float(np.mean(pnls)),
        "pnl_std": float(np.std(pnls)),
        "win_rate_mean": float(np.mean(win_rates)),
        "episodes": episode_results,
    }

    logger.info("=" * 60)
    logger.info("Evaluation Summary (%d episodes)", args.num_episodes)
    logger.info("  Reward:   %.4f +/- %.4f  [%.4f, %.4f]",
                summary["reward_mean"], summary["reward_std"],
                summary["reward_min"], summary["reward_max"])
    logger.info("  PnL:      %.4f +/- %.4f", summary["pnl_mean"], summary["pnl_std"])
    logger.info("  Win Rate: %.1f%%", summary["win_rate_mean"] * 100)
    logger.info("=" * 60)

    # --- Save ---
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name.strip() or Path(args.checkpoint_path).stem
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{run_name}_{stamp}"
    out_json = out_dir / f"evaluation_{base}.json"
    out_txt = out_dir / f"evaluation_{base}.txt"
    out_summary_csv = out_dir / f"summary_{base}.csv"
    out_trades_csv = out_dir / f"trades_{base}.csv"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    summary_fields = [
        "checkpoint",
        "data_path",
        "num_episodes",
        "seed",
        "reward_mean",
        "reward_std",
        "reward_min",
        "reward_max",
        "pnl_mean",
        "pnl_std",
        "win_rate_mean",
    ]
    with open(out_summary_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerow({k: summary.get(k, "") for k in summary_fields})

    trade_fields = [
        "episode_index",
        "total_reward",
        "steps",
        "num_trades",
        "num_wins",
        "num_losses",
        "total_gain_pct",
        "total_pnl",
        "peak_capital_employed",
        "days_with_positions",
        "days_without_positions",
        "win_rate",
    ]
    with open(out_trades_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=trade_fields)
        writer.writeheader()
        for idx, row in enumerate(summary["episodes"], start=1):
            out_row = dict(row)
            out_row["episode_index"] = idx
            writer.writerow({k: out_row.get(k, "") for k in trade_fields})

    lines = [
        f"Run: {run_name}",
        f"Checkpoint: {args.checkpoint_path}",
        f"Episodes: {args.num_episodes}",
        f"Reward mean/std: {summary['reward_mean']:.6f} / {summary['reward_std']:.6f}",
        f"PnL mean/std: {summary['pnl_mean']:.6f} / {summary['pnl_std']:.6f}",
        f"Win rate mean: {summary['win_rate_mean'] * 100:.2f}%",
        f"JSON: {out_json}",
        f"Summary CSV: {out_summary_csv}",
        f"Trades CSV: {out_trades_csv}",
    ]
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    logger.info("Results saved to %s", out_json)
    logger.info("Summary CSV saved to %s", out_summary_csv)
    logger.info("Trades CSV saved to %s", out_trades_csv)
    logger.info("Text summary saved to %s", out_txt)


if __name__ == "__main__":
    main()
