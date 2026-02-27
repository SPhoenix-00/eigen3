# Eigen2 vs Eigen3 Delta (Sync Reference)

This document lists changes in Eigen2 since Eigen3 was branched, and the corresponding Eigen3 files to update. Use it to sync Eigen3 with Eigen2 behavior.

**Eigen2 path:** `d:\GitHub\eigen2` (or `C:\Users\antoi\Documents\GitHub\eigen2`)  
**Last updated:** 2025-02-27

---

## 1. Data

### 1.1 Constants and format (Eigen2)

| Item | Eigen2 | Eigen3 (before sync) |
|------|--------|----------------------|
| TOTAL_COLUMNS | 117 (skinny) | 669 in docs/config |
| CONTEXT_WINDOW_DAYS | 151 | 504 |
| FEATURES_PER_CELL | 5 | 5 |
| data_array shape | [T, 117, 5] | [T, 669, 5] in places |
| data_array_full | [T, 117, 9] | [T, 669, 9] |
| Normalization | Identity (mean=0, std=1); Instance Norm in network | Configurable |
| Train/val/holdout | Three-tier: train, validation (503d), committee holdout (252d) | train_split fraction |

### 1.2 Eigen2 data/loader.py

- **StockDataLoader**: `load_csv()` (pkl or CSV), `parse_cell_data_full` (9 features), `parse_cell_data` (5 features from full indices [1,4,6,7,8]), `extract_features()`, `create_train_val_split()` (three-tier), `compute_normalization_stats()` returns identity (0, 1), `load_and_prepare(quiet=)`, `get_window()`, `normalize_window()`, `dates` array.
- **Output**: `data_array`, `data_array_full`, `normalization_stats` (identity), `dates`, `train_indices`, `val_indices`, `train_end_idx`, `val_start_idx`, `val_end_idx`, `column_names`.

### 1.3 Eigen3 files to update

- [eigen3/data/data_loader.py](eigen3/data/data_loader.py): Support 117 columns; add `load_trading_data` from pkl/csv matching Eigen2; identity norm option; three-tier split (validation_days, committee_holdout_days); return dates; DataConfig num_columns default 117, context_window_days 151.
- [eigen3/data/__init__.py](eigen3/data/__init__.py): Export any new loaders.
- [configs/env/trading.yaml](configs/env/trading.yaml): context_window_days 151, num_columns 117.

---

## 2. Environment

### 2.1 Eigen2 environment/trading_env.py

- **API**: Gymnasium `gym.Env`; constructor takes `data_array`, `dates`, `normalization_stats`, `start_idx`, `end_idx`, `trading_end_idx`, `data_array_full`, `is_training`, `consistency_mode`, `gauntlet_mode`, `maverick_mode`.
- **Position**: `stock_id`, `entry_price`, `coefficient`, `sale_target_pct`, `days_held`, `entry_day_idx`, `entry_date`; `sale_target_price` property.
- **Holding**: MIN_HOLDING_PERIOD (20), LIQUIDATION_WINDOW (10), MAX_HOLDING_PERIOD (30). No sell before day 20; days 21–30 can exit if target hit; day 30 forced liquidation.
- **Reward**: Hurdle rate (0.6%), conviction scaling (coefficient^1.25), asymmetric win/loss; forced exit penalty; consistency_mode loss multiplier; maverick_mode (lower hurdle, no forced exit penalty, FOMO/ROI).
- **Observation**: Identity norm then multiplicative observation noise (OBSERVATION_NOISE_STD 0.01) when `is_training`.
- **Methods**: `fast_step()` (no obs/info), `get_batch_observations(start_day_idx, count)` (zero-copy sliding window); `set_training_mode()`, `set_gauntlet_mode()`, `set_consistency_mode()`; `get_episode_summary()` (clears episode_actions/rewards), includes `peak_capital_employed`, `roi`, `market_return_pct` (maverick), `zero_trades_penalty` (gauntlet/consistency/normal).
- **Constants**: MAX_TRADES_PER_DAY (5), INACTION_PENALTY (0.0), FORCED_EXIT_PENALTY_PCT (0.01), ZERO_TRADES_PENALTY_*, HURDLE_RATE, CONVICTION_SCALING_POWER, etc.

### 2.2 Eigen3 files to update

- [eigen3/environment/trading_env.py](eigen3/environment/trading_env.py): Add constructor args for dates, trading_end_idx, is_training, consistency_mode, gauntlet_mode, maverick_mode. Implement min_holding_period (20), max_holding_period (30), liquidation window; hurdle + conviction scaling + forced exit penalty; observation noise when is_training; episode summary with peak_capital_employed, roi; optional fast_step and get_batch_observations (or document JAX equivalents). Keep JAX-native state and EvoRL Env interface.

---

## 3. Models

### 3.1 Eigen2 models/networks.py

- **FeatureExtractor**: Input [batch, context_days, num_columns, num_features]. **Instance normalization first** (per column, across time, affine=False), then CNN, then LSTM; gradient checkpointing off by default; output [batch, num_columns, lstm_output_size]. Average last 3 timesteps.
- **AttentionModule**: Cross or self; learnable query for cross; LayerNorm; `_apply_attention_dropout` (bernoulli mask + renormalize) when return_attention_weights.
- **Actor**: Self-attention (use_cross_attention=False) for “each stock attends to all columns”; investable slice Config.INVESTABLE_START_COL:INVESTABLE_END_COL+1 (9:117 → 108 stocks); **output head init**: coefficient head xavier gain 5, bias 0.5; sale target head bias = (MIN_SALE_TARGET+MAX_SALE_TARGET)/2; **coefficient**: ReLU then leverage_multiplier (optional) then **clamp [0, 100]**; no sigmoid on sale target, clip to [10, 50].
- **Critic**: Optional attention; **mean pool** over columns; concat state_features + action_flat → fc; gradient checkpointing.

### 3.2 Eigen3 files to update

- [eigen3/models/feature_extractor.py](eigen3/models/feature_extractor.py): Add **instance normalization first** (before CNN), same axis as Eigen2 (per column, across time). Input shape [batch, context_days, num_columns, num_features] (or document current convention). Default num_columns 117.
- [eigen3/models/attention.py](eigen3/models/attention.py): Attention dropout (bernoulli + renormalize) when returning weights; support self vs cross.
- [eigen3/models/actor.py](eigen3/models/actor.py): Self-attention mode; investable slice from config (9:117); **output head init** (Flax kernel init gain 5, bias 0.5 and center); coefficient ReLU then **clamp max 100**; no sigmoid sale target.
- [eigen3/models/critic.py](eigen3/models/critic.py): Mean pool over columns; optional attention.

---

## 4. Agent and workflow

### 4.1 Eigen2 models/ddpg_agent.py

- **DDPGAgent**: agent_id, is_elite; move_to_device(device, recreate_optimizers); **GradScaler** (mixed precision); **gradient accumulation** (accumulate=True); noise_decay (NOISE_DECAY 0.99995, MIN_NOISE 0.01); **select_actions_batch(states, add_noise)**; **load_weights_only(path)**; **clone()** (deep copy, no param sharing); **get_stats()** (update_count, noise_scale, avg losses); loss history deque maxlen 1000; coefficient clip 0–100 after noise.

### 4.2 Eigen3 files to update

- [eigen3/agents/trading_agent.py](eigen3/agents/trading_agent.py): Exploration noise decay (config); clip coefficient to 100 after noise; get_stats equivalent (update_count, noise_scale, avg losses); document clone/load_weights_only for EvoRL (or implement if EvoRL supports). Ensure action clip [0, 100] for coefficient.

---

## 5. Config and scripts

### 5.1 Eigen2 utils/config.py (excerpts)

- CONTEXT_WINDOW_DAYS=151, TOTAL_COLUMNS=117, INVESTABLE_START_COL=9, NUM_INVESTABLE_STOCKS=108, FEATURES_PER_CELL=5.
- MIN_HOLDING_PERIOD=20, MAX_HOLDING_PERIOD=30, MAX_TRADES_PER_DAY=5, INACTION_PENALTY=0.0, HURDLE_RATE=0.006, CONVICTION_SCALING_POWER=1.25, FORCED_EXIT_PENALTY_PCT=0.01.
- TRADING_PERIOD_DAYS=125, SETTLEMENT_PERIOD_DAYS=30.
- VALIDATION_DAYS=503, COMMITTEE_HOLDOUT_DAYS=252.
- OBSERVATION_NOISE_STD=0.01.
- ACTOR_HIDDEN_DIMS=[256,128,64], CRITIC_HIDDEN_DIMS=[256,128], TAU=0.005, GAMMA=0.99, NOISE_SCALE=0.125, NOISE_DECAY=0.99995, MIN_NOISE=0.01.

### 5.2 Eigen3 files to update

- [configs/agent/trading_erl.yaml](configs/agent/trading_erl.yaml): Align with Eigen2 (context_window 151, 117 cols, holding periods, hurdle, noise decay, etc.).
- [configs/env/trading.yaml](configs/env/trading.yaml): context_window_days 151, num_columns 117, min_holding_period 20, max_holding_period 30, inaction_penalty 0.0, observation_noise_std 0.01.
- [scripts/train.py](scripts/train.py): Wire to load data (load_trading_data or load_eigen2_data), create env, build workflow from config, run training loop, checkpointing, logging.

---

## 6. New Eigen2 components (no direct Eigen3 port yet)

- **committee/**: Committee selection, validation, optimization (Python; could be ported later).
- **erl/hall_of_fame.py**, **erl/global_hof.py**: Hall of Fame, league rules.
- **training/**: erl_trainer, fitness, breakthrough, validation_slices, workers, checkpoint, multi_agent, local_evaluator, episode, etc. Eigen3 uses EvoRL for workflow; document mapping from ERLTrainer to EvoRL workflow.
- **inference/**: engine, state. Eigen3 can add inference script that loads checkpoint and runs agent.
- **global50.py**, **evaluate_*.py**, **main.py**: Entry points; Eigen3 equivalent: scripts/train.py, scripts/evaluate.py.

---

## 7. Summary checklist

- [x] Data: 117 columns, 151 context, identity norm, three-tier split, dates.
- [ ] Env: dates, trading_end_idx, min/max holding, hurdle, conviction, forced exit, observation noise, gauntlet/consistency/maverick, episode summary (peak capital, ROI).
  - [x] Core: min/max holding, hurdle, conviction, forced exit, observation noise.
  - [ ] Remaining: dates array, trading_end_idx, gauntlet/consistency/maverick modes, episode summary (peak capital, ROI).
- [x] Models: Instance norm first in FeatureExtractor; Actor self-attention, head init, coefficient clamp 100; Critic mean pool; attention dropout.
- [ ] Agent: Noise decay, coefficient clip 100, get_stats.
  - [x] Coefficient clip 100 after noise.
  - [ ] Remaining: Noise decay (NOISE_DECAY 0.99995, MIN_NOISE 0.01), get_stats (update_count, noise_scale, avg losses).
- [x] Config: All above constants in YAML.
- [x] Scripts: train.py wired end-to-end (data + env + all config params forwarded).
- [x] Bug fixes: workflow attribute names, stop_gradient, AgentState wrapping, Critic default, norm_stats None guard, stale 669/504 dimensions.
