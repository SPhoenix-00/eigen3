"""Central defaults for trading env reward shaping.

Hydra ``configs/env/*.yaml`` should keep the same values so CLI overrides stay
the single source at run time; these constants supply defaults for
``TradingEnv`` and for ``scripts/train.py`` when a key is absent.
"""

# Fraction of return (e.g. 0.005 = 0.5% hurdle before positive net reward)
DEFAULT_HURDLE_RATE: float = 0.005

# 1.0 = linear coefficient scaling (no superlinear conviction boost)
DEFAULT_CONVICTION_SCALING_POWER: float = 1.0

# > 1.0 magnifies negative net PnL in the reward (after hurdle)
DEFAULT_LOSS_PENALTY_MULTIPLIER: float = 1.25

# Scales the per-episode reward/penalty (agent PnL vs buy-and-hold benchmark)
DEFAULT_EPISODE_REWARD_MULTIPLIER: float = 1.0
