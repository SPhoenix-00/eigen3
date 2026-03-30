"""DDPG-based trading agents"""

from eigen3.agents.trading_agent import (
    TradingAgent,
    TradingNetworkParams,
    params_for_flax_msgpack,
    trading_params_from_msgpack_bytes,
    soft_target_update,
)

__all__ = [
    "TradingAgent",
    "TradingNetworkParams",
    "params_for_flax_msgpack",
    "trading_params_from_msgpack_bytes",
    "soft_target_update",
]
