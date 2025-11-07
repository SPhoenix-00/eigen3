"""DDPG-based trading agents"""

from eigen3.agents.trading_agent import TradingAgent, TradingNetworkParams, soft_target_update

__all__ = [
    "TradingAgent",
    "TradingNetworkParams",
    "soft_target_update",
]
