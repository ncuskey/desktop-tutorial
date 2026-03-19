"""Strategy implementations and registry."""

from forex_research_lab.strategies.base import BaseStrategy
from forex_research_lab.strategies.breakout import RangeBreakoutStrategy, VolatilityExpansionBreakoutStrategy
from forex_research_lab.strategies.carry import CarryProxyStrategy
from forex_research_lab.strategies.mean_reversion import BollingerFadeStrategy, RSIReversalStrategy
from forex_research_lab.strategies.trend import DonchianBreakoutStrategy, MovingAverageCrossoverStrategy

STRATEGY_REGISTRY = {
    "ma_crossover": MovingAverageCrossoverStrategy,
    "donchian_breakout": DonchianBreakoutStrategy,
    "rsi_reversal": RSIReversalStrategy,
    "bollinger_fade": BollingerFadeStrategy,
    "range_breakout": RangeBreakoutStrategy,
    "volatility_expansion_breakout": VolatilityExpansionBreakoutStrategy,
    "carry_proxy": CarryProxyStrategy,
}

__all__ = [
    "BaseStrategy",
    "BollingerFadeStrategy",
    "CarryProxyStrategy",
    "DonchianBreakoutStrategy",
    "MovingAverageCrossoverStrategy",
    "RSIReversalStrategy",
    "RangeBreakoutStrategy",
    "STRATEGY_REGISTRY",
    "VolatilityExpansionBreakoutStrategy",
]
