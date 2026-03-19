"""Signal-generation strategies."""

from .base import BaseStrategy
from .breakout import RangeBreakoutStrategy, VolatilityExpansionBreakoutStrategy
from .carry import CarryProxyStrategy
from .mean_reversion import BollingerFadeStrategy, RSIReversalStrategy
from .trend import DonchianBreakoutStrategy, MovingAverageCrossoverStrategy

__all__ = [
    "BaseStrategy",
    "BollingerFadeStrategy",
    "CarryProxyStrategy",
    "DonchianBreakoutStrategy",
    "MovingAverageCrossoverStrategy",
    "RSIReversalStrategy",
    "RangeBreakoutStrategy",
    "VolatilityExpansionBreakoutStrategy",
]
