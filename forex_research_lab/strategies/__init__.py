"""Strategy registry and exports."""

from __future__ import annotations

from typing import Callable, Mapping

import pandas as pd

from .breakout import (
    range_breakout_generate_signals,
    volatility_expansion_breakout_generate_signals,
)
from .carry import carry_proxy_generate_signals
from .mean_reversion import (
    bollinger_fade_generate_signals,
    rsi_reversal_generate_signals,
)
from .trend import (
    donchian_breakout_generate_signals,
    ma_crossover_generate_signals,
)

StrategyFn = Callable[[pd.DataFrame, Mapping[str, float]], pd.Series]


STRATEGY_REGISTRY: dict[str, StrategyFn] = {
    "trend_ma_crossover": ma_crossover_generate_signals,
    "trend_donchian_breakout": donchian_breakout_generate_signals,
    "mean_reversion_rsi": rsi_reversal_generate_signals,
    "mean_reversion_bollinger_fade": bollinger_fade_generate_signals,
    "breakout_range": range_breakout_generate_signals,
    "breakout_volatility_expansion": volatility_expansion_breakout_generate_signals,
    "carry_proxy": carry_proxy_generate_signals,
}

__all__ = [
    "StrategyFn",
    "STRATEGY_REGISTRY",
    "ma_crossover_generate_signals",
    "donchian_breakout_generate_signals",
    "rsi_reversal_generate_signals",
    "bollinger_fade_generate_signals",
    "range_breakout_generate_signals",
    "volatility_expansion_breakout_generate_signals",
    "carry_proxy_generate_signals",
]
