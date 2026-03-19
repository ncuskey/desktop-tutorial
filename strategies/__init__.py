"""Strategy signal generators.

All strategy functions expose:
    generate_signals(df: pd.DataFrame, params: dict) -> pd.Series
and return a position series in {-1, 0, +1}.
"""

from .breakout import range_breakout_signals, volatility_expansion_breakout_signals
from .carry import carry_proxy_signals
from .filters import apply_filter
from .mean_reversion import bollinger_fade_signals, rsi_reversal_signals
from .trend import donchian_breakout_signals, ma_crossover_signals

__all__ = [
    "apply_filter",
    "ma_crossover_signals",
    "donchian_breakout_signals",
    "rsi_reversal_signals",
    "bollinger_fade_signals",
    "range_breakout_signals",
    "volatility_expansion_breakout_signals",
    "carry_proxy_signals",
]
