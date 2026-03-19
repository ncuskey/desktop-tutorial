from __future__ import annotations

import numpy as np
import pandas as pd


def _window_drawdown(returns_window: np.ndarray) -> float:
    equity = np.cumprod(1.0 + returns_window)
    peak = np.maximum.accumulate(equity)
    dd = (equity / peak) - 1.0
    return float(np.min(dd))


def compute_switch_diagnostics(
    regime_series: pd.Series,
    returns: pd.Series,
    n_bars: int = 24,
) -> pd.DataFrame:
    aligned_regime = regime_series.fillna("UNKNOWN").astype(str)
    aligned_returns = returns.reindex(aligned_regime.index).fillna(0.0)
    switch_mask = aligned_regime != aligned_regime.shift(1)
    switch_idx = aligned_regime.index[switch_mask.fillna(False)]

    if len(switch_idx) <= 1:
        return pd.DataFrame(
            [
                {
                    "SwitchCount": 0.0,
                    "AvgReturnAfterSwitch": 0.0,
                    "AvgDrawdownAfterSwitch": 0.0,
                    "PctSwitchesImproveNextNExpectancy": 0.0,
                    "NextNBars": float(n_bars),
                }
            ]
        )

    window_returns: list[float] = []
    window_drawdowns: list[float] = []
    benchmark = float(aligned_returns.mean() * n_bars)
    better_than_expectancy = []

    for switch_time in switch_idx[1:]:
        start_loc = aligned_regime.index.get_loc(switch_time)
        end_loc = min(start_loc + n_bars, len(aligned_returns))
        if end_loc - start_loc < 2:
            continue
        window = aligned_returns.iloc[start_loc:end_loc].to_numpy()
        window_ret = float(np.prod(1.0 + window) - 1.0)
        window_dd = _window_drawdown(window)
        window_returns.append(window_ret)
        window_drawdowns.append(window_dd)
        better_than_expectancy.append(window_ret > benchmark)

    if not window_returns:
        return pd.DataFrame(
            [
                {
                    "SwitchCount": 0.0,
                    "AvgReturnAfterSwitch": 0.0,
                    "AvgDrawdownAfterSwitch": 0.0,
                    "PctSwitchesImproveNextNExpectancy": 0.0,
                    "NextNBars": float(n_bars),
                }
            ]
        )

    return pd.DataFrame(
        [
            {
                "SwitchCount": float(len(window_returns)),
                "AvgReturnAfterSwitch": float(np.mean(window_returns)),
                "AvgDrawdownAfterSwitch": float(np.mean(window_drawdowns)),
                "PctSwitchesImproveNextNExpectancy": float(np.mean(better_than_expectancy)),
                "NextNBars": float(n_bars),
            }
        ]
    )
