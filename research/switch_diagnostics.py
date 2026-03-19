from __future__ import annotations

import json

import numpy as np
import pandas as pd


def _window_drawdown(returns_window: np.ndarray) -> float:
    equity = np.cumprod(1.0 + returns_window)
    peak = np.maximum.accumulate(equity)
    dd = (equity / peak) - 1.0
    return float(np.min(dd))


def _regime_runs(regime_series: pd.Series) -> pd.DataFrame:
    labels = regime_series.fillna("UNKNOWN").astype(str)
    if labels.empty:
        return pd.DataFrame(columns=["Regime", "RunLength"])
    run_id = (labels != labels.shift(1)).cumsum()
    runs = (
        pd.DataFrame({"Regime": labels, "RunId": run_id})
        .groupby(["RunId", "Regime"], as_index=False)
        .size()
        .rename(columns={"size": "RunLength"})
    )
    return runs[["Regime", "RunLength"]]


def compute_regime_duration_stats(regime_series: pd.Series) -> pd.DataFrame:
    runs = _regime_runs(regime_series)
    if runs.empty:
        return pd.DataFrame(
            columns=[
                "Regime",
                "AverageDurationBars",
                "MedianDurationBars",
                "RunCount",
                "RunLengthDistribution",
            ]
        )

    grouped = []
    for regime, g in runs.groupby("Regime", sort=True):
        lengths = [int(x) for x in g["RunLength"].tolist()]
        grouped.append(
            {
                "Regime": regime,
                "AverageDurationBars": float(np.mean(lengths)),
                "MedianDurationBars": float(np.median(lengths)),
                "RunCount": float(len(lengths)),
                "RunLengthDistribution": json.dumps(lengths),
            }
        )
    return pd.DataFrame(grouped)


def compute_switches_per_1000_bars(regime_series: pd.Series) -> pd.DataFrame:
    labels = regime_series.fillna("UNKNOWN").astype(str)
    if labels.empty:
        switches = 0.0
    else:
        switches = float((labels != labels.shift(1)).sum() - 1)
        switches = max(switches, 0.0)
    per_1000 = (switches / max(len(labels), 1)) * 1_000.0
    return pd.DataFrame(
        [
            {
                "SwitchCount": switches,
                "Bars": float(len(labels)),
                "SwitchesPer1000Bars": float(per_1000),
            }
        ]
    )


def compute_switch_diagnostics(
    regime_series: pd.Series,
    returns: pd.Series,
    n_bars: int = 24,
) -> pd.DataFrame:
    aligned_regime = regime_series.fillna("UNKNOWN").astype(str)
    aligned_returns = returns.reindex(aligned_regime.index).fillna(0.0)
    switch_mask = (aligned_regime != aligned_regime.shift(1)).fillna(False)
    switch_idx = aligned_regime.index[switch_mask.fillna(False)]
    runs = _regime_runs(aligned_regime)
    duration_dist = (
        runs.groupby("Regime")["RunLength"]
        .apply(lambda s: [int(x) for x in s.tolist()])
        .to_dict()
    )
    switches_per_1000 = compute_switches_per_1000_bars(aligned_regime).iloc[0]
    avg_duration = float(runs["RunLength"].mean()) if not runs.empty else 0.0
    median_duration = float(runs["RunLength"].median()) if not runs.empty else 0.0

    if len(switch_idx) <= 1:
        return pd.DataFrame(
            [
                {
                    "SwitchCount": 0.0,
                    "SwitchesPer1000Bars": float(switches_per_1000["SwitchesPer1000Bars"]),
                    "AvgReturnAfterSwitch": 0.0,
                    "AvgDrawdownAfterSwitch": 0.0,
                    "PctSwitchesImproveNextNExpectancy": 0.0,
                    "AverageRegimeDurationBars": avg_duration,
                    "MedianRegimeDurationBars": median_duration,
                    "RegimeLengthDistribution": json.dumps(duration_dist, sort_keys=True),
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
                    "SwitchesPer1000Bars": float(switches_per_1000["SwitchesPer1000Bars"]),
                    "AvgReturnAfterSwitch": 0.0,
                    "AvgDrawdownAfterSwitch": 0.0,
                    "PctSwitchesImproveNextNExpectancy": 0.0,
                    "AverageRegimeDurationBars": avg_duration,
                    "MedianRegimeDurationBars": median_duration,
                    "RegimeLengthDistribution": json.dumps(duration_dist, sort_keys=True),
                    "NextNBars": float(n_bars),
                }
            ]
        )

    return pd.DataFrame(
        [
            {
                "SwitchCount": float(len(window_returns)),
                "SwitchesPer1000Bars": float(switches_per_1000["SwitchesPer1000Bars"]),
                "AvgReturnAfterSwitch": float(np.mean(window_returns)),
                "AvgDrawdownAfterSwitch": float(np.mean(window_drawdowns)),
                "PctSwitchesImproveNextNExpectancy": float(np.mean(better_than_expectancy)),
                "AverageRegimeDurationBars": avg_duration,
                "MedianRegimeDurationBars": median_duration,
                "RegimeLengthDistribution": json.dumps(duration_dist, sort_keys=True),
                "NextNBars": float(n_bars),
            }
        ]
    )
