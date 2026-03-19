from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from core.types import AllocationResult

from .rebalance import apply_rebalance_schedule, build_rebalance_mask
from .risk_budget import apply_weight_constraints, normalize_positive_scores


AllocatorMode = Literal["equal_weight", "inverse_volatility", "expectancy_score"]


def _rolling_volatility(returns: pd.DataFrame, window: int = 120) -> pd.DataFrame:
    return returns.rolling(window=window, min_periods=max(10, window // 3)).std(ddof=0)


@dataclass
class V2PortfolioAllocator:
    mode: AllocatorMode = "equal_weight"
    max_symbol_exposure: float = 0.35
    gross_exposure_cap: float = 1.0
    inverse_vol_window: int = 120
    score_lookback: int = 120
    rebalance_rule: str = "1D"
    flat_on_missing_data: bool = True

    def allocate(
        self,
        signal_frame: pd.DataFrame,
        returns_frame: pd.DataFrame | None = None,
        expectancy_scores: pd.DataFrame | pd.Series | None = None,
    ) -> AllocationResult:
        if signal_frame.empty:
            return AllocationResult(weights=signal_frame.copy(), diagnostics={"status": "empty"})

        signals = signal_frame.astype(float).copy()
        index = pd.DatetimeIndex(signals.index)
        cols = signals.columns.tolist()
        target_weights = pd.DataFrame(0.0, index=index, columns=cols)

        rolling_vol = None
        if returns_frame is not None:
            rolling_vol = _rolling_volatility(returns_frame.astype(float), window=self.inverse_vol_window)

        if isinstance(expectancy_scores, pd.Series):
            score_df = pd.DataFrame(np.tile(expectancy_scores.values, (len(signals), 1)), index=index, columns=expectancy_scores.index)
            score_df = score_df.reindex(columns=cols).fillna(0.0)
        elif isinstance(expectancy_scores, pd.DataFrame):
            score_df = expectancy_scores.reindex(index=index, columns=cols).fillna(0.0)
        else:
            score_df = pd.DataFrame(0.0, index=index, columns=cols)

        for ts in index:
            s = signals.loc[ts]
            if self.flat_on_missing_data:
                s = s.where(~s.isna(), other=0.0)
            active = s[s != 0.0]
            if active.empty:
                target_weights.loc[ts] = 0.0
                continue

            side = np.sign(active)
            if self.mode == "equal_weight":
                base = pd.Series(1.0, index=active.index, dtype=float)
            elif self.mode == "inverse_volatility":
                if rolling_vol is None:
                    base = pd.Series(1.0, index=active.index, dtype=float)
                else:
                    vol = rolling_vol.loc[ts, active.index].replace(0.0, np.nan)
                    inv = (1.0 / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
                    base = normalize_positive_scores(inv)
            elif self.mode == "expectancy_score":
                raw_scores = score_df.loc[ts, active.index]
                base = normalize_positive_scores(raw_scores.clip(lower=0.0))
            else:
                raise ValueError(f"Unsupported allocator mode: {self.mode}")

            if self.mode == "equal_weight":
                base = normalize_positive_scores(base)
            weights_active = base * side
            constrained = apply_weight_constraints(
                weights_active,
                max_symbol_exposure=self.max_symbol_exposure,
                gross_exposure_cap=self.gross_exposure_cap,
            )
            target_weights.loc[ts, constrained.index] = constrained

        rebalance_mask = build_rebalance_mask(index, rule=self.rebalance_rule)
        live_weights = apply_rebalance_schedule(target_weights, rebalance_mask=rebalance_mask)
        diagnostics = {
            "mode": self.mode,
            "max_symbol_exposure": float(self.max_symbol_exposure),
            "gross_exposure_cap": float(self.gross_exposure_cap),
            "avg_gross_exposure": float(live_weights.abs().sum(axis=1).mean()),
            "rebalance_count": int(rebalance_mask.sum()),
        }
        return AllocationResult(weights=live_weights, diagnostics=diagnostics)
