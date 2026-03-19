from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from combinations import combine_specialist_sleeves
from execution import apply_no_trade_filter_high_vol, apply_volatility_targeting


@dataclass
class RegimeSpecialistOrchestrator:
    regime_column: str = "regime_label"
    vol_regime_column: str = "vol_regime"
    regime_to_sleeve: dict[str, str] = field(default_factory=dict)
    sleeve_weights: dict[str, float] | None = None
    fallback: str = "flat"
    default_strategy: str | None = None
    allow_high_vol_entries: bool = False
    use_vol_targeting: bool = False
    target_atr_norm: float = 0.001
    max_leverage: float = 1.0

    def orchestrate(
        self,
        df: pd.DataFrame,
        sleeve_signals: dict[str, pd.Series],
    ) -> pd.Series:
        if self.regime_column not in df.columns:
            raise ValueError(f"Missing required regime column: {self.regime_column}")
        if self.vol_regime_column not in df.columns:
            raise ValueError(f"Missing required volatility regime column: {self.vol_regime_column}")

        weighted_signals: dict[str, pd.Series] = {}
        for sleeve_name, signal in sleeve_signals.items():
            weight = 1.0
            if self.sleeve_weights is not None:
                weight = float(self.sleeve_weights.get(sleeve_name, 0.0))
            weighted_signals[sleeve_name] = signal.astype(float) * weight

        out = combine_specialist_sleeves(
            strategy_outputs=weighted_signals,
            regime_series=df[self.regime_column],
            regime_to_sleeve=self.regime_to_sleeve,
            fallback=self.fallback,
            default_strategy=self.default_strategy,
        )

        out = apply_no_trade_filter_high_vol(
            out,
            vol_regime=df[self.vol_regime_column],
            allow_high_vol=self.allow_high_vol_entries,
        )

        if self.use_vol_targeting:
            if "atr_norm" not in df.columns:
                raise ValueError("atr_norm is required in df when use_vol_targeting=True")
            out = apply_volatility_targeting(
                out,
                atr_normalized=df["atr_norm"],
                target_atr_norm=self.target_atr_norm,
                max_leverage=self.max_leverage,
            )
        else:
            out = out.clip(lower=-self.max_leverage, upper=self.max_leverage)

        return out.astype(float)
