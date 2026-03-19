from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from combinations import combine_specialist_sleeves
from execution import apply_no_trade_filter_high_vol, apply_volatility_targeting


@dataclass
class RegimeSpecialistOrchestrator:
    regime_column: str = "regime_label"
    stable_regime_column: str = "stable_regime_label"
    use_stable_regime: bool = True
    vol_regime_column: str = "vol_regime"
    regime_to_sleeve: dict[str, str] = field(default_factory=dict)
    sleeve_weights: dict[str, float] | None = None
    fallback: str = "flat"
    default_strategy: str | None = None
    switch_cooldown_bars: int = 12
    switch_penalty_bps: float = 0.0
    allow_high_vol_entries: bool = False
    use_vol_targeting: bool = False
    target_atr_norm: float = 0.001
    max_leverage: float = 1.0
    latest_switch_flags: pd.Series | None = None
    latest_switch_penalty: pd.Series | None = None

    def _resolve_regime_series(self, df: pd.DataFrame) -> pd.Series:
        if self.use_stable_regime:
            if self.stable_regime_column not in df.columns:
                raise ValueError(
                    f"use_stable_regime=True but column '{self.stable_regime_column}' not found."
                )
            return df[self.stable_regime_column]
        if self.regime_column not in df.columns:
            raise ValueError(f"Missing required regime column: {self.regime_column}")
        return df[self.regime_column]

    def _apply_switch_cooldown(self, signal: pd.Series) -> tuple[pd.Series, pd.Series]:
        if self.switch_cooldown_bars <= 0:
            switches = (signal != signal.shift(1)).astype(float).fillna(0.0)
            return signal, switches

        out = signal.copy().astype(float)
        switch_flags = pd.Series(0.0, index=out.index)
        if len(out) == 0:
            return out, switch_flags

        last_switch_idx = -10_000_000
        out.iloc[0] = float(out.iloc[0])
        if abs(out.iloc[0]) > 1e-12:
            switch_flags.iloc[0] = 1.0
            last_switch_idx = 0

        for i in range(1, len(out)):
            desired = float(out.iloc[i])
            prev = float(out.iloc[i - 1])
            if np.isclose(desired, prev):
                out.iloc[i] = prev
                continue
            if i - last_switch_idx < self.switch_cooldown_bars:
                out.iloc[i] = prev
                continue
            out.iloc[i] = desired
            switch_flags.iloc[i] = 1.0
            last_switch_idx = i
        return out, switch_flags

    def orchestrate(
        self,
        df: pd.DataFrame,
        sleeve_signals: dict[str, pd.Series],
    ) -> pd.Series:
        if self.vol_regime_column not in df.columns:
            raise ValueError(f"Missing required volatility regime column: {self.vol_regime_column}")

        resolved_regime = self._resolve_regime_series(df)

        weighted_signals: dict[str, pd.Series] = {}
        for sleeve_name, signal in sleeve_signals.items():
            weight = 1.0
            if self.sleeve_weights is not None:
                weight = float(self.sleeve_weights.get(sleeve_name, 0.0))
            weighted_signals[sleeve_name] = signal.astype(float) * weight

        out = combine_specialist_sleeves(
            strategy_outputs=weighted_signals,
            regime_series=resolved_regime,
            regime_to_sleeve=self.regime_to_sleeve,
            fallback=self.fallback,
            default_strategy=self.default_strategy,
        )
        # Entry decoupling is naturally preserved: regime gates allowed sleeve,
        # but position only appears when the selected sleeve emits non-zero signal.
        out = out.fillna(0.0)

        out, switch_flags = self._apply_switch_cooldown(out)

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

        switch_penalty = pd.Series(0.0, index=out.index)
        if self.switch_penalty_bps > 0:
            switch_penalty = switch_flags * (self.switch_penalty_bps / 10_000.0)
        self.latest_switch_flags = switch_flags.astype(float)
        self.latest_switch_penalty = switch_penalty.astype(float)

        return out.astype(float)
