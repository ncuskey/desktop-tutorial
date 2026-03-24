from __future__ import annotations

import numpy as np
import pandas as pd

from .filters import apply_filter


def _rolling_percentile_rank(series: pd.Series, window: int = 252) -> pd.Series:
    min_periods = max(30, window // 3)
    return series.rolling(window=window, min_periods=min_periods).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )


def trend_breakout_signals(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Legacy trend breakout sleeve (kept for V2/V2.1 compatibility).
    Supports Donchian and optional ATR breakout confirmation.
    """
    lookback = int(params.get("donchian_lookback", params.get("lookback", 20)))
    atr_col = str(params.get("atr_col", "atr_14"))
    atr_breakout_mult = float(params.get("atr_breakout_mult", 1.2))
    combine_mode = str(params.get("combine_mode", "either")).lower()  # either / both
    use_donchian = bool(params.get("use_donchian", True))
    use_atr_breakout = bool(params.get("use_atr_breakout", True))
    trailing_stop_atr_mult = params.get("trailing_stop_atr_mult", None)
    if trailing_stop_atr_mult is not None:
        trailing_stop_atr_mult = float(trailing_stop_atr_mult)

    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    atr = (
        pd.to_numeric(df[atr_col], errors="coerce")
        if atr_col in df.columns
        else pd.Series(np.nan, index=df.index)
    )

    donchian_upper = high.rolling(lookback, min_periods=lookback).max().shift(1)
    donchian_lower = low.rolling(lookback, min_periods=lookback).min().shift(1)
    donchian_long = close > donchian_upper
    donchian_short = close < donchian_lower

    atr_anchor = close.shift(1)
    atr_long = close > (atr_anchor + (atr_breakout_mult * atr))
    atr_short = close < (atr_anchor - (atr_breakout_mult * atr))

    if use_donchian and use_atr_breakout:
        if combine_mode == "both":
            long_entry = donchian_long & atr_long
            short_entry = donchian_short & atr_short
        else:
            long_entry = donchian_long | atr_long
            short_entry = donchian_short | atr_short
    elif use_donchian:
        long_entry = donchian_long
        short_entry = donchian_short
    elif use_atr_breakout:
        long_entry = atr_long
        short_entry = atr_short
    else:
        raise ValueError("At least one breakout engine must be enabled.")

    signal = pd.Series(0.0, index=df.index, dtype=float)
    current = 0.0
    trailing_stop = np.nan

    for i in range(len(df)):
        price = float(close.iloc[i]) if pd.notna(close.iloc[i]) else np.nan
        atr_i = float(atr.iloc[i]) if pd.notna(atr.iloc[i]) else np.nan
        if not np.isfinite(price):
            signal.iloc[i] = current
            continue

        go_long = bool(long_entry.iloc[i]) if pd.notna(long_entry.iloc[i]) else False
        go_short = bool(short_entry.iloc[i]) if pd.notna(short_entry.iloc[i]) else False

        if current > 0:
            if trailing_stop_atr_mult is not None and np.isfinite(atr_i):
                new_stop = price - trailing_stop_atr_mult * atr_i
                trailing_stop = (
                    max(trailing_stop, new_stop) if np.isfinite(trailing_stop) else new_stop
                )
            stop_hit = np.isfinite(trailing_stop) and price < trailing_stop
            if stop_hit:
                current = 0.0
                trailing_stop = np.nan
        elif current < 0:
            if trailing_stop_atr_mult is not None and np.isfinite(atr_i):
                new_stop = price + trailing_stop_atr_mult * atr_i
                trailing_stop = (
                    min(trailing_stop, new_stop) if np.isfinite(trailing_stop) else new_stop
                )
            stop_hit = np.isfinite(trailing_stop) and price > trailing_stop
            if stop_hit:
                current = 0.0
                trailing_stop = np.nan

        if current == 0.0:
            if go_long and not go_short:
                current = 1.0
                if trailing_stop_atr_mult is not None and np.isfinite(atr_i):
                    trailing_stop = price - trailing_stop_atr_mult * atr_i
            elif go_short and not go_long:
                current = -1.0
                if trailing_stop_atr_mult is not None and np.isfinite(atr_i):
                    trailing_stop = price + trailing_stop_atr_mult * atr_i
        elif current > 0 and go_short:
            current = -1.0
            trailing_stop = (
                price + trailing_stop_atr_mult * atr_i
                if trailing_stop_atr_mult is not None and np.isfinite(atr_i)
                else np.nan
            )
        elif current < 0 and go_long:
            current = 1.0
            trailing_stop = (
                price - trailing_stop_atr_mult * atr_i
                if trailing_stop_atr_mult is not None and np.isfinite(atr_i)
                else np.nan
            )

        signal.iloc[i] = current

    warmup_invalid = donchian_upper.isna() & donchian_lower.isna()
    signal[warmup_invalid] = 0.0
    out = signal.astype(int)
    filter_condition = params.get("filter_condition")
    if filter_condition is not None:
        out = apply_filter(out, condition=filter_condition)
    return out


def trend_breakout_v2_signals(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Hardened trend breakout sleeve for V2.3 edge amplification.
    """
    lookback = int(params.get("breakout_lookback", params.get("lookback", 20)))
    atr_col = str(params.get("atr_col", "atr_14"))
    atr_norm_col = str(params.get("atr_norm_col", "atr_norm"))
    vol_rank_col = str(params.get("atr_pct_rank_col", "atr_norm_pct_rank"))

    vol_compression_max_pct = float(
        params.get(
            "compression_pct_rank_threshold",
            params.get("vol_compression_max_pct", 0.40),
        )
    )
    breakout_strength_atr_mult = float(
        params.get(
            "breakout_strength_atr_multiple",
            params.get("breakout_strength_atr_mult", 0.25),
        )
    )
    velocity_lookback = int(params.get("velocity_lookback", 6))
    velocity_threshold = float(params.get("velocity_threshold", 1.0))
    confirmation_bars = int(params.get("confirmation_bars", 2))
    expansion_lookback = int(params.get("expansion_lookback", 12))
    expansion_threshold = float(params.get("expansion_threshold", 1.10))

    retest_entry_mode = bool(
        params.get("retest_entry_enabled", params.get("retest_entry_mode", False))
    )
    retest_expiry_bars = int(params.get("retest_expiry_bars", 10))
    retest_tolerance_atr_mult = float(params.get("retest_tolerance_atr_mult", 0.15))
    retest_confirm_buffer_atr_mult = float(params.get("retest_confirm_buffer_atr_mult", 0.05))

    trailing_stop_atr_mult = float(
        params.get("atr_stop_multiplier", params.get("trailing_stop_atr_mult", 1.8))
    )
    max_holding_bars = int(params.get("time_exit_bars", params.get("max_holding_bars", 72)))
    vol_contraction_exit_mult = float(params.get("vol_contraction_exit_mult", 0.80))
    vol_contraction_window = int(params.get("vol_contraction_window", 20))
    vol_exit_pct_rank_threshold = float(params.get("vol_exit_pct_rank_threshold", 0.25))
    contraction_exit_enabled = bool(params.get("contraction_exit_enabled", True))

    winner_extension_enabled = bool(params.get("winner_extension_enabled", True))
    extension_trigger_atr_multiple = float(params.get("extension_trigger_atr_multiple", 2.0))
    extension_stop_multiplier = float(params.get("extension_stop_multiplier", 2.5))
    extension_max_holding_bars = int(
        params.get("extension_max_holding_bars", max_holding_bars * 3)
    )

    partial_take_profit_enabled = bool(params.get("partial_take_profit_enabled", True))
    partial_take_profit_rr = float(
        params.get(
            "partial_take_profit_atr_multiple",
            params.get("partial_take_profit_rr", 1.2),
        )
    )
    if not partial_take_profit_enabled:
        partial_take_profit_rr = 0.0
    partial_take_profit_size = float(params.get("partial_take_profit_size", 0.4))
    partial_take_profit_size = float(np.clip(partial_take_profit_size, 0.1, 0.9))

    min_bars_between_trades = int(params.get("min_bars_between_trades", 24))
    dynamic_cooldown_by_vol = bool(params.get("dynamic_cooldown_by_vol", False))
    high_vol_cooldown_mult = float(params.get("high_vol_cooldown_mult", 1.5))

    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    atr = pd.to_numeric(df.get(atr_col, np.nan), errors="coerce")
    atr_norm = pd.to_numeric(df.get(atr_norm_col, np.nan), errors="coerce")

    if vol_rank_col in df.columns:
        atr_rank = pd.to_numeric(df[vol_rank_col], errors="coerce")
    else:
        atr_rank = _rolling_percentile_rank(atr_norm, window=252)

    donchian_upper = high.rolling(lookback, min_periods=lookback).max().shift(1)
    donchian_lower = low.rolling(lookback, min_periods=lookback).min().shift(1)
    atr_ma = atr_norm.rolling(
        vol_contraction_window,
        min_periods=max(10, vol_contraction_window // 2),
    ).mean()
    directional_velocity = (close - close.shift(velocity_lookback)) / atr.replace(0.0, np.nan)
    recent_range = (
        high.rolling(expansion_lookback, min_periods=expansion_lookback).max()
        - low.rolling(expansion_lookback, min_periods=expansion_lookback).min()
    ) / close.abs().replace(0.0, np.nan)
    prior_range = recent_range.shift(expansion_lookback)
    range_expansion_ratio = recent_range / prior_range.replace(0.0, np.nan)

    signal = pd.Series(0.0, index=df.index, dtype=float)
    current_pos = 0.0
    entry_price = np.nan
    entry_atr = np.nan
    bars_in_trade = 0
    trail_stop = np.nan
    partial_tp_taken = False
    extended_mode = False
    next_entry_i = -10_000_000

    pending_dir = 0
    pending_level = np.nan
    pending_expiry_i = -1
    long_break_streak = 0
    short_break_streak = 0

    for i in range(len(df)):
        px = float(close.iloc[i]) if pd.notna(close.iloc[i]) else np.nan
        atr_i = float(atr.iloc[i]) if pd.notna(atr.iloc[i]) else np.nan
        atrn_i = float(atr_norm.iloc[i]) if pd.notna(atr_norm.iloc[i]) else np.nan
        atr_rank_i = float(atr_rank.iloc[i]) if pd.notna(atr_rank.iloc[i]) else np.nan
        up_i = float(donchian_upper.iloc[i]) if pd.notna(donchian_upper.iloc[i]) else np.nan
        dn_i = float(donchian_lower.iloc[i]) if pd.notna(donchian_lower.iloc[i]) else np.nan
        atrn_ma_i = float(atr_ma.iloc[i]) if pd.notna(atr_ma.iloc[i]) else np.nan
        vel_i = (
            float(directional_velocity.iloc[i])
            if pd.notna(directional_velocity.iloc[i])
            else np.nan
        )
        expansion_i = (
            float(range_expansion_ratio.iloc[i])
            if pd.notna(range_expansion_ratio.iloc[i])
            else np.nan
        )

        if not np.isfinite(px):
            signal.iloc[i] = current_pos
            continue

        can_open_new = i >= next_entry_i
        compression_ok = (not np.isfinite(atr_rank_i)) or (atr_rank_i <= vol_compression_max_pct)
        expansion_ok = np.isfinite(expansion_i) and (expansion_i >= expansion_threshold)
        velocity_long_ok = np.isfinite(vel_i) and (vel_i >= velocity_threshold)
        velocity_short_ok = np.isfinite(vel_i) and (vel_i <= -velocity_threshold)

        long_break = False
        short_break = False
        if np.isfinite(up_i) and np.isfinite(atr_i):
            long_break = px > up_i and (px - up_i) > (breakout_strength_atr_mult * atr_i)
        if np.isfinite(dn_i) and np.isfinite(atr_i):
            short_break = px < dn_i and (dn_i - px) > (breakout_strength_atr_mult * atr_i)
        if long_break:
            long_break_streak += 1
        else:
            long_break_streak = 0
        if short_break:
            short_break_streak += 1
        else:
            short_break_streak = 0

        long_entry = False
        short_entry = False
        long_ready = (
            can_open_new
            and compression_ok
            and expansion_ok
            and velocity_long_ok
            and long_break_streak >= confirmation_bars
        )
        short_ready = (
            can_open_new
            and compression_ok
            and expansion_ok
            and velocity_short_ok
            and short_break_streak >= confirmation_bars
        )

        if retest_entry_mode:
            if long_ready and long_break:
                pending_dir = 1
                pending_level = up_i
                pending_expiry_i = i + retest_expiry_bars
            elif short_ready and short_break:
                pending_dir = -1
                pending_level = dn_i
                pending_expiry_i = i + retest_expiry_bars

            if pending_dir != 0 and i <= pending_expiry_i and np.isfinite(atr_i):
                tol = retest_tolerance_atr_mult * atr_i
                confirm = retest_confirm_buffer_atr_mult * atr_i
                if pending_dir == 1:
                    touched = float(low.iloc[i]) <= (pending_level + tol)
                    confirmed = px >= (pending_level + confirm)
                    if touched and confirmed and long_ready:
                        long_entry = True
                        pending_dir = 0
                else:
                    touched = float(high.iloc[i]) >= (pending_level - tol)
                    confirmed = px <= (pending_level - confirm)
                    if touched and confirmed and short_ready:
                        short_entry = True
                        pending_dir = 0
            if pending_dir != 0 and i > pending_expiry_i:
                pending_dir = 0
        else:
            long_entry = long_ready and long_break
            short_entry = short_ready and short_break

        exit_now = False
        if current_pos != 0.0:
            bars_in_trade += 1
            pos_side = 1.0 if current_pos > 0.0 else -1.0

            if (
                winner_extension_enabled
                and not extended_mode
                and np.isfinite(entry_price)
                and np.isfinite(entry_atr)
            ):
                unrealized = (px - entry_price) * pos_side
                if unrealized >= (extension_trigger_atr_multiple * entry_atr):
                    extended_mode = True

            if np.isfinite(atr_i):
                trail_mult = trailing_stop_atr_mult
                if extended_mode:
                    trail_mult = trailing_stop_atr_mult * extension_stop_multiplier
                if pos_side > 0:
                    new_stop = px - trail_mult * atr_i
                    trail_stop = max(trail_stop, new_stop) if np.isfinite(trail_stop) else new_stop
                    if np.isfinite(trail_stop) and px < trail_stop:
                        exit_now = True
                else:
                    new_stop = px + trail_mult * atr_i
                    trail_stop = min(trail_stop, new_stop) if np.isfinite(trail_stop) else new_stop
                    if np.isfinite(trail_stop) and px > trail_stop:
                        exit_now = True

            if (not extended_mode) and bars_in_trade >= max_holding_bars:
                exit_now = True
            if extended_mode and bars_in_trade >= extension_max_holding_bars:
                exit_now = True

            if contraction_exit_enabled:
                if np.isfinite(atrn_i) and np.isfinite(atrn_ma_i):
                    if atrn_i <= (vol_contraction_exit_mult * atrn_ma_i):
                        exit_now = True
                if np.isfinite(atr_rank_i) and atr_rank_i <= vol_exit_pct_rank_threshold:
                    exit_now = True

            if (
                partial_take_profit_rr > 0
                and not partial_tp_taken
                and np.isfinite(entry_price)
                and np.isfinite(entry_atr)
            ):
                if pos_side > 0 and (px - entry_price) >= (partial_take_profit_rr * entry_atr):
                    current_pos = current_pos * (1.0 - partial_take_profit_size)
                    partial_tp_taken = True
                elif pos_side < 0 and (entry_price - px) >= (partial_take_profit_rr * entry_atr):
                    current_pos = current_pos * (1.0 - partial_take_profit_size)
                    partial_tp_taken = True

        if exit_now:
            current_pos = 0.0
            entry_price = np.nan
            entry_atr = np.nan
            bars_in_trade = 0
            trail_stop = np.nan
            partial_tp_taken = False
            extended_mode = False
            cooldown_bars = min_bars_between_trades
            if dynamic_cooldown_by_vol and np.isfinite(atr_rank_i):
                cooldown_scale = 1.0 + high_vol_cooldown_mult * max(0.0, atr_rank_i - 0.5)
                cooldown_bars = int(round(min_bars_between_trades * cooldown_scale))
            next_entry_i = i + max(cooldown_bars, min_bars_between_trades)

        if current_pos == 0.0:
            if long_entry and not short_entry:
                current_pos = 1.0
                entry_price = px
                entry_atr = atr_i
                bars_in_trade = 0
                trail_stop = px - trailing_stop_atr_mult * atr_i if np.isfinite(atr_i) else np.nan
                partial_tp_taken = False
                extended_mode = False
                pending_dir = 0
            elif short_entry and not long_entry:
                current_pos = -1.0
                entry_price = px
                entry_atr = atr_i
                bars_in_trade = 0
                trail_stop = px + trailing_stop_atr_mult * atr_i if np.isfinite(atr_i) else np.nan
                partial_tp_taken = False
                extended_mode = False
                pending_dir = 0
        elif current_pos > 0 and short_entry and can_open_new:
            current_pos = -1.0
            entry_price = px
            entry_atr = atr_i
            bars_in_trade = 0
            trail_stop = px + trailing_stop_atr_mult * atr_i if np.isfinite(atr_i) else np.nan
            partial_tp_taken = False
            extended_mode = False
            pending_dir = 0
        elif current_pos < 0 and long_entry and can_open_new:
            current_pos = 1.0
            entry_price = px
            entry_atr = atr_i
            bars_in_trade = 0
            trail_stop = px - trailing_stop_atr_mult * atr_i if np.isfinite(atr_i) else np.nan
            partial_tp_taken = False
            extended_mode = False
            pending_dir = 0

        signal.iloc[i] = current_pos

    warmup_invalid = donchian_upper.isna() | donchian_lower.isna()
    signal[warmup_invalid] = 0.0
    return signal.astype(float).clip(-1.0, 1.0)


def generate_signals(df: pd.DataFrame, params: dict) -> pd.Series:
    return trend_breakout_signals(df, params)
