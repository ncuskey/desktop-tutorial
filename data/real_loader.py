from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.loader import resample_ohlcv


DEFAULT_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "timestamp": ("timestamp", "time", "date", "datetime", "Date", "Timestamp"),
    "open": ("open", "Open", "o", "bid_open"),
    "high": ("high", "High", "h", "bid_high"),
    "low": ("low", "Low", "l", "bid_low"),
    "close": ("close", "Close", "c", "bid_close"),
    "volume": ("volume", "Volume", "tick_volume", "vol"),
    "spread_bps": ("spread_bps", "spread", "Spread"),
}


def _detect_source_column(
    df: pd.DataFrame,
    target: str,
    column_map: dict[str, str] | None,
    required: bool = False,
) -> str | None:
    if column_map and target in column_map:
        mapped = column_map[target]
        if mapped in df.columns:
            return mapped
        if required:
            raise ValueError(f"Column map provided {target} -> {mapped}, but source column not found.")
        return None
    for alias in DEFAULT_COLUMN_ALIASES.get(target, ()):
        if alias in df.columns:
            return alias
    return None


def _parse_timestamp(ts: pd.Series, timezone: str | None = None) -> pd.Series:
    parsed = pd.to_datetime(ts, errors="coerce", utc=False)
    if parsed.isna().all():
        raise ValueError("Failed to parse timestamps from source data.")
    if timezone:
        if getattr(parsed.dt, "tz", None) is None:
            parsed = parsed.dt.tz_localize(timezone)
        else:
            parsed = parsed.dt.tz_convert(timezone)
        parsed = parsed.dt.tz_convert("UTC")
    else:
        if getattr(parsed.dt, "tz", None) is None:
            parsed = parsed.dt.tz_localize("UTC")
        else:
            parsed = parsed.dt.tz_convert("UTC")
    return parsed


def infer_timeframe_from_series(ts: pd.Series) -> str:
    ts = pd.Series(pd.to_datetime(ts, utc=True)).sort_values().drop_duplicates()
    if len(ts) < 3:
        return "unknown"
    deltas = ts.diff().dropna().dt.total_seconds()
    median_sec = float(deltas.median())
    if median_sec <= 0:
        return "unknown"
    if abs(median_sec - 60) < 1:
        return "1m"
    if abs(median_sec - 300) < 1:
        return "5m"
    if abs(median_sec - 900) < 1:
        return "15m"
    if abs(median_sec - 1800) < 1:
        return "30m"
    if abs(median_sec - 3600) < 1:
        return "1h"
    if abs(median_sec - 14400) < 1:
        return "4h"
    if abs(median_sec - 86400) < 1:
        return "1d"
    return f"{int(round(median_sec))}s"


def load_real_fx_csv(
    filepath: str | Path,
    symbol: str,
    column_map: dict[str, str] | None = None,
    timezone: str | None = None,
) -> pd.DataFrame:
    """Load provider CSV and normalize to canonical FX schema."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Real FX source file not found: {filepath}")

    src = pd.read_csv(path)
    if src.empty:
        raise ValueError(f"Source file has no rows: {filepath}")

    col_ts = _detect_source_column(src, "timestamp", column_map, required=True)
    col_open = _detect_source_column(src, "open", column_map, required=True)
    col_high = _detect_source_column(src, "high", column_map, required=True)
    col_low = _detect_source_column(src, "low", column_map, required=True)
    col_close = _detect_source_column(src, "close", column_map, required=True)
    col_volume = _detect_source_column(src, "volume", column_map, required=False)
    col_spread = _detect_source_column(src, "spread_bps", column_map, required=False)

    missing_required = [name for name, col in {
        "timestamp": col_ts,
        "open": col_open,
        "high": col_high,
        "low": col_low,
        "close": col_close,
    }.items() if col is None]
    if missing_required:
        raise ValueError(f"Missing required columns in source CSV: {missing_required}")

    out = pd.DataFrame(
        {
            "timestamp": _parse_timestamp(src[col_ts], timezone=timezone),
            "symbol": symbol,
            "open": pd.to_numeric(src[col_open], errors="coerce"),
            "high": pd.to_numeric(src[col_high], errors="coerce"),
            "low": pd.to_numeric(src[col_low], errors="coerce"),
            "close": pd.to_numeric(src[col_close], errors="coerce"),
        }
    )
    out["volume"] = (
        pd.to_numeric(src[col_volume], errors="coerce") if col_volume is not None else np.nan
    )
    out["spread_bps"] = (
        pd.to_numeric(src[col_spread], errors="coerce") if col_spread is not None else np.nan
    )

    raw_row_count = len(out)
    out = out.dropna(subset=["timestamp", "open", "high", "low", "close"]).copy()
    out = out.sort_values("timestamp")
    duplicate_count = int(out.duplicated(subset=["timestamp"]).sum())
    out = out.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)

    timeframe = infer_timeframe_from_series(out["timestamp"])
    out.attrs["ingestion_audit"] = {
        "source_file": str(path),
        "row_count_raw": raw_row_count,
        "row_count_clean": int(len(out)),
        "duplicate_rows_removed": duplicate_count,
        "detected_timeframe": timeframe,
        "columns_found": sorted(src.columns.tolist()),
        "spread_source_used": "csv" if col_spread is not None else "static_default",
    }
    return out


def normalize_fx_dataframe(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str | None = None,
) -> pd.DataFrame:
    """Normalize a dataframe to canonical schema and optional timeframe."""
    required = {"timestamp", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Cannot normalize data. Missing columns: {sorted(missing)}")

    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    out = out.dropna(subset=["timestamp", "open", "high", "low", "close"]).copy()
    out["symbol"] = symbol
    if "volume" not in out.columns:
        out["volume"] = np.nan
    if "spread_bps" not in out.columns:
        out["spread_bps"] = np.nan

    out = out.sort_values("timestamp")
    out = out.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)

    if timeframe:
        tf_map = {
            "m1": "1min",
            "1m": "1min",
            "5m": "5min",
            "15m": "15min",
            "30m": "30min",
            "h1": "1h",
            "h4": "4h",
            "d1": "1d",
            "1h": "1h",
            "4h": "4h",
            "1d": "1d",
        }
        tf = tf_map.get(timeframe.lower(), timeframe.lower())
        source_tf = infer_timeframe_from_series(out["timestamp"])
        source_tf_norm = tf_map.get(source_tf.lower(), source_tf.lower()) if isinstance(source_tf, str) else source_tf
        if source_tf_norm != tf:
            # Aggregate to requested research timeframe when source bar size differs.
            spread_agg = (
                out.set_index("timestamp")["spread_bps"]
                .resample(tf)
                .mean()
                .reset_index()
                .rename(columns={"spread_bps": "spread_bps"})
            )
            source_for_ohlc = out[["timestamp", "symbol", "open", "high", "low", "close", "volume"]]
            base = resample_ohlcv(source_for_ohlc, tf)
            out = base.merge(spread_agg, on="timestamp", how="left")
            out["symbol"] = symbol

    cols = ["timestamp", "symbol", "open", "high", "low", "close", "volume", "spread_bps"]
    return out[cols].sort_values("timestamp").reset_index(drop=True)


def estimate_missing_bars(df: pd.DataFrame, timeframe: str | None = None) -> int:
    if df.empty:
        return 0
    ts = pd.to_datetime(df["timestamp"], utc=True).sort_values()
    if timeframe is None or timeframe == "unknown":
        timeframe = infer_timeframe_from_series(ts)
    tf_map = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1h",
        "4h": "4h",
        "1d": "1d",
        "H1": "1h",
        "H4": "4h",
        "D1": "1d",
    }
    freq = tf_map.get(timeframe, timeframe)
    if freq == "unknown":
        return 0
    full = pd.date_range(ts.iloc[0], ts.iloc[-1], freq=freq, tz="UTC")
    # FX cash trading is typically closed on weekends; exclude weekend bars from expectation.
    full = full[full.dayofweek < 5]
    missing = len(full.difference(pd.DatetimeIndex(ts)))
    return int(max(missing, 0))


def build_data_quality_flags(df: pd.DataFrame, timeframe: str | None = None) -> dict[str, Any]:
    ts = pd.to_datetime(df["timestamp"], utc=True)
    non_monotonic = int((ts.diff().dropna().dt.total_seconds() <= 0).sum())
    duplicate_count = int(df.duplicated(subset=["timestamp"]).sum())
    missing_ohlc = int(df[["open", "high", "low", "close"]].isna().any(axis=1).sum())
    zero_range = int(((df["high"] - df["low"]).abs() <= 1e-12).sum())

    detected_tf = timeframe or infer_timeframe_from_series(ts)
    missing_bars_est = estimate_missing_bars(df, timeframe=detected_tf)
    abnormal_gaps = 0
    if len(ts) >= 3:
        delta = ts.diff().dropna().dt.total_seconds()
        med = float(delta.median()) if len(delta) > 0 else 0.0
        if med > 0:
            prev_ts = ts.shift(1).loc[delta.index]
            curr_ts = ts.loc[delta.index]
            weekend_like = (
                (prev_ts.dt.weekday >= 4)
                & ((curr_ts.dt.weekday <= 1) | (curr_ts.dt.weekday == 6))
                & (delta >= 40 * 3600)
                & (delta <= 80 * 3600)
            )
            abnormal_gaps = int(((delta > (3.0 * med)) & (~weekend_like)).sum())

    return {
        "missing_ohlc_rows": missing_ohlc,
        "non_monotonic_timestamps": non_monotonic,
        "abnormal_gaps": int(abnormal_gaps),
        "zero_range_candles_count": zero_range,
        "duplicate_timestamp_count": duplicate_count,
        "missing_bars_estimate": int(missing_bars_est),
        "detected_timeframe": detected_tf,
    }
