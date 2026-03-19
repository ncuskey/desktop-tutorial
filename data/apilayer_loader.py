from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


DEFAULT_APILAYER_LIVE_URL = "http://apilayer.net/api/live"


@dataclass
class ApiLayerLiveResponse:
    timestamp: pd.Timestamp
    source: str
    quotes: dict[str, float]
    raw_success: bool
    raw_error: str | None = None


def _to_utc_timestamp(unix_ts: int | float | None) -> pd.Timestamp:
    if unix_ts is None:
        return pd.Timestamp(datetime.now(timezone.utc))
    return pd.Timestamp(datetime.fromtimestamp(float(unix_ts), tz=timezone.utc))


def fetch_apilayer_live_quotes(
    access_key: str,
    currencies: Iterable[str],
    source: str = "USD",
    endpoint: str = DEFAULT_APILAYER_LIVE_URL,
    timeout_seconds: int = 30,
) -> ApiLayerLiveResponse:
    """
    Fetch latest FX quotes from apilayer live endpoint.

    Returns quotes in raw provider format, e.g. "USDEUR", "USDJPY".
    """
    currency_csv = ",".join(sorted({c.upper() for c in currencies if c}))
    source = source.upper()
    params = {
        "access_key": access_key,
        "currencies": currency_csv,
        "source": source,
        "format": 1,
    }
    url = f"{endpoint}?{urlencode(params)}"

    req = Request(url=url, method="GET")
    with urlopen(req, timeout=timeout_seconds) as resp:
        payload = resp.read().decode("utf-8")
    data = json.loads(payload)

    success = bool(data.get("success", False))
    if not success:
        err_obj = data.get("error", {})
        err_msg = err_obj.get("info") or str(err_obj) or "Unknown apilayer error"
        return ApiLayerLiveResponse(
            timestamp=_to_utc_timestamp(data.get("timestamp")),
            source=source,
            quotes={},
            raw_success=False,
            raw_error=err_msg,
        )

    raw_quotes = data.get("quotes", {}) or {}
    quotes: dict[str, float] = {}
    for key, value in raw_quotes.items():
        try:
            quotes[str(key).upper()] = float(value)
        except (TypeError, ValueError):
            continue

    return ApiLayerLiveResponse(
        timestamp=_to_utc_timestamp(data.get("timestamp")),
        source=source,
        quotes=quotes,
        raw_success=True,
        raw_error=None,
    )


def _symbol_to_price(symbol: str, source: str, source_quotes: dict[str, float]) -> float | None:
    """
    Convert source-based quotes to target 6-char FX symbol price.

    Example:
      source=USD, quote USDEUR -> EURUSD = 1 / USDEUR
      source=USD, quote USDJPY -> USDJPY = USDJPY
      source=USD, cross EURGBP -> USDGBP / USDEUR
    """
    symbol = symbol.upper()
    source = source.upper()
    if len(symbol) != 6:
        return None
    base = symbol[:3]
    quote = symbol[3:]

    # Direct same-as-source base.
    if base == source:
        return source_quotes.get(f"{source}{quote}")

    # Inverse same-as-source quote.
    if quote == source:
        src_base = source_quotes.get(f"{source}{base}")
        if src_base and src_base > 0:
            return 1.0 / src_base
        return None

    # Cross conversion via source:
    # base/quote = (source->quote) / (source->base)
    src_quote = source_quotes.get(f"{source}{quote}")
    src_base = source_quotes.get(f"{source}{base}")
    if src_quote and src_base and src_base > 0:
        return src_quote / src_base
    return None


def build_canonical_snapshot(
    response: ApiLayerLiveResponse,
    symbols: Iterable[str],
) -> pd.DataFrame:
    """
    Convert live quote payload into canonical rows.

    open/high/low/close are equal for a single snapshot row.
    """
    rows: list[dict[str, object]] = []
    for symbol in symbols:
        price = _symbol_to_price(symbol=symbol, source=response.source, source_quotes=response.quotes)
        if price is None or not np.isfinite(price) or price <= 0:
            continue
        rows.append(
            {
                "timestamp": response.timestamp,
                "symbol": symbol.upper(),
                "open": float(price),
                "high": float(price),
                "low": float(price),
                "close": float(price),
                "volume": np.nan,
                "spread_bps": np.nan,
            }
        )
    return pd.DataFrame(rows)


def append_snapshot_rows_to_symbol_csv(
    snapshot_df: pd.DataFrame,
    output_dir: str | Path = "data/real",
) -> list[Path]:
    """
    Append one-row snapshots to symbol CSV files compatible with real_loader.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if snapshot_df.empty:
        return written

    expected_cols = ["timestamp", "symbol", "open", "high", "low", "close", "volume", "spread_bps"]
    available_cols = [c for c in expected_cols if c in snapshot_df.columns]
    frame = snapshot_df[available_cols].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values(["symbol", "timestamp"])

    for symbol, grp in frame.groupby("symbol", sort=True):
        path = out_dir / f"{symbol}_H1.csv"
        chunk = grp.copy()
        if path.exists():
            prev = pd.read_csv(path)
            prev["timestamp"] = pd.to_datetime(prev["timestamp"], utc=True, errors="coerce")
            chunk = pd.concat([prev, chunk], axis=0, ignore_index=True)
        chunk = chunk.dropna(subset=["timestamp", "open", "high", "low", "close"])
        chunk = chunk.drop_duplicates(subset=["timestamp"], keep="last")
        chunk = chunk.sort_values("timestamp")
        chunk.to_csv(path, index=False)
        written.append(path)
    return written
