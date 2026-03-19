from __future__ import annotations

import pandas as pd


def carry_proxy_signals(df: pd.DataFrame, params: dict) -> pd.Series:
    """Simple carry placeholder using interest differential proxy.

    Expects either:
      - `interest_diff` column in df, or
      - `constant_diff` in params.
    Positive differential implies long (+1), negative implies short (-1).
    """

    if "interest_diff" in df.columns:
        diff = df["interest_diff"]
    else:
        diff = pd.Series(float(params.get("constant_diff", 0.0)), index=df.index)

    signal = pd.Series(0, index=df.index, dtype=int)
    signal[diff > 0] = 1
    signal[diff < 0] = -1
    return signal
