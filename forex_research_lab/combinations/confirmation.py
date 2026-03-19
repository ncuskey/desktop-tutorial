"""Signal confirmation logic."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


def confirmation_signal(signals: Mapping[str, pd.Series], min_confirmations: int | None = None) -> pd.Series:
    """Only trade when enough strategies agree on the same direction."""
    if not signals:
        raise ValueError("At least one signal series is required")

    frame = pd.concat(signals, axis=1).fillna(0.0)
    required = min_confirmations or frame.shape[1]

    positive_votes = (frame > 0.0).sum(axis=1)
    negative_votes = (frame < 0.0).sum(axis=1)

    confirmed = pd.Series(0.0, index=frame.index, dtype=float)
    confirmed.loc[positive_votes >= required] = 1.0
    confirmed.loc[negative_votes >= required] = -1.0
    return confirmed
