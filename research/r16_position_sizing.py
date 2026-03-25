from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

EARLY_MFE_THRESHOLD = 0.00065
EARLY_RETURN3_THRESHOLD = -0.00024


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def compute_position_size(row: pd.Series | dict[str, Any]) -> float:
    """Map fixed early-feature quality score to a size multiplier."""
    early_mfe = _safe_float(row.get("early_mfe"))
    early_return_3 = _safe_float(row.get("early_return_3"))
    early_return_1 = _safe_float(row.get("early_return_1"))
    early_slope = _safe_float(row.get("early_slope"))

    score = 0
    if np.isfinite(early_mfe) and early_mfe < EARLY_MFE_THRESHOLD:
        score -= 1
    if np.isfinite(early_return_3) and early_return_3 < EARLY_RETURN3_THRESHOLD:
        score -= 1
    if np.isfinite(early_return_1) and early_return_1 < 0.0:
        score -= 1
    if np.isfinite(early_slope) and early_slope < 0.0:
        score -= 1

    if score <= -3:
        return 0.5
    if score == -2:
        return 0.75
    if score == -1:
        return 1.0
    return 1.25


__all__ = [
    "EARLY_MFE_THRESHOLD",
    "EARLY_RETURN3_THRESHOLD",
    "compute_position_size",
]
