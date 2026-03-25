from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# R1.5.3 fixed, non-tuned thresholds.
EARLY_MFE_THRESHOLD = 0.00065
EARLY_RETURN3_THRESHOLD = -0.00024
FAILURE_SCORE_CUTOFF = -2


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def compute_failure_score(row: pd.Series | dict[str, Any]) -> int:
    """Compute fixed composite failure score from early features.

    Score starts at 0 and decrements by 1 for each weak condition.
    """
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
    return int(score)


__all__ = [
    "EARLY_MFE_THRESHOLD",
    "EARLY_RETURN3_THRESHOLD",
    "FAILURE_SCORE_CUTOFF",
    "compute_failure_score",
]

