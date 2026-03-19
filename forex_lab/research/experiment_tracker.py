"""Experiment tracking - log runs to DataFrame or SQLite."""

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd


class ExperimentTracker:
    """
    Log experiment runs with strategy, params, symbol, timeframe, metrics.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path
        self.runs: list[dict] = []

    def log(
        self,
        strategy: str,
        params: dict[str, Any],
        symbol: str,
        timeframe: str,
        metrics: dict[str, float],
        extra: Optional[dict] = None,
    ) -> None:
        """Log a single experiment run."""
        run = {
            "strategy": strategy,
            "params": json.dumps(params),
            "symbol": symbol,
            "timeframe": timeframe,
            **metrics,
        }
        if extra:
            run["extra"] = json.dumps(extra)
        self.runs.append(run)

    def to_dataframe(self) -> pd.DataFrame:
        """Export runs to DataFrame."""
        return pd.DataFrame(self.runs)

    def save(self, path: Optional[str] = None) -> None:
        """Save to CSV or SQLite."""
        path = path or self.db_path
        if not path:
            return
        df = self.to_dataframe()
        if path.endswith(".db") or path.endswith(".sqlite"):
            import sqlite3
            conn = sqlite3.connect(path)
            df.to_sql("experiments", conn, if_exists="append", index=False)
            conn.close()
        else:
            df.to_csv(path, index=False)

    def load(self, path: str) -> pd.DataFrame:
        """Load previous runs from CSV."""
        return pd.read_csv(path)
