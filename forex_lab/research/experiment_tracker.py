"""Experiment tracking — log and query strategy evaluation runs."""

from __future__ import annotations

import pandas as pd
import numpy as np
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Any


class ExperimentTracker:
    """Track experiment runs in a SQLite database or in-memory DataFrame.

    Each run records: strategy name, params, symbol, timeframe, all metrics,
    and a timestamp.
    """

    def __init__(self, db_path: str | Path | None = None):
        self._runs: list[dict[str, Any]] = []
        self.db_path = db_path
        if db_path:
            self._init_db(db_path)

    def log_run(
        self,
        strategy: str,
        params: dict[str, Any],
        symbol: str,
        timeframe: str,
        metrics: dict[str, float],
        tags: dict[str, str] | None = None,
    ) -> None:
        """Log a single experiment run."""
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "strategy": strategy,
            "params": json.dumps(params),
            "symbol": symbol,
            "timeframe": timeframe,
            **(tags or {}),
            **metrics,
        }
        self._runs.append(record)
        if self.db_path:
            self._insert_db(record)

    def to_dataframe(self) -> pd.DataFrame:
        """Return all logged runs as a DataFrame."""
        return pd.DataFrame(self._runs)

    def best_runs(
        self,
        metric: str = "sharpe",
        n: int = 10,
        ascending: bool = False,
    ) -> pd.DataFrame:
        """Return top N runs sorted by a given metric."""
        df = self.to_dataframe()
        if metric not in df.columns or len(df) == 0:
            return df
        return df.sort_values(metric, ascending=ascending).head(n)

    def filter(self, **kwargs: Any) -> pd.DataFrame:
        """Filter runs by exact column matches."""
        df = self.to_dataframe()
        for key, value in kwargs.items():
            if key in df.columns:
                df = df[df[key] == value]
        return df

    def _init_db(self, db_path: str | Path) -> None:
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                strategy TEXT,
                params TEXT,
                symbol TEXT,
                timeframe TEXT,
                cagr REAL,
                total_return REAL,
                sharpe REAL,
                sortino REAL,
                max_drawdown REAL,
                profit_factor REAL,
                win_rate REAL,
                expectancy REAL,
                trade_count REAL,
                volatility REAL
            )
            """
        )
        conn.commit()
        conn.close()

    def _insert_db(self, record: dict[str, Any]) -> None:
        conn = sqlite3.connect(str(self.db_path))
        cols = [
            "timestamp", "strategy", "params", "symbol", "timeframe",
            "cagr", "total_return", "sharpe", "sortino", "max_drawdown",
            "profit_factor", "win_rate", "expectancy", "trade_count", "volatility",
        ]
        values = [record.get(c) for c in cols]
        placeholders = ", ".join(["?"] * len(cols))
        col_names = ", ".join(cols)
        conn.execute(f"INSERT INTO runs ({col_names}) VALUES ({placeholders})", values)
        conn.commit()
        conn.close()
