"""Experiment tracking — log runs to SQLite or in-memory DataFrame."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


class ExperimentTracker:
    """Lightweight experiment logger backed by SQLite."""

    def __init__(self, db_path: str = "experiments.db"):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._ensure_table()

    def _ensure_table(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                strategy TEXT,
                symbol TEXT,
                timeframe TEXT,
                params TEXT,
                metrics TEXT,
                notes TEXT
            )
            """
        )
        self._conn.commit()

    def log(
        self,
        strategy: str,
        symbol: str,
        timeframe: str,
        params: dict[str, Any],
        metrics: dict[str, float],
        notes: str = "",
    ) -> int:
        """Log an experiment run. Returns the row id."""
        ts = datetime.now(timezone.utc).isoformat()
        cur = self._conn.execute(
            """
            INSERT INTO experiments (timestamp, strategy, symbol, timeframe, params, metrics, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (ts, strategy, symbol, timeframe, json.dumps(params), json.dumps(metrics), notes),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def to_dataframe(self) -> pd.DataFrame:
        """Return all logged experiments as a DataFrame."""
        df = pd.read_sql_query("SELECT * FROM experiments", self._conn)
        if not df.empty:
            df["params"] = df["params"].apply(json.loads)
            df["metrics"] = df["metrics"].apply(json.loads)
        return df

    def close(self) -> None:
        self._conn.close()

    def __del__(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
