from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


class ExperimentTracker:
    def __init__(self, db_path: str | Path = "outputs/experiments.sqlite") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts_utc TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    metrics_json TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def log_run(
        self,
        strategy: str,
        params: dict,
        symbol: str,
        timeframe: str,
        metrics: dict,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO experiments (
                    ts_utc, strategy, params_json, symbol, timeframe, metrics_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    strategy,
                    json.dumps(params, sort_keys=True),
                    symbol,
                    timeframe,
                    json.dumps(metrics, sort_keys=True),
                ),
            )
            conn.commit()

    def to_dataframe(self) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query("SELECT * FROM experiments ORDER BY id ASC", conn)
