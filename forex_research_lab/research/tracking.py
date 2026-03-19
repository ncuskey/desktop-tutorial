"""Simple experiment tracking via in-memory DataFrame and optional SQLite."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


@dataclass
class ExperimentTracker:
    db_path: str | Path | None = None
    records: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.db_path is not None:
            self._init_db()

    def _init_db(self) -> None:
        assert self.db_path is not None
        path = Path(self.db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    run_timestamp TEXT,
                    run_name TEXT,
                    strategy TEXT,
                    symbol TEXT,
                    timeframe TEXT,
                    params_json TEXT,
                    metrics_json TEXT
                )
                """
            )
            conn.commit()

    def log_run(
        self,
        run_name: str,
        strategy: str,
        params: Mapping[str, Any],
        symbol: str,
        timeframe: str,
        metrics: Mapping[str, float],
    ) -> None:
        timestamp = datetime.now(tz=timezone.utc).isoformat()
        record = {
            "run_timestamp": timestamp,
            "run_name": run_name,
            "strategy": strategy,
            "symbol": symbol,
            "timeframe": timeframe,
            "params": dict(params),
            **{f"metric_{k}": float(v) for k, v in metrics.items()},
        }
        self.records.append(record)

        if self.db_path is not None:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO experiments (
                        run_timestamp, run_name, strategy, symbol, timeframe, params_json, metrics_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        timestamp,
                        run_name,
                        strategy,
                        symbol,
                        timeframe,
                        json.dumps(dict(params)),
                        json.dumps(dict(metrics)),
                    ),
                )
                conn.commit()

    def to_dataframe(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame()
        return pd.DataFrame(self.records)
