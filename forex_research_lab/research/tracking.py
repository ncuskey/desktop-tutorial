"""Simple experiment tracking to in-memory records and SQLite."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class ExperimentTracker:
    db_path: str | Path | None = None
    _records: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.db_path is None:
            self._conn = None
            return

        db_path = Path(self.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                run_id TEXT,
                strategy TEXT,
                params TEXT,
                symbol TEXT,
                timeframe TEXT,
                split TEXT,
                metric_name TEXT,
                metric_value REAL
            )
            """
        )
        self._conn.commit()

    def log_run(
        self,
        run_id: str,
        strategy: str,
        params: str,
        symbol: str,
        timeframe: str,
        split: str,
        metrics: dict[str, float],
    ) -> None:
        for metric_name, metric_value in metrics.items():
            record = {
                "run_id": run_id,
                "strategy": strategy,
                "params": params,
                "symbol": symbol,
                "timeframe": timeframe,
                "split": split,
                "metric_name": metric_name,
                "metric_value": float(metric_value),
            }
            self._records.append(record)
            if self._conn is not None:
                self._conn.execute(
                    """
                    INSERT INTO experiments (
                        run_id, strategy, params, symbol, timeframe, split, metric_name, metric_value
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record["run_id"],
                        record["strategy"],
                        record["params"],
                        record["symbol"],
                        record["timeframe"],
                        record["split"],
                        record["metric_name"],
                        record["metric_value"],
                    ),
                )
        if self._conn is not None:
            self._conn.commit()

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._records)

    def close(self) -> None:
        if getattr(self, "_conn", None) is not None:
            self._conn.close()
            self._conn = None
