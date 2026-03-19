"""Experiment logging to memory and optional SQLite storage."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _to_native(value: Any) -> Any:
    """Convert numpy and pandas objects into JSON-friendly Python objects."""
    if is_dataclass(value):
        return {key: _to_native(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _to_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_native(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


class ExperimentTracker:
    """Track research runs in memory and optionally persist them to SQLite."""

    def __init__(self, sqlite_path: str | Path | None = None) -> None:
        self.sqlite_path = Path(sqlite_path) if sqlite_path is not None else None
        self.records: list[dict[str, Any]] = []

        if self.sqlite_path is not None:
            self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            self._initialize_database()

    def _initialize_database(self) -> None:
        with sqlite3.connect(self.sqlite_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    context_json TEXT
                )
                """
            )

    def log_run(
        self,
        strategy: str,
        params: dict[str, Any],
        symbol: str,
        timeframe: str,
        metrics: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record a run in memory and, if configured, in SQLite."""
        timestamp = datetime.now(UTC).isoformat()
        record = {
            "timestamp": timestamp,
            "strategy": strategy,
            "symbol": symbol,
            "timeframe": timeframe,
            "params": _to_native(params),
            "context": _to_native(context or {}),
        }
        record.update({key: _to_native(value) for key, value in metrics.items()})
        self.records.append(record)

        if self.sqlite_path is not None:
            with sqlite3.connect(self.sqlite_path) as connection:
                connection.execute(
                    """
                    INSERT INTO experiments (timestamp, strategy, symbol, timeframe, params_json, metrics_json, context_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        timestamp,
                        strategy,
                        symbol,
                        timeframe,
                        json.dumps(_to_native(params), sort_keys=True),
                        json.dumps(_to_native(metrics), sort_keys=True),
                        json.dumps(_to_native(context or {}), sort_keys=True),
                    ),
                )

        return record

    def to_frame(self) -> pd.DataFrame:
        """Return all logged experiments as a dataframe."""
        return pd.DataFrame(self.records)
