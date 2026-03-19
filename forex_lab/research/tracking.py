from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class ExperimentTracker:
    db_path: Optional[str] = None
    _records: List[Dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.db_path:
            path = Path(self.db_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS experiments (
                        ts TEXT,
                        strategy TEXT,
                        symbol TEXT,
                        timeframe TEXT,
                        params_json TEXT,
                        metrics_json TEXT
                    )
                    """
                )

    def log(
        self,
        strategy: str,
        params: Dict,
        symbol: str,
        timeframe: str,
        metrics: Dict,
    ) -> None:
        row = {
            "ts": pd.Timestamp.utcnow().isoformat(),
            "strategy": strategy,
            "symbol": symbol,
            "timeframe": timeframe,
            "params": params,
            "metrics": metrics,
        }
        self._records.append(row)
        if self.db_path:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO experiments (ts, strategy, symbol, timeframe, params_json, metrics_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["ts"],
                        strategy,
                        symbol,
                        timeframe,
                        json.dumps(params, sort_keys=True),
                        json.dumps(metrics, sort_keys=True),
                    ),
                )

    def to_dataframe(self) -> pd.DataFrame:
        if not self._records:
            return pd.DataFrame(columns=["ts", "strategy", "symbol", "timeframe", "params", "metrics"])
        return pd.DataFrame(self._records)
