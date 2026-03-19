"""
Experiment tracking — logs all runs to an in-memory DataFrame and
optionally persists them to a SQLite database.

Each experiment record captures:
  - run_id, timestamp
  - strategy name + params
  - symbol + timeframe
  - all MetricsResult fields
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from metrics.performance import MetricsResult


class ExperimentTracker:
    """Logs experiment runs and provides querying / persistence.

    Parameters
    ----------
    db_path:
        Optional path to a SQLite file.  If None, runs are kept in memory only.
    """

    _TABLE = "experiments"

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._records: list[dict] = []
        self.db_path = Path(db_path) if db_path else None
        if self.db_path:
            self._init_db()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log(
        self,
        strategy: str,
        params: dict[str, Any],
        symbol: str,
        timeframe: str,
        metrics: MetricsResult,
        extra: dict[str, Any] | None = None,
    ) -> str:
        """Log a single experiment run.

        Returns
        -------
        run_id — a short hash identifying this run.
        """
        run_id = self._make_run_id(strategy, params, symbol, timeframe)
        record = {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "strategy": strategy,
            "symbol": symbol,
            "timeframe": timeframe,
            "params": json.dumps(params, default=str),
            **metrics.to_dict(),
        }
        if extra:
            record.update(extra)

        self._records.append(record)

        if self.db_path:
            self._write_to_db(record)

        return run_id

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    @property
    def results(self) -> pd.DataFrame:
        """Return all logged runs as a DataFrame."""
        if not self._records:
            return pd.DataFrame()
        df = pd.DataFrame(self._records)
        df["params"] = df["params"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )
        return df

    def best_runs(
        self,
        metric: str = "sharpe",
        n: int = 10,
        strategy: str | None = None,
        symbol: str | None = None,
    ) -> pd.DataFrame:
        """Return top-n runs sorted by metric."""
        df = self.results
        if df.empty:
            return df
        if strategy:
            df = df[df["strategy"] == strategy]
        if symbol:
            df = df[df["symbol"] == symbol]
        if metric not in df.columns:
            return df.head(n)
        return df.nlargest(n, metric)

    def pivot_heatmap(
        self, param_x: str, param_y: str, metric: str = "sharpe"
    ) -> pd.DataFrame:
        """Return a pivot table for heatmap visualisation."""
        df = self.results
        if df.empty:
            return df
        # Expand params dict column into separate columns
        param_df = df["params"].apply(pd.Series)
        full = pd.concat([df.drop(columns=["params"]), param_df], axis=1)
        if param_x not in full.columns or param_y not in full.columns:
            return pd.DataFrame()
        pivot = full.pivot_table(
            values=metric, index=param_y, columns=param_x, aggfunc="mean"
        )
        return pivot

    def load_from_db(self) -> pd.DataFrame:
        """Load all records from SQLite into memory."""
        if not self.db_path or not self.db_path.exists():
            return pd.DataFrame()
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql(f"SELECT * FROM {self._TABLE}", conn)

    # ------------------------------------------------------------------

    def _make_run_id(
        self, strategy: str, params: dict, symbol: str, timeframe: str
    ) -> str:
        payload = f"{strategy}|{sorted(params.items())}|{symbol}|{timeframe}|{datetime.now(timezone.utc)}"
        return hashlib.md5(payload.encode()).hexdigest()[:8]

    def _init_db(self) -> None:
        if self.db_path is None:
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._TABLE} (
                    run_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    strategy TEXT,
                    symbol TEXT,
                    timeframe TEXT,
                    params TEXT,
                    cagr REAL, sharpe REAL, sortino REAL,
                    max_drawdown REAL, profit_factor REAL,
                    win_rate REAL, expectancy REAL,
                    n_trades INTEGER, calmar REAL,
                    volatility_ann REAL
                )
                """
            )

    def _write_to_db(self, record: dict) -> None:
        if self.db_path is None:
            return
        cols = [
            "run_id", "timestamp", "strategy", "symbol", "timeframe", "params",
            "cagr", "sharpe", "sortino", "max_drawdown", "profit_factor",
            "win_rate", "expectancy", "n_trades", "calmar", "volatility_ann",
        ]
        values = {c: record.get(c) for c in cols}
        placeholders = ", ".join(f":{c}" for c in cols)
        col_names = ", ".join(cols)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO {self._TABLE} ({col_names}) VALUES ({placeholders})",
                values,
            )
