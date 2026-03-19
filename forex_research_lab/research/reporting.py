"""Artifact generation for research experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _json_safe_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    safe_metrics: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.floating)) and not np.isfinite(value):
            safe_metrics[key] = None
        else:
            safe_metrics[key] = value
    return safe_metrics


def _save_series_plot(series: pd.Series, title: str, y_label: str, output_path: Path) -> None:
    if series.empty:
        return

    figure, axis = plt.subplots(figsize=(11, 4))
    series.plot(ax=axis, linewidth=1.5)
    axis.set_title(title)
    axis.set_xlabel("Timestamp")
    axis.set_ylabel(y_label)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _save_heatmap(search_history: list[pd.DataFrame], output_directory: Path, metric: str) -> None:
    if not search_history:
        return

    combined = pd.concat(search_history, ignore_index=True)
    if combined.empty:
        return

    params_column = combined["params"].dropna()
    if params_column.empty:
        return

    param_keys = list(params_column.iloc[0].keys())
    if len(param_keys) < 2:
        combined.to_csv(output_directory / "parameter_robustness.csv", index=False)
        return

    x_key, y_key = param_keys[0], param_keys[1]
    grouped = combined.groupby([y_key, x_key], dropna=False)[metric].mean().reset_index()
    pivot = grouped.pivot(index=y_key, columns=x_key, values=metric).sort_index().sort_index(axis=1)
    pivot.to_csv(output_directory / "parameter_robustness_heatmap.csv")

    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(pivot.to_numpy(), aspect="auto", origin="lower", cmap="viridis")
    axis.set_xticks(range(len(pivot.columns)))
    axis.set_xticklabels([str(value) for value in pivot.columns])
    axis.set_yticks(range(len(pivot.index)))
    axis.set_yticklabels([str(value) for value in pivot.index])
    axis.set_xlabel(x_key)
    axis.set_ylabel(y_key)
    axis.set_title(f"{metric} robustness heatmap")
    figure.colorbar(image, ax=axis, label=metric)
    figure.tight_layout()
    figure.savefig(output_directory / "parameter_robustness_heatmap.png", dpi=150)
    plt.close(figure)


def save_experiment_outputs(result: Any, output_directory: str | Path, heatmap_metric: str = "Sharpe") -> Path:
    """Persist metrics, curves, split details, and a parameter robustness heatmap."""
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)

    metrics_path = destination / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe_metrics(result.aggregate_metrics), handle, indent=2, sort_keys=True)

    pd.DataFrame([result.aggregate_metrics]).to_csv(destination / "metrics.csv", index=False)

    if not result.equity_curve.empty:
        result.equity_curve.rename("equity").to_csv(destination / "equity_curve.csv", header=True)
        _save_series_plot(result.equity_curve, "Equity Curve", "Equity", destination / "equity_curve.png")

    if not result.drawdown_curve.empty:
        result.drawdown_curve.rename("drawdown").to_csv(destination / "drawdown_curve.csv", header=True)
        _save_series_plot(result.drawdown_curve, "Drawdown Curve", "Drawdown", destination / "drawdown_curve.png")

    if not result.split_summary.empty:
        result.split_summary.to_csv(destination / "walk_forward_splits.csv", index=False)

    if not result.trades.empty:
        result.trades.to_csv(destination / "trades.csv", index=False)

    _save_heatmap(result.search_history, destination, metric=heatmap_metric)
    return destination
