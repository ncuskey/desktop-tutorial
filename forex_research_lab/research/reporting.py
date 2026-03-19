"""Reporting and visualization helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_equity_drawdown_plot(frame: pd.DataFrame, output_path: str | Path, title: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(frame["timestamp"], frame["equity"], color="tab:blue", linewidth=1.2)
    axes[0].set_title(f"{title} - Equity Curve")
    axes[0].set_ylabel("Equity")
    axes[0].grid(alpha=0.25)

    axes[1].fill_between(frame["timestamp"], frame["drawdown"], 0.0, color="tab:red", alpha=0.35)
    axes[1].set_title(f"{title} - Drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_metrics_table(metrics: dict[str, float], output_path: str | Path) -> pd.DataFrame:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"metric": list(metrics.keys()), "value": list(metrics.values())})
    frame.to_csv(path, index=False)
    return frame


def save_parameter_heatmap(
    sweep_results: pd.DataFrame,
    output_path: str | Path,
    x: str,
    y: str,
    value: str = "objective",
    title: str | None = None,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pivot = sweep_results.pivot_table(index=y, columns=x, values=value, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, cmap="viridis", annot=True, fmt=".2f", ax=ax)
    ax.set_title(title or "Parameter Robustness Heatmap")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
