"""Plotting helpers for research artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def plot_equity_curves(equity_curves: dict[str, pd.Series], output_path: str | Path) -> None:
    path = _ensure_parent(output_path)
    plt.figure(figsize=(12, 6))
    for name, curve in equity_curves.items():
        plt.plot(curve.index, curve.values, label=name)
    plt.title("Equity Curves")
    plt.xlabel("Time")
    plt.ylabel("Equity")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_drawdown_curves(drawdown_curves: dict[str, pd.Series], output_path: str | Path) -> None:
    path = _ensure_parent(output_path)
    plt.figure(figsize=(12, 5))
    for name, curve in drawdown_curves.items():
        plt.plot(curve.index, curve.values, label=name)
    plt.title("Drawdown Curves")
    plt.xlabel("Time")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_heatmap(
    matrix: pd.DataFrame,
    title: str,
    output_path: str | Path,
    x_label: str = "",
    y_label: str = "",
) -> None:
    path = _ensure_parent(output_path)
    if matrix.empty:
        raise ValueError("Cannot plot empty heatmap matrix")

    values = matrix.values.astype(float)
    plt.figure(figsize=(8, 6))
    im = plt.imshow(values, aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(im, label="Value")
    plt.title(title)
    plt.xlabel(x_label or matrix.columns.name or "x")
    plt.ylabel(y_label or matrix.index.name or "y")

    plt.xticks(ticks=np.arange(len(matrix.columns)), labels=matrix.columns, rotation=45, ha="right")
    plt.yticks(ticks=np.arange(len(matrix.index)), labels=matrix.index)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
