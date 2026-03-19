"""
Plotting utilities for the Forex Strategy Research Lab.

All plots use matplotlib with a clean style.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")  # headless rendering
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import seaborn as sns
    _HAS_PLOT = True
except ImportError:
    _HAS_PLOT = False


_STYLE = "seaborn-v0_8-darkgrid"
_FIG_DPI = 120


def _check_deps() -> None:
    if not _HAS_PLOT:
        raise ImportError("matplotlib and seaborn required for plotting.")


def plot_equity_drawdown(
    equity: pd.Series,
    drawdown: pd.Series,
    title: str = "Strategy Performance",
    save_path: str | Path | None = None,
) -> None:
    """Plot equity curve and drawdown in a two-panel figure."""
    _check_deps()
    try:
        plt.style.use(_STYLE)
    except Exception:
        pass

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})

    ax_eq, ax_dd = axes
    ax_eq.plot(equity.index, equity.values, color="#2196F3", linewidth=1.5, label="Equity")
    ax_eq.set_ylabel("Equity (× initial)")
    ax_eq.set_title(title, fontsize=14, fontweight="bold")
    ax_eq.legend()

    ax_dd.fill_between(drawdown.index, drawdown.values, 0, color="#F44336", alpha=0.6)
    ax_dd.set_ylabel("Drawdown")
    ax_dd.set_xlabel("Date")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_walk_forward(
    wf_result,
    title: str = "Walk-Forward Out-of-Sample Equity",
    save_path: str | Path | None = None,
) -> None:
    """Plot concatenated OOS equity with per-window colouring."""
    _check_deps()
    try:
        plt.style.use(_STYLE)
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(14, 6))

    colors = plt.cm.tab10.colors
    for i, window in enumerate(wf_result.windows):
        c = colors[i % len(colors)]
        eq = (1 + window.test_returns).cumprod()
        ax.plot(eq.index, eq.values, color=c, linewidth=1.2,
                label=f"W{window.window_id} (Sharpe={window.test_metrics.sharpe:.2f})")

    if len(wf_result.oos_equity) > 0:
        eq = wf_result.oos_equity
        # Normalise to 1
        eq = eq / eq.iloc[0]
        ax.plot(eq.index, eq.values, color="black", linewidth=2.5,
                linestyle="--", label="OOS Combined", zorder=5)

    ax.set_ylabel("Equity")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=7, ncol=3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_table(
    metrics_dict: dict[str, Any],
    title: str = "Performance Metrics",
    save_path: str | Path | None = None,
) -> None:
    """Render a metrics table as a figure."""
    _check_deps()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")

    rows = [(k, f"{v:.4f}" if isinstance(v, float) else str(v))
            for k, v in metrics_dict.items()]
    table = ax.table(
        cellText=rows,
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=20)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_param_heatmap(
    sweep_results: pd.DataFrame,
    param_x: str,
    param_y: str,
    metric: str = "sharpe",
    title: str | None = None,
    save_path: str | Path | None = None,
) -> None:
    """Plot a heatmap of metric values over a 2D parameter grid."""
    _check_deps()

    if param_x not in sweep_results.columns or param_y not in sweep_results.columns:
        print(f"[plot_param_heatmap] params {param_x}/{param_y} not in results.")
        return

    pivot = sweep_results.pivot_table(
        values=metric, index=param_y, columns=param_x, aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(title or f"{metric} by {param_x} × {param_y}", fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_bootstrap_distribution(
    bootstrap_result,
    metric: str = "sharpe",
    title: str | None = None,
    save_path: str | Path | None = None,
) -> None:
    """Plot bootstrap distribution of a metric with original value marked."""
    _check_deps()

    dist = bootstrap_result.metric_distributions.get(metric, np.array([]))
    if len(dist) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(dist, bins=50, color="#42A5F5", edgecolor="white", alpha=0.8, density=True)

    if bootstrap_result.original_metrics:
        orig = getattr(bootstrap_result.original_metrics, metric, None)
        if orig is not None:
            ax.axvline(orig, color="red", linewidth=2, label=f"Original: {orig:.3f}")

    p5, p95 = bootstrap_result.confidence_interval(metric, alpha=0.10)
    ax.axvline(p5, color="orange", linestyle="--", linewidth=1.5, label=f"P5: {p5:.3f}")
    ax.axvline(p95, color="orange", linestyle="--", linewidth=1.5, label=f"P95: {p95:.3f}")
    ax.axvline(0, color="black", linewidth=1, linestyle=":")

    ax.set_xlabel(metric)
    ax.set_ylabel("Density")
    ax.set_title(title or f"Bootstrap Distribution: {metric}", fontsize=13)
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
