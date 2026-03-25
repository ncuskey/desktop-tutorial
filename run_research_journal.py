#!/usr/bin/env python3
"""Append timestamped research-journal entries to consolidated report."""

from __future__ import annotations

import argparse
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


def _fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _git(cmd: list[str]) -> str:
    try:
        return (
            subprocess.check_output(["git", *cmd], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def _summarize_r12(outputs_dir: Path) -> list[str]:
    path = outputs_dir / "regime_gated_comparison.csv"
    df = _safe_read_csv(path)
    if df is None or df.empty:
        return ["- R1.2.3: not available"]

    lines: list[str] = []
    required = {"symbol", "variant", "sharpe", "expectancy", "max_dd", "trade_count"}
    if not required.issubset(df.columns):
        return ["- R1.2.3: file present but missing expected columns"]

    for symbol, grp in df.groupby("symbol"):
        variants = set(grp["variant"].astype(str).tolist())
        if {"gated", "unfiltered"} - variants:
            continue
        gated = grp[grp["variant"] == "gated"].iloc[0]
        unfiltered = grp[grp["variant"] == "unfiltered"].iloc[0]
        lines.append(
            (
                f"- {symbol}: "
                f"dSharpe={_fmt(gated['sharpe'] - unfiltered['sharpe'])}, "
                f"dExpectancy={_fmt(gated['expectancy'] - unfiltered['expectancy'], 6)}, "
                f"dMaxDD={_fmt(gated['max_dd'] - unfiltered['max_dd'])}, "
                f"dTrades={int(gated['trade_count'] - unfiltered['trade_count'])}"
            )
        )

    if not lines:
        return ["- R1.2.3: unable to compute paired gated/unfiltered deltas"]
    return lines


def _summarize_fold_deltas(path: Path, name: str) -> list[str]:
    df = _safe_read_csv(path)
    if df is None or df.empty:
        return [f"- {name}: not available"]

    lines = [f"- {name}:"]
    for col in ("delta_sharpe", "delta_expectancy", "delta_max_dd", "delta_trade_count"):
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if values.empty:
            continue
        lines.append(
            f"  - {col}: median={_fmt(values.median())}, pct_gt_0={_fmt((values > 0).mean(), 3)}"
        )
    return lines


def _recent_artifacts(outputs_dir: Path, n: int = 8) -> list[str]:
    allowed = {".csv", ".json", ".md", ".png"}
    files = [
        p
        for p in outputs_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in allowed and ".git" not in p.parts
    ]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    lines: list[str] = []
    for p in files[:n]:
        ts = datetime.fromtimestamp(p.stat().st_mtime, tz=UTC).strftime("%Y-%m-%d %H:%M UTC")
        lines.append(f"- `{p.as_posix()}` ({ts})")
    return lines or ["- No matching artifacts found"]


def build_entry(title: str, notes: list[str], outputs_dir: Path) -> str:
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    commit = _git(["rev-parse", "--short", "HEAD"])
    branch = _git(["branch", "--show-current"])

    lines: list[str] = [
        "",
        f"## Journal Entry — {now} — {title}",
        "",
        f"- Commit: `{commit}`",
        f"- Branch: `{branch}`",
        "",
        "### Notes",
    ]

    if notes:
        lines.extend([f"- {note}" for note in notes])
    else:
        lines.append("- Iteration update recorded.")

    lines.extend(
        [
            "",
            "### Metric Snapshot (auto-generated)",
            "",
            "#### R1.2.3 deltas (gated - unfiltered)",
            *(_summarize_r12(outputs_dir)),
            "",
            "#### R1.3 / R1.3.1 fold deltas",
            *(_summarize_fold_deltas(outputs_dir / "gate_comparison_by_fold.csv", "R1.3")),
            "",
            "#### R1.4 fold deltas",
            *(_summarize_fold_deltas(outputs_dir / "meta_gate_comparison.csv", "R1.4")),
            "",
            "### Recently Updated Artifacts",
            *(_recent_artifacts(outputs_dir)),
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append a journal entry to the consolidated report.")
    parser.add_argument("--title", default="Iteration Update", help="Short title for this journal entry.")
    parser.add_argument(
        "--note",
        action="append",
        default=[],
        help="Add a note bullet. Can be provided multiple times.",
    )
    parser.add_argument(
        "--report-path",
        default="CONSOLIDATED_RESULTS_REPORT.md",
        help="Markdown report path to append into.",
    )
    parser.add_argument(
        "--outputs-dir",
        default="outputs",
        help="Outputs directory used for metric snapshot extraction.",
    )
    parser.add_argument(
        "--allow-duplicate-commit",
        action="store_true",
        help="Allow appending an entry even if the current commit hash already exists in the report.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print entry without writing file.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_path = Path(args.report_path)
    outputs_dir = Path(args.outputs_dir)

    if not report_path.exists():
        raise FileNotFoundError(f"Report not found: {report_path}")

    entry = build_entry(args.title, args.note, outputs_dir)
    commit = _git(["rev-parse", "--short", "HEAD"])
    report_text = report_path.read_text(encoding="utf-8")

    if (not args.allow_duplicate_commit) and f"- Commit: `{commit}`" in report_text:
        print(f"Skipped: commit `{commit}` is already logged in {report_path}.")
        return 0

    if args.dry_run:
        print(entry)
        return 0

    with report_path.open("a", encoding="utf-8") as f:
        f.write(entry)
    print(f"Appended journal entry to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
