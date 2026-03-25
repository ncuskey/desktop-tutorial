# Forex Strategy Research Lab

Production-oriented quantitative research framework for evaluating FX strategies, combinations, and orchestration layers under realistic assumptions.

## What this repo contains

- **Data & features**: `data/`, `regime/`
- **Strategies**: `strategies/`
- **Combinations & orchestration**: `combinations/`, `orchestrators/`
- **Execution & risk controls**: `execution/`
- **Metrics & validation**: `metrics/`, `research/`
- **Meta-labeling decision layer**: `metalabel/`
- **Runnable experiment entrypoint**: `prototype.py`

## Core capabilities

- Multi-symbol data support (`EURUSD`, `GBPUSD`, `USDJPY`, `AUDUSD`)
- Multi-timeframe support (`H1`, `H4`, `D1`)
- Cost-aware simulation (spread, slippage, commissions)
- Regime detection + stable regime state machine
- Specialist sleeves + regime-aware orchestration
- Strict walk-forward validation
- Bootstrap robustness analysis
- Meta-label trade filtering
- Experiment tracking to SQLite

## Quick start

```bash
python3 -m pip install -r requirements.txt
python3 prototype.py
```

Artifacts are written to `outputs/`.

## Research journal workflow

Keep `CONSOLIDATED_RESULTS_REPORT.md` as the living research journal.

After each iteration/run, append an entry:

```bash
python3 run_research_journal.py \
  --title "R1.x Iteration" \
  --note "What changed" \
  --note "Key result or decision"
```

Optional:

- `--dry-run` to preview the entry without writing.
- `--allow-duplicate-commit` if you need multiple entries for the same commit.

## Notes

- Current default run uses generated sample OHLCV data (`data/sample_ohlcv.csv`).
- To increase confidence, integrate external historical datasets and rerun strict walk-forward experiments across symbols/timeframes/providers.
