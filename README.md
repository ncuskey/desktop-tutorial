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
- **Real-data multi-symbol runner**: `run_real_data_evaluation.py`

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

### Real historical multi-symbol evaluation

1. Configure data sources in:
   - `configs/data_sources.yaml`
   - `configs/symbols.yaml`
2. Point each symbol to a real CSV path and provider column map.
3. Run:

```bash
python3 run_real_data_evaluation.py
```

If you want a local demo run and configured real CSVs are missing:

```bash
python3 run_real_data_evaluation.py --create-demo-if-missing
```

### Pull live FX snapshots from API (apilayer)

You can pull live quotes and append canonical rows to `data/real/*.csv`:

```bash
export APILAYER_ACCESS_KEY="YOUR_KEY"
python3 fetch_apilayer_live.py --symbols EURUSD,GBPUSD,USDJPY,AUDUSD
```

Then run the strict WF evaluator:

```bash
python3 run_real_data_evaluation.py
```

Or fetch + evaluate in one command:

```bash
python3 run_real_data_evaluation.py --pull-apilayer-live
```

Note: the apilayer `/live` endpoint returns point-in-time quotes (not full historical OHLC bars).  
For robust backtests you still need accumulated snapshots or provider historical endpoints.

### Pull Dukascopy historical data (duka-dl)

Install downloader dependency:

```bash
python3 -m pip install duka-dl
```

Download history and build canonical H1 CSV files for configured symbols:

```bash
python3 fetch_dukascopy_history.py --start 01-01-2024 --end 30-06-2024
```

Or do it in one step and immediately run strict WF evaluation:

```bash
python3 run_real_data_evaluation.py --pull-duka-history --duka-start 01-01-2024 --duka-end 30-06-2024
```

## Notes

- Current default run uses generated sample OHLCV data (`data/sample_ohlcv.csv`).
- To increase confidence, integrate external historical datasets and rerun strict walk-forward experiments across symbols/timeframes/providers.
