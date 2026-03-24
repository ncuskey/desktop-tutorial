# Strategy Specification: TrendBreakout_V2

- Version: `TrendBreakout_V2_R1_FINAL`
- Symbols: `EURUSD`
- Timeframe: `H1`
- Data Range: `2020-01-05T00:00:00+00:00` to `2021-02-12T03:00:00+00:00`
- Timestamp (UTC): `2026-03-24T21:47:04.577042+00:00`
- Commit: `4f46f6a`

## Overview
| Field | Value |
|---|---|
| strategy | TrendBreakout_V2 |
| symbols | ["EURUSD"] |
| timeframe | H1 |
| data_range | {"start": "2020-01-05T00:00:00+00:00", "end": "2021-02-12T03:00:00+00:00", "bar_count": 9904} |
| search_method | random |
| candidate_count | 40 |
| hardened_candidate_id | 22 |

## Core Thesis
HARDENED_DEFAULT for TrendBreakout_V2 on EURUSD (H1) targets a reproducible positive OOS expectancy of 0.000765 with OOS Sharpe 0.973 and max drawdown -0.006106. Fold-level positive expectancy appears in 40.0% of folds, with robustness score -7.520.

## Mechanics
### Entry Logic
| Field | Value |
|---|---|
| lookback | 30 |
| velocity_lookback | 4 |
| velocity_threshold | 0.8 |
| confirmation_bars | 2 |
| vol_compression_max_pct | 0.4 |
| breakout_strength_atr_mult | 0.3 |
| retest_entry_mode | False |
| expansion_lookback | 12 |
| expansion_threshold | 1.02 |

### Exit Logic
| Field | Value |
|---|---|
| trailing_stop_atr_mult | 2.2 |
| max_holding_bars | 72 |
| vol_exit_pct_rank_threshold | 0.15 |
| partial_take_profit_rr | 1.6 |
| partial_take_profit_size | 0.5 |
| winner_extension_enabled | False |
| extension_trigger_atr_multiple | 1.8 |
| extension_stop_multiplier | 2.4 |
| extension_max_holding_bars | 160 |
| vol_contraction_exit_mult | 0.8 |
| vol_contraction_window | 20 |

### Trade Management
| Field | Value |
|---|---|
| min_bars_between_trades | 20 |
| dynamic_cooldown_by_vol | True |
| high_vol_cooldown_mult | 2.0 |

## Parameters
| Field | Value |
|---|---|
| lookback | 30 |
| velocity_lookback | 4 |
| velocity_threshold | 0.8 |
| confirmation_bars | 2 |
| vol_compression_max_pct | 0.4 |
| breakout_strength_atr_mult | 0.3 |
| retest_entry_mode | False |
| trailing_stop_atr_mult | 2.2 |
| max_holding_bars | 72 |
| vol_exit_pct_rank_threshold | 0.15 |
| partial_take_profit_rr | 1.6 |
| partial_take_profit_size | 0.5 |
| winner_extension_enabled | False |
| extension_trigger_atr_multiple | 1.8 |
| extension_stop_multiplier | 2.4 |
| extension_max_holding_bars | 160 |
| min_bars_between_trades | 20 |
| dynamic_cooldown_by_vol | True |
| high_vol_cooldown_mult | 2.0 |
| expansion_lookback | 12 |
| expansion_threshold | 1.02 |
| vol_contraction_exit_mult | 0.8 |
| vol_contraction_window | 20 |

## Performance
| Field | Value |
|---|---|
| oos_expectancy | 0.0007654015473434 |
| oos_sharpe | 0.973467941167683 |
| oos_max_drawdown | -0.0061061822869274 |
| best_peak_expectancy | 0.0007876792244898 |
| best_robust_score | 0.748460267574955 |
| best_hardened_expectancy | 0.0007654015473434 |
| best_hardened_sharpe | 0.973467941167683 |
| best_hardened_max_drawdown | -0.0061061822869274 |
| candidate_count | 40 |
| search_method | random |
| fold_count | 15 |
| positive_expectancy_pct | 0.4 |
| positive_sharpe_pct | 0.3333333333333333 |
| drawdown_improved_folds_pct | 1.0 |
| expectancy_std | 0.001596650076888874 |
| sharpe_std | 3.0965646970478597 |
| zero_trade_fold_pct | 0.4666666666666667 |
| avg_trade_count_per_fold | 0.8 |
| worst_folds | [{"test_start": "2020-09-19 08:00:00+00:00", "test_end": "2020-10-10 03:00:00+00:00", "expectancy": -0.0006282508609451, "sharpe": -6.140507222076877, "max_drawdown": -0.0006925658512701}, {"test_start": "2020-10-10 04:00:00+00:00", "test_end": "2020-10-30 23:00:00+00:00", "expectancy": -0.0006146068822858, "sharpe": -6.798192841867094, "max_drawdown": -0.0039086506403219}, {"test_start": "2020-05-17 08:00:00+00:00", "test_end": "2020-06-07 03:00:00+00:00", "expectancy": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}] |

## Edge Profile
| Field | Value |
|---|---|
| expectancy_per_trade | 0.0007654015473434 |
| positive_fold_expectancy_pct | 0.4 |
| positive_fold_sharpe_pct | 0.3333333333333333 |
| expectancy_std_by_fold | 0.001596650076888874 |
| sharpe_std_by_fold | 3.0965646970478597 |
| avg_trades_per_fold | 0.8 |

## Robustness
| Field | Value |
|---|---|
| robustness_score | -7.520424790522311 |
| robustness_rank | 38 |
| expectancy_rank | 2 |
| parameter_isolation_penalty | 10.226080711116673 |
| plateau_bonus | -5.610579935115474 |
| neighbor_count | 5 |
| top_parameter_sensitivity | [{"parameter": "confirmation_bars", "abs_spearman_corr": 0.7256699900541159, "spearman_corr_with_expectancy": -0.7256699900541159, "grouped_expectancy_spread": 0.0006705389757215}, {"parameter": "vol_exit_pct_rank_threshold", "abs_spearman_corr": 0.3908412121957444, "spearman_corr_with_expectancy": -0.3908412121957444, "grouped_expectancy_spread": 0.0005131746191863}, {"parameter": "extension_max_holding_bars", "abs_spearman_corr": 0.3103661468591873, "spearman_corr_with_expectancy": 0.3103661468591873, "grouped_expectancy_spread": 0.0002786569000586}, {"parameter": "winner_extension_enabled", "abs_spearman_corr": 0.2003176739905326, "spearman_corr_with_expectancy": 0.2003176739905326, "grouped_expectancy_spread": 0.0001896706006906}, {"parameter": "vol_compression_max_pct", "abs_spearman_corr": 0.198905830723845, "spearman_corr_with_expectancy": 0.198905830723845, "grouped_expectancy_spread": 0.0002185788794681}] |
| false_peak_count | 5 |
| false_peak_examples | [{"candidate_id": 0, "oos_expectancy": 0.0007876792244898, "robustness_score": -8.090244745981678}, {"candidate_id": 22, "oos_expectancy": 0.0007654015473434, "robustness_score": -7.520424790522311}, {"candidate_id": 21, "oos_expectancy": 0.0003716573849789, "robustness_score": -5.418875832043159}] |

## Risks
| Field | Value |
|---|---|
| zero_trade_fold_pct | 0.4666666666666667 |
| worst_folds | [{"test_start": "2020-09-19 08:00:00+00:00", "test_end": "2020-10-10 03:00:00+00:00", "expectancy": -0.0006282508609451, "sharpe": -6.140507222076877, "max_drawdown": -0.0006925658512701}, {"test_start": "2020-10-10 04:00:00+00:00", "test_end": "2020-10-30 23:00:00+00:00", "expectancy": -0.0006146068822858, "sharpe": -6.798192841867094, "max_drawdown": -0.0039086506403219}, {"test_start": "2020-05-17 08:00:00+00:00", "test_end": "2020-06-07 03:00:00+00:00", "expectancy": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}] |
| drawdown_risk_note | Fold-level drawdown clusters are visible in the worst-fold sample. Validate stress periods before promotion. |

## Component Insights
| Field | Value |
|---|---|
| positive_components | [{"component_test": "partial_tp_off", "delta_expectancy": 0.0005027178586762, "delta_sharpe": 0.5085253209958265, "delta_max_drawdown": -2.220446049250313e-16, "delta_robust_score": 0.6450327517004584}, {"component_test": "winner_extension_on", "delta_expectancy": 0.0001404609029742, "delta_sharpe": 0.2513318865775635, "delta_max_drawdown": -4.440892098500626e-16, "delta_robust_score": 0.2047958880116576}, {"component_test": "contraction_exit_loose", "delta_expectancy": 0.0001217439357895, "delta_sharpe": -0.113069999562671, "delta_max_drawdown": -0.0022132352490444, "delta_robust_score": 0.050755847892062}] |
| negative_components | [{"component_test": "contraction_exit_tight", "delta_expectancy": -0.0003756949868449, "delta_sharpe": -0.706251789565223, "delta_max_drawdown": -0.0012781183738883, "delta_robust_score": -0.6148327885909766}, {"component_test": "retest_entry_on", "delta_expectancy": 0.0, "delta_sharpe": 0.0, "delta_max_drawdown": 0.0, "delta_robust_score": 0.0}, {"component_test": "retest_entry_off", "delta_expectancy": 0.0, "delta_sharpe": 0.0, "delta_max_drawdown": 0.0, "delta_robust_score": 0.0}] |
| neutral_components | [{"component_test": "retest_entry_on", "delta_expectancy": 0.0, "delta_sharpe": 0.0, "delta_max_drawdown": 0.0, "delta_robust_score": 0.0}, {"component_test": "retest_entry_off", "delta_expectancy": 0.0, "delta_sharpe": 0.0, "delta_max_drawdown": 0.0, "delta_robust_score": 0.0}, {"component_test": "winner_extension_off", "delta_expectancy": 0.0, "delta_sharpe": 0.0, "delta_max_drawdown": 0.0, "delta_robust_score": 0.0}, {"component_test": "partial_tp_on", "delta_expectancy": 0.0, "delta_sharpe": 0.0, "delta_max_drawdown": 0.0, "delta_robust_score": 0.0}, {"component_test": "dynamic_cooldown_off", "delta_expectancy": 0.0, "delta_sharpe": 0.0, "delta_max_drawdown": 0.0, "delta_robust_score": 0.0}] |

## Cross-Symbol Validation
### Per-Symbol Results
| Symbol | Expectancy | Sharpe | MaxDD | Classification |
|---|---:|---:|---:|---|
| EURUSD | 0.0007654015473434 | 0.973467941167683 | -0.0061061822869274 | REJECT |
| GBPUSD | 0.0017007476773515 | 1.1994828050640107 | -0.0021569261154581 | REJECT |
| AUDUSD | 0.0009801395875796 | 0.5409864403620183 | -0.0037450649585671 | REJECT |

### Overall Classification
`REJECT` (PROMOTE=0, CONDITIONAL=0, REJECT=3)

### Parameter Alignment Summary
| Field | Value |
|---|---|
| parameter_count | 23 |
| mean_alignment_score | 0.7955204216073782 |
| high_alignment_pct | 0.5217391304347826 |
| low_alignment_params | ["confirmation_bars", "dynamic_cooldown_by_vol", "extension_stop_multiplier", "extension_trigger_atr_multiple", "high_vol_cooldown_mult", "min_bars_between_trades", "partial_take_profit_rr", "partial_take_profit_size", "trailing_stop_atr_mult", "velocity_lookback", "winner_extension_enabled"] |

## Promotion Status
**Status:** `HOLD`

- Positive stitched OOS expectancy.
- Positive stitched OOS Sharpe.
- Less than half of folds show positive expectancy.
- Contained OOS max drawdown under 1%.
- Robustness score is not positive.

## Next Steps
- Increase fold count or extend history to improve confidence in fold-level expectancy stability.
- Review entry strictness to reduce no-trade folds while preserving expectancy.
- Densify local parameter search around robust candidates to avoid isolated false peaks.
- Stress-test and potentially disable 'contraction_exit_tight' given negative ablation impact.
