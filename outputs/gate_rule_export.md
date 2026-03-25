# Gate Rule Export (R1.3)

- Strategy: `TrendBreakout_V2`
- Gate mode: `score_threshold`
- Deterministic decision: `ALLOW if model_score >= train_fold_score_quantile_0.70`

## Threshold Stability
- Quantile target: `0.7`
- Threshold mean: `0.7748443548604598`
- Threshold std: `0.10085393242640486`
- Threshold min/max: `0.5728209071105466 / 0.9630682809942537`

## Coefficient Stability (median coefficient, sign consistency)

| feature | median_coef | sign_consistency |
|---|---:|---:|
| atr_percentile | 0.272732 | 0.689 |
| tsmom_48 | 0.244644 | 0.733 |
| vr_6 | 0.187540 | 0.600 |
| vr_12 | 0.177226 | 0.600 |
| ema_fast_12 | 0.170663 | 0.644 |
| ema_slow_48 | 0.157948 | 0.644 |
| tsmom_24 | 0.143989 | 0.667 |
| breakout_strength_atr_mult | -0.139709 | 0.578 |
| adx_14 | -0.085694 | 0.533 |
| ma_slope | 0.084432 | 0.689 |
| price_acceleration | 0.080567 | 0.556 |
| tsmom_avg | 0.074882 | 0.667 |
| adx_slope | -0.074643 | 0.622 |
| ma_spread | 0.072269 | 0.622 |
| range_ratio | 0.041902 | 0.511 |
| vr_24 | 0.041255 | 0.578 |
| atr_ratio | 0.029330 | 0.533 |
| distance_from_range_high | -0.024417 | 0.533 |
| atr_expansion_recent | 0.021722 | 0.556 |
| trend_variance_ratio | 0.011454 | 0.511 |
| tsmom_72 | -0.009940 | 0.533 |
| breakout_velocity | -0.007140 | 0.511 |

## Tree Rule Snapshots

### Fold Tree 1
```
|--- class: 1

```
### Fold Tree 2
```
|--- class: 1

```
### Fold Tree 3
```
|--- class: 1

```
### Fold Tree 4
```
|--- class: 1

```
### Fold Tree 5
```
|--- class: 1

```
