from __future__ import annotations

from typing import Any

import pandas as pd


def build_feature_pruning_tables(
    ablation_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert ablation output into per-context pruning impacts and global recommendations.
    """
    if ablation_df.empty:
        empty_local = pd.DataFrame(
            columns=[
                "symbol",
                "sleeve",
                "feature_group",
                "expectancy_impact",
                "sharpe_impact",
                "maxdd_impact",
                "recommendation",
            ]
        )
        empty_global = pd.DataFrame(
            columns=[
                "feature_group",
                "contexts",
                "expectancy_impact_mean",
                "sharpe_impact_mean",
                "maxdd_impact_mean",
                "pct_removal_hurts_expectancy",
                "pct_removal_improves_expectancy",
                "recommendation",
            ]
        )
        return empty_local, empty_global

    df = ablation_df.copy()
    if "sleeve" not in df.columns:
        df["sleeve"] = "UNKNOWN"

    local_rows: list[dict[str, Any]] = []
    for (symbol, sleeve), g in df.groupby(["symbol", "sleeve"], dropna=False):
        baseline = g[g["variant"] == "baseline"]
        if baseline.empty:
            continue
        b = baseline.iloc[0]
        base_exp = float(b.get("expectancy_filtered", 0.0))
        base_sharpe = float(b.get("sharpe_filtered", 0.0))
        base_dd = float(b.get("maxdd_filtered", 0.0))

        for _, row in g.iterrows():
            variant = str(row.get("variant", ""))
            if not variant.startswith("without_"):
                continue
            feature_group = variant.replace("without_", "")
            exp_imp = float(row.get("expectancy_filtered", 0.0)) - base_exp
            sharpe_imp = float(row.get("sharpe_filtered", 0.0)) - base_sharpe
            dd_imp = float(row.get("maxdd_filtered", 0.0)) - base_dd
            # positive dd_imp => less negative drawdown => improvement
            if exp_imp <= -2e-5:
                rec = "KEEP"
            elif exp_imp >= 2e-5 and sharpe_imp >= 0.0:
                rec = "DROP"
            else:
                rec = "CONDITIONAL"
            local_rows.append(
                {
                    "symbol": symbol,
                    "sleeve": sleeve,
                    "feature_group": feature_group,
                    "expectancy_impact": exp_imp,
                    "sharpe_impact": sharpe_imp,
                    "maxdd_impact": dd_imp,
                    "recommendation": rec,
                }
            )

    local_df = pd.DataFrame(local_rows)
    if local_df.empty:
        global_df = pd.DataFrame(
            columns=[
                "feature_group",
                "contexts",
                "expectancy_impact_mean",
                "sharpe_impact_mean",
                "maxdd_impact_mean",
                "pct_removal_hurts_expectancy",
                "pct_removal_improves_expectancy",
                "recommendation",
            ]
        )
        return local_df, global_df

    grouped = local_df.groupby("feature_group")
    global_rows: list[dict[str, Any]] = []
    for feature_group, g in grouped:
        exp_mean = float(g["expectancy_impact"].mean())
        sharpe_mean = float(g["sharpe_impact"].mean())
        dd_mean = float(g["maxdd_impact"].mean())
        hurts = float((g["expectancy_impact"] < -2e-5).mean())
        helps = float((g["expectancy_impact"] > 2e-5).mean())

        if hurts >= 0.6 and exp_mean < 0.0:
            rec = "KEEP"
        elif helps >= 0.6 and exp_mean >= 0.0:
            rec = "DROP"
        else:
            rec = "CONDITIONAL"

        global_rows.append(
            {
                "feature_group": feature_group,
                "contexts": int(len(g)),
                "expectancy_impact_mean": exp_mean,
                "sharpe_impact_mean": sharpe_mean,
                "maxdd_impact_mean": dd_mean,
                "pct_removal_hurts_expectancy": hurts,
                "pct_removal_improves_expectancy": helps,
                "recommendation": rec,
            }
        )

    global_df = pd.DataFrame(global_rows).sort_values(
        ["recommendation", "expectancy_impact_mean"],
        ascending=[True, False],
    )
    return local_df.sort_values(["symbol", "sleeve", "feature_group"]).reset_index(drop=True), global_df.reset_index(drop=True)
