from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from research.v21_runner import run_v21_refinement


def _safe_head(df: pd.DataFrame, n: int = 5) -> str:
    if df.empty:
        return "(empty)"
    return df.head(n).to_string(index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Forex Strategy Research Lab V2.1 refinement pass.")
    parser.add_argument("--data-sources", default="configs/data_sources.yaml")
    parser.add_argument("--symbols", default="configs/symbols.yaml")
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--longer-start", default=None, help="Optional UTC start timestamp for candidate validation window.")
    parser.add_argument("--longer-end", default=None, help="Optional UTC end timestamp for candidate validation window.")
    parser.add_argument("--use-purged-candidates", action="store_true", help="Enable purged candidate validation.")
    args = parser.parse_args()

    results = run_v21_refinement(
        data_sources_config=args.data_sources,
        symbols_config=args.symbols,
        output_dir=args.outputs,
        longer_start=args.longer_start,
        longer_end=args.longer_end,
        use_purged_candidates=args.use_purged_candidates,
    )

    out = Path(args.outputs)
    print("\nV2.1 outputs generated:")
    print(f"- {out / 'v21_filter_rate_diagnostics.csv'}")
    print(f"- {out / 'v21_sleeve_symbol_ranking.csv'}")
    print(f"- {out / 'v21_component_decisions.csv'}")
    print(f"- {out / 'v21_feature_pruning.csv'}")
    print(f"- {out / 'v21_feature_group_summary.csv'}")
    print(f"- {out / 'v21_candidate_validation.csv'}")
    print(f"- {out / 'v21_candidate_equity.png'}")

    ranking = results["v21_sleeve_symbol_ranking"]
    decisions = results["v21_component_decisions"]
    feature_summary = results["v21_feature_group_summary"]
    filter_diag = results["v21_filter_rate_diagnostics"]
    candidate = results["v21_candidate_validation"]

    top_promote = decisions[decisions["decision"] == "PROMOTE"].head(5)
    prune_groups = feature_summary[feature_summary["recommendation"] == "DROP"].head(5)
    filter_abs_err = pd.to_numeric(filter_diag.get("filter_rate_error", pd.Series(dtype=float)), errors="coerce").abs()
    avg_abs_err = float(filter_abs_err.mean()) if not filter_abs_err.empty else 0.0
    clipped_pct = float(pd.to_numeric(filter_diag.get("threshold_clipped", pd.Series(dtype=float)), errors="coerce").fillna(0).mean()) if not filter_diag.empty else 0.0

    print("\nConcise V2.1 summary:")
    print(f"- Top promoted components: {len(top_promote)} shown below")
    print(_safe_head(top_promote))
    print(f"- Feature groups to prune (top): {len(prune_groups)} shown below")
    print(_safe_head(prune_groups))
    print(f"- Filter-rate enforcement quality: avg |error|={avg_abs_err:.4f}, threshold-clipped-pct={clipped_pct:.2%}")
    print(f"- Candidate combinations selected: {len(candidate)}")
    if not candidate.empty:
        print(candidate[["symbol", "sleeve", "candidate_rank", "filtered_expectancy", "expectancy_delta"]].to_string(index=False))
    print("\nTop ranking preview:")
    print(_safe_head(ranking))


if __name__ == "__main__":
    main()
