
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def banner(text: str) -> None:
    width = 68
    print("\n" + "▓" * width)
    print(f"  {text}")
    print("▓" * width)


def main(skip_data: bool = False, no_plots: bool = False) -> None:
    t0 = time.time()

    print("""
╔══════════════════════════════════════════════════════════════════════╗
║   PRIMETRADE.AI — Bitcoin Sentiment × Trader Performance Analysis   ║
║   Data Science Assignment  |  Full Pipeline Runner                  ║
╚══════════════════════════════════════════════════════════════════════╝
""")

    # ── STEP 1: Data Generation ───────────────────────────────────────────────
    if not skip_data:
        banner("STEP 1 — Generating Synthetic Data")
        from data.generate_data import make_fear_greed, make_trades
        import pandas as pd

        print("  Generating Fear & Greed time-series …")
        fg     = make_fear_greed()
        fg_path = os.path.join(DATA_DIR, "fear_greed.csv")
        fg.to_csv(fg_path, index=False)
        print(f"  ✓ Fear & Greed saved ({len(fg):,} rows)")

        print("  Generating trade fills …")
        trades    = make_trades(fg)
        tr_path   = os.path.join(DATA_DIR, "trades.csv")
        trades.to_csv(tr_path, index=False)
        print(f"  ✓ Trades saved ({len(trades):,} rows)")
    else:
        print("  [--skip-data] Using existing CSV files in data/")

    # ── STEP 2: Load & Preprocess ─────────────────────────────────────────────
    banner("STEP 2 — Loading & Preprocessing")
    from src.data_loader import load_raw, preprocess, build_trader_summary, get_sentiment_summary

    trades_raw, fg_raw = load_raw()
    df         = preprocess(trades_raw, fg_raw)
    trader_df  = build_trader_summary(df)
    sent_summ  = get_sentiment_summary(df)

    print(f"  ✓ Preprocessed DataFrame  : {df.shape}")
    print(f"  ✓ Unique traders          : {df['account'].nunique()}")
    print(f"  ✓ Unique symbols          : {df['symbol'].nunique()}")
    print(f"  ✓ Date range              : {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    print("\n  Sentiment × Performance Table:")
    print(sent_summ.to_string(index=False))

    # ── STEP 3: Statistical Hypothesis Testing ────────────────────────────────
    banner("STEP 3 — Statistical Hypothesis Testing")
    from src.hypothesis_tests import run_all_tests
    test_results = run_all_tests(df)

    # ── STEP 4: Machine Learning ──────────────────────────────────────────────
    banner("STEP 4 — Random Forest Feature Attribution")
    from src.ml_model import train_and_evaluate, plot_learning_curve, plot_feature_importance_comparison

    ml_results  = train_and_evaluate(df, n_estimators=200, cv_folds=5)
    importances = ml_results["importances"]

    if not no_plots:
        plot_learning_curve(
            ml_results,
            os.path.join(OUTPUTS_DIR, "learning_curve.png")
        )
        plot_feature_importance_comparison(
            ml_results,
            os.path.join(OUTPUTS_DIR, "feature_importance_comparison.png")
        )

    # ── STEP 5: EDA Dashboards ────────────────────────────────────────────────
    if not no_plots:
        banner("STEP 5 — Generating Analytical Dashboards")
        from src.eda import plot_master_dashboard, plot_deep_dive

        print("  Building Figure 1 — Master 9-panel dashboard …")
        plot_master_dashboard(
            df, trader_df, importances,
            os.path.join(OUTPUTS_DIR, "figure1_master_dashboard.png")
        )

        print("  Building Figure 2 — Deep-dive analytics …")
        plot_deep_dive(
            df, trader_df, fg_raw,
            os.path.join(OUTPUTS_DIR, "figure2_deep_dive.png")
        )
    else:
        print("  [--no-plots] Skipping figure generation.")

    # ── STEP 6: Strategic Insights ────────────────────────────────────────────
    banner("STEP 6 — Strategic Recommendations & Key Stats")
    from src.insights import print_recommendations, compute_key_stats, plot_recommendations_summary

    print_recommendations()

    stats = compute_key_stats(df, trader_df)
    print("  Key Statistics:")
    for k, v in stats.items():
        print(f"    {k:<35} : {v}")

    if not no_plots:
        plot_recommendations_summary(
            os.path.join(OUTPUTS_DIR, "recommendations_summary.png")
        )

    # ── STEP 7: Save outputs to CSV ───────────────────────────────────────────
    banner("STEP 7 — Saving Output Tables")
    sent_summ.to_csv(os.path.join(OUTPUTS_DIR, "sentiment_performance_table.csv"), index=False)
    trader_df.to_csv(os.path.join(OUTPUTS_DIR, "trader_summary.csv"), index=False)
    print("  ✓ sentiment_performance_table.csv")
    print("  ✓ trader_summary.csv")

    # ── Done ──────────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'═'*68}")
    print(f"  ✅ Pipeline complete in {elapsed:.1f}s")
    print(f"  All outputs saved to: {OUTPUTS_DIR}/")
    print(f"{'═'*68}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Primetrade.ai Analysis Pipeline")
    parser.add_argument("--skip-data", action="store_true",
                        help="Skip data generation; use existing CSVs in data/")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip figure generation (faster CI runs)")
    args = parser.parse_args()
    main(skip_data=args.skip_data, no_plots=args.no_plots)
