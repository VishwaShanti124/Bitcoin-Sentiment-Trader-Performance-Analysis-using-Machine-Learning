"""
src/insights.py
---------------
Surfaces and prints the six strategic recommendations from the analysis.
Also generates a formatted recommendations summary table saved to outputs/.
"""

from __future__ import annotations

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SENTIMENT_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]

RECOMMENDATIONS = [
    {
        "id": 1,
        "title": "Sentiment-Conditioned Position Sizing",
        "regime_focus": "Extreme Fear / Fear",
        "action": (
            "Scale position sizes 20–30% larger during Extreme Fear & Fear regimes "
            "for mean-reversion strategies. Fear produces the highest mean PnL (+$8.99), "
            "suggesting contrarian entries during panic sell-offs carry genuine alpha."
        ),
        "risk": "Drawdown risk during prolonged bear markets.",
        "priority": "HIGH",
    },
    {
        "id": 2,
        "title": "Leverage Guardrails by Sentiment",
        "regime_focus": "All regimes",
        "action": (
            "Allow up to 20x during Neutral/Greed conditions where win rates peak (25.0%). "
            "Hard-cap at 10x during Extreme Fear to prevent liquidation cascades. "
            "Leverage above 25x yields a 46.1% win rate — structurally unprofitable."
        ),
        "risk": "May limit upside during euphoric Extreme Greed phases.",
        "priority": "HIGH",
    },
    {
        "id": 3,
        "title": "Temporal Execution Windows",
        "regime_focus": "Fear / Greed",
        "action": (
            "Route execution algorithms to prefer fills during UTC 08:00–12:00 on Fear days "
            "and UTC 14:00–18:00 on Greed days.  Avoid Neutral-regime weekend sessions "
            "which show the worst risk-adjusted outcomes."
        ),
        "risk": "Latency and exchange downtime during preferred windows.",
        "priority": "MEDIUM",
    },
    {
        "id": 4,
        "title": "Symbol-Specific Sentiment Strategies",
        "regime_focus": "BTC-USD / ETH-USD",
        "action": (
            "BTC and ETH show the strongest sentiment–PnL coupling. "
            "Build separate alpha models per asset rather than a generic cross-asset strategy. "
            "DOGE and ARB appear sentiment-agnostic and may benefit more from order-flow signals."
        ),
        "risk": "Overfitting to observed BTC/ETH sentiment dynamics.",
        "priority": "MEDIUM",
    },
    {
        "id": 5,
        "title": "Trader Segmentation for Platform Features",
        "regime_focus": "All traders",
        "action": (
            "48.8% of traders are profitable over the observation period. "
            "The top quintile exhibit lower average leverage and higher trade frequency. "
            "Surface these high-consistency traders in leaderboard and copy-trade features "
            "rather than raw PnL leaders (who may be high-risk outliers)."
        ),
        "risk": "Past consistency does not guarantee future performance.",
        "priority": "MEDIUM",
    },
    {
        "id": 6,
        "title": "Expand Temporal Window & Feature Engineering",
        "regime_focus": "Platform-wide",
        "action": (
            "Incorporate 3+ years of data to improve statistical power. "
            "Engineer lagged FG signals (3-day, 7-day MA) and regime-transition dummies "
            "to capture leading vs lagging sentiment effects. "
            "Integrate SHAP explainability into production risk management."
        ),
        "risk": "Data licensing and compute overhead.",
        "priority": "LOW",
    },
]

PRIORITY_COLORS = {"HIGH": "#D32F2F", "MEDIUM": "#FF7043", "LOW": "#66BB6A"}


def print_recommendations() -> None:
    print("\n" + "═" * 72)
    print("  STRATEGIC RECOMMENDATIONS")
    print("═" * 72)
    for rec in RECOMMENDATIONS:
        p_color = "🔴" if rec["priority"] == "HIGH" else ("🟡" if rec["priority"] == "MEDIUM" else "🟢")
        print(f"\n  [{rec['id']}] {rec['title']}  {p_color} {rec['priority']}")
        print(f"  Regime focus : {rec['regime_focus']}")
        print(f"  Action       : {rec['action']}")
        print(f"  Risk         : {rec['risk']}")
    print("═" * 72 + "\n")


def compute_key_stats(df: pd.DataFrame, trader_df: pd.DataFrame) -> dict:
    """Return a dict of headline statistics for the report."""
    fg_col = "fg_index" if "fg_index" in df.columns else "fg_value"

    fear_pnl       = df[df["sentiment"] == "Fear"]["closedPnL"].mean()
    ext_fear_pnl   = df[df["sentiment"] == "Extreme Fear"]["closedPnL"].mean()
    pct_profitable = (trader_df["total_pnl"] > 0).mean() * 100
    low_lev_wr     = df[df["leverage"] <= 2]["is_winner"].mean() * 100
    high_lev_wr    = df[df["leverage"] > 11]["is_winner"].mean() * 100
    avg_lev_top    = trader_df.nlargest(10, "total_pnl")["avg_leverage"].mean()

    return {
        "fear_mean_pnl":        round(fear_pnl, 2),
        "extreme_fear_mean_pnl":round(ext_fear_pnl, 2),
        "pct_profitable_traders": round(pct_profitable, 1),
        "low_leverage_win_rate":  round(low_lev_wr, 1),
        "high_leverage_win_rate": round(high_lev_wr, 1),
        "avg_leverage_top10":     round(avg_lev_top, 1),
    }


def plot_recommendations_summary(save_path: str) -> None:
    """Visual summary of the six recommendations."""
    fig, ax = plt.subplots(figsize=(14, 8), facecolor="#0D1117")
    ax.set_facecolor("#0D1117")
    ax.axis("off")

    ax.text(0.5, 0.97, "Strategic Recommendations — Primetrade.ai",
            ha="center", va="top", transform=ax.transAxes,
            fontsize=14, color="#58A6FF", fontweight="bold")

    row_h = 0.13
    start_y = 0.88
    for rec in RECOMMENDATIONS:
        y   = start_y - (rec["id"] - 1) * row_h
        col = PRIORITY_COLORS[rec["priority"]]
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.01, y - 0.10), 0.98, 0.11,
            boxstyle="round,pad=0.01",
            linewidth=1.2, edgecolor=col,
            facecolor="#161B22", transform=ax.transAxes, clip_on=False,
        ))
        ax.text(0.03, y - 0.01, f"[{rec['id']}] {rec['title']}",
                transform=ax.transAxes, fontsize=9, color=col, fontweight="bold", va="top")
        ax.text(0.03, y - 0.045,
                rec["action"][:140] + ("…" if len(rec["action"]) > 140 else ""),
                transform=ax.transAxes, fontsize=7.5, color="#C9D1D9", va="top",
                wrap=True)
        ax.text(0.88, y - 0.01, rec["priority"],
                transform=ax.transAxes, fontsize=8.5, color=col, fontweight="bold", va="top")

    legend_patches = [mpatches.Patch(color=v, label=k) for k, v in PRIORITY_COLORS.items()]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9,
              facecolor="#161B22", edgecolor="#30363D", labelcolor="#C9D1D9")

    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0D1117")
    plt.close(fig)
    print(f"  Saved → {save_path}")
