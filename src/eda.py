"""
src/eda.py
----------
Produces the 9-panel master analytical dashboard (Figure 1) and the
6-panel deep-dive analytics dashboard (Figure 2) described in the report.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.patches import Patch
from typing import Optional

warnings.filterwarnings("ignore")

# ── Style constants ──────────────────────────────────────────────────────────
PALETTE = {
    "Extreme Fear":  "#D32F2F",
    "Fear":          "#FF7043",
    "Neutral":       "#FDD835",
    "Greed":         "#66BB6A",
    "Extreme Greed": "#1B5E20",
}
SENTIMENT_ORDER = list(PALETTE.keys())
COLORS          = [PALETTE[s] for s in SENTIMENT_ORDER]

plt.rcParams.update({
    "figure.facecolor": "#0D1117",
    "axes.facecolor":   "#161B22",
    "axes.edgecolor":   "#30363D",
    "axes.labelcolor":  "#C9D1D9",
    "xtick.color":      "#8B949E",
    "ytick.color":      "#8B949E",
    "text.color":       "#C9D1D9",
    "grid.color":       "#21262D",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "font.family":      "DejaVu Sans",
})


# ── Helper ───────────────────────────────────────────────────────────────────
def _save(fig, path: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {path}")


# ── Panel builders ───────────────────────────────────────────────────────────

def _panel_sentiment_pie(ax, df):
    counts = df["sentiment"].value_counts().reindex(SENTIMENT_ORDER)
    wedge_colors = COLORS
    ax.pie(
        counts,
        labels=counts.index,
        colors=wedge_colors,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 7, "color": "#C9D1D9"},
        wedgeprops={"linewidth": 0.5, "edgecolor": "#0D1117"},
    )
    ax.set_title("Sentiment Distribution", fontsize=9, color="#58A6FF", pad=6)


def _panel_mean_pnl_bar(ax, df):
    summary = df.groupby("sentiment", observed=True)["closedPnL"].mean().reindex(SENTIMENT_ORDER)
    bars = ax.bar(SENTIMENT_ORDER, summary.values, color=COLORS, edgecolor="#0D1117", linewidth=0.6)
    for bar, val in zip(bars, summary.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (3 if val >= 0 else -6),
            f"${val:.2f}",
            ha="center", va="bottom", fontsize=6.5, color="#C9D1D9",
        )
    ax.axhline(0, color="#8B949E", linewidth=0.8, linestyle="--")
    ax.set_title("Mean Closed PnL by Sentiment", fontsize=9, color="#58A6FF", pad=6)
    ax.set_ylabel("Mean PnL (USD)", fontsize=7)
    ax.set_xticklabels(SENTIMENT_ORDER, rotation=20, ha="right", fontsize=6.5)


def _panel_win_rate_bar(ax, df):
    wr = df.groupby("sentiment", observed=True)["is_winner"].mean().reindex(SENTIMENT_ORDER)
    bars = ax.bar(SENTIMENT_ORDER, wr.values * 100, color=COLORS, edgecolor="#0D1117", linewidth=0.6)
    ax.axhline(50, color="#FDD835", linewidth=0.8, linestyle="--", label="50% baseline")
    ax.set_title("Win Rate by Sentiment", fontsize=9, color="#58A6FF", pad=6)
    ax.set_ylabel("Win Rate (%)", fontsize=7)
    ax.set_xticklabels(SENTIMENT_ORDER, rotation=20, ha="right", fontsize=6.5)
    ax.legend(fontsize=6, loc="upper right")


def _panel_violin(ax, df):
    present = [s for s in SENTIMENT_ORDER if (df["sentiment"] == s).sum() > 5]
    data_by_sentiment = [
        df[df["sentiment"] == s]["closedPnL"].clip(-500, 500).values
        for s in present
    ]
    colors_present = [PALETTE[s] for s in present]
    parts = ax.violinplot(data_by_sentiment, positions=range(len(present)),
                          showmedians=True, showextrema=False)
    for pc, color in zip(parts["bodies"], colors_present):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("#FFFFFF")
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(present, rotation=20, ha="right", fontsize=6.5)
    ax.set_title("PnL Distribution Across Sentiment Regimes", fontsize=9, color="#58A6FF", pad=6)
    ax.set_ylabel("Closed PnL (USD)", fontsize=7)


def _panel_leverage_bar(ax, df):
    lev = df.groupby("sentiment", observed=True)["leverage"].mean().reindex(SENTIMENT_ORDER)
    ax.bar(SENTIMENT_ORDER, lev.values, color=COLORS, edgecolor="#0D1117", linewidth=0.6)
    ax.set_title("Avg Leverage by Sentiment", fontsize=9, color="#58A6FF", pad=6)
    ax.set_ylabel("Average Leverage", fontsize=7)
    ax.set_xticklabels(SENTIMENT_ORDER, rotation=20, ha="right", fontsize=6.5)


def _panel_scatter_volume(ax, df):
    sample = df.sample(min(3000, len(df)), random_state=42)
    for s, color in PALETTE.items():
        sub = sample[sample["sentiment"] == s]
        ax.scatter(sub["timestamp"], sub["size"],
                   c=color, alpha=0.35, s=6, linewidths=0, label=s)
    ax.set_title("Daily Trade Volume Coloured by Sentiment", fontsize=9, color="#58A6FF", pad=6)
    ax.set_xlabel("Date", fontsize=7)
    ax.set_ylabel("Trade Size", fontsize=7)
    legend_elements = [Patch(facecolor=PALETTE[s], label=s) for s in SENTIMENT_ORDER]
    ax.legend(handles=legend_elements, fontsize=5.5, loc="upper left", ncol=3)


def _panel_hour_heatmap(ax, df):
    pivot = (
        df.groupby(["hour", "sentiment"], observed=True)["closedPnL"]
        .mean()
        .unstack("sentiment")
        .reindex(columns=SENTIMENT_ORDER)
        .reindex(range(24))
    )
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="RdYlGn",
        center=0,
        linewidths=0.3,
        linecolor="#0D1117",
        cbar_kws={"shrink": 0.6, "label": "Mean PnL"},
        annot=False,
    )
    ax.set_title("Hour × Sentiment Mean PnL", fontsize=9, color="#58A6FF", pad=6)
    ax.set_xlabel("Sentiment", fontsize=7)
    ax.set_ylabel("Hour (UTC)", fontsize=7)


def _panel_feature_importance(ax, importances: dict):
    sorted_items = sorted(importances.items(), key=lambda x: x[1])
    names, vals = zip(*sorted_items)
    bar_colors = ["#FF7043", "#66BB6A", "#FF00FF", "#00BFFF", "#FDD835"]
    ax.barh(names, vals, color=bar_colors[:len(names)], edgecolor="#0D1117", linewidth=0.5)
    for i, v in enumerate(vals):
        ax.text(v + 0.002, i, f"{v:.1%}", va="center", fontsize=7, color="#C9D1D9")
    ax.set_title("Feature Importance (Random Forest)", fontsize=9, color="#58A6FF", pad=6)
    ax.set_xlabel("Importance", fontsize=7)


def _panel_trader_pnl(ax, trader_df):
    trader_df = trader_df.sort_values("total_pnl")
    colors = ["#66BB6A" if v > 0 else "#D32F2F" for v in trader_df["total_pnl"]]
    ax.bar(range(len(trader_df)), trader_df["total_pnl"].values, color=colors,
           edgecolor="#0D1117", linewidth=0.3, width=0.8)
    ax.axhline(0, color="#8B949E", linewidth=0.8)
    ax.set_title("Trader Total PnL Distribution", fontsize=9, color="#58A6FF", pad=6)
    ax.set_xlabel("Traders (ranked)", fontsize=7)
    ax.set_ylabel("Total PnL (USD)", fontsize=7)


# ── Master 9-panel dashboard ─────────────────────────────────────────────────

def plot_master_dashboard(df: pd.DataFrame, trader_df: pd.DataFrame,
                          importances: dict, save_path: str) -> None:
    """Figure 1 — 9-panel master analytical dashboard."""
    fig = plt.figure(figsize=(20, 18), facecolor="#0D1117")
    fig.suptitle(
        "Bitcoin Market Sentiment × Hyperliquid Trader Performance\n"
        "Comprehensive EDA | Statistical Testing | ML Feature Attribution",
        fontsize=13, color="#58A6FF", y=0.98, fontweight="bold",
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.35)

    axes = [fig.add_subplot(gs[r, c]) for r in range(3) for c in range(3)]

    _panel_sentiment_pie(axes[0], df)
    _panel_mean_pnl_bar(axes[1], df)
    _panel_win_rate_bar(axes[2], df)
    _panel_violin(axes[3], df)
    _panel_leverage_bar(axes[4], df)
    _panel_scatter_volume(axes[5], df)
    _panel_hour_heatmap(axes[6], df)
    _panel_feature_importance(axes[7], importances)
    _panel_trader_pnl(axes[8], trader_df)

    _save(fig, save_path)


# ── Deep-dive 6-panel dashboard ───────────────────────────────────────────────

def plot_deep_dive(df: pd.DataFrame, trader_df: pd.DataFrame,
                   fg_df: pd.DataFrame, save_path: str) -> None:
    """Figure 2 — deep-dive analytics."""
    fig = plt.figure(figsize=(20, 14), facecolor="#0D1117")
    fig.suptitle(
        "Deep-Dive Analytics: Trader Behaviour & Sentiment Dynamics",
        fontsize=13, color="#58A6FF", y=0.98, fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.35)

    # 1. Mean PnL by symbol and sentiment
    ax1 = fig.add_subplot(gs[0, 0])
    sym_sent = (df.groupby(["symbol", "sentiment"], observed=True)["closedPnL"]
                .mean().unstack("sentiment").reindex(columns=SENTIMENT_ORDER))
    sym_sent.plot(kind="bar", ax=ax1, color=COLORS, edgecolor="#0D1117", linewidth=0.4, legend=False)
    ax1.set_title("Mean PnL by Symbol & Sentiment", fontsize=9, color="#58A6FF", pad=6)
    ax1.set_xlabel("Symbol", fontsize=7)
    ax1.set_ylabel("Mean PnL (USD)", fontsize=7)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=20, ha="right", fontsize=7)

    # 2. Top-10 trader cumulative PnL equity curves
    ax2 = fig.add_subplot(gs[0, 1])
    top10 = trader_df.nlargest(10, "total_pnl")["account"]
    cmap  = plt.cm.tab10
    for idx, acc in enumerate(top10):
        sub  = df[df["account"] == acc].sort_values("timestamp")
        cumsum = sub["closedPnL"].cumsum().values
        ax2.plot(cumsum, linewidth=1.1, color=cmap(idx / 10), alpha=0.85)
    ax2.set_title("Cumulative PnL – Top 10 Traders", fontsize=9, color="#58A6FF", pad=6)
    ax2.set_xlabel("Trade #", fontsize=7)
    ax2.set_ylabel("Cumulative PnL (USD)", fontsize=7)

    # 3. Win rate by leverage bucket
    ax3 = fig.add_subplot(gs[0, 2])
    lev_wr = df.groupby("leverage_bucket", observed=True)["is_winner"].mean() * 100
    lev_wr.plot(kind="bar", ax=ax3, color="#58A6FF", edgecolor="#0D1117", linewidth=0.5)
    ax3.axhline(50, color="#FDD835", linestyle="--", linewidth=0.8)
    ax3.set_title("Win Rate by Leverage Bucket", fontsize=9, color="#58A6FF", pad=6)
    ax3.set_ylabel("Win Rate (%)", fontsize=7)
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=20, ha="right", fontsize=7)

    # 4. Fear/Greed 30-day rolling MA
    ax4 = fig.add_subplot(gs[1, 0])
    fg_ts = fg_df.set_index("date")["value"].rolling(30).mean()
    ax4.plot(fg_ts.values, color="#FF7043", linewidth=1.2, label="30-day MA")
    ax4.fill_between(range(len(fg_ts)), fg_ts.values, alpha=0.15, color="#FF7043")
    ax4.set_title("Fear/Greed 30-day Rolling Average", fontsize=9, color="#58A6FF", pad=6)
    ax4.set_ylabel("FG Value", fontsize=7)
    ax4.legend(fontsize=6)

    # 5. Correlation matrix
    ax5 = fig.add_subplot(gs[1, 1])
    corr_features = ["total_pnl", "win_rate", "avg_leverage", "trade_count", "avg_size"]
    available     = [c for c in corr_features if c in trader_df.columns]
    corr_mat      = trader_df[available].corr()
    sns.heatmap(
        corr_mat, ax=ax5, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, linewidths=0.5, linecolor="#0D1117",
        annot_kws={"size": 7},
        cbar_kws={"shrink": 0.6},
    )
    ax5.set_title("Correlation Matrix (Trader Metrics)", fontsize=9, color="#58A6FF", pad=6)

    # 6. Mean PnL by day of week
    ax6 = fig.add_subplot(gs[1, 2])
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_pnl   = df.groupby("day_of_week")["closedPnL"].mean().reindex(dow_order)
    bar_colors = ["#66BB6A" if v > 0 else "#D32F2F" for v in dow_pnl.values]
    ax6.bar(dow_order, dow_pnl.values, color=bar_colors, edgecolor="#0D1117", linewidth=0.5)
    ax6.axhline(0, color="#8B949E", linewidth=0.8)
    ax6.set_title("Mean PnL by Day of Week", fontsize=9, color="#58A6FF", pad=6)
    ax6.set_ylabel("Mean PnL (USD)", fontsize=7)
    ax6.set_xticklabels(dow_order, rotation=30, ha="right", fontsize=6.5)

    _save(fig, save_path)
