"""
src/data_loader.py
------------------
Handles data ingestion, merging, and feature engineering.
Reads CSVs produced by data/generate_data.py or real exports.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

SENTIMENT_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
SENTIMENT_CODES = {s: i for i, s in enumerate(SENTIMENT_ORDER)}

LEVERAGE_BINS   = [0, 2, 5, 10, 25, 100]
LEVERAGE_LABELS = ["1-2x", "3-5x", "6-10x", "11-25x", "25x+"]


def load_raw() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (trades_df, fear_greed_df) from CSV files in data/."""
    trades_path = os.path.join(DATA_DIR, "trades.csv")
    fg_path     = os.path.join(DATA_DIR, "fear_greed.csv")

    if not os.path.exists(trades_path) or not os.path.exists(fg_path):
        raise FileNotFoundError(
            "Data files not found. Run `python data/generate_data.py` first."
        )

    trades = pd.read_csv(trades_path, parse_dates=["timestamp"])
    fg     = pd.read_csv(fg_path, parse_dates=["date"])
    return trades, fg


def preprocess(trades: pd.DataFrame, fg: pd.DataFrame) -> pd.DataFrame:
    """
    Merge trade fills with daily Fear & Greed, engineer features,
    and return a clean analysis-ready DataFrame.
    """
    # UTC normalise
    trades["timestamp"] = pd.to_datetime(trades["timestamp"], utc=True).dt.tz_localize(None)
    trades["date"]      = trades["timestamp"].dt.date.astype(str)
    fg["date"]          = fg["date"].astype(str)

    # Left-join on calendar date
    df = trades.merge(
        fg[["date", "value", "classification"]].rename(
            columns={"value": "fg_index", "classification": "sentiment"}
        ),
        on="date",
        how="left",
        suffixes=("", "_fg"),
    )

    # If sentiment already in trades (synthetic data), use that; else from merge
    if "sentiment_fg" in df.columns:
        df["sentiment"] = df["sentiment"].combine_first(df.pop("sentiment_fg"))
    if "fg_value" in df.columns and "fg_index" in df.columns:
        df["fg_index"] = df["fg_index"].combine_first(df["fg_value"])
    elif "fg_value" in df.columns:
        df.rename(columns={"fg_value": "fg_index"}, inplace=True)

    # Drop duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]

    # Temporal features
    df["hour"]       = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.day_name()

    # Derived fields
    df["is_winner"]      = (df["closedPnL"] > 0).astype(int)
    df["abs_pnl"]        = df["closedPnL"].abs()
    df["sentiment_code"] = df["sentiment"].map(SENTIMENT_CODES)

    # Leverage bucket
    df["leverage_bucket"] = pd.cut(
        df["leverage"],
        bins=LEVERAGE_BINS,
        labels=LEVERAGE_LABELS,
        right=True,
    )

    # Ordered sentiment category
    df["sentiment"] = pd.Categorical(
        df["sentiment"], categories=SENTIMENT_ORDER, ordered=True
    )

    # Drop rows with critical NaNs
    df.dropna(subset=["closedPnL", "leverage", "sentiment"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def build_trader_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-trader metrics used in deep-dive analytics."""
    summary = (
        df.groupby("account")
        .agg(
            total_pnl    =("closedPnL",  "sum"),
            win_rate     =("is_winner",  "mean"),
            avg_leverage =("leverage",   "mean"),
            trade_count  =("closedPnL",  "count"),
            avg_size     =("size",       "mean"),
        )
        .reset_index()
    )
    summary["is_profitable"] = (summary["total_pnl"] > 0).astype(int)
    return summary


def get_sentiment_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return the sentiment × performance table (Section 4 of report)."""
    return (
        df.groupby("sentiment", observed=True)
        .agg(
            trades      =("closedPnL",  "count"),
            mean_pnl    =("closedPnL",  "mean"),
            win_rate    =("is_winner",  "mean"),
            avg_leverage=("leverage",   "mean"),
        )
        .round(2)
        .reset_index()
    )
