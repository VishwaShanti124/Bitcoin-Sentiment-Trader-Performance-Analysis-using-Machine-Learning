"""
generate_data.py
----------------
Generates synthetic Hyperliquid trade fills and Fear & Greed Index data
that mirrors the statistical properties described in the Primetrade.ai
Data Science Assignment brief.

Run once before any other module:
    python data/generate_data.py
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

SEED = 42
np.random.seed(SEED)

# ── Constants ────────────────────────────────────────────────────────────────
START_DATE = datetime(2023, 1, 1)
END_DATE   = datetime(2024, 5, 31)
N_TRADES   = 20_000
N_TRADERS  = 80
SYMBOLS    = ["BTC-USD", "ETH-USD", "SOL-USD", "ARB-USD", "DOGE-USD"]
SYM_WEIGHTS = [0.40, 0.25, 0.15, 0.10, 0.10]       # trade-frequency weights

# Sentiment regime proportions (from EDA pie chart)
REGIMES = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
REGIME_PROPS = [0.224, 0.181, 0.139, 0.204, 0.252]  # sums to 1.0

# Mean PnL per regime (from assignment table, Section 4)
REGIME_MEAN_PNL = {
    "Extreme Fear":  -8.30,
    "Fear":          +8.99,
    "Neutral":       -1.76,
    "Greed":         -0.60,
    "Extreme Greed": -0.96,
}

# Regime win rates (approx from table)
REGIME_WIN_RATE = {
    "Extreme Fear":  0.235,
    "Fear":          0.249,
    "Neutral":       0.231,
    "Greed":         0.250,
    "Extreme Greed": 0.244,
}

# ── Build Fear & Greed time-series ────────────────────────────────────────────
def make_fear_greed() -> pd.DataFrame:
    n_days = (END_DATE - START_DATE).days + 1
    dates  = [START_DATE + timedelta(days=i) for i in range(n_days)]

    # Walk with mean-reversion so regimes cluster realistically
    fg = np.zeros(n_days)
    fg[0] = 50.0
    for i in range(1, n_days):
        fg[i] = np.clip(fg[i-1] + np.random.normal(0, 4) + 0.05*(50 - fg[i-1]), 1, 99)

    labels = pd.cut(
        fg,
        bins=[0, 20, 40, 60, 80, 100],
        labels=REGIMES,
        include_lowest=True,
    )

    return pd.DataFrame({"date": dates, "value": np.round(fg, 1), "classification": labels})


# ── Build trade fills ─────────────────────────────────────────────────────────
def make_trades(fg_df: pd.DataFrame) -> pd.DataFrame:
    date_to_regime = dict(zip(fg_df["date"].astype(str), fg_df["classification"]))
    date_to_fg     = dict(zip(fg_df["date"].astype(str), fg_df["value"]))

    accounts = [f"trader_{i:03d}" for i in range(N_TRADERS)]

    # Assign each trader a "style" that determines their leverage preference
    trader_leverage_mu = {a: np.random.choice([2, 5, 10, 15, 20, 25], p=[0.15,0.20,0.25,0.20,0.12,0.08])
                          for a in accounts}

    rows = []
    for _ in range(N_TRADES):
        # Random timestamp within date range
        delta_days  = np.random.randint(0, (END_DATE - START_DATE).days)
        trade_date  = START_DATE + timedelta(days=delta_days)
        trade_hour  = int(np.random.choice(range(24), p=_hour_weights()))
        ts = trade_date.replace(hour=trade_hour,
                                minute=np.random.randint(0, 60),
                                second=np.random.randint(0, 60))

        date_str = str(trade_date.date())
        regime   = date_to_regime.get(date_str, "Neutral")
        fg_val   = date_to_fg.get(date_str, 50.0)

        account  = np.random.choice(accounts)
        symbol   = np.random.choice(SYMBOLS, p=SYM_WEIGHTS)
        side     = np.random.choice(["long", "short"])

        lev_mu   = trader_leverage_mu[account]
        leverage = float(np.clip(np.random.normal(lev_mu, lev_mu * 0.3), 1, 30))

        # Execution price (rough realistic ranges)
        price_ranges = {
            "BTC-USD":  (16_000, 72_000),
            "ETH-USD":  (1_000,  4_000),
            "SOL-USD":  (10,     220),
            "ARB-USD":  (0.5,    2.0),
            "DOGE-USD": (0.05,   0.20),
        }
        lo, hi = price_ranges[symbol]
        exec_price = round(np.random.uniform(lo, hi), 4)

        # Size: larger position = more variance
        size = round(abs(np.random.lognormal(1.5, 1.0)), 4)
        value = round(exec_price * size, 2)

        # PnL: drawn from regime distribution, with fat tails
        mu    = REGIME_MEAN_PNL[str(regime)]
        sigma = abs(mu) * 12 + 30          # fat tails
        pnl   = round(np.random.normal(mu, sigma) * (1 + 0.05 * leverage), 2)

        rows.append({
            "timestamp":       ts,
            "account":         account,
            "symbol":          symbol,
            "side":            side,
            "execution_price": exec_price,
            "size":            size,
            "value":           value,
            "leverage":        round(leverage, 1),
            "closedPnL":       pnl,
            "sentiment":       str(regime),
            "fg_value":        fg_val,
        })

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    df["is_winner"] = (df["closedPnL"] > 0).astype(int)
    return df


def _hour_weights():
    """Heavier weight during UTC 8-18 to mimic market hours."""
    w = np.ones(24)
    w[8:18] *= 2.5
    w /= w.sum()
    return w


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    out_dir = os.path.join(os.path.dirname(__file__))

    print("Generating Fear & Greed data …")
    fg = make_fear_greed()
    fg_path = os.path.join(out_dir, "fear_greed.csv")
    fg.to_csv(fg_path, index=False)
    print(f"  Saved {len(fg):,} rows → {fg_path}")

    print("Generating trade fills …")
    trades = make_trades(fg)
    tr_path = os.path.join(out_dir, "trades.csv")
    trades.to_csv(tr_path, index=False)
    print(f"  Saved {len(trades):,} rows → {tr_path}")

    print("\nSample trade:\n", trades.head(2).to_string())
    print("\nFear/Greed sample:\n", fg.head(3).to_string())
    print("\nData generation complete.")
