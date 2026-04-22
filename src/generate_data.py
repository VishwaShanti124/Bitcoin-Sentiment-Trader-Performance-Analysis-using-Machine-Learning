"""
Generate sample datasets for the Primetrade.ai web application.
Run once to create data/fear_greed.csv and data/trader_data.csv
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random, os

np.random.seed(42)
random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# ── Fear & Greed Index ──────────────────────────────────────────────────────
start = datetime(2023, 1, 1)
dates = [start + timedelta(days=i) for i in range(500)]

fg_values = []
v = 50
for _ in dates:
    v = float(np.clip(v + np.random.normal(0, 6), 1, 99))
    fg_values.append(int(v))

def classify(x):
    if x <= 24: return "Extreme Fear"
    if x <= 44: return "Fear"
    if x <= 55: return "Neutral"
    if x <= 74: return "Greed"
    return "Extreme Greed"

fear_greed = pd.DataFrame({
    "date":           [d.strftime("%Y-%m-%d") for d in dates],
    "value":          fg_values,
    "classification": [classify(v) for v in fg_values],
})
fear_greed.to_csv(os.path.join(DATA_DIR, 'fear_greed.csv'), index=False)

# ── Trader Data ─────────────────────────────────────────────────────────────
symbols  = ["BTC-USD", "ETH-USD", "SOL-USD", "ARB-USD", "DOGE-USD"]
accounts = [f"0x{i:04x}{'a'*36}" for i in range(1, 81)]

rows = []
for _ in range(20_000):
    acct  = random.choice(accounts)
    sym   = random.choice(symbols)
    side  = random.choice(["BUY", "SELL"])
    ts    = start + timedelta(days=random.randint(0, 499),
                              hours=random.randint(0, 23),
                              minutes=random.randint(0, 59))
    base  = {"BTC-USD": 30000, "ETH-USD": 1800, "SOL-USD": 25,
             "ARB-USD": 1.2, "DOGE-USD": 0.08}[sym]
    px    = base * np.random.uniform(0.85, 1.15)
    size  = abs(np.random.lognormal(0, 1) * 0.5)
    lev   = random.choice([1, 2, 3, 5, 10, 20, 50])
    pnl   = np.random.normal(0, px * size * 0.03) if side == "SELL" else 0.0

    rows.append({
        "account":         acct,
        "symbol":          sym,
        "execution_price": round(px, 4),
        "size":            round(size, 6),
        "side":            side,
        "time":            ts.strftime("%Y-%m-%d %H:%M:%S"),
        "start_position":  round(np.random.uniform(-5, 5), 6),
        "event":           "FILL",
        "closedPnL":       round(pnl, 4),
        "leverage":        lev,
    })

pd.DataFrame(rows).to_csv(os.path.join(DATA_DIR, 'trader_data.csv'), index=False)
print("Data generated:", DATA_DIR)
