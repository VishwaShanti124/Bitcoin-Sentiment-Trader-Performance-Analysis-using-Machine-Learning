"""
analytics.py  –  All analysis logic for the Primetrade.ai dashboard.
Isolated from Flask so it can be unit-tested independently.
"""
from __future__ import annotations
import os
import pandas as pd
import numpy as np
from scipy.stats import kruskal, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
SENT_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]


# ── Data Loading ────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fg = pd.read_csv(os.path.join(DATA_DIR, 'fear_greed.csv'))
    tr = pd.read_csv(os.path.join(DATA_DIR, 'trader_data.csv'))

    fg['date'] = pd.to_datetime(fg['date'])
    tr['time'] = pd.to_datetime(tr['time'])
    tr['date'] = tr['time'].dt.normalize()
    tr['hour'] = tr['time'].dt.hour
    tr['day_of_week'] = tr['time'].dt.day_name()

    df = tr.merge(fg, on='date', how='left')
    df['is_winner'] = (df['closedPnL'] > 0).astype(int)
    df['trade_value'] = df['execution_price'] * df['size']
    df['classification'] = pd.Categorical(
        df['classification'], categories=SENT_ORDER, ordered=True)

    return fg, tr, df


# ── KPI Summary ─────────────────────────────────────────────────────────────

def get_kpis(df: pd.DataFrame) -> dict:
    trader_stats = df.groupby('account').agg(
        total_pnl=('closedPnL', 'sum')).reset_index()
    profitable_pct = (trader_stats['total_pnl'] > 0).mean() * 100

    return {
        'total_trades':    int(len(df)),
        'unique_traders':  int(df['account'].nunique()),
        'unique_symbols':  int(df['symbol'].nunique()),
        'profitable_pct':  round(profitable_pct, 1),
        'total_volume':    round(df['trade_value'].sum(), 0),
        'avg_leverage':    round(df['leverage'].mean(), 1),
    }


# ── Sentiment Performance ────────────────────────────────────────────────────

def get_sentiment_performance(df: pd.DataFrame) -> list[dict]:
    sp = df.groupby('classification', observed=False).agg(
        mean_pnl  =('closedPnL', 'mean'),
        median_pnl=('closedPnL', 'median'),
        win_rate  =('is_winner', 'mean'),
        trade_cnt =('closedPnL', 'count'),
        avg_lev   =('leverage',  'mean'),
    ).reset_index()
    sp['win_rate'] = (sp['win_rate'] * 100).round(1)
    sp['mean_pnl']  = sp['mean_pnl'].round(2)
    sp['avg_lev']   = sp['avg_lev'].round(1)
    return sp.to_dict(orient='records')


# ── Statistical Tests ────────────────────────────────────────────────────────

def run_statistical_tests(df: pd.DataFrame) -> dict:
    groups = [df[df['classification'] == s]['closedPnL'].dropna()
              for s in SENT_ORDER]
    kw_stat, kw_pval = kruskal(*groups)
    corr_s, p_s = spearmanr(
        df['value'].dropna(),
        df.loc[df['value'].notna(), 'closedPnL'])
    return {
        'kruskal_h':   round(float(kw_stat), 3),
        'kruskal_p':   round(float(kw_pval), 4),
        'spearman_r':  round(float(corr_s), 4),
        'spearman_p':  round(float(p_s), 4),
        'kw_significant': bool(kw_pval < 0.05),
    }


# ── ML Feature Importance ───────────────────────────────────────────────────

def get_feature_importance(df: pd.DataFrame) -> list[dict]:
    ml = df[['closedPnL', 'leverage', 'size', 'execution_price',
             'value', 'hour', 'is_winner']].dropna()
    X = ml[['leverage', 'size', 'execution_price', 'value', 'hour']]
    y = ml['is_winner']
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    cv_auc = cross_val_score(rf, X, y, cv=5, scoring='roc_auc').mean()
    rf.fit(X, y)
    fi = pd.Series(rf.feature_importances_, index=X.columns)\
           .sort_values(ascending=False)
    return {
        'features': [{'name': k, 'importance': round(float(v)*100, 1)}
                     for k, v in fi.items()],
        'cv_auc': round(float(cv_auc), 4),
    }


# ── Time Series ─────────────────────────────────────────────────────────────

def get_timeseries(fg: pd.DataFrame) -> list[dict]:
    fg_s = fg.sort_values('date').copy()
    fg_s['roll7']  = fg_s['value'].rolling(7,  min_periods=1).mean().round(1)
    fg_s['roll30'] = fg_s['value'].rolling(30, min_periods=1).mean().round(1)
    fg_s['date']   = fg_s['date'].dt.strftime('%Y-%m-%d')
    return fg_s[['date', 'value', 'classification', 'roll7', 'roll30']]\
               .to_dict(orient='records')


# ── Daily Volume ─────────────────────────────────────────────────────────────

def get_daily_volume(df: pd.DataFrame) -> list[dict]:
    dv = df.groupby(['date', 'classification'], observed=False)\
           .size().reset_index(name='count')
    dv['date'] = dv['date'].dt.strftime('%Y-%m-%d')
    return dv.to_dict(orient='records')


# ── PnL Distribution ─────────────────────────────────────────────────────────

def get_pnl_distribution(df: pd.DataFrame) -> dict:
    result = {}
    clipped = df[df['closedPnL'].between(-500, 500)]
    for s in SENT_ORDER:
        vals = clipped[clipped['classification'] == s]['closedPnL'].dropna().tolist()
        result[s] = vals[:2000]          # cap for JSON size
    return result


# ── Trader Leaderboard ───────────────────────────────────────────────────────

def get_trader_leaderboard(df: pd.DataFrame, top_n: int = 20) -> list[dict]:
    ts = df.groupby('account').agg(
        total_pnl   =('closedPnL', 'sum'),
        win_rate    =('is_winner', 'mean'),
        trade_count =('closedPnL', 'count'),
        avg_leverage=('leverage',  'mean'),
        symbols     =('symbol',    'nunique'),
    ).reset_index()
    ts['win_rate']     = (ts['win_rate'] * 100).round(1)
    ts['total_pnl']    = ts['total_pnl'].round(2)
    ts['avg_leverage'] = ts['avg_leverage'].round(1)
    ts['rank']         = range(1, len(ts) + 1)
    top = ts.nlargest(top_n, 'total_pnl').reset_index(drop=True)
    top['rank'] = range(1, top_n + 1)
    # shorten address for display
    top['address'] = top['account'].str[:6] + '…' + top['account'].str[-4:]
    return top[['rank','address','total_pnl','win_rate',
                'trade_count','avg_leverage','symbols']].to_dict(orient='records')


# ── Symbol Breakdown ─────────────────────────────────────────────────────────

def get_symbol_breakdown(df: pd.DataFrame) -> list[dict]:
    sb = df.groupby(['symbol', 'classification'], observed=False)\
           .agg(mean_pnl=('closedPnL', 'mean'),
                trade_cnt=('closedPnL', 'count'))\
           .reset_index()
    sb['mean_pnl'] = sb['mean_pnl'].round(2)
    return sb.to_dict(orient='records')


# ── Hour Heatmap ─────────────────────────────────────────────────────────────

def get_hour_heatmap(df: pd.DataFrame) -> list[dict]:
    hm = df.groupby(['hour', 'classification'], observed=False)\
           ['closedPnL'].mean().reset_index()
    hm['closedPnL'] = hm['closedPnL'].round(2)
    hm.columns = ['hour', 'sentiment', 'mean_pnl']
    return hm.to_dict(orient='records')


# ── Day of Week ──────────────────────────────────────────────────────────────

def get_dow_pnl(df: pd.DataFrame) -> list[dict]:
    dow = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    res = df.groupby('day_of_week')['closedPnL'].mean().reindex(dow).reset_index()
    res.columns = ['day', 'mean_pnl']
    res['mean_pnl'] = res['mean_pnl'].round(2)
    return res.to_dict(orient='records')


# ── Leverage Bucket ──────────────────────────────────────────────────────────

def get_leverage_winrate(df: pd.DataFrame) -> list[dict]:
    df2 = df.copy()
    df2['lev_bucket'] = pd.cut(df2['leverage'],
                                bins=[0, 2, 5, 10, 25, 100],
                                labels=["1-2x","3-5x","6-10x","11-25x","26x+"])
    lb = df2.groupby('lev_bucket', observed=True)['is_winner']\
            .mean().reset_index()
    lb.columns = ['bucket', 'win_rate']
    lb['win_rate'] = (lb['win_rate'] * 100).round(1)
    return lb.to_dict(orient='records')


# ── Full bundle ──────────────────────────────────────────────────────────────

def get_full_dashboard_data() -> dict:
    fg, tr, df = load_data()
    return {
        'kpis':                get_kpis(df),
        'sentiment_perf':      get_sentiment_performance(df),
        'stats_tests':         run_statistical_tests(df),
        'feature_importance':  get_feature_importance(df),
        'timeseries':          get_timeseries(fg),
        'daily_volume':        get_daily_volume(df),
        'pnl_distribution':    get_pnl_distribution(df),
        'trader_leaderboard':  get_trader_leaderboard(df),
        'symbol_breakdown':    get_symbol_breakdown(df),
        'hour_heatmap':        get_hour_heatmap(df),
        'dow_pnl':             get_dow_pnl(df),
        'leverage_winrate':    get_leverage_winrate(df),
    }
