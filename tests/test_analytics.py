
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import pandas as pd
import numpy as np
from analytics import (
    load_data, get_kpis, get_sentiment_performance,
    run_statistical_tests, get_feature_importance,
    get_trader_leaderboard, get_leverage_winrate,
    SENT_ORDER
)


@pytest.fixture(scope='module')
def loaded():
    fg, tr, df = load_data()
    return fg, tr, df


# ── Data Loading ─────────────────────────────────────────────────────────────

def test_load_returns_three_frames(loaded):
    fg, tr, df = loaded
    assert isinstance(fg, pd.DataFrame)
    assert isinstance(tr, pd.DataFrame)
    assert isinstance(df, pd.DataFrame)

def test_merged_df_has_classification(loaded):
    _, _, df = loaded
    assert 'classification' in df.columns

def test_no_null_classification(loaded):
    _, _, df = loaded
    assert df['classification'].notna().all()

def test_is_winner_binary(loaded):
    _, _, df = loaded
    assert set(df['is_winner'].unique()).issubset({0, 1})

# ── KPIs ─────────────────────────────────────────────────────────────────────

def test_kpi_keys(loaded):
    _, _, df = loaded
    kpis = get_kpis(df)
    for key in ['total_trades','unique_traders','unique_symbols',
                'profitable_pct','total_volume','avg_leverage']:
        assert key in kpis

def test_kpi_total_trades(loaded):
    _, _, df = loaded
    kpis = get_kpis(df)
    assert kpis['total_trades'] == len(df)

def test_kpi_profitable_range(loaded):
    _, _, df = loaded
    kpis = get_kpis(df)
    assert 0 <= kpis['profitable_pct'] <= 100

# ── Sentiment Performance ─────────────────────────────────────────────────────

def test_sentiment_perf_all_regimes(loaded):
    _, _, df = loaded
    sp = get_sentiment_performance(df)
    names = [s['classification'] for s in sp]
    for s in SENT_ORDER:
        assert s in names

def test_sentiment_win_rate_range(loaded):
    _, _, df = loaded
    for s in get_sentiment_performance(df):
        assert 0 <= s['win_rate'] <= 100

# ── Statistical Tests ─────────────────────────────────────────────────────────

def test_stat_keys(loaded):
    _, _, df = loaded
    t = run_statistical_tests(df)
    for k in ['kruskal_h','kruskal_p','spearman_r','spearman_p','kw_significant']:
        assert k in t

def test_pvalue_in_range(loaded):
    _, _, df = loaded
    t = run_statistical_tests(df)
    assert 0 <= t['kruskal_p'] <= 1
    assert 0 <= t['spearman_p'] <= 1

# ── ML ────────────────────────────────────────────────────────────────────────

def test_feature_importance_sums_100(loaded):
    _, _, df = loaded
    fi = get_feature_importance(df)
    total = sum(f['importance'] for f in fi['features'])
    assert abs(total - 100.0) < 1.0   # allow floating point

def test_cv_auc_reasonable(loaded):
    _, _, df = loaded
    fi = get_feature_importance(df)
    assert 0.4 <= fi['cv_auc'] <= 1.0

# ── Leaderboard ───────────────────────────────────────────────────────────────

def test_leaderboard_top20(loaded):
    _, _, df = loaded
    lb = get_trader_leaderboard(df, top_n=20)
    assert len(lb) == 20

def test_leaderboard_sorted(loaded):
    _, _, df = loaded
    lb = get_trader_leaderboard(df)
    pnls = [t['total_pnl'] for t in lb]
    assert pnls == sorted(pnls, reverse=True)

# ── Leverage ──────────────────────────────────────────────────────────────────

def test_leverage_buckets(loaded):
    _, _, df = loaded
    lv = get_leverage_winrate(df)
    buckets = [l['bucket'] for l in lv]
    assert set(buckets) == {"1-2x","3-5x","6-10x","11-25x","26x+"}

def test_winrate_range_leverage(loaded):
    _, _, df = loaded
    for l in get_leverage_winrate(df):
        assert 0 <= l['win_rate'] <= 100
