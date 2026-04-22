"""
app.py  –  Flask web application for Primetrade.ai Analysis Dashboard
Run:  python app.py
Then open:  http://localhost:5000
"""
from __future__ import annotations
import os, sys, json, threading, time
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, jsonify, render_template, send_from_directory, request
from functools import lru_cache
from analytics import get_full_dashboard_data, load_data, \
    get_sentiment_performance, get_trader_leaderboard, run_statistical_tests

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), '..', 'templates'),
    static_folder=os.path.join(os.path.dirname(__file__), '..', 'static'),
)

# ── Cache dashboard data (expensive ML + stats) ────────────────────────────
_CACHE: dict = {}
_CACHE_LOCK = threading.Lock()

def _warm_cache():
    global _CACHE
    data = get_full_dashboard_data()
    with _CACHE_LOCK:
        _CACHE = data
    print("✓ Dashboard data cached")

# Warm on startup in background so first request is fast
threading.Thread(target=_warm_cache, daemon=True).start()


# ── Pages ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


# ── API Endpoints ──────────────────────────────────────────────────────────

@app.route('/api/dashboard')
def api_dashboard():
    """Full dashboard payload – all charts in one call."""
    with _CACHE_LOCK:
        data = dict(_CACHE)
    if not data:
        data = get_full_dashboard_data()
    return jsonify(data)


@app.route('/api/kpis')
def api_kpis():
    with _CACHE_LOCK:
        return jsonify(_CACHE.get('kpis', {}))


@app.route('/api/sentiment')
def api_sentiment():
    with _CACHE_LOCK:
        return jsonify(_CACHE.get('sentiment_perf', []))


@app.route('/api/stats')
def api_stats():
    with _CACHE_LOCK:
        return jsonify(_CACHE.get('stats_tests', {}))


@app.route('/api/features')
def api_features():
    with _CACHE_LOCK:
        return jsonify(_CACHE.get('feature_importance', {}))


@app.route('/api/timeseries')
def api_timeseries():
    with _CACHE_LOCK:
        return jsonify(_CACHE.get('timeseries', []))


@app.route('/api/leaderboard')
def api_leaderboard():
    with _CACHE_LOCK:
        return jsonify(_CACHE.get('trader_leaderboard', []))


@app.route('/api/symbols')
def api_symbols():
    with _CACHE_LOCK:
        return jsonify(_CACHE.get('symbol_breakdown', []))


@app.route('/api/heatmap')
def api_heatmap():
    with _CACHE_LOCK:
        return jsonify(_CACHE.get('hour_heatmap', []))


@app.route('/api/leverage')
def api_leverage():
    with _CACHE_LOCK:
        return jsonify(_CACHE.get('leverage_winrate', []))


@app.route('/api/dow')
def api_dow():
    with _CACHE_LOCK:
        return jsonify(_CACHE.get('dow_pnl', []))


@app.route('/api/pnl_dist')
def api_pnl_dist():
    with _CACHE_LOCK:
        return jsonify(_CACHE.get('pnl_distribution', {}))


@app.route('/api/refresh', methods=['POST'])
def api_refresh():
    """Force re-run of all analytics (e.g. after uploading new data)."""
    threading.Thread(target=_warm_cache, daemon=True).start()
    return jsonify({'status': 'refreshing'})


@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'cached': bool(_CACHE)})


if __name__ == '__main__':
    # Generate data if not present
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    if not os.path.exists(os.path.join(data_dir, 'trader_data.csv')):
        print("Generating sample data...")
        from generate_data import *  # noqa
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
