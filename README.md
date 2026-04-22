# Primetrade.ai — Sentiment Analysis Web Dashboard

> Bitcoin Market Sentiment × Hyperliquid Trader Performance  
> Full-stack Flask + Vanilla JS interactive analytics dashboard

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate sample data (first time only)
```bash
python src/generate_data.py
```

### 3. Run the web server
```bash
python src/app.py
```

### 4. Open your browser
```
http://localhost:5000
```

---

## 📁 Project Structure

```
primetrade_analysis/
├── src/
│   ├── app.py              # Flask application + all API routes
│   ├── analytics.py        # Core analysis engine (pure Python, no Flask)
│   └── generate_data.py    # Sample data generator
├── data/
│   ├── fear_greed.csv       # Fear/Greed Index (auto-generated)
│   └── trader_data.csv      # Hyperliquid trade fills (auto-generated)
├── templates/
│   └── index.html           # Single-page dashboard UI
├── static/                  # Static assets (CSS/JS if extracted)
├── tests/
│   └── test_analytics.py    # pytest unit tests
├── outputs/                 # Generated charts / reports
├── requirements.txt
└── README.md
```

---

## 🌐 API Endpoints

| Endpoint          | Description                              |
|-------------------|------------------------------------------|
| `GET /`           | Dashboard UI                             |
| `GET /api/dashboard` | Full data payload (all charts)        |
| `GET /api/kpis`   | Top-level KPI metrics                    |
| `GET /api/sentiment` | Sentiment performance breakdown       |
| `GET /api/stats`  | Statistical test results                 |
| `GET /api/features` | ML feature importance                  |
| `GET /api/timeseries` | Fear/Greed time series              |
| `GET /api/leaderboard` | Top 20 trader rankings             |
| `GET /api/symbols` | Symbol × sentiment breakdown            |
| `GET /api/heatmap` | Hour × sentiment PnL heatmap            |
| `GET /api/leverage` | Leverage bucket win rates              |
| `GET /api/dow`    | Day-of-week PnL patterns                 |
| `GET /api/pnl_dist` | PnL distribution by sentiment          |
| `GET /api/health` | Health check                             |
| `POST /api/refresh` | Re-run all analytics                  |

---

## 🧪 Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

---

## 📊 Dashboard Sections

| Tab | Contents |
|-----|----------|
| **Overview** | KPI strip, FG rolling average, sentiment distribution, win rates |
| **Sentiment Analysis** | Per-regime PnL, volume, symbol breakdown |
| **Trader Performance** | Day-of-week PnL, leverage chart, hour heatmap |
| **Statistical Tests** | Kruskal-Wallis + Spearman results, PnL histograms |
| **ML Insights** | Feature importance bars + radar chart, CV AUC |
| **Temporal Patterns** | Full FG time series, stacked daily volume |
| **Leaderboard** | Top 20 traders by PnL with win rate & leverage |

---

## 🔧 Production Deployment

```bash
# Using gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 src.app:app

# Or with Docker
docker build -t primetrade-dashboard .
docker run -p 5000:5000 primetrade-dashboard
```

---

## 📈 Key Findings

- **Fear regime** produces the highest mean PnL (+$8.99), consistent with contrarian strategies
- **Fear/Greed Index** is the 3rd most important ML feature at ~19% importance  
- **High leverage** (>25x) correlates with below 50% win rate — structural headwind
- **48.8%** of traders are profitable over the observation period

---

## 🏗️ Tech Stack

- **Backend:** Python 3.10+, Flask 3.x, Pandas, NumPy, Scipy, Scikit-learn
- **Frontend:** Vanilla JS, Chart.js 4, CSS Grid, Google Fonts (Syne + DM Mono)
- **Testing:** pytest
- **Deployment:** Gunicorn / Docker-ready
