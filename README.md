# RetailLens — Retail Sales Forecasting & Inventory Dashboard

A predictive time-series platform that combines Deep Learning (AI) and Machine Learning (ML) to analyze historical retail sales, capture complex seasonal patterns, and project future inventory needs — with multi-step forecasts, uncertainty intervals, and dynamic low-stock alerts surfaced through an interactive dashboard.

## Overview

Retailers need to know not just what sold yesterday, but what is likely to sell next week and next month, per store and per item. This project implements a **dual-model architecture**:
1. **AI Model (Deep Learning)**: Long Short-Term Memory (**LSTM**) neural network for capturing sequential context, non-linear temporal dynamics, and long-range dependencies.
2. **ML Model (Machine Learning)**: Extreme Gradient Boosting (**XGBoost**) utilizing tabular feature engineering (calendar signals, multi-horizon lags, rolling statistics).

Both models compete on a strict chronological holdout test period. A data-driven `ModelSelector` dynamically identifies the superior model per series based on holdout RMSE/MAE (with no hard-coded winner), passing the optimal forecast to the inventory optimization engine and Streamlit dashboard.

## Objectives

- Clean and structure raw daily sales data into an analysis-ready dataset.
- Understand seasonality, trend, and demand patterns through EDA.
- Build an AI deep learning model (**LSTM**) and a machine learning model (**XGBoost**) and compare them rigorously on an identical chronological holdout window.
- Dynamically select the best model per series based on holdout performance (lowest RMSE).
- Generate multi-step forecasts (30/60/90 days) with confidence intervals.
- Turn low-stock risk into concrete, explainable inventory recommendations (Reorder Point & Safety Stock).
- Provide an interactive Streamlit dashboard for store managers and supply chain planners.

## Dataset

**Source:** [Store Item Demand Forecasting Challenge](https://www.kaggle.com/competitions/demand-forecasting-kernels-only) (Kaggle)

| Property | Value |
|---|---|
| Records | ~913,000 daily rows |
| Coverage | 10 stores × 50 items = 500 time series |
| Date range | 2013–2017 |
| Columns | `date`, `store`, `item`, `sales` |
| Known gap | No price, promotion, or holiday data |

Because there is no promo/holiday data, seasonality analysis (day-of-week, monthly, yearly cycles) and lag/rolling feature engineering are key levers for predictive accuracy.

## Architecture & Methodology

**Pipeline:** Data Ingestion → Cleaning → Feature Engineering / Sequencing → Dual-Model Training (LSTM + XGBoost) → Holdout Evaluation → Dynamic Model Selection → Inventory Optimization → Streamlit Dashboard

1. **Cleaning** — Datetime conversion, missing/duplicate validation, chronological sorting (`src/clean_data.py`).
2. **AI Sequence Pipeline** — MinMax scaling fitted strictly on training partition, 30-day sliding-window sequence generation for LSTM (`src/features/lstm_features.py`).
3. **ML Feature Pipeline** — Lag features (7, 14, 21, 28, 30, 60, 90, 365), rolling aggregations (7, 14, 30, 90-day mean & std), and calendar encodings (`src/features/feature_engineering.py`).
4. **AI Deep Learning Model (LSTM)** — 2-layer stacked LSTM with dropout regularization and multi-step autoregressive projection (`src/models/lstm_model.py`).
5. **Machine Learning Model (XGBoost)** — Gradient boosted regression trees trained on engineered lag/rolling features with chronological holdout validation (`src/models/xgboost_model.py`).
6. **Model Selection** — Objective, data-driven selection comparing holdout RMSE and MAE (`src/models/model_selector.py`).
7. **Inventory Optimization** — Lead-time demand, statistical safety stock ($SS = Z \times \sigma \times \sqrt{L}$ at 95% service level), and dynamic Reorder Point (ROP) (`src/inventory.py`).
8. **Interactive Dashboard** — Multi-tab Streamlit dashboard surfacing forecasts, model comparisons, feature importances, and low-stock alerts (`app/streamlit_app.py`).

## Quickstart

### 1. Installation
```bash
git clone <repo-url>
cd aiproject
python -m venv venv
venv\Scripts\activate      # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

### 2. Train Models (LSTM + XGBoost)
```bash
# Train both AI (LSTM) and ML (XGBoost) models on Store 1, Item 1
python src/train.py --store 1 --item 1

# Optional: specify custom holdout or force retrain
python src/train.py --store 1 --item 1 --holdout 90 --force-retrain
```

### 3. Launch Dashboard
```bash
streamlit run app/streamlit_app.py
```

## Running Tests

The test suite validates data leakage prevention, sequence generation, feature engineering, LSTM & XGBoost training/prediction, inventory calculations, and dashboard integrations:

```bash
python -m pytest tests/ -v
```

## Dashboard Features

- **Overview**: High-level store & item sales metrics, aggregate KPIs.
- **Historical Analysis**: Daily, weekly, and monthly sales trends and seasonality.
- **Forecast**: Interactive Plotly projections with confidence bands across 30, 60, and 90-day horizons.
- **Model Comparison**: Side-by-side metric tables (AI LSTM vs ML XGBoost).
- **ML Evaluation**: Model architecture badges, feature importance, holdout error breakdown.
- **Stock Alerts**: Automated inventory recommendations, safety stock, and reorder point alerts.
- **Data Explorer**: Raw data filtering, summaries, and CSV downloads.

## Tech Stack

| Layer | Technologies |
|---|---|
| Core Language | Python 3.10+ |
| Data Processing | `pandas`, `numpy` |
| AI / Deep Learning | `tensorflow`, `keras` |
| Machine Learning | `xgboost`, `scikit-learn`, `joblib` |
| Visualization & Metrics | `plotly`, `matplotlib`, `scipy` |
| Web Application | `streamlit` |
| Testing | `pytest`, `unittest` |

## License

This project is licensed under the **MIT License**.
