# RetailLens — Retail Forecasting Platform

RetailLens compares LSTM and XGBoost demand forecasts and serves historical sales, saved model outputs, and inventory calculations through FastAPI and a React frontend. The frontend uses a light beige and sap-green palette, with a slide-out feature menu and chart, table, and split forecast layouts.

## Overview

For a trained store-item series, the pipeline evaluates two forecasting approaches:

- **LSTM**: sequence model trained on historical sales.
- **XGBoost**: gradient-boosted trees using calendar, lag, and rolling features.

The training pipeline selects a model using chronological holdout RMSE, with MAE as a tie-breaker. The API reads saved reports and forecasts; it does not train models or write project data. The frontend provides forecast exploration, historical sales, model comparisons, and inventory calculations.

## Features

- Filter a store-item series and inspect historical sales.
- Search historical sales by a start and end date on the **Forecast** and **Sales History** tabs. Model Comparison and Inventory are not affected by this date range.
- Explore saved forecast output as a chart, table, or split view, with 30-, 60-, or 90-day horizons where those forecast rows exist.
- Compare LSTM and XGBoost holdout results for a trained series.
- Request a reorder recommendation using a saved forecast and current stock supplied by the user.
- Access the same data through the FastAPI endpoints and interactive API docs.

## Dataset

**Source:** [Store Item Demand Forecasting Challenge](https://www.kaggle.com/competitions/demand-forecasting-kernels-only) (Kaggle)

| Property | Value |
|---|---|
| Records | ~913,000 daily rows |
| Coverage | 10 stores × 50 items = 500 time series |
| Date range | 2013–2017 |
| Columns | `date`, `store`, `item`, `sales` |
| Known gaps | No price, promotion, holiday, or inventory data |

The inventory recommendation endpoint requires current stock to be supplied by the caller. Safety stock uses forecast variability as a proxy; it is not a substitute for an inventory feed or measured demand variability.

The source dataset, processed CSVs, and generated model/forecast files are not committed to Git. To train on the Kaggle data, place its `train.csv` at `data/raw/train.csv`. If raw data is missing, `src/clean_data.py` generates synthetic sample data for local demonstrations; results from that data are not Kaggle results.

## Architecture & Methodology

**Pipeline:** Sales CSV → Cleaning → Feature Engineering / Sequence Building → LSTM and XGBoost Training → Chronological Holdout Evaluation → Saved Reports and Forecasts → FastAPI → React Frontend

Training runs for one store-item series at a time (Store 1, Item 1 by default) and saves forecasts for a 90-day horizon. The frontend can display a shorter horizon by selecting 30 or 60 days. Model selection and reported metrics currently use the same holdout window, so treat results as prototype evaluation rather than an independent final test.

Inventory recommendations calculate lead-time demand, safety stock, reorder point, and suggested order quantity from a saved forecast and caller-provided stock. Validate the forecast-variability safety-stock assumption, lead times, and service level against operational data before using recommendations for purchasing.

## Quickstart

### 1. Installation
```bash
git clone https://github.com/Sandipannnn/RetailLens.git
cd RetailLens
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
```

For an API-only Python environment, install `api/requirements.txt` instead.

### 2. Prepare data and train models
```bash
# Place the Kaggle train.csv at data/raw/train.csv first.
# Train both LSTM and XGBoost for Store 1, Item 1
python src/train.py --store 1 --item 1

# Optional: set the holdout period or force retraining
python src/train.py --store 1 --item 1 --holdout 90 --force-retrain
```

Training defaults to one store-item series. Model files and forecast CSVs are generated locally under ignored `data/processed/` paths; comparison reports are written under `reports/`. A sample Store 1, Item 1 report is checked in, but fresh forecasts and inventory recommendations require generated forecast files. Train each series you want to explore.

### 3. Launch the API
```bash
python -m uvicorn api.main:app --reload
```

API documentation is available at <http://127.0.0.1:8000/docs>.

### 4. Launch the frontend
In a second terminal:
```bash
cd frontend
npm ci
npm run dev
```

Open the Vite URL printed in the terminal (normally <http://localhost:5173>). Keep both the API and frontend terminals running while using the app. The API's default CORS configuration allows this origin. If the API uses a different host or port, create `frontend/.env.local` based on `frontend/.env.example` and set `VITE_API_BASE_URL`.

## Running Tests

The test suite validates the API, time-series models, feature engineering, holdout evaluation, and inventory calculations:

```bash
python -m pytest tests/ -v
```

## API Endpoints

- `GET /health` — service status.
- `GET /sales/options` — available stores, store-specific items, and date bounds for the frontend.
- `GET /sales/summary` and `GET /sales` — historical sales, filtered by `store`, `item`, `start_date`, and `end_date`; `/sales` also supports `limit` and `offset`.
- `GET /forecasts` — saved future forecast rows; historical holdout predictions are excluded.
- `GET /models/comparison` and `GET /models/evaluation` — current per-series reports plus legacy evaluation artifacts, optionally filtered by `store` and `item`.
- `GET /inventory/recommendation` — calculate reorder point and suggested reorder quantity from a saved forecast and supplied `store`, `item`, and `current_stock`; `model` defaults to the saved report's selected model.

Date filtering applies to historical sales data. The frontend exposes the date search on Forecast and Sales History only; model comparison and inventory recommendations are not date-filtered. Set `RETAILLENS_CORS_ORIGINS` to a comma-separated origin list when the frontend runs on a non-default local origin.

## Tech Stack

| Layer | Technologies |
|---|---|
| Core Language | Python 3.10+ |
| Data Processing | `pandas`, `numpy` |
| AI / Deep Learning | `tensorflow`, `keras` |
| Machine Learning | `xgboost`, `scikit-learn`, `joblib` |
| Visualization & Metrics | `matplotlib`, `scipy`, custom SVG forecast chart |
| API | `fastapi`, `uvicorn` |
| Frontend | React, TypeScript, Vite, Lucide icons |
| Testing | `pytest`, `unittest` |

## License

This project is licensed under the **MIT License**.
