# RetailLens FastAPI

Read-only API for RetailLens historical sales, saved future forecasts, model reports, and inventory recommendations. The service does not train models or write project data.

## Install and run

From the project root, install the small API dependency set and start the service:

```powershell
python -m pip install -r api/requirements.txt
python -m uvicorn api.main:app --reload
```

For model training and the complete test suite, install the root `requirements.txt` instead.

Open <http://127.0.0.1:8000/docs> for Swagger UI or <http://127.0.0.1:8000/redoc> for ReDoc.

## Endpoints

- `GET /health` — API process status.
- `GET /sales/options` — available stores, store-specific items, and historical date bounds.
- `GET /sales/summary` — totals, statistics, and date coverage; optional `store`, `item`, `start_date`, and `end_date`.
- `GET /sales` — paginated historical observations; same filters plus `limit` (1–10000) and `offset`.
- `GET /forecasts` — saved future forecast rows; optional `store`, `item`, `limit`, and `offset`. Holdout prediction artifacts are excluded.
- `GET /models/comparison` — current per-series training reports and legacy comparison artifacts; optional `store` and `item`.
- `GET /models/evaluation` — current per-series training reports, saved metrics, feature importance, and holdout prediction counts.
- `GET /inventory/recommendation` — calculate reorder point and suggested reorder quantity from a saved forecast. Requires `store`, `item`, and `current_stock`; `model` can be `best`, `lstm`, or `xgboost`. `best` uses the saved training report. Optional inputs are `lead_time_days`, `cycle_stock_days`, and `service_level`.

The inventory calculation needs an actual on-hand stock value, which is not included in the Kaggle sales dataset. Its safety-stock estimate uses forecast variability as a proxy; validate the result against observed demand variability and operational lead times before relying on it.

## Data and generated files

Sales reads `data/processed/sales_clean.csv` first and falls back to `data/raw/train.csv`. Both files must use `date,store,item,sales`. If neither is present, sales routes return a not-found response. Model binaries and forecast files are generated locally and ignored by Git. A Store 1, Item 1 comparison report is checked in as an example; run `src/train.py` for fresh forecasts and recommendations or for other series.

Configure browser access for a different local frontend origin with `RETAILLENS_CORS_ORIGINS`, a comma-separated list of allowed origins. The default allows common Vite and React development ports on localhost.
