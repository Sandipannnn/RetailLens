# RetailLens Sales Analytics API

A separate, read-only FastAPI service for the existing RetailLens data and saved outputs. It does not use Django, retrain models, or change the Streamlit dashboard.

## Install and run (PowerShell, from the RetailLens project root)

```powershell
python -m pip install -r api/requirements.txt
python -m uvicorn api.main:app --reload
```

If your existing `.venv` is active in VS Code, these commands use it. The API files are separate from `ecommerce/`, and it is safe to stop the service with `Ctrl+C`.

## Interactive documentation

Open <http://127.0.0.1:8000/docs> for Swagger UI or <http://127.0.0.1:8000/redoc> for ReDoc.

## Endpoints

- `GET /health` — service status.
- `GET /sales/summary` — totals and statistics; optional `store`, `item`, `start_date`, `end_date` (ISO `YYYY-MM-DD`).
- `GET /sales` — historical sales; same filters plus `limit` (1–10000) and `offset`.
- `GET /forecasts` — saved future forecasts from `data/processed/forecasts/*.csv`; optional `store`, `item`, `limit`, `offset`. Historical holdout predictions are excluded. If no future forecast files exist, returns `available: false` with an empty `records` array.
- `GET /models/comparison` — time-series report and ML comparison outputs, returned separately.
- `GET /models/evaluation` — saved ML metrics, feature importance, and counts of saved holdout prediction rows.

The sales endpoints prefer `data/processed/sales_clean.csv` and fall back to `data/raw/train.csv`. They expect the existing `date,store,item,sales` schema. All endpoints only read project files; the API does not write outputs or train models.
