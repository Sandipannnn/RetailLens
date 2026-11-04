# RetailLens Frontend

Responsive React + TypeScript frontend for the RetailLens FastAPI service. The UI uses a light beige and sap-green palette, a slide-out feature menu, and chart/table/split forecast layouts.

## Run locally

Use Node.js 20.19 or newer.

Start the API from the repository root in one terminal:

```powershell
python -m uvicorn api.main:app --reload
```

In another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL (normally <http://localhost:5173>). The API allows this local origin by default. To use a different API host, create `.env.local` based on `.env.example` and set `VITE_API_BASE_URL`.

The UI reads historical sales, forecast exports, and model reports from the API. Store, product, forecast model, horizon, and date changes automatically refresh the displayed data. Changing stores preserves the selected product when available. Use the `From` and `To` date fields to filter historical summaries, records, and chart points; **Search** can also refresh the current selection. **Reset** restores the full available date range. Forecast output remains a future horizon and is shown separately. Inventory recommendations require the user to enter current stock because the source sales dataset has no inventory values. Recommendations then update automatically when the store, product, model, stock, lead time, or order cycle changes. Older requests are cancelled so their results cannot replace the current selection.


## Forecast table data

Demand Outlook lists future predictions first, followed by the latest 90 historical
observations. Historical rows intentionally have no forecast or bounds. Missing
forecast files must be generated for the selected store and item; sales rows are
never substituted for predictions.

For an XGBoost-only export, run from the repository root:

```powershell
python -m src.export_xgboost --store 2 --item 1
```

This generates 90 future rows and approximate 90% empirical absolute-error bounds
calibrated on a recursive holdout forecast. Future coverage is not guaranteed.
The full `src/train.py` pipeline also exports bounds for LSTM and XGBoost using
its holdout predictions. Refresh the dashboard after generating forecast files.

## Generate forecasts for every selection

From the repository root, run `python -m src.generate_missing_forecasts` in an environment with the root requirements installed. This trains missing series with both models, exports 90 future days, and updates the comparison reports. Complete exports are skipped on subsequent runs. Progress is saved in `data/processed/forecasts/generation_status.json`; a failed series is retried on the next run. Keep the process running until its status is `complete`.

When predictions are unavailable, the dashboard checks every 5 seconds for new exports. Forecast totals and daily averages use only the selected model's generated predictions.

When a selected model has no predictions, the two demand cards show **Recent sales** and **Historical daily average**, calculated from the latest 30, 60, or 90 available daily observations within the selected date range. These are actual historical values, not substitute forecasts. Once predictions arrive, the labels and values switch to forecast demand and projected daily average. Background checks preserve the loaded historical values.
