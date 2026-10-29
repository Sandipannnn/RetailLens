# RetailLens Presentation Outline

## Slide 1: RetailLens
Retail Sales Forecasting API

## Slide 2: Problem
Retail teams need reliable insight into historical sales and future store-item demand to make better planning decisions.

## Slide 3: Project objective
- Clean and analyze daily retail sales.
- Train and compare LSTM and XGBoost forecasting models.
- Expose historical sales, saved forecasts, evaluation results, and inventory calculations through an API.

## Slide 4: Architecture
Sales CSV → Cleaning → Feature engineering and sequence building → LSTM/XGBoost → Chronological holdout evaluation → Model selection and forecast exports → FastAPI.

The API is implemented. The React frontend uses a beige and sap-green theme, a slide-out feature menu, and chart/table/split forecasting layouts.

## Slide 5: Data
- Kaggle Store Item Demand Forecasting Challenge.
- 10 stores × 50 items, with daily observations from 2013 through 2017.
- No price, promotions, holidays, or current inventory fields.
- Source data and trained model/forecast binaries are local and ignored by Git; one example comparison report is checked in.

## Slide 6: Model evaluation example
The saved Store 1, Item 1 training report records a 90-day chronological holdout:

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| LSTM | 4.4893 | 5.5637 | 9.5615% |
| XGBoost | 4.9928 | 6.1196 | 10.6410% |

LSTM is selected for this series by the lower RMSE. These are one-series results, not an aggregate score across all 500 series. Regenerate reports when models, data, or training settings change.

## Slide 7: API
- Filter and summarize historical sales.
- Retrieve saved future forecasts separately from historical holdout predictions.
- Access per-series comparison reports and evaluation artifacts.
- Explore the API interactively through `/docs`.
- Use the slide-out menu to switch between forecast, sales history, model comparison, and inventory views.
- Switch the forecast display between chart, table, or split layouts.

## Slide 8: Inventory recommendation
The caller supplies current stock. The API calculates lead-time demand, safety stock, reorder point, and suggested reorder quantity from a saved forecast. The source dataset has no inventory data; safety stock uses forecast variability as a proxy and requires operational validation.

## Slide 9: Limitations
- No external demand signals or inventory feed.
- Model selection and performance reporting currently share a holdout window.
- Training and saved reports are per store-item series.
- A production deployment pipeline is not yet included.

## Slide 10: Next steps
- Add a separate validation process and final untouched test window.
- Connect an authoritative current-stock source and evaluate inventory recommendations with business data.
- Add browser-level UI tests and accessibility checks.
- Add automated CI and API integration checks.
