# RetailLens Presentation Outline

## Slide 1: RetailLens
Retail Sales Forecasting & Analytics

## Slide 2: Problem Statement
Retail teams need visibility into sales trends, future demand, product performance, and potential stock risks.

## Slide 3: Project Objective
- Analyze historical sales
- Forecast future demand
- Compare forecasting models
- Surface operational attention areas in a dashboard

## Slide 4: System Architecture
Dataset → Data Cleaning → Exploratory Analysis → Feature Engineering → Prophet and XGBoost → Evaluation → Best Model → Forecast → Dashboard → Alerts

The current repository implements the cleaning, Prophet, SARIMA benchmark, evaluation, and dashboard layers. XGBoost integration is not present in the current source tree.

## Slide 5: Dashboard Features
- Interactive store, product, and date filters
- KPI cards for sales and evaluation metrics
- Daily, weekly, and monthly historical views
- Forecast horizon and model controls
- Model comparison, data tables, downloads, and alerts

## Slide 6: Model Comparison
The available evaluation report compares the following hold-out results:

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| Prophet (Default Baseline) | 5.25 | 6.81 | 11.64% |
| Prophet (Tuned) | 4.95 | 6.26 | 10.75% |
| SARIMA (1,1,1)x(1,1,1,7) | 8.36 | 10.50 | 18.80% |

The dashboard selects the best model dynamically by RMSE, then MAE, then MAPE. The current report therefore selects Prophet (Tuned).

## Slide 7: Forecasting
Historical sales and future forecasts are shown together. Lower and upper bounds are displayed when the forecast export provides them. Supported horizons are 30, 60, and 90 days where output files exist.

## Slide 8: Stock Alerts
When an inventory column exists, the dashboard compares available stock with expected demand. Without inventory data, it uses a clearly labelled Demand-Based Attention Threshold and never presents forecast demand as actual stock.

## Slide 9: Technology Stack
Python, Pandas, NumPy, Plotly, Streamlit, Prophet, SARIMA, XGBoost, scikit-learn, and pytest.

## Slide 10: Results
The tuned Prophet model has the lowest reported RMSE (6.26), MAE (4.95), and MAPE (10.75%) among the available evaluated models.

## Slide 11: Limitations
- The source dataset has no inventory, price, promotion, or holiday business features.
- Forecast accuracy depends on historical data quality and coverage.
- XGBoost outputs are not currently available in the repository.
- External demand drivers are not represented.

## Slide 12: Future Improvements
- Add XGBoost forecast exports and integrated comparison
- Connect real-time sales and inventory systems
- Automate model retraining and monitoring
- Add promotion-aware and external-variable forecasting
- Deploy to a managed cloud environment