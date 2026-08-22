# RetailLens — Retail Sales Forecasting & Inventory Dashboard

A predictive time-series system that analyzes historical retail sales, seasonality, and demand
patterns to project future inventory needs — with multi-step forecasts, confidence intervals,
and low-stock alerts surfaced through an interactive dashboard.

## Overview

Retailers need to know not just what sold yesterday, but what's likely to sell next week and
next month, per store and per item. This project builds and compares two forecasting
approaches (Prophet and XGBoost), evaluates them rigorously, and exposes the results through a
Streamlit dashboard that a non-technical store manager could actually use.

## Objectives

- Clean and structure raw daily sales data into an analysis-ready dataset.
- Understand seasonality, trend, and demand patterns through EDA.
- Build a classical time-series model (Prophet) and a machine-learning model (XGBoost) and
  compare them on equal footing.
- Generate multi-step forecasts (30/60/90 days) with honest uncertainty estimates.
- Turn low-stock risk into a concrete, explainable alert rather than a flat threshold.
- Ship an interactive dashboard, not just a notebook.

## Dataset

**Source:** [Store Item Demand Forecasting Challenge](https://www.kaggle.com/competitions/demand-forecasting-kernels-only) (Kaggle)

| | |
|---|---|
| Records | ~913,000 daily rows |
| Coverage | 10 stores × 50 items = 500 time series |
| Date range | 2013–2017 |
| Columns | `date`, `store`, `item`, `sales` |
| Known gap | No price, promotion, or holiday data |

Because there's no promo/holiday data, seasonality analysis (day-of-week, monthly, yearly
cycles) is the main lever for EDA insight rather than promo-impact analysis.

## Methodology

**Pipeline:** Data Collection → Cleaning → EDA → Feature Engineering → Modeling → Evaluation →
Best Model Selection → Dashboard

1. **Cleaning** — datetime conversion, duplicate/missing checks, sorted by store/item/date.
2. **Feature engineering** — lag features (t-1, t-7, t-28), rolling mean/std windows, calendar
   features (day-of-week, month, is_weekend).
3. **Prophet model** — per-series or aggregated forecasts with built-in confidence intervals,
   validated with Prophet's rolling-origin cross-validation.
4. **XGBoost model** — trained on lag/rolling features with `TimeSeriesSplit` cross-validation
   and light Optuna tuning; quantile objective used for its own prediction intervals so it's
   comparable to Prophet on more than just point accuracy.
5. **Hybrid (stretch goal)** — Prophet handles trend/seasonality, XGBoost models the residuals
   using the engineered features. Typically outperforms either model alone.
6. **Explainability** — SHAP values on the XGBoost model to surface which features (recent
   lags, day-of-week, etc.) actually drive predictions.
7. **Evaluation** — MAE, RMSE, and MAPE on a hold-out test set, computed overall and
   per-store/per-item, not just in aggregate.

## Dashboard Features

- Store / product / date-range filters
- Historical sales alongside forecast charts, with confidence bands
- Prophet vs. XGBoost toggle so the model comparison is visible, not just a metric in a report
- Low-stock alerts based on a reorder-point calculation (lead-time demand + safety buffer from
  forecast uncertainty), not a flat threshold
- Cached data loading (`st.cache_data`) so filtering doesn't re-run the full pipeline

## Tech Stack

`pandas` · `numpy` · `Matplotlib` / `Plotly` · `Prophet` · `XGBoost` · `scikit-learn` ·
`statsmodels` · `SHAP` · `Optuna` · `Streamlit`

## Team & Roles

| Member | Role |
|---|---|
| Sandipan Biswas | Project Lead & Data Engineer |
| Soumili Das | EDA & Visualization |
| Rudra Pratap Singh | Time-Series Forecasting (Prophet) |
| Sohel Mallik | Machine Learning & Evaluation (XGBoost) |
| Jeet Jana | Dashboard & Deployment (Streamlit) |

**Review/dependency order:** Data Engineer → EDA → Forecasting → ML/Evaluation → Dashboard —
each stage depends on the frozen output of the one before it.

## Project Structure

```
project-root/
├── data/
│   ├── raw/          # untouched downloaded files
│   └── processed/    # cleaned, frozen dataset (single source of truth)
├── notebooks/         # one subfolder or filename prefix per person
├── src/                # reusable scripts (cleaning, features, models)
├── reports/            # final report, PPT drafts
└── README.md
```

## Setup

```bash
git clone <repo-url>
cd project-root
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Download the dataset (requires Kaggle API credentials)
kaggle competitions download -c demand-forecasting-kernels-only
unzip demand-forecasting-kernels-only.zip -d data/raw
```

## Roadmap

| Week | Task |
|---|---|
| 1 | Dataset selection, scope decision, project planning |
| 1 | Understand business problem |
| 2 | Data cleaning and preprocessing |
| 2 | Initial EDA |
| 3 | Detailed visual analysis |
| 3–4 | Prophet model |
| 4 | XGBoost model |
| 5 | Model evaluation, comparison, hybrid experiment |
| 6 | Streamlit dashboard |
| 6 | Low-stock alert logic |
| 7 | Integration and testing |
| 8 | Report, PPT, demo, deployment |

## Evaluation

Models are compared using **MAE**, **RMSE**, and **MAPE** on a hold-out test set, with results
logged to a shared sheet as each model finishes so the Week 5 comparison doesn't require
re-running everything from scratch.

## Deployment

Final dashboard deployed to Streamlit Community Cloud or Hugging Face Spaces for a live demo
link, rather than demoed locally only.

## Contributing

- Branch naming: `feature/<name>-<task>` — e.g. `feature/soumili-eda`, `feature/rudra-prophet`
- Open a PR into `main` for review rather than pushing directly
- Keep `requirements.txt` updated when adding a new dependency

## Future Scope

- Extend to hierarchical forecasting (store → region → national rollups)
- Automated retraining pipeline as new data arrives
- Anomaly detection layer to flag stockout-driven sales dips before they bias training
