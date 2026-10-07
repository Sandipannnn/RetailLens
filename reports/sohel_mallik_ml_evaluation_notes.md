# Sohel Mallik (BWU/BTS/24/157)

## Machine Learning & Evaluation

This note covers only the Machine Learning & Evaluation contribution:

- Build an XGBoost baseline model with lag features.
- Split the data into chronological training and hold-out test sets.
- Calculate MAE, RMSE, and MAPE.
- Compare model results and select the best-performing model.

It does not describe the EDA, LSTM, Prophet implementation details, inventory logic, or Streamlit UI except where those parts consume this track's outputs.

---

## 1. Contribution Summary

The ML pipeline predicts daily retail sales for a selected store-item series. It converts historical sales into supervised-learning features, trains an `XGBRegressor`, evaluates predictions on the final hold-out period, calculates error metrics, exports the results, and compares XGBoost with Prophet.

The complete flow is:

```text
sales_clean.csv
    -> select one store and item
    -> create calendar, lag, and rolling features
    -> remove rows without enough historical feature values
    -> chronological train / hold-out split
    -> train XGBoost on the training period
    -> predict the hold-out period
    -> calculate MAE, RMSE, MAPE
    -> compare XGBoost and Prophet
    -> select the lowest-error model
    -> save model, predictions, metrics, reports, and charts
```

---

## 2. Files and Folders for This Part

### Main source files

| Path | Role in Machine Learning & Evaluation |
| --- | --- |
| `src/features/feature_engineering.py` | Creates calendar features, lag features, rolling features, the final feature table, and the ordered feature-column list. |
| `src/models/xgboost_model.py` | Defines `RetailXGBoostForecaster`, trains XGBoost, performs chronological hold-out evaluation, clips negative predictions to zero, calculates metrics, saves/loads the model, and returns feature importance. |
| `src/models/ml_evaluation.py` | Runs the complete ML evaluation pipeline, audits leakage, evaluates XGBoost and Prophet on the same hold-out period, compares metrics, selects the best model, and writes output files. |
| `src/utils/metrics.py` | Implements MAE, RMSE, MAPE, sMAPE, and the `calculate_metrics()` helper used by the model evaluation. |
| `src/models/model_selector.py` | Reusable selector for comparing registered models. Its default selection rule is lowest RMSE, then lowest MAE. |
| `src/clean_data.py` | Supplies the cleaned input dataset when the processed file is missing. This is an upstream dependency of the ML pipeline. |
| `src/data_generator.py` | Generates deterministic sample data used by the tests. It is not the production training dataset. |

### Test files

| Path | What it verifies for this part |
| --- | --- |
| `tests/test_ml_evaluation.py` | Tests lag correctness, no cross-series leakage, rolling-feature leakage prevention, chronological splitting, XGBoost training/prediction, non-negative finite predictions, model persistence, metric calculations, and dynamic best-model selection. |
| `tests/test_forecasting.py` | Related forecasting tests that may exercise shared forecasting behavior. |
| `tests/test_new_architecture.py` | Additional project architecture/integration tests. |

### Input data folders and files

| Path | Use |
| --- | --- |
| `data/processed/sales_clean.csv` | Main cleaned sales input. Expected columns are `date`, `store`, `item`, and `sales`. |
| `data/raw/train.csv` | Original raw sales data. It is cleaned upstream and is not used directly by the XGBoost feature builder. |
| `data/processed/` | Folder containing the cleaned dataset and generated forecasting assets. |
| `data/` | Project data root. |

### Generated ML output folders and files

| Path | Generated result |
| --- | --- |
| `models/xgboost_model.pkl` | Persisted trained XGBoost model created by `ml_evaluation.py`. |
| `data/processed/models/xgb_s1_i1.joblib` | Existing XGBoost model artifact for Store 1, Item 1. |
| `outputs/predictions/xgboost_test_predictions.csv` | Hold-out dates, actual sales, and XGBoost predicted sales. |
| `outputs/metrics/xgboost_feature_importance.csv` | Feature names and their XGBoost importance scores. |
| `outputs/metrics/model_comparison.csv` | Model-by-model MAE, RMSE, MAPE, training time, and notes. |
| `outputs/metrics/model_comparison.json` | Machine-readable metrics, selected model, split dates, and record counts. |
| `reports/ts_model_comparison.csv` | Shared comparison report containing model metrics. |
| `reports/ts_model_comparison.json` | Shared JSON comparison report and best-model metadata. |
| `outputs/figures/xgboost_actual_vs_predicted.png` | Chart comparing actual and predicted hold-out sales. |
| `outputs/figures/model_comparison.png` | Chart comparing MAE, RMSE, and MAPE between models. |
| `outputs/figures/` | Folder for generated evaluation charts. |
| `outputs/metrics/` | Folder for generated metric and feature-importance files. |
| `outputs/predictions/` | Folder for generated prediction files. |
| `reports/` | Folder for generated evaluation reports and this note. |
| `models/` | Folder for persisted trained models. |

### Configuration and dependency files

| Path | Relevance |
| --- | --- |
| `requirements.txt` | Includes `pandas`, `numpy`, `xgboost`, `scikit-learn`, `joblib`, `matplotlib`, `pytest`, and other project dependencies. |
| `pyrightconfig.json` | Python type-checking configuration. |
| `README.md` | Project-level setup, architecture, commands, and testing documentation. |
| `agent.md` | Project ownership matrix and engineering/testing guidance. |

---

## 3. Feature Engineering

The input schema is:

```text
date  : observation date
store : store identifier
item  : product identifier
sales : target value to predict
```

`build_features()` in `src/features/feature_engineering.py` creates three feature groups.

### 3.1 Calendar features

The date is converted to datetime and transformed into:

- `day_of_week`
- `day_of_month`
- `week_of_year`
- `month`
- `quarter`
- `year`
- `is_weekend`

These help XGBoost learn weekly, monthly, quarterly, yearly, and weekend demand patterns.

### 3.2 Lag features

The default lag configuration is:

```python
LAG_DAYS = [1, 7, 14, 28]
```

The generated columns are:

- `lag_1`: sales one day earlier
- `lag_7`: sales seven days earlier
- `lag_14`: sales fourteen days earlier
- `lag_28`: sales twenty-eight days earlier

Lags are calculated separately for each `store` and `item` group. Therefore, values from one store-item series cannot enter another series.

### 3.3 Rolling features

The default rolling windows are:

```python
ROLLING_WINDOWS = [7, 14, 28]
```

The generated columns are:

- `rolling_mean_7`, `rolling_mean_14`, `rolling_mean_28`
- `rolling_std_7`, `rolling_std_14`, `rolling_std_28`

The target is shifted by one row before rolling statistics are calculated. For a date `T`, the rolling values use observations before `T`, never `sales[T]`. This prevents target leakage.

### 3.4 Rows removed before training

The first rows in each series do not have enough history for all lag and rolling features. These rows contain `NaN` values and are removed with:

```python
featured.dropna(subset=feature_cols)
```

The model is trained only on complete feature rows.

---

## 4. XGBoost Baseline Model

The class `RetailXGBoostForecaster` is implemented in `src/models/xgboost_model.py`.

The baseline uses `xgboost.XGBRegressor` with these main defaults:

```python
n_estimators = 500
max_depth = 6
learning_rate = 0.05
subsample = 0.8
colsample_bytree = 0.8
min_child_weight = 5
reg_alpha = 0.1
reg_lambda = 1.0
random_state = 42
objective = "reg:squarederror"
```

These settings provide a reproducible gradient-boosted tree baseline. The model receives the calendar, store/item, lag, and rolling columns returned by `get_feature_columns()`.

Training steps in `fit()`:

1. Copy the cleaned DataFrame.
2. Optionally filter to one `store` and one `item`.
3. Build calendar, lag, and rolling features.
4. Drop incomplete feature rows.
5. Set `X` to the feature columns and `y` to `sales`.
6. Fit `XGBRegressor`.
7. Mark the forecaster as fitted.

---

## 5. Train and Hold-Out Test Split

The project uses a chronological split because sales forecasting must not train on future observations.

For a hold-out period of `90` days:

```python
max_date = data["date"].max()
split_date = max_date - pd.Timedelta(days=holdout_days - 1)

train = data[data["date"] < split_date]
test = data[data["date"] >= split_date]
```

The default command evaluates Store 1 and Item 1 over the last 90 days:

```bash
python src/models/ml_evaluation.py --store 1 --item 1 --holdout-days 90
```

The split is performed by date, not by random sampling. The training period must end before the test period starts:

```text
max(training date) < min(test date)
```

### Why the hold-out is valid

Features are constructed using prior observations. A test row may use historical sales from before that row's date for its lag and rolling features, but it does not use its own future target value. The pipeline also runs `audit_leakage()` and stops if the chronological condition fails.

---

## 6. Evaluation Metrics

The metrics are implemented in `src/utils/metrics.py` and called through `calculate_metrics()`.

### MAE: Mean Absolute Error

```text
MAE = mean(abs(actual - predicted))
```

MAE is the average absolute prediction error in sales units. Lower is better. It is easy to explain: an MAE of `5` means the prediction is off by about five sales units on average.

### RMSE: Root Mean Squared Error

```text
RMSE = sqrt(mean((actual - predicted)^2))
```

RMSE is also measured in sales units, but it penalizes large errors more strongly than MAE. Lower is better.

### MAPE: Mean Absolute Percentage Error

```text
MAPE = mean(abs((actual - predicted) / actual)) * 100
```

The implementation replaces an actual value close to zero with a small epsilon (`1e-5`) so the result remains finite. MAPE is reported as a percentage. Lower is better.

The evaluation returns these required keys:

```python
{
    "MAE": ...,
    "RMSE": ...,
    "MAPE": ...,
    "sMAPE": ...,
}
```

`sMAPE` is also calculated for reporting, but the required metrics for this contribution are MAE, RMSE, and MAPE.

---

## 7. Best-Performing Model Selection

`src/models/ml_evaluation.py` evaluates XGBoost and Prophet on the same hold-out window and creates a metrics dictionary:

```python
model_metrics = {
    "Prophet": {"MAE": ..., "RMSE": ..., "MAPE": ...},
    "XGBoost": {"MAE": ..., "RMSE": ..., "MAPE": ...},
}
```

The function `select_best_model()` ranks models using:

1. Lowest MAE.
2. Lowest RMSE if MAE is tied.
3. Lowest MAPE if both MAE and RMSE are tied.

The winner is calculated from the current evaluation values; it is not hard-coded. The output includes:

- `best_model`
- `best_model_reason`
- each model's MAE, RMSE, and MAPE

There is also a reusable `ModelSelector` in `src/models/model_selector.py`. Its default API uses RMSE as the primary metric and MAE as the tie-breaker. For the command-line ML evaluation pipeline described in this note, the active selection function is `select_best_model()` in `src/models/ml_evaluation.py`, whose primary metric is MAE.

---

## 8. How to Run and Verify

From the project root:

```bash
pip install -r requirements.txt
python src/models/ml_evaluation.py --store 1 --item 1 --holdout-days 90
python -m pytest tests/test_ml_evaluation.py -v
```

A successful run should:

- load `data/processed/sales_clean.csv`;
- select the requested store-item series;
- build lag, rolling, and calendar features;
- print the train/test date ranges;
- pass the leakage audit;
- train and evaluate XGBoost;
- print MAE, RMSE, and MAPE;
- write predictions, metrics, feature importance, reports, and figures;
- report the dynamically selected best model.

The focused tests verify that:

- `lag_1` equals the previous sales value within a series;
- different store-item series do not share lag values;
- rolling features exclude the current target value;
- train and test dates do not overlap;
- XGBoost trains successfully;
- predictions are finite and non-negative;
- MAE, RMSE, and MAPE are finite;
- model selection changes when the input metric values change.

---

## 9. Explanation for Presentation or Viva

### Short explanation

"My responsibility was the Machine Learning and Evaluation part of RetailLens. I implemented an XGBoost baseline for daily retail sales forecasting. First, I generated calendar, lag, and rolling features from the cleaned sales data. The lag features use previous sales values such as one, seven, fourteen, and twenty-eight days earlier, calculated separately for every store-item series. I then split each time series chronologically, keeping the final 90 days as an unseen hold-out test period. XGBoost was trained only on the earlier period and evaluated on the hold-out period. I calculated MAE, RMSE, and MAPE, where lower values indicate better performance. Finally, I compared XGBoost with Prophet and selected the best model using the evaluation metrics rather than hard-coding a winner."

### What to say if asked about leakage

"I avoided leakage in two ways. First, I used a chronological split instead of a random split. Second, lag and rolling features are based only on previous observations. Rolling features shift the sales series before calculation, so the current day's target cannot enter its own feature values. The pipeline also checks that the latest training date is earlier than the earliest test date."

### What to say if asked about the metrics

"MAE gives the average absolute error in sales units. RMSE gives more weight to large errors. MAPE expresses the average error as a percentage, and the implementation uses an epsilon for zero actual values so it does not produce infinite or undefined results."

### What to say if asked how the best model is selected

"Both models are evaluated on the same hold-out dates. The pipeline ranks them by lower MAE, then lower RMSE, then lower MAPE for ties. Therefore, the selected model depends on the measured results for the selected store, item, and hold-out period."

---

## 10. Scope Boundary

Owned by Sohel Mallik:

- `src/features/feature_engineering.py`
- `src/models/xgboost_model.py`
- `src/models/ml_evaluation.py`
- `src/utils/metrics.py`
- relevant ML tests in `tests/test_ml_evaluation.py`
- XGBoost evaluation outputs under `models/`, `outputs/`, and `reports/`

Used as dependencies but not the main ownership area:

- `src/clean_data.py`
- `data/processed/sales_clean.csv`
- Prophet implementation files
- LSTM implementation files
- `app/streamlit_app.py`
- inventory and dashboard modules
