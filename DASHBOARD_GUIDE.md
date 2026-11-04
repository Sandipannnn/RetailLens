# RetailLens Dashboard Guide

This document explains what each dashboard section means, how the forecast is
produced, how LSTM and XGBoost are used, and how inventory recommendations are
calculated.

## 1. What RetailLens does

RetailLens forecasts daily demand for one **store-item series** at a time.
The source data contains:

| Field | Meaning |
|---|---|
| `date` | The sales date |
| `store` | Numeric store identifier |
| `item` | Numeric product identifier |
| `sales` | Units sold on that date |

The original dataset covers 10 stores, 50 items, and daily records from 2013
through 2017. The dashboard uses numeric IDs with friendly demo labels such as
Store 1 and Item 1. These labels do not represent real store or product names.

## 2. Dashboard controls

### Store

Selects the store series to display.

### Product

Selects the product within the selected store. A forecast is trained separately
for each store-item combination.

### Forecast model

Selects which saved model forecast is displayed:

- **LSTM**
- **XGBoost**

The selection changes the displayed forecast and the inventory model request.
The backend's saved report identifies the best model using holdout metrics.

### Forecast horizon

The dashboard can display the first:

- 30 days
- 60 days
- 90 days

The training pipeline generates 90 days. Selecting 30 or 60 days only displays
the beginning of that saved 90-day forecast; it does not retrain the model.

### Date search

The date fields filter historical sales and historical comparison rows.
Click **Search** to apply the selected range. Editing a date field alone does
not reload data. Click **Reset** to restore the complete historical range.

The date search does not change the saved future forecast horizon or inventory
calculation.

## 3. Forecast dashboard cards

### Historical sales

The total number of units actually sold in the selected store-item series and
selected historical date range:

```text
Historical sales = sum(actual sales rows)
```

The smaller value below it is the number of daily records included.

### Forecast demand

When forecast rows are available, this is the total predicted demand for the
selected horizon:

```text
Forecast demand = forecast_day_1 + forecast_day_2 + ... + forecast_day_N
```

For example, if the 30 daily predictions sum to 1,350:

```text
Forecast demand = 1,350 units
```

If no future forecast is available, the card displays recent actual sales
instead of forecast demand.

### Daily average

For a forecast:

```text
Daily average = forecast demand / number of forecast days
```

For historical data:

```text
Historical daily average = total actual sales / number of observations
```

### Selected model

Shows the model currently selected in the dashboard. The report below it
contains the model's chronological holdout evaluation, normally using 90 days.

## 4. Forecast chart and table

### Chart

The chart displays:

- **Actual sales**: recorded historical units
- **Model forecast**: predicted future units
- **Uncertainty range**: approximate lower and upper forecast bounds

The chart shows the latest 90 historical days to keep the graph readable. The
table can show all historical rows in the selected date range.

### Table columns

| Column | Meaning |
|---|---|
| Date | Date represented by the row |
| Period | `Historical` for recorded sales or `Forecast` for predictions |
| Actual sales | Units actually recorded in the source data |
| Forecast demand | Model prediction for that date |
| Lower bound | Approximate lower prediction range |
| Upper bound | Approximate upper prediction range |

### Why can a row show both actual sales and forecast demand?

Historical comparison rows can contain both values. For example:

```text
Actual sales:      45
Forecast demand: 45.7
```

Here, 45 is what actually sold and 45.7 is what the model estimated for that
historical date. This is used to evaluate model accuracy. It is not the total
future demand.

Future forecast rows have a prediction but no actual sales yet:

```text
Actual sales:      Not yet recorded
Forecast demand:  45.7
```

The future forecast begins after the last recorded date.

### Lower and upper bounds

Bounds are calibrated from historical holdout errors:

```text
absolute error = |actual sales - predicted sales|
width = 90th percentile of holdout absolute errors

lower bound = max(0, forecast - width)
upper bound = forecast + width
```

These are approximate planning ranges, not guaranteed statistical confidence
intervals.

## 5. Sales History section

The Sales History page shows the recorded daily sales rows.

### Total units

```text
Total units = sum of actual sales in the selected date range
```

### Daily average

```text
Daily average = total units / observations
```

### Observations

The number of daily sales records returned for the selected store, product, and
date range. An observation is one date's sales value.

## 6. How the models work

RetailLens trains two different forecasting approaches and compares them.

### LSTM

LSTM means Long Short-Term Memory. It is a neural network designed for
sequences.

This project uses:

- The previous 35 days of sales as input
- A MinMaxScaler fitted only on training data
- An LSTM layer followed by dense layers
- Mean squared error training loss
- Early stopping to reduce overfitting

Conceptually:

```text
last 35 sales values -> LSTM -> next-day demand prediction
```

For a 90-day forecast, the first prediction is fed back into the input window
to produce the next prediction. This is called recursive forecasting.

### XGBoost

XGBoost is a gradient-boosted decision-tree regression model. It receives
structured features for each date:

#### Calendar features

- Day of week
- Day of month
- Week of year
- Month
- Quarter
- Year
- Weekend indicator

#### Lag features

- Sales one day earlier: `lag_1`
- Sales seven days earlier: `lag_7`
- Sales 14 days earlier: `lag_14`
- Sales 28 days earlier: `lag_28`

#### Rolling features

- Seven-day rolling mean and standard deviation
- 14-day rolling mean and standard deviation
- 28-day rolling mean and standard deviation

Rolling features use only earlier days, so the current day's sales are not
leaked into its own features.

Conceptually:

```text
calendar + lag + rolling features -> XGBoost -> next-day demand prediction
```

For future dates, XGBoost predicts one day at a time. Each prediction is added
to the history used to build the next day's lag and rolling features.

## 7. Why XGBoost may be better than LSTM

XGBoost is not always better. The project selects the better model for each
series using measured holdout performance:

1. Reserve the final chronological holdout period.
2. Train/evaluate LSTM on that period.
3. Train/evaluate XGBoost on that period.
4. Choose the lower RMSE.
5. Use MAE as the tie-breaker.

Lower values are better:

- **MAE**: average absolute prediction error
- **RMSE**: error metric that penalizes large mistakes more strongly
- **MAPE**: percentage error, where applicable

XGBoost can perform better in this project because the sales data has strong
calendar and short-term lag patterns, and the engineered features expose those
patterns directly. It also trains and serves faster and usually needs less data
and tuning than a neural network.

LSTM can be better when long sequence patterns are important and sufficient
training data and tuning are available. The dashboard should therefore trust
the measured holdout result, not assume that one algorithm is always best.

## 8. Inventory planning

The Inventory page uses a saved future forecast and the current stock entered
by the user. It does not train a new model.

Inputs:

- Current stock: units currently available
- Lead time: days between placing and receiving an order
- Order cycle: days of demand covered by one replenishment cycle

The recommendation is calculated only after clicking **Calculate
recommendation**.

### Lead-time demand

Expected demand while waiting for the order:

```text
Lead-time demand = sum of the first lead_time_days forecast values
```

With a seven-day lead time, the first seven forecast days are summed.

### Safety stock

Safety stock protects against forecast variability:

```text
sigma = standard deviation of lead-time forecast values
safety stock = z × sigma × sqrt(lead time days)
```

The default service level is 95%, which uses:

```text
z = 1.645
```

The current implementation uses forecast variability as a proxy for demand
variability. It should be validated against real operational demand and lead
time data before being used for purchasing.

### Reorder point

```text
Reorder point = lead-time demand + safety stock
```

If current stock is at or below the reorder point, the dashboard shows a
low-stock/reorder alert.

### Cycle stock

```text
average daily demand = total forecast demand / forecast horizon
cycle stock = average daily demand × order cycle days
```

### Suggested order quantity

The recommendation replenishes stock to the reorder point plus one cycle:

```text
target stock = reorder point + cycle stock
suggested order = max(0, target stock - current stock)
```

The result is never negative.

### Overstock alert

The dashboard marks overstock when:

```text
current stock > reorder point + 2 × cycle stock
```

This is a planning alert, not an automatic purchase or disposal instruction.

## 9. Data and API flow

```text
Sales CSV
   |
   v
Training pipeline
   |
   +--> LSTM model
   +--> XGBoost model
   |
   v
Holdout comparison and model selection
   |
   v
Saved forecast CSV files and model reports
   |
   v
FastAPI
   |
   v
React/Vite dashboard
```

The API reads saved artifacts. It does not retrain models during normal
dashboard requests.

Important endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /health` | Check API availability |
| `GET /sales/options` | Load stores, products, and date bounds |
| `GET /sales` | Load historical sales |
| `GET /sales/summary` | Load totals, averages, and observation counts |
| `GET /sales/comparison` | Load historical actual-versus-predicted values |
| `GET /forecasts` | Load saved future forecasts |
| `GET /models/comparison` | Load LSTM/XGBoost holdout metrics |
| `GET /inventory/recommendation` | Calculate inventory guidance |

## 10. Limitations

- The source dataset has sales but no actual inventory, price, promotion, or
  holiday fields.
- Current stock must be entered manually.
- Forecast bounds are approximate error-based ranges.
- Recursive multi-day forecasts can accumulate errors.
- Forecast results from synthetic fallback data are not Kaggle results.
- Inventory recommendations are planning guidance and require operational
  validation.

## 11. Run the project

From the repository root, start the API:

```powershell
python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

In another terminal, start the frontend:

```powershell
cd frontend
npm run dev -- --host 127.0.0.1
```

Open `http://127.0.0.1:5173`.

This project uses time-series retail sales data. Each row represents the daily sales of one product in one store.

| Column | Data Type | Example | Meaning |
|---|---|---|---|
| date | Date | 2017-12-31 | Sales date |
| store | Integer | 1 | Store ID |
| item | Integer | 1 | Product ID |
| sales | Numeric / Integer | 45 | Units sold |


## 1. Units Sold

**Units sold** means the number of pieces of a selected product sold on a specific date at a specific store.

### Example

| Date | Store | Product | Units Sold |
|---|---|---|---:|
| Dec 31, 2017 | Store 1 | Item 1 | 45 |

This means **45 units of Item 1 were sold at Store 1 on Dec 31, 2017**.

In the project, it comes from the `sales` column:

- `date` → Sales date
- `store` → Store ID
- `item` → Product ID
- `sales` → Number of units sold

It does not mean revenue or price. It is only the quantity of products sold.

---

## 2. Observations

**Observations** means the total number of daily sales records in the selected store, product, and date range.

### Example

- Start Date: 2013-01-01
- End Date: 2017-12-31
- Store: Store 1
- Product: Item 1
- Total Observations: 1,826

This means the dataset contains **1,826 recorded days of sales**.

Each observation represents one row, for example:

| Date | Store | Product | Units Sold |
|---|---|---|---:|
| Jan 1, 2013 | Store 1 | Item 1 | 12 |
| Jan 2, 2013 | Store 1 | Item 1 | 15 |
| Jan 3, 2013 | Store 1 | Item 1 | 18 |
| ... | ... | ... | ... |
| Dec 31, 2017 | Store 1 | Item 1 | 45 |

It does **not** mean total units sold. It means the number of records/days included.



## 1. MAE — Mean Absolute Error

**MAE shows the average number of units by which the model's predictions differ from actual sales.**

### Example

| Actual | Predicted | Absolute Error |
|---:|---:|---:|
| 50 | 45 | 5 |
| 40 | 44 | 4 |
| 60 | 65 | 5 |

### Calculation

MAE = average(|actual sales - predicted sales|)

MAE = (5 + 4 + 5) / 3

MAE = **4.67 units**

**Meaning:** The model's predictions are wrong by an average of 4.67 units.

---

## 2. RMSE — Root Mean Squared Error

**RMSE is similar to MAE, but it gives more penalty to large forecasting errors.**

### Example

| Actual | Predicted | Absolute Error |
|---:|---:|---:|
| 50 | 45 | 5 |
| 40 | 44 | 4 |
| 60 | 80 | 20 |

The 20-unit error is much larger, so RMSE increases significantly.

### Why RMSE is Important for RetailLens

A very large forecasting mistake can cause:

- Stockouts
- Excess inventory
- Poor reorder decisions

The project uses **RMSE as the primary model-selection metric**.

### Formula

RMSE = sqrt(average((actual sales - predicted sales)^2))

### Calculation

RMSE = sqrt((5² + 4² + 20²) / 3)

RMSE = sqrt((25 + 16 + 400) / 3)

RMSE = sqrt(147)

RMSE = **12.12 units**

**Meaning:** RMSE penalizes the large 20-unit error more heavily than MAE.

---

## 3. MAPE — Mean Absolute Percentage Error

**MAPE shows the average prediction error as a percentage of actual sales.**

### Example

| Actual | Predicted | Percentage Error |
|---:|---:|---:|
| 50 | 45 | 10% |
| 40 | 44 | 10% |
| 60 | 54 | 10% |

### Calculation

MAPE = average(|actual - predicted| / actual) × 100

MAPE = (10% + 10% + 10%) / 3

MAPE = **10%**

**Meaning:** The model's predictions are approximately 10% away from actual sales on average.

MAPE is useful because it is easier to understand and compare across products with different sales volumes.

**Note:** MAPE is undefined when actual sales are zero.
