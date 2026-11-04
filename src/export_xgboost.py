"""Generate a per-series XGBoost forecast with approximate holdout-error bounds.

Run from the repository root: python -m src.export_xgboost --store 2 --item 1
"""
import argparse
from pathlib import Path

import pandas as pd

from src.forecast_intervals import add_prediction_bounds
from src.models.xgboost_model import RetailXGBoostForecaster


def export_forecast(store: int, item: int) -> Path:
    root = Path(__file__).resolve().parents[1]
    data = pd.read_csv(root / "data/processed/sales_clean.csv", parse_dates=["date"])
    data = data[(data.store == store) & (data.item == item)].sort_values("date")
    if len(data) < 180:
        raise ValueError("At least 180 historical observations are required.")
    model = RetailXGBoostForecaster()
    # Train before the holdout, then calibrate on a recursive 90-day forecast.
    model.evaluate_holdout(data, holdout_days=90, store=store, item=item)
    cutoff = data.date.max() - pd.Timedelta(days=89)
    calibration = model.predict_future(data[data.date < cutoff], 90, store, item)
    observed = calibration.merge(data, left_on="ds", right_on="date")
    model.fit(data, store=store, item=item)
    forecast = add_prediction_bounds(
        model.predict_future(data, 90, store, item), observed.sales, observed.yhat,
    )
    destination = root / f"data/processed/forecasts/xgboost_s{store}_i{item}.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(destination, index=False)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=int, required=True)
    parser.add_argument("--item", type=int, required=True)
    args = parser.parse_args()
    print(export_forecast(args.store, args.item))
