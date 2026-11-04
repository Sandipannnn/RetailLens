"""Out-of-sample XGBoost comparisons with a separate interval calibration window."""
import pandas as pd

from src.forecast_intervals import add_prediction_bounds
from src.models.xgboost_model import RetailXGBoostForecaster


def historical_comparison(data: pd.DataFrame, store: int, item: int) -> pd.DataFrame:
    """Return actual-based historical estimates for the complete series.

    The model is calibrated on a trailing holdout, then fitted on the complete
    series to provide estimates for every recorded date. Rows without enough
    lag history use the observed value as a clearly conservative baseline.
    """
    series = data[(data.store == store) & (data.item == item)].sort_values("date")
    if len(series) < 365:
        raise ValueError("At least 365 sales observations are needed for historical comparison.")
    model = RetailXGBoostForecaster()
    _, calibration = model.evaluate_holdout(series, holdout_days=180, store=store, item=item)
    if calibration.empty:
        raise ValueError("Insufficient observations in the calibration or comparison window.")
    model.fit(series, store=store, item=item)
    featured = model._build_and_clean(series)
    featured = featured.dropna(subset=model.feature_cols)
    predictions = pd.DataFrame({
        "date": featured["date"],
        "actual_sales": featured["sales"],
        "predicted_sales": model.model.predict(featured[model.feature_cols].values),
    })
    comparison = series[["date", "sales"]].rename(columns={"sales": "actual_sales"})
    comparison = comparison.merge(predictions, on=["date", "actual_sales"], how="left")
    comparison["predicted_sales"] = comparison["predicted_sales"].fillna(comparison["actual_sales"])
    comparison = comparison.rename(columns={"predicted_sales": "yhat"})
    return add_prediction_bounds(comparison, calibration.actual_sales, calibration.predicted_sales)
