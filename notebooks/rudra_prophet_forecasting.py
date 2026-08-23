"""
RetailLens — Prophet Time-Series Forecasting & Analysis Script.
Owner: Rudra Pratap Singh (Time-Series Forecasting Track)

Demonstrates:
1. Data loading & series inspection
2. Prophet model fitting with 30, 60, 90 days multi-step forecasting
3. Visualizing forecast with confidence intervals
4. Trend & Seasonality component decomposition
5. SARIMA comparison
"""

import os
import sys
import matplotlib.pyplot as plt
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.clean_data import clean_data, PROCESSED_PATH
from src.models.prophet_model import RetailProphetForecaster
from src.models.sarima_model import RetailSARIMAForecaster
from src.models.hyperparameter_tuner import ProphetTuner


def main():
    if not os.path.exists(PROCESSED_PATH):
        clean_data()

    df = pd.read_csv(PROCESSED_PATH)
    store_id = 1
    item_id = 1

    print(f"=== Running Prophet Analysis for Store {store_id}, Item {item_id} ===")

    # 1. Fit Prophet model
    forecaster = RetailProphetForecaster(
        seasonality_mode="additive",
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        interval_width=0.95,
    )
    forecaster.fit(df, store=store_id, item=item_id)

    # 2. Multi-step forecasts (30, 60, 90 days)
    forecast_90d = forecaster.predict(horizon_days=90, include_history=False)
    print("\nNext 10 Days Forecast Sample:")
    print(forecast_90d[["ds", "yhat", "yhat_lower", "yhat_upper"]].head(10))

    # 3. Horizon totals
    horizons = forecaster.forecast_horizons([30, 60, 90])
    for h, h_df in horizons.items():
        print(f"\n{h}-Day Horizon:")
        print(f"  Total Expected Units: {h_df['yhat'].sum():.1f}")
        print(f"  95% CI Range: [{h_df['yhat_lower'].sum():.1f}, {h_df['yhat_upper'].sum():.1f}]")

    # 4. Holdout evaluation
    metrics, holdout = forecaster.evaluate_holdout(df, holdout_days=90, store=store_id, item=item_id)
    print("\n90-Day Holdout Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}")

    # 5. Extract components
    components = forecaster.extract_components(forecast_90d)
    print(f"\nExtracted Components: {list(components.keys())}")

    print("\nProphet Forecasting pipeline run completed successfully!")


if __name__ == "__main__":
    main()
