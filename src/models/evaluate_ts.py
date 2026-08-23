"""
Time-Series Model Evaluation & Comparison Pipeline for RetailLens.
Evaluates Prophet (Baseline vs Tuned) against SARIMA benchmark across multi-step horizons,
computes metrics (MAE, RMSE, MAPE, Coverage), and exports summary reports.

Owner: Rudra Pratap Singh (Time-Series Forecasting Track)
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.clean_data import clean_data, PROCESSED_PATH
from src.models.prophet_model import RetailProphetForecaster
from src.models.sarima_model import RetailSARIMAForecaster
from src.models.hyperparameter_tuner import ProphetTuner
from src.utils.metrics import calculate_metrics


def run_evaluation_pipeline(
    store_id: int = 1,
    item_id: int = 1,
    horizons: List[int] = [30, 60, 90],
    tune_prophet: bool = True,
    use_optuna: bool = False,
    compare_sarima: bool = True,
    data_path: str = PROCESSED_PATH,
    output_dir: str = "data/processed/forecasts",
    report_dir: str = "reports",
) -> pd.DataFrame:
    """
    Executes end-to-end evaluation pipeline on a specific store-item time series.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    if not os.path.exists(data_path):
        print(f"Cleaned dataset not found at {data_path}. Running clean_data()...")
        clean_data()

    print(f"\n=======================================================")
    print(f" RetailLens Time-Series Evaluation Pipeline")
    print(f" Target Series: Store {store_id}, Item {item_id}")
    print(f" Forecast Horizons: {horizons} days")
    print(f"=======================================================\n")

    df = pd.read_csv(data_path)
    series_data = df[(df["store"] == store_id) & (df["item"] == item_id)]
    print(f"Loaded series data: {len(series_data)} rows ({series_data['date'].min()} to {series_data['date'].max()})")

    results = []

    # -------------------------------------------------------------
    # 1. Baseline Prophet Model
    # -------------------------------------------------------------
    print("\n--- 1. Evaluating Baseline Prophet Forecaster ---")
    t0 = time.time()
    baseline_prophet = RetailProphetForecaster(
        seasonality_mode="additive",
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        interval_width=0.95,
    )
    bp_metrics, bp_holdout = baseline_prophet.evaluate_holdout(df, holdout_days=90, store=store_id, item=item_id)
    bp_fit_time = time.time() - t0

    # Fit on full data and generate multi-step forecasts
    baseline_prophet.fit(df, store=store_id, item=item_id)
    bp_forecasts = baseline_prophet.predict(horizon_days=max(horizons))
    bp_fc_path = os.path.join(output_dir, f"prophet_baseline_s{store_id}_i{item_id}.csv")
    baseline_prophet.save_forecast(bp_forecasts, bp_fc_path)

    results.append({
        "Model": "Prophet (Default Baseline)",
        "Store": store_id,
        "Item": item_id,
        "MAE": bp_metrics["MAE"],
        "RMSE": bp_metrics["RMSE"],
        "MAPE (%)": bp_metrics["MAPE"],
        "sMAPE (%)": bp_metrics["sMAPE"],
        "Coverage (%)": bp_metrics.get("Coverage", 0.0),
        "Avg Interval Width": bp_metrics.get("Avg_Interval_Width", 0.0),
        "Train Time (s)": round(bp_fit_time, 2),
        "Notes": "Additive, default priors",
    })
    print(f"Baseline Prophet: MAE={bp_metrics['MAE']:.2f}, RMSE={bp_metrics['RMSE']:.2f}, MAPE={bp_metrics['MAPE']:.2f}%, Coverage={bp_metrics.get('Coverage', 0.0):.1f}%")

    # -------------------------------------------------------------
    # 2. Hyperparameter Tuned Prophet Model
    # -------------------------------------------------------------
    best_prophet_model = baseline_prophet
    if tune_prophet:
        print("\n--- 2. Tuning Prophet Hyperparameters ---")
        tuner = ProphetTuner(metric="rmse")
        t0 = time.time()
        if use_optuna:
            best_params, best_score = tuner.tune_with_optuna(df, store=store_id, item=item_id, n_trials=15)
        else:
            best_params, trials_df = tuner.grid_search_cv(df, store=store_id, item=item_id, max_trials=16)
        tune_time = time.time() - t0
        print(f"Best Hyperparameters found ({tune_time:.1f}s): {best_params}")

        tuned_prophet = tuner.get_best_forecaster(interval_width=0.95)
        tp_metrics, tp_holdout = tuned_prophet.evaluate_holdout(df, holdout_days=90, store=store_id, item=item_id)

        tuned_prophet.fit(df, store=store_id, item=item_id)
        tp_forecasts = tuned_prophet.predict(horizon_days=max(horizons))
        tp_fc_path = os.path.join(output_dir, f"prophet_tuned_s{store_id}_i{item_id}.csv")
        tuned_prophet.save_forecast(tp_forecasts, tp_fc_path)
        best_prophet_model = tuned_prophet

        results.append({
            "Model": "Prophet (Tuned)",
            "Store": store_id,
            "Item": item_id,
            "MAE": tp_metrics["MAE"],
            "RMSE": tp_metrics["RMSE"],
            "MAPE (%)": tp_metrics["MAPE"],
            "sMAPE (%)": tp_metrics["sMAPE"],
            "Coverage (%)": tp_metrics.get("Coverage", 0.0),
            "Avg Interval Width": tp_metrics.get("Avg_Interval_Width", 0.0),
            "Train Time (s)": round(tune_time, 2),
            "Notes": str(best_params),
        })
        print(f"Tuned Prophet: MAE={tp_metrics['MAE']:.2f}, RMSE={tp_metrics['RMSE']:.2f}, MAPE={tp_metrics['MAPE']:.2f}%, Coverage={tp_metrics.get('Coverage', 0.0):.1f}%")

    # -------------------------------------------------------------
    # 3. SARIMA Benchmark Model
    # -------------------------------------------------------------
    if compare_sarima:
        print("\n--- 3. Evaluating SARIMA Benchmark Model ---")
        t0 = time.time()
        sarima_forecaster = RetailSARIMAForecaster(
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 7),
            interval_width=0.95,
        )
        sarima_metrics, sarima_holdout = sarima_forecaster.evaluate_holdout(df, holdout_days=90, store=store_id, item=item_id)
        sarima_fit_time = time.time() - t0

        sarima_forecaster.fit(df, store=store_id, item=item_id)
        sarima_forecasts = sarima_forecaster.predict(horizon_days=max(horizons))
        sarima_fc_path = os.path.join(output_dir, f"sarima_benchmark_s{store_id}_i{item_id}.csv")
        sarima_forecaster.save_forecast(sarima_forecasts, sarima_fc_path)

        results.append({
            "Model": "SARIMA (1,1,1)x(1,1,1,7)",
            "Store": store_id,
            "Item": item_id,
            "MAE": sarima_metrics["MAE"],
            "RMSE": sarima_metrics["RMSE"],
            "MAPE (%)": sarima_metrics["MAPE"],
            "sMAPE (%)": sarima_metrics["sMAPE"],
            "Coverage (%)": sarima_metrics.get("Coverage", 0.0),
            "Avg Interval Width": sarima_metrics.get("Avg_Interval_Width", 0.0),
            "Train Time (s)": round(sarima_fit_time, 2),
            "Notes": "Weekly seasonal state-space",
        })
        print(f"SARIMA: MAE={sarima_metrics['MAE']:.2f}, RMSE={sarima_metrics['RMSE']:.2f}, MAPE={sarima_metrics['MAPE']:.2f}%, Coverage={sarima_metrics.get('Coverage', 0.0):.1f}%")

    # -------------------------------------------------------------
    # 4. Multi-Horizon Breakdown
    # -------------------------------------------------------------
    print("\n--- 4. Multi-Step Forecast Breakdown (30, 60, 90 Days) ---")
    horizon_forecasts = best_prophet_model.forecast_horizons(horizons=horizons)
    for h, h_df in horizon_forecasts.items():
        total_projected_demand = h_df["yhat"].sum()
        lower_bound = h_df["yhat_lower"].sum()
        upper_bound = h_df["yhat_upper"].sum()
        print(f"Horizon {h:2d} Days: Total Expected Sales = {total_projected_demand:.0f} units (95% CI: [{lower_bound:.0f}, {upper_bound:.0f}])")

    # Save summary report
    summary_df = pd.DataFrame(results)
    csv_report = os.path.join(report_dir, "ts_model_comparison.csv")
    json_report = os.path.join(report_dir, "ts_model_comparison.json")

    summary_df.to_csv(csv_report, index=False)
    with open(json_report, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved evaluation comparison report to:")
    print(f"  - {csv_report}")
    print(f"  - {json_report}")
    print(f"\n{summary_df.to_string(index=False)}")

    return summary_df


def main():
    parser = argparse.ArgumentParser(description="RetailLens Time-Series Forecasting Evaluation Pipeline")
    parser.add_argument("--store", type=int, default=1, help="Store ID (default: 1)")
    parser.add_argument("--item", type=int, default=1, help="Item ID (default: 1)")
    parser.add_argument("--horizons", nargs="+", type=int, default=[30, 60, 90], help="Forecast horizons (e.g. 30 60 90)")
    parser.add_argument("--no-tune", action="store_true", help="Skip Prophet hyperparameter tuning")
    parser.add_argument("--optuna", action="store_true", help="Use Optuna for Bayesian tuning")
    parser.add_argument("--no-sarima", action="store_true", help="Skip SARIMA benchmark comparison")
    parser.add_argument("--data-path", type=str, default=PROCESSED_PATH, help="Path to processed sales CSV")
    parser.add_argument("--output-dir", type=str, default="data/processed/forecasts", help="Output directory for forecasts")
    parser.add_argument("--report-dir", type=str, default="reports", help="Output directory for reports")

    args = parser.parse_args()

    run_evaluation_pipeline(
        store_id=args.store,
        item_id=args.item,
        horizons=args.horizons,
        tune_prophet=not args.no_tune,
        use_optuna=args.optuna,
        compare_sarima=not args.no_sarima,
        data_path=args.data_path,
        output_dir=args.output_dir,
        report_dir=args.report_dir,
    )


if __name__ == "__main__":
    main()
