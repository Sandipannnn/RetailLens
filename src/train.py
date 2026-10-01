"""
RetailLens Model Training Pipeline.

Trains and evaluates:
  - LSTM (AI / Deep Learning) — src/models/lstm_model.py
  - XGBoost (Traditional ML)  — src/models/xgboost_model.py (RetailXGBoostForecaster)

Selects the best model by holdout RMSE (no hard-coding).
Saves all trained models + forecasts + report to disk.

Run from project root:
    python src/train.py
    python src/train.py --store 2 --item 5
    python src/train.py --store 1 --item 1 --force-retrain

IMPORTANT:
- Run this before launching the Streamlit dashboard.
- The dashboard loads saved models — it does NOT retrain on startup.
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import List

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.clean_data import clean_data, PROCESSED_PATH
from src.models.lstm_model import LSTMForecaster
from src.models.xgboost_model import RetailXGBoostForecaster
from src.models.model_selector import ModelSelector
from src.inventory import compute_inventory_recommendation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")

LSTM_MODEL_DIR = "data/processed/models"
XGB_MODEL_PATH_TPL = "data/processed/models/xgb_s{store}_i{item}.joblib"
FORECAST_DIR = "data/processed/forecasts"
REPORT_DIR = "reports"


def train_and_evaluate(
    store_id: int = 1,
    item_id: int = 1,
    holdout_days: int = 90,
    data_path: str = PROCESSED_PATH,
    force_retrain: bool = False,
) -> dict:
    """
    Full training + evaluation + model selection pipeline for one (store, item).

    Returns dict with: lstm_metrics, xgb_metrics, best_model_name, comparison_df
    """
    os.makedirs(LSTM_MODEL_DIR, exist_ok=True)
    os.makedirs(FORECAST_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    # Ensure clean data exists
    if not os.path.exists(data_path):
        logger.info("Cleaned data not found. Running clean_data()...")
        clean_data()

    logger.info("\n" + "=" * 60)
    logger.info(" RetailLens Training Pipeline  (LSTM + XGBoost)")
    logger.info(f" Target: Store {store_id}, Item {item_id}")
    logger.info(f" Holdout: {holdout_days} days")
    logger.info("=" * 60 + "\n")

    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    series_df = df[(df["store"] == store_id) & (df["item"] == item_id)]
    logger.info(f"Series: {len(series_df)} rows  "
                f"({series_df['date'].min().date()} → {series_df['date'].max().date()})")

    selector = ModelSelector()
    results = {}

    # ---------------------------------------------------------------
    # 1. LSTM — AI / Deep Learning
    # ---------------------------------------------------------------
    logger.info("\n--- [1/2] LSTM — AI / Deep Learning ---")
    lstm = LSTMForecaster(model_dir=LSTM_MODEL_DIR)
    xgb_model_path = XGB_MODEL_PATH_TPL.format(store=store_id, item=item_id)

    lstm_model_exists = LSTMForecaster.model_exists(store_id, item_id, LSTM_MODEL_DIR)
    if not force_retrain and lstm_model_exists:
        logger.info("  Loading saved LSTM model...")
        lstm.load(store_id, item_id)
        lstm_metrics, _ = lstm.evaluate_holdout(
            df, holdout_days=holdout_days, store=store_id, item=item_id, retrain=False
        )
    else:
        logger.info("  Training LSTM model...")
        t0 = time.time()
        lstm_metrics, _ = lstm.evaluate_holdout(
            df, holdout_days=holdout_days, store=store_id, item=item_id, retrain=True
        )
        logger.info(f"  LSTM training completed in {time.time() - t0:.1f}s")
        lstm.save(store_id, item_id)
    logger.info(
        f"  LSTM Holdout  — MAE={lstm_metrics['MAE']:.2f}, "
        f"RMSE={lstm_metrics['RMSE']:.2f}, MAPE={lstm_metrics['MAPE']:.2f}%"
    )
    results["lstm_metrics"] = lstm_metrics
    selector.register("LSTM (AI)", lstm_metrics, lstm)

    # ---------------------------------------------------------------
    # 2. XGBoost — Traditional ML  (RetailXGBoostForecaster)
    # ---------------------------------------------------------------
    logger.info("\n--- [2/2] XGBoost — Traditional ML ---")
    xgb = RetailXGBoostForecaster()

    if not force_retrain and os.path.exists(xgb_model_path):
        logger.info("  Loading saved XGBoost model...")
        xgb = RetailXGBoostForecaster.load(xgb_model_path)
        # Re-evaluate on holdout for a fair metric
        xgb_metrics, _ = xgb.evaluate_holdout(
            df, holdout_days=holdout_days, store=store_id, item=item_id
        )
    else:
        logger.info("  Training XGBoost model...")
        t0 = time.time()
        xgb_metrics, _ = xgb.evaluate_holdout(
            df, holdout_days=holdout_days, store=store_id, item=item_id
        )
        logger.info(f"  XGBoost training completed in {time.time() - t0:.1f}s")
        xgb.save(xgb_model_path)

    logger.info(
        f"  XGBoost Holdout — MAE={xgb_metrics['MAE']:.2f}, "
        f"RMSE={xgb_metrics['RMSE']:.2f}, MAPE={xgb_metrics['MAPE']:.2f}%"
    )
    results["xgb_metrics"] = xgb_metrics
    selector.register("XGBoost (ML)", xgb_metrics, xgb)

    # ---------------------------------------------------------------
    # 3. Model Comparison & Selection
    # ---------------------------------------------------------------
    comparison_df = selector.comparison_table()
    best_name, best_model, best_metrics = selector.select()

    logger.info("\n--- Model Comparison ---")
    logger.info(f"\n{comparison_df.to_string(index=False)}\n")
    logger.info(f"✓ Selected: {best_name}")

    results["best_model_name"] = best_name
    results["best_metrics"] = best_metrics
    results["comparison_df"] = comparison_df

    # ---------------------------------------------------------------
    # 4. Generate Forecasts (both models, 90-day horizon)
    # ---------------------------------------------------------------
    logger.info("\n--- Generating Forecasts ---")
    lstm_fc = lstm.predict(horizon_days=90)
    xgb_fc = xgb.predict_future(df, horizon_days=90, store=store_id, item=item_id)

    lstm_fc.to_csv(f"{FORECAST_DIR}/lstm_s{store_id}_i{item_id}.csv", index=False)
    xgb_fc.to_csv(f"{FORECAST_DIR}/xgboost_s{store_id}_i{item_id}.csv", index=False)
    logger.info(f"  Forecasts saved to {FORECAST_DIR}/")

    results["lstm_forecast"] = lstm_fc
    results["xgb_forecast"] = xgb_fc

    # ---------------------------------------------------------------
    # 5. Save Evaluation Report (JSON)
    # ---------------------------------------------------------------
    report = {
        "store": store_id,
        "item": item_id,
        "holdout_days": holdout_days,
        "lstm_metrics": {k: round(float(v), 4) for k, v in lstm_metrics.items()},
        "xgb_metrics": {k: round(float(v), 4) for k, v in xgb_metrics.items()},
        "best_model": best_name,
        "comparison": comparison_df.to_dict(orient="records"),
    }
    report_path = f"{REPORT_DIR}/model_comparison_s{store_id}_i{item_id}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"  Report saved to {report_path}")

    logger.info("\n" + "=" * 60)
    logger.info(f" Training complete.  Best model: {best_name}")
    logger.info(f" Dashboard: streamlit run app/streamlit_app.py")
    logger.info("=" * 60 + "\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="RetailLens Training Pipeline (LSTM + XGBoost)"
    )
    parser.add_argument("--store", type=int, default=1)
    parser.add_argument("--item", type=int, default=1)
    parser.add_argument("--holdout", type=int, default=90)
    parser.add_argument("--data-path", type=str, default=PROCESSED_PATH)
    parser.add_argument("--force-retrain", action="store_true")
    args = parser.parse_args()

    train_and_evaluate(
        store_id=args.store,
        item_id=args.item,
        holdout_days=args.holdout,
        data_path=args.data_path,
        force_retrain=args.force_retrain,
    )


if __name__ == "__main__":
    main()
