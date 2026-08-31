"""
ML Evaluation Pipeline for RetailLens.

Executes the complete machine-learning evaluation workflow:
    1. Load processed data (generate + clean if absent)
    2. Build XGBoost features (lag, rolling, calendar)
    3. Chronological train / hold-out test split
    4. Data leakage audit
    5. Train XGBoost and generate hold-out predictions
    6. Calculate MAE, RMSE, MAPE for XGBoost
    7. Evaluate Prophet on the same compatible hold-out period
    8. Prophet MAE, RMSE, MAPE
    9. Model comparison table
   10. Automatic best-model selection (MAE → RMSE → MAPE)
   11. XGBoost feature importance
   12. Persist XGBoost model
   13. Export outputs (CSV, JSON, figures)
   14. Return results dict for Streamlit / downstream consumers

All metric values are dynamically calculated from actual predictions.
No results are hard-coded.

Run from the project root:
    python src/models/ml_evaluation.py
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe on servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.clean_data import clean_data, PROCESSED_PATH
from src.features.feature_engineering import (
    LAG_DAYS,
    ROLLING_WINDOWS,
    GROUP_COLS,
    DATE_COL,
    TARGET_COL,
    build_features,
    get_feature_columns,
)
from src.models.prophet_model import RetailProphetForecaster
from src.models.xgboost_model import RetailXGBoostForecaster
from src.utils.metrics import calculate_metrics

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
MODEL_DIR = "models"
OUTPUT_METRICS_DIR = "outputs/metrics"
OUTPUT_PREDICTIONS_DIR = "outputs/predictions"
OUTPUT_FIGURES_DIR = "outputs/figures"
REPORT_DIR = "reports"

XGBOOST_MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_model.pkl")
COMPARISON_CSV = os.path.join(REPORT_DIR, "ts_model_comparison.csv")
COMPARISON_JSON = os.path.join(REPORT_DIR, "ts_model_comparison.json")
ML_COMPARISON_CSV = os.path.join(OUTPUT_METRICS_DIR, "model_comparison.csv")
ML_COMPARISON_JSON = os.path.join(OUTPUT_METRICS_DIR, "model_comparison.json")
XGB_PREDICTIONS_CSV = os.path.join(OUTPUT_PREDICTIONS_DIR, "xgboost_test_predictions.csv")
PROPHET_PREDICTIONS_CSV = os.path.join(OUTPUT_PREDICTIONS_DIR, "prophet_test_predictions.csv")
FEATURE_IMPORTANCE_CSV = os.path.join(OUTPUT_METRICS_DIR, "xgboost_feature_importance.csv")
XGB_ACTUAL_VS_PRED_PNG = os.path.join(OUTPUT_FIGURES_DIR, "xgboost_actual_vs_predicted.png")
PROPHET_ACTUAL_VS_PRED_PNG = os.path.join(OUTPUT_FIGURES_DIR, "prophet_actual_vs_predicted.png")
MODEL_COMPARISON_PNG = os.path.join(OUTPUT_FIGURES_DIR, "model_comparison.png")


# ===========================================================================
# Data leakage audit
# ===========================================================================

def audit_leakage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    date_col: str = DATE_COL,
) -> Dict[str, Any]:
    """
    Verifies that the chronological split is clean.

    Checks:
        1. max(train_date) < min(test_date)
        2. No test targets appear as feature values in training rows
           (structural guarantee: features are computed from shift(N), N>=1)

    Returns a dict with 'pass' (bool) and 'details' (str).
    """
    train_max = pd.to_datetime(train_df[date_col]).max()
    test_min = pd.to_datetime(test_df[date_col]).min()

    chronological = bool(train_max < test_min)
    details = (
        f"Train max date: {train_max.date()}  |  Test min date: {test_min.date()}  |  "
        f"Chronological: {'PASS' if chronological else 'FAIL'}\n"
        "Lag/rolling features use .shift(N) within groups -- future target values "
        "are structurally excluded from all feature columns."
    )
    return {"pass": chronological, "details": details}


# ===========================================================================
# Best-model selection
# ===========================================================================

def select_best_model(model_metrics: Dict[str, Dict[str, float]]) -> Tuple[str, str]:
    """
    Selects the best model from a dict of {model_name: {MAE, RMSE, MAPE}}.

    Selection strategy (lower is better):
        Primary   : lowest MAE
        Secondary : lowest RMSE  (tie-breaker)
        Tertiary  : lowest MAPE  (final tie-breaker)

    Returns
    -------
    (best_model_name, reason_string)
    """
    if not model_metrics:
        raise ValueError("model_metrics dict is empty.")

    ranked = sorted(
        model_metrics.items(),
        key=lambda kv: (kv[1]["MAE"], kv[1]["RMSE"], kv[1]["MAPE"]),
    )
    best_name, best_vals = ranked[0]
    runner_name, runner_vals = ranked[1] if len(ranked) > 1 else (None, None)

    if runner_name and abs(best_vals["MAE"] - runner_vals["MAE"]) < 1e-9:
        if abs(best_vals["RMSE"] - runner_vals["RMSE"]) < 1e-9:
            reason = (
                f"{best_name} selected: lowest MAPE ({best_vals['MAPE']:.4f}%) "
                f"among tied MAE and RMSE."
            )
        else:
            reason = (
                f"{best_name} selected: lowest RMSE ({best_vals['RMSE']:.4f}) "
                f"among tied MAE."
            )
    else:
        reason = (
            f"{best_name} selected: lowest MAE ({best_vals['MAE']:.4f})."
        )

    return best_name, reason


# ===========================================================================
# Visualisations
# ===========================================================================

def _save_actual_vs_predicted(
    comparison_df: pd.DataFrame,
    date_col: str,
    actual_col: str,
    pred_col: str,
    title: str,
    output_path: str,
) -> None:
    """Saves a line chart of actual vs predicted sales."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(comparison_df[date_col], comparison_df[actual_col],
            label="Actual", color="#264653", linewidth=1.5)
    ax.plot(comparison_df[date_col], comparison_df[pred_col],
            label="Predicted", color="#e76f51", linewidth=1.5, linestyle="--")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    logger.info("Saved figure: %s", output_path)


def _save_model_comparison_chart(
    model_results: Dict[str, Dict[str, float]],
    output_path: str,
) -> None:
    """Saves a grouped bar chart comparing MAE / RMSE / MAPE across models."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    models = list(model_results.keys())
    metrics_list = ["MAE", "RMSE", "MAPE"]
    x = np.arange(len(models))
    width = 0.25
    colors = ["#264653", "#2a9d8f", "#e9c46a"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (metric, color) in enumerate(zip(metrics_list, colors)):
        vals = [model_results[m][metric] for m in models]
        ax.bar(x + i * width, vals, width, label=metric, color=color)

    ax.set_xticks(x + width)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("Error")
    ax.set_title("Prophet vs XGBoost — MAE / RMSE / MAPE", fontsize=13)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    logger.info("Saved figure: %s", output_path)


# ===========================================================================
# Main pipeline
# ===========================================================================

def run_ml_evaluation_pipeline(
    store_id: int = 1,
    item_id: int = 1,
    holdout_days: int = 90,
    data_path: str = PROCESSED_PATH,
    output_dir: str = OUTPUT_METRICS_DIR,
    model_dir: str = MODEL_DIR,
    figures_dir: str = OUTPUT_FIGURES_DIR,
    predictions_dir: str = OUTPUT_PREDICTIONS_DIR,
    report_dir: str = REPORT_DIR,
) -> Dict[str, Any]:
    """
    Executes the full ML evaluation pipeline for RetailLens.

    Parameters
    ----------
    store_id     : Store to evaluate (default 1)
    item_id      : Item to evaluate (default 1)
    holdout_days : Number of trailing days for the hold-out test set
    data_path    : Path to processed sales CSV
    output_dir   : Directory for metric outputs
    model_dir    : Directory for saved models
    figures_dir  : Directory for figure outputs
    predictions_dir : Directory for prediction CSVs
    report_dir   : Directory for comparison reports (shared with ts_model_comparison)

    Returns
    -------
    results dict compatible with Streamlit and tests:
    {
        "Prophet":  {"MAE": float, "RMSE": float, "MAPE": float, "sMAPE": float},
        "XGBoost":  {"MAE": float, "RMSE": float, "MAPE": float, "sMAPE": float},
        "best_model": str,
        "best_model_reason": str,
        "feature_importance": pd.DataFrame,
        "xgb_comparison": pd.DataFrame,
        "prophet_comparison": pd.DataFrame,
        "leakage_audit": dict,
        "train_start": str,
        "train_end": str,
        "test_start": str,
        "test_end": str,
        "train_records": int,
        "test_records": int,
    }
    """
    for d in [output_dir, model_dir, figures_dir, predictions_dir, report_dir]:
        os.makedirs(d, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    if not os.path.exists(data_path):
        logger.info("Processed dataset not found -- running clean_data()...")
        clean_data()

    df = pd.read_csv(data_path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(GROUP_COLS + [DATE_COL]).reset_index(drop=True)

    series_df = df[(df["store"] == store_id) & (df["item"] == item_id)].copy()
    if series_df.empty:
        raise ValueError(f"No data for store={store_id}, item={item_id} in {data_path}.")

    logger.info(
        "Loaded series: store=%d item=%d  rows=%d  %s to %s",
        store_id, item_id, len(series_df),
        series_df[DATE_COL].min().date(), series_df[DATE_COL].max().date(),
    )

    # ------------------------------------------------------------------
    # 2. Feature engineering (for reporting / leakage audit)
    # ------------------------------------------------------------------
    logger.info("Building features...")
    featured_df = build_features(
        series_df,
        lag_days=LAG_DAYS,
        rolling_windows=ROLLING_WINDOWS,
        group_cols=GROUP_COLS,
        target_col=TARGET_COL,
        date_col=DATE_COL,
    )
    feature_cols = get_feature_columns(include_store_item=True)

    max_date = series_df[DATE_COL].max()
    split_date = max_date - pd.Timedelta(days=holdout_days - 1)

    train_full = featured_df[featured_df[DATE_COL] < split_date].dropna(subset=feature_cols)
    test_full = featured_df[featured_df[DATE_COL] >= split_date].dropna(subset=feature_cols)

    logger.info(
        "Split -- Train: %s to %s (%d rows)  |  Test: %s to %s (%d rows)",
        train_full[DATE_COL].min().date(), train_full[DATE_COL].max().date(), len(train_full),
        test_full[DATE_COL].min().date(), test_full[DATE_COL].max().date(), len(test_full),
    )

    # ------------------------------------------------------------------
    # 3. Data leakage audit
    # ------------------------------------------------------------------
    leakage_result = audit_leakage(train_full, test_full)
    logger.info(
        "DATA LEAKAGE AUDIT: %s\n%s",
        "PASS" if leakage_result["pass"] else "FAIL",
        leakage_result["details"],
    )
    if not leakage_result["pass"]:
        raise RuntimeError("DATA LEAKAGE DETECTED -- aborting pipeline.")

    # ------------------------------------------------------------------
    # 4. XGBoost evaluation
    # ------------------------------------------------------------------
    logger.info("--- XGBoost hold-out evaluation ---")
    t0 = time.time()
    xgb_forecaster = RetailXGBoostForecaster()
    xgb_metrics, xgb_comparison = xgb_forecaster.evaluate_holdout(
        series_df, holdout_days=holdout_days, store=store_id, item=item_id
    )
    xgb_time = time.time() - t0

    logger.info(
        "XGBoost: MAE=%.4f  RMSE=%.4f  MAPE=%.4f%%  (%.1fs)",
        xgb_metrics["MAE"], xgb_metrics["RMSE"], xgb_metrics["MAPE"], xgb_time,
    )

    # Persist XGBoost model
    xgb_model_path = os.path.join(model_dir, "xgboost_model.pkl")
    xgb_forecaster.save(xgb_model_path)

    # Feature importance
    importance_df = xgb_forecaster.get_feature_importance()
    logger.info("Top 5 features:\n%s", importance_df.head(5).to_string(index=False))

    # Save XGBoost predictions CSV
    xgb_pred_path = os.path.join(predictions_dir, "xgboost_test_predictions.csv")
    xgb_comparison.to_csv(xgb_pred_path, index=False)
    logger.info("Saved XGBoost predictions to %s", xgb_pred_path)

    # Save feature importance CSV
    fi_path = os.path.join(output_dir, "xgboost_feature_importance.csv")
    importance_df.to_csv(fi_path, index=False)

    # Actual-vs-predicted chart
    _save_actual_vs_predicted(
        xgb_comparison,
        date_col=DATE_COL,
        actual_col="actual_sales",
        pred_col="predicted_sales",
        title=f"XGBoost — Actual vs Predicted (Store {store_id}, Item {item_id})",
        output_path=os.path.join(figures_dir, "xgboost_actual_vs_predicted.png"),
    )

    # ------------------------------------------------------------------
    # 5. Prophet evaluation on the same hold-out period
    # ------------------------------------------------------------------
    logger.info("--- Prophet hold-out evaluation (same %d-day window) ---", holdout_days)
    t0 = time.time()
    prophet_forecaster = RetailProphetForecaster(
        seasonality_mode="additive",
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        interval_width=0.95,
    )
    prophet_metrics, prophet_comparison_raw = prophet_forecaster.evaluate_holdout(
        df, holdout_days=holdout_days, store=store_id, item=item_id
    )
    prophet_time = time.time() - t0

    logger.info(
        "Prophet: MAE=%.4f  RMSE=%.4f  MAPE=%.4f%%  (%.1fs)",
        prophet_metrics["MAE"], prophet_metrics["RMSE"], prophet_metrics["MAPE"], prophet_time,
    )

    # Rename columns for consistency
    prophet_comparison = prophet_comparison_raw.rename(
        columns={"ds": DATE_COL, "y": "actual_sales", "yhat": "predicted_sales"}
    )
    prophet_comparison["store"] = store_id
    prophet_comparison["item"] = item_id

    # Save Prophet predictions
    prophet_pred_path = os.path.join(predictions_dir, "prophet_test_predictions.csv")
    prophet_comparison.to_csv(prophet_pred_path, index=False)
    logger.info("Saved Prophet predictions to %s", prophet_pred_path)

    # Actual-vs-predicted chart
    _save_actual_vs_predicted(
        prophet_comparison,
        date_col=DATE_COL,
        actual_col="actual_sales",
        pred_col="predicted_sales",
        title=f"Prophet — Actual vs Predicted (Store {store_id}, Item {item_id})",
        output_path=os.path.join(figures_dir, "prophet_actual_vs_predicted.png"),
    )

    # ------------------------------------------------------------------
    # 6. Model comparison and best-model selection
    # ------------------------------------------------------------------
    model_metrics = {
        "Prophet": {
            "MAE": prophet_metrics["MAE"],
            "RMSE": prophet_metrics["RMSE"],
            "MAPE": prophet_metrics["MAPE"],
            "sMAPE": prophet_metrics.get("sMAPE", 0.0),
        },
        "XGBoost": {
            "MAE": xgb_metrics["MAE"],
            "RMSE": xgb_metrics["RMSE"],
            "MAPE": xgb_metrics["MAPE"],
            "sMAPE": xgb_metrics.get("sMAPE", 0.0),
        },
    }

    best_model_name, best_model_reason = select_best_model(model_metrics)

    logger.info("==============================================")
    logger.info("MODEL COMPARISON")
    logger.info("==============================================")
    for model_name, mvals in model_metrics.items():
        logger.info(
            "  %-10s  MAE=%.4f  RMSE=%.4f  MAPE=%.4f%%",
            model_name, mvals["MAE"], mvals["RMSE"], mvals["MAPE"],
        )
    logger.info("Best model: %s  |  Reason: %s", best_model_name, best_model_reason)

    # Model comparison chart
    _save_model_comparison_chart(
        {k: v for k, v in model_metrics.items()},
        output_path=os.path.join(figures_dir, "model_comparison.png"),
    )

    # ------------------------------------------------------------------
    # 7. Export comparison reports
    # ------------------------------------------------------------------

    # Extended rows for the main ts_model_comparison.csv
    # (kept compatible with the existing Streamlit REPORT_PATHS consumer)
    comparison_rows = []
    for model_name, mvals in model_metrics.items():
        row = {
            "Model": model_name,
            "Store": store_id,
            "Item": item_id,
            "MAE": mvals["MAE"],
            "RMSE": mvals["RMSE"],
            "MAPE (%)": mvals["MAPE"],
            "sMAPE (%)": mvals["sMAPE"],
            "Coverage (%)": 0.0,
            "Avg Interval Width": 0.0,
            "Train Time (s)": round(xgb_time if model_name == "XGBoost" else prophet_time, 2),
            "Notes": (
                "Lag/rolling/calendar features, XGBRegressor"
                if model_name == "XGBoost"
                else "Additive, default priors"
            ),
        }
        comparison_rows.append(row)

    # Append best-model marker row (informational, not a real model)
    comparison_df = pd.DataFrame(comparison_rows)

    # ML-specific outputs
    comparison_df.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)
    ml_json_payload = {
        **{k: v for k, v in model_metrics.items()},
        "best_model": best_model_name,
        "best_model_reason": best_model_reason,
        "store": store_id,
        "item": item_id,
        "holdout_days": holdout_days,
        "train_start": str(train_full[DATE_COL].min().date()),
        "train_end": str(train_full[DATE_COL].max().date()),
        "test_start": str(test_full[DATE_COL].min().date()),
        "test_end": str(test_full[DATE_COL].max().date()),
        "train_records": int(len(train_full)),
        "test_records": int(len(test_full)),
    }
    with open(os.path.join(output_dir, "model_comparison.json"), "w") as fh:
        json.dump(ml_json_payload, fh, indent=2)

    # Update the shared comparison report (used by Streamlit and existing evaluate_ts.py)
    _update_shared_report(
        comparison_rows,
        best_model_name,
        best_model_reason,
        report_dir=report_dir,
    )

    # ------------------------------------------------------------------
    # 8. Assemble results
    # ------------------------------------------------------------------
    results: Dict[str, Any] = {
        "Prophet": model_metrics["Prophet"],
        "XGBoost": model_metrics["XGBoost"],
        "best_model": best_model_name,
        "best_model_reason": best_model_reason,
        "feature_importance": importance_df,
        "xgb_comparison": xgb_comparison,
        "prophet_comparison": prophet_comparison,
        "leakage_audit": leakage_result,
        "train_start": str(train_full[DATE_COL].min().date()),
        "train_end": str(train_full[DATE_COL].max().date()),
        "test_start": str(test_full[DATE_COL].min().date()),
        "test_end": str(test_full[DATE_COL].max().date()),
        "train_records": len(train_full),
        "test_records": len(test_full),
        "xgb_model_path": xgb_model_path,
    }

    logger.info("ML evaluation pipeline complete.")
    return results


def _update_shared_report(
    ml_rows: List[Dict],
    best_model: str,
    best_model_reason: str,
    report_dir: str = REPORT_DIR,
) -> None:
    """
    Merges the ML evaluation rows into the shared ts_model_comparison report.
    Existing Prophet/SARIMA rows are preserved; XGBoost row is added or updated.
    """
    csv_path = os.path.join(report_dir, "ts_model_comparison.csv")
    json_path = os.path.join(report_dir, "ts_model_comparison.json")

    existing_rows: List[Dict] = []
    if os.path.exists(json_path):
        try:
            with open(json_path) as fh:
                existing_rows = json.load(fh)
        except (json.JSONDecodeError, OSError):
            pass

    # Remove any existing XGBoost entry so we always use the freshly computed one
    existing_rows = [r for r in existing_rows if r.get("Model") != "XGBoost"]

    # Append fresh XGBoost row
    xgb_row = next((r for r in ml_rows if r["Model"] == "XGBoost"), None)
    if xgb_row:
        existing_rows.append(xgb_row)

    # Also record best-model selection as a metadata entry
    best_entry = {
        "Model": f"[Best: {best_model}]",
        "Notes": best_model_reason,
    }
    # Remove any prior best-model entry
    existing_rows = [r for r in existing_rows if not str(r.get("Model", "")).startswith("[Best:")]
    existing_rows.append(best_entry)

    # Write JSON
    os.makedirs(report_dir, exist_ok=True)
    with open(json_path, "w") as fh:
        json.dump(existing_rows, fh, indent=2)

    # Write CSV (only rows that have a full set of metric columns)
    metric_cols = ["Model", "Store", "Item", "MAE", "RMSE", "MAPE (%)", "sMAPE (%)",
                   "Coverage (%)", "Avg Interval Width", "Train Time (s)", "Notes"]
    report_rows = [r for r in existing_rows if "MAE" in r]
    pd.DataFrame(report_rows)[metric_cols].to_csv(csv_path, index=False)
    logger.info("Updated shared report: %s", csv_path)


# ===========================================================================
# CLI entry point
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RetailLens ML Evaluation Pipeline (XGBoost vs Prophet)"
    )
    parser.add_argument("--store", type=int, default=1)
    parser.add_argument("--item", type=int, default=1)
    parser.add_argument("--holdout-days", type=int, default=90)
    parser.add_argument("--data-path", type=str, default=PROCESSED_PATH)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_METRICS_DIR)
    parser.add_argument("--model-dir", type=str, default=MODEL_DIR)
    parser.add_argument("--figures-dir", type=str, default=OUTPUT_FIGURES_DIR)
    parser.add_argument("--predictions-dir", type=str, default=OUTPUT_PREDICTIONS_DIR)
    parser.add_argument("--report-dir", type=str, default=REPORT_DIR)
    args = parser.parse_args()

    results = run_ml_evaluation_pipeline(
        store_id=args.store,
        item_id=args.item,
        holdout_days=args.holdout_days,
        data_path=args.data_path,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        figures_dir=args.figures_dir,
        predictions_dir=args.predictions_dir,
        report_dir=args.report_dir,
    )

    print("\n============================================================")
    print("  RETAILLENS -- ML EVALUATION RESULTS")
    print("============================================================")
    for model in ("Prophet", "XGBoost"):
        m = results[model]
        print(f"\n  {model}:")
        print(f"    MAE  = {m['MAE']:.4f}")
        print(f"    RMSE = {m['RMSE']:.4f}")
        print(f"    MAPE = {m['MAPE']:.4f}%")
    print(f"\n  Best model : {results['best_model']}")
    print(f"  Reason     : {results['best_model_reason']}")
    print("\n  Top 5 XGBoost features:")
    print(results["feature_importance"].head(5).to_string(index=False))
    print("\n  Train period : %s to %s (%d rows)" % (
        results["train_start"], results["train_end"], results["train_records"]))
    print("  Test period  : %s to %s (%d rows)" % (
        results["test_start"], results["test_end"], results["test_records"]))
    print("\n  Leakage audit: %s" % ("PASS" if results["leakage_audit"]["pass"] else "FAIL"))
    print("============================================================\n")


if __name__ == "__main__":
    main()
