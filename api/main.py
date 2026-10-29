"""Read-only FastAPI endpoints for the existing RetailLens sales artifacts."""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.inventory import compute_inventory_recommendation

ROOT = Path(__file__).resolve().parents[1]
DATA_CANDIDATES = (
    ROOT / "data" / "processed" / "sales_clean.csv",
    ROOT / "data" / "raw" / "train.csv",
)
FORECAST_DIR = ROOT / "data" / "processed" / "forecasts"
ML_METRICS_JSON = ROOT / "outputs" / "metrics" / "model_comparison.json"
ML_METRICS_CSV = ROOT / "outputs" / "metrics" / "model_comparison.csv"
TS_METRICS_JSON = ROOT / "reports" / "ts_model_comparison.json"
TS_METRICS_CSV = ROOT / "reports" / "ts_model_comparison.csv"
FEATURE_IMPORTANCE_CSV = ROOT / "outputs" / "metrics" / "xgboost_feature_importance.csv"
REPORT_DIR = ROOT / "reports"
PREDICTION_DIR = ROOT / "outputs" / "predictions"

app = FastAPI(
    title="RetailLens Sales Analytics API",
    description=(
        "Read-only access to RetailLens historical sales, exported future forecasts, "
        "and saved model evaluation artifacts. This API never trains models or writes to project data."
    ),
    version="1.0.0",
)
cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "RETAILLENS_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["GET"],
    allow_headers=["Accept"],
)


def _json_safe(value: Any) -> Any:
    """Convert pandas/numpy values into JSON-compatible Python values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


@lru_cache(maxsize=1)
def _sales_frame() -> tuple[pd.DataFrame, Path]:
    for path in DATA_CANDIDATES:
        if path.is_file():
            try:
                frame = pd.read_csv(path)
            except (OSError, UnicodeError, pd.errors.ParserError) as exc:
                raise HTTPException(status_code=500, detail=f"Unable to read sales data {path.name}: {exc}") from exc
            required = {"date", "store", "item", "sales"}
            missing = sorted(required.difference(frame.columns))
            if missing:
                raise HTTPException(status_code=500, detail=f"Sales data is missing required columns: {', '.join(missing)}")
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            frame["store"] = pd.to_numeric(frame["store"], errors="coerce")
            frame["item"] = pd.to_numeric(frame["item"], errors="coerce")
            frame["sales"] = pd.to_numeric(frame["sales"], errors="coerce")
            frame = frame.dropna(subset=["date", "store", "item", "sales"]).sort_values(["date", "store", "item"])
            return frame, path
    raise HTTPException(status_code=404, detail="No sales CSV found. Expected data/processed/sales_clean.csv or data/raw/train.csv.")


def _filtered_sales(store: int | None, item: int | None, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    frame, _ = _sales_frame()
    if start_date:
        start = pd.Timestamp(start_date)
    else:
        start = None
    if end_date:
        end = pd.Timestamp(end_date)
    else:
        end = None
    if start is not None and end is not None and start > end:
        raise ValueError("start_date must be on or before end_date.")
    if start is not None:
        frame = frame[frame["date"] >= start]
    if end is not None:
        frame = frame[frame["date"] < end + pd.Timedelta(days=1)]
    if store is not None:
        frame = frame[frame["store"] == store]
    if item is not None:
        frame = frame[frame["item"] == item]
    return frame


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"Unable to read {path.relative_to(ROOT)}: {exc}") from exc


def _read_csv_records(path: Path) -> list[dict[str, Any]] | None:
    if not path.is_file():
        return None
    try:
        return _json_safe(pd.read_csv(path).to_dict(orient="records"))
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise HTTPException(status_code=500, detail=f"Unable to read {path.relative_to(ROOT)}: {exc}") from exc


def _training_reports(store: int | None = None, item: int | None = None) -> list[dict[str, Any]]:
    reports = []
    for path in sorted(REPORT_DIR.glob("model_comparison_s*_i*.json")):
        match = re.fullmatch(r"model_comparison_s(\d+)_i(\d+)", path.stem)
        if not match:
            continue
        report_store, report_item = map(int, match.groups())
        if store is not None and report_store != store:
            continue
        if item is not None and report_item != item:
            continue
        report = _read_json(path)
        if report is None:
            continue
        reports.append({
            "store": report.get("store", report_store),
            "item": report.get("item", report_item),
            "report": report,
        })
    return reports


def _forecast_frame(store: int | None = None, item: int | None = None) -> pd.DataFrame:
    files = sorted(FORECAST_DIR.glob("*.csv")) if FORECAST_DIR.is_dir() else []
    parts: list[pd.DataFrame] = []
    for path in files:
        try:
            frame = pd.read_csv(path)
        except (OSError, UnicodeError, pd.errors.ParserError) as exc:
            raise HTTPException(status_code=500, detail=f"Unable to read forecast {path.name}: {exc}") from exc
        if frame.empty or {"actual", "actual_sales", "actuals"}.intersection(str(col).lower() for col in frame.columns):
            continue

        match = re.search(r"_s(\d+)_i(\d+)$", path.stem)
        inferred_store, inferred_item = (map(int, match.groups()) if match else (None, None))
        store_col = next((c for c in frame.columns if str(c).lower() in {"store", "store_id", "location", "branch"}), None)
        item_col = next((c for c in frame.columns if str(c).lower() in {"item", "product", "product_id", "sku"}), None)
        if store is not None:
            if store_col:
                frame = frame[pd.to_numeric(frame[store_col], errors="coerce") == store]
            elif inferred_store != store:
                continue
        if item is not None:
            if item_col:
                frame = frame[pd.to_numeric(frame[item_col], errors="coerce") == item]
            elif inferred_item != item:
                continue
        if frame.empty:
            continue
        if store_col is None and inferred_store is not None:
            frame["store"] = inferred_store
        if item_col is None and inferred_item is not None:
            frame["item"] = inferred_item
        frame["_source"] = path.stem
        parts.append(frame)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


@app.get("/health", tags=["service"])
def health() -> dict[str, str]:
    """Report that the API process is responding."""
    return {"status": "ok"}


@app.get("/sales/options", tags=["sales"])
def sales_options(
    store: int | None = Query(default=None, ge=1),
) -> dict[str, Any]:
    """Return available store/item choices and sales date bounds for the frontend."""
    frame, _ = _sales_frame()
    if frame.empty:
        raise HTTPException(status_code=404, detail="Sales data has no valid records.")
    scoped = frame[frame["store"] == store] if store is not None else frame
    return {
        "stores": sorted(int(value) for value in frame["store"].unique()),
        "items": sorted(int(value) for value in scoped["item"].unique()),
        "date_start": frame["date"].min().date().isoformat(),
        "date_end": frame["date"].max().date().isoformat(),
    }


@app.get("/sales/summary", tags=["sales"])
def sales_summary(
    store: int | None = Query(default=None, ge=1),
    item: int | None = Query(default=None, ge=1),
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """Return sales totals, averages, date coverage, and row counts for optional filters."""
    try:
        data = _filtered_sales(store, item, start_date, end_date)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid date filter: {exc}") from exc
    return {
        "filters": {"store": store, "item": item, "start_date": start_date, "end_date": end_date},
        "records": int(len(data)),
        "total_sales": float(data["sales"].sum()) if not data.empty else 0.0,
        "average_daily_sales_per_record": float(data["sales"].mean()) if not data.empty else None,
        "minimum_sales": float(data["sales"].min()) if not data.empty else None,
        "maximum_sales": float(data["sales"].max()) if not data.empty else None,
        "stores": int(data["store"].nunique()),
        "items": int(data["item"].nunique()),
        "date_start": data["date"].min().date().isoformat() if not data.empty else None,
        "date_end": data["date"].max().date().isoformat() if not data.empty else None,
    }


@app.get("/sales", tags=["sales"])
def historical_sales(
    store: int | None = Query(default=None, ge=1),
    item: int | None = Query(default=None, ge=1),
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = Query(default=1000, ge=1, le=10000),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    """Return paginated historical observations with optional store/item/date filters."""
    try:
        data = _filtered_sales(store, item, start_date, end_date)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid date filter: {exc}") from exc
    total = len(data)
    records = data.iloc[offset : offset + limit].copy()
    records["date"] = records["date"].dt.strftime("%Y-%m-%d")
    return {"total": int(total), "limit": limit, "offset": offset, "records": records.to_dict(orient="records")}


@app.get("/forecasts", tags=["forecasts"])
def forecasts(
    store: int | None = Query(default=None, ge=1),
    item: int | None = Query(default=None, ge=1),
    limit: int = Query(default=1000, ge=1, le=10000),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    """Return saved future forecast rows; holdout prediction files are deliberately excluded."""
    combined = _forecast_frame(store, item)
    if combined.empty:
        return {"available": False, "message": "No exported future forecast CSV files are available.", "total": 0, "limit": limit, "offset": offset, "records": []}
    total = len(combined)
    return {"available": True, "total": int(total), "limit": limit, "offset": offset, "records": _json_safe(combined.iloc[offset : offset + limit].to_dict(orient="records"))}


@app.get("/models/comparison", tags=["models"])
def model_comparison(
    store: int | None = Query(default=None, ge=1),
    item: int | None = Query(default=None, ge=1),
) -> dict[str, Any]:
    """Return current per-series reports plus the legacy aggregate comparisons."""
    return {
        "training_reports": _training_reports(store, item),
        "time_series": {"json": _read_json(TS_METRICS_JSON), "csv": _read_csv_records(TS_METRICS_CSV)},
        "machine_learning": {"json": _read_json(ML_METRICS_JSON), "csv": _read_csv_records(ML_METRICS_CSV)},
    }


@app.get("/models/evaluation", tags=["models"])
def model_evaluation(
    store: int | None = Query(default=None, ge=1),
    item: int | None = Query(default=None, ge=1),
) -> dict[str, Any]:
    """Return current per-series metrics and legacy evaluation artifacts."""
    metrics = _read_json(ML_METRICS_JSON)
    prediction_summaries: dict[str, Any] = {}
    for name in ("prophet_test_predictions.csv", "xgboost_test_predictions.csv"):
        path = PREDICTION_DIR / name
        rows = _read_csv_records(path)
        prediction_summaries[name] = {"available": rows is not None, "records": len(rows) if rows is not None else 0}
    return {
        "training_reports": _training_reports(store, item),
        "metrics": metrics,
        "feature_importance": _read_csv_records(FEATURE_IMPORTANCE_CSV),
        "holdout_predictions": prediction_summaries,
        "note": "Prediction artifacts are historical holdout evaluations and are not future forecasts.",
    }


@app.get("/inventory/recommendation", tags=["inventory"])
def inventory_recommendation(
    store: int = Query(ge=1),
    item: int = Query(ge=1),
    current_stock: float = Query(ge=0),
    model: Literal["best", "lstm", "xgboost"] = "best",
    lead_time_days: int = Query(default=7, ge=1),
    cycle_stock_days: int = Query(default=14, ge=1),
    service_level: float = Query(default=0.95, gt=0.5, lt=1.0),
) -> dict[str, Any]:
    """Calculate an inventory recommendation from the selected saved forecast."""
    report_rows = _training_reports(store, item)
    if model == "best":
        if not report_rows:
            raise HTTPException(
                status_code=404,
                detail=f"No trained model report found for store {store}, item {item}.",
            )
        selected_model = str(report_rows[0]["report"].get("best_model", "")).lower()
        if "lstm" in selected_model:
            forecast_model = "lstm"
        elif "xgboost" in selected_model:
            forecast_model = "xgboost"
        else:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported best_model value in saved report: {selected_model or 'missing'}.",
            )
    else:
        forecast_model = model

    forecast_path = FORECAST_DIR / f"{forecast_model}_s{store}_i{item}.csv"
    if not forecast_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"No {forecast_model} future forecast found for store {store}, item {item}.",
        )
    try:
        forecast_frame = pd.read_csv(forecast_path)
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise HTTPException(status_code=500, detail=f"Unable to read forecast {forecast_path.name}: {exc}") from exc

    try:
        report = compute_inventory_recommendation(
            forecast_df=forecast_frame,
            current_stock=current_stock,
            store=store,
            item=item,
            model_name=forecast_model.upper(),
            lead_time_days=lead_time_days,
            cycle_stock_days=cycle_stock_days,
            service_level=service_level,
        )
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=f"Unable to calculate inventory recommendation: {exc}") from exc

    return {
        "store": store,
        "item": item,
        "forecast_model": forecast_model,
        "recommendation": report.to_dict(),
        "note": "Safety stock uses forecast variability as a proxy; validate against operational lead-time and demand data.",
    }
