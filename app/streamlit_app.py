"""RetailLens Streamlit dashboard.

The dashboard consumes the frozen data and exported reports when they exist. It
does not retrain models during a page interaction, which keeps the UI suitable
for local demos and hosted deployment.
"""

from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
DATA_CANDIDATES = (ROOT / "data" / "processed" / "sales_clean.csv", ROOT / "data" / "raw" / "train.csv")
FORECAST_DIR = ROOT / "data" / "processed" / "forecasts"
REPORT_PATHS = (ROOT / "reports" / "ts_model_comparison.csv", ROOT / "reports" / "ts_model_comparison.json")
# ML evaluation outputs (written by src/models/ml_evaluation.py)
ML_COMPARISON_PATH = ROOT / "outputs" / "metrics" / "model_comparison.json"
ML_FEATURE_IMPORTANCE_PATH = ROOT / "outputs" / "metrics" / "xgboost_feature_importance.csv"
ML_PREDICTIONS_DIR = ROOT / "outputs" / "predictions"


def _find_column(columns: Iterable[str], aliases: Iterable[str]) -> Optional[str]:
    normalized = {str(column).strip().lower(): str(column) for column in columns}
    for alias in aliases:
        if alias.lower() in normalized:
            return normalized[alias.lower()]
    return None


def detect_schema(frame: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Map common data contracts to the columns actually present in a frame."""
    return {
        "date": _find_column(frame.columns, ("date", "ds", "timestamp")),
        "sales": _find_column(frame.columns, ("sales", "y", "quantity", "demand", "units_sold")),
        "store": _find_column(frame.columns, ("store", "store_id", "location", "branch")),
        "product": _find_column(frame.columns, ("item", "product", "product_id", "sku")),
        "category": _find_column(frame.columns, ("category", "product_category")),
        "inventory": _find_column(frame.columns, ("inventory", "stock", "available_stock", "on_hand")),
    }


@st.cache_data(show_spinner=False)
def load_sales_data() -> Tuple[pd.DataFrame, Dict[str, Optional[str]], Optional[str]]:
    for path in DATA_CANDIDATES:
        if path.exists():
            try:
                frame = pd.read_csv(path)
                if frame.empty:
                    return pd.DataFrame(), {}, f"The dataset at {path} is empty."
                schema = detect_schema(frame)
                if schema.get("date"):
                    frame[schema["date"]] = pd.to_datetime(frame[schema["date"]], errors="coerce")
                if schema.get("sales"):
                    frame[schema["sales"]] = pd.to_numeric(frame[schema["sales"]], errors="coerce")
                return frame.dropna(subset=[schema["date"]] if schema.get("date") else []), schema, None
            except (OSError, ValueError, pd.errors.ParserError) as exc:
                return pd.DataFrame(), {}, f"Unable to read {path.name}: {exc}"
    return pd.DataFrame(), {}, "No dataset found. Add data/raw/train.csv or data/processed/sales_clean.csv."


@st.cache_data(show_spinner=False)
def load_comparison() -> pd.DataFrame:
    path = REPORT_PATHS[0]
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError):
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_ml_results() -> Dict[str, Any]:
    """
    Loads ML evaluation results (XGBoost metrics, best model, feature importance)
    from the outputs written by src/models/ml_evaluation.py.

    Returns an empty dict if the ML pipeline has not been run yet.
    """
    if not ML_COMPARISON_PATH.exists():
        return {}
    try:
        import json
        with open(ML_COMPARISON_PATH) as fh:
            return json.load(fh)
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


@st.cache_data(show_spinner=False)
def load_feature_importance() -> pd.DataFrame:
    """Loads XGBoost feature importance CSV if available."""
    if not ML_FEATURE_IMPORTANCE_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(ML_FEATURE_IMPORTANCE_PATH)
    except (OSError, ValueError, pd.errors.ParserError):
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_forecasts() -> pd.DataFrame:
    dirs_to_search = []
    if FORECAST_DIR.exists():
        dirs_to_search.append(FORECAST_DIR)
    if ML_PREDICTIONS_DIR.exists():
        dirs_to_search.append(ML_PREDICTIONS_DIR)
    if not dirs_to_search:
        return pd.DataFrame()
    frames: List[pd.DataFrame] = []
    for search_dir in dirs_to_search:
        for path in sorted(search_dir.glob("*.csv")):
            try:
                frame = pd.read_csv(path)
                if frame.empty:
                    continue
                frame["_source"] = path.stem
                frames.append(frame)
            except (OSError, ValueError, pd.errors.ParserError):
                continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def best_model(comparison: pd.DataFrame) -> Optional[str]:
    if comparison.empty or "Model" not in comparison.columns:
        return None
    metric_columns = [column for column in ("RMSE", "MAE", "MAPE (%)") if column in comparison.columns]
    if not metric_columns:
        return None
    ranked = comparison.copy()
    ranked[metric_columns] = ranked[metric_columns].apply(pd.to_numeric, errors="coerce")
    ranked = ranked.dropna(subset=metric_columns)
    if ranked.empty:
        return None
    return str(ranked.sort_values(metric_columns, ascending=True).iloc[0]["Model"])


def _label_column(schema: Dict[str, Optional[str]], key: str) -> Optional[str]:
    return schema.get(key)


def filter_data(frame: pd.DataFrame, schema: Dict[str, Optional[str]], stores: List[Any], products: List[Any], dates: Tuple[Any, Any]) -> pd.DataFrame:
    filtered = frame.copy()
    date_column = schema.get("date")
    if date_column and len(dates) == 2:
        filtered = filtered[filtered[date_column].dt.date.between(dates[0], dates[1])]
    if schema.get("store") and stores:
        filtered = filtered[filtered[schema["store"]].isin(stores)]
    if schema.get("product") and products:
        filtered = filtered[filtered[schema["product"]].isin(products)]
    return filtered


def filter_forecasts(frame: pd.DataFrame, stores: List[Any], products: List[Any]) -> pd.DataFrame:
    """Apply available store and product keys to exported forecast rows."""
    filtered = frame.copy()
    store_column = _find_column(filtered.columns, ("store", "store_id", "location", "branch"))
    product_column = _find_column(filtered.columns, ("item", "product", "product_id", "sku"))
    if store_column and stores:
        filtered = filtered[filtered[store_column].isin(stores)]
    if product_column and products:
        filtered = filtered[filtered[product_column].isin(products)]
    return filtered


def select_forecast_model(frame: pd.DataFrame, selected_model: str, comparison: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    selected = frame.copy()
    model_column = _find_column(selected.columns, ("model", "model_name"))
    model_name = best_model(comparison) if selected_model == "Best Model" else selected_model
    if not model_name:
        return selected
    if model_column:
        return selected[selected[model_column].astype(str).str.contains(model_name, case=False, na=False)]
    if "_source" in selected.columns:
        tokens = re.findall(r"[a-z0-9]+", model_name.lower())
        token = "_".join(tokens[:2]) if len(tokens) > 1 else tokens[0]
        return selected[selected["_source"].str.lower().str.replace("-", "_", regex=False).str.contains(token, na=False)]
    return selected


def sales_figure(frame: pd.DataFrame, date_column: str, sales_column: str, title: str, frequency: Optional[str] = None) -> go.Figure:
    chart_data = frame[[date_column, sales_column]].dropna().copy()
    if frequency:
        chart_data = chart_data.set_index(date_column)[sales_column].resample(frequency).sum().reset_index()
    figure = go.Figure(go.Scatter(x=chart_data[date_column], y=chart_data[sales_column], mode="lines", line={"color": "#d95f02", "width": 2}))
    figure.update_layout(title=title, height=360, margin={"l": 20, "r": 20, "t": 50, "b": 20}, hovermode="x unified")
    return figure


def show_metric(label: str, value: Any) -> None:
    st.metric(label, value if value not in (None, "") else "Not Available")


def forecast_view(forecasts: pd.DataFrame, filtered: pd.DataFrame, schema: Dict[str, Optional[str]], selected_model: str, horizon: int) -> None:
    if forecasts.empty:
        st.info("Forecast files are not available. Run the existing evaluation or forecasting pipeline first.")
        return
    date_column = _find_column(forecasts.columns, ("ds", "date", "forecast_date"))
    prediction_column = _find_column(forecasts.columns, ("yhat", "prediction", "predicted_sales", "forecast"))
    if not date_column or not prediction_column:
        st.warning("Forecast output needs a date and prediction column. Forecast visualization is unavailable.")
        return
    forecast_data = forecasts.copy()
    forecast_data[date_column] = pd.to_datetime(forecast_data[date_column], errors="coerce")
    forecast_data = select_forecast_model(forecast_data, selected_model, load_comparison())
    forecast_data = forecast_data.sort_values(date_column).head(horizon)
    if forecast_data.empty:
        st.info("No forecast records found for the selected model and horizon.")
        return
    date_key = schema.get("date")
    sales_key = schema.get("sales")
    figure = go.Figure()
    if date_key and sales_key and not filtered.empty:
        history = filtered.groupby(date_key, as_index=False)[sales_key].sum()
        figure.add_trace(go.Scatter(x=history[date_key], y=history[sales_key], name="Historical Sales", line={"color": "#264653"}))
    figure.add_trace(go.Scatter(x=forecast_data[date_column], y=forecast_data[prediction_column].clip(lower=0), name="Forecast", line={"color": "#e76f51", "width": 3}))
    lower = _find_column(forecast_data.columns, ("yhat_lower", "lower_bound", "lower"))
    upper = _find_column(forecast_data.columns, ("yhat_upper", "upper_bound", "upper"))
    if lower and upper:
        figure.add_trace(go.Scatter(x=forecast_data[date_column], y=forecast_data[upper].clip(lower=0), name="Upper Bound", line={"width": 0}, showlegend=False))
        figure.add_trace(go.Scatter(x=forecast_data[date_column], y=forecast_data[lower].clip(lower=0), name="Confidence Interval", fill="tonexty", line={"width": 0}, fillcolor="rgba(231,111,81,0.18)"))
    figure.update_layout(height=430, hovermode="x unified", margin={"l": 20, "r": 20, "t": 20, "b": 20})
    st.plotly_chart(figure, use_container_width=True)
    st.dataframe(forecast_data, use_container_width=True, hide_index=True)
    st.download_button("Download forecast CSV", forecast_data.to_csv(index=False), "retail_lens_forecast.csv", "text/csv")


def main() -> None:
    st.set_page_config(page_title="RetailLens", page_icon="📈", layout="wide")
    st.title("RetailLens")
    st.subheader("Retail Sales Forecasting & Analytics")
    st.write("Analyze historical retail performance, compare forecasting models, explore future demand, and identify potential stock risks.")

    frame, schema, data_error = load_sales_data()
    comparison = load_comparison()
    forecasts = load_forecasts()
    ml_results = load_ml_results()
    feature_importance = load_feature_importance()
    if data_error:
        st.error(data_error)
    if frame.empty or not schema.get("date") or not schema.get("sales"):
        st.warning("Sales column could not be identified. Historical sales analysis is unavailable.")
        return

    date_column, sales_column = schema["date"], schema["sales"]
    min_date, max_date = frame[date_column].min().date(), frame[date_column].max().date()
    with st.sidebar:
        st.header("Filters")
        store_values = sorted(frame[schema["store"]].dropna().unique().tolist()) if schema.get("store") else []
        selected_stores = st.multiselect("Store", store_values, default=store_values) if store_values else []
        product_source = frame[frame[schema["store"]].isin(selected_stores)] if schema.get("store") and selected_stores else frame
        product_values = sorted(product_source[schema["product"]].dropna().unique().tolist()) if schema.get("product") else []
        selected_products = st.multiselect("Product / Item", product_values, default=product_values) if product_values else []
        selected_dates = st.date_input("Date range", (min_date, max_date), min_value=min_date, max_value=max_date)
        if not isinstance(selected_dates, (tuple, list)) or len(selected_dates) != 2:
            selected_dates = (min_date, max_date)
        horizon = st.selectbox("Forecast horizon", [30, 60, 90], index=0)
        model_options = ["Best Model"]
        if not comparison.empty and "Model" in comparison.columns:
            model_options.extend([str(value) for value in comparison["Model"].dropna().unique()])
        selected_model = st.selectbox("Model", model_options)

    filtered = filter_data(frame, schema, selected_stores, selected_products, (selected_dates[0], selected_dates[1]))
    forecasts = filter_forecasts(forecasts, selected_stores, selected_products)
    forecasts = select_forecast_model(forecasts, selected_model, comparison)
    if filtered.empty:
        st.info("No records found for the selected filters.")
        return

    # Best model: prefer ML evaluation results if available, fall back to comparison CSV
    selected_best = ml_results.get("best_model") or best_model(comparison)
    selected_metrics = comparison[comparison["Model"].astype(str) == (selected_best if selected_model == "Best Model" else selected_model)] if selected_best and not comparison.empty else pd.DataFrame()
    total_sales = filtered[sales_column].sum()
    average_sales = filtered[sales_column].mean()
    forecast_column = _find_column(forecasts.columns, ("yhat", "prediction", "predicted_sales", "forecast")) if not forecasts.empty else None
    forecast_total = forecasts[forecast_column].head(horizon).sum() if forecast_column else None
    kpis = st.columns(7)
    with kpis[0]: show_metric("Historical Sales", f"{total_sales:,.0f}")
    with kpis[1]: show_metric("Average Sales", f"{average_sales:,.1f}")
    with kpis[2]: show_metric("Forecasted Sales", f"{forecast_total:,.0f}" if forecast_total is not None else None)
    with kpis[3]: show_metric("Best Model", selected_best)
    for index, metric in enumerate(("MAE", "RMSE", "MAPE (%)"), start=4):
        with kpis[index]: show_metric(metric, f"{selected_metrics.iloc[0][metric]:,.2f}" if not selected_metrics.empty and metric in selected_metrics else None)

    overview, history_tab, forecast_tab, comparison_tab, ml_tab, alert_tab, explorer_tab = st.tabs(
        ["Overview", "Historical Analysis", "Forecast", "Model Comparison", "ML Evaluation", "Stock Alerts", "Data Explorer"]
    )
    with overview:
        st.plotly_chart(sales_figure(filtered, date_column, sales_column, "Daily sales"), use_container_width=True)
        st.plotly_chart(sales_figure(filtered, date_column, sales_column, "Weekly sales", "W"), use_container_width=True)
    with history_tab:
        st.plotly_chart(sales_figure(filtered, date_column, sales_column, "Daily sales", None), use_container_width=True)
        st.plotly_chart(sales_figure(filtered, date_column, sales_column, "Monthly sales", "MS"), use_container_width=True)
        period = filtered.groupby(date_column)[sales_column].sum()
        if len(period) > 1:
            st.info(f"Highest sales period: {period.idxmax().date()} ({period.max():,.0f} units). Lowest: {period.idxmin().date()} ({period.min():,.0f} units).")
    with forecast_tab:
        forecast_view(forecasts, filtered, schema, selected_model, horizon)
    with comparison_tab:
        if comparison.empty:
            st.info("Model comparison report is not available.")
        else:
            st.dataframe(comparison, use_container_width=True, hide_index=True)
            if selected_best:
                st.success(f"Best model by MAE, then RMSE, then MAPE: {selected_best}")
    with ml_tab:
        st.subheader("Machine Learning Evaluation — XGBoost vs Prophet")
        if not ml_results:
            st.info(
                "ML evaluation results are not available. "
                "Run `python src/models/ml_evaluation.py` to generate them."
            )
        else:
            col_p, col_x = st.columns(2)
            with col_p:
                st.markdown("**Prophet**")
                for metric in ("MAE", "RMSE", "MAPE"):
                    val = ml_results.get("Prophet", {}).get(metric)
                    show_metric(metric, f"{val:.4f}" if val is not None else None)
            with col_x:
                st.markdown("**XGBoost**")
                for metric in ("MAE", "RMSE", "MAPE"):
                    val = ml_results.get("XGBoost", {}).get(metric)
                    show_metric(metric, f"{val:.4f}" if val is not None else None)
            best_name = ml_results.get("best_model")
            best_reason = ml_results.get("best_model_reason", "")
            if best_name:
                st.success(f"**Best model: {best_name}** — {best_reason}")
            holdout = ml_results.get("holdout_days")
            train_start = ml_results.get("train_start", "")
            train_end = ml_results.get("train_end", "")
            test_start = ml_results.get("test_start", "")
            test_end = ml_results.get("test_end", "")
            if holdout:
                st.caption(
                    f"Hold-out: {holdout} days ({test_start} → {test_end})  |  "
                    f"Training: {train_start} → {train_end}"
                )
        if not feature_importance.empty:
            st.subheader("XGBoost Feature Importance")
            top_fi = feature_importance.head(15)
            fig_fi = go.Figure(go.Bar(
                x=top_fi["importance"][::-1],
                y=top_fi["feature"][::-1],
                orientation="h",
                marker_color="#2a9d8f",
            ))
            fig_fi.update_layout(
                height=420,
                margin={"l": 20, "r": 20, "t": 10, "b": 20},
                xaxis_title="Importance (normalised gain)",
            )
            st.plotly_chart(fig_fi, use_container_width=True)
    with alert_tab:
        prediction = forecasts[forecast_column].clip(lower=0) if forecast_column else pd.Series(dtype=float)
        inventory_column = schema.get("inventory")
        threshold = st.number_input("Demand-Based Attention Threshold", min_value=0.0, value=100.0, step=10.0) if not inventory_column else None
        if prediction.empty:
            st.info("Forecast data is unavailable, so no alerts can be generated.")
        elif inventory_column and inventory_column in filtered:
            available = float(filtered[inventory_column].iloc[-1])
            demand = float(prediction.sum())
            st.error("LOW STOCK ALERT" if available < demand else "Stock covers expected demand")
            st.dataframe(pd.DataFrame({"Available Stock": [available], "Forecasted Demand": [demand], "Alert Status": ["LOW STOCK ALERT" if available < demand else "OK"]}), hide_index=True)
        else:
            demand = float(prediction.sum())
            attention = demand > float(threshold)
            st.warning("Attention Required: high expected demand may require inventory review." if attention else "No demand-based attention alert")
            st.dataframe(pd.DataFrame({"Forecasted Demand": [demand], "Threshold": [threshold], "Alert Status": ["HIGH DEMAND ATTENTION" if attention else "OK"]}), hide_index=True)
    with explorer_tab:
        st.download_button("Download historical CSV", filtered.to_csv(index=False), "retail_lens_historical.csv", "text/csv")
        st.dataframe(filtered, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()