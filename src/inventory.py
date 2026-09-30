"""
Inventory Recommendation Engine for RetailLens.
Uses deterministic business rules based on demand forecasts.

Formulas:
- Expected Lead-Time Demand = sum(forecast[0:lead_time_days])
- Safety Stock = z * sigma_forecast * sqrt(lead_time_days)
  where z = 1.645 (95% service level)
- Reorder Point (ROP) = Lead-Time Demand + Safety Stock
- Suggested Reorder Quantity = max(0, ROP + cycle_stock - current_stock)
- Low-stock Alert: current_stock <= ROP
- Overstock Alert: current_stock > ROP + 2 * cycle_stock

No ML model is used here — these are deterministic rules based on the
demand forecast produced by LSTM or XGBoost.

Owner: RetailLens ML Track
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


# Default inventory parameters
DEFAULT_LEAD_TIME_DAYS: int = 7       # Days between placing and receiving an order
DEFAULT_CYCLE_STOCK_DAYS: int = 14    # How many days of demand to order at a time
DEFAULT_SERVICE_LEVEL: float = 0.95   # 95% service level (z = 1.645)
Z_SCORE_95: float = 1.645


@dataclass
class InventoryReport:
    """Summary of inventory recommendation for one (store, item) pair."""
    store: int
    item: int
    current_stock: float
    lead_time_days: int
    expected_lead_time_demand: float
    safety_stock: float
    reorder_point: float
    cycle_stock: float
    suggested_reorder_qty: float
    low_stock_alert: bool
    overstock_alert: bool
    forecast_horizon_days: int
    total_expected_demand: float
    model_used: str

    def to_dict(self) -> Dict:
        return {
            "Store": self.store,
            "Item": self.item,
            "Current Stock": round(self.current_stock, 1),
            "Lead Time Demand": round(self.expected_lead_time_demand, 1),
            "Safety Stock": round(self.safety_stock, 1),
            "Reorder Point": round(self.reorder_point, 1),
            "Cycle Stock": round(self.cycle_stock, 1),
            "Suggested Reorder Qty": round(self.suggested_reorder_qty, 1),
            "Low-Stock Alert": self.low_stock_alert,
            "Overstock Alert": self.overstock_alert,
            "Forecast Horizon (days)": self.forecast_horizon_days,
            "Total Expected Demand": round(self.total_expected_demand, 1),
            "Model Used": self.model_used,
        }


def compute_inventory_recommendation(
    forecast_df: pd.DataFrame,
    current_stock: float,
    store: int,
    item: int,
    model_name: str = "Unknown",
    lead_time_days: int = DEFAULT_LEAD_TIME_DAYS,
    cycle_stock_days: int = DEFAULT_CYCLE_STOCK_DAYS,
    service_level: float = DEFAULT_SERVICE_LEVEL,
) -> InventoryReport:
    """
    Computes inventory metrics and alerts from a demand forecast.

    Args:
        forecast_df: DataFrame with at least a 'yhat' column (daily demand forecasts)
        current_stock: Current on-hand inventory count
        store: Store ID
        item: Item ID
        model_name: Name of the model that produced the forecast
        lead_time_days: Replenishment lead time in days
        cycle_stock_days: Days of inventory to order per cycle
        service_level: Desired service level (0.95 = 95%)

    Returns:
        InventoryReport dataclass with all computed values and alerts.
    """
    if "yhat" not in forecast_df.columns:
        raise ValueError("forecast_df must contain a 'yhat' column.")
    if len(forecast_df) == 0:
        raise ValueError("forecast_df is empty.")

    yhats = forecast_df["yhat"].values.astype(float)
    yhats = np.maximum(0, yhats)  # clip negative forecasts

    # -- Lead-Time Demand --
    lt_slice = yhats[:lead_time_days]
    expected_lead_time_demand = float(np.sum(lt_slice))

    # -- Safety Stock --
    # sigma of the forecast (as a proxy for demand variability)
    sigma = float(np.std(lt_slice)) if len(lt_slice) > 1 else float(np.mean(lt_slice) * 0.2)
    z = _service_level_to_z(service_level)
    safety_stock = z * sigma * np.sqrt(lead_time_days)

    # -- Reorder Point --
    reorder_point = expected_lead_time_demand + safety_stock

    # -- Cycle Stock --
    # Average demand per day × cycle stock days
    avg_daily_demand = float(np.mean(yhats))
    cycle_stock = avg_daily_demand * cycle_stock_days

    # -- Suggested Reorder Quantity --
    # Bring stock up to ROP + one full cycle
    target_stock = reorder_point + cycle_stock
    suggested_qty = max(0.0, target_stock - current_stock)

    # -- Alerts --
    low_stock_alert = bool(current_stock <= reorder_point)
    overstock_alert = bool(current_stock > reorder_point + 2 * cycle_stock)

    # -- Total forecast demand --
    total_expected_demand = float(np.sum(yhats))

    return InventoryReport(
        store=store,
        item=item,
        current_stock=float(current_stock),
        lead_time_days=lead_time_days,
        expected_lead_time_demand=expected_lead_time_demand,
        safety_stock=safety_stock,
        reorder_point=reorder_point,
        cycle_stock=cycle_stock,
        suggested_reorder_qty=suggested_qty,
        low_stock_alert=low_stock_alert,
        overstock_alert=overstock_alert,
        forecast_horizon_days=len(yhats),
        total_expected_demand=total_expected_demand,
        model_used=model_name,
    )


def _service_level_to_z(service_level: float) -> float:
    """Converts service level to z-score (approximate)."""
    # Common service levels
    levels = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326, 0.999: 3.090}
    if service_level in levels:
        return levels[service_level]
    # Approximate using normal distribution
    from scipy.stats import norm  # type: ignore[import]
    try:
        return float(norm.ppf(service_level))
    except Exception:
        return 1.645  # Default to 95%


def batch_inventory_recommendations(
    forecasts: Dict[tuple, pd.DataFrame],
    stock_levels: Dict[tuple, float],
    model_name: str = "Unknown",
    lead_time_days: int = DEFAULT_LEAD_TIME_DAYS,
    cycle_stock_days: int = DEFAULT_CYCLE_STOCK_DAYS,
) -> pd.DataFrame:
    """
    Computes inventory recommendations for multiple (store, item) pairs.

    Args:
        forecasts: Dict mapping (store, item) -> forecast DataFrame
        stock_levels: Dict mapping (store, item) -> current stock level
        model_name: Name of the forecasting model used

    Returns:
        DataFrame with one row per (store, item)
    """
    reports = []
    for (store, item), fc_df in forecasts.items():
        stock = stock_levels.get((store, item), 0.0)
        try:
            report = compute_inventory_recommendation(
                forecast_df=fc_df,
                current_stock=stock,
                store=store,
                item=item,
                model_name=model_name,
                lead_time_days=lead_time_days,
                cycle_stock_days=cycle_stock_days,
            )
            reports.append(report.to_dict())
        except Exception as e:
            reports.append({
                "Store": store,
                "Item": item,
                "Error": str(e),
            })

    return pd.DataFrame(reports)
