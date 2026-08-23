"""
Synthetic Data Generator for RetailLens.
Generates realistic multi-store, multi-item daily sales data with trend,
yearly seasonality, and weekly cycles matching the Kaggle dataset schema.
"""

import os
from typing import Optional
import numpy as np
import pandas as pd


def generate_sample_sales_data(
    num_stores: int = 10,
    num_items: int = 50,
    start_date: str = "2013-01-01",
    end_date: str = "2017-12-31",
    random_seed: int = 42,
    output_path: Optional[str] = "data/raw/train.csv",
) -> pd.DataFrame:
    """
    Generates synthetic daily sales records.
    
    Formula:
    sales = base_item_level 
            * store_multiplier 
            * (1 + trend_slope * day_idx) 
            * (1 + 0.3 * sin(2*pi*doy/365))  # Yearly seasonality
            * (1 + 0.2 * weekend_boost)      # Weekly seasonality
            + noise
    """
    np.random.seed(random_seed)
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    n_days = len(date_range)

    # Base item popularity: between 15 and 80 mean units
    item_baselines = {item: np.random.uniform(15, 80) for item in range(1, num_items + 1)}
    # Store foot-traffic factor: between 0.75 and 1.35
    store_factors = {store: np.random.uniform(0.75, 1.35) for store in range(1, num_stores + 1)}

    day_idx = np.arange(n_days)
    day_of_year = np.array([d.timetuple().tm_yday for d in date_range])
    day_of_week = np.array([d.weekday() for d in date_range])

    # Yearly pattern (peak in summer months around July/Aug)
    yearly_factor = 1.0 + 0.35 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
    # Weekly pattern (peak on Friday=4, Saturday=5, Sunday=6)
    weekly_weights = np.array([0.85, 0.90, 0.95, 1.00, 1.20, 1.35, 1.15])
    weekly_factor = weekly_weights[day_of_week]
    # Linear multi-year upward trend (~4% growth per year)
    trend_factor = 1.0 + (0.04 / 365.0) * day_idx

    records = []
    for store in range(1, num_stores + 1):
        s_fac = store_factors[store]
        for item in range(1, num_items + 1):
            base = item_baselines[item] * s_fac
            mu = base * trend_factor * yearly_factor * weekly_factor
            # Poisson or Gaussian noise
            noise = np.random.normal(0, np.sqrt(mu) * 0.8, size=n_days)
            sales = np.maximum(0, np.round(mu + noise)).astype(int)

            df_series = pd.DataFrame({
                "date": date_range.strftime("%Y-%m-%d"),
                "store": store,
                "item": item,
                "sales": sales,
            })
            records.append(df_series)

    df_all = pd.DataFrame(pd.concat(records, ignore_index=True))
    df_all = df_all.sort_values(["store", "item", "date"]).reset_index(drop=True)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_all.to_csv(output_path, index=False)
        print(f"Generated sample dataset ({df_all.shape[0]} rows) saved to {output_path}")

    return df_all


if __name__ == "__main__":
    generate_sample_sales_data()
