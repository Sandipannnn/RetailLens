"""
EDA STARTER SCRIPT — Soumili Das
Role: Exploratory Data Analysis (EDA) & Visualization
Project: RetailLens

- Sales show a clear upward trend from 2013 to 2017
- Weekends (Sat/Sun) have noticeably higher average sales than weekdays
- December and July show seasonal peaks
- No promo data available in this dataset for promo-impact analysis

"""

import pandas as pd
import matplotlib.pyplot as plt
import os

# ----------------------------------------------------------------------
# SETTINGS — change these to match your actual file
# ----------------------------------------------------------------------

DATA_PATH = "../data/processed/sales_clean.csv"   # relative path from the /eda folder to Sandipan's cleaned file

# COLUMN NAMES — confirmed from Sandipan's clean_data.py (src/clean_data.py)
DATE_COL = "date"      # column with the date
SALES_COL = "sales"    # column with number of units sold
STORE_COL = "store"    # column with store id
ITEM_COL = "item"      # column with item/product id
PROMO_COL = None       # this dataset has no promo column, so this stays None (promo step will just be skipped)

OUTPUT_DIR = "eda_charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    """Step 1: Load the CSV and make sure the date column is a real date."""
    df = pd.read_csv(DATA_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    return df


def inspect_data(df):
    """Step 2: Print basic info so you know what you're working with."""
    print("=" * 60)
    print("BASIC INFO ABOUT YOUR DATA")
    print("=" * 60)
    print(f"Number of rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Date range: {df[DATE_COL].min()} to {df[DATE_COL].max()}")
    print("\nFirst 5 rows:")
    print(df.head())
    print()


def plot_daily_weekly_monthly(df):
    """Step 3: Aggregate sales by day, week, and month, and plot each."""

    # --- Daily ---
    daily = df.groupby(DATE_COL)[SALES_COL].sum()
    plt.figure(figsize=(12, 4))
    daily.plot()
    plt.title("Daily Total Sales")
    plt.xlabel("Date")
    plt.ylabel("Units Sold")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/1_daily_sales.png")
    plt.close()

    # --- Weekly (resample groups by week automatically) ---
    weekly = df.set_index(DATE_COL)[SALES_COL].resample("W").sum()
    plt.figure(figsize=(12, 4))
    weekly.plot()
    plt.title("Weekly Total Sales")
    plt.xlabel("Week")
    plt.ylabel("Units Sold")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/2_weekly_sales.png")
    plt.close()

    # --- Monthly ---
    monthly = df.set_index(DATE_COL)[SALES_COL].resample("ME").sum()
    plt.figure(figsize=(12, 4))
    monthly.plot(kind="bar")
    plt.title("Monthly Total Sales")
    plt.xlabel("Month")
    plt.ylabel("Units Sold")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/3_monthly_sales.png")
    plt.close()

    print("Saved: 1_daily_sales.png, 2_weekly_sales.png, 3_monthly_sales.png")


def plot_seasonality(df):
    """Step 4: Look for patterns by day-of-week and by month."""

    df = df.copy()
    df["day_of_week"] = df[DATE_COL].dt.day_name()
    df["month"] = df[DATE_COL].dt.month_name()

    # --- Average sales by day of week ---
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]
    by_day = df.groupby("day_of_week")[SALES_COL].mean().reindex(day_order)

    plt.figure(figsize=(8, 4))
    by_day.plot(kind="bar", color="skyblue")
    plt.title("Average Sales by Day of Week")
    plt.ylabel("Avg Units Sold")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/4_sales_by_day_of_week.png")
    plt.close()

    # --- Average sales by month ---
    month_order = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
    by_month = df.groupby("month")[SALES_COL].mean().reindex(month_order)

    plt.figure(figsize=(10, 4))
    by_month.plot(kind="bar", color="salmon")
    plt.title("Average Sales by Month")
    plt.ylabel("Avg Units Sold")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/5_sales_by_month.png")
    plt.close()

    print("Saved: 4_sales_by_day_of_week.png, 5_sales_by_month.png")


def plot_promo_effect(df):
    """Step 5: Compare promo vs non-promo sales, only if a promo column exists."""
    if PROMO_COL is None or PROMO_COL not in df.columns:
        print("No promo column set — skipping promo analysis.")
        return

    avg_by_promo = df.groupby(PROMO_COL)[SALES_COL].mean()

    plt.figure(figsize=(6, 4))
    avg_by_promo.plot(kind="bar", color="mediumseagreen")
    plt.title("Average Sales: Promo vs Non-Promo")
    plt.ylabel("Avg Units Sold")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/6_promo_effect.png")
    plt.close()

    print("Saved: 6_promo_effect.png")
    print(avg_by_promo)


def main():
    df = load_data()
    inspect_data(df)
    plot_daily_weekly_monthly(df)
    plot_seasonality(df)
    plot_promo_effect(df)
    print("\nAll done! Check the 'eda_charts' folder for your images.")
    print("Next: look at each chart and write 3-5 bullet-point insights,")
    print("e.g. 'Sales peak on Saturdays' or 'Store sales spike in December'.")


if __name__ == "__main__":
    main()
