"""
First inspection pass on the Store Item Demand Forecasting dataset.
Owner: Sandipan (Data Engineer)
"""

import pandas as pd

# Load raw data
df = pd.read_csv("data/raw/train.csv")

# --- Basic shape and structure ---
print("Shape:", df.shape)
print("\nDtypes:\n", df.dtypes)
print("\nFirst 5 rows:\n", df.head())

# --- Data quality checks ---
print("\nMissing values per column:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

# --- Date handling ---
df["date"] = pd.to_datetime(df["date"])
print("\nDate range:", df["date"].min(), "to", df["date"].max())

# --- Coverage checks ---
print("\nUnique stores:", df["store"].nunique())
print("Unique items:", df["item"].nunique())
print("Expected store-item combos:", df["store"].nunique() * df["item"].nunique())
print("Actual unique store-item combos:", df.groupby(["store", "item"]).ngroups)

# --- Sales sanity checks ---
print("\nSales summary:\n", df["sales"].describe())
print("\nAny negative sales?", (df["sales"] < 0).any())
print("Zero-sales rows:", (df["sales"] == 0).sum())

# --- Rows per series (should be consistent if no gaps) ---
rows_per_series = df.groupby(["store", "item"]).size()
print("\nRows per store-item series - min/max:", rows_per_series.min(), rows_per_series.max())