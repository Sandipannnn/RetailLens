"""
Cleans the raw Store Item Demand dataset and freezes a processed version.
Owner: Sandipan (Data Engineer)

Run from the project root:
    python src/clean_data.py
"""

import os
import pandas as pd
from src.data_generator import generate_sample_sales_data

RAW_PATH = "data/raw/train.csv"
PROCESSED_PATH = "data/processed/sales_clean.csv"

def clean_data():
    if not os.path.exists(RAW_PATH):
        print(f"Raw dataset not found at {RAW_PATH}. Generating sample dataset...")
        generate_sample_sales_data(output_path=RAW_PATH)

    df = pd.read_csv(RAW_PATH)

    # Defensive checks (dataset is already clean, but re-verify in case
    # someone reruns this on a different export)
    missing = df.isnull().sum().sum()
    dupes = df.duplicated().sum()
    negatives = (df["sales"] < 0).sum()

    if missing > 0:
        print(f"Warning: {missing} missing values found — dropping rows.")
        df = df.dropna()

    if dupes > 0:
        print(f"Warning: {dupes} duplicate rows found — dropping.")
        df = df.drop_duplicates()

    if negatives > 0:
        print(f"Warning: {negatives} negative sales values found — check before proceeding.")

    # Datetime conversion
    df["date"] = pd.to_datetime(df["date"])

    # Sort so the series are in proper chronological order per store-item
    df = df.sort_values(["store", "item", "date"]).reset_index(drop=True)

    # Save frozen processed dataset
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Saved cleaned dataset: {df.shape} -> {PROCESSED_PATH}")

if __name__ == "__main__":
    clean_data()