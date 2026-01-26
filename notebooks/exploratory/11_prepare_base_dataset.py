import pandas as pd
import numpy as np
import os

# --- Configuration ---
RAW_DATA_PATH = "data/raw/crop_yield.csv"
PROCESSED_DATA_DIR = "data/processed"
BASE_DATASET_PATH = os.path.join(PROCESSED_DATA_DIR, "crop_yield_cleaned_base.csv")


def load_and_basic_clean(path):
    """Load data and perform basic cleaning (strings, NaNs, zeros)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")

    df = pd.read_csv(path)
    print(f"Initial Shape: {df.shape}")

    # 1. Standardize column names
    df.columns = df.columns.str.strip()

    # 2. Strip string columns
    str_cols = df.select_dtypes(include=["object"]).columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()

    # 3. Drop duplicates
    initial_len = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {initial_len - len(df)} duplicates.")

    # 4. Drop missing values (Rows with any NaN)
    # Since we have plenty of data, dropping is safer than imputing for a "base" cleaner
    initial_len = len(df)
    df = df.dropna()
    print(f"Dropped {initial_len - len(df)} rows with missing values.")

    # 5. Sanity Checks (Physical Constraints - Row wise)
    # Area must be > 0
    df = df[df["Area"] > 0]
    # Inputs/Yield cannot be negative
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df = df[df[col] >= 0]

    print(f"Shape after sanity checks: {df.shape}")
    return df


def row_wise_feature_engineering(df):
    """
    Apply Feature Engineering that is purely row-based (NO global stats).
    Safe to do before splitting.
    """
    df = df.copy()

    # 1. Create Per-Unit Features
    # Logic: Fertilizer and Pesticide are totals, need to be per hectare
    df["Fertilizer_per_Area"] = df["Fertilizer"] / df["Area"]
    df["Pesticide_per_Area"] = df["Pesticide"] / df["Area"]

    # 2. Remove Redundant/Leakage Columns
    # Production is (Yield * Area). Since we predict Yield, and know Area,
    # Production is effectively the target.
    # Raw Fertilizer/Pesticide totals are replaced by rates.
    cols_to_drop = ["Production", "Fertilizer", "Pesticide"]
    df = df.drop(columns=cols_to_drop)

    return df


def remove_outliers_iqr(df):
    """
    Remove Statistical Outliers using IQR.

    NOTE ON DATA LEAKAGE:
    Technically, calculating IQR on the full dataset leaks information about the
    distribution of the test set into the training set (e.g., what constitutes an 'extreme' value).

    However, for the purpose of "Data Cleaning" (removing garbage/sensor errors),
    this is often performed on the base dataset to ensure the model doesn't train
    on impossible or highly erroneous data points.

    We apply it here to create a "Clean Base", but be aware of this distinction.
    """
    # Columns to check for outliers
    numeric_cols = [
        "Yield",
        "Annual_Rainfall",
        "Fertilizer_per_Area",
        "Pesticide_per_Area",
    ]

    initial_len = len(df)

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        # Using a standard 1.5 factor
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    print(f"Outlier Removal (IQR): Dropped {initial_len - len(df)} rows.")
    return df


def main():
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    print("--- 1. Loading & Basic Cleaning ---")
    df = load_and_basic_clean(RAW_DATA_PATH)

    print("\n--- 2. Row-wise Feature Engineering ---")
    df = row_wise_feature_engineering(df)

    print("\n--- 3. Outlier Removal (Cleaning) ---")
    df = remove_outliers_iqr(df)

    print(f"\nFinal Base Dataset Shape: {df.shape}")
    print(df.head())

    print(f"\nSaving base dataset to {BASE_DATASET_PATH}...")
    df.to_csv(BASE_DATASET_PATH, index=False)
    print("Done. This file is robustly cleaned but NOT normalized/encoded.")


if __name__ == "__main__":
    main()
