# this is old version of data processing script kept for reference
# newer version is in notebooks/exploratory/04_process_data.py
import pandas as pd
import numpy as np
import os

# --- Configuration ---
RAW_DATA_PATH = "data/raw/crop_yield.csv"
PROCESSED_DATA_DIR = "data/processed"
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "crop_yield_cleaned.csv")


def clean_column_names(df):
    """Standardize column names to lower_snake_case (optional, but good practice)."""
    # map current names for clarity based on source.md if needed,
    # but currently they match well (Crop, State, etc).
    # Let's just strip whitespace from column names.
    df.columns = df.columns.str.strip()
    return df


def clean_strings(df):
    """Strip whitespace and unify casing for categorical columns."""
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col] = df[col].str.strip()
        # Optional: df[col] = df[col].str.lower() # specific standardization
    return df


def handle_duplicates(df):
    """Remove duplicate rows."""
    initial_count = len(df)
    df = df.drop_duplicates()
    final_count = len(df)
    print(f"Removed {initial_count - final_count} duplicate rows.")
    return df


def handle_missing_values(df):
    """Handle missing values."""
    # Check for missing values first
    print("\nMissing values before handling:")
    print(df.isnull().sum())

    # Strategy: Drop rows with missing critical information.
    # If Yield, State, or Crop is missing, the row is likely useless for this prediction task.
    # For numeric features like Fertilizer/Pesticide, one might impute,
    # but let's drop for high-quality data first.
    initial_count = len(df)
    df = df.dropna()
    final_count = len(df)
    print(f"Dropped {initial_count - final_count} rows containing missing values.")

    return df


def clean_numeric_outliers(df):
    """Basic sanity checks for numerical data."""
    # Ensure no negative values for physical quantities
    cols_to_check = [
        "Area",
        "Production",
        "Annual_Rainfall",
        "Fertilizer",
        "Pesticide",
        "Yield",
    ]
    for col in cols_to_check:
        if col in df.columns:
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                print(
                    f"Found {negative_count} negative values in {col}. Removing entries."
                )
                df = df[df[col] >= 0]

    # Optional: Remove 0 values for Area if they exist (division by zero issues for yield calc)
    if "Area" in df.columns:
        zero_area = (df["Area"] == 0).sum()
        if zero_area > 0:
            print(f"Removing {zero_area} rows with 0 Area.")
            df = df[df["Area"] > 0]

    return df


def main():
    # Ensure processed directory exists
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    print("Loading data...")
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Error: Raw data not found at {RAW_DATA_PATH}")
        return

    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Initial shape: {df.shape}")

    # 1. Cleaning Strings
    print("Standardizing string format...")
    df = clean_column_names(df)
    df = clean_strings(df)

    # 2. Duplicates
    df = handle_duplicates(df)

    # 3. Missing Values
    df = handle_missing_values(df)

    # 4. Outliers / Sanity Checks
    df = clean_numeric_outliers(df)

    # 5. Feature Correction (if needed)
    # The source metadata mentions Yield = Production / Area.
    # We can verify this or re-calculate to ensure consistency.
    # Be careful with division by zero implies Area > 0 check above.
    # df['Yield_Calculated'] = df['Production'] / df['Area']

    print(f"Final shape: {df.shape}")
    print(df.head())

    # Save
    print(f"Saving processed data to {PROCESSED_DATA_PATH}...")
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
