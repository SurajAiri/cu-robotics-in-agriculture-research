import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# --- Configuration ---
RAW_DATA_PATH = "data/raw/crop_yield.csv"
PROCESSED_DATA_DIR = "data/processed"
PREPROCESSOR_DIR = "models/preprocessors"
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "crop_yield_normalized.csv")
SCALER_PATH = os.path.join(PREPROCESSOR_DIR, "scaler.joblib")


def load_and_basic_clean(path):
    """Load data and perform basic cleaning (strings, NaNs, zeros)."""
    df = pd.read_csv(path)

    # Standardize column names
    df.columns = df.columns.str.strip()

    # Strip string columns
    str_cols = df.select_dtypes(include=["object"]).columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    # Drop duplicates
    df = df.drop_duplicates()

    # Drop missing values
    df = df.dropna()

    # Drop rows with 0 Area (can't calculate per-unit)
    df = df[df["Area"] > 0]

    # Drop likely invalid data (negative numbers)
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df = df[df[col] >= 0]

    return df


def feature_engineering(df):
    """Create per-unit features and drop redundant ones."""
    # Ensure fertilizer and pesticide are per unit area
    # Note: User hypothesis: "fertilizer, pesticides are for total area"
    # We create normalized features
    df["Fertilizer_per_Area"] = df["Fertilizer"] / df["Area"]
    df["Pesticide_per_Area"] = df["Pesticide"] / df["Area"]

    # Production is logically Yield * Area. Since we predict Yield,
    # and Area is a feature, Production is redundant/leakage.
    cols_to_drop = ["Production", "Fertilizer", "Pesticide"]
    df = df.drop(columns=cols_to_drop)

    return df


def remove_outliers(df, cols, method="IQR", factor=1.5):
    """Remove outliers from specified columns."""
    initial_len = len(df)

    if method == "IQR":
        for col in cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR

            # Simple filtering
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    print(
        f"Outlier Removal ({method}): Reduced data from {initial_len} to {len(df)} rows."
    )
    return df


def normalize_features(df):
    """Normalize numerical features using StandardScaler."""
    scaler = StandardScaler()

    # Identify numerical columns to scale
    # We typically scale inputs using Z-score.
    # We might NOT want to scale the Target (Yield) depending on the model,
    # but for "normalized dataset" usually inputs are scaled.
    # Let's scale all continuous numerical features: Annual_Rainfall, Area, Fertilizer_per_Area, Pesticide_per_Area
    # Target 'Yield' is usually kept scaling-free or scaled separately.
    # Let's scale inputs only to keep Yield interpretable, or scale all.
    # User said "normalize dataset". Let's scale inputs.

    scale_cols = [
        "Area",
        "Annual_Rainfall",
        "Fertilizer_per_Area",
        "Pesticide_per_Area",
    ]

    # It's good practice to fit scalar on train only, but for this "processing" step
    # we are preparing the whole dataset.
    df[scale_cols] = scaler.fit_transform(df[scale_cols])

    return df, scaler


def main():
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(PREPROCESSOR_DIR, exist_ok=True)

    print("Loading data...")
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Error: {RAW_DATA_PATH} not found.")
        return

    df = load_and_basic_clean(RAW_DATA_PATH)
    print(f"Data Loaded & Basic Cleaned: {df.shape}")

    # Feature Engineering
    df = feature_engineering(df)
    print(f"After Feature Engineering: {df.shape}")
    print(
        "Created 'Fertilizer_per_Area', 'Pesticide_per_Area'. Dropped 'Production', 'Fertilizer', 'Pesticide'."
    )

    # Outlier Removal
    # Focus on physical constraints and extreme anomalies
    outlier_check_cols = [
        "Yield",
        "Annual_Rainfall",
        "Fertilizer_per_Area",
        "Pesticide_per_Area",
    ]
    df = remove_outliers(df, outlier_check_cols, method="IQR", factor=1.5)

    # Normalize
    # Note: We are transforming the dataframe in place.
    df, scaler = normalize_features(df)
    print("Numeric features normalized (StandardScaler).")

    # Save Scaler
    print(f"Saving scaler to {SCALER_PATH}...")
    joblib.dump(scaler, SCALER_PATH)

    # Save
    print(f"Saving to {PROCESSED_DATA_PATH}...")
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print("Done.")

    print("\nSample of processed data:")
    print(df.head())


if __name__ == "__main__":
    main()
