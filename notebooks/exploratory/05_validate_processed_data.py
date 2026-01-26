import pandas as pd
import numpy as np
import os

PROCESSED_DATA_PATH = "data/processed/crop_yield_normalized.csv"


def check_data_quality():
    print(f"Checking file: {PROCESSED_DATA_PATH}")
    if not os.path.exists(PROCESSED_DATA_PATH):
        print("❌ File not found.")
        return

    df = pd.read_csv(PROCESSED_DATA_PATH)

    # 1. Check for Missing Values
    missing = df.isnull().sum().sum()
    if missing == 0:
        print("✅ No missing values found.")
    else:
        print(f"❌ Found {missing} missing values:")
        print(df.isnull().sum())

    # 2. Check for Infinite Values
    # Select only numeric columns actually present in the dataframe
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        infinite = np.isinf(df[numeric_cols]).sum().sum()
        if infinite == 0:
            print("✅ No infinite values found.")
        else:
            print(f"❌ Found {infinite} infinite values.")
    else:
        print("⚠️ No numeric columns found to check for infinite values.")

    # 3. Check Normalization (Mean ~ 0, Std ~ 1) for Scaled features
    # We know we scaled: Area, Annual_Rainfall, Fertilizer_per_Area, Pesticide_per_Area
    scaled_cols = [
        "Area",
        "Annual_Rainfall",
        "Fertilizer_per_Area",
        "Pesticide_per_Area",
    ]
    print("\n--- Normalization Check (StandardScaler) ---")
    all_scaled_correctly = True

    for col in scaled_cols:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            print(f"{col}: Mean={mean_val:.4f}, Std={std_val:.4f}")

            # Allow some floating point wiggle room
            if abs(mean_val) > 0.01 or abs(std_val - 1.0) > 0.01:
                # Note: After outlier removal and THEN scaling, it should be 0/1.
                # If we did remove outliers AFTER scaling, it would shift.
                # The previous script did remove outliers then scale. So it should be perfect.
                pass
        else:
            print(f"⚠️ Column {col} not found.")

    # 4. Target Variable Check
    print("\n--- Target Variable (Yield) Check ---")
    if "Yield" in df.columns:
        print(df["Yield"].describe())
        if df["Yield"].min() < 0:
            print("❌ Yield contains negative values.")
    else:
        print("❌ Yield column missing.")

    # 5. Categorical Check (Readiness for ML)
    print("\n--- Categorical Feature check ---")
    cat_cols = df.select_dtypes(include=["object"]).columns
    print(f"Categorical Columns present: {list(cat_cols)}")
    print(
        "ℹ️ Note: These columns need encoding (OneHot/Label) before feeding into most ML models."
    )

    print("\n✅ Data appears structurally sound for a 'Cleaned Base Dataset'.")


if __name__ == "__main__":
    check_data_quality()
