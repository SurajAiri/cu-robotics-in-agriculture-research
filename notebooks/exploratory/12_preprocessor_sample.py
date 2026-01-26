import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# --- Configuration ---
BASE_DATA_PATH = "data/processed/crop_yield_cleaned_base.csv"
MODEL_DIR = "models/preprocessors"

# Output files
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor_pipeline.joblib")
# We usually don't save "train/test" CSVs unless for debugging,
# but let's save a "sample_processed_train.csv" to show what it looks like.


def main():
    if not os.path.exists(BASE_DATA_PATH):
        print(
            f"Error: {BASE_DATA_PATH} not found. Run 11_prepare_base_dataset.py first."
        )
        return

    os.makedirs(MODEL_DIR, exist_ok=True)

    print("--- Loading Base Dataset ---")
    df = pd.read_csv(BASE_DATA_PATH)
    print(f"Shape: {df.shape}")

    # --- 1. Split Data (Train/Test) ---
    # This is CRITICAL. We must split BEFORE fitting scalers/encoders
    # to avoid data leakage.

    # Define Features (X) and Target (y)
    target = "Yield"
    X = df.drop(columns=[target])
    y = df[target]

    # Stratify by State to ensure all states are represented in Train & Test
    # (Optional but good for crop data across regions)
    print("\n--- Splitting Data ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=X["State"]
    )
    print(f"Train Set: {X_train.shape}")
    print(f"Test Set:  {X_test.shape}")

    # --- 2. Define Preprocessing Pipeline ---
    # Identify column types
    numeric_features = [
        "Area",
        "Annual_Rainfall",
        "Fertilizer_per_Area",
        "Pesticide_per_Area",
    ]
    categorical_features = ["Crop", "Season", "State"]

    # Ensure they exist (sanity check)
    missing_nums = [c for c in numeric_features if c not in X.columns]
    missing_cats = [c for c in categorical_features if c not in X.columns]
    if missing_nums or missing_cats:
        print(f"Missing columns! Nums: {missing_nums}, Cats: {missing_cats}")
        return

    # Create Transformers
    # Numeric: Standard Scaling (Z-score)
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    # Categorical: One-Hot Encoding
    # handle_unknown='ignore' so model doesn't crash if new crop appears in production
    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )

    # Combine into ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",  # Drop any other cols not specified
    )

    # --- 3. Fit on TRAIN Data Only ---
    print("\n--- Fitting Preprocessor on TRAIN data ---")
    preprocessor.fit(X_train)

    # --- 4. Save the Fitted Preprocessor ---
    print(f"Saving preprocessor to {PREPROCESSOR_PATH}...")
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    # --- 5. Verify Transformation ---
    # Transform Train (for checking)
    X_train_processed = preprocessor.transform(X_train)

    # Get feature names (StandardScaler keeps names usually, OneHot makes new ones)
    # Note: ColumnTransformer makes getting names distinct slightly complex in older sklearn,
    # but let's try standard approach.
    try:
        num_names = numeric_features
        cat_names = preprocessor.named_transformers_["cat"][
            "onehot"
        ].get_feature_names_out(categorical_features)
        feature_names = np.r_[num_names, cat_names]

        X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
        print("\n--- Sample Processed Train Data ---")
        print(X_train_df.head())
        print(f"Processed Shape: {X_train_df.shape}")
    except Exception as e:
        print(f"Could not reconstruction dataframe with names (minor issue): {e}")

    print("\n✅ Success. The pipeline is fitted on Train data without leakage.")
    print("You can now load 'preprocessor_pipeline.joblib' to transform new data.")


if __name__ == "__main__":
    main()
