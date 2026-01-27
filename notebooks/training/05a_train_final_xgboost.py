import pandas as pd
import numpy as np
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBRegressor

# --- Configuration ---
DATA_PATH = "data/processed/crop_yield_cleaned_base.csv"
MODEL_DIR = "models/production"
MODEL_FILENAME = "champion_xgboost_pipeline.joblib"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found.")

    print("--- Loading Full Dataset for Production Training ---")
    df = pd.read_csv(DATA_PATH)

    target = "Yield"
    X = df.drop(columns=[target])
    y = df[target]

    print(f"Training on {len(df)} samples...")

    # Preprocessing
    # Using the same configuration as the successful experiment
    numeric_features = [
        "Area",
        "Annual_Rainfall",
        "Fertilizer_per_Area",
        "Pesticide_per_Area",
    ]
    categorical_features = ["Crop", "Season", "State"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    # Defined XGBoost with Champion Hyperparameters
    # RMSE=0.3519, MAE=0.1938, R2=0.8617
    xgb_model = XGBRegressor(
        n_estimators=975,
        learning_rate=0.11,
        max_depth=9,
        subsample=0.82,
        colsample_bytree=0.87,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", xgb_model),
        ]
    )

    print("Fitting model on full dataset...")
    pipeline.fit(X, y)

    print(f"Saving production model to {MODEL_PATH}...")
    joblib.dump(pipeline, MODEL_PATH)
    print("Done.")

    # Optional: Quick verification
    print("\n--- Quick In-Sample Verification ---")
    y_pred = pipeline.predict(X.head())
    print("Actual:", y.head().values)
    print("Predicted:", y_pred)


if __name__ == "__main__":
    main()
