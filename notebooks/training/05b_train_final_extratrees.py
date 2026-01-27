import pandas as pd
import numpy as np
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import ExtraTreesRegressor

# --- Configuration ---
DATA_PATH = "data/processed/crop_yield_cleaned_base.csv"
MODEL_DIR = "models/production"
MODEL_FILENAME = "rollback_extratrees_pipeline.joblib"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found.")

    print("--- Loading Full Dataset for Production Training (Rollback Model) ---")
    df = pd.read_csv(DATA_PATH)

    target = "Yield"
    X = df.drop(columns=[target])
    y = df[target]

    print(f"Training on {len(df)} samples...")

    # Preprocessing
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

    # Defined Extra Trees with Best Hyperparameters
    # RMSE=0.3636, MAE=0.1926, R2=0.8524
    # Tuned params: 'regressor__bootstrap': True, 'regressor__min_samples_leaf': 2, 'regressor__min_samples_split': 5, 'regressor__n_estimators': 800
    et_model = ExtraTreesRegressor(
        n_estimators=800,
        min_samples_split=5,
        min_samples_leaf=2,
        bootstrap=True,
        # max_depth=None, # Default
        # max_features=None, # Default (auto/sqrt/log2 - Tuned was None aka 1.0)
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", et_model),
        ]
    )

    print("Fitting model on full dataset...")
    pipeline.fit(X, y)

    print(f"Saving rollback model to {MODEL_PATH}...")
    joblib.dump(pipeline, MODEL_PATH)
    print("Done.")

    # Optional: Quick verification
    print("\n--- Quick In-Sample Verification ---")
    y_pred = pipeline.predict(X.head())
    print("Actual:", y.head().values)
    print("Predicted:", y_pred)


if __name__ == "__main__":
    main()
