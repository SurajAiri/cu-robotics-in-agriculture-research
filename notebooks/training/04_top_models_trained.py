import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

# --- Configuration ---
DATA_PATH = "data/processed/crop_yield_cleaned_base.csv"
RESULTS_PATH = "results/top_model_result.csv"
os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)


# --- Custom Wrapper for CatBoost ---
class CatBoostWrapper(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        iterations=500,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3,
        random_state=42,
        silent=True,
    ):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.random_state = random_state
        self.silent = silent

    def fit(self, X, y):
        self.estimator_ = CatBoostRegressor(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            random_state=self.random_state,
            silent=self.silent,
            allow_writing_files=False,
        )
        self.estimator_.fit(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self)
        return self.estimator_.predict(X)


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found.")

    print("--- Loading Data ---")
    df = pd.read_csv(DATA_PATH)

    target = "Yield"
    X = df.drop(columns=[target])
    y = df[target]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=X["State"]
    )

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

    # Define Models with Hypertuned Parameters
    # Values taken from results/hypertuned_info.md
    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=533,
            min_samples_split=4,
            min_samples_leaf=1,
            max_depth=25,
            random_state=42,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=890,
            min_samples_split=4,
            min_samples_leaf=1,
            max_depth=29,
            bootstrap=False,
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=975,
            learning_rate=0.11,
            max_depth=9,
            subsample=0.82,
            colsample_bytree=0.87,
            random_state=42,
            n_jobs=-1,
        ),
        "CatBoost": CatBoostWrapper(
            depth=9,
            iterations=808,
            l2_leaf_reg=2,
            learning_rate=0.15439975445336496,
            random_state=42,
            silent=True,
        ),
    }

    results = []

    print("--- Training Top Models on Train/Test Split ---")
    for name, model in models.items():
        print(f"Training {name}...")
        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("regressor", model)]
        )

        # Train
        pipeline.fit(X_train, y_train)

        # Predict
        y_pred = pipeline.predict(X_test)

        # Evaluate
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"  RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

        results.append({"Model": name, "RMSE": rmse, "MAE": mae, "R2": r2})

    # Save Results
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved performance summary to {RESULTS_PATH}")
    print(results_df)


if __name__ == "__main__":
    main()
