import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from scipy.stats import randint

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


def tune_model(pipeline, param_dist, X_train, y_train, model_name):
    print(f"\n--- Tuning {model_name} ---")
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=30,
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1,
        scoring="neg_root_mean_squared_error",
    )
    random_search.fit(X_train, y_train)
    print(f"Best params for {model_name}: {random_search.best_params_}")
    return random_search.best_estimator_


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

    # Using StandardScaler as requested/consistent with previous good results
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

    models = {}

    # 1. Random Forest - Re-tune
    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(random_state=42, n_jobs=-1)),
        ]
    )
    rf_params = {
        "regressor__n_estimators": randint(200, 1000),  # Expanded range to cover ~500
        "regressor__max_depth": [None, 10, 20, 30, 40, 50],
        "regressor__min_samples_split": randint(2, 11),
        "regressor__min_samples_leaf": randint(1, 10),
        "regressor__max_features": ["sqrt", "log2", None],  # Crucial parameter
    }
    models["Random Forest"] = tune_model(
        rf_pipeline, rf_params, X_train, y_train, "Random Forest"
    )

    # 2. Extra Trees - Re-tune
    et_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", ExtraTreesRegressor(random_state=42, n_jobs=-1)),
        ]
    )
    et_params = {
        "regressor__n_estimators": randint(200, 1000),  # Expanded range to cover ~890
        "regressor__max_depth": [None, 10, 20, 30, 40, 50],
        "regressor__min_samples_split": randint(2, 11),
        "regressor__min_samples_leaf": randint(1, 10),
        "regressor__max_features": ["sqrt", "log2", None],
        "regressor__bootstrap": [True, False],
    }
    models["Extra Trees"] = tune_model(
        et_pipeline, et_params, X_train, y_train, "Extra Trees"
    )

    # 3. XGBoost - Use Champion parameters from previous step
    print("\n--- Configuring XGBoost (Fixed) ---")
    xgb_model = XGBRegressor(
        n_estimators=975,
        learning_rate=0.11,
        max_depth=9,
        subsample=0.82,
        colsample_bytree=0.87,
        random_state=42,
        n_jobs=-1,
    )
    models["XGBoost"] = Pipeline(
        steps=[("preprocessor", preprocessor), ("regressor", xgb_model)]
    )
    # Fit fixed model
    models["XGBoost"].fit(X_train, y_train)

    # 4. CatBoost - Use Champion parameters from previous step
    print("\n--- Configuring CatBoost (Fixed) ---")
    cat_model = CatBoostWrapper(
        depth=9,
        iterations=808,
        l2_leaf_reg=2,
        learning_rate=0.15439975445336496,
        random_state=42,
        silent=True,
    )
    models["CatBoost"] = Pipeline(
        steps=[("preprocessor", preprocessor), ("regressor", cat_model)]
    )
    # Fit fixed model
    models["CatBoost"].fit(X_train, y_train)

    # --- Evaluation ---
    results = []
    print("\n--- Final Evaluation on Test Set ---")
    for name, pipeline in models.items():
        y_pred = pipeline.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"{name}: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

        results.append({"Model": name, "RMSE": rmse, "MAE": mae, "R2": r2})

    # Save Results
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved performance summary to {RESULTS_PATH}")
    print(results_df)


if __name__ == "__main__":
    main()
