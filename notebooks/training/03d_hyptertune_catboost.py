import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import randint, uniform
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

# --- Configuration ---
DATA_PATH = "data/processed/crop_yield_cleaned_base.csv"
MODEL_SAVE_PATH = "models/boosting_regressors/best_tuned_catboost.joblib"
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)


# --- Custom Wrapper for CatBoost to fix sklearn compatibility (AttributeError: __sklearn_tags__) ---
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
        # self.estimator_ is not initialized here to comply with sklearn check_is_fitted logic

    def fit(self, X, y):
        # Initialize the actual CatBoostRegressor with current parameters
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

    print("--- Loading Data for CatBoost Tuning ---")
    df = pd.read_csv(DATA_PATH)

    target = "Yield"
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=X["State"]
    )

    numeric_features = [
        "Area",
        "Annual_Rainfall",
        "Fertilizer_per_Area",
        "Pesticide_per_Area",
    ]
    categorical_features = ["Crop", "Season", "State"]

    # CatBoost works best with categorical features specified directly,
    # but to maintain Pipeline compatibility with RandomizedSearchCV (standard sklearn),
    # we'll stick to OHE.
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

    # Use the Wrapper instead of CatBoostRegressor directly
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", CatBoostWrapper(random_state=42, silent=True)),
        ]
    )

    param_dist = {
        "regressor__iterations": randint(500, 1500),  # Increased range slightly
        "regressor__learning_rate": uniform(0.01, 0.2),
        "regressor__depth": randint(4, 10),
        "regressor__l2_leaf_reg": randint(1, 10),
    }

    print("Starting RandomizedSearchCV for CatBoost (Wrapped)...")

    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=10,
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1,
        scoring="neg_root_mean_squared_error",
    )

    random_search.fit(X_train, y_train)

    print(f"\nBest Parameters: {random_search.best_params_}")
    print(f"Best CV Score (RMSE): {-random_search.best_score_:.4f}")

    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print("\n--- Test Set Evaluation ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R2:   {r2:.4f}")

    joblib.dump(best_model, MODEL_SAVE_PATH)
    print(f"Saved tuned model to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
