import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import randint

# --- Configuration ---
DATA_PATH = "data/processed/crop_yield_cleaned_base.csv"
MODEL_SAVE_PATH = "models/scikit_regressors/best_tuned_extra_trees.joblib"
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found.")

    print("--- Loading Data for Extra Trees Tuning ---")
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

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", ExtraTreesRegressor(random_state=42, n_jobs=-1)),
        ]
    )

    # Extra Trees params are similar to Random Forest
    param_dist = {
        "regressor__n_estimators": randint(100, 500),
        "regressor__max_depth": [None, 10, 20, 30, 40],
        "regressor__min_samples_split": randint(2, 11),
        "regressor__min_samples_leaf": randint(1, 5),
        "regressor__max_features": ["sqrt", "log2", None],
        "regressor__bootstrap": [
            True,
            False,
        ],  # Extra Trees can optionally use bootstrap
    }

    print("Starting RandomizedSearchCV...")
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=20,
        cv=3,
        verbose=2,
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
