import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# --- Configuration ---
DATA_PATH = "data/processed/crop_yield_cleaned_base.csv"
MODEL_SAVE_DIR = "models/scikit_regressors"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


def train_evaluate_models():
    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found. Run clean/prep scripts first.")

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded data shape: {df.shape}")

    target = "Yield"
    X = df.drop(columns=[target])
    y = df[target]

    # 2. Split Data (Stratify by State for better distribution)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=X["State"]
    )

    # 3. Define Preprocessing
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

    # 4. Define Models to Explore
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, random_state=42
        ),
    }

    results = []

    print("\n--- Training & Evaluating Models ---")
    for name, model in models.items():
        # Create Pipeline
        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("regressor", model)]
        )

        # Train
        print(f"Training {name}...")
        pipeline.fit(X_train, y_train)

        # Predict
        y_pred = pipeline.predict(X_test)

        # Evaluate
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results.append(
            {
                "Model": name,
                "RMSE": rmse,
                "MAE": mae,
                "R2": r2,
                "Pipeline_Object": pipeline,
            }
        )

        print(f"  RMSE: {rmse:.4f}, R2: {r2:.4f}")

    # 5. Summary & Visualization
    results_df = pd.DataFrame(results).sort_values(by="RMSE")
    print("\n--- Final Leaderboard ---")
    print(results_df[["Model", "RMSE", "MAE", "R2"]])

    # Save best model
    best_model_info = results_df.iloc[0]
    best_model_name = best_model_info["Model"]
    best_pipeline = best_model_info["Pipeline_Object"]

    save_path = os.path.join(
        MODEL_SAVE_DIR, f"best_model_{best_model_name.replace(' ', '_').lower()}.joblib"
    )
    joblib.dump(best_pipeline, save_path)
    print(f"\nSaved best model ({best_model_name}) to {save_path}")

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x="RMSE", y="Model", data=results_df, palette="viridis")
    plt.title("Model Comparison (RMSE Lower is Better)")
    plt.show()


if __name__ == "__main__":
    train_evaluate_models()
