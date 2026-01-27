import pandas as pd
import joblib
import os
import numpy as np

# --- Configuration ---
# Paths assuming script is run from project root
MODEL_DIR = "models/production"
CHAMPION_MODEL_PATH = os.path.join(MODEL_DIR, "champion_xgboost_pipeline.joblib")
ROLLBACK_MODEL_PATH = os.path.join(MODEL_DIR, "rollback_extratrees_pipeline.joblib")


def load_prediction_model(model_path):
    """
    Load a trained model pipeline from disk.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    print(f"Loading model from {model_path}...")
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")


def get_sample_input():
    """
    Create a sample DataFrame matching the training schema.
    Schema expected by Pipeline:
    - Numerical: [Area, Annual_Rainfall, Fertilizer_per_Area, Pesticide_per_Area]
    - Categorical: [Crop, Season, State]
    """
    data = [
        {
            "Crop": "Rice",
            "Season": "Kharif",
            "State": "Punjab",
            "Area": 1200.0,
            "Annual_Rainfall": 650.0,
            "Fertilizer_per_Area": 145.5,
            "Pesticide_per_Area": 0.45,
        },
        {
            "Crop": "Wheat",
            "Season": "Rabi",
            "State": "Uttar Pradesh",
            "Area": 2500.0,
            "Annual_Rainfall": 800.0,
            "Fertilizer_per_Area": 120.0,
            "Pesticide_per_Area": 0.30,
        },
        {
            "Crop": "Maize",
            "Season": "Kharif",
            "State": "Karnataka",
            "Area": 500.0,
            "Annual_Rainfall": 1200.0,
            "Fertilizer_per_Area": 100.0,
            "Pesticide_per_Area": 0.10,
        },
    ]
    return pd.DataFrame(data)


def predict_with_fallback(input_df, champion_model, rollback_model):
    """
    Demonstrate a production inference logic with fallback.
    """
    try:
        print("\nAttempting inference with Champion Model (XGBoost)...")
        predictions = champion_model.predict(input_df)
        source = "Champion (XGBoost)"
    except Exception as e:
        print(f"ERROR: Champion model inference failed: {e}")
        print("Switching to Rollback Model (Extra Trees)...")
        try:
            predictions = rollback_model.predict(input_df)
            source = "Rollback (Extra Trees)"
        except Exception as e2:
            print(f"CRITICAL: Rollback model also failed: {e2}")
            return None, "FAILED"

    return predictions, source


def main():
    # 1. Load Models
    try:
        champion_pipeline = load_prediction_model(CHAMPION_MODEL_PATH)
        rollback_pipeline = load_prediction_model(ROLLBACK_MODEL_PATH)
    except Exception as e:
        print(f"Initialization Failed: {e}")
        return

    # 2. Get Data
    input_df = get_sample_input()
    print("\n--- Input Data ---")
    print(input_df)

    # 3. Run Inference
    predictions, source = predict_with_fallback(
        input_df, champion_pipeline, rollback_pipeline
    )

    # 4. Show Results
    if predictions is not None:
        print(f"\n--- Inference Successful (Source: {source}) ---")

        # Attach predictions to dataframe for display
        results = input_df.copy()
        results["Predicted_Yield"] = predictions

        print("\nResults:")
        print(results[["Crop", "State", "Predicted_Yield"]])

        # Verify both models manually for comparison (Demostration purpose only)
        print("\n--- Model Comparison (For Demo Only) ---")
        champ_preds = champion_pipeline.predict(input_df)
        roll_preds = rollback_pipeline.predict(input_df)

        comparison = pd.DataFrame(
            {
                "Crop": input_df["Crop"],
                "Champion_Pred": champ_preds,
                "Rollback_Pred": roll_preds,
                "Difference": np.abs(champ_preds - roll_preds),
            }
        )
        print(comparison)


if __name__ == "__main__":
    main()
