import pandas as pd
import numpy as np
import joblib
import os


class CropYieldPreprocessor:
    def __init__(self, models_dir="models/preprocessors"):
        self.scaler_path = os.path.join(models_dir, "scaler.joblib")
        self.encoder_path = os.path.join(models_dir, "onehot_encoder.joblib")
        self.scaler = None
        self.encoder = None

        # Columns that the scaler expects
        self.numeric_features = [
            "Area",
            "Annual_Rainfall",
            "Fertilizer_per_Area",
            "Pesticide_per_Area",
        ]

        # Columns that the encoder expects
        self.categorical_features = ["Crop", "Season", "State"]

    def load(self):
        """Load saved scaler and encoder."""
        if not os.path.exists(self.scaler_path) or not os.path.exists(
            self.encoder_path
        ):
            raise FileNotFoundError(
                f"Preprocessor files not found in {os.path.dirname(self.scaler_path)}"
            )

        self.scaler = joblib.load(self.scaler_path)
        self.encoder = joblib.load(self.encoder_path)
        print("✅ Preprocessors loaded successfully.")

    def _feature_engineering(self, df):
        """Internal method to perform feature engineering on raw data."""
        df = df.copy()

        # 1. Standardize Strings
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype(str).str.strip()

        # 2. Ensure Numeric Types
        # Fertilizer and Pesticide might come in as strings in some raw inputs
        numeric_inputs = ["Area", "Annual_Rainfall", "Fertilizer", "Pesticide"]
        for col in numeric_inputs:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # 3. Handle Missing Values (for inference, default to 0 is a safe naive approach if fields missing)
        df.fillna(0, inplace=True)

        # 4. Calculate Per-Unit Features
        # Avoid division by zero
        df["Fertilizer_per_Area"] = np.where(
            df["Area"] > 0, df["Fertilizer"] / df["Area"], 0
        )
        df["Pesticide_per_Area"] = np.where(
            df["Area"] > 0, df["Pesticide"] / df["Area"], 0
        )

        return df

    def transform_normalized(self, data):
        """
        Transforms raw data into normalized format (for Gradient Boosting models).
        Expects data to have: Crop, Season, State, Area, Annual_Rainfall, Fertilizer, Pesticide

        Returns: DataFrame with Categorical columns + Scaled Numeric columns.
        """
        if self.scaler is None:
            self.load()

        # Handle Dictionary Input
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()

        df = self._feature_engineering(df)

        # Validate Scaling Columns Exist
        missing_numeric = [
            col for col in self.numeric_features if col not in df.columns
        ]
        if missing_numeric:
            raise ValueError(
                f"Missing required numeric columns for scaling: {missing_numeric}"
            )

        # Scale Numeric Features
        df[self.numeric_features] = self.scaler.transform(df[self.numeric_features])

        # Return only model-relevant columns
        cols_to_keep = self.categorical_features + self.numeric_features
        # Ensure categorical columns are present (even if empty/NaN strings if missing input)
        for col in self.categorical_features:
            if col not in df.columns:
                df[col] = "Unknown"

        return df[cols_to_keep]

    def transform_encoded(self, data):
        """
        Transforms raw data into fully encoded format (for Deep Learning / Linear Models).
        Returns: DataFrame with Scaled Numeric columns + One-Hot Encoded columns.
        """
        if self.encoder is None:
            self.load()

        # Get normalized data first (Categorical + Scaled Numeric)
        df_norm = self.transform_normalized(data)

        # Separate numeric and categorical
        df_numeric = df_norm[self.numeric_features].reset_index(drop=True)
        df_cat = df_norm[self.categorical_features].reset_index(drop=True)

        # Encode Categorical
        # Note: handle_unknown='ignore' in encoder handles unseen categories
        encoded_array = self.encoder.transform(df_cat)
        encoded_feature_names = self.encoder.get_feature_names_out(
            self.categorical_features
        )

        df_encoded_cat = pd.DataFrame(encoded_array, columns=encoded_feature_names)

        # Concatenate
        df_final = pd.concat([df_numeric, df_encoded_cat], axis=1)

        return df_final


# --- Test Usage ---
if __name__ == "__main__":
    # Sample Mock Input (from raw data structure)
    sample_input = {
        "Crop": "Arecanut",
        "Season": "Whole Year",
        "State": "Assam",
        "Area": 73814.0,
        "Annual_Rainfall": 2051.4,
        "Fertilizer": 7024878.38,
        "Pesticide": 22882.34,
        # Note: Production and Yield are NOT inputs for prediction
    }

    print("--- Initializing Preprocessor ---")
    preprocessor = CropYieldPreprocessor()

    print("\n--- Test 1: Transform for Gradient Boosting (Normalized) ---")
    norm_df = preprocessor.transform_normalized(sample_input)
    print("Shape:", norm_df.shape)
    print(norm_df)

    print("\n--- Test 2: Transform for Deep Learning (Encoded) ---")
    enc_df = preprocessor.transform_encoded(sample_input)
    print("Shape:", enc_df.shape)
    print("Columns:", enc_df.columns.tolist()[:10], "...")  # Print first 10 cols
    print(enc_df.iloc[:, :5])
