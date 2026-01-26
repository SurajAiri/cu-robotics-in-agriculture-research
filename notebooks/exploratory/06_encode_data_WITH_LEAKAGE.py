import pandas as pd
import os
import joblib
from sklearn.preprocessing import OneHotEncoder

INPUT_PATH = "data/processed/crop_yield_normalized.csv"
OUTPUT_PATH = "data/processed/crop_yield_encoded.csv"
PREPROCESSOR_DIR = "models/preprocessors"
ENCODER_PATH = os.path.join(PREPROCESSOR_DIR, "onehot_encoder.joblib")


def main():
    if not os.path.exists(INPUT_PATH):
        print(f"Error: {INPUT_PATH} not found.")
        return

    os.makedirs(PREPROCESSOR_DIR, exist_ok=True)

    print(f"Loading normalized data from {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)

    # Identify categorical columns
    cat_cols = ["Crop", "Season", "State"]
    # Verify they exist
    existing_cat_cols = [c for c in cat_cols if c in df.columns]

    if not existing_cat_cols:
        print("No categorical columns found.")
        return

    print(f"Encoding columns: {existing_cat_cols}")

    # One-Hot Encoding
    # sparse_output=False allows us to stick it back into a pandas DataFrame easily
    # handle_unknown='ignore' ensures robustness if new categories appear later (though sparse=False usually wants errors usually)
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore", dtype=int)

    encoded_data = encoder.fit_transform(df[existing_cat_cols])
    encoded_feature_names = encoder.get_feature_names_out(existing_cat_cols)

    encoded_df = pd.DataFrame(
        encoded_data, columns=encoded_feature_names, index=df.index
    )

    # Concatenate
    df_final = pd.concat([df.drop(columns=existing_cat_cols), encoded_df], axis=1)

    print(f"Encoded shape: {df_final.shape}")

    # Save Encoder
    print(f"Saving encoder to {ENCODER_PATH}...")
    joblib.dump(encoder, ENCODER_PATH)

    # Save Data
    print(f"Saving encoded data to {OUTPUT_PATH}...")
    df_final.to_csv(OUTPUT_PATH, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
