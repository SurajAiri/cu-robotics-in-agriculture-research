import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set plot style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

DATA_PATH = "data/raw/crop_yield.csv"


def analyze_relationships(df):
    """Analyze relationships between total inputs and Area."""
    print("\n--- Analyzing Area vs Input Relationships ---")

    # create temporary per-unit features
    df["Fertilizer_per_Area"] = df["Fertilizer"] / df["Area"]
    df["Pesticide_per_Area"] = df["Pesticide"] / df["Area"]

    # 1. Scatter plots: Total Input vs Area
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Total Fertilizer vs Area
    sns.scatterplot(x="Area", y="Fertilizer", data=df, ax=axes[0, 0], alpha=0.5)
    axes[0, 0].set_title("Total Fertilizer vs Area")

    # Total Pesticide vs Area
    sns.scatterplot(x="Area", y="Pesticide", data=df, ax=axes[0, 1], alpha=0.5)
    axes[0, 1].set_title("Total Pesticide vs Area")

    # Per-unit Fertilizer Dist
    sns.histplot(df["Fertilizer_per_Area"], ax=axes[1, 0], kde=True, bins=50)
    axes[1, 0].set_title("Distribution of Fertilizer per Unit Area")

    # Per-unit Pesticide Dist
    sns.histplot(df["Pesticide_per_Area"], ax=axes[1, 1], kde=True, bins=50)
    axes[1, 1].set_title("Distribution of Pesticide per Unit Area")

    plt.tight_layout()
    plt.show()  # In CLI this might not show, usually saved or just skipped.
    # Assuming user might run in interactive window or see code.

    # Correlation check
    print("\nCorrelation between Area and Total Inputs:")
    print(df[["Area", "Fertilizer", "Pesticide"]].corr())


def check_outliers(df):
    """Visualize outliers in key numerical columns."""
    print("\n--- Checking for Outliers ---")

    # Recalculate per-unit for checking
    df["Fertilizer_per_Area"] = df["Fertilizer"] / df["Area"]
    df["Pesticide_per_Area"] = df["Pesticide"] / df["Area"]

    cols_to_check = [
        "Yield",
        "Annual_Rainfall",
        "Fertilizer_per_Area",
        "Pesticide_per_Area",
    ]

    plt.figure(figsize=(15, 8))
    for i, col in enumerate(cols_to_check, 1):
        plt.subplot(2, 2, i)
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")

    plt.tight_layout()
    plt.show()

    # IQR Stats
    for col in cols_to_check:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        print(
            f"{col}: Found {len(outliers)} outliers (Wait, this includes legitimate high yields?). Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]"
        )


def main():
    if not os.path.exists(DATA_PATH):
        print(f"File not found: {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)

    # Basic cleaning before explore (from previous step awareness)
    df = df.dropna()
    df = df[df["Area"] > 0]  # Avoid div by zero

    analyze_relationships(df)
    check_outliers(df)


if __name__ == "__main__":
    main()
