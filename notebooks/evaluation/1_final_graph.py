import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
RESULTS_PATH = "results/top_model_result.csv"
OUTPUT_DIR = "reports/figures"
COMBINED_FILE = os.path.join(OUTPUT_DIR, "final_model_performance_comparison.png")
R2_FILE = os.path.join(OUTPUT_DIR, "final_model_r2_score.png")
ERROR_FILE = os.path.join(OUTPUT_DIR, "final_model_error_metrics.png")


def plot_r2(df, ax=None):
    """Refactored plotting logic for R2"""
    # If no axis provided, create a new figure (for single plot mode)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        standalone = True
    else:
        standalone = False

    sns.barplot(
        data=df, x="Model", y="R2", ax=ax, palette="viridis", hue="Model", legend=False
    )

    ax.set_title(
        "Model Accuracy (R² Score)\nHigher is Better",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_ylabel("R² Score", fontsize=12)
    ax.set_xlabel("Model Name", fontsize=12)
    ax.set_ylim(0, 1.05)

    # Annotations
    for i, v in enumerate(df["R2"]):
        ax.text(
            i,
            v + 0.015,
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    return ax.figure if standalone else None


def plot_errors(df, ax=None):
    """Refactored plotting logic for Errors"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        standalone = True
    else:
        standalone = False

    df_melted = df.melt(
        id_vars="Model",
        value_vars=["RMSE", "MAE"],
        var_name="Metric",
        value_name="Error Value",
    )

    sns.barplot(
        data=df_melted, x="Model", y="Error Value", hue="Metric", ax=ax, palette="magma"
    )

    ax.set_title(
        "Error Metrics (RMSE & MAE)\nLower is Better",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_ylabel("Error Value", fontsize=12)
    ax.set_xlabel("Model Name", fontsize=12)
    ax.legend(title="Metric")

    # Annotations
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=10)

    return ax.figure if standalone else None


def main():
    if not os.path.exists(RESULTS_PATH):
        raise FileNotFoundError(f"{RESULTS_PATH} not found.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load and Sort Data
    df = pd.read_csv(RESULTS_PATH).sort_values(by="R2", ascending=False)

    # Global Style
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    # --- 1. Generate Combined Figure (Best for Paper) ---
    print(f"Generating Combined Figure...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    plot_r2(df, ax=axes[0])
    plot_errors(df, ax=axes[1])
    plt.tight_layout()
    plt.savefig(COMBINED_FILE, dpi=300, bbox_inches="tight")
    plt.close()  # Free memory

    # --- 2. Generate Separate R2 Figure ---
    print(f"Generating R2 Figure...")
    fig_r2 = plot_r2(df)
    plt.tight_layout()
    fig_r2.savefig(R2_FILE, dpi=300, bbox_inches="tight")
    plt.close()

    # --- 3. Generate Separate Error Figure ---
    print(f"Generating Error Figure...")
    fig_error = plot_errors(df)
    plt.tight_layout()
    fig_error.savefig(ERROR_FILE, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nDone! saved to:\n1. {COMBINED_FILE}\n2. {R2_FILE}\n3. {ERROR_FILE}")


if __name__ == "__main__":
    main()
