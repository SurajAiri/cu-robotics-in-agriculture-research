import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set plot style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# cd ../.. # Navigate to project root if needed
DATA_PATH = "data/raw/crop_yield.csv"
df = pd.read_csv(DATA_PATH)

# --- 1. Basic Data Inspection ---
print("--- First 5 rows ---")
print(df.head())

print("\n--- Data Info ---")
print(df.info())

print("\n--- Summary Statistics ---")
print(df.describe())

print("\n--- Missing Values ---")
print(df.isnull().sum())

print(f"\n--- Duplicate Rows: {df.duplicated().sum()} ---")

# --- 2. Target Variable Analysis (Yield) ---
plt.figure(figsize=(10, 6))
sns.histplot(df["Yield"], bins=50, kde=True)
plt.title("Distribution of Crop Yield")
plt.xlabel("Yield (Production per Unit Area)")
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x=df["Yield"])
plt.title("Boxplot of Crop Yield")
plt.show()

# --- 3. Categorical Feature Analysis ---
# Top 10 Crops by frequency
plt.figure(figsize=(12, 6))
df["Crop"].value_counts().head(10).plot(kind="bar")
plt.title("Top 10 Crops by Frequency")
plt.xlabel("Crop")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# Distribution of Seasons
plt.figure(figsize=(8, 5))
sns.countplot(x="Season", data=df)
plt.title("Count of Records per Season")
plt.show()

# Top 10 States by frequency
plt.figure(figsize=(12, 6))
df["State"].value_counts().head(10).plot(kind="bar")
plt.title("Top 10 States by Record Count")
plt.xticks(rotation=45)
plt.show()

# --- 4. Numerical Feature Analysis ---
numerical_cols = ["Area", "Production", "Annual_Rainfall", "Fertilizer", "Pesticide"]
df[numerical_cols].hist(bins=30, figsize=(15, 10), layout=(2, 3))
plt.suptitle("Histograms of Numerical Features")
plt.show()

# --- 5. Bivariate Analysis (Correlations) ---
# Correlation Matrix
plt.figure(figsize=(10, 8))
# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

# Scatter plots: Key Inputs vs Yield
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.scatterplot(x="Annual_Rainfall", y="Yield", data=df, ax=axes[0], alpha=0.5)
axes[0].set_title("Annual Rainfall vs Yield")

sns.scatterplot(x="Fertilizer", y="Yield", data=df, ax=axes[1], alpha=0.5)
axes[1].set_title("Fertilizer vs Yield")

sns.scatterplot(x="Pesticide", y="Yield", data=df, ax=axes[2], alpha=0.5)
axes[2].set_title("Pesticide vs Yield")

plt.tight_layout()
plt.show()

# --- 6. Temporal Analysis (Yield over Years) ---
plt.figure(figsize=(12, 6))
sns.lineplot(x="Crop_Year", y="Yield", data=df)  # aggregations default to mean with ci
plt.title("Average Crop Yield Over Years")
plt.xlabel("Year")
plt.ylabel("Average Yield")
plt.show()

# --- 7. Yield by State (Top 10) ---
# Calculate average yield per state and sort
avg_yield_by_state = (
    df.groupby("State")["Yield"].mean().sort_values(ascending=False).head(15)
)

plt.figure(figsize=(12, 6))
sns.barplot(x=avg_yield_by_state.index, y=avg_yield_by_state.values)
plt.title("Top 15 States by Average Yield")
plt.xticks(rotation=90)
plt.ylabel("Average Yield")
plt.show()
