import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DATA_PATH = "data/raw/crop_yield.csv"

df = pd.read_csv(DATA_PATH)
df.head()

df.tail()

df.shape

df.info()

cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
num_cols = df.select_dtypes(include=["number"]).columns.tolist()

print("Categorical columns:", cat_cols)
print("Numerical columns:", num_cols)

df[num_cols].describe().T

for col in cat_cols:
    print(f"Value counts for {col}:")
    # print(df[col].value_counts())
    # print(df[col].unique())
    print(df[col].nunique())
    print("\n")


for col in num_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()
# shows production and yield are highly correlated, so we drop production

num_cols.remove("Production")

for col in num_cols:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df[col], y=df["Yield"])
    plt.title(f"Yield vs {col}")
    plt.show()
