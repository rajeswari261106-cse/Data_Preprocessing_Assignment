import pandas as pd
import numpy as np

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder
)

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv("dataset.csv")

print("Dataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nFirst 5 Rows:")
print(df.head())

# ----------------------------
# Missing Values
# ----------------------------
print("\nMissing Values:")
print(df.isnull().sum())

# Fill numeric missing values
for col in df.select_dtypes(include=np.number):
    df[col] = df[col].fillna(df[col].median())

# Fill categorical missing values
for col in df.select_dtypes(include="object"):
    df[col] = df[col].fillna(df[col].mode()[0])

# ----------------------------
# Remove Duplicates
# ----------------------------
duplicates = df.duplicated().sum()
print("\nDuplicate Rows:", duplicates)

df = df.drop_duplicates()

# ----------------------------
# Outlier Handling (IQR Method)
# ----------------------------
numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df = df[(df[col] >= lower) & (df[col] <= upper)]

# ----------------------------
# Correct Data Types
# ----------------------------
print("\nData Types:")
print(df.dtypes)

# ----------------------------
# Categorical Encoding
# ----------------------------
categorical_cols = df.select_dtypes(include="object").columns

df = pd.get_dummies(
    df,
    columns=categorical_cols,
    drop_first=True
)

# ----------------------------
# Feature Scaling
# ----------------------------
scaler = StandardScaler()

num_cols = df.select_dtypes(include=np.number).columns

df[num_cols] = scaler.fit_transform(df[num_cols])

# ----------------------------
# Skewness Handling
# ----------------------------
for col in num_cols:
    if abs(df[col].skew()) > 1:
        df[col] = np.log1p(df[col] - df[col].min() + 1)

# ----------------------------
# Save Cleaned Dataset
# ----------------------------
df.to_csv("cleaned_dataset.csv", index=False)

print("\nPreprocessing Completed Successfully!")
print("Cleaned dataset saved as cleaned_dataset.csv")