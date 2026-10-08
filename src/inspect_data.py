import pandas as pd

# Load dataset
df = pd.read_csv("dataset/creditcard.csv")

# Basic information
print("Dataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nClass Distribution:")
print(df["Class"].value_counts())

print("\nClass Distribution (%):")
print(df["Class"].value_counts(normalize=True) * 100)