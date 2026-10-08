import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
df = pd.read_csv("dataset/creditcard.csv")

# Separate features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Split data BEFORE any preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale Time and Amount
scaler = StandardScaler()

X_train[["Time", "Amount"]] = scaler.fit_transform(
    X_train[["Time", "Amount"]]
)

X_test[["Time", "Amount"]] = scaler.transform(
    X_test[["Time", "Amount"]]
)

# Save scaler for later use in the web application
joblib.dump(scaler, "models/scaler.pkl")

# Save processed datasets
X_train.to_csv("output/X_train.csv", index=False)
X_test.to_csv("output/X_test.csv", index=False)
y_train.to_csv("output/y_train.csv", index=False)
y_test.to_csv("output/y_test.csv", index=False)

# Display information
print("Preprocessing completed.")
print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())