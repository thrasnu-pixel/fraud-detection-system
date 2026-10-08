import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

# Load the original training data
X = pd.read_csv("output/X_train.csv")
y = pd.read_csv("output/y_train.csv").squeeze()

# Split training data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))

# Train a NEW Random Forest only on the training portion
print("\nTraining validation Random Forest...")

model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Get probabilities on validation data
val_probabilities = model.predict_proba(X_val)[:, 1]

# Test different thresholds
results = []

for threshold in [
    0.30, 0.35, 0.40, 0.45,
    0.50, 0.55, 0.60, 0.65, 0.70
]:

    predictions = (val_probabilities >= threshold).astype(int)

    precision = precision_score(
        y_val,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        predictions,
        zero_division=0
    )

    results.append(
        (threshold, precision, recall, f1)
    )

# Display results
print("\nValidation Results:")
print(
    f"{'Threshold':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
)

for threshold, precision, recall, f1 in results:
    print(
        f"{threshold:<12.2f}"
        f"{precision:<12.3f}"
        f"{recall:<12.3f}"
        f"{f1:<12.3f}"
    )

# Select threshold with highest F1
best_result = max(
    results,
    key=lambda x: x[3]
)

best_threshold = best_result[0]

print("\nBest threshold:")
print(best_threshold)

# Save the threshold
with open("models/threshold.txt", "w") as file:
    file.write(str(best_threshold))

print("\nThreshold saved.")