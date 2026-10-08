import pandas as pd
import joblib

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# Load untouched test data
X_test = pd.read_csv("output/X_test.csv")
y_test = pd.read_csv("output/y_test.csv").squeeze()

# Load final Random Forest model
model = joblib.load("models/random_forest_model.pkl")

# Load selected threshold
with open("models/threshold.txt", "r") as file:
    threshold = float(file.read())

# Get fraud probabilities
y_prob = model.predict_proba(X_test)[:, 1]

# Apply selected threshold
y_pred = (y_prob >= threshold).astype(int)

print("FINAL MODEL EVALUATION")
print("=======================")

print("\nModel: Random Forest")
print("Threshold:", threshold)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nROC-AUC Score:")
print(roc_auc_score(y_test, y_prob))