import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# Load processed data
X_train = pd.read_csv("output/X_train.csv")
X_test = pd.read_csv("output/X_test.csv")
y_train = pd.read_csv("output/y_train.csv").squeeze()
y_test = pd.read_csv("output/y_test.csv").squeeze()

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# Train
print("Training Random Forest...")
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nROC-AUC Score:")
print(roc_auc_score(y_test, y_prob))

# Save model
joblib.dump(model, "models/random_forest_model.pkl")

print("\nRandom Forest model saved.")