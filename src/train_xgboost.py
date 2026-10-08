import pandas as pd
import joblib

from xgboost import XGBClassifier
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

# Calculate class imbalance ratio
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

print("Scale positive weight:", scale_pos_weight)

# Create XGBoost model
model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss"
)

# Train
print("\nTraining XGBoost...")
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
joblib.dump(model, "models/xgboost_model.pkl")

print("\nXGBoost model saved.")