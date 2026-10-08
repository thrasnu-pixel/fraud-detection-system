import pandas as pd
import joblib

from sklearn.metrics import precision_score, recall_score, f1_score

# Load test data
X_test = pd.read_csv("output/X_test.csv")
y_test = pd.read_csv("output/y_test.csv").squeeze()

# Load models
logistic_model = joblib.load("models/logistic_model.pkl")
random_forest_model = joblib.load("models/random_forest_model.pkl")
xgboost_model = joblib.load("models/xgboost_model.pkl")


def evaluate_thresholds(model, model_name):
    probabilities = model.predict_proba(X_test)[:, 1]

    print(f"\n{'=' * 60}")
    print(model_name)
    print(f"{'=' * 60}")

    print(
        f"{'Threshold':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1':<12}"
    )

    for threshold in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:

        predictions = (probabilities >= threshold).astype(int)

        precision = precision_score(
            y_test, predictions, zero_division=0
        )

        recall = recall_score(
            y_test, predictions, zero_division=0
        )

        f1 = f1_score(
            y_test, predictions, zero_division=0
        )

        print(
            f"{threshold:<12.2f}"
            f"{precision:<12.3f}"
            f"{recall:<12.3f}"
            f"{f1:<12.3f}"
        )


# Evaluate all models
evaluate_thresholds(logistic_model, "Logistic Regression")
evaluate_thresholds(random_forest_model, "Random Forest")
evaluate_thresholds(xgboost_model, "XGBoost")