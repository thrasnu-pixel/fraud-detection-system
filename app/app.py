from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os
import json


app = Flask(__name__)


# ============================================================
# Deployment-safe project paths
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "random_forest_model.pkl"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "scaler.pkl"
)

THRESHOLD_PATH = os.path.join(
    BASE_DIR,
    "models",
    "threshold.txt"
)

DEMO_PATH = os.path.join(
    BASE_DIR,
    "app",
    "demo_samples.json"
)


# ============================================================
# Load trained model and preprocessing objects
# ============================================================

model = joblib.load(MODEL_PATH)

scaler = joblib.load(SCALER_PATH)

with open(THRESHOLD_PATH, "r") as file:
    threshold = float(file.read())


# ============================================================
# Exact feature order used by the trained model
# ============================================================

feature_names = [
    "Time",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",
    "Amount"
]


# ============================================================
# Home page
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


# ============================================================
# Load legitimate demo transaction
# ============================================================

@app.route("/sample")
def sample():

    try:

        with open(DEMO_PATH, "r") as file:
            demo_data = json.load(file)

        return jsonify(demo_data["legitimate"])

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# Load fraudulent demo transaction
# ============================================================

@app.route("/sample-fraud")
def sample_fraud():

    try:

        with open(DEMO_PATH, "r") as file:
            demo_data = json.load(file)

        return jsonify(demo_data["fraud"])

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# Prediction
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        if data is None:

            return jsonify({
                "error": "No JSON data received."
            }), 400


        # Check that all required features are present

        for feature in feature_names:

            if feature not in data:

                return jsonify({
                    "error": f"Missing feature: {feature}"
                }), 400


        # Keep exact feature order used during training

        values = [
            float(data[feature])
            for feature in feature_names
        ]


        # Create DataFrame

        X = pd.DataFrame(
            [values],
            columns=feature_names
        )


        # ====================================================
        # IMPORTANT:
        # The scaler was trained ONLY on Time and Amount.
        # V1-V28 are already PCA-transformed features.
        # ====================================================

        X[["Time", "Amount"]] = scaler.transform(
            X[["Time", "Amount"]]
        )


        # ====================================================
        # Random Forest prediction
        # ====================================================

        probability = model.predict_proba(X)[0][1]


        # Apply optimized threshold

        prediction = int(
            probability >= threshold
        )


        if prediction == 1:

            result = "⚠️ Fraudulent Transaction"

        else:

            result = "✅ Legitimate Transaction"


        return jsonify({

            "prediction": prediction,

            "result": result,

            "fraud_probability": round(
                float(probability),
                4
            ),

            "threshold": threshold

        })


    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 400


# ============================================================
# Start application
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )