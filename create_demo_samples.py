import pandas as pd
import json

df = pd.read_csv("dataset/creditcard.csv")

feature_names = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19",
    "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount"
]

legitimate_row = df.iloc[0]
fraudulent_row = df[df["Class"] == 1].iloc[0]

demo_data = {
    "legitimate": {
        feature: float(legitimate_row[feature])
        for feature in feature_names
    },
    "fraud": {
        feature: float(fraudulent_row[feature])
        for feature in feature_names
    }
}

with open("app/demo_samples.json", "w") as file:
    json.dump(demo_data, file, indent=4)

print("Demo samples created successfully.")