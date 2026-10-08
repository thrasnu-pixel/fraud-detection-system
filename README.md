# 💳 Fraud Detection System

A machine learning-based credit card fraud detection system built using Python, Scikit-learn, Random Forest, XGBoost, and Flask.

The project takes transaction features as input, estimates the probability of fraud, and classifies the transaction as either legitimate or fraudulent.

---

## 🚀 Project Overview

Credit card fraud detection is a challenging machine learning problem because fraudulent transactions are extremely rare compared with legitimate transactions.

This project builds an end-to-end fraud detection pipeline that includes:

- Exploratory Data Analysis
- Data preprocessing
- Handling of highly imbalanced data
- Multiple machine learning models
- Model comparison
- Probability threshold optimization
- Final model evaluation
- Flask-based web application
- Demo transaction samples
- Deployment-ready configuration

---

## ✨ Features

- 🔍 Fraudulent transaction detection
- 📊 Exploratory Data Analysis
- ⚖️ Imbalanced classification handling
- 🤖 Multiple ML models
- 🎯 Optimized fraud detection threshold
- 📈 Fraud probability prediction
- 🌐 Interactive Flask web application
- ✅ Legitimate transaction demo
- ⚠️ Fraudulent transaction demo
- ☁️ Deployment-ready architecture

---

## 📂 Dataset

This project uses the **Credit Card Fraud Detection Dataset**.

Dataset characteristics:

| Property | Value |
|---|---:|
| Total transactions | 284,807 |
| Legitimate transactions | 284,315 |
| Fraudulent transactions | 492 |
| Fraud rate | ~0.17% |
| Input features | 30 |
| Target variable | Class |

The dataset contains:

- `Time`
- `V1`–`V28`
- `Amount`
- `Class`

Where:

```text
Class = 0 → Legitimate
Class = 1 → Fraud