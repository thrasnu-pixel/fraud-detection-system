import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("dataset/creditcard.csv")

# -----------------------------
# 1. Class distribution
# -----------------------------
plt.figure(figsize=(6, 4))
sns.countplot(x="Class", data=df)
plt.title("Transaction Class Distribution")
plt.xlabel("Class (0 = Legitimate, 1 = Fraud)")
plt.ylabel("Number of Transactions")
plt.savefig("output/class_distribution.png")
plt.show()

# -----------------------------
# 2. Transaction amount
# -----------------------------
plt.figure(figsize=(8, 5))
sns.histplot(df["Amount"], bins=50, kde=True)
plt.title("Transaction Amount Distribution")
plt.xlabel("Amount")
plt.ylabel("Frequency")
plt.savefig("output/amount_distribution.png")
plt.show()

# -----------------------------
# 3. Amount by class
# -----------------------------
plt.figure(figsize=(8, 5))
sns.boxplot(x="Class", y="Amount", data=df)
plt.title("Transaction Amount by Class")
plt.xlabel("Class (0 = Legitimate, 1 = Fraud)")
plt.ylabel("Amount")
plt.savefig("output/amount_by_class.png")
plt.show()

# -----------------------------
# 4. Correlation with Class
# -----------------------------
correlation = df.corr()["Class"].sort_values(ascending=False)

print("\nCorrelation with Class:")
print(correlation)

print("\nEDA completed.")