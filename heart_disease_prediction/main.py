# 🌸 Task 3: Heart Disease Prediction

## 📌 Objective
# The goal of this task is to predict whether a person is at risk of heart disease using health-related features.
# A Logistic Regression model is applied to classify patients as at-risk or not.


# ----------------------------
# Import Required Libraries
# ----------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, RocCurveDisplay
from sklearn.preprocessing import LabelEncoder

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv("heart.csv")
print("Dataset Info:")
print(df.info())
print("\nMissing values per column:\n", df.isnull().sum())

### 📌 Dataset Overview
# - Source: Heart Disease UCI Dataset (CSV)
# - Total Features: Numeric and categorical health features
# - Target Column: `num` (will be converted to binary `target`)
# - Missing values handled before modeling


# ----------------------------
# Handle Missing Values
# ----------------------------
# Fill numeric columns with median
for col in ["trestbps","chol","thalch","oldpeak","ca"]:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical columns with mode
for col in ["sex","cp","fbs","restecg","exang","slope","thal","dataset"]:
    df[col] = df[col].fillna(df[col].mode()[0])

# Encode categorical columns
label_cols = ["sex","cp","fbs","restecg","exang","slope","thal","dataset"]
le = LabelEncoder()
for col in label_cols:
    df[col] = le.fit_transform(df[col])

### 📌 Notes
# - Missing numeric values replaced by median.
# - Missing categorical values replaced by mode.
# - All categorical columns encoded using LabelEncoder.


# ----------------------------
# Create Binary Target
# ----------------------------
df["target"] = df["num"].apply(lambda x: 1 if x > 0 else 0)

### 📌 Target Creation Insight
# - Original `num` column has multiple values (>0 indicates heart disease)
# - Converted to binary target: 0 = No heart disease, 1 = Heart disease


# ----------------------------
# Exploratory Data Analysis EDA - Plots
# ----------------------------
# Target Distribution
plt.figure(figsize=(6,4))
sns.countplot(x="target", data=df)
plt.title("Target Distribution (Heart Disease)")
plt.savefig("outputs/target_distribution.png")
plt.show()
# plt.close()

### 📌 Target Distribution Insight
# - The dataset is moderately balanced.
# - Slightly more patients without heart disease than with.


# Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("outputs/correlation_heatmap.png")
plt.show()
# plt.close()

### 📌 Correlation Insight
# - Some features like `cp`, `thal`, `exang` show strong correlation with target.
# - Multicollinearity is minimal among numeric features.


# ----------------------------
# Features & Target
# ----------------------------
X = df.drop(["num","target","id"], axis=1)
y = df["target"]

# ----------------------------
# Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# Train Logistic Regression Model
# ----------------------------
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

### 📌 Model Insight
# - Logistic Regression used for binary classification.
# - Max iterations increased to ensure convergence.


# ----------------------------
# Predictions & Evaluation
# ----------------------------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:\n", conf_matrix)
print(f"ROC-AUC: {roc_auc:.4f}")

# ----------------------------
# ROC Curve
# ----------------------------
RocCurveDisplay.from_predictions(y_test, y_proba)
plt.title("ROC Curve")
plt.savefig("outputs/roc_curve.png")
plt.show()
# plt.close()

### 📌 ROC Curve Insight
# - ROC-AUC indicates how well the model distinguishes patients at risk.
# - Higher ROC-AUC (>0.85) shows strong model performance.

# ----------------------------
# Feature Importance
# ----------------------------
feature_importance = pd.Series(model.coef_[0], index=X.columns)
feature_importance.sort_values().plot(kind="barh")
plt.title("Feature Importance (Logistic Regression)")
plt.xlabel("Coefficient Value")
plt.savefig("outputs/feature_importance.png")
plt.show()
# plt.close()

### 📌 Feature Importance Insight
# - Features like `cp`, `thal`, and `exang` are highly influential in predicting heart disease.
# - Numeric features like `trestbps` and `chol` have moderate importance.

print("✅ Task 3 Complete! Plots saved in 'outputs/' folder.")