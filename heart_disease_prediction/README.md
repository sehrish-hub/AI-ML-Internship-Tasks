# ❤️ Task 3: Heart Disease Prediction 

## 🎯 Objective
Build a machine learning model to predict whether a person is at risk of heart disease based on health data.

---

## 📊 Dataset
- Source: [Heart Disease UCI Dataset](https://www.kaggle.com/datasets/ronitf/heart-disease-uci)
- Format: CSV (`heart.csv`)
- Note: The dataset contains numeric and categorical health features of patients.

---

## 🛠 Tools & Libraries
- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

## 🧠 Model
- Logistic Regression (binary classification)
- Features: All health-related features except `id` and original target `num`
- Target: `target` (0 = no heart disease, 1 = heart disease)

---

## 🔍 Steps Performed
1. Loaded dataset (`heart.csv`) into a Pandas DataFrame.
2. Handled missing values:
   - Numeric columns → median
   - Categorical columns → mode
3. Encoded categorical features using LabelEncoder.
4. Created a binary target column (`target`) from the original `num` column.
5. Performed EDA:
   - Target distribution plot
   - Correlation heatmap
6. Split dataset into training and testing sets.
7. Trained Logistic Regression model.
8. Evaluated model using:
   - Accuracy
   - Confusion Matrix
   - ROC-AUC score
9. Visualized ROC curve and feature importance.

---

## 📈 Output
All plots are saved in the `outputs/` folder:
- `target_distribution.png` – shows the distribution of patients with and without heart disease
- `correlation_heatmap.png` – correlation between features
- `roc_curve.png` – ROC curve for model evaluation
- `feature_importance.png` – importance of each feature in prediction

Example evaluation metrics:
Accuracy: 0.7989
Confusion Matrix:
[[63 12]
[25 84]]
ROC-AUC: 0.8733

---
## 👩‍💻 Author

**Sehrish Shafiq**


## 🚀 How to Run
1. Clone the repository or download the project folder.
2. Install dependencies:

```bash
python -m venv .venv
# Activate environment
# Windows
.venv\Scripts\activate
# Linux / WSL
source .venv/bin/activate

pip install -r requirements.txt
Run the script:

python main.py

---
