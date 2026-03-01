# 🏠 House Price Prediction using Machine Learning

## 📌 Project Overview

This project predicts house prices using Machine Learning based on important housing features such as living area, number of bedrooms, bathrooms, and neighborhood. The project follows the complete Machine Learning lifecycle including data preprocessing, exploratory data analysis (EDA), feature engineering, model training, evaluation, visualization, and saving the trained model for future use.

This project is designed to be production-ready and suitable for portfolio, internship submissions, and real-world deployment.

---

## 🎯 Objectives

* Load and explore housing dataset
* Clean and preprocess data
* Perform Exploratory Data Analysis (EDA)
* Encode categorical features
* Scale numerical features
* Train a Machine Learning regression model
* Evaluate model performance
* Visualize results
* Save trained model for reuse
* Predict house prices using saved model

---

## 🧠 Machine Learning Workflow

### 1. Data Loading

* Dataset loaded using pandas
* Dataset structure analyzed using `.info()` and `.head()`

### 2. Data Cleaning

* Missing numerical values filled using median
* Missing categorical values filled using mode

### 3. Feature Engineering

* Categorical feature `Neighborhood` encoded using LabelEncoder

### 4. Exploratory Data Analysis (EDA)

* Price distribution visualization
* Correlation heatmap

### 5. Feature Scaling

* StandardScaler used to normalize feature values

### 6. Train-Test Split

* Dataset split into training and testing sets (80% train, 20% test)

### 7. Model Training

* Linear Regression model used
* Model trained using training dataset

### 8. Model Evaluation

Model performance evaluated using:

* MAE (Mean Absolute Error)
* RMSE (Root Mean Squared Error)
* R² Score (Coefficient of Determination)

### 9. Visualization

Generated plots:

* Price Distribution
* Correlation Heatmap
* Actual vs Predicted Prices
* Feature Importance

### 10. Model Saving

Model saved using joblib including:

* Trained Model
* Scaler
* Label Encoder

Saved file:

```
outputs/house_price_model.pkl
```

---

## 📂 Project Structure

```
House-Price-Prediction/
│
├── house_prices.csv
├── main.py
├── test_model.py
│
├── outputs/
│   ├── house_price_model.pkl
│   ├── price_distribution.png
│   ├── correlation_heatmap.png
│   ├── actual_vs_predicted.png
│   ├── feature_importance.png
│
└── README.md
```

---

## ⚙️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib

---

## 📊 Features Used for Prediction

* GrLivArea (Living Area)
* BedroomAbvGr (Number of Bedrooms)
* FullBath (Number of Bathrooms)
* Neighborhood (Location)

---

## ▶️ How to Run the Project

### Step 1: Install dependencies

```
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

### Step 2: Train the model

```
python main.py
```

### Step 3: Test saved model

```
python test_model.py
```

---

## 💾 Model Saving and Loading

### Save model

```
joblib.dump({
    "model": model,
    "scaler": scaler,
    "label_encoder": le
}, "outputs/house_price_model.pkl")
```

### Load model

```
data = joblib.load("outputs/house_price_model.pkl")
model = data["model"]
scaler = data["scaler"]
label_encoder = data["label_encoder"]
```

---

## 📈 Example Prediction

Input:

* Living Area: 2000
* Bedrooms: 3
* Bathrooms: 2
* Neighborhood: NAmes

Output:

```
Predicted House Price: 185000 (example)
```

---


## 🎓 Learning Outcomes

This project demonstrates:

* End-to-end Machine Learning workflow
* Data preprocessing and feature engineering
* Model training and evaluation
* Model persistence using joblib
* Real-world ML project structure

---

## 👩‍💻 Author

Sehrish Shafiq
AI Engineer | Machine Learning Developer | Generative AI Enthusiast

---

## ⭐ Project Status

✅ Complete
✅ Production-ready
✅ Portfolio-ready
