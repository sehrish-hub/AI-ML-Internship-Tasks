# AI-ML-Internship-Tasks

📌 ## Overview

This repository contains my completed AI/ML internship tasks.  
These projects demonstrate my skills in:

- Data analysis and visualization  
- Machine learning model development  
- Prediction systems using regression and classification  
- Prompt engineering and AI chatbot development using LLMs  

Each task includes:

- Python script (.py)
- Dataset (or dataset link)
- Output visualizations (if applicable)
- README documentation

---

# 🛠 Tools & Technologies Used

- Python 3.x
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- yfinance
- OpenAI Agents SDK
- Gemini LLM (gemini-2.5-flash)
- dotenv, os
- Prompt Engineering

---

# 📂 Folder Structure

AI-ML-Internship-Tasks/
│
├── Task-1_Iris_Dataset/
│ ├── main.py
│ ├── outputs/
│ └── README.md
│
├── Task-2_Stock_Price_Prediction/
│ ├── main.py
│ ├── outputs/
│ └── README.md
│
├── Task-3_Heart_Disease_Prediction/
│ ├── main.py
│ ├── outputs/
│ ├── heart.csv
│ └── README.md
│
├── Task-4_General_Health_Query_Chatbot/
│ ├── chatbot.py
│ └── README.md
│
└── README.md\


---

# 📋 Tasks Details

---

# 🌸 Task 1 – Iris Dataset Exploration & Visualization

## 🎯 Goal

Explore and visualize the Iris dataset to understand feature distributions, relationships, and patterns.

## 🧠 Skills Learned

- Data loading and inspection
- Data visualization
- Exploratory Data Analysis (EDA)
- Feature analysis

## 🛠 Tools Used

- Python
- Pandas
- Seaborn
- Matplotlib

## 🔍 Steps Performed

- Loaded dataset using seaborn
- Inspected dataset structure
- Generated scatter plots
- Created histograms
- Generated box plots
- Saved visualizations

## 📈 Output

- Feature distribution plots
- Outlier detection plots
- Species comparison plots

## 📊 Insights

- Setosa species is clearly separable
- Petal features are strong predictors

---

# 📈 Task 2 – Stock Price Prediction

## 🎯 Goal

Predict next-day stock closing prices using historical stock data.

## 🧠 Skills Learned

- Time series data handling
- Regression modeling
- Feature engineering
- Model evaluation

## 🛠 Tools Used

- Python
- Pandas
- yfinance
- Scikit-learn
- Matplotlib

## 🔍 Steps Performed

- Loaded stock data using yfinance
- Created input features
- Created prediction target
- Trained Linear Regression model
- Evaluated model performance
- Visualized predictions

## 📈 Output

- Actual vs predicted price plots
- Prediction trend visualization

## 📊 Insights

- Model predicts trends effectively
- Minor errors during high volatility

---

# ❤️ Task 3 – Heart Disease Prediction

## 🎯 Goal

Predict whether a patient is at risk of heart disease using health data.

## 🧠 Skills Learned

- Data preprocessing
- Feature engineering
- Classification modeling
- Model evaluation metrics
- Medical dataset analysis

## 🛠 Tools Used

- Python
- Pandas
- Seaborn
- Scikit-learn
- Logistic Regression

## 🔍 Steps Performed

- Loaded heart disease dataset
- Cleaned missing values
- Encoded categorical features
- Split data into training and testing
- Trained Logistic Regression model
- Evaluated model using:

  - Accuracy
  - Confusion Matrix
  - ROC-AUC score

## 📈 Output

- ROC curve
- Feature importance plot
- Correlation heatmap

## 📊 Results

- Accuracy: ~80%
- ROC-AUC: ~87%
- Important features identified

---

# 🤖 Task 4 – General Health Query Chatbot (LLM Based)

## 🎯 Goal

Build an AI chatbot that answers general health-related questions safely using an LLM.

## 🧠 Skills Learned

- Prompt engineering
- LLM integration
- Agent architecture
- Safety filtering
- Conversational AI development

## 🛠 Tools Used

- Python
- OpenAI Agents SDK
- Gemini LLM
- dotenv
- Prompt Engineering

## 🔍 Steps Performed

- Configured Gemini LLM API
- Created medical assistant agent
- Designed prompt instructions
- Implemented safety filter
- Built chatbot interaction loop
- Tested chatbot with health queries

## 💬 Example Queries

- What causes a sore throat?
- What are symptoms of cold?
- Is paracetamol safe?

## 📈 Output

Example interaction:

You: What causes sore throat?

# 🚀 How to Run
Navigate to task folder:

cd Task-4_General_Health_Query_Chatbot


Create virtual environment:

python -m venv .venv


Activate environment:

Windows:

.venv\Scripts\activate


Linux / WSL:

source .venv/bin/activate


Install dependencies:

pip install -r requirements.txt


Run project:

python chatbot.py

---

# 👩‍💻 Author

**Sehrish Shafiq** 
LinkedIn: https://www.linkedin.com/in/sehrish-shafiq

---
# ⭐ Summary

This repository demonstrates my ability to:

- Perform data analysis
- Build machine learning models
- Develop prediction systems
- Implement AI chatbots using LLMs
- Apply prompt engineering and safety handling

These projects reflect practical implementation of AI/ML concepts.
