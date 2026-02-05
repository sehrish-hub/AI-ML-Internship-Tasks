# 🌸 Task 1: Exploring and Visualizing Iris Dataset

## 📌 Objective
The purpose of this project is to explore, analyze, and visualize the Iris dataset
to understand feature distributions, relationships, and data patterns using Python.

---

## 📊 Dataset
- Iris Dataset (loaded using Seaborn)
- 150 rows and 5 columns

### Features:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width
- Species

---

## 🛠 Tools & Libraries
- Python
- Pandas
- Matplotlib
- Seaborn

---

## 🔍 Steps Performed
1. Loaded dataset using Seaborn
2. Inspected dataset using:
   - `.head()`
   - `.info()`
   - `.describe()`
3. Visualized data using:
   - Scatter plots (feature relationships)
   - Histograms (distribution analysis)
   - Box plots (outlier detection)

---

## 📈 Visual Outputs
All generated plots are saved in the `outputs/` folder:
- `scatter_plot.png`
- `histograms.png`
- `box_plot.png`

---

## ✅ Conclusion
The Iris dataset is clean, well-structured, and ideal for learning data exploration
and visualization techniques. Visual analysis helps identify feature relationships
and data distribution effectively.

---

## 👩‍💻 Author
**Sehrish Shafiq**



uv init --package <project name>
uv run <project name>
Python Environment Setup
python -m venv venv
Activate: venv\Scripts\activate
Linux / Mac: source venv/bin/activate
Install libraries: uv add  pandas matplotlib seaborn/ uv add pandas matplotlib seaborn --frozen

📊 Step 5: Visualization Samjho (Concept)
🔹 Scatter Plot

Feature relationships dikhata hai

Species ke clusters clearly nazar aate hain

🔹 Histogram

Data distribution (spread, skewness)

Feature values ka range samajh aata hai

🔹 Box Plot

Outliers identify karta hai

Median aur spread dikhata hai






















🌸 Task 1 – WSL Setup & Run Guide
1️⃣ Navigate to Project Folder
cd /mnt/c/Users/sehri/OneDrive/Desktop/AI&ML/iris_dataset

2️⃣ Create Linux venv
python3 -m venv .venv-linux


.venv-linux → Linux ke liye dedicated venv

Windows .venv se completely separate

3️⃣ Activate Linux venv
source .venv-linux/bin/activate


Prompt me (venv-linux) show hoga

4️⃣ Install Required Packages
pip install --upgrade pip
pip install pandas matplotlib seaborn


Ye Task 1 ke liye sufficient hai

WSL me bhi same Python environment ready ho jayega

5️⃣ Run Script

Agar script root folder me hai:

python3 main.py


Agar script src folder me hai:

python3 src/iris_analysis.py





WSL/Ubuntu me Python ka venv module missing hai, isliye environment create nahi ho pa raha.

🔹 Solution:

Install the required package for venv:

sudo apt update
sudo apt install python3-venv -y


python3-venv → ye module provide karta hai python3 -m venv functionality

sudo required hai system-level installation ke liye

Verify installation:

python3 -m venv --help


Agar help text show hua → ready ho venv create karne ke liye

Create Linux venv again:

python3 -m venv .venv-linux


.venv-linux → naya Linux-compatible virtual environment

Activate it:

source .venv-linux/bin/activate


Prompt me (venv-linux) show hoga

Install required packages:

pip install --upgrade pip
pip install pandas matplotlib seaborn


Run your Task 1 script:

python3 main.py


Plots outputs/ folder me save honge ✅




🔹 Step 2: Upgrade pip
python3 -m pip install --upgrade pip





































# 🌸 Task 1: Exploring and Visualizing Iris Dataset

## 📌 Objective
The purpose of this project is to explore, analyze, and visualize the Iris dataset
to understand feature distributions, relationships, and data patterns using Python.

---

## 📊 Dataset
- Iris Dataset (loaded using Seaborn)
- 150 rows and 5 columns

### Features:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width
- Species

---

## 🛠 Tools & Libraries
- Python
- Pandas
- Matplotlib
- Seaborn

---

## 🔍 Steps Performed
1. Loaded dataset using Seaborn
2. Inspected dataset using:
   - `.head()`
   - `.info()`
   - `.describe()`
3. Visualized data using:
   - Scatter plots (feature relationships)
   - Histograms (distribution analysis)
   - Box plots (outlier detection)

---

## 📈 Visual Outputs
All generated plots are saved in the `outputs/` folder:
- `scatter_plot.png`
- `histograms.png`
- `box_plot.png`

---

## 💡 WSL / Linux Users Note
- WSL me matplotlib GUI window open nahi hoti (`plt.show()` kaam nahi karta).  
- Sab plots automatically `outputs/` folder me save ho jate hain.  
- Plots dekhne ke liye folder kholen aur images ko kisi image viewer ya browser me open karein.  
- Example: `outputs/scatter_plot.png`, `outputs/histograms.png`, `outputs/box_plot.png`

---

## ✅ Conclusion
The Iris dataset is clean, well-structured, and ideal for learning data exploration
and visualization techniques. Visual analysis helps identify feature relationships
and data distribution effectively.

---

## 👩‍💻 Author
**Sehrish Shafiq**
