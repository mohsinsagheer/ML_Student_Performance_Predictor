# 🎓 Student Performance Prediction System

An end-to-end Machine Learning project featuring an **interactive Streamlit web dashboard** and a **step-by-step Jupyter Notebook** for predicting student academic outcomes (**Pass** vs. **Fail**). 

The system implements and compares **Decision Tree**, **Gaussian Naïve Bayes**, and **K-Nearest Neighbors (KNN)** classifiers, along with a **Majority-Vote Ensemble**.

---

## 🌟 Key Features

1. **Interactive Multi-Page Streamlit Dashboard (`app.py`)**:
   - **Dataset & Memory Explorer**: Inspect dataset dimensions, statistics, 0 missing value verification, deep memory profiling (`df.memory_usage(deep=True)`), and automatic downcasting optimization.
   - **Visualizations**: Grade distributions ($G1, G2, G3$) with pass/fail threshold lines, weekly study time vs. pass rate, and alcohol/absence correlations using Plotly.
   - **Model Evaluation & Comparison**: Side-by-side metric tables (Accuracy, Precision, Recall, F1-Score), annotated interactive Confusion Matrices, and Decision Tree Feature Importance.
   - **Real-Time Live Predictor**: Form with categorized inputs (Academic, Demographics, Lifestyle), preset student personas (High Achiever, At-Risk, Average Hardworking, High Absences), model probability bars, and actionable academic recommendations.
   - **Batch CSV Predictions**: Upload custom student CSV files or test holdout sets with 1-click downloadable results.
   - **Educational Guide**: Beginner-friendly explanations of Decision Tree, Naïve Bayes, and KNN algorithms.

2. **Jupyter Notebook (`notebooks/student_performance_pipeline.ipynb`)**:
   - A complete 10-phase laboratory notebook with pre-executed outputs, charts, and markdown explanations.

3. **Software Engineering & ML Best Practices**:
   - Modular code architecture (`src/data_loader.py`, `src/preprocessor.py`, `src/models.py`, `src/utils.py`).
   - Stratified train/test splitting to maintain Pass/Fail balance.
   - One-Hot Encoding and Standard Scaling using Scikit-Learn `ColumnTransformer`.
   - Defensive memory management and parameter type hinting.

---

## 📂 Project Structure

```text
Lab-Mid-Project/
├── app.py                             # Interactive Streamlit Web Application
├── requirements.txt                   # Pinned project dependencies
├── generate_notebook.py               # Automated notebook generator script
├── README.md                          # Project documentation and user guide
├── .venv/                             # Dedicated Python Virtual Environment
├── student-data/                      # UCI Student Performance Dataset
│   ├── student-mat.csv                # Mathematics course records (395 students)
│   ├── student-por.csv                # Portuguese course records (649 students)
│   ├── student-merge.R                # R script for student merging
│   └── student.txt                    # Dataset description & attribute metadata
├── notebooks/
│   └── student_performance_pipeline.ipynb  # Comprehensive laboratory notebook
└── src/                               # Modular Python Package
    ├── __init__.py                    # Source package initializer
    ├── data_loader.py                 # Data loading, memory profiling & cleaning
    ├── preprocessor.py                # Encoding, scaling, and train/test splitting
    ├── models.py                      # Decision Tree, Naïve Bayes, KNN & Majority Vote
    └── utils.py                       # Plotly visualizations & student personas
```

---

## 🚀 Step-by-Step Installation & Setup

### 1. Prerequisites
- Python 3.10+ (Tested on Python 3.12)
- Linux / macOS / Windows terminal

### 2. Activate the Virtual Environment
A dedicated `.venv` virtual environment is configured in this repository. Activate it using:

```bash
# On Linux / macOS:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

*(Optional: If recreating the environment from scratch)*:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 🖥️ Running the Streamlit Dashboard

To launch the web dashboard, run:

```bash
streamlit run app.py
```

Once started, open your web browser at:
👉 **`http://localhost:8501`**

---

## 📓 Running the Jupyter Notebook

To explore the step-by-step pipeline in Jupyter:

```bash
jupyter notebook notebooks/student_performance_pipeline.ipynb
```
or open the notebook directly in your IDE (VS Code / Antigravity IDE).

---

## 🧠 Model Architectures & Methodologies

| Model | Underlying Concept | Strengths | Ideal Scenario |
| :--- | :--- | :--- | :--- |
| **Decision Tree** | Recursive binary partitioning minimizing Gini Impurity | Highly interpretable, identifies feature importance | Explaining factors (e.g. past failures) to teachers |
| **Naïve Bayes** | Bayes' Theorem with conditional independence assumption | Fast training, gives probabilistic confidence | Rapid baseline classification and posterior tracking |
| **KNN** | Instance-based majority vote from $k$ nearest neighbors | Non-parametric, adapts easily to new records | Identifying students similar to past cohorts |
| **Majority Vote** | Democratic ensemble consensus across all 3 models | Lowers variance, reduces individual model bias | Final high-stakes pass/fail assessment |

---

## 📊 Performance Benchmarks (Math Cohort)

| Classifier | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Decision Tree** | ~86.1% | ~97.7% | ~81.1% | ~88.7% |
| **Naïve Bayes** | ~81.0% | ~88.0% | ~83.0% | ~85.4% |
| **K-Nearest Neighbors** | ~76.0% | ~79.3% | ~86.8% | ~82.9% |
| **Majority Vote Ensemble** | **~86.1%** | **~92.0%** | **~86.8%** | **~89.3%** |

*Note: Results obtained with standard 80/20 stratified split.*
