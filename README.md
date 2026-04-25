# 🚀 Collision Risk Classification in LEO using Machine Learning

## 📌 Overview

This project focuses on the classification of collision risk between satellites in Low Earth Orbit (LEO) using machine learning techniques.

The objective is to develop and compare different supervised learning models capable of distinguishing between low-risk and potentially dangerous conjunction events, based on real Conjunction Data Messages (CDMs) provided by the European Space Agency (ESA).

This work has been developed as part of a Master's Thesis in Artificial Intelligence.

---

## 🛰️ Dataset

The dataset consists of Conjunction Data Messages (CDMs), which contain information about close approaches between space objects.

* Source: European Space Agency (ESA)
* Platform: Zenodo
* Time period: 2015–2019
* Original size: ~199,000 records
* Features: 103 variables

### 🔧 Preprocessing strategy

Each conjunction event contains multiple CDMs over time.
To avoid temporal dependency and data leakage:

✔ Only the **last CDM per event** (minimum `time_to_tca`) is used
✔ Final dataset size: **15,321 events**

---

## 🎯 Problem Definition

The task is formulated as a **binary classification problem**:

* **Class 0 (Low risk)** → `risk == -30`
* **Class 1 (Potential risk)** → `risk > -30`

---

## ⚙️ Data Processing Pipeline

The data processing includes:

1. Selection of last CDM per event
2. Creation of binary target variable (`risk_binary`)
3. Handling missing values:

   * Removal of high-null feature (`c_rcs_estimate`)
   * Median imputation for numerical features
4. Categorical encoding:

   * Grouping rare categories
   * One-hot encoding
5. Feature redundancy reduction:

   * Correlation threshold: 0.95
6. Feature scaling:

   * Standardization using `StandardScaler`
7. Data leakage prevention:

   * Removal of `max_risk_scaling` and `max_risk_estimate`
8. Train-test split:

   * 80% training / 20% testing (stratified)

---

## 🤖 Models Implemented

The following machine learning models were evaluated:

* Decision Tree
* Random Forest
* Support Vector Machine (SVM)
* Naïve Bayes
* Gradient Boosting

Final optimized models:

* Random Forest
* Gradient Boosting
* SVM

---

## 📊 Evaluation Metrics

Models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score

Additionally:

* Confusion matrices
* Precision-Recall curves

---

## ⚠️ Important Note: Data Leakage

During initial experiments, unrealistically high scores (>0.99) were obtained.

This was due to **data leakage**, caused by features directly derived from the target:

* `max_risk_scaling`
* `max_risk_estimate`

These features were removed to ensure a fair and realistic evaluation.

---

## 📂 Project Structure

project/
│
├── code/
│   └── models_final.py
│
└── README.md

---

## ▶️ How to Run

### 1. Install dependencies

pip install pandas numpy scikit-learn matplotlib seaborn

### 2. Place dataset

Ensure the dataset file is available:

cleanup_cdm_history_2015-2019.csv

### 3. Run the script

python code/models_final.py

---

## 📊 Output

The script will generate:

* Dataset statistics
* Class distribution
* Model performance metrics
* Bar chart comparing models
* Confusion matrices for each model

---

## 🧠 Key Insights

* Ensemble methods outperform simpler models
* Gradient Boosting provides the best balance between precision and recall
* Proper handling of data leakage is critical in ML workflows
* Orbital uncertainty and relative dynamics are key predictors

---

## 📚 References

* European Space Agency (ESA)
* Collision Avoidance Challenge Dataset (Zenodo)

---

## 👨‍🎓 Author

**Rafael Martín Priego**

---

## 📄 License

This project is for academic and research purposes.
