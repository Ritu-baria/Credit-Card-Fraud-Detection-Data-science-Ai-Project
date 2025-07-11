# Credit-Card-Fraud-Detection-Data-science-Ai-Project# 💳 Credit Card Fraud Detection using AI & Machine Learning

Detect fraudulent transactions in credit card data using machine learning models. This project leverages real-world data, advanced analytics, and classification techniques to prevent fraud and secure financial transactions.

---

## 📌 Project Overview

Credit card fraud is a growing concern in the finance sector. Using AI and data science, we can identify and flag suspicious transactions before they cause harm. This project builds a fraud detection model using classification algorithms on an imbalanced dataset.

---

## 📂 Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions
- **Fraudulent Transactions**: 492 (highly imbalanced)
- **Features**: Numerical (V1 to V28), `Time`, `Amount`, and `Class` (0 = legit, 1 = fraud)

---

## 📊 Workflow

1. **Import Libraries & Load Dataset**
2. **Exploratory Data Analysis (EDA)**
3. **Preprocessing & Feature Scaling**
4. **Handle Class Imbalance with SMOTE**
5. **Train-Test Split**
6. **Model Building (XGBoost, Logistic Regression, etc.)**
7. **Evaluation (Precision, Recall, ROC-AUC)**
8. **Conclusion & Insights**

---

## 📌 Technologies Used

- **Python**
- **Pandas**, **NumPy**
- **Matplotlib**, **Seaborn**
- **Scikit-learn**
- **Imbalanced-learn (SMOTE)**
- **XGBoost**

---

## 🧠 Models Used

| Model               | Description                            |
|---------------------|----------------------------------------|
| Logistic Regression | Simple baseline model                  |
| Random Forest       | Handles nonlinear relationships        |
| XGBoost             | Fast, accurate gradient boosting model |
| Isolation Forest    | Anomaly detection                      |

---

## 🧪 Evaluation Metrics

- **Precision**
- **Recall (Important for fraud!)**
- **F1-Score**
- **ROC-AUC**
- **Confusion Matrix**

---

## 🔍 Key Learnings

- Fraud detection is a classic **imbalanced classification** problem
- **Recall** is more important than accuracy in fraud detection
- **SMOTE** helps synthesize minority class samples
- **XGBoost** works well on structured tabular data
- Business impact is high: even small improvements can save millions

---

## ✅ Sample Results

```text
Classification Report:
              precision    recall  f1-score   support
           0       0.99      0.99      0.99      5682
           1       0.99      0.99      0.99      5691

Accuracy: 99.2%
ROC AUC Score: 0.995
