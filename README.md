# 💳 Credit Card Fraud Detection

An end-to-end machine learning pipeline to detect fraudulent credit card transactions using **XGBoost**, **SMOTE** for class imbalance, and **SHAP** for model explainability — deployed as an interactive **Streamlit** web application.

---

## 🔍 Problem Statement

Financial fraud costs institutions **billions of dollars annually**. This project builds a production-ready ML pipeline that predicts fraudulent transactions in real time, enabling financial institutions to flag suspicious activity before it causes damage.

---

## 📊 Dataset

- **Source:** ULB Machine Learning Group via [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions with only 492 fraud cases
- **Class imbalance:** 99.83% legitimate vs 0.17% fraud
- **Features:** 28 PCA-transformed variables (V1–V28) + Amount + Time

---

## ⚡ Key Challenge — Class Imbalance

With only 0.17% fraud cases, a naive model would predict "legitimate" for everything and still achieve 99.8% accuracy — completely useless in practice.

**Solution:** Applied **SMOTE (Synthetic Minority Oversampling Technique)** to balance the training data, improving fraud recall from 0.61 → 0.89.

---

## 🤖 Models Compared

| Model | AUC-ROC | F1 (Fraud) | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression | 0.97 | 0.76 | 0.88 | 0.67 |
| Random Forest | 0.99 | 0.87 | 0.91 | 0.84 |
| **XGBoost** ✅ | **0.99** | **0.89** | **0.92** | **0.89** |

**XGBoost selected** as the final model based on highest F1 score and best balance of precision vs recall — critical in fraud detection where both false positives (blocking legitimate transactions) and false negatives (missing fraud) have real business costs.

---

## 🔎 SHAP Explainability

Model explainability is essential in fintech — regulators and business stakeholders need to understand *why* a transaction was flagged.

**Top fraud predictors identified via SHAP:**
- **V14** — strongest negative indicator of fraud
- **V17** — high importance in fraud classification  
- **V12** — significant contributor to fraud probability
- **V10** — notable fraud signal

![SHAP Summary](../Downloads/shap_summary.png)

---

## 📈 ROC Curve

![ROC Curve](../Downloads/roc_curve.png)

---

## 🚀 Streamlit App

Interactive web app allowing users to input transaction features and get real-time fraud probability with prediction confidence.

**To run locally:**
```bash
git clone https://github.com/saanvireddy/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt
python -m streamlit run app/app.py
```

---

## 🗂️ Project Structure

```
credit-card-fraud-detection/
│
├── app/
│   └── app.py                  # Streamlit web application
├── data/
│   └── creditcard.csv          # Dataset (download from Kaggle)
├── models/
│   ├── xgb_fraud_model.pkl     # Trained XGBoost model
│   └── scaler.pkl              # Fitted StandardScaler
├── notebooks/
│   └── fraud_detection.ipynb   # Full EDA, modeling, SHAP analysis
├── roc_curve.png               # ROC curve comparison
├── shap_summary.png            # SHAP feature importance plot
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| ML Models | Scikit-learn, XGBoost |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| Visualization | Matplotlib, Seaborn |
| Deployment | Streamlit |
| Model Persistence | Joblib |

---

## 💡 Key Findings

- SMOTE oversampling improved fraud recall by **46%** (0.61 → 0.89)
- XGBoost outperformed Logistic Regression baseline by **13%** on F1 score
- V14 and V17 are the most critical features — consistent with known PCA patterns in card transaction fraud literature
- Precision-Recall tradeoff analysis shows XGBoost maintains high precision (0.92) while maximizing recall — minimizing both false alarms and missed fraud

---

## 👩‍💻 Author

**Saanvi Reddy Baradi**  
M.S. Artificial Intelligence & Business Analytics — University of South Florida  
[LinkedIn](https://linkedin.com/in/saanvireddy) | [GitHub](https://github.com/saanvireddy)
