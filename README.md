# Credit Card Fraud Detection

## Problem Statement
Financial fraud costs institutions billions annually. 
This project builds an ML pipeline to detect fraudulent 
credit card transactions with high precision.

## Dataset
- 284,807 transactions, 492 fraud cases (0.17% fraud rate)
- Source: ULB Machine Learning Group via Kaggle
- Features: 28 PCA-transformed variables + Amount + Time

## Key Challenges
- Severe class imbalance (99.8% legitimate vs 0.2% fraud)
- Solved using SMOTE oversampling

## Models Compared
| Model | AUC-ROC | F1 (Fraud) |
|-------|---------|------------|
| Logistic Regression | ~0.97 | ~0.76 |
| Random Forest | ~0.99 | ~0.87 |
| XGBoost | ~0.99 | ~0.89 |

## Key Findings
- V14, V17, V12 are strongest fraud predictors (SHAP)
- XGBoost outperforms baseline by 13% on F1 score
- SMOTE improved recall from 0.61 to 0.89

## Tech Stack
Python, Scikit-learn, XGBoost, SHAP, SMOTE, Streamlit

## How to Run
pip install -r requirements.txt
streamlit run app/app.py