# Credit-Risk-Classification-Pipeline
A machine learning pipeline that predicts credit risk (good/bad) for loan applicants using the German Credit Dataset. Built with Python and deployed as an interactive web app using Streamlit.

## Overview

This project benchmarks multiple supervised ML models to identify the best-performing algorithm for credit risk classification. The final model is deployed via a Streamlit interface where users can input applicant details and receive an instant risk prediction.

## Models Evaluated

| Model | Evaluation Metrics | 
|----------|----------|
| Logistic Regression | ROC-AUC, Precision, Recall, F1 |
|Random Forest | ROC-AUC, Precision, Recall, F1 |
|Extra Trees | ROC-AUC, Precision, Recall, F1 |
|XGBoost | ROC-AUC, Precision, Recall, F1 |
|Neural Network (MLP) | ROC-AUC, Precision, Recall, F1 |

**Extra Trees Classifier** was identified as the best-performing model and is used for final predictions.

## Features Used

- Age
- Sex
- Job type (0–3)
- Housing status (own / rent / free)
- Saving accounts (little / moderate / quite rich / rich)
- Checking account (little / moderate / rich)
- Credit amount
- Loan duration (months)

## Project Structure

```
credit-risk-pipeline
|
|-- Credit_Risk_Modeling.ipynb       # Full pipeline: EDA, preprocessing, model training & evaluation
|-- app.py                           # Streamlit web app for predictions
|-- extra_trees_model.pkl            # Saved Extra Trees model
|-- Sex_encoder.pkl                  # Label encoders for categorical features
|-- Housing_encoder.pkl
|-- Saving accounts_encoder.pkl
|-- Checking account_encoder.pkl
|-- README.md

```

## App Preview

Enter applicant details (age, job, housing, account status, loan amount and duration) and click Predict Risk to get an instant good/bad credit risk classification.

## Tech Stack

- Python — Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn
- Deployment — Streamlit
- Model Persistence — Joblib

## Dataset

German Credit Dataset — publicly available on [Kaggle](https://www.kaggle.com/datasets/uciml/german-credit)






