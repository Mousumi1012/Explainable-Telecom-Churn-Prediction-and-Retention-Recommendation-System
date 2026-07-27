# Explainable Telecom Churn Prediction and Retention Recommendation System

A machine learning project that predicts telecom customer churn, explains prediction drivers using SHAP, and suggests retention actions through a Streamlit app.

## Features

- Telecom churn prediction using supervised machine learning
- SHAP-based explainability
- Risk-level classification
- Retention action recommendations
- Streamlit deployment for interactive use

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, XGBoost, SHAP, Streamlit, Joblib

## Dataset

- IBM Telco Customer Churn dataset
- Target variable: `Churn`

## Models Used

- Logistic Regression
- Random Forest
- XGBoost

Final selected model:
- XGBoost

## Project Structure

```bash
app/
data/
models/
notebooks/
src/
requirements.txt
README.md