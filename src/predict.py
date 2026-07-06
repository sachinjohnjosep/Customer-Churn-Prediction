"""
Prediction Module

This script:
- Loads new customer data
- Applies feature engineering
- Uses the trained model
- Predicts customer churn risk
"""


import pandas as pd
import pickle


from feature_engineering import create_features

from data_preprocessing import (
    remove_unnecessary_columns,
    encode_features
)



# -----------------------------
# Load New Customers
# -----------------------------

new_customers = pd.read_csv(
    "D:/Customer Churn Prediction/data/new_customers_1.csv"
)



# Keep original copy

output = new_customers.copy()



# -----------------------------
# Feature Engineering
# -----------------------------

new_customers = create_features(
    new_customers
)



# -----------------------------
# Preprocessing
# -----------------------------

new_customers = remove_unnecessary_columns(
    new_customers
)



new_customers = encode_features(
    new_customers
)



# -----------------------------
# Load Model
# -----------------------------

with open(
    "../models/churn_model.pkl",
    "rb"
) as file:


    model_data = pickle.load(
        file
    )



model = model_data[
    "model"
]


scaler = model_data[
    "scaler"
]


columns = model_data[
    "columns"
]



# Match training columns

new_customers = new_customers.reindex(
    columns=columns,
    fill_value=0
)



# Scale Data

new_customers_scaled = scaler.transform(
    new_customers
)



# -----------------------------
# Predict Churn
# -----------------------------

prediction = model.predict(
    new_customers_scaled
)



probability = model.predict_proba(
    new_customers_scaled
)[:,1]



# -----------------------------
# Results
# -----------------------------

output[
    "Churn_Prediction"
] = prediction



output[
    "Churn_Risk_%"
] = (
    probability * 100
).round(2)



print(
    output
)



# Save predictions

output.to_csv(
    "data/churn_predictions.csv",
    index=False
)



print(
    "Predictions saved: data/churn_predictions.csv"
)