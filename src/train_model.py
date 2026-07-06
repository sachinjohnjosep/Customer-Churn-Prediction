"""
Model Training Module

This script:
- Loads customer churn data
- Applies feature engineering
- Preprocesses the data
- Trains multiple ML models
- Selects the best model
- Saves the trained model
"""


import pandas as pd
import pickle
import os

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import roc_auc_score


from feature_engineering import create_features

from data_preprocessing import preprocess_training_data



# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv(
    "D:\Customer Churn Prediction\data\customer_churn.csv"
)



# -----------------------------
# Feature Engineering
# -----------------------------

df = create_features(
    df
)



# -----------------------------
# Data Preprocessing
# -----------------------------

(
    X_train,
    X_test,
    y_train,
    y_test,
    scaler,
    columns

) = preprocess_training_data(
    df
)



# -----------------------------
# Define Models
# -----------------------------

models = {

    "Logistic Regression":
        LogisticRegression(),


    "Random Forest":
        RandomForestClassifier(
            random_state=42
        ),


    "Gradient Boosting":
        GradientBoostingClassifier(
            random_state=42
        )

}



# -----------------------------
# Train and Select Best Model
# -----------------------------

best_model = None

best_score = 0

best_model_name = ""



for name, model in models.items():


    model.fit(
        X_train,
        y_train
    )


    probabilities = model.predict_proba(
        X_test
    )[:, 1]


    score = roc_auc_score(
        y_test,
        probabilities
    )


    print(
        f"{name} ROC-AUC: {score:.3f}"
    )



    if score > best_score:


        best_score = score

        best_model = model

        best_model_name = name



print(
    f"Best Model: {best_model_name}"
)



# -----------------------------
# Save Model
# -----------------------------

os.makedirs(
    "../models",
    exist_ok=True
)



model_data = {

    "model": best_model,

    "scaler": scaler,

    "columns": columns

}



with open(
    "../models/churn_model.pkl",
    "wb"
) as file:


    pickle.dump(
        model_data,
        file
    )



print(
    "Model saved successfully: models/churn_model.pkl"
)