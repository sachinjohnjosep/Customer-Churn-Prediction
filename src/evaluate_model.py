"""
Model Evaluation Module

This script:
- Loads the trained churn prediction model
- Tests model performance
- Displays evaluation metrics
"""


import pandas as pd
import pickle

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


from feature_engineering import create_features

from data_preprocessing import (
    remove_unnecessary_columns,
    encode_features
)



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
# Prepare Data
# -----------------------------

df = remove_unnecessary_columns(
    df
)


df = encode_features(
    df
)


X = df.drop(
    "Churn",
    axis=1
)


y = df[
    "Churn"
]



# -----------------------------
# Load Saved Model
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



# Match columns

X = X.reindex(
    columns=columns,
    fill_value=0
)



X_scaled = scaler.transform(
    X
)



# -----------------------------
# Predictions
# -----------------------------

predictions = model.predict(
    X_scaled
)


probabilities = model.predict_proba(
    X_scaled
)[:,1]



# -----------------------------
# Evaluation Results
# -----------------------------

print(
    "Accuracy:",
    accuracy_score(
        y,
        predictions
    )
)


print(
    "Precision:",
    precision_score(
        y,
        predictions
    )
)


print(
    "Recall:",
    recall_score(
        y,
        predictions
    )
)


print(
    "F1 Score:",
    f1_score(
        y,
        predictions
    )
)


print(
    "ROC-AUC:",
    roc_auc_score(
        y,
        probabilities
    )
)



print(
    "\nClassification Report\n"
)


print(
    classification_report(
        y,
        predictions
    )
)



print(
    "\nConfusion Matrix\n"
)


print(
    confusion_matrix(
        y,
        predictions
    )
)