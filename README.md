# Customer Churn Prediction Using Machine Learning

## Project Overview

Customer churn is a major challenge for subscription-based businesses. Losing existing customers increases business costs and reduces long-term revenue.

This project builds an end-to-end machine learning pipeline to predict whether a customer is likely to churn based on customer behavior and account information.

The project focuses on:
- Customer behavior analysis
- Feature engineering
- Machine learning model comparison
- Churn risk prediction
- Business insights for customer retention


## Problem Statement

The goal is to predict customer churn using historical customer data.

Machine Learning Task:

Binary Classification

Target:

- 0 → Customer stays
- 1 → Customer churns


## Dataset

The dataset contains customer information including:

- Age
- Total Purchase Amount
- Account Manager Assignment
- Customer Relationship Duration
- Number of Sites Used
- Company Details
- Churn Status


## Project Structure

```text
Customer-Churn-Prediction/

├── data/
│   ├── customer_churn.csv
│   ├── new_customers_1.csv
│   └── churn_predictions.csv
│
├── notebooks/
│   └── customer_churn_analysis.ipynb
│
├── src/
│   ├── feature_engineering.py
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict.py
│
├── models/
│   └── churn_model.pkl
│
├── README.md
├── requirements.txt
└── .gitignore
```


## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Machine Learning


## Exploratory Data Analysis

Performed analysis on:

- Customer churn distribution
- Customer age patterns
- Purchase behavior
- Account manager impact
- Customer engagement level
- Feature relationships


## Feature Engineering

Created additional customer behavior features:

### Customer Value

Measures customer spending compared with relationship duration.

### Engagement Level

Segments customers based on product usage.

Categories:

- Low
- Medium
- High

### Age Group

Segments customers into:

- Young
- Middle
- Senior


## Machine Learning Models

The following models were trained and compared:

- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier


## Evaluation Metrics

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score


## Model Pipeline

1. Load customer data
2. Perform feature engineering
3. Clean and preprocess data
4. Train multiple ML models
5. Select best performing model
6. Save trained model
7. Predict churn risk for new customers


## Prediction Output Example

The final model predicts:

- Customer churn status
- Churn probability percentage

Example:

| Customer | Prediction | Churn Risk |
|---|---|---|
| Customer A | Stay | 6.56% |
| Customer B | Churn | 88.53% |


## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train model:

```bash
python src/train_model.py
```

Evaluate model:

```bash
python src/evaluate_model.py
```

Predict new customers:

```bash
python src/predict.py
```


## Results

The machine learning pipeline successfully identifies customers with a high probability of churn.

The predictions can help businesses:

- Identify risky customers early
- Improve customer retention strategies
- Make data-driven customer management decisions


## Future Improvements

- Hyperparameter tuning
- Advanced feature engineering
- SHAP explainability
- Deployment with a web application