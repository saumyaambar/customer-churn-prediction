# Customer Churn Prediction & Business Proposal

Predicting customer churn for a consumer finance/telecom company using machine learning, and translating the findings into a data-driven retention strategy.

## Problem Statement

The company is experiencing customer churn well above the industry average. The goal is to identify **which customers are likely to leave** and **why**, so the business can act proactively instead of reactively.

## Business Objective

- Predict customer churn using machine learning
- Identify the key factors driving churn
- Propose a data-driven business solution to reduce churn rate

## Dataset

- `Client.csv` — customer demographic and account information
- `Record.csv` — customer usage and behavior records
- Merged on `Customer_ID` → ~100,000 rows × 101 columns
- **Target variable:** `churn` (1 = churned, 0 = retained)

> Note: raw data files are not included in this repo. Add your own `Client.csv` and `Record.csv` to the project root to reproduce the notebook.

## Approach

1. Exploratory Data Analysis (EDA)
2. Data cleaning & missing value treatment
3. Feature engineering
4. Encoding & preprocessing
5. Model building — Logistic Regression & Random Forest
6. Model evaluation & feature importance
7. Business insight extraction & proposal

## Methodology

- **Missing values:** columns with >50% missing data dropped; remaining values median-imputed
- **Feature engineering:** `revenue_per_call`, `revenue_per_minute`, and `customer_age_group` (tenure-based segments: New / Regular / Loyal / Very Loyal)
- **Encoding:** one-hot encoding for categorical variables
- **Split:** 80/20 stratified train-test split

## Models & Results

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | ~60% | ~0.634 |
| Random Forest (200 trees) | ~62% | ~0.672 |

**Random Forest** was selected as the final model for its higher AUC and ability to capture non-linear relationships.

## Key Findings

1. **Equipment age (`eqpdays`)** is the strongest churn predictor — customers with older devices churn more, likely due for an upgrade elsewhere.
2. **Declining usage (`change_mou`)** is the second-strongest signal — falling usage predicts drift toward churn.
3. **Churners disengage silently** — they make *fewer* customer care calls, not more, making them hard to catch without a model.
4. **Dropped calls (`drop_vce_Mean`)** indicate poor network quality directly drives churn.
5. Engineered features (`revenue_per_call`, `revenue_per_minute`) confirm: low engagement + low spend = high churn risk.

## Business Proposal: AI-Powered Churn Prevention System

- **Early Warning Dashboard** — flag customers with >70% predicted churn probability monthly
- **Targeted Retention Campaigns** — loyalty offers for new customers at month 3 and 6; proactive outreach for frequent care-callers
- **Network Quality Investment** — use dropped-call data to prioritize infrastructure upgrades in high-churn regions

## Tech Stack

- Python, pandas, NumPy
- scikit-learn (Logistic Regression, Random Forest, preprocessing, metrics)
- matplotlib, seaborn

## Repository Structure

```
.
├── customer-churn-prediction.ipynb   # Main analysis notebook
├── README.md
└── data/                             # (not included) Client.csv, Record.csv
```

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook customer-churn-prediction.ipynb
```

## Future Improvements

- Hyperparameter tuning (GridSearch/RandomizedSearch) to push AUC higher
- Try gradient boosting models (XGBoost/LightGBM)
- Address class imbalance (SMOTE or class weighting)
- Deploy the model behind a simple API/dashboard for the early-warning system
