
# SuperKart Sales Forecasting

## Project Overview

SuperKart is a retail business that wants to forecast product-store sales for the upcoming quarter.

This project develops a machine learning regression solution to predict
`Product_Store_Sales_Total` using product and store characteristics.

## Business Objective

Build a reliable sales forecasting model that can help SuperKart:

- Forecast expected product-store sales.
- Support inventory and store planning.
- Identify important product and store characteristics.
- Provide single and batch prediction through a web application.

## Dataset

The dataset contains 8,763 observations and 12 variables, including:

- Product characteristics
- Store characteristics
- Product-store sales

The target variable is:

`Product_Store_Sales_Total`

## Machine Learning Workflow

1. Data overview and quality checks
2. Exploratory Data Analysis
3. Data preprocessing
4. Feature engineering and selection
5. Outlier analysis
6. Train-test split
7. Baseline model development
8. Hyperparameter tuning
9. Model comparison
10. Final model selection
11. Model serialization
12. Flask backend API
13. Streamlit frontend
14. Docker containerization
15. Deployment

## Models Evaluated

The following regression models were evaluated:

- Random Forest Regressor
- XGBoost Regressor
- Tuned Random Forest Regressor
- Tuned XGBoost Regressor

## Final Model

The **Tuned Random Forest Regressor** was selected as the final model.

Test-set performance:

| Metric | Score |
|---|---:|
| RMSE | 277.54 |
| MAE | 104.61 |
| R² | 0.93 |
| MAPE | 3.75% |

## Deployment Architecture

```text
User
 |
 v
Streamlit Frontend
 |
 | HTTP POST
 v
Flask Backend API
 |
 v
Serialized Random Forest Model
 |
 v
Sales Prediction
