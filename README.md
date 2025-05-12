# Shaikmay
# Power BI End-to-End Churn Analysis | SQL + Machine Learning

## Project Overview
This project focuses on churn analysis for a telecom company using a full BI + ML pipeline. It combines SQL data processing, Power BI dashboarding, and Python machine learning to predict customer churn and identify business improvement opportunities.

## Tools & Technologies
- SQL (Data Cleaning & Transformation)
- Power BI (Dashboard Design)
- Python (Pandas, Scikit-learn, XGBoost)
- Jupyter Notebook

## Key Features
- Interactive Power BI dashboards to monitor churn metrics, customer segments, and service usage.
- Predictive modeling using Random Forest and XGBoost (achieved ~82% accuracy).
- Churn classification based on contract type, monthly charges, tenure, and service subscriptions.
- ML outputs exported to Power BI for risk-based customer segmentation.

## Business Impact
Key Insights from the Churn Analysis
Churn Rate: Approximately 26.5% of customers were found to have churned. That’s a significant proportion and highlights a critical area to focus on for customer retention.

## Top Churn Reasons

- High monthly charges for short tenure customers.
- Lack of tech support and online security features.
- Customers using month-to-month contracts were more likely to churn.

## Demographics: Senior citizens had a higher churn rate.

Services Impact: Those without add-on services (like streaming TV or device protection) had higher churn rates.

Tenure Analysis: Customers in their early months (0–12 months) were more prone to churn.



## Recommended solutions: 
- promote annual contracts, add service bundles, and early loyalty interventions.



## Process and Steps

## Framework
The project followed an end-to-end BI + ML pipeline, involving:

1. Data Preprocessing (SQL + Python):
    Cleaned and imported customer, service, and churn data.
    Joined related tables (e.g., customer and services).
    Handled missing values and engineered new features (like total charges, tenure groups).

2. Power BI Dashboard Development:
    
    Churn Overview    
    Demographics Analysis
    Contract & Service Impact
    Predictive Segmentation
    Used DAX to calculate churn KPIs and applied slicers to filter views.

3. Machine Learning Integration (Python):
    Applied Logistic Regression, Random Forest, and XGBoost models.
    Split the data (80/20) and balanced the churned vs. non-churned classes using SMOTE.
    Achieved over 80% accuracy with Random Forest and XGBoost.
    Deployed the model in a Jupyter Notebook and exported predictions for integration into Power BI.



## Power BI Dashboard & Code
- Power BI Report: Screenshot
![Churn Analysis Summary_1](https://github.com/user-attachments/assets/6a693308-3202-4598-9272-b8fa56bcf756)
![Churn Analysis Prediction_2](https://github.com/user-attachments/assets/7beef630-e2f8-4ef9-a2ca-e084234c7cee)

- Python & SQL scripts:
[SQL Queries.docx](https://github.com/user-attachments/files/20120916/SQL.Queries.docx)
[Python Code with Random Forest.docx](https://github.com/user-attachments/files/20120915/Python.Code.with.Random.Forest.docx)
[Power Query Transformations_Shaik.docx](https://github.com/user-attachments/files/20120913/Power.Query.Transformations_Shaik.docx)
[Customer_Data.csv](https://github.com/user-attachments/files/20120920/Customer_Data.csv)


