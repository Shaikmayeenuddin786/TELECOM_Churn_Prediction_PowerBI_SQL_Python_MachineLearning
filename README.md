#  Telecom Customer Churn Analysis & Prediction

**Tools Used:** Power BI | SQL Server | Python (Pandas, Scikit-learn) | Machine Learning (Random Forest)

## Project Overview

This project aims to analyze telecom customer behavior and predict churn using machine learning, SQL, and Power BI. The goal is to help telecom companies identify at-risk customers and take proactive steps to improve retention.

Using data analysis, we modeled churn behavior and created an interactive dashboard that provides key demographic, geographic, and behavioral insights to support business decisions.

---

## Business Problem

Customer churn is a significant issue in the telecom industry. Losing existing customers not only impacts revenue but also increases the cost of acquiring new ones. Understanding **why customers leave** and **who is likely to churn** can help businesses implement effective retention strategies.

---

## Objectives

- Predict customers who are likely to churn using historical data.
- Analyze churn patterns across age, gender, state, services, contract type, and more.
- Deliver actionable insights via a Power BI dashboard.
- Enable leadership teams to track churn KPIs in real time.

---

## Data Summary

- Total Records: **6,418 Customers**
- Fields Included: Demographics, Services, Contract, Charges, Usage Patterns, etc.
- Target Variable: `Churn` (Yes/No)

---

## Project Workflow

### 1. **Data Cleaning & Preparation (SQL Server)**
- Cleaned and filtered the raw telecom dataset using SQL queries.
- Created SQL Views for:
  - `Churned Customers`
  - `Joined Customers`
  - `Monthly Charge & Tenure Buckets`

### 2. **Churn Prediction Modeling (Python)**
- Preprocessed the data using Pandas (encoding, missing values, feature selection).
- Trained a **Random Forest Classifier**:
  - Accuracy: **~81%**
  - Evaluated using Confusion Matrix, Precision, Recall, and F1-score.
- Identified key churn drivers (e.g., Contract Type, Device Protection, Tenure).

### 3. **Dashboarding (Power BI)**
- Created multiple pages:
  - **Summary Dashboard**: High-level KPIs (Churn %, Customer Count, Joiners)
  - **Prediction Dashboard**: Predicted Churners with filters and breakdowns
- Used DAX measures, slicers, and drill-through filters to explore:
  - Churn by State, Age, Gender, Contract, Payment Type, Services
  - Churn by Category (e.g., Competition, Price, Dissatisfaction)

---

## Key Insights

- **27% churn rate** across the customer base.
- Customers aged **50+** and those with **month-to-month contracts** are at higher risk.
- States like **Jammu & Kashmir, Assam**, and **Jharkhand** had the highest churn %.
- **Lack of services** like **Device Protection**, **Online Security**, and **Premium Support** was common among churners.
- Customers on **Mailed Check** payments churned more than **Credit Card** users.
- Customers with **Fiber Optic** and **multiple services** were more likely to leave.

---

##  Sample Dashboard Snapshots

###  Summary Dashboard
- Power BI Report: Screenshot
![Churn Analysis Summary_1](https://github.com/user-attachments/assets/6a693308-3202-4598-9272-b8fa56bcf756)
![Churn Analysis Prediction_2](https://github.com/user-attachments/assets/7beef630-e2f8-4ef9-a2ca-e084234c7cee)



## Project Structure

---

##  How to Use

1. Clone the repo or download as ZIP.
2. Open the `.pbix` file in Power BI Desktop to explore the dashboard.
3. Open `telecom_churn_model.ipynb` in Jupyter to review or retrain the ML model.
4. Check SQL folder for the views used in ETL processing.

---

## Author

**Shaik Mayeenuddin**  
Business Analyst | Data Analytics | AI & ML Student  
🔗 [LinkedIn](https://www.linkedin.com/in/shaikmayeenuddin) | [GitHub](https://github.com/Shaikmayeenuddin786)
