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

## Executive Summary and Project Workflow

- Total Records: **6,418 Customers**
- Fields Included: Demographics, Services, Contract, Charges, Usage Patterns, etc.
- Target Variable: `Churn` (Yes/No)

---

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

## Insights Deep Dive

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

---
# Top Churn Reasons – ( Insights from Churn Analysis Projects):


- Month-to-Month Contracts :  Customers on short-term, flexible plans are more likely to churn compared to annual or two-year contracts.

- 	High Monthly Charges : Customers paying higher-than-average fees often leave due to perceived lack of value or pricing dissatisfaction.
- 	Low Tenure / New Customers : Newer customers (0–12 months) are more likely to churn, often due to unmet expectations or poor onboarding.
- 	Lack of Add-On Services : Churn is higher among customers not using services like online backup, tech support, or security features.
- 	No Internet Service / DSL Users : Customers not using internet services or using slower options (like DSL) tend to churn more than fiber users.
- 	Senior Citizen Segment : This demographic may show higher churn depending on digital literacy, support needs, or service relevance.
-	Multiple Complaints or Support Calls : High interaction with customer support (especially unresolved issues) often correlates with dissatisfaction and churn.
- 	No Paperless Billing or AutoPay: Indicates lower engagement or trust, often a churn signal.
- 	Service Downtime / Technical Issues: Repeated service failures or outages lead to frustration and switching.
- 	Lack of Loyalty Incentives :Absence of targeted offers, discounts, or retention strategies contributes to early exits.

# _____________________________________

# Recommended Solutions to Reduce Churn:

- Promote Long-Term Contracts : Offer discounts or loyalty perks for customers who switch from month-to-month to 1- or 2-year plans.

- Introduce Tiered Pricing Models : Align pricing with usage patterns and customer segments to provide more value at lower perceived cost.
- Improve Onboarding for New Customers : Implement welcome programs, personalized setup guides, and first-30-day follow-up calls to reduce early churn.
- 	Upsell Value-Add Services : Bundle services like online security, tech support, or streaming to increase stickiness and engagement.
- 	Target At-Risk Segments Proactively: Use churn risk scores from your model to send retention offers or personalized support to high-risk customers.
- 	Enhance Service Quality for DSL Users : Migrate legacy customers to higher-speed options like fiber with promotional pricing or device upgrades.
- 	Reward Loyalty : Implement rewards, referral bonuses, or exclusive discounts for long-term, low-complaint customers.
- 	Reduce Support Friction :Improve first-call resolution and reduce wait times; invest in AI chatbots or self-service tools.
- 	Incentivize Paperless Billing and AutoPay: Offer small credits or priority support to customers who switch to these options (shows commitment).
- 	Monitor and Act on Feedback : Regularly review NPS, CSAT, and complaint data; close the loop with customers when issues are resolved.


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
