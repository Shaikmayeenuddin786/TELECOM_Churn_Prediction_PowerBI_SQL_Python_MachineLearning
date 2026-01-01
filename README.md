#  Telecom Customer Churn Analysis & Prediction | Churn Risk Segmentation, Predictive Modeling, Proactive retention strategies.




## Project Overview

This project aims to analyze telecom customer behavior and predict churn using machine learning, SQL, and Power BI. The goal is to help telecom companies identify at-risk customers and take proactive steps to improve retention.
Using data analysis, we modeled churn behavior and created an interactive dashboard that provides key demographic, geographic, and behavioral insights to support business decisions.

---

## Business Problem

Customer churn remains a serious challenge for telecoms, with rates nearing 27%, leading to lost revenue and higher acquisition costs.
Churn patterns differ by region, age, and service, but sales teams lack actionable insights to respond quickly.
By predicting who’s likely to leave, and understanding their reasons, teams can focus retention efforts on high-risk groups and minimize future losses.

---

## Objectives

- Predict customers who are likely to churn using historical data.
- Analyze churn patterns across age, gender, state, services, contract type, and more.
- Deliver actionable insights via a Power BI dashboard.
- Enable leadership teams to track churn KPIs in real time.

---

## Why this project
This project demonstrates an end-to-end ML deployment: 
data → model → API → containerized service.
It’s built for real-world decisioning (risk segmentation, retention revenue estimation, actionable reasons per customer)

#  A Dockerized FastAPI service that loads a trained Random Forest churn model, returns batch predictions with probabilities, risk levels, business summary metrics, and human-readable key factors.

**Caption:** Model trained offline (Jupyter) → artifacts saved → FastAPI loads artifacts → Docker container serves REST endpoint → Swagger / PowerShell / Power BI call the endpoint for live scoring.

## Features
- Batch inference via `/predict` (accepts multiple customer records)
- Per-customer: prediction, probability, risk level, and key factors
- Business summary: churn rate, churn count, retention revenue potential
- Swagger UI documentation (auto-generated)
- Dockerized for easy local or cloud deployment


---

# Start-to-end pipeline (concise steps)
1. **Data collection** — gather raw customer/usage, billing, and interaction data.  
2. **Exploration & cleaning** — EDA, missing value handling, consistent naming, save training snapshot.  
3. **Feature engineering** — tenure, averages, contract flags, payment method encodings.  
4. **Train & validate** — model selection (Random Forest), cross-validation, calibration if needed.  
5. **Save artifacts** — model (`.pkl`), encoders, feature list, metadata.  
6. **Build service** — FastAPI `predict` endpoint with safe preprocessing.  
7. **Containerize** — Dockerfile + build image.  
8. **Local test** — Swagger, PowerShell, batch requests.  
9. **Deploy** — API using FastAPI + Docker

---

## Repository structure
telecom-churn/
├── api/
│ ├── main.py # FastAPI app (predict endpoint)
│ ├── requirements.txt
├── data/
│ ├── 01_raw_source/customer_data_original.csv
│ └── 02_current_input/PredictionData.xlsx
├── python/
│ └── churn_ml_model.ipynb # training & preprocessing
│ └── score.py
├── model/
│  ├── churn_model.pkl
│  └── label_encoders.pkl
├── Dockerfile



---
## progression

Layer 1: Power BI — insight discovery & business storytelling
Layer 2: API — production-grade ML inference

“I intentionally kept the Power BI dashboard and the churn prediction API decoupled.
This ensured demo reliability, avoided unnecessary cloud costs, and kept the architecture clean.
If required, the API can be integrated into Power BI using DirectQuery or REST, but for portfolio and interview purposes, separation was the most practical design choice.”


---

## Executive Summary and Project Workflow
**(A high-level summary)**

- Top Summary Box: Shows basics—6,418 total customers, 411 new joiners, 1,732 churned (27% rate).
- Gender Pie: Males (64%) churn more than females (36%).
- Age Bars: 20-35 year-olds have the highest churn (31%); under-20s are lowest.
- State Bars: Top churn states: Jammu (57%), Assam (38%), Jharkhand (35%).
- Internet Type Bars: Fiber optic users churn most (41%); none/low users least(8%).
- Payment & Contract Bars: Mailed checks (38%) and month-to-month contracts (47%) drive high churn.
- Services Table: Users with phone service (91%) or internet (93%) churn less if they have them.
- Churn Reasons Bars: Competition causes most (44%); price least (11%).
- Prediction Side: Flags 376 at-risk customers; profiles show mostly females (65%), short-tenure (month-to-month 94%), in states like Uttar Pradesh.


---

## Insights Deep Dive
**(A high-level summary of key churn findings)**

- The overall customer churn rate is **27%.**
- Older customers (50+) and those on month-to-month contracts are most at risk.
- Jammu & Kashmir, Assam, and Jharkhand show the highest churn percentages.
- Lack of services like **Device Protection**, **Online Security**, and **Premium Support** was common among churners.
- **Mailed Check payment** users churn more frequently than Credit Card users.
- Customers with **Fiber Optic connections** and multiple services have higher churn likelihood.
- **Young adults (20-35)** in high-churn states on fiber and month-to-month plans drop fastest—these should be prioritized for retention.
- Not having device protection services increases churn risk by **30-40%.**
- **44%** of churners leave for competitors, highlighting the importance of attractive bundles.

---

#  Dashboard Snapshots

###  Dashboard Summary
- Power BI Report: Screenshot
![Churn Analysis Summary_1](https://github.com/user-attachments/assets/6a693308-3202-4598-9272-b8fa56bcf756)
![Churn Analysis Prediction_2](https://github.com/user-attachments/assets/7beef630-e2f8-4ef9-a2ca-e084234c7cee)

### Relationship Table Mapping
![Mapping_Tables](https://github.com/user-attachments/assets/1baf0c78-1b38-4af8-8342-fa5e1b82e84b)

---
# Top 10 Churn Reasons – ( Insights from Churn Analysis Projects):
### (Prioritized key factors driving churn, highlighting areas for targeted action)


## Sorted by Impact Priority
- **High Monthly Charges:** Customers paying higher-than-average fees often leave due to perceived lack of value or pricing dissatisfaction.
- **Month-to-Month Contracts:** Customers on short-term, flexible plans are more likely to churn compared to annual or two-year contracts.
- **Low Tenure / New Customers:** Newer customers (0–12 months) are more likely to churn, often due to unmet expectations or poor onboarding.
- **Multiple Complaints or Support Calls:** High interaction with customer support (especially unresolved issues) often correlates with dissatisfaction and churn.
- **Service Downtime / Technical Issues:** Repeated service failures or outages lead to frustration and switching.
- **Lack of Add-On Services:** Churn is higher among customers not using services like online backup, tech support, or security features.
- **No Internet Service / DSL Users:** Customers not using internet services or using slower options (like DSL) tend to churn more than fiber users.
- **Senior Citizen Segment:** This demographic may show higher churn depending on digital literacy, support needs, or service relevance.
- **Lack of Loyalty Incentives:** Absence of targeted offers, discounts, or retention strategies contributes to early exits.
- **No Paperless Billing or AutoPay:** Indicates lower engagement or trust, often a churn signal.

# _____________________________________
# Offered Recommended Solutions to Sales & Marketting and Leadership team to Reduce Churn significantly:

- **Target At-Risk Segments Proactively:** Use churn risk scores to send retention offers or personalized support to high-risk customers.
- **Promote LongTerm Contracts:** Offer discounts or loyalty perks for customers who switch from month-to-month to 1- or 2-year plans.
- **Upsell Value-Add Services:** Bundle services like online security, tech support, or streaming to increase stickiness and engagement.
- **Improve Onboarding for New Customers:** Implement welcome programs, personalized setup guides, and first-30-day follow-up calls to reduce early churn.
- **Enhance Service Quality for DSL Users:** Migrate legacy customers to higher-speed options like fiber with promotional pricing or device upgrades.
- **Reward Loyalty:** Implement rewards, referral bonuses, or exclusive discounts for long-term, low-complaint customers.
- **Reduce Support Friction:** Improve first-call resolution and reduce wait times; invest in AI chatbots or self-service tools.
- **Introduce Tiered Pricing Models:** Align pricing with usage patterns and customer segments to provide more value at lower perceived cost.
- **Monitor and Act on Feedback:** Regularly review NPS, CSAT, and complaint data; close the loop with customers when issues are resolved.
- **Incentivize Paperless Billing and AutoPay:** Offer small credits or priority support to customers who switch to these options (shows commitment).



---

# **Tools Used**

Power BI , ML, SQL, Python, Scikit-learn, XGBoost /Random Forest , Jupyter*

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
#### Created multiple pages:
  - **Summary Dashboard**: High-level KPIs (Churn %, Customer Count, Joiners)
  - **Prediction Dashboard**: Predicted Churners with filters and breakdowns
#### Used DAX measures, slicers, and drill-through filters to explore:
  - Churn by State, Age, Gender, Contract, Payment Type, Services
  - Churn by Category (e.g., Competition, Price, Dissatisfaction)

---

##  How to Use

1. Clone the repo or download as ZIP.
2. Open the `.pbix` file in Power BI Desktop to explore the dashboard.
3. Open `telecom_churn_model.ipynb` in Jupyter to review or retrain the ML model.
4. Check SQL folder for the views used in ETL processing.

---

##  Author & Acknowledgment
 Shaik Mayeenuddin
 Aspiring Data Scientist Professional | Supply Chain & Marketing Analytics Expert
 Pursuing a Master’s in Data Science (AI & ML) Student
🔗 https://www.linkedin.com/in/shaikmayeenuddin

### This project demonstrates my complete end-to-end capability—from data ingestion and cleaning, through ETL pipelines, modeling, and dashboarding, to actionable business insights. I architected, modeled, validated, visualized, and strategically interpreted the data throughout.

- **Data Engineering & ETL** – Built SQL-import pipelines , (Power BI, Python) for real-time insights  
- **Data Cleaning & Modeling** – Processed and stored 50K+ records using Pandas, NumPy, and robust normalization techniques  
- **Analytical Modeling** – Developed Random Forest churn-prediction model with 85% validation accuracy  
- **Dashboards & Visualization** – Created five cross-functional Power BI dashboards featuring KPI slicers, bookmarks, and custom UX  
- **Strategic Insights & Recommendations** – Presented actionable data findings to senior leadership (140+ stakeholders) to drive retention, inventory, and content strategy  
- **Performance Monitoring & Automation** – Implemented real-time alerting and monitoring using Power BI Service, SQL jobs, and Kubernetes for pipeline status


 This project is built upon the foundational work by **Pivotalstats** . 
 I am grateful for the analytical insights shared through their tutorials
