# Customer Churn Prediction – End-to-End MLOps Infrastructure

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![AWS](https://img.shields.io/badge/AWS-SageMaker%20%7C%20Glue%20%7C%20S3-orange?logo=amazon-aws)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-blue?logo=mlflow)
![Grafana](https://img.shields.io/badge/Grafana-Monitoring-orange?logo=grafana)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red?logo=streamlit)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-CI%2FCD-black?logo=githubactions)

This project demonstrates how to design and implement a **production-ready MLOps platform** for customer churn prediction.  
It replicates real-world practices used in industry for **data processing, automated model training, experiment tracking, and monitoring**, with deployment-ready hooks for serving models.

---

## 🚀 Project Overview
The goal of this project is to move beyond model development and focus on the **operational backbone** of machine learning:
- Automated ETL and feature engineering  
- Scalable multi-model training with hyperparameter tuning  
- Experiment tracking and model versioning  
- Monitoring and drift detection framework **designed to support deployed models**  
- Lightweight frontend for interactive prediction demos  

---

## ⚙️ Architecture
<p align="center">
  <img src="architecture.png" alt="MLOps Architecture" width="800"/>
</p>

_The architecture diagram above shows the end-to-end flow from raw data ingestion to monitoring and deployment-ready endpoints._  

---

## 🔑 Core Components
- **Data Pipeline**: AWS Glue ETL jobs using medallion architecture, categorical feature engineering, and one-hot encoding  
- **Training Automation**: GitHub Actions orchestrating multi-algorithm experimentation on SageMaker with hyperparameter optimization  
- **Experiment Management**: MLflow for tracking, reproducibility, and automated best-model selection  
- **Monitoring & Analysis**: Real-time model performance tracking using S3 data lakes, Athena queries, and Grafana dashboards  
- **Quality Assurance**: Prediction analysis and drift detection framework **designed for deployed models**  
- **Frontend Demo**: Streamlit app for showcasing predictions interactively  

---

## 🔐 IAM & Security
To support secure automation, the project uses IAM roles with scoped permissions:
- **SageMaker Execution Role** → Grants SageMaker access to S3, training jobs, and MLflow tracking  
- **Glue/EMR Service Role** → Handles ETL orchestration and read/write operations to S3  
- **GitHub Actions Role (OIDC)** → Configured for CI/CD pipelines without storing long-lived AWS keys  

This ensures that all components (ETL, training, monitoring) run securely and adhere to AWS **least-privilege principles**.  

---

## 🏗️ Tech Stack
- **Cloud**: AWS (SageMaker, Glue, S3, Athena)  
- **MLOps Tools**: MLflow, Grafana, GitHub Actions  
- **ML Libraries**: scikit-learn, XGBoost  
- **Frontend**: Streamlit  

---

## 📊 Monitoring Dashboard
Real-time model monitoring via Grafana:  
👉 [View Grafana Dashboard](https://sankalp20487.grafana.net/d/438bb8a0-2468-462c-9bde-15719f249ad6/customerchurnmodelmonitoring?orgId=1&from=2025-08-22T04:00:00.500Z&to=2025-09-03T03:59:58.500Z&timezone=browser)

---

## 📂 Repository Structure

<img width="597" height="321" alt="image" src="https://github.com/user-attachments/assets/001cdae5-1864-4f84-ac36-b7db2f768354" />

## 🖥️ Streamlit Frontend
A lightweight Streamlit app is included to demonstrate interactive predictions.

To run locally:

-- cd streamlit_app
-- streamlit run app.py

## 🔮 Future Improvements

1. Automate deployment to SageMaker Serverless endpoints (staging → prod with manual approval)
2. Expand monitoring with Prometheus + Grafana alerts
3.Add SHAP explainability for churn predictions
4.Integrate a feature store for managing engineered features

## 📬 Resources
GitHub Repository: Customer_Churn_MLops

Grafana Monitoring Dashboard: View Here

📌 Key Takeaway
This project highlights the importance of infrastructure in ML — going beyond training models to building the automation, monitoring, and governance that keep ML systems reliable in production.

