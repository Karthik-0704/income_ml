# Income Classification Service

A production-grade Machine Learning pipeline designed to classify individual income levels based on census data. This project implements a full **MLOps lifecycle**, from automated training and experiment tracking to cloud-based containerized deployment.

## 🏗️ System Architecture



The workflow follows a modern CI/CD pattern:
1.  **Experimentation**: Models are trained locally or via CLI, with all metrics and artifacts logged to **DagsHub**.
2.  **Version Control**: Code changes pushed to **GitHub** trigger automated workflows.
3.  **CI/CD**: **GitHub Actions** builds a Docker image, pushes it to Docker Hub, and signals the **AWS EC2** runner.
4.  **Deployment**: The API is hosted inside a **Docker** container on EC2, accessible via a ligthweight **Streamlit** frontend hosted on a separate hosting platform.

---

## 🛠️ Tech Stack & Services

* **MLflow & DagsHub**: Used for experiment tracking, parameter logging, and artifact storage.
* **FastAPI**: The high-performance backend engine that handles on-demand model swapping and inference.
* **Streamlit**: A lightweight frontend hosted separately for UI/UX.
* **Docker**: Ensures environment consistency across development and production.
* **AWS EC2 (t2.small)**: Cloud infrastructure hosting the live API.
* **GitHub Actions**: Orchestrates the automated deployment and runner services.

---

## 🎥 Product Demo

Check out the engine in action!

https://github.com/user-attachments/assets/2d6a528f-a5a9-435e-ad03-4543dbef3c60

### **Walkthrough Highlights:**
* **Real-time Configuration**: Toggling Algorithms and Balancing Strategies.
* **JSON Inference**: Executing a live prediction against the FastAPI production endpoint.

## 📊 Model Performance Highlights
The engine currently supports four model configurations with the following benchmarks tracked in MLflow:

The engine currently supports four model configurations with the following benchmarks tracked in MLflow. Performance varies based on the class-balancing strategy used during training:

| Algorithm Family | Balancing Strategy | Accuracy | Precision | Recall | Test ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost** | Balanced | 86.73% | 32.03% | 90.24% | 0.953 |
| **XGBoost** | Unbalanced | 95.54% | 74.48% | 49.27% | 0.955 |
| **Random Forest** | Balanced | 93.89% | 52.96% | 65.62% | 0.947 |
| **Random Forest** | Unbalanced | 95.40% | 81.16% | 39.33% | 0.950 |

> **Note**: Balanced models prioritize **Recall** (finding as many >$50k earners as possible), while Unbalanced models prioritize **Precision** (ensuring those labeled >$50k are highly likely to be correct).

🔒 Security Configuration
GitHub Secrets: Sensitive data like Server IPv4, DAGSHUB_TOKEN and MLFLOW_TRACKING_URI are injected at runtime.

AWS Security Groups: Limited port configurations are opened for inbound TCP traffic to allow only authorized frontend communication.

## 🎥 Product Demo (Test Deployment)

This project is currently in a **Phase 1: Test Deployment** stage. The primary focus is verifying the end-to-end connection between the GitHub runner, the AWS-hosted FastAPI backend, and the DagsHub artifact storage.

### **Current Capability:**
* **Live Inference**: The frontend successfully sends raw JSON to the FastAPI endpoint on AWS.
* **On-Demand Loading**: The backend dynamically fetches and loads the requested model version from DagsHub artifacts.
* **Scalable Infrastructure**: The system is containerized with Docker, allowing for consistent deployment across different cloud environments.

---

## 🔮 Future Scope & Business Implementation

As this project transitions from a technical prototype to a production-ready solution, the following enhancements are planned to maximize its socio-economic impact:

### **1. Technical Roadmap**
* **Dynamic Dashboards**: Integrating real-time visualization of Drift Detection and Model Health metrics directly into the frontend.
* **Auto-Scaling Infrastructure**: Transitioning from a single t2.small instance to an auto-scaling group or Kubernetes (EKS) to handle high-concurrency traffic during peak hours.
* **Automated Retraining**: Implementing event-driven triggers that automatically retrain the model when data drift is detected in the census features.

### **2. Business & Socio-Economic Applications**
This engine provides high-value insights for several key industries:
* **Financial Services & Banking**: Assisting in credit risk assessment by providing an additional data point for an applicant's potential income bracket.
* **Targeted Marketing**: Enabling e-commerce platforms to segment users by socio-economic tiers for more relevant product recommendations and premium service offers.
* **Public Policy & Urban Planning**: Helping government agencies identify economic trends in specific demographic segments to better allocate resources and social welfare programs.
* **Smart City Integration**: Using income distribution data to optimize urban development, such as planning public transportation routes and affordable housing projects.
