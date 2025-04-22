# 01 Streaming Fraud Detection

- [data ingestion pipeline](./data_ingestion_pipeline.ipynb)
- [model serving fastapi](./model_serving_fastapi.ipynb)
- [monitoring dashboard](./monitoring_dashboard.ipynb)
- [real time feature engineering](./real_time_feature_engineering.ipynb)

---

## 📘 **Capstone: Streaming Fraud Detection – Structured Index**

---

### 🧩 **01. Real-Time Data Ingestion and Processing**

#### 📌 **Includes: `data_ingestion_pipeline.ipynb`**

##### **Subtopics:**
- **Stream vs Batch Ingestion**
  - Understanding the need for real-time fraud detection pipelines
- **Apache Kafka / Flink / Spark Streaming**
  - Tools for high-throughput message ingestion
- **Streaming Pipeline Architecture**
  - Producers, brokers, consumers, and schema design
- **Example:** Simulating real-time transaction flow using Kafka + Python

---

### 🧩 **02. Real-Time Feature Engineering and Model Serving**

#### 📌 **Includes: `real_time_feature_engineering.ipynb`, `model_serving_fastapi.ipynb`**

##### **Subtopics:**
- **Feature Stores and Stream-Aware Engineering**
  - Windowed features (e.g., last 5 mins), aggregates, feature freshness
- **Online vs Offline Features**
  - Ensuring consistency between training and live inference
- **FastAPI for Low-Latency Model Serving**
  - Deploying a trained fraud detection model with FastAPI
- **Example:** Creating features on the fly and scoring them via a REST API

---

### 🧩 **03. Monitoring and Observability**

#### 📌 **Includes: `monitoring_dashboard.ipynb`**

##### **Subtopics:**
- **Why Monitoring Is Critical in Fraud Detection**
  - Detecting concept drift, latency spikes, and false positives
- **Metrics to Track**
  - Prediction volume, feature value distributions, model confidence
- **Visualization Tools**
  - Using Grafana, Prometheus, or Streamlit for dashboards
- **Example:** Real-time fraud monitoring dashboard tracking model performance over time

---

### 🧠 Bonus Ideas:
- Integrate **anomaly detection** as a fallback when the model is uncertain
- Add a **feature drift monitor** using `evidently` or `Riverml`
- Connect with a **CI/CD pipeline** for model updates on real-world data drift

---
