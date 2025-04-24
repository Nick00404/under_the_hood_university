
- [ 07 lab data drift simulation with evidently.ipynb ](./07_lab_data_drift_simulation_with_evidently.ipynb)  
- [ 08 lab concept drift tracking with river.ipynb ](./08_lab_concept_drift_tracking_with_river.ipynb)  
- [ 09 lab alerting pipeline with prometheus.ipynb ](./09_lab_alerting_pipeline_with_prometheus.ipynb) 
- [ 10_lab_monitor_model_latency_with_prometheus.ipynb ](./10_lab_monitor_model_latency_with_prometheus.ipynb)
- [ 11_lab_concurrent_traffic_with_locust.ipynb ](./11_lab_concurrent_traffic_with_locust.ipynb)
- [ 12_lab_grafana_dashboard_for_live_model_metrics.ipynb ](./12_lab_grafana_dashboard_for_live_model_metrics.ipynb)

## ✅ **05_model_monitoring**

This module focuses on tracking the health, performance, and integrity of deployed models over time. It covers techniques for monitoring model performance, detecting data and concept drift, and setting up alerting mechanisms to ensure robust and reliable model operations.

---

### **1. Data Drift Detection with Evidently**

**1.1 Core Concepts of Data Drift**  
- Identifying shifts in input feature distributions  
- Statistical methods for drift detection (e.g., KL divergence, Kolmogorov-Smirnov test)  

**1.2 Evidently Overview**  
- Setting up Evidently for monitoring model inputs and outputs  
- Customizing drift detection reports and visualizations  

**1.3 Monitoring Drift at Scale**  
- Handling large-scale feature monitoring (batch vs. streaming)  
- Integrating with data lakes or warehouses (e.g., S3, BigQuery)  

*Lab: Set up Evidently to monitor data drift for a classification model over time.*

---

### **2. Concept Drift Detection with River**

**2.1 Concept Drift Fundamentals**  
- Difference between data drift and concept drift  
- Detection algorithms (e.g., ADWIN, DDM) and their applications  

**2.2 River Library for Streaming Data**  
- Working with River for online learning and drift detection  
- Implementing concept drift detection in a streaming model environment  

**2.3 Adapting Models to Concept Drift**  
- Incremental learning with River  
- Re-training models to adapt to detected drift patterns  

*Lab: Detect and handle concept drift in a live recommendation system using River.*

---

### **3. Performance Tracking with Prometheus & Grafana**

**3.1 Prometheus for Metrics Collection**  
- Exposing custom model metrics via Prometheus client library  
- Defining key performance indicators (KPIs): latency, throughput, success rate  

**3.2 Grafana Dashboards for Visualization**  
- Designing custom dashboards to track model performance metrics  
- Real-time monitoring and alerting in Grafana based on Prometheus data  

**3.3 Scaling and Alerting Mechanisms**  
- Defining alert thresholds and anomaly detection  
- Integrating Slack or email notifications for performance anomalies  

*Lab: Set up Prometheus to monitor model latency and throughput, and visualize it in Grafana.*

---

### **4. Model Performance Dashboards**

**4.1 Key Performance Metrics**  
- Accuracy, F1 score, precision/recall, and business-specific metrics (e.g., customer satisfaction score)  
- Tracking model outputs over time to identify performance degradation  

**4.2 Interactive Dashboards**  
- Using Dash or Streamlit to build interactive visualizations  
- Displaying performance metrics and drift indicators in real-time  

**4.3 Performance Benchmarks**  
- Comparing model versions (e.g., comparing A/B test results)  
- Historical performance and trend analysis  

*Lab: Build an interactive dashboard to track the F1 score and data drift of your deployed model.*

---

### **5. Alerting & Integration with Slack**

**5.1 Defining Alert Rules**  
- Alerting thresholds based on key metrics (e.g., 10% drop in accuracy, increased latency)  
- Setting up anomaly detection algorithms for automatic alerts  

**5.2 Integrating Alerts with Slack**  
- Configuring Slack Webhooks for automated notifications  
- Building rich message formatting with alert details and links  

**5.3 Post-Alert Actions**  
- Triggering automated re-training workflows from alerts  
- Integrating with incident management tools (e.g., PagerDuty)  

*Lab: Set up an alerting system in Slack that triggers when model accuracy falls below a predefined threshold.*

---

### ✳️ **Pedagogical Goals Across the Module**

- **Proactive Monitoring**: Emphasize the importance of continuously monitoring models for performance issues and data shifts.  
- **Data and Concept Drift**: Teach students to differentiate and handle both types of drift in real-time systems.  
- **Automation & Integration**: Ensure that students can integrate monitoring tools with CI/CD pipelines and alerting systems for end-to-end automation.

---

| Lab Filename                                                  | Purpose |
|---------------------------------------------------------------|---------|
| `07_lab_data_drift_simulation_with_evidently.ipynb`           | Inject synthetic feature drift and visualize it using the Evidently dashboard |
| `08_lab_concept_drift_tracking_with_river.ipynb`              | Track concept drift in real-time using a streaming model and rolling accuracy stats |
| `09_lab_alerting_pipeline_with_prometheus.ipynb`              | Connect model metrics to Prometheus and trigger Slack alerts when thresholds are crossed |
| `10_lab_monitor_model_latency_with_prometheus.ipynb`          | Record model inference times and set Prometheus-based alerts for latency spikes |
| `11_lab_concurrent_traffic_with_locust.ipynb`                 | Simulate concurrent API load on a model server using Locust and monitor failure patterns |
| `12_lab_grafana_dashboard_for_live_model_metrics.ipynb`       | Build a Grafana dashboard for visualizing real-time ML metrics like accuracy, latency, drift |