# 01 End To End Mlops Platform

- [data versioning dvc](./data_versioning_dvc.ipynb)
- [drift monitoring](./drift_monitoring.ipynb)
- [kubeflow pipelines](./kubeflow_pipelines.ipynb)
- [model serving seldon](./model_serving_seldon.ipynb)

---

### 🧮 **01. Data Versioning & Experiment Tracking with DVC**

#### 📌 **Subtopics Covered:**
- Setting up **DVC** for data & model version control  
- Connecting to remote storage (S3, GDrive, etc.)  
- Pipelines: `dvc.yaml` stages for preprocessing → training → evaluation  
- Linking experiments to **Git commits** for reproducibility  

---

### 🔄 **02. Model Drift Monitoring & Alerts**

#### 📌 **Subtopics Covered:**
- Detecting **covariate drift**, **label drift**, and **concept drift**  
- Using tools like **Evidently AI**, **Alibi Detect**, or **Fiddler**  
- Dashboard visualizations for live drift stats  
- Setting up thresholds + alerting via Slack/Webhooks  

---

### ⚠️ **03. Incident Response Playbook** (`incident_response_playbook.md`)

#### 📌 **Contents Covered:**
- Actionable steps for ML incidents (e.g., drift, latency spikes, data corruption)  
- Role assignment: Who handles what  
- Logging best practices & rollback strategies  
- Communication templates for reporting and escalation  

---

### 🧪 **04. Kubeflow Pipelines for Scalable Workflows**

#### 📌 **Subtopics Covered:**
- Building pipeline components (preprocess → train → validate → deploy)  
- Parameterizing hyperparameters and datasets  
- Running on Kubernetes cluster (with GPU support)  
- Managing pipeline versions and artifacts  

---

### 📦 **05. Model Serving with Seldon Core**

#### 📌 **Subtopics Covered:**
- Creating custom Docker models with `s2i` or Python wrappers  
- Deploying with Seldon CRDs (SeldonDeployment)  
- Load balancing, canary rollouts, and autoscaling  
- Monitoring model inputs/outputs with Prometheus + Grafana  

---

### ✅ Summary

> This capstone builds a **robust MLOps backbone** — covering **versioning**, **orchestration**, **deployment**, and **production monitoring**. It's everything a modern AI company needs to scale responsibly.

---
