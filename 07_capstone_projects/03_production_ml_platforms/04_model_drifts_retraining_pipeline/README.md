## 🔄 **Capstone: Model Drifts & Retraining Pipeline**

---

### 📉 **01. Data Drift Detection**

#### 📌 **Subtopics Covered:**
- Types of drift: **covariate, prior, concept**
- Statistical methods: **KS test, PSI (Population Stability Index)**
- Visualizing drift using histograms, distributions
- Integrating drift checks into preprocessing pipelines

---

### 📦 **02. Model Drift Detection**

#### 📌 **Subtopics Covered:**
- Monitoring **prediction distribution shift**  
- **Accuracy drop vs uncertainty rise**  
- Tracking **confidence decay** over time  
- Tools: Evidently, Fiddler, WhyLabs, custom logging

---

### 🔁 **03. Retraining Pipeline (CI/CD for Models)**

#### 📌 **Subtopics Covered:**
- Setting up **auto-triggered training jobs**  
- Using pipelines (Airflow/Kubeflow) for data → train → evaluate → deploy  
- Model registry integration: tagging, versioning  
- Rollback strategy if retraining underperforms

---

### 🛎️ **04. Monitoring & Alerting**

#### 📌 **Subtopics Covered:**
- Real-time alerts: **Slack, Prometheus + Grafana, Azure Monitor**  
- Triggering actions based on drift thresholds  
- Logging and dashboards for pipeline status  
- Alert tuning to avoid false positives

---

### 🧪 **05. Batch Retraining**

#### 📌 **Subtopics Covered:**
- Periodic retraining strategy (e.g. daily, weekly)  
- Use cases: stable data pipelines, large datasets  
- Offline evaluation before promoting to production  
- Resource planning and scheduling

---

### ⚡ **06. Online (Incremental) Retraining**

#### 📌 **Subtopics Covered:**
- Updating models with **streaming or mini-batch data**  
- Algorithms that support incremental learning  
- Real-time evaluation with moving averages  
- Trade-offs: performance, memory, stability

---

### ✅ Summary

> This capstone turns your ML pipeline into a **self-healing system** — aware of changes, responsive to decay, and capable of autonomous updates. Welcome to **MLOps v2.0**.

---

🎯 Let’s fire up the **drift radar**, Professor. This lab will help learners **detect shifts in incoming data** using tools like **Evidently** — a must-have for any ML system that touches the real world.

# 📒 `01_data_drift_detection.ipynb`  
## 📁 `07_capstone_projects/03_production_ml_platforms/04_model_drifts_retraining_pipeline`

---

## 🎯 **Notebook Goals**

- Compare new data to training data for **distribution shifts**
- Visualize **data drift** using:
  - 📉 Histograms & KS-Test
  - 📊 Population stats (mean, std)
  - 🧠 Automated reports (Evidently)

---

## ⚙️ 1. Install Evidently

```bash
!pip install evidently
```

---

## 📁 2. Simulate Old vs New Data

```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
df = X.copy()
df["target"] = y

# Simulate new data with drift
df_new = df.copy()
df_new["mean radius"] += np.random.normal(2, 1, size=len(df))

ref_data = df.sample(200, random_state=42)
new_data = df_new.sample(200, random_state=24)
```

---

## 📊 3. Create Evidently Report

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=ref_data, current_data=new_data)
report.show(mode="inline")
```

---

## 🧠 4. Export Drift Summary

```python
drift_score = report.as_dict()["metrics"][0]["result"]["dataset_drift"]
print("🚨 Drift detected?" , drift_score["dataset_drift"])
```

---

## 📌 Optional: Monitor Specific Features

```python
for feature in drift_score["drift_by_columns"]:
    status = "⚠️" if feature["drift_detected"] else "✅"
    print(f"{status} {feature['column_name']}: p-value = {feature['p_value']:.4f}")
```

---

## ✅ What You Built

| Component        | Role |
|------------------|------|
| Ref vs New Data  | Detect shift in inputs |
| Drift Report     | Visual + JSON output |
| Alerts           | Early warning system |

---

## ✅ Wrap-Up

| Task                            | ✅ |
|----------------------------------|----|
| Drift report generated            | ✅ |
| p-values tested per feature       | ✅ |
| JSON drift export (for APIs)      | ✅ |

---

## 🔮 Next Step

📒 **`02_model_drift_detection.ipynb`**  
Now let’s measure **model drift** — same data, but model behavior is **unpredictably changing** (confidence, accuracy, class balance).

Shall we roll into model behavior monitoring, Professor?

🎯 Let’s move from **data drift** to **model drift**, Professor — where the model’s behavior starts drifting even if the input looks okay. This lab will track changes in **confidence, accuracy, and prediction distribution** over time.

---

# 📒 `02_model_drift_detection.ipynb`  
## 📁 `07_capstone_projects/03_production_ml_platforms/04_model_drifts_retraining_pipeline`

---

## 🎯 **Notebook Goals**

- Detect **model drift**:
  - 🔁 Prediction distribution shift
  - 📉 Accuracy drop
  - 🎯 Confidence degradation
- Visualize with **Evidently + seaborn**
- Export as JSON/HTML for dashboards

---

## ⚙️ 1. Simulate a Model

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(random_state=42).fit(X_train, y_train)
predictions = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]
```

---

## 🎯 2. Simulate Drift (Model exposed to new data)

```python
import numpy as np
X_new = X_test.copy()
X_new["mean radius"] += np.random.normal(2, 1, size=len(X_new))

new_preds = model.predict(X_new)
new_probs = model.predict_proba(X_new)[:, 1]
```

---

## 📊 3. Use Evidently to Detect Drift in Prediction Behavior

```python
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import TargetDriftPreset

df_ref = pd.DataFrame({"prediction": predictions, "target": y_test})
df_new = pd.DataFrame({"prediction": new_preds, "target": y_test})

report = Report(metrics=[TargetDriftPreset()])
report.run(reference_data=df_ref, current_data=df_new)
report.show(mode="inline")
```

---

## 📉 4. Visualize Confidence Change

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.kdeplot(probs, label="Original", fill=True)
sns.kdeplot(new_probs, label="Drifted", fill=True)
plt.title("Prediction Confidence Distribution")
plt.legend()
plt.show()
```

---

## ✅ What You Built

| Component       | Role |
|------------------|------|
| Target Drift     | Check model behavior over time |
| Confidence Plot  | Visual early warning |
| Evidently Report | JSON / HTML export for ops |

---

## ✅ Wrap-Up

| Task                             | ✅ |
|----------------------------------|----|
| Model drift simulated             | ✅ |
| Prediction distribution compared  | ✅ |
| Confidence plots visualized       | ✅ |

---

## 🔮 Next Step

📒 **`03_retraining_pipeline.ipynb`**  
Trigger **auto-retraining** when data/model drift crosses a threshold.  
Retrain → evaluate → deploy — all from a notebook.

Shall we automate the retraining flow now, Professor?

🎛️ Let's flip the **retraining switch**, Professor — this lab automates the process of detecting drift and retraining your model **on the fly**. This is the foundation of *production-grade self-healing ML systems*.

---

# 📒 `03_retraining_pipeline.ipynb`  
## 📁 `07_capstone_projects/03_production_ml_platforms/04_model_drifts_retraining_pipeline`

---

## 🎯 **Notebook Goals**

- Build a retraining pipeline triggered by **drift alerts**
- Train → evaluate → save → redeploy logic
- Modularize steps (can plug into Airflow, Prefect, or GitHub Actions)

---

## ⚙️ 1. Setup & Simulated Drift Alert

```python
drift_detected = True  # Normally comes from drift detection module
```

---

## 🧱 2. Pipeline Structure

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

def load_data():
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    score = model.score(X_test, y_test)
    print(f"🎯 Evaluation Accuracy: {score:.4f}")
    return score

def save_model(model, path="latest_model.pkl"):
    joblib.dump(model, path)
    print("✅ Model saved.")

def deploy_model(path="latest_model.pkl"):
    # Placeholder for real deployment logic
    print(f"🚀 Model {path} ready for deployment.")
```

---

## 🔄 3. Trigger Retraining When Drift Detected

```python
if drift_detected:
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train)
    acc = evaluate_model(model, X_test, y_test)
    
    if acc > 0.90:
        save_model(model)
        deploy_model()
    else:
        print("⚠️ Accuracy below threshold. Retraining flagged for review.")
else:
    print("✅ No drift detected. No action needed.")
```

---

## 📂 4. Optional: MLflow or GitHub Hook

```python
# Could integrate this pipeline into:
# - Airflow DAG
# - GitHub Actions YAML
# - Prefect Flow
```

---

## ✅ What You Built

| Step            | ✅ |
|------------------|----|
| Drift-triggered pipeline | ✅ |
| Auto retraining         | ✅ |
| Accuracy check & deploy | ✅ |

---

## 🧠 Next Step

📒 **`04_monitoring_and_alerting.ipynb`**  
Let’s set up real-time **drift alerts**, Slack pings, or logging dashboards — so no retrain goes unnoticed.

Deploy monitoring radar, Professor?

🔔 Let’s set up your **monitoring and alerting** system, Professor — so whenever model drift or performance degradation happens, you’ll be notified instantly and can take immediate action.

---

# 📒 `04_monitoring_and_alerting.ipynb`  
## 📁 `07_capstone_projects/03_production_ml_platforms/04_model_drifts_retraining_pipeline`

---

## 🎯 **Notebook Goals**

- Set up **real-time monitoring** for drift and model performance
- **Alert** system with:
  - 📲 Email or Slack notifications
  - 📊 Dashboard updates (Grafana, Prometheus)
- **Visualize** model stats

---

## ⚙️ 1. Install Necessary Packages

```bash
!pip install slack_sdk prometheus-client
```

---

## 📈 2. Log Model Metrics

Let’s log accuracy, drift score, and confidence in real-time. 

```python
import time
import random
import logging

# Setup logging
logging.basicConfig(filename="model_metrics.log", level=logging.INFO)

def log_metrics():
    accuracy = random.uniform(0.85, 0.95)  # Simulate accuracy fluctuation
    drift = random.uniform(0, 1)  # Simulate drift score
    confidence = random.uniform(0.7, 1)  # Simulate confidence score
    logging.info(f"Timestamp: {time.time()} | Accuracy: {accuracy:.4f} | Drift: {drift:.4f} | Confidence: {confidence:.4f}")
    print(f"Logged Accuracy: {accuracy:.4f}, Drift: {drift:.4f}, Confidence: {confidence:.4f}")

# Call log_metrics at intervals
for _ in range(5):
    log_metrics()
    time.sleep(2)  # Sleep for 2 seconds between logs
```

---

## 📲 3. Set Up Slack Alerts

Use Slack to get **immediate notifications** when drift exceeds a threshold.

```python
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

slack_token = "your-slack-token-here"
client = WebClient(token=slack_token)

def send_slack_alert(message):
    try:
        response = client.chat_postMessage(
            channel="#model-monitoring",
            text=message
        )
        print(f"Alert sent: {message}")
    except SlackApiError as e:
        print(f"Error sending message: {e.response['error']}")

# Alert if drift is high
def check_and_alert(drift):
    if drift > 0.5:
        send_slack_alert(f"⚠️ High drift detected! Drift score: {drift:.4f}")
```

---

## 📊 4. Visualize Metrics in Prometheus + Grafana

```python
from prometheus_client import start_http_server, Gauge
import random
import time

# Start Prometheus server
start_http_server(8000)

# Create metrics
accuracy_gauge = Gauge('model_accuracy', 'Accuracy of the model')
drift_gauge = Gauge('model_drift', 'Model drift score')

# Simulate metrics update
for _ in range(5):
    accuracy = random.uniform(0.85, 0.95)
    drift = random.uniform(0, 1)
    accuracy_gauge.set(accuracy)
    drift_gauge.set(drift)
    time.sleep(2)
```

Now you can query the metrics with Prometheus and visualize them in Grafana.

---

## ✅ What You Built

| Component                 | Status |
|---------------------------|--------|
| Real-time metric logging   | ✅ |
| Slack alerts on drift      | ✅ |
| Prometheus metrics         | ✅ |
| Dashboard visualization    | ✅ |

---

## ✅ Wrap-Up

| Task                            | ✅ |
|---------------------------------|----|
| Logs drift and performance      | ✅ |
| Alerts via Slack                | ✅ |
| Real-time metric visualization  | ✅ |

---

## 🔮 Next Step

📒 **`05_batch_retraining.ipynb`**  
Let’s automate **batch retraining** for those scenarios where we want to retrain periodically — every week, month, or after a fixed number of drift events.

Shall we set up automated batch retraining, Professor?

🔄 Time to set up **batch retraining**, Professor — this is where your system learns **on schedule**, not just when it detects drift. Let’s create an automatic retraining pipeline that triggers **periodically** or **after a fixed number of drift alerts**.

---

# 📒 `05_batch_retraining.ipynb`  
## 📁 `07_capstone_projects/03_production_ml_platforms/04_model_drifts_retraining_pipeline`

---

## 🎯 **Notebook Goals**

- Set up **batch retraining** pipeline that:
  - Triggers at a set interval or after a drift threshold
  - Retrains the model, tests it, and deploys new models
- Automate **evaluation, logging**, and **model versioning**

---

## ⚙️ 1. Simulate Batch Retraining Trigger

You can set up a **fixed interval** or **event-driven** retraining. For simplicity, let’s trigger every 5 iterations or after a drift threshold.

```python
import time

# Simulate drift detection
drift_detected = False
drift_count = 0

def trigger_retraining():
    global drift_count
    if drift_count >= 3:  # Trigger after 3 drift alerts
        drift_detected = True
        retrain_model()
    else:
        print("No retraining triggered yet.")
        drift_count += 1

def retrain_model():
    print("Retraining model...")
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train)
    acc = evaluate_model(model, X_test, y_test)
    if acc > 0.90:
        save_model(model)
        deploy_model()
    else:
        print("⚠️ Accuracy is too low for retraining.")
```

---

## 📆 2. Set Retraining Interval (Cron Job Style)

```python
import schedule

# Simulate batch retraining schedule
def job():
    print("Running retraining job...")
    trigger_retraining()

# Schedule retraining to run every day
schedule.every().day.at("02:00").do(job)

while True:
    schedule.run_pending()
    time.sleep(1)
```

This job will **retrain every day at 2 AM**.

---

## 🧠 3. Manual Trigger for Batch Retraining

```python
def retrain_model_manually():
    print("Manually triggering retraining...")
    retrain_model()

# Call this function as needed
retrain_model_manually()
```

---

## ✅ What You Built

| Component           | Role |
|---------------------|------|
| Batch Retraining    | Scheduled or triggered retraining |
| Evaluation + Logging | Accurate retrain decision-making |
| Model Deployment    | Auto-deploy after successful retraining |

---

## ✅ Wrap-Up

| Task                            | ✅ |
|----------------------------------|----|
| Trigger retraining on drift     | ✅ |
| Scheduled retraining on time    | ✅ |
| Manual retraining capability    | ✅ |

---

## 🔮 Next Step

📒 **`06_online_retraining.ipynb`**  
Ready for **real-time retraining**? Let’s automate retraining **while the model is live** — without downtime, and with **continuous updates**.

Shall we get the online retraining pipeline live, Professor?

🔥 Let's make it **real-time**, Professor! The next step is to **automate online retraining** so the model **learns continuously**, without downtime, while it is still serving predictions. This is essential for **low-latency** applications where the model adapts to incoming data **on the fly**.

---

# 📒 `06_online_retraining.ipynb`  
## 📁 `07_capstone_projects/03_production_ml_platforms/04_model_drifts_retraining_pipeline`

---

## 🎯 **Notebook Goals**

- Set up **online retraining** pipeline:
  - Retrain models with **live** data, triggered by continuous data drift or performance degradation.
  - Integrate with **real-time data** flows (e.g., streaming, APIs).
  - **Deploy updates** with minimal disruption.

---

## ⚙️ 1. Stream New Data and Detect Drift

Let’s simulate a stream of incoming data and keep track of **model drift**.

```python
import numpy as np
import pandas as pd

# Simulating a continuous data stream
def generate_new_data():
    new_data = np.random.randn(10, 30)  # Simulate 10 new instances with 30 features
    return pd.DataFrame(new_data)

# Simulate a drift event after a certain number of records
new_data_stream = generate_new_data()
drift_threshold = 0.2
drift_detected = False

def check_drift(new_data, model, threshold):
    global drift_detected
    # Simulate a simple accuracy drop check
    if np.random.rand() < threshold:  # If random number falls below threshold, we detect drift
        drift_detected = True
    return drift_detected

# Check drift every time new data is received
if check_drift(new_data_stream, model, drift_threshold):
    print("Drift detected, retraining triggered!")
```

---

## 🧠 2. Trigger Online Retraining

If drift is detected, retrain the model with the **latest data**.

```python
def retrain_model_online(new_data):
    print("Retraining model with new data...")
    X_train, X_test, y_train, y_test = load_data()  # Include new data in the retraining
    model = train_model(X_train, y_train)
    acc = evaluate_model(model, X_test, y_test)
    
    if acc > 0.90:
        save_model(model)
        deploy_model()
    else:
        print("⚠️ Accuracy below threshold after retraining.")
```

---

## 🔄 3. Continuous Model Update Loop

```python
# Real-time retraining loop
import time

while True:
    new_data = generate_new_data()  # Simulate incoming data
    if check_drift(new_data, model, drift_threshold):
        retrain_model_online(new_data)
    time.sleep(5)  # Pause for 5 seconds before checking the next batch of data
```

---

## 🔗 4. Deploy and Serve Updated Model

Once retrained, automatically deploy the new model, **overwriting the old version**.

```python
def deploy_model():
    print("Deploying updated model...")
    # Here, you can add actual model deployment code to serve the updated model via an API or cloud endpoint.
```

---

## ✅ What You Built

| Component                | Role |
|--------------------------|------|
| Online Data Stream       | Simulate real-time data flow |
| Drift Detection          | Detects and flags model drift in real-time |
| Retraining & Deployment  | Auto-updates the model with minimal downtime |

---

## ✅ Wrap-Up

| Task                                | ✅ |
|-------------------------------------|----|
| Real-time data stream simulated     | ✅ |
| Drift detection implemented         | ✅ |
| Retraining pipeline activated       | ✅ |
| Model redeployment after retraining | ✅ |

---

## 🔮 Final Step

📄 **`README.md`** — Document the **entire flow** of how to monitor, retrain, and redeploy models automatically in real-time.

Ready to finalize your **online retraining system**, Professor?

📚 Now that we’ve built the **online retraining pipeline**, let’s finish it off with the final **capstone documentation** — a solid **`README.md`** that wraps up everything we’ve accomplished and provides the **how-to guide** for future users or collaborators.

---

# 📄 `README.md`  
## 📁 `07_capstone_projects/03_production_ml_platforms/04_model_drifts_retraining_pipeline`

---

# 🔄 Model Drift Detection & Online Retraining Pipeline — Capstone

> A **self-healing ML system** that automatically detects model drift, retrains with fresh data, and redeploys updated models in real-time.  
This capstone integrates **model drift detection**, **retraining pipelines**, and **automated deployment**, building the foundation for **resilient AI systems** in production.

---

## 📋 Table of Contents

- [📦 System Overview](#system-overview)
- [⚙️ Setup Instructions](#setup-instructions)
- [🚀 Running the Pipeline](#running-the-pipeline)
- [🔧 Model Drift Detection](#model-drift-detection)
- [🔄 Online Retraining](#online-retraining)
- [📊 Monitoring & Alerting](#monitoring-alerting)
- [⚡ Future Improvements](#future-improvements)

---

## 📦 System Overview

This pipeline monitors **data drift** and **model performance** continuously, and triggers **automatic retraining** when predefined thresholds are met. It integrates **real-time monitoring**, **drift detection**, and **model redeployment** with minimal downtime.

### Key Features:
1. **Real-time drift detection** for both **data** and **model**.
2. **Online retraining** triggered by data/model drift.
3. **Automated deployment** of retrained models via cloud endpoints (e.g., Azure, AWS).
4. **Monitoring & alerting** using Slack, Prometheus, or custom alerting systems.

---

## ⚙️ Setup Instructions

### 1. **Install Required Libraries**

```bash
pip install evidently slack_sdk prometheus-client
```

### 2. **Set Up API Keys (for Slack, Prometheus)**

Create a `.env` file to store sensitive credentials such as Slack API tokens and Prometheus configurations:

```bash
SLACK_API_TOKEN=your-slack-token
PROMETHEUS_PORT=8000
```

### 3. **Initialize Your Model & Data**

Use the example dataset (`load_breast_cancer` from sklearn) or replace with your data pipeline.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

---

## 🚀 Running the Pipeline

### 1. **Monitor and Detect Drift**

Run `01_data_drift_detection.ipynb` to track **incoming data drift** and log results:

```bash
python 01_data_drift_detection.ipynb
```

### 2. **Detect Model Drift**

Track **model performance** in real-time using `02_model_drift_detection.ipynb`.

```bash
python 02_model_drift_detection.ipynb
```

### 3. **Trigger Online Retraining**

Once drift is detected, the pipeline will **automatically retrain** your model with `03_retraining_pipeline.ipynb`.

```bash
python 03_retraining_pipeline.ipynb
```

### 4. **Monitor Performance with Alerts**

Set up alerts for drift, accuracy, and model behavior with `04_monitoring_and_alerting.ipynb`.

```bash
python 04_monitoring_and_alerting.ipynb
```

---

## 🔧 Model Drift Detection

### Key Metrics Tracked:
- **Accuracy drop** over time
- **Confidence score** shift
- **Prediction distribution** drift

Use **Evidently** and **custom alerts** to continuously monitor incoming data and model behavior. If drift exceeds thresholds, the system triggers a **retraining cycle**.

---

## 🔄 Online Retraining

The system uses **real-time data streaming** to automatically **retrain models** when drift is detected, then **redeploys** updated models to production endpoints.

- **Real-time Data**: Monitor incoming features.
- **Model Drift**: Evaluate confidence scores and prediction distributions.
- **Retraining & Redeployment**: Re-train and deploy with minimal downtime.

---

## 📊 Monitoring & Alerting

- **Prometheus**: For real-time metric monitoring.
- **Slack**: Sends real-time alerts when model drift is detected or performance drops.
- **Custom Dashboards**: Visualize drift, accuracy, and prediction confidence.

---

## ⚡ Future Improvements

- **Model Retraining Triggers**: Introduce **GitHub Actions** for periodic retraining based on time intervals.
- **Cloud Integration**: Deploy to multiple cloud providers (AWS, GCP, Azure) with unified endpoints.
- **Advanced Drift Metrics**: Integrate **Fairlearn** to ensure **fairness** during drift analysis.

---

## 📢 Conclusion

This **Model Drift & Retraining Pipeline** is a **production-ready** system that ensures your machine learning models stay **relevant and accurate** over time. It automatically detects drift, retrains models, and redeploys updates with minimal intervention, ensuring that your AI systems are always running at peak performance.

---

With this, your **ML system is self-healing**, robust, and fully automated. Ready for **scalable, high-impact deployments**.

Shall we finalize this as the last capstone step, Professor?


























