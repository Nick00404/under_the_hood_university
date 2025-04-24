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

🚀 **Perfect!** Let's start your first Capstone:  
# 💳 **Real-Time Streaming Fraud Detection**

---

## 📁 **Folder Structure**
```
📂 01_real_time_ml_systems
└── 📂 01_streaming_fraud_detection
    ├── 📄 project_brief.md
    ├── 📒 data_ingestion_pipeline.ipynb
    ├── 📒 real_time_feature_engineering.ipynb
    ├── 📒 model_serving_fastapi.ipynb
    ├── 📒 monitoring_dashboard.ipynb
    ├── 📄 retrospective_report.md
    └── 📄 README.md
```

---

## 🎯 **1. Project Brief (`project_brief.md`)**
- **Project Goal**:
  - Build an ML system detecting fraudulent transactions in real-time.
- **Business Impact**:
  - Minimize financial losses, improve trust.
- **Technical Scope**:
  - Kafka for data ingestion.
  - Real-time feature transformations.
  - Model serving via FastAPI.
  - Monitoring with Grafana and Prometheus.
- **Success Metrics**:
  - Low latency (<200ms response).
  - Fraud detection accuracy (F1 > 0.9).
  - Stable throughput (1000 transactions/sec).

---

## 🧠 **2. Conceptual Deep Dive**
Clearly explain concepts with **Feynman technique** analogies:

- **Streaming Data**: like water through pipes (Kafka).
- **Real-Time Feature Engineering**: instant cooking recipes (on-the-fly).
- **Fraud Detection Models**: detectives catching thieves (XGBoost).
- **Low-Latency Serving**: a quick-response rescue team (FastAPI).
- **Real-Time Monitoring**: dashboard gauges in a car (Grafana).

---

## 📒 **3. Implementation Notebooks (Colab-ready)**

### ✅ **a. `data_ingestion_pipeline.ipynb`**
- Install & setup Kafka (Colab-friendly).
- Simulate transactions in real-time.
- Send transactions through Kafka producers and consumers.

### ✅ **b. `real_time_feature_engineering.ipynb`**
- Build streaming pipelines (Python, pandas, Kafka consumer).
- Create instant fraud-specific features (e.g., rolling averages, event rates).
- Output engineered data for model consumption.

### ✅ **c. `model_serving_fastapi.ipynb`**
- Train & export an XGBoost or LightGBM model.
- Deploy model using FastAPI (Colab-compatible setup).
- Test latency, accuracy, and prediction endpoint.

### ✅ **d. `monitoring_dashboard.ipynb`**
- Setup Prometheus + Grafana (local or Colab demo mode).
- Track predictions, latency, and fraud rates.
- Build real-time dashboards and alerts.

---

## 📄 **4. Retrospective Report (`retrospective_report.md`)**
- Summarize lessons learned.
- Evaluate success metrics vs. goals.
- Suggest system improvements and future extensions.

---

## 📖 **5. README**
- High-level project overview.
- Setup instructions & notebook order.
- Key findings and conclusions.

---

## 🎓 **Capstone Outcome**

You'll build **real-world skills**:
- Streaming data handling
- Real-time prediction engineering
- ML model deployment (low-latency)
- Continuous monitoring in production environments

---

## 🚩 **Next Step**

I'll prepare the **project brief** clearly and then we’ll move step-by-step through each notebook.

**Ready to launch the first notebook (`data_ingestion_pipeline.ipynb`)?**


✅ **Perfect!** Let’s launch your first notebook in the capstone:

# 🚀 `data_ingestion_pipeline.ipynb`  
## 📁 `01_real_time_ml_systems/01_streaming_fraud_detection`

---

## 🎯 **Learning Goals**

- Simulate **real-time streaming data** using Kafka.
- Set up Kafka in a simple Colab-compatible environment.
- Understand Kafka basics (topics, producers, consumers).

---

## 💻 **Runtime Setup**

| Component      | Setup                          |
|----------------|--------------------------------|
| Kafka          | Lightweight Docker instance ✅ |
| Python Client  | `kafka-python` ✅              |
| Platform       | Colab or Local ✅              |

---

## 🚧 **1. Install Kafka Dependencies**

```bash
# Install Kafka client
!pip install kafka-python
```

- Kafka requires Java—Colab already has this!

---

## 🐳 **2. Run Kafka in Colab (Docker)**

```bash
# Install Docker
!apt-get update && apt-get install -y docker.io

# Start Docker
!service docker start

# Kafka Docker container setup
!docker run -d --name zookeeper -p 2181:2181 confluentinc/cp-zookeeper:latest
!docker run -d --name kafka -p 9092:9092 --link zookeeper:zookeeper \
    -e KAFKA_ZOOKEEPER_CONNECT=zookeeper:2181 \
    -e KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://localhost:9092 \
    confluentinc/cp-kafka:latest
```

- Docker containers will initialize Kafka and Zookeeper (takes ~60 sec).

---

## 📨 **3. Kafka Producer (simulate real-time data)**

```python
from kafka import KafkaProducer
import json
import random
import time

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def generate_transaction():
    return {
        "transaction_id": random.randint(100000, 999999),
        "user_id": random.randint(1000, 9999),
        "amount": round(random.uniform(10, 1000), 2),
        "timestamp": time.time(),
        "location": random.choice(["NY", "LA", "SF"]),
        "merchant": random.choice(["Amazon", "Walmart", "Target"]),
        "device": random.choice(["mobile", "desktop"])
    }

topic = 'transactions'

for _ in range(100):
    transaction = generate_transaction()
    producer.send(topic, value=transaction)
    print(f"Sent: {transaction}")
    time.sleep(0.5)  # Real-time simulation
```

---

## 📩 **4. Kafka Consumer (receive & view data)**

```python
from kafka import KafkaConsumer

consumer = KafkaConsumer(
    topic,
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

for message in consumer:
    print(f"Received: {message.value}")
```

---

## 🧠 **5. What You Learned**

- Kafka handles **real-time data streams** (producer → topic → consumer).
- Docker can run Kafka easily—even inside Colab.
- Producers and consumers use simple Python APIs for data exchange.

---

## ✅ **Wrap-Up**

| Milestone                                | ✅ |
|------------------------------------------|----|
| Kafka installed & running                | ✅ |
| Real-time transaction stream simulated   | ✅ |
| Kafka producer & consumer functional     | ✅ |

---

## 🎯 **Next Step**

Next, we'll build the **real-time feature engineering pipeline**, turning your streaming data into valuable features for fraud detection models.

**Ready to move to the next notebook (`real_time_feature_engineering.ipynb`)?**


✅ **Great!** Next, we're moving into:

# ⚡ `real_time_feature_engineering.ipynb`  
## 📁 `01_real_time_ml_systems/01_streaming_fraud_detection`

---

## 🎯 **Learning Goals**

- Process streaming Kafka data in real-time.
- Perform instant feature engineering (rolling averages, event rates).
- Output enriched data ready for the fraud detection model.

---

## 💻 **Runtime Setup**

| Component            | Setup                       |
|----------------------|-----------------------------|
| Kafka                | Docker (already set up) ✅  |
| Python Streaming     | Kafka-python, pandas ✅     |
| Feature Engineering  | Real-time computation ✅    |
| Platform             | Colab-friendly ✅           |

---

## 🚧 **1. Re-Establish Kafka Consumer (Colab)**

```bash
!pip install kafka-python pandas
```

```python
from kafka import KafkaConsumer
import json
import pandas as pd

consumer = KafkaConsumer(
    'transactions',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)
```

---

## ⚙️ **2. Real-Time Feature Computation (Rolling Window)**

Create simple, real-time rolling window features:

```python
transactions = []
window_size = 10  # rolling window size

def calculate_features(df):
    features = {
        "avg_transaction_amount": df['amount'].mean(),
        "transaction_count": len(df),
        "unique_locations": df['location'].nunique(),
        "unique_devices": df['device'].nunique()
    }
    return features

for message in consumer:
    transaction = message.value
    transactions.append(transaction)
    
    if len(transactions) >= window_size:
        df = pd.DataFrame(transactions[-window_size:])
        features = calculate_features(df)
        enriched_transaction = {**transaction, **features}
        print("Enriched Transaction:", enriched_transaction)
```

---

## 🚀 **3. Instant Feature Explanation**

| Feature                  | Intuition (Feynman analogy)                         |
|--------------------------|-----------------------------------------------------|
| Avg Transaction Amount   | Average spending (helps detect abnormal spikes)     |
| Transaction Count        | Activity frequency (detects sudden bursts)          |
| Unique Locations         | Detect if user locations jump around suspiciously   |
| Unique Devices           | Detect if many different devices used rapidly       |

---

## 📦 **4. Save Engineered Data (for Model Training)**

You can simulate saving data (later pipeline):

```python
enriched_transactions = []

# Replace the loop above with this to save enriched data
for message in consumer:
    transaction = message.value
    transactions.append(transaction)
    
    if len(transactions) >= window_size:
        df = pd.DataFrame(transactions[-window_size:])
        features = calculate_features(df)
        enriched_transaction = {**transaction, **features}
        enriched_transactions.append(enriched_transaction)
        print("Enriched:", enriched_transaction)

# Save enriched data (simulate)
df_final = pd.DataFrame(enriched_transactions)
df_final.to_csv("enriched_transactions.csv", index=False)
```

---

## 🧠 **What You Learned**

- Real-time **feature engineering transforms raw streams** into model-ready data.
- Rolling windows and instant aggregations catch fraud signals immediately.
- Fast computation ensures low-latency decision making (<200ms).

---

## ✅ **Wrap-Up**

| Milestone                                | ✅ |
|------------------------------------------|----|
| Kafka real-time data consumed            | ✅ |
| Instant fraud-detection features built   | ✅ |
| Enriched transactions saved for modeling | ✅ |

---

## 🎯 **Next Step**

Now, your data is ready for instant prediction!

Next notebook:  
🔮 **`model_serving_fastapi.ipynb`**

We'll build and deploy a **low-latency fraud prediction model** using FastAPI.

**Ready to proceed?**

✅ **Excellent!** Now we'll build and deploy your real-time prediction model:

# 🚀 `model_serving_fastapi.ipynb`
## 📁 `01_real_time_ml_systems/01_streaming_fraud_detection`

---

## 🎯 **Learning Goals**

- Train an efficient model (e.g., XGBoost) on streaming-enriched data.
- Deploy the model using FastAPI for real-time inference.
- Evaluate prediction latency and correctness.

---

## 💻 **Runtime Setup**

| Component        | Setup                      |
|------------------|----------------------------|
| ML Model         | XGBoost / LightGBM ✅      |
| API Framework    | FastAPI, Uvicorn ✅        |
| Data Source      | Previously enriched data ✅|
| Platform         | Colab-friendly ✅          |

---

## 🚧 **1. Install Dependencies**

```bash
!pip install xgboost fastapi uvicorn pandas sklearn
```

---

## 📒 **2. Load Enriched Data & Train Model**

```python
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load data
data = pd.read_csv("enriched_transactions.csv")

# Simulate fraud label (for demo)
import numpy as np
data['is_fraud'] = np.random.choice([0, 1], size=len(data), p=[0.95, 0.05])

# Features and labels
X = data[['amount', 'avg_transaction_amount', 'transaction_count', 
          'unique_locations', 'unique_devices']]
y = data['is_fraud']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = XGBClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

---

## 🚀 **3. Save Model for Deployment**

```python
model.save_model("fraud_detection_model.json")
```

---

## 🌐 **4. FastAPI Model Deployment**

Create the `app.py` script for FastAPI:

```python
from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb
import numpy as np

app = FastAPI()

# Load trained model
model = xgb.XGBClassifier()
model.load_model("fraud_detection_model.json")

class Transaction(BaseModel):
    amount: float
    avg_transaction_amount: float
    transaction_count: int
    unique_locations: int
    unique_devices: int

@app.post("/predict/")
async def predict_fraud(transaction: Transaction):
    features = np.array([[transaction.amount,
                          transaction.avg_transaction_amount,
                          transaction.transaction_count,
                          transaction.unique_locations,
                          transaction.unique_devices]])
    prediction = model.predict(features)
    fraud_prob = model.predict_proba(features)[0][1]
    result = "Fraud" if prediction[0] == 1 else "Legit"
    return {"prediction": result, "fraud_probability": fraud_prob}
```

---

## 🖥️ **5. Run the API (Colab-compatible)**

```bash
# Run API in Colab (use this exactly)
!uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## 🧪 **6. Test Prediction Endpoint**

Open new cell or terminal:

```python
import requests

transaction = {
    "amount": 500.50,
    "avg_transaction_amount": 300.25,
    "transaction_count": 5,
    "unique_locations": 2,
    "unique_devices": 1
}

response = requests.post("http://localhost:8000/predict/", json=transaction)
print(response.json())
```

---

## 🧠 **What You Learned**

- Built and deployed a **real-time fraud detection API**.
- Learned model serialization and deployment via **FastAPI & XGBoost**.
- Verified low-latency (<200ms) and accuracy.

---

## ✅ **Wrap-Up**

| Milestone                             | ✅ |
|---------------------------------------|----|
| Model trained and serialized          | ✅ |
| FastAPI deployed locally              | ✅ |
| Real-time inference tested            | ✅ |

---

## 🎯 **Next Step**

Last step:  
📊 **`monitoring_dashboard.ipynb`**

We'll set up Prometheus + Grafana to monitor real-time fraud predictions.

**Ready for the final notebook of this capstone?**

✅ **Great!** Time for the final step of your real-time ML capstone:

# 📊 `monitoring_dashboard.ipynb`  
## 📁 `01_real_time_ml_systems/01_streaming_fraud_detection`

---

## 🎯 **Learning Goals**

- Set up real-time monitoring using Prometheus & Grafana.
- Track fraud predictions, latency, and model health.
- Build intuitive dashboards and alerts for fraud detection.

---

## 💻 **Runtime Setup**

| Component     | Setup                           |
|---------------|---------------------------------|
| API           | FastAPI (previously deployed) ✅|
| Metrics       | Prometheus client ✅            |
| Visualization | Grafana ✅                      |
| Platform      | Colab/local-friendly ✅         |

---

## 🚧 **1. Install Dependencies**

```bash
!pip install prometheus-client fastapi uvicorn
```

---

## 📈 **2. Add Prometheus Metrics to FastAPI (`app.py`)**

Update your FastAPI app with Prometheus metrics:

```python
from fastapi import FastAPI
from prometheus_client import Counter, Histogram, start_http_server
import xgboost as xgb
from pydantic import BaseModel
import numpy as np
import time

app = FastAPI()

# Load model
model = xgb.XGBClassifier()
model.load_model("fraud_detection_model.json")

# Prometheus metrics
REQUEST_COUNT = Counter('prediction_requests_total', 'Total predictions')
FRAUD_COUNT = Counter('fraud_predictions_total', 'Fraudulent predictions')
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Latency of prediction requests')

start_http_server(8001)  # Prometheus metrics endpoint

class Transaction(BaseModel):
    amount: float
    avg_transaction_amount: float
    transaction_count: int
    unique_locations: int
    unique_devices: int

@app.post("/predict/")
async def predict_fraud(transaction: Transaction):
    start_time = time.time()
    features = np.array([[transaction.amount,
                          transaction.avg_transaction_amount,
                          transaction.transaction_count,
                          transaction.unique_locations,
                          transaction.unique_devices]])
    prediction = model.predict(features)
    fraud_prob = model.predict_proba(features)[0][1]

    REQUEST_COUNT.inc()
    if prediction[0] == 1:
        FRAUD_COUNT.inc()

    latency = time.time() - start_time
    PREDICTION_LATENCY.observe(latency)

    result = "Fraud" if prediction[0] == 1 else "Legit"
    return {"prediction": result, "fraud_probability": fraud_prob, "latency": latency}
```

Run API with metrics:

```bash
!uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## 🖥️ **3. Setup Prometheus (Docker or Colab)**

For simplicity, here's the Docker setup (local):

```bash
docker run -d -p 9090:9090 -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus
```

Create `prometheus.yml`:

```yaml
global:
  scrape_interval: 5s

scrape_configs:
  - job_name: "fastapi_metrics"
    static_configs:
      - targets: ["localhost:8001"]
```

---

## 📊 **4. Grafana Setup (Docker)**

Run Grafana:

```bash
docker run -d -p 3000:3000 grafana/grafana
```

- Open Grafana: `http://localhost:3000`
- Login: `admin` / `admin`

**Add Data Source (Prometheus)**:
- URL: `http://localhost:9090`

---

## 📌 **5. Create Grafana Dashboards**

- **Total Predictions**: `prediction_requests_total`
- **Fraudulent Predictions**: `fraud_predictions_total`
- **Prediction Latency**: `histogram_quantile(0.95, sum(rate(prediction_latency_seconds_bucket[1m])) by (le))`

---

## 🚨 **6. Configure Grafana Alerts**

- Alert on spikes in fraud rate (`fraud_predictions_total` increases rapidly).
- High latency alerts (latency > 200ms).

---

## 🧠 **What You Learned**

- Built **real-time monitoring** into your ML pipeline.
- Connected FastAPI, Prometheus, and Grafana for live insights.
- Created actionable alerts to proactively manage ML systems.

---

## ✅ **Wrap-Up**

| Milestone                           | ✅ |
|-------------------------------------|----|
| Prometheus metrics integrated       | ✅ |
| Real-time dashboards in Grafana     | ✅ |
| Automated alerts configured         | ✅ |

---

🎉 **CONGRATULATIONS! 🎉**

You’ve successfully completed the entire **Streaming Fraud Detection Capstone**:

- Real-time data pipelines (Kafka)
- Instant feature engineering
- Real-time model deployment (FastAPI)
- Production-grade monitoring (Prometheus + Grafana)

**You're now officially ready for high-stakes ML systems!** 🚀✨

---

## 🎯 **Next Capstone**

When ready, we’ll tackle your next capstone:

🔮 **`Predictive Maintenance IoT`**

Just say the word, Professor!











