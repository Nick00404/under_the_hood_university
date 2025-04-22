
## ✅ **03_model_serving**

This module focuses on translating trained models into production-grade APIs or services. It covers serving frameworks, scaling nuances, and serving modalities for deep learning models — balancing latency, concurrency, and cost.

---

### **1. API-Driven Model Serving with FastAPI**

**1.1 Core Principles**  
- FastAPI lifecycle (`startup`, `shutdown`) for model loading  
- Request parsing with `Pydantic`, data validation  

**1.2 Endpoint Design & Versioning**  
- `/predict`, `/health`, `/v1/model-name` patterns  
- Schema evolution without breaking consumers  

**1.3 Concurrency & Performance**  
- Async IO vs sync execution  
- Uvicorn workers and process/thread tuning  

*Lab: Serve a scikit-learn or XGBoost model via FastAPI with request validation and async inference.*

---

### **2. Flask Serving & Scaling Limitations**

**2.1 Minimal App Patterns**  
- Flask WSGI structure for simple model serving  
- JSON serialization and response strategies  

**2.2 Production Pitfalls**  
- GIL impact under multithreading  
- Handling long inference times and timeouts  

**2.3 Mitigations and Extensions**  
- Using Gunicorn with multiple workers  
- Offloading heavy computation via task queues (Celery, Redis)  

*Lab: Deploy a Flask app for inference, stress-test it under concurrent requests, and diagnose bottlenecks.*

---

### **3. TorchServe for PyTorch Models**

**3.1 Model Archive (MAR) Packaging**  
- Creating a `.mar` file with handler and model state  
- `torch-model-archiver` CLI usage and folder structure  

**3.2 Custom Inference Handlers**  
- Preprocess/postprocess logic hooks  
- Batch request handling in custom `handle()` methods  

**3.3 Lifecycle Management & Metrics**  
- Configuring `config.properties` for scalable inference  
- Model registration, scale-out, and Prometheus metrics  

*Lab: Serve a fine-tuned transformer or CNN using TorchServe with a custom inference handler.*

---

### **4. TensorFlow Serving via Docker**

**4.1 SavedModel Format & Signatures**  
- Exporting TensorFlow models with `tf.saved_model.save()`  
- Understanding `serving_default` signature  

**4.2 REST & gRPC Inference**  
- Predict endpoint usage, protobuf formats  
- Latency comparison: REST vs gRPC  

**4.3 Dockerized Deployment**  
- Running TF Serving container with mounted model volume  
- Dynamic model loading with config file and polling  

*Lab: Package and deploy a TensorFlow model using Dockerized TF Serving with RESTful inference.*

---

### **5. Batch vs. Real-Time Serving Patterns**

**5.1 Architectural Trade-offs**  
- Latency-sensitive vs. throughput-oriented applications  
- Synchronous vs. asynchronous inference strategies  

**5.2 Batch Inference Pipelines**  
- Triggering inference via job schedulers (Airflow, Argo)  
- Writing batch outputs to stores (S3, BigQuery, Parquet)  

**5.3 Multi-Model Serving Patterns**  
- Model routing strategies (A/B, version headers)  
- Dynamic model loading for multi-tenant use cases  

*Lab: Build both real-time (FastAPI) and batch (Airflow-triggered) inference pipelines on the same model.*

---

### ✳️ **Pedagogical Goals Across the Module**

- **Separation of Concerns**: Clearly distinguish between model logic, API logic, and deployment mechanics.  
- **Scalability Under Load**: Highlight concurrency challenges and containerized scaling solutions.  
- **Serving Mode Fitness**: Guide students to match serving strategies with business constraints (SLA, user load, infra availability).

---
