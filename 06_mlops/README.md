# Folder structure 
         
├───01_model_development
│       01_feature_store_feast_vs_custom.ipynb
│       02_data_versioning_with_dvc.ipynb
│       03_experiment_tracking_with_mlflow.ipynb
│       04_experiment_tracking_with_wandb.ipynb
│       05_model_versioning_strategies.ipynb
│       06_reproducibility_with_conda_docker.ipynb
│       README.md
│
├───02_model_training_pipeline
│       01_pipeline_with_scikit_learn_pipeline.ipynb
│       02_pipeline_with_kubeflow.ipynb
│       03_training_on_gcp_vertex_ai.ipynb
│       04_training_on_sagemaker.ipynb
│       05_training_on_local_cluster_with_ray.ipynb
│       README.md
│
├───03_model_serving
│       01_fastapi_model_serving_basics.ipynb
│       02_flask_serving_and_scaling_issues.ipynb
│       03_model_serving_with_torchserve.ipynb
│       04_tensorflow_serving_with_docker.ipynb
│       05_batch_vs_realtime_serving_patterns.ipynb
│       README.md
│
├───04_deployment_strategies
│       01_building_docker_images_for_models.ipynb
│       02_kubernetes_deployment_helm_kustomize.ipynb
│       03_model_deployment_with_seldon_kserve.ipynb
│       04_deploy_to_cloudrun_lambda.ipynb
│       05_blue_green_canary_shadow_deployments.ipynb
│       README.md
│
├───05_model_monitoring
│       01_data_drift_with_evidently.ipynb
│       02_concept_drift_with_river.ipynb
│       03_monitoring_with_prometheus_grafana.ipynb
│       04_model_performance_tracking_dashboards.ipynb
│       05_alerting_and_slack_integration.ipynb
│       README.md
│
└───06_model_optimization_and_costs
        01_model_quantization_with_onnx_tflite.ipynb
        02_pruning_and_sparsity_strategies.ipynb
        03_tensor_rt_for_fast_inference.ipynb
        04_serverless_inference_sagemaker_endpoint.ipynb
        05_gpu_vs_cpu_cost_tradeoffs.ipynb
        06_batching_and_request_optimization.ipynb
        README.md


# INDEX

---

## ✅ **01_model_development**

This module establishes the foundational reproducibility, traceability, and governance practices for production-grade machine learning workflows. It ties together data lineage, experimentation, model lifecycle tracking, and environment consistency.

---

### **1. Feature Store Design & Management**

**1.1 Architectural Patterns**  
- Centralized vs. decentralized feature store architectures  
- Online + offline store bifurcation (e.g., Redis + Parquet)  
- Feature freshness and point-in-time correctness  

**1.2 Feast vs. Custom Pipelines**  
- Feast registry, provider setup, materialization flow  
- Custom store implementations using DB/warehouse + scheduler  
- Tradeoffs: abstraction vs. flexibility, scaling, governance  

**1.3 Consistency Guarantees**  
- Online/offline training-serving skew mitigation  
- Feature versioning strategies  
- TTLs, backfilling, and late-arriving data handling  

*Lab: Build a time-aware feature store using Feast + Redis/PostgreSQL.*

---

### **2. Data Versioning & Lineage**

**2.1 Reproducibility Principles**  
- Deterministic pipelines & lineage tracing  
- Dataset immutability and fingerprinting  

**2.2 Versioning with DVC**  
- Remote backend setup (S3, GCS, Azure)  
- DVC pipelines (`dvc.yaml`, `params.yaml`) for DAG definition  
- `dvc diff`, checksum validation, and CI integration  

**2.3 Lineage Integration**  
- Linking DVC lineage to ML metadata (MLflow, W&B)  
- Hash consistency across pipeline steps  

*Lab: Reproduce an ML experiment from versioned data + model inputs.*

---

### **3. Experiment Tracking & Comparability**

**3.1 Experiment Logging Foundations**  
- Unified model metadata: hyperparams, code, data, env  
- Best practices for consistent metric tracking  

**3.2 MLflow Implementation**  
- Local tracking server, UI usage, artifact logging  
- Model Registry and stage transitions (Staging → Prod)  
- Auto-logging for scikit-learn, TensorFlow, etc.  

**3.3 Weights & Biases (W&B)**  
- Advanced dashboarding, system metrics, alerting  
- Experiment grouping, collaboration workflows  
- Artifact versioning for models and datasets  

*Lab: Track and compare model runs across MLflow and W&B for the same experiment.*

---

### **4. Model Versioning Strategies**

**4.1 Lifecycle Mapping**  
- Checkpointing during training vs. full pipeline versioning  
- Snapshotting preprocessing + model bundle together  

**4.2 Manual vs. Automated Versioning**  
- Git tag–based snapshots  
- Automated CI/CD-driven version bumps  

**4.3 Integration with Registries**  
- Model cards and metadata  
- Linking to CI workflows (e.g., test → promote → deploy triggers)  

*Lab: Register two versions of a model and automate promotion via GitHub Actions.*

---

### **5. Environment Reproducibility**

**5.1 Dependency Management**  
- Conda environments (`environment.yml`, lockfiles)  
- Pip-tools and `pyproject.toml` workflows  
- Environment pinning for hardware-specific builds  

**5.2 Dockerization for ML**  
- Image layering, Conda + pip combo in Docker  
- Deterministic builds using hashable Dockerfiles  
- Caching strategies and multi-stage Docker builds  

**5.3 Reproducibility Hashing**  
- Validating run environments using hash digests  
- Reproducing model artifacts from recorded config  

*Lab: Build and run the same experiment in two different environments (local + Docker) with identical results.*

---

### **6. Integration & Governance Foundations**  

**6.1 ML Governance Readiness**  
- Audit trails for data, experiments, and model versions  
- Source-to-prediction lineage for enterprise audits  

**6.2 Cross-Tool Integration**  
- Linking DVC + MLflow + Docker for reproducible, deployable workflows  
- Managing experiment metadata in Git-friendly formats  

**6.3 Pre-production Checklist**  
- Environment parity checklist (dev, staging, prod)  
- Model release documentation with data + env refs  

*Lab: Generate a reproducibility report (data, code, env, metrics) for regulatory compliance.*

---

### ✳️ **Key Principles**  

- **Environment-First Thinking**: Every model output is a function of data, code, environment.  
- **Governance from Day 1**: Build with auditability, not as an afterthought.  
- **Tool-Agnostic Patterns**: Core ideas apply across MLflow, W&B, DVC, Feast, etc.  

---

## ✅ **02_model_training_pipeline**

This module focuses on building scalable, modular, and reproducible training pipelines using classical frameworks, workflow orchestrators, and cloud-native infrastructure. It emphasizes pipeline abstraction, distributed execution, and cloud-specific training constructs.

---

### **1. Classical ML Pipelines (Scikit-Learn)**

**1.1 Pipeline Composition**  
- `Pipeline` and `ColumnTransformer` abstractions  
- Chaining preprocessing with estimators  
- Ensuring transform integrity between training and inference  

**1.2 Custom Components**  
- Building custom `TransformerMixin` classes  
- Handling temporal features, categorical encodings  

**1.3 Persistence and Reuse**  
- Persisting pipelines with `joblib`, `pickle`, or `skops`  
- Reusing preprocessing logic in model serving  

*Lab: Build a complete scikit-learn pipeline with train-test consistency and deploy-ready persistence.*

---

### **2. Pipeline Authoring with Kubeflow**

**2.1 Componentization Fundamentals**  
- Defining pipeline components with input/output typing  
- Containerized component authoring (`@component`, YAML specs)  

**2.2 Orchestrating with Kubeflow DSL**  
- Creating DAGs with dependencies  
- Parameters, loops, conditional branching  

**2.3 Pipeline Execution & Tracking**  
- Pipeline runs, metadata visualization  
- Output artifact tracking and lineage via ML Metadata  

*Lab: Convert a local training workflow into a fully modular Kubeflow pipeline.*

---

### **3. Training on GCP Vertex AI**

**3.1 Managed Training Jobs**  
- Prebuilt container usage and custom training jobs  
- Defining custom training specs (entry point, region, resource allocation)  

**3.2 Artifact and Model Registry Integration**  
- Vertex AI Experiments for run tracking  
- Storing trained models in the Vertex Model Registry  

**3.3 MLOps Integration Points**  
- CI/CD deployment of training code via Cloud Build  
- Scheduling retraining workflows with Cloud Functions or Workflows  

*Lab: Train and register a model using a custom container on Vertex AI.*

---

### **4. Training on AWS SageMaker**

**4.1 Training Job Types**  
- Built-in algorithms, script mode, and custom containers  
- Spot instance training with checkpointing  

**4.2 Estimators and Training APIs**  
- Using SageMaker Python SDK for orchestration  
- Fine-grained control of training input/output paths  

**4.3 Model Registry and Endpoint Ready Artifacts**  
- SageMaker Model Registry lifecycle  
- Automatic packaging for deployment with real-time endpoints  

*Lab: Launch a managed training job on SageMaker and register it for deployment.*

---

### **5. Distributed Training with Ray**

**5.1 Ray Core Concepts**  
- Ray actors, tasks, and object store  
- Cluster setup for local and distributed use  

**5.2 Ray Train API**  
- Distributed model training with `TorchTrainer`, `XGBoostTrainer`  
- Resource-aware training configuration and scheduling  

**5.3 Scaling Workflows with Ray Tune**  
- Hyperparameter tuning with distributed grid/random/bayesian search  
- Early stopping, metric tracking, and checkpointing  

*Lab: Train a model with parallel HPO using Ray Tune on a multi-node Ray cluster.*

---

### ✳️ **Pedagogical Threads Across All Sections**  

- **Training Pipeline as a First-Class Citizen**: No monoliths — modularity and reproducibility are non-negotiable.  
- **Infrastructure-Agnostic Patterns**: Whether local or cloud-native, every pipeline must be portable and composable.  
- **System Thinking**: Treat model training as part of a broader MLOps DAG — not a standalone script.

---

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

## ✅ **04_deployment_strategies**

This module covers strategies for deploying machine learning models into scalable, fault-tolerant, and secure environments. It includes containerization, orchestration with Kubernetes, deployment on serverless infrastructures, and advanced deployment strategies such as blue/green and canary releases.

---

### **1. Docker Image Creation for ML Models**

**1.1 Containerization Principles**  
- Writing efficient `Dockerfile` for ML models  
- Multi-stage builds for lean, secure images  

**1.2 Managing Dependencies**  
- Managing Python packages with `requirements.txt`, `Pipenv`, `Conda`  
- Handling OS-level dependencies (e.g., `libtensorflow`, `libcuda`)  

**1.3 Versioning and Model Artifacting**  
- Embedding model weights and configurations into images  
- Image tagging strategies for version control  

*Lab: Build and optimize a Docker image for serving a TensorFlow model with dependencies.*

---

### **2. Kubernetes Deployment with Helm & Kustomize**

**2.1 Kubernetes Core Concepts**  
- Pods, Deployments, and ReplicaSets  
- Horizontal Pod Autoscaling (HPA) for efficient resource allocation  

**2.2 Helm Charts for ML Models**  
- Using `helm` to package, configure, and deploy ML models  
- Customizing charts for specific model environments (e.g., TensorFlow, PyTorch)  

**2.3 Kustomize for Dynamic Deployment Configuration**  
- Layering Kubernetes YAMLs with Kustomize for different environments  
- Parametrizing deployments with overlays  

*Lab: Deploy an ML model to Kubernetes using Helm charts, and customize it with Kustomize.*

---

### **3. Model Deployment with Seldon & KServe**

**3.1 Introduction to Seldon & KServe**  
- Seldon’s ML deployment architecture and integration with Kubernetes  
- KServe’s inference service framework for model deployment  

**3.2 Model Serving with Explainability**  
- Integrating explainers (SHAP, LIME) with Seldon or KServe  
- Real-time model explainability through REST/gRPC endpoints  

**3.3 Multi-Model Deployment**  
- Managing model versions and performing rolling upgrades  
- A/B testing and performance tracking with Seldon/KServe  

*Lab: Deploy a Scikit-learn model with explainability integrated using Seldon/KServe.*

---

### **4. Serverless Deployment to Cloud Run & Lambda**

**4.1 Serverless Frameworks Overview**  
- Cloud Run for fully managed containerized applications  
- AWS Lambda for event-driven model inference  

**4.2 Deploying Containers to Cloud Run**  
- Packaging ML models as Docker containers for Cloud Run  
- Scaling policies and auto-scaling inference requests  

**4.3 Deploying Functions to AWS Lambda**  
- Using Lambda’s limited runtime for lightweight inferences  
- Packaging models and dependencies for Lambda (e.g., AWS Lambda Layers)  

*Lab: Deploy a FastAPI-based model on Cloud Run and a lightweight model in AWS Lambda.*

---

### **5. Advanced Deployment Patterns (Blue-Green, Canary, Shadow)**

**5.1 Blue/Green Deployment**  
- Zero-downtime updates with production rollbacks  
- Automating traffic routing between blue/green environments  

**5.2 Canary Releases**  
- Gradual rollout with feedback loops  
- Implementing A/B testing in production environments  

**5.3 Shadow Deployment**  
- Routing live traffic to new models for benchmarking without affecting production  
- Anomaly detection during shadowing for safe model rollouts  

*Lab: Implement a blue/green deployment strategy for model updates on Kubernetes.*

---

### ✳️ **Pedagogical Goals Across the Module**

- **Infrastructure Flexibility**: Teach how to deploy on diverse infrastructures — cloud, serverless, or on-prem.  
- **Robust Release Strategies**: Ensure safe rollouts with real-time feedback and progressive deployment.  
- **Decoupling Inference from Deployment**: Address the independence of deployment pipelines from model training pipelines, ensuring smooth model updates.

---

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

## ✅ **06_model_optimization_and_costs**

This module focuses on optimizing machine learning models for performance and cost efficiency. It explores techniques like quantization, pruning, inference acceleration, serverless deployment, and cost engineering to ensure models run efficiently at scale while minimizing operational costs.

---

### **1. Model Quantization with ONNX & TFLite**

**1.1 Introduction to Model Quantization**  
- The role of quantization in reducing model size and inference time  
- Trade-offs between model accuracy and resource savings (e.g., quantization-aware training)  

**1.2 Quantization with ONNX**  
- Converting models to ONNX format for cross-platform compatibility  
- Applying post-training quantization with ONNX Runtime for optimized inference  

**1.3 TFLite for Mobile & Edge Devices**  
- Converting models to TensorFlow Lite for mobile and IoT devices  
- Performance gains and optimizations in low-power environments  

*Lab: Convert a trained model to ONNX format, apply quantization, and benchmark its performance.*

---

### **2. Pruning and Sparsity Strategies**

**2.1 Pruning Fundamentals**  
- Techniques for pruning models (e.g., weight pruning, neuron pruning)  
- Regularization methods to prevent overfitting during pruning  

**2.2 Sparsity in Neural Networks**  
- Enforcing sparsity through structured pruning (e.g., channels, layers)  
- Using sparse matrix libraries for efficient computation  

**2.3 Impact on Inference Speed and Accuracy**  
- Comparing inference speed before and after pruning  
- Balancing pruning aggressiveness with model accuracy  

*Lab: Apply pruning to a convolutional neural network and evaluate its impact on inference performance.*

---

### **3. TensorRT for Fast Inference**

**3.1 Introduction to TensorRT**  
- NVIDIA TensorRT for high-performance deep learning inference  
- Overview of TensorRT optimization pipelines (e.g., layer fusion, precision calibration)  

**3.2 Using TensorRT with Deep Learning Frameworks**  
- Integrating TensorRT with TensorFlow, PyTorch, and ONNX models  
- Conversion of models into TensorRT optimized formats (e.g., FP16, INT8)  

**3.3 Performance Benchmarks**  
- Benchmarking TensorRT’s speedup over CPU/GPU-based inference  
- Optimizing batch sizes and multi-threading for maximum throughput  

*Lab: Convert a PyTorch model to TensorRT and measure the inference speed on a GPU.*

---

### **4. Serverless Inference with SageMaker Endpoint**

**4.1 Serverless Inference Overview**  
- Introduction to serverless inference and its use cases  
- Benefits of serverless models: automatic scaling, cost efficiency  

**4.2 SageMaker Endpoint Setup**  
- Deploying ML models on AWS SageMaker for serverless inference  
- Auto-scaling endpoints and managing resource usage  

**4.3 Cost Management and Monitoring**  
- Monitoring the cost of serverless endpoints based on usage  
- Strategies for optimizing cold start and response time  

*Lab: Deploy a trained model on SageMaker for serverless inference and track its performance and costs.*

---

### **5. GPU vs. CPU Cost Trade-offs**

**5.1 GPU and CPU Architecture Comparison**  
- Differences in processing power between CPUs and GPUs for ML tasks  
- When to use GPUs vs. CPUs for model inference  

**5.2 Cost Considerations**  
- Cost-per-inference on different platforms (AWS, GCP, on-prem)  
- Estimating cost savings by using GPUs for batch processing vs. CPUs for real-time predictions  

**5.3 Efficient Resource Allocation**  
- Leveraging spot instances, GPU virtualization, and multi-GPU setups for cost optimization  
- Scaling inference workloads to balance cost with performance  

*Lab: Compare the cost and performance of running a model on GPU vs. CPU on cloud platforms.*

---

### **6. Batching and Request Optimization**

**6.1 Batch Processing for Cost Efficiency**  
- Benefits of batching requests for inference (reducing idle time, optimizing throughput)  
- Techniques for dynamic batching in real-time environments  

**6.2 Minimizing Latency in Batch Inference**  
- Balancing batch size with latency requirements  
- Strategies for optimizing batch processing in cloud-native environments (e.g., Cloud Functions, Kubernetes)  

**6.3 Auto-scaling and Request Load Management**  
- Implementing auto-scaling based on traffic patterns and batch processing needs  
- Load balancing between multiple model endpoints for optimized request handling  

*Lab: Implement batching for inference requests in a scalable cloud infrastructure (e.g., Kubernetes or SageMaker).*

---

### ✳️ **Pedagogical Goals Across the Module**

- **Cost Efficiency**: Teach methods to optimize ML models without sacrificing performance, aiming to minimize computational costs.  
- **Performance Gains**: Encourage students to experiment with pruning, quantization, and inference acceleration techniques to achieve faster models.  
- **Real-World Application**: Ensure students understand the trade-offs between cloud infrastructure choices, model optimizations, and performance costs in production environments.

---

