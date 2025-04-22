
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
