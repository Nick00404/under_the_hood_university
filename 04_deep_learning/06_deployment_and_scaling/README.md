# 06 Deployment And Scaling

- [01 exporting models onnx torchscript savedmodel](./01_exporting_models_onnx_torchscript_savedmodel.ipynb)
- [02 docker for ml](./02_docker_for_ml.ipynb)
- [02a advanced docker for ml](./02a_advanced_docker_for_ml.ipynb)
- [02b serving with torchserve tensorflow serving](./02b_serving_with_torchserve_tensorflow_serving.ipynb)
- [03 edge deployment tflite raspberry pi](./03_edge_deployment_tflite_raspberry_pi.ipynb)
- [04 distributed training horovod pytorch ddp](./04_distributed_training_horovod_pytorch_ddp.ipynb)
- [05 monitoring models prometheus grafana](./05_monitoring_models_prometheus_grafana.ipynb)
- [06 scaling with kubernetes kubeflow](./06_scaling_with_kubernetes_kubeflow.ipynb)
- [ 07 lab export pytorch to onnx and run.ipynb ](./07_lab_export_pytorch_to_onnx_and_run.ipynb)  
- [ 08 lab dockerize and_test flask model server.ipynb ](./08_lab_dockerize_and_test_flask_model_server.ipynb)  
- [ 09 lab k8s microservice mock deploy.ipynb ](./09_lab_k8s_microservice_mock_deploy.ipynb)  
---

## 📘 **Model Export, Deployment, and Monitoring – Structured Index**

---

### 🧩 **01. Exporting Models: ONNX, TorchScript, and SavedModel**

#### 📌 **Subtopics:**
- **ONNX (Open Neural Network Exchange) Format**
  - What is ONNX and why is it useful for cross-framework compatibility?
  - Exporting models from PyTorch or TensorFlow to ONNX format
  - Example: Converting a PyTorch model to ONNX and running inference in different frameworks
- **TorchScript**
  - What is TorchScript and how does it enable PyTorch models to be serialized and deployed in production?
  - Exporting a PyTorch model to TorchScript and optimizing for inference
  - Example: Exporting a model to TorchScript for efficient deployment
- **SavedModel (TensorFlow)**
  - Overview of TensorFlow's SavedModel format for exporting trained models
  - How to save, load, and serve TensorFlow models in the SavedModel format
  - Example: Exporting a TensorFlow model to SavedModel for deployment

---

### 🧩 **02. Docker for ML Deployment**

#### 📌 **Subtopics:**
- **Introduction to Docker for Machine Learning**
  - What is Docker and how does it facilitate reproducible and portable machine learning environments?
  - Setting up Docker containers for ML development and deployment
  - Example: Creating a Docker container for a machine learning model
- **Advanced Docker for ML**
  - Advanced techniques for creating custom Docker images for different ML frameworks
  - Best practices for managing dependencies, version control, and optimizing Docker images for ML
  - Example: Building a multi-stage Dockerfile for PyTorch-based applications
- **Serving Models with TorchServe and TensorFlow Serving**
  - Overview of TorchServe for serving PyTorch models
  - Overview of TensorFlow Serving for TensorFlow models
  - Example: Deploying a model using TorchServe and TensorFlow Serving

---

### 🧩 **03. Edge Deployment: TFLite and Raspberry Pi**

#### 📌 **Subtopics:**
- **Introduction to Edge Deployment**
  - Challenges of deploying machine learning models on edge devices like Raspberry Pi
  - How to use TensorFlow Lite (TFLite) to optimize models for edge devices
- **TensorFlow Lite (TFLite) for Edge Deployment**
  - Steps to convert a TensorFlow model to TFLite format
  - Example: Deploying a TFLite model on a Raspberry Pi for real-time inference
- **Raspberry Pi Setup for ML**
  - Configuring the Raspberry Pi for machine learning applications
  - Installing dependencies and running TFLite models on Raspberry Pi
  - Example: Running an object detection model on Raspberry Pi using TFLite

---

### 🧩 **04. Distributed Training: Horovod, PyTorch DDP**

#### 📌 **Subtopics:**
- **Horovod for Distributed Training**
  - What is Horovod, and how does it accelerate distributed deep learning training?
  - Integrating Horovod with TensorFlow and PyTorch for multi-GPU training
  - Example: Setting up Horovod for distributed training on multiple GPUs
- **PyTorch Distributed Data Parallel (DDP)**
  - Introduction to DDP in PyTorch for efficient multi-GPU training
  - Best practices for using DDP to parallelize training and speed up model convergence
  - Example: Implementing DDP for distributed training with PyTorch

---

### 🧩 **05. Monitoring Models: Prometheus and Grafana**

#### 📌 **Subtopics:**
- **Introduction to Model Monitoring**
  - Why is monitoring critical for production-level machine learning models?
  - Overview of Prometheus for collecting metrics and Grafana for visualization
- **Prometheus for Model Monitoring**
  - How to integrate Prometheus to monitor ML model performance and resource usage (e.g., GPU utilization, inference times)
  - Example: Setting up Prometheus to collect model metrics
- **Grafana for Visualizing Model Performance**
  - Visualizing the metrics collected by Prometheus using Grafana
  - Example: Creating dashboards in Grafana to monitor model health and performance

---

### 🧩 **06. Scaling with Kubernetes and Kubeflow**

#### 📌 **Subtopics:**
- **Introduction to Kubernetes for Scaling ML**
  - What is Kubernetes, and how does it help manage containerized machine learning workloads?
  - Setting up Kubernetes clusters to deploy machine learning models at scale
  - Example: Deploying a machine learning model on Kubernetes
- **Kubeflow for ML Pipelines**
  - Overview of Kubeflow for building, deploying, and managing machine learning workflows
  - Example: Using Kubeflow to create and deploy a full ML pipeline in a Kubernetes environment
- **Scaling Inference with Kubernetes and Kubeflow**
  - Best practices for scaling inference workloads using Kubernetes and Kubeflow
  - Example: Scaling a model inference service with Kubernetes and Kubeflow

---

### 🧠 **Bonus:**
- **Real-World Deployment Considerations**
  - Deployment challenges: Latency, load balancing, and versioning of models
  - Best practices for managing and updating models in production environments
  - Continuous integration/continuous deployment (CI/CD) pipelines for ML models

---
