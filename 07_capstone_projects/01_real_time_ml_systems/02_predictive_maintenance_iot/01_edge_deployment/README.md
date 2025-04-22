## 📘 **Capstone: Edge Deployment – Structured Index**

---

### 🧩 **01. Deploying ML Models on Edge Devices**

#### 📌 **Includes: `edge_device_deployment_guide.md`, `edge_deployment_tflite.ipynb`**

##### **Subtopics:**
- **What Is Edge Deployment?**
  - Use cases: offline inference, privacy, reduced latency
- **Supported Hardware**
  - Raspberry Pi, Android devices, Coral TPU, microcontrollers
- **TensorFlow Lite (TFLite) Overview**
  - Lightweight format for on-device inference
- **End-to-End Deployment Guide**
  - Installing dependencies, deploying models, running test inference
- **Example:** Running a digit recognition or object detection model on a Raspberry Pi using TFLite

---

### 🧩 **02. Model Optimization and Compression Techniques**

#### 📌 **Includes: `model_compression.ipynb`**

##### **Subtopics:**
- **Why Compress Models for Edge?**
  - Reduce memory, power, and inference latency
- **Pruning and Weight Clustering**
  - Techniques to eliminate redundancy while retaining accuracy
- **Knowledge Distillation**
  - Train a smaller model to mimic a larger one
- **Model Size vs Accuracy Tradeoffs**
  - How much compression is “too much”?
- **Example:** Applying pruning and visualizing model size reduction

---

### 🧩 **03. Quantization for Efficient Inference**

#### 📌 **Includes: `quantization_tflite.ipynb`**

##### **Subtopics:**
- **What Is Quantization?**
  - Reducing model precision (e.g., float32 → int8)
- **Types of Quantization in TFLite**
  - Post-training dynamic, full integer, and quantization-aware training
- **Performance Gains and Accuracy Drops**
  - Benchmarking before/after quantization
- **Deployment Compatibility**
  - Ensuring quantized models run on edge hardware
- **Example:** Converting a model to int8 and benchmarking latency improvement on an edge device

---
