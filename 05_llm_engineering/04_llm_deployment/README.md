# 04 Llm Deployment

- [01 serving frameworks vllm tgi](./01_serving_frameworks_vllm_tgi.ipynb)
- [02 quantization ggml awq gptq](./02_quantization_ggml_awq_gptq.ipynb)
- [03 distributed inference tensorrt llm](./03_distributed_inference_tensorrt_llm.ipynb)
- [04 edge deployment ollama mlc](./04_edge_deployment_ollama_mlc.ipynb)
- [05 caching and request batching](./05_caching_and_request_batching.ipynb)
- [06 cost monitoring and autoscaling](./06_cost_monitoring_and_autoscaling.ipynb)
- [`07_lab_vllm_vs_tgi_latency_comparison.ipynb`](./07_lab_vllm_vs_tgi_latency_comparison.ipynb)  
- [`08_lab_quantize_with_gptq_and_awq.ipynb`](./08_lab_quantize_with_gptq_and_awq.ipynb)  
- [`09_lab_batching_and_request_queuing_testbed.ipynb`](./09_lab_batching_and_request_queuing_testbed.ipynb)  

---

## 📘 **LLM Deployment – Structured Index**

---

### 🧩 **01. Serving Frameworks: vLLM, TGI**

#### 📌 **Subtopics:**
- **Overview of LLM Serving Needs**
  - Latency, throughput, scalability considerations
- **vLLM (Virtualized LLM Inference)**
  - Efficient memory management with PagedAttention
  - Running OpenAI-compatible APIs at scale
- **Text Generation Inference (TGI)**
  - Hugging Face's optimized inference engine
  - Features like tensor parallelism, streaming
- **Example:** Deploying a LLaMA model with both vLLM and TGI

---

### 🧩 **02. Quantization: GGML, AWQ, GPTQ**

#### 📌 **Subtopics:**
- **Why Quantization?**
  - Trade-offs between model size, speed, and accuracy
- **GGML and CPU Inference**
  - Lightweight inference on local hardware
- **GPTQ and AWQ**
  - 4-bit quantization for fast GPU inference with minimal loss
- **Example:** Quantizing a model with GPTQ and comparing performance pre/post

---

### 🧩 **03. Distributed Inference: TensorRT-LLM**

#### 📌 **Subtopics:**
- **Inference at Scale**
  - Handling large models across multiple GPUs or nodes
- **TensorRT-LLM Overview**
  - NVIDIA’s high-performance inference stack for LLMs
- **Pipeline and Tensor Parallelism**
  - Techniques for parallelizing model execution
- **Example:** Deploying a 70B model across multiple GPUs with TensorRT-LLM

---

### 🧩 **04. Edge Deployment: Ollama, MLC**

#### 📌 **Subtopics:**
- **Why Edge LLMs?**
  - Offline usage, data privacy, reduced latency
- **Ollama Runtime**
  - Running quantized models locally with GPU/CPU support
- **MLC (Machine Learning Compilation)**
  - Deploying LLMs on mobile and embedded devices
- **Example:** Running a quantized Mistral model on a MacBook with Ollama

---

### 🧩 **05. Caching and Request Batching**

#### 📌 **Subtopics:**
- **Importance of Request Optimization**
  - Reducing compute for repeated or similar queries
- **Caching Mechanisms**
  - Embedding cache, prompt cache, KV cache
- **Request Batching**
  - Grouping requests to maximize GPU utilization
- **Example:** Implementing KV caching and batching in a TGI server

---

### 🧩 **06. Cost Monitoring and Autoscaling**

#### 📌 **Subtopics:**
- **Tracking Inference Costs**
  - GPU hours, API usage, and model-specific metrics
- **Monitoring Tools**
  - Prometheus, Grafana, custom dashboards
- **Autoscaling Strategies**
  - Horizontal pod autoscaling in Kubernetes
  - Scaling based on token throughput or latency
- **Example:** Setting up autoscaling for a vLLM instance in Kubernetes

---
