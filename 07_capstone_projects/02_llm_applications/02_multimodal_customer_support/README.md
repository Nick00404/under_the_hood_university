# 02 Multimodal Customer Support

- [deployment with nvidia triton](./deployment_with_nvidia_triton.ipynb)
- [image text retrieval](./image_text_retrieval.ipynb)
- [latency optimization](./latency_optimization.ipynb)
- [llava visual qa finetuning](./llava_visual_qa_finetuning.ipynb)

---

### 🖼️ **01. Image-Text Retrieval for Product Queries**

#### 📌 **Subtopics Covered:**
- **Multimodal embeddings**: CLIP, BLIP for visual-text matching  
- **Use-case**: Retrieve matching product images from a query like "red sneakers with white soles"  
- Fine-tuning on domain-specific product catalogs  
- Evaluation: Recall@K, median rank, precision curves  

---

### 👁️‍🗨️ **02. LLaVA Visual QA Finetuning**

#### 📌 **Subtopics Covered:**
- Intro to **LLaVA**: Vision-Language model for visual Q&A  
- Dataset curation: Customer-uploaded screenshots + issue descriptions  
- Finetuning for customer support FAQs (e.g., damaged product images)  
- Inference: Visual Q&A chatbot with contextual grounding  

---

### 🚀 **03. Deployment with NVIDIA Triton Inference Server**

#### 📌 **Subtopics Covered:**
- Deploying image-text + visual QA models on Triton  
- Concurrent model serving (CLIP + LLaVA + reranker)  
- Batching, model versioning, and shared memory optimizations  
- GPU utilization, memory pinning, and perf analysis  

---

### ⏱ **04. Latency Optimization Techniques**

#### 📌 **Subtopics Covered:**
- Profiling end-to-end query time (API to response)  
- Quantization + ONNX conversion for CLIP/LLaVA  
- Async queuing, multithreaded preprocessing  
- Batch size vs latency trade-offs for production SLAs  

---

### 📊 **05. A/B Testing & Results Report** (`a_b_testing_results.md`)

#### 📌 **Contents Covered:**
- Experimental design for chatbot vs human support fallback  
- Metrics: First Response Time (FRT), CSAT score, resolution rate  
- Summary of statistical results with visuals (charts, tables)  
- Key learning: When to invoke human-in-the-loop, cost-to-benefit  

---
