# 01 Reproduce Sota Paper

- [distributed training](./distributed_training.ipynb)
- [quantization impact](./quantization_impact.ipynb)
- [reimplementation pytorch](./reimplementation_pytorch.ipynb)

---

### 📖 **01. Paper Analysis & Experimental Planning**

#### 📌 **Subtopics Covered:**
- Dissecting top-tier ML/NLP/CV papers (from NeurIPS, CVPR, ICLR, ACL)
- Understanding architecture diagrams, ablation studies, and benchmarks
- Identifying reproducibility gaps
- Drafting a **reproduction plan** (datasets, metrics, codebase requirements)

---

### ⚙️ **02. Reimplementation in PyTorch**

#### 📌 **Subtopics Covered:**
- Converting paper equations into functional code  
- Writing **modular, well-documented PyTorch code**
- Training from scratch with identical configs
- Logging reproducibility metrics (wandb, TensorBoard, CSV logs)

---

### 🧪 **03. Distributed Training & Scaling Experiments**

#### 📌 **Subtopics Covered:**
- Using **DDP (DistributedDataParallel)** in PyTorch
- Setup on multi-GPU / multi-node (NCCL backend)
- Profiling compute and memory bottlenecks
- Scalability analysis vs baseline models

---

### 🧠 **Bonus: Contribution & Impact**

#### 📂 `contribution_to_open_source.md`
- Documenting code contributions to open-source repos  
- Pull request practices, reproducibility badges, and citations  
- Maintaining compatibility with `torch>=2.0` and community standards  

#### 📦 `quantizati` (fix filename typo?)
- Likely intended for **post-training quantization** of reproduced models for deployment
- Apply `torch.quantization` or ONNX-based optimizations

---

### ✅ Summary

> Reproducing SOTA work shows you **don’t just learn ML — you question, revalidate, and scale it**. This track proves you're **research-aware and implementation-proven**, a rare combo in industry.

---
