# 06 Advanced Topics

- [01 mixture of experts implementation](./01_mixture_of_experts_implementation.ipynb)
- [02 long context processing ring attention](./02_long_context_processing_ring_attention.ipynb)
- [03 multi agent llm systems](./03_multi_agent_llm_systems.ipynb)
- [04 llm os agi prototyping](./04_llm_os_agi_prototyping.ipynb)
- [05 compression sparse pruning](./05_compression_sparse_pruning.ipynb)
- [06 energy efficient llms](./06_energy_efficient_llms.ipynb)

---

## 📘 **Advanced Topics – Structured Index**

---

### 🧩 **01. Mixture of Experts (MoE) Implementation**

#### 📌 **Subtopics:**
- **What is a Mixture of Experts?**
  - Sparse activation of model submodules for scalability
- **Architecture Design**
  - Router layers, gating mechanisms, and expert parallelism
- **Popular MoE Frameworks**
  - DeepSpeed MoE, GShard, Switch Transformer
- **Example:** Implementing a 2-expert MoE model using PyTorch and DeepSpeed

---

### 🧩 **02. Long Context Processing: Ring Attention and Beyond**

#### 📌 **Subtopics:**
- **Challenges with Long Contexts**
  - Quadratic memory and compute bottlenecks
- **Ring Attention and Related Techniques**
  - Sliding window, ring attention, and dilated attention mechanisms
- **Segmented Context and Chunk Memory Models**
  - Recurrent memory and retrieval-augmented mechanisms
- **Example:** Comparing vanilla attention vs ring attention on 32k-token input

---

### 🧩 **03. Multi-Agent LLM Systems**

#### 📌 **Subtopics:**
- **Agentic LLM Architecture**
  - Agent roles, communication, coordination mechanisms
- **Planning, Tool Use, and Memory**
  - Agents with tools, shared memory, and long-term goals
- **Frameworks and Runtimes**
  - CrewAI, AutoGPT, LangGraph, and custom agents
- **Example:** Multi-agent system for research + code generation tasks

---

### 🧩 **04. LLM OS and AGI Prototyping**

#### 📌 **Subtopics:**
- **What Is an LLM OS?**
  - Abstracting operating system-like behavior with LLMs
- **Autonomy and Task Decomposition**
  - Scheduling, inter-process communication, reasoning loops
- **AGI Prototype Architectures**
  - Architecting LLMs with perception, memory, planning
- **Example:** Prototyping an LLM “desktop agent” that operates local tools

---

### 🧩 **05. Compression: Sparse Models and Pruning**

#### 📌 **Subtopics:**
- **Need for Model Compression**
  - Memory efficiency, latency reduction, deployment at edge
- **Sparsity and Pruning Techniques**
  - Unstructured, structured, and dynamic sparsity
- **Knowledge Distillation**
  - Transferring knowledge from large to small models
- **Example:** Pruning a BERT model to 50% sparsity without major loss in accuracy

---

### 🧩 **06. Energy-Efficient LLMs**

#### 📌 **Subtopics:**
- **Environmental Impact of LLMs**
  - Training and inference energy consumption metrics
- **Strategies for Efficiency**
  - Quantization, efficient attention, hardware-aware architecture
- **Monitoring and Reporting**
  - Carbon tracking tools, FLOPs tracking, and sustainability dashboards
- **Example:** Estimating and reducing energy cost of inference with quantized models

---
