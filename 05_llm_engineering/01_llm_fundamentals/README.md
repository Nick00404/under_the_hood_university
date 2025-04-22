# 01 Llm Fundamentals

- [01 transformer architecture in depth](./01_transformer_architecture_in_depth.ipynb)
- [02 tokenization bytepair unicode](./02_tokenization_bytepair_unicode.ipynb)
- [03 prompt engineering patterns](./03_prompt_engineering_patterns.ipynb)
- [04 scaling laws and compute optimization](./04_scaling_laws_and_compute_optimization.ipynb)
- [05 model architectures gpt llama mistral](./05_model_architectures_gpt_llama_mistral.ipynb)
- [06 attention optimizations flash paged](./06_attention_optimizations_flash_paged.ipynb)

---

## 📘 **LLM Fundamentals – Structured Index**

---

### 🧩 **01. Transformer Architecture In-Depth**

#### 📌 **Subtopics:**
- **Overview of Transformer Models**
  - Key components: Attention, Feed-Forward Networks, Residual Connections
  - Encoder-decoder vs decoder-only architectures
- **Self-Attention and Multi-Head Attention**
  - How self-attention works and its role in capturing contextual information
  - Benefits of multi-head attention and its implementation
- **Positional Encoding**
  - Why positional encoding is needed in Transformers
  - Common approaches: sinusoidal vs learned embeddings
- **Example:** Visual walkthrough of input flow through a Transformer block

---

### 🧩 **02. Tokenization: Byte Pair Encoding and Unicode**

#### 📌 **Subtopics:**
- **Tokenization in LLMs**
  - Why tokenization is crucial for text processing in large models
- **Byte Pair Encoding (BPE)**
  - How BPE works and its advantages in language modeling
  - Example: Tokenizing text using BPE
- **Unicode Handling**
  - Unicode representation and challenges in tokenization
  - How modern tokenizers handle multilingual and special character inputs

---

### 🧩 **03. Prompt Engineering Patterns**

#### 📌 **Subtopics:**
- **Introduction to Prompt Engineering**
  - Why prompt design matters in LLM performance
- **Common Prompting Techniques**
  - Zero-shot, one-shot, few-shot prompting
  - Chain-of-thought prompting and its impact on reasoning tasks
- **Advanced Prompting Patterns**
  - Instruction tuning, roleplay prompts, meta prompts
- **Example:** Using different prompting strategies to solve a logic puzzle

---

### 🧩 **04. Scaling Laws and Compute Optimization**

#### 📌 **Subtopics:**
- **Understanding Scaling Laws**
  - The relationship between model size, dataset size, and compute
  - Key findings from OpenAI, DeepMind, and others
- **Optimization for Training Efficiency**
  - Mixed precision training, gradient checkpointing, memory-efficient techniques
- **Example:** Applying scaling laws to estimate performance of larger models

---

### 🧩 **05. Model Architectures: GPT, LLaMA, Mistral**

#### 📌 **Subtopics:**
- **GPT Family Overview**
  - Architecture highlights from GPT-2 to GPT-4
- **LLaMA Architecture**
  - Design goals and differences from GPT
  - Performance and use cases
- **Mistral and Derivatives**
  - Innovative features of Mistral (e.g., sliding window attention)
  - How it compares to LLaMA and GPT
- **Example:** Side-by-side comparison of architecture diagrams

---

### 🧩 **06. Attention Optimizations: Flash and Paged**

#### 📌 **Subtopics:**
- **Challenges with Standard Attention**
  - Memory and compute bottlenecks in large-scale models
- **Flash Attention**
  - What it is, how it reduces memory usage and boosts speed
  - Use in modern LLMs like GPT-4 and Mistral
- **Paged Attention**
  - Efficient attention for long context windows
  - How it enables faster inference with larger contexts
- **Example:** Benchmark comparison of attention mechanisms

---
