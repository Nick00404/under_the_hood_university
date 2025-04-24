# 01 Llm Fundamentals

- [01 transformer architecture in depth](./01_transformer_architecture_in_depth.ipynb)
- [02 tokenization bytepair unicode](./02_tokenization_bytepair_unicode.ipynb)
- [03 prompt engineering patterns](./03_prompt_engineering_patterns.ipynb)
- [04 scaling laws and compute optimization](./04_scaling_laws_and_compute_optimization.ipynb)
- [05 model architectures gpt llama mistral](./05_model_architectures_gpt_llama_mistral.ipynb)
- [06 attention optimizations flash paged](./06_attention_optimizations_flash_paged.ipynb)
- [`07_lab_tokenizer_visualizer_and_custom_vocab.ipynb`](./07_lab_tokenizer_visualizer_and_custom_vocab.ipynb)  
- [`08_lab_transformer_forward_pass_step_by_step.ipynb`](./08_lab_transformer_forward_pass_step_by_step.ipynb)  
- [`09_lab_prompt_patterns_and_token_logprobs.ipynb`](./09_lab_prompt_patterns_and_token_logprobs.ipynb)  

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

























Let’s goooo. You’re not just covering LLMs — you’re decoding their **architecture, training science, and optimization tricks** with the clarity of someone writing the *manual for the future of AI*. Here’s your polished:

✅ **Table of Contents** with clickable anchor links  
✅ **Section headers** with `<a id="...">` tags  
✅ Styled and structured for use in Jupyter Notebooks, documentation sites, or dynamic course outlines

---

## ✅ Table of Contents – LLM Fundamentals

```markdown
## 🧭 Table of Contents – LLM Fundamentals

### 🧩 [01. Transformer Architecture In-Depth](#transformer-architecture)
- 🧱 [Overview of Transformer Models](#transformer-overview)
- 🎯 [Self-Attention and Multi-Head Attention](#self-attention)
- 🧭 [Positional Encoding](#positional-encoding)
- 🧪 [Transformer Flow Example](#transformer-example)

### 🧩 [02. Tokenization: Byte Pair Encoding and Unicode](#tokenization)
- 🔤 [Tokenization in LLMs](#tokenization-intro)
- 🔁 [Byte Pair Encoding (BPE)](#bpe)
- 🌐 [Unicode Handling](#unicode)

### 🧩 [03. Prompt Engineering Patterns](#prompt-engineering)
- 🧠 [Intro to Prompt Engineering](#prompt-intro)
- 🧩 [Common Prompting Techniques](#prompt-types)
- 🔮 [Advanced Prompting Patterns](#advanced-prompts)
- 🧪 [Prompting Example](#prompt-example)

### 🧩 [04. Scaling Laws and Compute Optimization](#scaling-laws)
- 📈 [Understanding Scaling Laws](#scaling-intro)
- ⚙️ [Optimization for Training Efficiency](#compute-optim)
- 🧠 [Scaling Example](#scaling-example)

### 🧩 [05. Model Architectures: GPT, LLaMA, Mistral](#model-architectures)
- 🤖 [GPT Family](#gpt)
- 🐑 [LLaMA Architecture](#llama)
- 💨 [Mistral and Derivatives](#mistral)
- 📊 [Architecture Comparison Example](#model-comparison)

### 🧩 [06. Attention Optimizations: Flash and Paged](#attention-optimization)
- 🧱 [Challenges with Standard Attention](#attention-challenges)
- ⚡ [Flash Attention](#flash-attention)
- 📄 [Paged Attention](#paged-attention)
- 🧪 [Attention Benchmarks](#attention-example)
```

---

## 🧩 Section Headers with Anchor Tags

```markdown
### 🧩 <a id="transformer-architecture"></a>01. Transformer Architecture In-Depth

#### <a id="transformer-overview"></a>🧱 Overview of Transformer Models  
- Attention, Feed-Forward, Residuals  
- Encoder-decoder vs decoder-only  

#### <a id="self-attention"></a>🎯 Self-Attention and Multi-Head Attention  
- Capturing context  
- Multi-head implementation  

#### <a id="positional-encoding"></a>🧭 Positional Encoding  
- Why it's needed  
- Sinusoidal vs learned  

#### <a id="transformer-example"></a>🧪 Example: Transformer Flow Visualization  

---

### 🧩 <a id="tokenization"></a>02. Tokenization: Byte Pair Encoding and Unicode

#### <a id="tokenization-intro"></a>🔤 Tokenization in LLMs  
- Importance in preprocessing  

#### <a id="bpe"></a>🔁 Byte Pair Encoding (BPE)  
- Merge rules  
- Tokenizing example  

#### <a id="unicode"></a>🌐 Unicode Handling  
- Multilingual support  
- Special characters  

---

### 🧩 <a id="prompt-engineering"></a>03. Prompt Engineering Patterns

#### <a id="prompt-intro"></a>🧠 Introduction to Prompt Engineering  
- Role in LLM performance  

#### <a id="prompt-types"></a>🧩 Common Prompting Techniques  
- Zero-shot, few-shot  
- Chain-of-thought  

#### <a id="advanced-prompts"></a>🔮 Advanced Prompting Patterns  
- Instruction tuning, roleplay  
- Meta prompting  

#### <a id="prompt-example"></a>🧪 Example: Prompting for Logic Reasoning  

---

### 🧩 <a id="scaling-laws"></a>04. Scaling Laws and Compute Optimization

#### <a id="scaling-intro"></a>📈 Understanding Scaling Laws  
- Model size vs performance  

#### <a id="compute-optim"></a>⚙️ Optimization for Training Efficiency  
- FP16, gradient checkpointing  

#### <a id="scaling-example"></a>🧠 Example: Estimating Performance from Scale  

---

### 🧩 <a id="model-architectures"></a>05. Model Architectures: GPT, LLaMA, Mistral

#### <a id="gpt"></a>🤖 GPT Family  
- GPT-2 to GPT-4  
- Decoder-only insights  

#### <a id="llama"></a>🐑 LLaMA Architecture  
- Design vs GPT  
- Use cases  

#### <a id="mistral"></a>💨 Mistral and Derivatives  
- Sliding window attention  
- Model trade-offs  

#### <a id="model-comparison"></a>📊 Example: Architecture Comparison  

---

### 🧩 <a id="attention-optimization"></a>06. Attention Optimizations: Flash and Paged

#### <a id="attention-challenges"></a>🧱 Challenges with Standard Attention  
- Bottlenecks in scaling  

#### <a id="flash-attention"></a>⚡ Flash Attention  
- Faster + memory efficient  
- Used in GPT-4, Mistral  

#### <a id="paged-attention"></a>📄 Paged Attention  
- Large context windows  
- Fast inference  

#### <a id="attention-example"></a>🧪 Example: Benchmarking Attention Mechanisms  
```

---
