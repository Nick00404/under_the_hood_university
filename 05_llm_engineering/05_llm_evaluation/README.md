# 05 Llm Evaluation

- [01 automated metrics bleu rouge bertscore](./01_automated_metrics_bleu_rouge_bertscore.ipynb)
- [02 human eval setups and crowdsourcing](./02_human_eval_setups_and_crowdsourcing.ipynb)
- [03 toxicity bias detection](./03_toxicity_bias_detection.ipynb)
- [04 red teaming adversarial testing](./04_red_teaming_adversarial_testing.ipynb)
- [05 latency throughput benchmarking](./05_latency_throughput_benchmarking.ipynb)
- [06 model cards and audit trails](./06_model_cards_and_audit_trails.ipynb)

---

## 📘 **LLM Evaluation – Structured Index**

---

### 🧩 **01. Automated Metrics: BLEU, ROUGE, BERTScore**

#### 📌 **Subtopics:**
- **Why Use Automated Metrics?**
  - Quick, scalable, and reproducible evaluation for generative tasks
- **BLEU and ROUGE**
  - Classical n-gram based metrics for translation and summarization
- **BERTScore and Embedding-Based Evaluation**
  - Semantic similarity using contextual embeddings
- **Limitations and Pitfalls**
  - Why these metrics may not align with human judgment
- **Example:** Comparing BLEU vs BERTScore on a summarization dataset

---

### 🧩 **02. Human Eval Setups and Crowdsourcing**

#### 📌 **Subtopics:**
- **Designing Human Evaluation Studies**
  - Quality, helpfulness, factuality, and preference comparisons
- **Crowdsourcing Platforms**
  - Using MTurk, Scale, Surge AI, and open-source alternatives
- **Prompt-Based Evaluations**
  - Binary, Likert, and ranking-based formats
- **Example:** Human eval protocol for chatbot helpfulness rating

---

### 🧩 **03. Toxicity and Bias Detection**

#### 📌 **Subtopics:**
- **Measuring Harmful Outputs**
  - Toxicity, hate speech, and sensitive content
- **Bias in LLMs**
  - Gender, race, political, and geographic biases
- **Detection Tools and Datasets**
  - Perspective API, Detoxify, RealToxicityPrompts
- **Example:** Evaluating a chatbot for toxicity across different prompt types

---

### 🧩 **04. Red Teaming and Adversarial Testing**

#### 📌 **Subtopics:**
- **What is Red Teaming in AI?**
  - Purposefully breaking or probing model behavior
- **Adversarial Prompting Techniques**
  - Jailbreak prompts, prompt injections, hidden queries
- **Structured Red Teaming Workflows**
  - Scenarios, roles, and risk assessments
- **Example:** Red teaming an LLM for safety and alignment under edge cases

---

### 🧩 **05. Latency and Throughput Benchmarking**

#### 📌 **Subtopics:**
- **Why Performance Metrics Matter**
  - Responsiveness and scalability in production
- **Latency Benchmarks**
  - Time-to-first-token, total generation time
- **Throughput Measurement**
  - Tokens per second, concurrent user capacity
- **Example:** Benchmarking vLLM vs TGI on latency and throughput

---

### 🧩 **06. Model Cards and Audit Trails**

#### 📌 **Subtopics:**
- **What Are Model Cards?**
  - Transparency reports for datasets, training, performance, and limitations
- **Audit Trails for LLMs**
  - Tracking data lineage, training logs, deployment history
- **Governance and Compliance**
  - Responsible AI practices and regulatory alignment
- **Example:** Creating a model card using Hugging Face's model card template

---
