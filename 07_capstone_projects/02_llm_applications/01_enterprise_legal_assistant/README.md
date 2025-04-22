# 01 Enterprise Legal Assistant

- [adversarial testing](./adversarial_testing.ipynb)
- [finetuning mistral lora](./finetuning_mistral_lora.ipynb)
- [gdpr compliance checks](./gdpr_compliance_checks.ipynb)
- [legal rag pipeline](./legal_rag_pipeline.ipynb)
- [vllm serving cost analysis](./vllm_serving_cost_analysis.ipynb)

---

### 🔐 **01. Adversarial Testing for Legal LLMs**

#### 📌 **Subtopics Covered:**
- Prompt injection & prompt leaking in legal context  
- Red-teaming LLMs with deceptive inputs  
- Jailbreak attempts on sensitive compliance queries  
- Evaluating response consistency & hallucination detection  

---

### 🧠 **02. Finetuning Mistral with LoRA**

#### 📌 **Subtopics Covered:**
- Why finetune for legal tasks (contracts, GDPR, policies)  
- Intro to **LoRA (Low-Rank Adaptation)** for efficient finetuning  
- Dataset setup: clause tagging, question answering, legal entailment  
- Training + evaluation pipeline using PEFT libraries  

---

### 🛡 **03. GDPR Compliance Checks with LLMs**

#### 📌 **Subtopics Covered:**
- Auto-extraction of GDPR-relevant clauses from policies  
- Mapping legal text to GDPR Articles (e.g., Article 5, 13, 17)  
- Use of rule-based + LLM hybrid systems for compliance flagging  
- Risk scoring and policy gap detection  

---

### 📚 **04. Retrieval-Augmented Generation (RAG) for Legal QA**

#### 📌 **Subtopics Covered:**
- Designing a **Legal-RAG** system with structured document ingestion  
- Chunking legal PDFs intelligently (sections, sub-clauses)  
- Embedding models: Legal-BERT / OpenAI embeddings / SBERT  
- LangChain / Haystack pipelines with court cases, regulations, or policies  
- Evaluation: faithfulness, citation integrity, hallucination score  

---

### 💰 **05. Serving Cost & Scalability Analysis with vLLM**

#### 📌 **Subtopics Covered:**
- Intro to [vLLM](https://github.com/vllm-project/vllm) for high-throughput serving  
- Cost breakdown of different deployment setups (GPU, CPU, quantized)  
- Comparison: OpenAI API vs local deployment  
- Latency vs. concurrency vs. cost trade-offs  

---

### 📘 **README.md Highlights**

- Overview of the Legal Assistant Capstone  
- System architecture diagram  
- Requirements & setup instructions  
- Sample use-cases: NDA audit, GDPR chatbot, internal compliance assistant  
- Dataset links (e.g., EU Legislation, case law, terms-of-service collections)

---
