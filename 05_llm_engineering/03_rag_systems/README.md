# 03 Rag Systems

- [01 vector databases pinecone weaviate](./01_vector_databases_pinecone_weaviate.ipynb)
- [02 advanced retrieval hybrid search](./02_advanced_retrieval_hybrid_search.ipynb)
- [03 document chunking and metadata](./03_document_chunking_and_metadata.ipynb)
- [04 evaluation with ragas trl](./04_evaluation_with_ragas_trl.ipynb)
- [05 multimodal rag images tables](./05_multimodal_rag_images_tables.ipynb)
- [06 production rag with llamaindex](./06_production_rag_with_llamaindex.ipynb)

---

## 📘 **RAG Systems – Structured Index**

---

### 🧩 **01. Vector Databases: Pinecone, Weaviate**

#### 📌 **Subtopics:**
- **Introduction to Vector Databases**
  - What are vector embeddings and why they’re essential for RAG
- **Pinecone Overview**
  - Setting up, indexing, and querying embeddings with Pinecone
- **Weaviate Overview**
  - Schema-based search and modular architecture of Weaviate
- **Example:** Comparing indexing/querying workflows in Pinecone vs Weaviate

---

### 🧩 **02. Advanced Retrieval: Hybrid Search**

#### 📌 **Subtopics:**
- **Retrieval Strategies**
  - Dense retrieval, sparse retrieval (BM25), and hybrid methods
- **Hybrid Search Techniques**
  - Combining semantic and keyword-based results
- **Reranking with Cross-Encoders**
  - Improving top-k relevance with reranking models
- **Example:** Implementing hybrid retrieval using FAISS + BM25

---

### 🧩 **03. Document Chunking and Metadata**

#### 📌 **Subtopics:**
- **Chunking Strategies**
  - Fixed-size, semantic, and recursive chunking
- **Metadata Enrichment**
  - Attaching source, title, and custom tags to chunks
- **Optimizing for Retrieval Performance**
  - Tradeoffs in chunk size, overlap, and context preservation
- **Example:** Chunking long PDFs for RAG using LangChain tools

---

### 🧩 **04. Evaluation with RAGAS and TRL**

#### 📌 **Subtopics:**
- **Why Evaluate RAG?**
  - Key metrics: faithfulness, relevance, grounding
- **RAGAS (RAG Assessment)**
  - Open-source toolkit for evaluating RAG pipelines
- **TRL Integration**
  - Using TRL for RLHF-style feedback and reward modeling in RAG
- **Example:** Running RAGAS to benchmark a RAG pipeline on a QA dataset

---

### 🧩 **05. Multimodal RAG: Images, Tables**

#### 📌 **Subtopics:**
- **Expanding Beyond Text**
  - Incorporating image and tabular data into RAG systems
- **Embedding and Indexing Multimodal Data**
  - Visual embedding models and handling structured tables
- **Multimodal Retrieval Techniques**
  - Cross-modal retrieval and generation
- **Example:** RAG pipeline that answers questions from PDFs with embedded charts and tables

---

### 🧩 **06. Production RAG with LlamaIndex**

#### 📌 **Subtopics:**
- **LlamaIndex Overview**
  - Core components: document loaders, indices, query engines
- **Building End-to-End RAG Pipelines**
  - Retrieval, context construction, response generation
- **Best Practices for Deployment**
  - Caching, observability, latency reduction
- **Example:** Deploying a production-grade RAG pipeline using LlamaIndex and OpenAI API

---
