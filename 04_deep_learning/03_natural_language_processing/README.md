# 03 Natural Language Processing

- [01 word embeddings word2vec glove fasttext](./01_word_embeddings_word2vec_glove_fasttext.ipynb)
- [02 rnns lstms gru for sequence modeling](./02_rnns_lstms_gru_for_sequence_modeling.ipynb)
- [03 attention mechanisms bahdanau transformer](./03_attention_mechanisms_bahdanau_transformer.ipynb)
- [04 pretrained transformers bert gpt finetuning](./04_pretrained_transformers_bert_gpt_finetuning.ipynb)
- [05 text generation with beam search sampling](./05_text_generation_with_beam_search_sampling.ipynb)
- [06 multilingual nlp xlm roberta mt5](./06_multilingual_nlp_xlm_roberta_mt5.ipynb)
- [07_ ab finetuning gpt2 text generation.ipynb](./07_lab_finetuning_gpt2_text_generation.ipynb)  
- [08 lab masked language modeling from_scratch.ipynb](./08_lab_masked_language_modeling_from_scratch.ipynb)  
- [09 lab attention visualization.ipynb](./09_lab_attention_visualization.ipynb)  

---

## 📘 **Deep Learning for Natural Language Processing (NLP) – Structured Index**

---

### 🧩 **01. Word Embeddings: Word2Vec, GloVe, FastText**

#### 📌 **Subtopics:**
- **Introduction to Word Embeddings**
  - What are word embeddings, and why are they used?
  - How do word embeddings capture semantic meaning in text?
- **Word2Vec**
  - Overview of Word2Vec: Continuous bag of words (CBOW) vs. Skip-Gram
  - Training Word2Vec on text data using Gensim
  - Example: Visualizing word embeddings with t-SNE
- **GloVe (Global Vectors for Word Representation)**
  - Difference between Word2Vec and GloVe
  - GloVe’s matrix factorization approach
  - Using pre-trained GloVe embeddings in NLP tasks
- **FastText**
  - FastText’s approach to representing words as subword units
  - Handling out-of-vocabulary words with FastText
  - Example: Training and using FastText for word vector representation

---

### 🧩 **02. RNNs, LSTMs, and GRUs for Sequence Modeling**

#### 📌 **Subtopics:**
- **Introduction to Recurrent Neural Networks (RNNs)**
  - What is a RNN and how does it handle sequential data?
  - Understanding vanishing and exploding gradient problems in RNNs
  - Implementing a simple RNN for text classification
- **Long Short-Term Memory (LSTM) Networks**
  - The LSTM architecture: Forget, input, and output gates
  - Solving the vanishing gradient problem with LSTMs
  - Example: Using LSTMs for sentiment analysis on text data
- **Gated Recurrent Units (GRUs)**
  - Differences between GRUs and LSTMs
  - When to choose GRUs over LSTMs
  - Implementing GRUs for language modeling or sequence prediction tasks

---

### 🧩 **03. Attention Mechanisms: Bahdanau, Transformer**

#### 📌 **Subtopics:**
- **Understanding Attention Mechanisms**
  - What is attention in the context of NLP?
  - Why attention mechanisms improve sequence-to-sequence models (like translation)?
  - Example: Implementing Bahdanau attention for sequence-to-sequence tasks
- **Bahdanau Attention (Additive Attention)**
  - Key components: Query, Key, and Value
  - How Bahdanau attention works and its application in machine translation
  - Example: Using Bahdanau attention with an RNN encoder-decoder
- **Transformers and Self-Attention**
  - Introduction to transformers and their self-attention mechanism
  - How transformers outperform RNNs in terms of parallelization and long-range dependencies
  - Example: Building a simple transformer model for text classification

---

### 🧩 **04. Pretrained Transformers: BERT, GPT, and Fine-Tuning**

#### 📌 **Subtopics:**
- **Introduction to Pretrained Transformers**
  - The rise of transformer-based models in NLP
  - Key transformer models: BERT, GPT, and their architecture differences
  - Why pretraining is effective for NLP tasks
- **BERT (Bidirectional Encoder Representations from Transformers)**
  - BERT's bidirectional attention mechanism
  - Fine-tuning BERT for downstream tasks like question answering or sentiment analysis
  - Example: Using Hugging Face’s `transformers` library for fine-tuning BERT on a custom dataset
- **GPT (Generative Pretrained Transformer)**
  - Understanding the autoregressive nature of GPT models
  - How GPT is used for text generation and language modeling
  - Example: Fine-tuning GPT-2 or GPT-3 for specific text generation tasks

---

### 🧩 **05. Text Generation with Beam Search and Sampling**

#### 📌 **Subtopics:**
- **Introduction to Text Generation**
  - What is text generation, and how does it work in NLP?
  - Common applications: chatbots, automated writing, and storytelling
- **Beam Search for Text Generation**
  - What is beam search, and how does it help improve text generation?
  - Beam width and trade-offs between speed and quality
  - Example: Implementing beam search for more coherent text generation
- **Sampling Methods for Text Generation**
  - Exploring different sampling techniques: greedy, random, temperature-based sampling
  - How temperature affects randomness and creativity in generated text
  - Example: Using temperature sampling for creative text generation

---

### 🧩 **06. Multilingual NLP: XLM, RoBERTa, mT5**

#### 📌 **Subtopics:**
- **Introduction to Multilingual NLP**
  - Why multilingual models are important for global applications
  - Challenges in multilingual NLP: tokenization, language diversity, etc.
- **XLM (Cross-lingual Language Model)**
  - XLM’s approach to learning multilingual representations
  - Using XLM for translation and cross-lingual tasks
  - Example: Fine-tuning XLM for multilingual text classification
- **RoBERTa and mT5 for Multilingual Tasks**
  - RoBERTa’s improvements over BERT and its impact on NLP
  - Using mT5 for multilingual text generation and translation
  - Example: Fine-tuning mT5 on a multilingual question answering task

---

### 🧠 **Bonus:**
- **Advanced Topics in NLP**
  - Exploring recent innovations in NLP, such as multilingual BERT (mBERT) and zero-shot learning
  - Multi-task learning in NLP: Combining different tasks to improve performance
- **Real-World NLP Applications**
  - Use cases: machine translation, sentiment analysis, named entity recognition (NER), and summarization
  - Industry-standard datasets: GLUE, SuperGLUE, SQuAD, CoNLL, and others
  - Deploying NLP models in production: Challenges and strategies

---
