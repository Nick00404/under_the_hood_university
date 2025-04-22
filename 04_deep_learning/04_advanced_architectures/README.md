# 04 Advanced Architectures

- [01 graph neural networks with pyg dgl](./01_graph_neural_networks_with_pyg_dgl.ipynb)
- [02 memory augmented nets neural turing machines](./02_memory_augmented_nets_neural_turing_machines.ipynb)
- [03 meta learning maml prototypical nets](./03_meta_learning_maml_prototypical_nets.ipynb)
- [04 attention free architectures mlp mixer](./04_attention_free_architectures_mlp_mixer.ipynb)
- [05 spiking neural nets surrogate gradients](./05_spiking_neural_nets_surrogate_gradients.ipynb)
- [06 diffusion models for generation](./06_diffusion_models_for_generation.ipynb)

---

## 📘 **Advanced Deep Learning – Structured Index**

---

### 🧩 **01. Graph Neural Networks (GNNs) with PyG and DGL**

#### 📌 **Subtopics:**
- **Introduction to Graph Neural Networks (GNNs)**
  - Understanding the concept of graphs in deep learning
  - Applications of GNNs: Social networks, recommendation systems, molecular graphs, etc.
- **PyTorch Geometric (PyG) and Deep Graph Library (DGL)**
  - Overview of PyG and DGL for building GNNs
  - How to implement basic GNN models using PyG and DGL
  - Example: Graph classification and node classification with PyG/DGL
- **Graph Convolutional Networks (GCNs)**
  - Convolution operations on graph structures
  - The spectral and spatial perspectives of GCNs
  - Example: Implementing a simple GCN for node classification

---

### 🧩 **02. Memory-Augmented Networks and Neural Turing Machines**

#### 📌 **Subtopics:**
- **Memory-Augmented Networks (MANNs)**
  - What are memory-augmented networks, and why do they require external memory?
  - Applications: Tasks that require reasoning and memory beyond traditional neural networks
- **Neural Turing Machines (NTMs)**
  - Architecture of NTMs: Neural networks with a differentiable memory matrix
  - How NTMs solve problems like algorithmic tasks and memory-based reasoning
  - Example: Implementing a simple Neural Turing Machine using PyTorch
- **Memory Networks and Their Applications**
  - Overview of memory networks and how they store and retrieve information
  - How they perform well in NLP and question answering tasks
  - Example: Implementing a memory network for QA systems

---

### 🧩 **03. Meta-Learning: MAML and Prototypical Networks**

#### 📌 **Subtopics:**
- **Introduction to Meta-Learning**
  - What is meta-learning, and why is it important for few-shot learning?
  - Applications of meta-learning in real-world tasks like robotics, computer vision, and NLP
- **Model-Agnostic Meta-Learning (MAML)**
  - Overview of MAML for few-shot learning
  - How MAML trains models to adapt quickly to new tasks with minimal data
  - Example: Implementing MAML for few-shot classification in PyTorch
- **Prototypical Networks for Few-Shot Learning**
  - Overview of prototypical networks and how they learn representations for few-shot learning
  - How prototypical networks compare to other meta-learning methods
  - Example: Implementing a Prototypical Network for few-shot classification tasks

---

### 🧩 **04. Attention-Free Architectures: MLP-Mixer**

#### 📌 **Subtopics:**
- **Introduction to Attention-Free Architectures**
  - The need for attention mechanisms in traditional architectures (transformers) vs. attention-free models
  - How attention-free models aim to reduce computational complexity
- **MLP-Mixer Architecture**
  - Overview of MLP-Mixer and its key components: Token-mixing and channel-mixing layers
  - Comparison to transformer-based models in terms of performance and efficiency
  - Example: Implementing MLP-Mixer for image classification
- **Alternatives to Attention Mechanisms**
  - How other attention-free methods (e.g., ConvNeXt) compare to traditional attention-based models
  - Pros and cons of attention-free models for NLP and computer vision tasks

---

### 🧩 **05. Spiking Neural Networks (SNNs) and Surrogate Gradients**

#### 📌 **Subtopics:**
- **Introduction to Spiking Neural Networks (SNNs)**
  - The biological inspiration behind SNNs
  - How SNNs differ from traditional neural networks in terms of information processing (spikes and time-based events)
  - Applications of SNNs in neuroscience and neuromorphic computing
- **Surrogate Gradients for SNN Training**
  - Challenges of training SNNs with traditional backpropagation
  - Introduction to surrogate gradients and how they allow gradient-based learning for SNNs
  - Example: Implementing SNNs with surrogate gradients for simple tasks like classification
- **Neuromorphic Computing**
  - Overview of neuromorphic computing and its role in AI
  - How SNNs are used in edge devices and low-power systems
  - The future of SNNs and their role in AI research

---

### 🧩 **06. Diffusion Models for Image Generation**

#### 📌 **Subtopics:**
- **Introduction to Diffusion Models**
  - What are diffusion models, and how do they generate data (e.g., images)?
  - Comparison to other generative models like GANs and VAEs
  - How diffusion models iteratively denoise noisy data to produce high-quality outputs
- **Training and Sampling with Diffusion Models**
  - Overview of the training process: noise schedules, forward and reverse diffusion
  - Sampling from diffusion models using various techniques (e.g., Langevin dynamics)
  - Example: Implementing a basic diffusion model for image generation
- **Applications of Diffusion Models**
  - Use cases: Image generation, inpainting, and super-resolution
  - How diffusion models have outperformed GANs in some tasks
  - Example: Comparing diffusion models with GANs on image generation tasks

---

### 🧠 **Bonus:**
- **Emerging Trends in Deep Learning**
  - Explore how advanced architectures like Graph Neural Networks, Memory-Augmented Networks, and Spiking Neural Networks are evolving.
  - Dive deeper into cutting-edge topics in AI, such as meta-learning, attention-free models, and diffusion models.
- **Real-World Applications and Research Directions**
  - Use cases in fields like robotics, self-driving cars, healthcare, and personalized AI.
  - The future of AI: From neuromorphic computing to scalable, energy-efficient models.

---
