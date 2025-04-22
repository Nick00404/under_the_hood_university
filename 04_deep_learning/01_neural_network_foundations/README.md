# 01 Neural Network Foundations
[01 tensor operations](./04_deep_learning/01_neural_network_foundations/01_tensor_operations_with_pytorch_tensorflow.ipynb)

- [01 tensor operations with pytorch tensorflow](./01_tensor_operations_with_pytorch_tensorflow.ipynb)
- [02 building mlps from scratch](./02_building_mlps_from_scratch.ipynb)
- [03 activation functions and vanishing gradients](./03_activation_functions_and_vanishing_gradients.ipynb)
- [04 loss functions mse crossentropy contrastive](./04_loss_functions_mse_crossentropy_contrastive.ipynb)
- [05 backpropagation autograd custom rules](./05_backpropagation_autograd_custom_rules.ipynb)
- [06 regularization dropout batchnorm l1l2](./06_regularization_dropout_batchnorm_l1l2.ipynb)

---

## 📘 Deep Learning – Structured Index

---

### 🧩 **01. Tensor Operations with PyTorch & TensorFlow**

#### 📌 Subtopics:
- **Introduction to Tensors**
  - What are Tensors? A generalization of matrices
  - Tensors in PyTorch vs TensorFlow
  - Common tensor operations (addition, multiplication, reshaping, etc.)
- **PyTorch Tensors**
  - Creating tensors and manipulating shapes in PyTorch
  - Indexing and slicing tensors in PyTorch
  - Broadcasting and its importance in deep learning
- **TensorFlow Tensors**
  - TensorFlow vs PyTorch: Key differences in tensor operations
  - Operations in TensorFlow (tf.Variable, tf.constant, tf.placeholder)
  - TensorFlow operations for deep learning models

---

### 🧩 **02. Building MLPs from Scratch**

#### 📌 Subtopics:
- **Introduction to Multi-Layer Perceptrons (MLPs)**
  - Understanding the architecture of MLPs
  - Components of MLP: Input layer, hidden layers, output layer
  - Activation functions and weights in MLPs
- **Building an MLP from Scratch in PyTorch**
  - Setting up a simple MLP model using `torch.nn.Module`
  - Forward pass and backward pass in MLP
  - Example code: Training a simple MLP on a dataset
- **Building an MLP from Scratch in TensorFlow**
  - Implementing MLP using TensorFlow’s Keras API
  - Defining the model architecture in Keras
  - Example code: Training and evaluating an MLP model

---

### 🧩 **03. Activation Functions and Vanishing Gradients**

#### 📌 Subtopics:
- **Activation Functions in Neural Networks**
  - Role of activation functions in deep networks
  - Common activation functions: ReLU, Sigmoid, Tanh, Softmax
  - Why activation functions are crucial for learning complex patterns
- **Vanishing Gradient Problem**
  - What is vanishing gradients and why does it occur?
  - How the vanishing gradient problem affects deep neural networks
  - Solutions to vanishing gradients (e.g., ReLU, He Initialization)
- **Improving Learning with Activation Functions**
  - Leaky ReLU, ELU, SELU and their advantages over traditional ReLU
  - Exploding gradients and gradient clipping
  - Implementing these solutions in both PyTorch and TensorFlow

---

### 🧩 **04. Loss Functions: MSE, Cross-Entropy, Contrastive**

#### 📌 Subtopics:
- **Mean Squared Error (MSE) Loss**
  - Understanding MSE as a loss function for regression
  - How to compute MSE and its derivatives
  - PyTorch/TensorFlow implementation of MSE
- **Cross-Entropy Loss**
  - Why cross-entropy is used for classification tasks
  - Binary vs multiclass cross-entropy
  - Softmax with cross-entropy loss function
- **Contrastive Loss**
  - What is contrastive loss and its use in metric learning
  - Understanding Siamese networks and how contrastive loss fits in
  - Implementing contrastive loss with PyTorch/TensorFlow

---

### 🧩 **05. Backpropagation, Autograd, and Custom Rules**

#### 📌 Subtopics:
- **Backpropagation Overview**
  - What is backpropagation and how does it work?
  - Understanding the chain rule in neural networks
  - Steps in the forward pass and backward pass
- **Autograd in PyTorch**
  - PyTorch’s autograd mechanism: how it calculates gradients automatically
  - How `autograd` tracks operations and computes derivatives
  - Practical example: Training a network using autograd
- **Custom Gradient Rules**
  - Implementing custom backpropagation rules in PyTorch
  - Using `torch.autograd.Function` for custom gradient computation
  - Example of defining custom gradient calculations

---

### 🧩 **06. Regularization: Dropout, BatchNorm, L1/L2**

#### 📌 Subtopics:
- **Dropout Regularization**
  - What is dropout and why it helps prevent overfitting?
  - How dropout is implemented in PyTorch and TensorFlow
  - Trade-offs in choosing the right dropout rate
- **Batch Normalization**
  - Role of Batch Normalization in accelerating convergence and stabilizing training
  - How it works: Normalizing activations per mini-batch
  - Implementing BatchNorm in both PyTorch and TensorFlow
- **L1/L2 Regularization**
  - L1 vs L2 regularization: What they are and how they work
  - How to apply L1/L2 regularization in PyTorch and TensorFlow
  - How regularization affects model complexity and generalization

---

### 🧠 Bonus:
- **Advanced Neural Network Architectures**
  - Overview of CNNs, RNNs, and GANs as part of deep learning progression
  - Case studies and real-world applications for MLPs and regularization techniques
- **Hyperparameter Tuning**
  - Techniques to optimize neural network performance
  - Implementing grid search and random search for neural networks
  - Using TensorBoard for visualizing model training

---
