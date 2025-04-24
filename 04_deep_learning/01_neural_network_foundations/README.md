# 01 Neural Network Foundations
- [01 tensor operations with pytorch tensorflow](./01_tensor_operations_with_pytorch_tensorflow.ipynb)
- [02 building mlps from scratch](./02_building_mlps_from_scratch.ipynb)
- [03 activation functions and vanishing gradients](./03_activation_functions_and_vanishing_gradients.ipynb)
- [04 loss functions mse crossentropy contrastive](./04_loss_functions_mse_crossentropy_contrastive.ipynb)
- [05 backpropagation autograd custom rules](./05_backpropagation_autograd_custom_rules.ipynb)
- [06 regularization dropout batchnorm l1l2](./06_regularization_dropout_batchnorm_l1l2.ipynb)
- [07 lab manual tensor_ops_and shapes.ipynb](./07_lab_manual_tensor_ops_and_shapes.ipynb)  
- [08 lab xor problem_with mlp.ipynb](./08_lab_xor_problem_with_mlp.ipynb)  
- [09 lab autograd from scratch.ipynb](./09_lab_autograd_from_scratch.ipynb)  

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











---

## 🔧 `04_deep_learning`

---

### 📁 `01_neural_network_foundations`

| Lab Filename                                      | Purpose |
|--------------------------------------------------|---------|
| `07_lab_manual_tensor_ops_and_shapes.ipynb`      | Hands-on with PyTorch & TensorFlow tensor basics |
| `08_lab_xor_problem_with_mlp.ipynb`              | Implement an MLP that solves XOR (non-linearity test) |
| `09_lab_autograd_from_scratch.ipynb`             | Build your own `autograd` engine for full backprop understanding |

---

### 📁 `02_computer_vision`

| Lab Filename                                       | Purpose |
|---------------------------------------------------|---------|
| `07_lab_cnn_feature_maps_visualization.ipynb`     | Visualize CNN filters & activations layer-by-layer |
| `08_lab_data_augmentation_comparison.ipynb`       | Try flips, crops, cutout, mixup and compare accuracy impact |
| `09_lab_finetune_resnet_on_custom_data.ipynb`     | Mini fine-tune ResNet on e.g. flower dataset or your dataset |

---

### 📁 `03_natural_language_processing`

| Lab Filename                                            | Purpose |
|---------------------------------------------------------|---------|
| `07_lab_finetuning_gpt2_text_generation.ipynb`          | Fine-tune GPT-2 on custom text + compare before/after |
| `08_lab_masked_language_modeling_from_scratch.ipynb`    | Train a mini BERT-style model on small dataset |
| `09_lab_attention_visualization.ipynb`                  | Use `bertviz` or equivalent to see attention heads in action |

---

### 📁 `04_advanced_architectures`

| Lab Filename                                             | Purpose |
|----------------------------------------------------------|---------|
| `07_lab_gnn_node_classification_with_cora.ipynb`         | Build Graph Neural Network with PyG or DGL on citation graphs |
| `08_lab_memory_augmented_net_tiny_tasks.ipynb`           | Simple NTM use-case: copy tasks or associative recall |
| `09_lab_diffusion_model_toy_image_gen.ipynb`             | Implement or run toy diffusion-based image generator |

---

### 📁 `05_model_optimization`

| Lab Filename                                             | Purpose |
|----------------------------------------------------------|---------|
| `07_lab_weight_pruning_and_accuracy_tracking.ipynb`      | Visualize what happens to accuracy as you prune weights |
| `08_lab_quantize_resnet_fp32_to_int8.ipynb`              | Quantize pretrained ResNet with ONNX or TFLite and compare perf |
| `09_lab_distill_teacher_student_on_mnist.ipynb`          | Train a large teacher → distill knowledge to a small student |

---

### 📁 `06_deployment_and_scaling`

| Lab Filename                                               | Purpose |
|------------------------------------------------------------|---------|
| `07_lab_export_pytorch_to_onnx_and_run.ipynb`              | Convert a trained PyTorch model to ONNX and infer locally |
| `08_lab_dockerize_and_test_flask_model_server.ipynb`       | Package model with Flask + Docker + run test API calls |
| `09_lab_k8s_microservice_mock_deploy.ipynb`                | Deploy a basic REST-serving model to Minikube or GKE (mock style) |

