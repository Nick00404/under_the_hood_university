# 05 Model Optimization

- [01 quantization post training qat](./01_quantization_post_training_qat.ipynb)
- [02 pruning magnitude optimal brain](./02_pruning_magnitude_optimal_brain.ipynb)
- [03 knowledge distillation teacher student](./03_knowledge_distillation_teacher_student.ipynb)
- [04 onnx and tensorrt conversion](./04_onnx_and_tensorrt_conversion.ipynb)
- [05 tflite and coreml for mobile](./05_tflite_and_coreml_for_mobile.ipynb)
- [06 mixed precision training](./06_mixed_precision_training.ipynb)
- [ 07 lab weight pruning and accuracy tracking.ipynb ](./07_lab_weight_pruning_and_accuracy_tracking.ipynb)  
- [ 08 lab quantize resnet fp32 to_int8.ipynb ](./08_lab_quantize_resnet_fp32_to_int8.ipynb)  
- [ 09 lab distill teacher student on mnist.ipynb ](./09_lab_distill_teacher_student_on_mnist.ipynb)  
---

## 📘 **Model Optimization and Deployment – Structured Index**

---

### 🧩 **01. Quantization: Post-Training and QAT (Quantization-Aware Training)**

#### 📌 **Subtopics:**
- **Introduction to Quantization**
  - What is quantization, and why is it important for model optimization?
  - Overview of different types: post-training quantization vs. quantization-aware training (QAT)
  - Benefits: Reducing model size, improving inference speed, and lowering power consumption
- **Post-Training Quantization (PTQ)**
  - The process of reducing precision (e.g., 32-bit to 8-bit) after model training
  - Example: Applying PTQ to a pre-trained model using PyTorch or TensorFlow
- **Quantization-Aware Training (QAT)**
  - How QAT fine-tunes the model with quantization constraints during training
  - Example: Implementing QAT using TensorFlow or PyTorch

---

### 🧩 **02. Pruning: Magnitude-Based and Optimal Brain Pruning**

#### 📌 **Subtopics:**
- **Introduction to Model Pruning**
  - What is pruning, and how does it reduce the number of model parameters?
  - Applications: Speeding up inference and reducing memory footprint
- **Magnitude-Based Pruning**
  - A simple approach to pruning where smaller weight magnitudes are set to zero
  - Example: Implementing magnitude-based pruning using TensorFlow or PyTorch
- **Optimal Brain Pruning (OBP)**
  - A more sophisticated pruning technique that optimizes the model's brain structure
  - Advantages of OBP over magnitude-based pruning in terms of model accuracy and efficiency
  - Example: Applying OBP to a neural network model

---

### 🧩 **03. Knowledge Distillation: Teacher-Student Models**

#### 📌 **Subtopics:**
- **Introduction to Knowledge Distillation**
  - What is knowledge distillation, and how does it work?
  - The role of the "teacher" and "student" models
  - Benefits: Compressing large models into smaller ones without losing performance
- **Training the Teacher-Student Model**
  - How to train a smaller "student" model to replicate the outputs of a larger "teacher" model
  - Example: Implementing knowledge distillation with a teacher-student setup in PyTorch
- **Applications of Knowledge Distillation**
  - Use cases: Deploying efficient models on edge devices or in production environments with limited resources
  - Example: Distilling a BERT model into a smaller version for faster inference

---

### 🧩 **04. ONNX and TensorRT Conversion for Inference Optimization**

#### 📌 **Subtopics:**
- **Introduction to ONNX (Open Neural Network Exchange)**
  - What is ONNX, and how does it facilitate model interoperability between different frameworks?
  - Converting models from PyTorch/TensorFlow to ONNX format
  - Example: Exporting a model from PyTorch to ONNX and running inference on ONNX Runtime
- **TensorRT for Model Optimization**
  - What is TensorRT, and how does it optimize models for NVIDIA GPUs?
  - Converting ONNX models to TensorRT format for optimized inference
  - Example: Using TensorRT for faster inference on NVIDIA GPUs

---

### 🧩 **05. TFLite and CoreML for Mobile Deployment**

#### 📌 **Subtopics:**
- **Introduction to Mobile Deployment**
  - Challenges of deploying models on mobile devices with limited resources
  - Overview of TFLite (TensorFlow Lite) and CoreML for mobile deployment
- **TensorFlow Lite (TFLite)**
  - Converting TensorFlow models to TFLite format for mobile devices
  - Optimizing models for mobile performance with TFLite (e.g., quantization and pruning)
  - Example: Deploying a model on Android using TFLite
- **CoreML for iOS Devices**
  - Converting models to CoreML format for deployment on iOS devices
  - Optimizing models with CoreML tools (e.g., quantization and pruning)
  - Example: Deploying a model on an iOS device using CoreML

---

### 🧩 **06. Mixed Precision Training**

#### 📌 **Subtopics:**
- **Introduction to Mixed Precision Training**
  - What is mixed precision training, and why is it useful?
  - The difference between single-precision (FP32) and half-precision (FP16) arithmetic
  - Benefits: Faster training, reduced memory usage, and improved efficiency
- **Implementing Mixed Precision Training**
  - How mixed precision training speeds up model training on compatible hardware (e.g., NVIDIA A100 GPUs)
  - Example: Enabling mixed precision training with PyTorch and TensorFlow
- **Hardware and Libraries for Mixed Precision**
  - Leveraging hardware like Tensor Cores and libraries like NVIDIA's Apex for mixed precision
  - Example: Configuring mixed precision training for TensorFlow and PyTorch

---

### 🧠 **Bonus:**
- **Emerging Trends in Model Optimization**
  - Explore new advancements in model optimization and deployment for edge and mobile devices.
  - Consider future improvements to distillation, pruning, and quantization techniques.
- **Deployment Strategies**
  - How to deploy optimized models into real-world applications, including considerations for cloud, edge, and mobile environments.

---
