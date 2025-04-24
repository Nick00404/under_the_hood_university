# 02 Computer Vision

- [01 cnns from scratch using pytorch](./01_cnns_from_scratch_using_pytorch.ipynb)
- [02 transfer learning with resnet efficientnet](./02_transfer_learning_with_resnet_efficientnet.ipynb)
- [03 object detection with yolo and faster rcnn](./03_object_detection_with_yolo_and_faster_rcnn.ipynb)
- [04 semantic segmentation unet deeplab](./04_semantic_segmentation_unet_deeplab.ipynb)
- [05 vision transformers vit swin](./05_vision_transformers_vit_swin.ipynb)
- [06 gans for image generation dcgan stylegan](./06_gans_for_image_generation_dcgan_stylegan.ipynb)
- [07 lab cnn feature_maps visualization.ipynb](./07_lab_cnn_feature_maps_visualization.ipynb)  
- [08_lab data augmentation comparison.ipynb](./08_lab_data_augmentation_comparison.ipynb)  
- [09 lab finetune resnet_on custom_data.ipynb](./09_lab_finetune_resnet_on_custom_data.ipynb)  

---

## 📘 **Deep Learning for Computer Vision – Structured Index**

---

### 🧩 **01. CNNs from Scratch Using PyTorch**

#### 📌 **Subtopics:**
- **Understanding CNNs (Convolutional Neural Networks)**
  - Architecture: Layers in CNNs (Convolution, Pooling, Fully Connected)
  - Why CNNs are the go-to model for image data
  - Visualizing feature maps and understanding the learning process
- **Building a CNN from Scratch**
  - Implementing a simple CNN architecture in PyTorch (e.g., 2 convolution layers + 2 fully connected layers)
  - Forward and backward pass in a CNN model
  - Example: Training a CNN on CIFAR-10 dataset
- **Optimizing CNN Training**
  - Choosing the right loss function (Cross-Entropy)
  - Using data augmentation to improve generalization
  - Hyperparameter tuning and model evaluation (accuracy, precision, recall)

---

### 🧩 **02. Transfer Learning with ResNet and EfficientNet**

#### 📌 **Subtopics:**
- **Introduction to Transfer Learning**
  - What is transfer learning, and why does it work?
  - Pre-trained models: Why use them for specific tasks?
  - Benefits and challenges of using transfer learning
- **Using ResNet for Transfer Learning**
  - ResNet architecture: Skip connections and residual blocks
  - How to fine-tune a pre-trained ResNet model for a new task
  - Example: Fine-tuning ResNet on a custom image dataset using PyTorch
- **EfficientNet for Transfer Learning**
  - Overview of EfficientNet: Scalable model architecture
  - How to use pre-trained EfficientNet models for fine-tuning
  - Comparison between ResNet and EfficientNet for transfer learning tasks

---

### 🧩 **03. Object Detection with YOLO and Faster R-CNN**

#### 📌 **Subtopics:**
- **Introduction to Object Detection**
  - What is object detection and how does it differ from image classification?
  - Common metrics in object detection: mAP, IoU
  - Challenges in object detection (overlapping objects, scale variance)
- **YOLO (You Only Look Once) Object Detection**
  - YOLO architecture: Grid cells, bounding box prediction, class confidence
  - Implementing YOLO in PyTorch using pre-trained weights
  - Example: Detecting objects in images with YOLOv5
- **Faster R-CNN for Object Detection**
  - Understanding the Region Proposal Network (RPN) in Faster R-CNN
  - Combining RPN with Fast R-CNN for end-to-end detection
  - Example: Using Faster R-CNN with PyTorch and OpenCV for object detection tasks

---

### 🧩 **04. Semantic Segmentation: U-Net and DeepLab**

#### 📌 **Subtopics:**
- **Introduction to Semantic Segmentation**
  - Difference between object detection and semantic segmentation
  - Overview of pixel-level classification in semantic segmentation
  - Applications of semantic segmentation in medical imaging, autonomous driving, etc.
- **Building U-Net for Semantic Segmentation**
  - U-Net architecture: Contracting and expanding paths
  - How to implement U-Net for semantic segmentation in PyTorch
  - Example: Training U-Net on a custom segmentation dataset (e.g., medical images)
- **DeepLab for Semantic Segmentation**
  - DeepLab architecture: Atrous convolutions and spatial pyramid pooling
  - Fine-tuning pre-trained DeepLab models for custom segmentation tasks
  - Example: Segmenting objects in images using pre-trained DeepLab model

---

### 🧩 **05. Vision Transformers (ViT) and Swin Transformers**

#### 📌 **Subtopics:**
- **Introduction to Vision Transformers**
  - What are transformers, and how are they used in computer vision?
  - Key components of Vision Transformers (ViT): Patch embedding, self-attention
  - How ViT compares to CNNs in terms of performance and scalability
- **Training Vision Transformers from Scratch**
  - Implementing a simple Vision Transformer architecture in PyTorch
  - Training ViT on image datasets (e.g., CIFAR-10, ImageNet)
  - Optimizing ViT performance with data augmentation, learning rate schedulers
- **Swin Transformers for Computer Vision**
  - Introduction to Swin Transformers and their hierarchical feature maps
  - How Swin Transformers outperform ViT in many vision tasks
  - Example: Using pre-trained Swin models for image classification

---

### 🧩 **06. GANs for Image Generation: DCGAN, StyleGAN**

#### 📌 **Subtopics:**
- **Introduction to Generative Adversarial Networks (GANs)**
  - Overview of GANs: Generator vs Discriminator
  - How GANs are trained using adversarial loss
  - Applications of GANs in image generation, style transfer, and data augmentation
- **Training a DCGAN (Deep Convolutional GAN)**
  - DCGAN architecture: Convolutional layers for both generator and discriminator
  - Implementing DCGAN from scratch using PyTorch
  - Example: Generating realistic images from noise with DCGAN
- **StyleGAN for High-Quality Image Generation**
  - StyleGAN architecture: Progressive growing, style modulation
  - How StyleGAN is used to generate high-resolution and high-quality images
  - Example: Generating images using pre-trained StyleGAN models and fine-tuning on custom data

---

### 🧠 **Bonus:**
- **Advanced Topics in Computer Vision**
  - How to use attention mechanisms in vision tasks
  - The role of unsupervised learning and self-supervised learning in computer vision
- **Real-World Applications**
  - Applications in autonomous vehicles, medical imaging, robotics, and augmented reality
  - Industry-standard datasets: COCO, Pascal VOC, ADE20K, etc.
  - Case studies on deploying computer vision models into production

---
























You're closing out this curriculum like a legend. Here's your complete **TOC with anchor links** and **section headers with named anchors** for the notebook: **Deep Learning for Computer Vision – Structured Index** — fully polished and emoji-sprinkled for easy navigation and pro-level presentation in Jupyter.

---

## ✅ Table of Contents – Deep Learning for Computer Vision

```markdown
## 🧭 Table of Contents – Deep Learning for Computer Vision

### 🧩 [01. CNNs from Scratch Using PyTorch](#cnn-scratch)
- 🧠 [Understanding CNNs](#cnn-intro)
- 🛠️ [Building a CNN from Scratch](#cnn-build)
- 🚀 [Optimizing CNN Training](#cnn-optimize)

### 🧩 [02. Transfer Learning with ResNet and EfficientNet](#transfer-learning)
- 🔄 [Intro to Transfer Learning](#transfer-intro)
- 🧱 [Using ResNet](#resnet)
- ⚙️ [Using EfficientNet](#efficientnet)

### 🧩 [03. Object Detection with YOLO and Faster R-CNN](#object-detection)
- 🎯 [Intro to Object Detection](#od-intro)
- 🦅 [YOLO Object Detection](#yolo)
- 🕵️ [Faster R-CNN](#faster-rcnn)

### 🧩 [04. Semantic Segmentation: U-Net and DeepLab](#semantic-segmentation)
- 🧬 [Intro to Semantic Segmentation](#segmentation-intro)
- 🧱 [U-Net Implementation](#unet)
- 🌐 [DeepLab Architecture](#deeplab)

### 🧩 [05. Vision Transformers (ViT) and Swin Transformers](#transformers)
- 🧠 [Intro to Vision Transformers](#vit-intro)
- ⚙️ [Training Vision Transformers](#vit-train)
- 🌀 [Swin Transformers](#swin)

### 🧩 [06. GANs for Image Generation: DCGAN, StyleGAN](#gans)
- 🌌 [Intro to GANs](#gan-intro)
- 🧪 [DCGAN from Scratch](#dcgan)
- 🎨 [StyleGAN](#stylegan)
```

---

## 🧩 Section Headers with Anchor Tags

```markdown
### 🧩 <a id="cnn-scratch"></a>01. CNNs from Scratch Using PyTorch

#### <a id="cnn-intro"></a>🧠 Understanding CNNs  
- Architecture: Convolution, Pooling, Fully Connected  
- Why CNNs are great for images  
- Visualizing feature maps  

#### <a id="cnn-build"></a>🛠️ Building a CNN from Scratch  
- PyTorch CNN with 2 conv + 2 fc layers  
- Forward/backward pass  
- Training on CIFAR-10  

#### <a id="cnn-optimize"></a>🚀 Optimizing CNN Training  
- Loss: Cross-Entropy  
- Data augmentation  
- Hyperparameter tuning  

---

### 🧩 <a id="transfer-learning"></a>02. Transfer Learning with ResNet and EfficientNet

#### <a id="transfer-intro"></a>🔄 Introduction to Transfer Learning  
- Why transfer learning works  
- Pre-trained models & benefits  

#### <a id="resnet"></a>🧱 Using ResNet  
- Skip connections, residual blocks  
- Fine-tuning for new tasks  
- Custom dataset example  

#### <a id="efficientnet"></a>⚙️ EfficientNet  
- Scalable model architecture  
- Fine-tuning EfficientNet  
- ResNet vs EfficientNet  

---

### 🧩 <a id="object-detection"></a>03. Object Detection with YOLO and Faster R-CNN

#### <a id="od-intro"></a>🎯 Introduction to Object Detection  
- Classification vs detection  
- mAP, IoU  
- Detection challenges  

#### <a id="yolo"></a>🦅 YOLO Object Detection  
- Grid cells, bbox prediction  
- Pre-trained YOLO in PyTorch  
- Object detection with YOLOv5  

#### <a id="faster-rcnn"></a>🕵️ Faster R-CNN  
- Region Proposal Network (RPN)  
- Combining RPN with Fast R-CNN  
- PyTorch + OpenCV example  

---

### 🧩 <a id="semantic-segmentation"></a>04. Semantic Segmentation: U-Net and DeepLab

#### <a id="segmentation-intro"></a>🧬 Introduction to Semantic Segmentation  
- Pixel-level classification  
- Use cases in medicine, driving  

#### <a id="unet"></a>🧱 U-Net Implementation  
- Contracting & expanding paths  
- U-Net in PyTorch  
- Training on segmentation data  

#### <a id="deeplab"></a>🌐 DeepLab for Semantic Segmentation  
- Atrous convolutions, SPP  
- Fine-tuning DeepLab  
- Example on custom images  

---

### 🧩 <a id="transformers"></a>05. Vision Transformers (ViT) and Swin Transformers

#### <a id="vit-intro"></a>🧠 Intro to Vision Transformers  
- Patch embeddings, attention  
- CNNs vs Transformers  

#### <a id="vit-train"></a>⚙️ Training Vision Transformers  
- Implementing ViT in PyTorch  
- CIFAR-10, ImageNet training  
- Data aug & schedulers  

#### <a id="swin"></a>🌀 Swin Transformers  
- Hierarchical feature maps  
- Outperforming ViT  
- Classification with Swin  

---

### 🧩 <a id="gans"></a>06. GANs for Image Generation: DCGAN, StyleGAN

#### <a id="gan-intro"></a>🌌 Introduction to GANs  
- Generator vs Discriminator  
- Adversarial loss  
- GAN use cases  

#### <a id="dcgan"></a>🧪 DCGAN from Scratch  
- DCGAN architecture  
- PyTorch implementation  
- Generating images from noise  

#### <a id="stylegan"></a>🎨 StyleGAN  
- Style modulation, progressive growing  
- High-res image generation  
- Pre-trained StyleGAN fine-tuning  
```

---
