## 28_fastai

#### **01_tabular_workflow.ipynb**

**Why it matters**:  
FastAI provides a high-level API that simplifies building deep learning models, especially for tabular data. It abstracts complex processes, making it easier for practitioners to apply deep learning techniques to structured datasets.

**What you'll learn**:  
- How to handle tabular data using FastAI's `TabularDataLoaders`.
- Creating and training tabular models efficiently with FastAI.
- Model interpretability, diagnostics, and fine-tuning techniques.

1. **Introduction to FastAI and the Tabular Workflow**  
   - FastAI’s philosophy and the `TabularDataLoaders`.

2. **Preparing Tabular Data for Deep Learning**  
   - Handling categorical and continuous variables, and transformations like `Categorify`, `Normalize`, and `FillMissing`.

3. **Creating a Tabular Model**  
   - Building neural networks for tabular data using `tabular_learner()`.

4. **Training a Tabular Model**  
   - Training with FastAI’s `fit_one_cycle()` and monitoring performance.

5. **Interpretability and Model Diagnostics**  
   - Using `InterpretableModel` to analyze and visualize predictions.

6. **Fine-Tuning Tabular Models**  
   - Hyperparameter tuning and using `callbacks` for regularization.

7. **Evaluation and Model Export**  
   - Evaluating performance and exporting models for deployment.

---

#### **02_transfer_learning.ipynb**

**Why it matters**:  
Transfer learning allows leveraging pre-trained models, improving the performance and efficiency of training, especially when data is scarce. FastAI makes this process straightforward across different domains like image, text, and tabular data.

**What you'll learn**:  
- Applying transfer learning with FastAI to fine-tune pre-trained models.
- Techniques for feature extraction and domain-specific fine-tuning.

1. **Introduction to Transfer Learning**  
   - Fine-tuning vs. feature extraction and the power of pre-trained models.

2. **Using Pre-trained Models in FastAI**  
   - Accessing models like ResNet, VGG, and using frozen layers.

3. **Fine-tuning Pre-trained Models**  
   - Freezing and unfreezing layers, and training with smaller learning rates.

4. **Transfer Learning for Tabular Data**  
   - Applying transfer learning techniques to tabular data.

5. **Advanced Transfer Learning Techniques**  
   - Domain-specific fine-tuning and regularization strategies.

6. **Evaluating Fine-tuned Models**  
   - Comparing fine-tuned models to base models and evaluating results.

7. **Exporting and Deploying Transfer Learning Models**  
   - Converting models for production environments and optimizing for inference.

---

#### **03_custom_models.ipynb**

**Why it matters**:  
Creating custom models allows for maximum flexibility and control over model architecture, enabling you to tailor models to specific tasks, whether they involve complex architectures or custom loss functions.

**What you'll learn**:  
- How to build custom neural networks and integrate them with FastAI.
- Implementing custom loss functions, augmentations, and optimizing models.

1. **Introduction to Custom Models in FastAI**  
   - Overview of creating custom models with FastAI.

2. **Building a Custom Neural Network Architecture**  
   - Designing and combining custom layers using `nn.Module`.

3. **FastAI's `Learner` for Custom Models**  
   - Training custom models with FastAI’s `Learner`.

4. **Building Custom Loss Functions**  
   - Defining and using custom loss functions in training.

5. **Custom Data Augmentation Techniques**  
   - Implementing and using custom augmentations in FastAI pipelines.

6. **Debugging and Optimizing Custom Models**  
   - Troubleshooting and optimizing custom architectures.

7. **Fine-Tuning Custom Models**  
   - Hyperparameter tuning and using callbacks for custom models.

8. **Deploying Custom Models**  
   - Exporting models to different formats (ONNX, TensorFlow) and deploying them.

9. **Real-World Applications**  
   - Building custom models for various use cases like image classification and time-series forecasting.
