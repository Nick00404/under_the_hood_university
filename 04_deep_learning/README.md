# 🧠 Deep Learning: From Neurons to Vision, Language & Optimization

Welcome to **Deep Learning @ UTHU** — the serious playground of neurons, weights, and high-level perception.

This module takes you from the math of neural networks to production-ready models for vision and NLP.  
Everything is **Colab-friendly**, GPU-scalable, and visually explained.

---

## 🧭 Book Structure

> This track is split into 6 progressive sub-books:

1. Neural Network Foundations  
2. Computer Vision  
3. Natural Language Processing  
4. Advanced Architectures  
5. Model Optimization  
6. Deployment & Scaling

---

## 🧠 1. Neural Network Foundations

> _Understand how neural networks work, from scratch._

| Notebook | Description |
|----------|-------------|
| `01_tensor_operations_with_pytorch_tensorflow.ipynb` | Tensor ops, shape logic, GPU usage |
| `02_building_mlps_from_scratch.ipynb` | Manual forward pass & backprop |
| `03_activation_functions_and_vanishing_gradients.ipynb` | ReLU, tanh, sigmoid & saturation pitfalls |
| `04_loss_functions_mse_crossentropy_contrastive.ipynb` | Loss math for regression, classification, embeddings |
| `05_backpropagation_autograd_custom_rules.ipynb` | Autograd internals + custom gradients |
| `06_regularization_dropout_batchnorm_l1l2.ipynb` | Fight overfitting with L1, L2, dropout, BN |
| `07_lab_manual_tensor_ops_and_shapes.ipynb` | Tensor shaping drills (PyTorch + TF) |
| `08_lab_xor_problem_with_mlp.ipynb` | Solve XOR with MLP: non-linear intro |
| `09_lab_autograd_from_scratch.ipynb` | Build autograd engine, visualize graph |

---

## 👁️‍🗨️ 2. Computer Vision

> _Learn how deep learning sees and understands pixels._

| Notebook | Description |
|----------|-------------|
| `01_cnns_from_scratch_using_pytorch.ipynb` | Manual conv layers + backprop |
| `02_transfer_learning_with_resnet_efficientnet.ipynb` | Fine-tune SOTA CNNs |
| `03_object_detection_with_yolo_and_faster_rcnn.ipynb` | Boxes, anchors, detection heads |
| `04_semantic_segmentation_unet_deeplab.ipynb` | Per-pixel predictions |
| `05_vision_transformers_vit_swin.ipynb` | From patches to global attention |
| `06_gans_for_image_generation_dcgan_stylegan.ipynb` | Image generation from noise |
| `07_lab_cnn_feature_maps_visualization.ipynb` | Visualize CNN filters and layers |
| `08_lab_data_augmentation_comparison.ipynb` | Augmentations: mixup, cutout, flips |
| `09_lab_finetune_resnet_on_custom_data.ipynb` | Custom dataset finetuning demo |

---

## 🗣️ 3. Natural Language Processing

> _Turn tokens into meaning. Words → Vectors → Logic._

| Notebook | Description |
|----------|-------------|
| `01_word_embeddings_word2vec_glove_fasttext.ipynb` | Semantic vector spaces |
| `02_rnns_lstms_gru_for_sequence_modeling.ipynb` | Recurrent nets for text and time |
| `03_attention_mechanisms_bahdanau_transformer.ipynb` | Foundations of Transformers |
| `04_pretrained_transformers_bert_gpt_finetuning.ipynb` | BERT/GPT + Hugging Face finetune |
| `05_text_generation_with_beam_search_sampling.ipynb` | Greedy vs beam vs sampling |
| `06_multilingual_nlp_xlm_roberta_mt5.ipynb` | Translate, summarize, cross-lingual |
| `07_lab_finetuning_gpt2_text_generation.ipynb` | GPT2 on your own corpus |
| `08_lab_masked_language_modeling_from_scratch.ipynb` | Mini-BERT from scratch |
| `09_lab_attention_visualization.ipynb` | Show Transformer attention heads |

---

## 🧪 4. Advanced Architectures

> _Beyond feedforward: memory, graphs, and generative power._

| Notebook | Description |
|----------|-------------|
| `01_graph_neural_networks_with_pyg_dgl.ipynb` | GNNs for citation/classification |
| `02_memory_augmented_nets_neural_turing_machines.ipynb` | External memory models |
| `03_meta_learning_maml_prototypical_nets.ipynb` | Few-shot learning engines |
| `04_attention_free_architectures_mlp_mixer.ipynb` | No-attention deep nets |
| `05_spiking_neural_nets_surrogate_gradients.ipynb` | Neuromorphic computing |
| `06_diffusion_models_for_generation.ipynb` | Denoise to generate |
| `07_lab_gnn_node_classification_with_cora.ipynb` | GCN on Cora graph |
| `08_lab_memory_augmented_net_tiny_tasks.ipynb` | Repeat/copy memory tasks |
| `09_lab_diffusion_model_toy_image_gen.ipynb` | MNIST diffusion from scratch |

---

## ⚙️ 5. Model Optimization

> _Compress, quantize, and make models production-worthy._

| Notebook | Description |
|----------|-------------|
| `01_quantization_post_training_qat.ipynb` | FP32 → INT8, static & dynamic |
| `02_pruning_magnitude_optimal_brain.ipynb` | Sparse networks for speed |
| `03_knowledge_distillation_teacher_student.ipynb` | Small student, big teacher |
| `04_onnx_and_tensorrt_conversion.ipynb` | Format shifts for deployment |
| `05_tflite_and_coreml_for_mobile.ipynb` | Edge AI deployment |
| `06_mixed_precision_training.ipynb` | Faster training via FP16 |

---

## 🚀 6. Deployment & Scaling

> _Ship it to the world. Or the edge._

| Notebook | Description |
|----------|-------------|
| `01_exporting_models_onnx_torchscript_savedmodel.ipynb` | Export formats + loading |
| `02_docker_for_ml.ipynb` | Containerize with Docker |
| `02a_advanced_docker_for_ml.ipynb` | Advanced image build flows |
| `02b_serving_with_torchserve_tensorflow_serving.ipynb` | Production model serving |
| `03_edge_deployment_tflite_raspberry_pi.ipynb` | Raspberry Pi / Edge demos |
| `04_distributed_training_horovod_pytorch_ddp.ipynb` | Multi-GPU training |
| `05_monitoring_models_prometheus_grafana.ipynb` | Metrics and dashboards |
| `06_scaling_with_kubernetes_kubeflow.ipynb` | Orchestrate with K8s |

---

## 🧠 Prereqs

Before diving in, you should know:

- Python + NumPy
- Matrix math (see `02_math_for_ai`)
- Scikit-learn basics
- Git + Notebooks

---

## 🔥 Ideal For...

- Devs entering AI
- CV/NLP beginners
- MLEs deploying models
- Researchers building prototypes

> _Recommended pairing: `05_llm_engineering` if you’re going into large language models next._

---

