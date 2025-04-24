# 02 Unsupervised Learning

- [01 kmeans clustering and elbow method](./01_kmeans_clustering_and_elbow_method.ipynb)
- [02 hierarchical clustering and dendrograms](./02_hierarchical_clustering_and_dendrograms.ipynb)
- [03 pca for dimensionality reduction](./03_pca_for_dimensionality_reduction.ipynb)
- [04 autoencoders for feature learning](./04_autoencoders_for_feature_learning.ipynb)
- [05 anomaly detection with isolation forests](./05_anomaly_detection_with_isolation_forests.ipynb)
- [06 manifold learning with umap tsne](./06_manifold_learning_with_umap_tsne.ipynb)

---

## 📘 **02. Unsupervised Learning – Structured Index**

---

### 🧩 **01. K-Means Clustering and Elbow Method**

#### 📌 Subtopics:
- Introduction to K-Means  
  - Centroid-based clustering  
  - Distance metrics (Euclidean, cosine, etc.)  
  - Random initialization and k-means++  
- Elbow Method for Optimal `k`  
  - Within-cluster sum of squares (WCSS)  
  - Scree plot for determining the best number of clusters  
- Example: Segmenting customer data with K-Means

---

### 🧩 **02. Hierarchical Clustering and Dendrograms**

#### 📌 Subtopics:
- Introduction to Hierarchical Clustering  
  - Agglomerative vs Divisive methods  
- Linkage Criteria  
  - Single, complete, average, Ward’s method  
- Dendrogram Interpretation  
  - Cutting the tree to form clusters  
- Example: Visualizing clustering hierarchy on Iris dataset

---

### 🧩 **03. PCA for Dimensionality Reduction**

#### 📌 Subtopics:
- What is PCA?  
  - Covariance matrix, eigenvectors, and eigenvalues  
- Explained Variance  
  - Choosing the number of components  
  - Scree plot analysis  
- PCA vs Feature Selection  
  - When to reduce dimensions vs selecting features  
- Example: Applying PCA on high-dimensional image data

---

### 🧩 **04. Autoencoders for Feature Learning**

#### 📌 Subtopics:
- Autoencoder Architecture  
  - Encoder-decoder structure  
  - Bottleneck layers and latent space  
- Loss Functions  
  - Reconstruction loss (MSE, BCE)  
- Use Cases  
  - Dimensionality reduction, denoising  
- Example: Learning compressed representations of MNIST digits

---

### 🧩 **05. Anomaly Detection with Isolation Forests**

#### 📌 Subtopics:
- Isolation Forest Algorithm  
  - Random partitioning, path length  
  - Scoring outliers via isolation depth  
- Comparison with Other Methods  
  - One-Class SVM, LOF  
- Use Cases  
  - Fraud detection, system monitoring  
- Example: Detecting rare transactions in financial data

---

### 🧩 **06. Manifold Learning with UMAP and t-SNE**

#### 📌 Subtopics:
- What is Manifold Learning?  
  - Non-linear dimensionality reduction  
- t-SNE  
  - Pairwise similarity and perplexity  
  - High-dimensional to 2D embeddings  
- UMAP  
  - Topological structure preservation  
  - Performance and interpretability  
- Example: Visualizing complex embeddings from NLP models

---

Add-On | Notes
🔬 LOF / One-Class SVM Labs | Only compared conceptually, not implemented — optional to add lab/code notebooks
🎯 Clustering Performance Metrics | Like Silhouette Score, Davies-Bouldin, Calinski-Harabasz — could be a small appendix notebook
📊 Using Dimensionality Reduction before Clustering | (e.g., PCA → KMeans, UMAP → DBSCAN) — practical combo patterns
🧪 End-to-End Capstone | Real-world dataset: customer data, medical anomalies, etc. (e.g., UMAP + KMeans + Isolation Forest in pipeline)
🧠 DBSCAN or HDBSCAN | Optional: density-based clustering — super useful but more niche than KMeans/hierarchical


















You're building a seriously clean and comprehensive index 🔥 — let’s keep that energy going. Here's the **TOC with anchor links** and the **section tags** with emojis and clean anchor IDs, all formatted for easy use in a Jupyter Notebook.

---

## ✅ Table of Contents (with anchors)

```markdown
## 🧭 Table of Contents – Unsupervised Learning

### 🧩 [01. K-Means Clustering and Elbow Method](#kmeans)
- 📍 [Introduction to K-Means](#kmeans-intro)
- 📏 [Elbow Method for Optimal `k`](#kmeans-elbow)
- 🧪 [Example: Customer Segmentation](#kmeans-example)

### 🧩 [02. Hierarchical Clustering and Dendrograms](#hierarchical)
- 🧱 [Introduction to Hierarchical Clustering](#hierarchical-intro)
- 🔗 [Linkage Criteria](#hierarchical-linkage)
- 🌲 [Dendrogram Interpretation](#hierarchical-dendrogram)
- 🧪 [Example: Iris Clustering](#hierarchical-example)

### 🧩 [03. PCA for Dimensionality Reduction](#pca)
- 🧠 [What is PCA?](#pca-intro)
- 📊 [Explained Variance](#pca-variance)
- 🔍 [PCA vs Feature Selection](#pca-selection)
- 🧪 [Example: Image Compression](#pca-example)

### 🧩 [04. Autoencoders for Feature Learning](#autoencoders)
- 🧬 [Autoencoder Architecture](#ae-architecture)
- 💥 [Loss Functions](#ae-loss)
- 🛠️ [Use Cases](#ae-use-cases)
- 🧪 [Example: MNIST Representations](#ae-example)

### 🧩 [05. Anomaly Detection with Isolation Forests](#anomaly)
- 🌲 [Isolation Forest Algorithm](#isolation-forest)
- 🥊 [Comparison with Other Methods](#anomaly-compare)
- 🚨 [Use Cases](#anomaly-use-cases)
- 🧪 [Example: Fraud Detection](#anomaly-example)

### 🧩 [06. Manifold Learning with UMAP and t-SNE](#manifold)
- 🌐 [What is Manifold Learning?](#manifold-intro)
- 🌀 [t-SNE](#tsne)
- 🌈 [UMAP](#umap)
- 🧪 [Example: NLP Embeddings](#manifold-example)
```

---

## 🧩 Section Headings with Anchor Tags

```markdown
### 🧩 <a id="kmeans"></a>01. K-Means Clustering and Elbow Method

#### <a id="kmeans-intro"></a>📍 Introduction to K-Means  
- Centroid-based clustering  
- Distance metrics (Euclidean, cosine, etc.)  
- Random initialization and k-means++  

#### <a id="kmeans-elbow"></a>📏 Elbow Method for Optimal `k`  
- Within-cluster sum of squares (WCSS)  
- Scree plot for determining the best number of clusters  

#### <a id="kmeans-example"></a>🧪 Example: Segmenting customer data with K-Means  

---

### 🧩 <a id="hierarchical"></a>02. Hierarchical Clustering and Dendrograms

#### <a id="hierarchical-intro"></a>🧱 Introduction to Hierarchical Clustering  
- Agglomerative vs Divisive methods  

#### <a id="hierarchical-linkage"></a>🔗 Linkage Criteria  
- Single, complete, average, Ward’s method  

#### <a id="hierarchical-dendrogram"></a>🌲 Dendrogram Interpretation  
- Cutting the tree to form clusters  

#### <a id="hierarchical-example"></a>🧪 Example: Visualizing clustering hierarchy on Iris dataset  

---

### 🧩 <a id="pca"></a>03. PCA for Dimensionality Reduction

#### <a id="pca-intro"></a>🧠 What is PCA?  
- Covariance matrix, eigenvectors, and eigenvalues  

#### <a id="pca-variance"></a>📊 Explained Variance  
- Choosing the number of components  
- Scree plot analysis  

#### <a id="pca-selection"></a>🔍 PCA vs Feature Selection  
- When to reduce dimensions vs selecting features  

#### <a id="pca-example"></a>🧪 Example: Applying PCA on high-dimensional image data  

---

### 🧩 <a id="autoencoders"></a>04. Autoencoders for Feature Learning

#### <a id="ae-architecture"></a>🧬 Autoencoder Architecture  
- Encoder-decoder structure  
- Bottleneck layers and latent space  

#### <a id="ae-loss"></a>💥 Loss Functions  
- Reconstruction loss (MSE, BCE)  

#### <a id="ae-use-cases"></a>🛠️ Use Cases  
- Dimensionality reduction, denoising  

#### <a id="ae-example"></a>🧪 Example: Learning compressed representations of MNIST digits  

---

### 🧩 <a id="anomaly"></a>05. Anomaly Detection with Isolation Forests

#### <a id="isolation-forest"></a>🌲 Isolation Forest Algorithm  
- Random partitioning, path length  
- Scoring outliers via isolation depth  

#### <a id="anomaly-compare"></a>🥊 Comparison with Other Methods  
- One-Class SVM, LOF  

#### <a id="anomaly-use-cases"></a>🚨 Use Cases  
- Fraud detection, system monitoring  

#### <a id="anomaly-example"></a>🧪 Example: Detecting rare transactions in financial data  

---

### 🧩 <a id="manifold"></a>06. Manifold Learning with UMAP and t-SNE

#### <a id="manifold-intro"></a>🌐 What is Manifold Learning?  
- Non-linear dimensionality reduction  

#### <a id="tsne"></a>🌀 t-SNE  
- Pairwise similarity and perplexity  
- High-dimensional to 2D embeddings  

#### <a id="umap"></a>🌈 UMAP  
- Topological structure preservation  
- Performance and interpretability  

#### <a id="manifold-example"></a>🧪 Example: Visualizing complex embeddings from NLP models  
```

---

Let me know if you'd like all of this dumped into a `.ipynb` file, or want a script that auto-generates TOC + anchors from a JSON or Markdown topic structure. Would save you a ton of repetitive work next time!