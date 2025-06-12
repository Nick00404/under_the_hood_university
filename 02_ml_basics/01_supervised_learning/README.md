# 01 Supervised Learning

- [01 linear regression and cost functions](./01_linear_regression_and_cost_functions.ipynb)
- [02 logistic regression and classification metrics](./02_logistic_regression_and_classification_metrics.ipynb)
- [03 decision trees and ensemble methods](./03_decision_trees_and_ensemble_methods.ipynb)
- [04 svm and kernel tricks for nonlinear data](./04_svm_and_kernel_tricks_for_nonlinear_data.ipynb)
- [05 regularization l1 l2 elasticnet](./05_regularization_l1_l2_elasticnet.ipynb)
- [06 bayesian models and naive bayes](./06_bayesian_models_and_naive_bayes.ipynb)



## 📘 `01_linear_regression_and_cost_functions.ipynb`

### 🧩 **1. Building Blocks of Linear Regression**
- Hypothesis Function
- Line Fitting (Geometric intuition)
- Assumptions of Linear Models

### 🧩 **2. Cost Function & Optimization**
- Squared Error / MSE
- Gradient Descent (Single variable)
- Gradient Descent (Multivariable)
- Vectorization for Speedup

### 🧩 **3. Evaluation & Interpretation**
- R² Score
- Underfitting & Model Diagnostics
- Visualizing Cost Surface

---

## 📘 `02_logistic_regression_and_classification_metrics.ipynb`

### 🧩 **1. Understanding Logistic Regression**
- Binary Classification Motivation
- Sigmoid Function & Probability Output
- Decision Boundary Interpretation

### 🧩 **2. Training the Model**
- Cost Function for Logistic Regression
- Gradient Descent for Logistic Regression
- Feature Scaling

### 🧩 **3. Evaluation & Performance**
- Accuracy, Precision, Recall, F1
- Confusion Matrix
- ROC Curve & AUC
- Overfitting in Classification Models

---

## 📘 `03_decision_trees_and_ensemble_methods.ipynb`

### 🧩 **1. Decision Trees Explained**
- Splitting Criteria: Gini vs Entropy
- Tree Depth & Pruning
- Overfitting in Trees

### 🧩 **2. Ensemble Techniques**
- Bagging & Random Forests
- Feature Importance
- Boosting Basics (GBM, XGBoost)

### 🧩 **3. Model Tuning & Comparison**
- Hyperparameters for Forests & Boosters
- When to Use Trees vs Linear Models
- Bias-Variance Tradeoff Visualized

---

## 📘 `04_svm_and_kernel_tricks_for_nonlinear_data.ipynb`

### 🧩 **1. Core Concepts of SVM**
- Max-Margin Intuition
- Hard vs Soft Margins
- Hinge Loss Function

### 🧩 **2. Going Nonlinear with Kernels**
- Polynomial & RBF Kernels
- Visualizing Transformations
- Kernelized Decision Boundaries

### 🧩 **3. Practical Usage**
- Parameter Tuning: C and Gamma
- Linear vs Non-linear SVM
- Comparison with Logistic Regression

---

## 📘 `05_regularization_l1_l2_elasticnet.ipynb`

### 🧩 **1. Motivation & Math of Regularization**
- Overfitting Intuition
- Regularized Cost Functions
- Effect of λ on Loss

### 🧩 **2. Types of Regularization**
- L2 (Ridge)
- L1 (Lasso)
- ElasticNet (Combining L1 + L2)

### 🧩 **3. Practical Model Fitting**
- Regularization in Scikit-Learn
- Cross-Validation for λ Selection
- Visual Demos: Shrinkage of Weights

---

## 📘 `06_bayesian_models_and_naive_bayes.ipynb`

### 🧩 **1. Foundations of Bayesian Thinking**
- Bayes Theorem Refresher
- Likelihood vs Prior vs Posterior
- Probabilistic Classification Intuition

### 🧩 **2. Naive Bayes Classifiers**
- Gaussian, Multinomial, Bernoulli
- Conditional Independence Assumption
- When Naive Bayes Works Well

### 🧩 **3. Evaluation & Usage**
- Use Cases (Spam, Sentiment, etc.)
- Comparison to Logistic Regression
- Performance on Imbalanced Data

---





















## 🧭 Master Table of Contents



---

### 📘 [03 Decision Trees & Ensemble Methods](#trees-explained)
- 🌳 [Decision Trees Explained](#trees-explained)
  - 🧪 [Splitting Criteria: Gini vs Entropy](#splitting-criteria)
  - ✂️ [Tree Depth & Pruning](#pruning)
  - 🚨 [Overfitting in Trees](#overfitting-trees)
- 🌲 [Ensemble Techniques](#ensemble-techniques)
  - 🧺 [Bagging & Random Forests](#bagging-forests)
  - ⭐ [Feature Importance](#feature-importance)
  - 🚀 [Boosting Basics (GBM, XGBoost)](#boosting)
- 🧪 [Model Tuning & Comparison](#model-tuning)
  - ⚙️ [Hyperparameters for Forests & Boosters](#hyperparameters)
  - 🔍 [When to Use Trees vs Linear Models](#trees-vs-linear)
  - 📉 [Bias-Variance Tradeoff Visualized](#bias-variance)

---

### 📘 [04 SVM & Kernel Tricks for Nonlinear Data](#svm-core)
- 🧭 [Core Concepts of SVM](#svm-core)
  - 📐 [Max-Margin Intuition](#max-margin)
  - 🧊 [Hard vs Soft Margins](#hard-soft)
  - 📏 [Hinge Loss Function](#hinge-loss)
- 🌌 [Going Nonlinear with Kernels](#svm-kernels)
  - 🧮 [Polynomial & RBF Kernels](#kernel-types)
  - 🔍 [Visualizing Transformations](#kernel-visual)
  - 🚧 [Kernelized Decision Boundaries](#kernel-boundaries)
- 🧰 [Practical Usage](#svm-practical)
  - ⚙️ [Parameter Tuning: C and Gamma](#svm-tuning)
  - 🔀 [Linear vs Non-linear SVM](#svm-linear-vs-nonlinear)
  - 🔄 [Comparison with Logistic Regression](#svm-vs-logistic)

---

### 📘 [05 Regularization (L1, L2, ElasticNet)](#regularization-motivation)
- 📉 [Motivation & Math of Regularization](#regularization-motivation)
  - 🧠 [Overfitting Intuition](#overfitting-intuition)
  - 📊 [Regularized Cost Functions](#regularized-cost)
  - 📉 [Effect of λ on Loss](#lambda-effect)
- 🧮 [Types of Regularization](#regularization-types)
  - 🏔️ [L2 (Ridge)](#ridge)
  - 🎯 [L1 (Lasso)](#lasso)
  - 🧷 [ElasticNet (Combining L1 + L2)](#elasticnet)
- 🛠️ [Practical Model Fitting](#regularization-practical)
  - 🧰 [Regularization in Scikit-Learn](#sklearn-regularization)
  - 🔁 [Cross-Validation for λ Selection](#cv-lambda)
  - 🎨 [Visual Demos: Shrinkage of Weights](#shrinkage-visuals)

---

### 📘 [06 Bayesian Models & Naive Bayes](#bayesian-foundations)
- 🧠 [Foundations of Bayesian Thinking](#bayesian-foundations)
  - 📚 [Bayes Theorem Refresher](#bayes-theorem)
  - 🧪 [Likelihood vs Prior vs Posterior](#likelihood-prior)
  - 🎲 [Probabilistic Classification Intuition](#probabilistic-intuition)
- 🐦 [Naive Bayes Classifiers](#naive-bayes)
  - 📊 [Gaussian, Multinomial, Bernoulli](#nb-types)
  - 🔗 [Conditional Independence Assumption](#independence-assumption)
  - ✅ [When Naive Bayes Works Well](#nb-use-cases)
- 📈 [Evaluation & Usage](#bayesian-evaluation)
  - ✉️ [Use Cases (Spam, Sentiment, etc.)](#nb-applications)
  - ⚔️ [Comparison to Logistic Regression](#nb-vs-logistic)
  - ⚖️ [Performance on Imbalanced Data](#imbalanced-performance)




















---

## 📘 `03_decision_trees_and_ensemble_methods.ipynb`

### 🌳 <a id="trees-explained"></a>**1. Decision Trees Explained**
#### <a id="splitting-criteria"></a>🧪 Splitting Criteria: Gini vs Entropy  
#### <a id="pruning"></a>✂️ Tree Depth & Pruning  
#### <a id="overfitting-trees"></a>🚨 Overfitting in Trees  

### 🌲 <a id="ensemble-techniques"></a>**2. Ensemble Techniques**
#### <a id="bagging-forests"></a>🧺 Bagging & Random Forests  
#### <a id="feature-importance"></a>⭐ Feature Importance  
#### <a id="boosting"></a>🚀 Boosting Basics (GBM, XGBoost)  

### 🧪 <a id="model-tuning"></a>**3. Model Tuning & Comparison**
#### <a id="hyperparameters"></a>⚙️ Hyperparameters for Forests & Boosters  
#### <a id="trees-vs-linear"></a>🔍 When to Use Trees vs Linear Models  
#### <a id="bias-variance"></a>📉 Bias-Variance Tradeoff Visualized  

---

## 📘 `04_svm_and_kernel_tricks_for_nonlinear_data.ipynb`

### 🧭 <a id="svm-core"></a>**1. Core Concepts of SVM**
#### <a id="max-margin"></a>📐 Max-Margin Intuition  
#### <a id="hard-soft"></a>🧊 Hard vs Soft Margins  
#### <a id="hinge-loss"></a>📏 Hinge Loss Function  

### 🌌 <a id="svm-kernels"></a>**2. Going Nonlinear with Kernels**
#### <a id="kernel-types"></a>🧮 Polynomial & RBF Kernels  
#### <a id="kernel-visual"></a>🔍 Visualizing Transformations  
#### <a id="kernel-boundaries"></a>🚧 Kernelized Decision Boundaries  

### 🧰 <a id="svm-practical"></a>**3. Practical Usage**
#### <a id="svm-tuning"></a>⚙️ Parameter Tuning: C and Gamma  
#### <a id="svm-linear-vs-nonlinear"></a>🔀 Linear vs Non-linear SVM  
#### <a id="svm-vs-logistic"></a>🔄 Comparison with Logistic Regression  

---

## 📘 `05_regularization_l1_l2_elasticnet.ipynb`

### 📉 <a id="regularization-motivation"></a>**1. Motivation & Math of Regularization**
#### <a id="overfitting-intuition"></a>🧠 Overfitting Intuition  
#### <a id="regularized-cost"></a>📊 Regularized Cost Functions  
#### <a id="lambda-effect"></a>📉 Effect of λ on Loss  

### 🧮 <a id="regularization-types"></a>**2. Types of Regularization**
#### <a id="ridge"></a>🏔️ L2 (Ridge)  
#### <a id="lasso"></a>🎯 L1 (Lasso)  
#### <a id="elasticnet"></a>🧷 ElasticNet (Combining L1 + L2)  

### 🛠️ <a id="regularization-practical"></a>**3. Practical Model Fitting**
#### <a id="sklearn-regularization"></a>🧰 Regularization in Scikit-Learn  
#### <a id="cv-lambda"></a>🔁 Cross-Validation for λ Selection  
#### <a id="shrinkage-visuals"></a>🎨 Visual Demos: Shrinkage of Weights  

---

## 📘 `06_bayesian_models_and_naive_bayes.ipynb`

### 🧠 <a id="bayesian-foundations"></a>**1. Foundations of Bayesian Thinking**
#### <a id="bayes-theorem"></a>📚 Bayes Theorem Refresher  
#### <a id="likelihood-prior"></a>🧪 Likelihood vs Prior vs Posterior  
#### <a id="probabilistic-intuition"></a>🎲 Probabilistic Classification Intuition  

### 🐦 <a id="naive-bayes"></a>**2. Naive Bayes Classifiers**
#### <a id="nb-types"></a>📊 Gaussian, Multinomial, Bernoulli  
#### <a id="independence-assumption"></a>🔗 Conditional Independence Assumption  
#### <a id="nb-use-cases"></a>✅ When Naive Bayes Works Well  

### 📈 <a id="bayesian-evaluation"></a>**3. Evaluation & Usage**
#### <a id="nb-applications"></a>✉️ Use Cases (Spam, Sentiment, etc.)  
#### <a id="nb-vs-logistic"></a>⚔️ Comparison to Logistic Regression  
#### <a id="imbalanced-performance"></a>⚖️ Performance on Imbalanced Data  
---

