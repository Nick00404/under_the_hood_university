# 04 Model Evaluation

- [01 metrics for imbalanced datasets](./01_metrics_for_imbalanced_datasets.ipynb)
- [02 cross validation strategies](./02_cross_validation_strategies.ipynb)
- [03 bias variance analysis](./03_bias_variance_analysis.ipynb)
- [04 model interpretability with shap lime](./04_model_interpretability_with_shap_lime.ipynb)
- [05 statistical tests for model comparison](./05_statistical_tests_for_model_comparison.ipynb)
- [06 ml model auditing and fairness](./06_ml_model_auditing_and_fairness.ipynb)

---

## 📘 ML Evaluation and Fairness – Structured Index

---

### 🧩 **01. Metrics for Imbalanced Datasets**

#### 📌 Subtopics:
- **Understanding Imbalanced Datasets**
  - Definition and importance of addressing imbalance
  - Impact of imbalanced datasets on model performance
- **Evaluation Metrics for Imbalanced Data**
  - Precision, Recall, F1-Score, and their importance
  - Area Under the Precision-Recall Curve (PR AUC)
  - F2 Score vs F1 Score for specific use cases
- **Resampling Techniques**
  - Oversampling (SMOTE) vs Undersampling
  - Using class weights to balance datasets
  - How to adjust metrics during training with imbalanced data

---

### 🧩 **02. Cross-Validation Strategies**

#### 📌 Subtopics:
- **K-Fold Cross-Validation**
  - What it is and how it works
  - Benefits and limitations of K-Fold
  - How K is selected for different types of models
- **Stratified K-Fold Cross-Validation**
  - Importance for imbalanced datasets
  - How stratification ensures proper class distribution
  - When to use vs regular K-Fold
- **Leave-One-Out and Other Cross-Validation Methods**
  - LOOCV and its computational cost
  - Time Series Cross-Validation and its requirements
  - Nested Cross-Validation for hyperparameter tuning

---

### 🧩 **03. Bias-Variance Analysis**

#### 📌 Subtopics:
- **Bias-Variance Tradeoff**
  - What is bias and variance?
  - How does the tradeoff affect model performance?
  - Visualizing bias and variance with model complexity
- **Types of Bias and Variance**
  - High bias (underfitting) vs low bias
  - High variance (overfitting) vs low variance
  - Diagnosing the model's errors with the bias-variance curve
- **Mitigating Bias and Variance**
  - Techniques to reduce bias: simpler models, adding features
  - Techniques to reduce variance: regularization, more data
  - The role of ensemble methods in balancing bias and variance

---

### 🧩 **04. Model Interpretability with SHAP & LIME**

#### 📌 Subtopics:
- **SHAP (SHapley Additive exPlanations)**
  - What SHAP values are and why they are used
  - SHAP’s connection to cooperative game theory
  - How SHAP explains model predictions globally and locally
- **LIME (Local Interpretable Model-agnostic Explanations)**
  - Introduction to LIME and its model-agnostic nature
  - Explaining individual predictions with LIME
  - Using LIME with black-box models (e.g., deep neural networks)
- **Comparing SHAP and LIME**
  - Key differences and when to use one over the other
  - Pros and cons for different types of models
  - Visualizations with SHAP and LIME (force plots, decision boundaries)

---

### 🧩 **05. Statistical Tests for Model Comparison**

#### 📌 Subtopics:
- **Chi-Square Test for Model Comparison**
  - Using Chi-square to test goodness-of-fit
  - Comparing categorical model outputs
  - Understanding p-values and statistical significance
- **Paired t-Test vs. Wilcoxon Signed-Rank Test**
  - When to use parametric vs non-parametric tests
  - Comparing two models on the same dataset
  - Understanding the assumptions of each test
- **ANOVA (Analysis of Variance)**
  - How ANOVA can compare multiple models
  - F-test and its use in hypothesis testing
  - One-way vs two-way ANOVA in model comparison

---

### 🧩 **06. ML Model Auditing and Fairness**

#### 📌 Subtopics:
- **Auditing ML Models for Fairness**
  - What constitutes fairness in machine learning
  - Bias detection techniques in model outcomes
  - Auditing model predictions for equality of opportunity
- **Fairness Metrics and Their Use**
  - Demographic Parity, Equalized Odds, and Disparate Impact
  - Evaluating model fairness with statistical tests
  - Applying fairness constraints to models
- **Ethical Considerations and Model Impact**
  - The societal impact of biased predictions
  - The importance of explainable AI in fairness auditing
  - Real-world applications: Hiring algorithms, Credit scoring, and Healthcare models

---

### 🧠 Bonus:
- Practical examples and code snippets to implement each fairness metric
- Visualizations for bias-variance tradeoff and model comparison
- Case studies of real-world ML fairness audits (e.g., facial recognition systems)

---
