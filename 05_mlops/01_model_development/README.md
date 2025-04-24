
### **1. Feature Store Design & Management**

**1.1 Architectural Patterns**  
- Centralized vs. decentralized feature store architectures  
- Online + offline store bifurcation (e.g., Redis + Parquet)  
- Feature freshness and point-in-time correctness  

**1.2 Feast vs. Custom Pipelines**  
- Feast registry, provider setup, materialization flow  
- Custom store implementations using DB/warehouse + scheduler  
- Tradeoffs: abstraction vs. flexibility, scaling, governance  

**1.3 Consistency Guarantees**  
- Online/offline training-serving skew mitigation  
- Feature versioning strategies  
- TTLs, backfilling, and late-arriving data handling  

*Lab: Build a time-aware feature store using Feast + Redis/PostgreSQL.*

---

### **2. Data Versioning & Lineage**

**2.1 Reproducibility Principles**  
- Deterministic pipelines & lineage tracing  
- Dataset immutability and fingerprinting  

**2.2 Versioning with DVC**  
- Remote backend setup (S3, GCS, Azure)  
- DVC pipelines (`dvc.yaml`, `params.yaml`) for DAG definition  
- `dvc diff`, checksum validation, and CI integration  

**2.3 Lineage Integration**  
- Linking DVC lineage to ML metadata (MLflow, W&B)  
- Hash consistency across pipeline steps  

*Lab: Reproduce an ML experiment from versioned data + model inputs.*

---

### **3. Experiment Tracking & Comparability**

**3.1 Experiment Logging Foundations**  
- Unified model metadata: hyperparams, code, data, env  
- Best practices for consistent metric tracking  

**3.2 MLflow Implementation**  
- Local tracking server, UI usage, artifact logging  
- Model Registry and stage transitions (Staging → Prod)  
- Auto-logging for scikit-learn, TensorFlow, etc.  

**3.3 Weights & Biases (W&B)**  
- Advanced dashboarding, system metrics, alerting  
- Experiment grouping, collaboration workflows  
- Artifact versioning for models and datasets  

*Lab: Track and compare model runs across MLflow and W&B for the same experiment.*

---

### **4. Model Versioning Strategies**

**4.1 Lifecycle Mapping**  
- Checkpointing during training vs. full pipeline versioning  
- Snapshotting preprocessing + model bundle together  

**4.2 Manual vs. Automated Versioning**  
- Git tag–based snapshots  
- Automated CI/CD-driven version bumps  

**4.3 Integration with Registries**  
- Model cards and metadata  
- Linking to CI workflows (e.g., test → promote → deploy triggers)  

*Lab: Register two versions of a model and automate promotion via GitHub Actions.*

---

### **5. Environment Reproducibility**

**5.1 Dependency Management**  
- Conda environments (`environment.yml`, lockfiles)  
- Pip-tools and `pyproject.toml` workflows  
- Environment pinning for hardware-specific builds  

**5.2 Dockerization for ML**  
- Image layering, Conda + pip combo in Docker  
- Deterministic builds using hashable Dockerfiles  
- Caching strategies and multi-stage Docker builds  

**5.3 Reproducibility Hashing**  
- Validating run environments using hash digests  
- Reproducing model artifacts from recorded config  

*Lab: Build and run the same experiment in two different environments (local + Docker) with identical results.*

---

### **6. Integration & Governance Foundations**  

**6.1 ML Governance Readiness**  
- Audit trails for data, experiments, and model versions  
- Source-to-prediction lineage for enterprise audits  

**6.2 Cross-Tool Integration**  
- Linking DVC + MLflow + Docker for reproducible, deployable workflows  
- Managing experiment metadata in Git-friendly formats  

**6.3 Pre-production Checklist**  
- Environment parity checklist (dev, staging, prod)  
- Model release documentation with data + env refs  

*Lab: Generate a reproducibility report (data, code, env, metrics) for regulatory compliance.*

---

### ✳️ **Key Principles**  

- **Environment-First Thinking**: Every model output is a function of data, code, environment.  
- **Governance from Day 1**: Build with auditability, not as an afterthought.  
- **Tool-Agnostic Patterns**: Core ideas apply across MLflow, W&B, DVC, Feast, etc.  

---
