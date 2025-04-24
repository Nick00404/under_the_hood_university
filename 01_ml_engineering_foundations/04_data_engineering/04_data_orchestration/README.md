### **04_data_orchestration**  
**1. Airflow DAG Design**  
1.1 Core Concepts  
- DAG structure (operators, sensors, hooks)  
- Task dependencies and dynamic pipelines  

1.2 Advanced Patterns  
- Dynamic DAG generation (Jinja templating, `dag_factory`)  
- Cross-DAG communication (XComs, TriggerDagRunOperator)  

1.3 Error Handling & Observability  
- Retries with exponential backoff  
- Monitoring with Airflow’s UI and logs  

1.4 Cost-Efficient Scheduling  
- Sensor optimization (reschedule vs. poke mode)  
- Resource-aware task queues  

*Lab: Build a DAG that ingests, validates, and processes data with atomic retries.*  

---  

**2. Prefect Flow Optimization**  
2.1 Prefect 2.0 Fundamentals  
- Flows, tasks, and state handlers  
- Hybrid execution (local + cloud runners)  

2.2 Error Recovery & Observability  
- Task retries with circuit breakers  
- Integrated logging (Prometheus/Grafana)  

2.3 Performance Optimization  
- Parallel task execution (TaskRunner API)  
- Caching intermediate results  

2.4 Security & Credential Management  
- Prefect Cloud secrets  
- OAuth2 for API integrations  

*Lab: Optimize a slow legacy pipeline using Prefect’s async tasks and caching.*  

---  

**3. Dagster Asset Management**  
3.1 Software-Defined Assets  
- Defining assets and dependencies  
- Auto-materialization and freshness policies  

3.2 Cross-Pipeline Coordination  
- Asset sensors for event-driven workflows  
- Partitioned assets for incremental processing  

3.3 Data Quality Integration  
- Embedding Great Expectations checks  
- Alerting on asset validation failures  

3.4 Cost Governance  
- Tracking compute/storage costs per asset  

*Lab: Create an asset pipeline with automated validation and cost tagging.*  

---  

**4. ML Pipelines with Kubeflow**  
4.1 Pipeline Authoring Basics  
- Components (lightweight vs. containerized)  
- Input/output artifacts and type checking  

4.2 Reusable Components  
- Shared component libraries  
- Versioning components with Git  

4.3 Experiment Tracking  
- Logging metrics/parameters with MLflow  
- Artifact lineage (linked to OpenLineage)  

4.4 Testing & Validation  
- Unit testing components (no deployment)  
- Integration testing with mocked data  

*Lab: Build a reusable training pipeline with hyperparameter logging.*  

---  

### **Key Features**  
- **No Overlaps**:  
  - Avoids deployment (covered elsewhere).  
  - Data validation linked to `05_data_quality`, not reimplemented here.  
- **Cross-Module Alignment**:  
  - Git for versioning components (links to `01_git_collaboration`).  
  - Cost governance ties to `06_data_governance`.  
- **Tool-Agnostic Principles**:  
  - Concepts apply to Airflow/Prefect/Dagster.  
  - Focus on patterns (e.g., atomic retries, reusable components).  
- **Real-World Focus**:  
  - Labs emphasize measurable outcomes (e.g., "optimize slow pipeline").  
