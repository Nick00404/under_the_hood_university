### **05_data_quality**  
**1. Data Profiling with Pandas**  
1.1 Core Concepts  
- Summary statistics, distributions, and missing value analysis  
- Cardinality checks and pattern recognition (e.g., email formats)  

1.2 Advanced Profiling Techniques  
- Correlation analysis for numerical/categorical data  
- Temporal profiling (seasonality, trend detection)  

1.3 Scalability Considerations  
- Profiling large datasets with `pandas-profiling` or `Dask`  
- Sampling strategies for iterative analysis  

*Lab: Profile a messy dataset and identify 5 critical quality issues.*  

---  

**2. Data Validation with Great Expectations & Pandera**  
2.1 Validation Fundamentals  
- Schema vs. statistical validation  
- Batch-aware validation (e.g., row count thresholds)  

2.2 Great Expectations in Practice  
- Suite-based expectations (e.g., `expect_column_values_to_be_unique`)  
- Automated documentation and data docs  

2.3 Pandera for DataFrame Validation  
- Statistical assertions (e.g., `Check.mean() > 0`)  
- Integration with Pandas/Spark workflows  

2.4 CI/CD Integration  
- Blocking pipeline execution on validation failures  
- Alerting via Slack/email  

*Lab: Validate a production dataset and generate a data quality report.*  

---  

**3. Statistical Anomaly Detection**  
3.1 Threshold-Based Methods  
- Z-scores, IQR, and rolling averages  
- Domain-specific thresholds (e.g., sensor error bounds)  

3.2 Time-Series Anomalies  
- STL decomposition for trend/seasonality  
- Change point detection (CUSUM, Prophet)  

3.3 Root Cause Analysis  
- Triaging anomalies (data drift vs. pipeline failures)  
- Linking anomalies to lineage/metadata  

*Lab: Detect and diagnose anomalies in a financial transaction dataset.*  

---  

**4. Data Observability with Monte Carlo**  
4.1 Observability Pillars  
- Freshness, volume, distribution, schema, lineage  
- Column-level health scoring  

4.2 Automated Monitoring  
- Setting up detectors for schema drift  
- Correlating incidents across pipelines  

4.3 Cost of Poor Data Quality  
- Quantifying downtime (e.g., $/hour of broken pipelines)  
- SLA tracking for data products  

*Lab: Set up end-to-end observability for a critical pipeline.*  

---  

**5. Data Lineage with OpenLineage & Marquez**  
5.1 Lineage Fundamentals  
- Code-to-data lineage (Airflow tasks, Spark jobs)  
- Impact analysis for schema/dataset changes  

5.2 OpenLineage Integrations  
- Extracting lineage from Airflow/Dagster/Spark  
- Visualizing lineage graphs  

5.3 Marquez for Metadata  
- Dataset versioning and lifecycle tracking  
- Querying lineage via API/UI  

*Lab: Trace a broken dashboard value back to its root cause using lineage.*  

---  

**6. Data Cleaning & Deduplication**  
6.1 Cleaning Patterns  
- Standardization (dates, currencies, units)  
- Imputation strategies (mean, forward-fill, ML-based)  

6.2 Deduplication Techniques  
- Exact matching (hashing) vs. fuzzy matching (Levenshtein)  
- Rule-based vs. ML-driven approaches  

6.3 Pipeline Integration  
- Cleaning as a pipeline step (pre- vs. post-validation)  
- Versioning cleaned datasets  

*Lab: Clean and deduplicate a messy customer database.*  

---  

**7. Pipeline Testing Strategies**  
7.1 Test Types  
- Unit tests (individual functions/transformations)  
- Integration tests (end-to-end pipeline runs)  

7.2 Testing Frameworks  
- `pytest` for Python-based pipelines  
- dbt tests for SQL transformations  

7.3 Mocking & Fixtures  
- Generating synthetic test data  
- Mocking external APIs/databases  

7.4 CI/CD Integration  
- Automated testing in GitHub Actions/GitLab CI  
- Code coverage metrics  

*Lab: Build a test suite for a PySpark ETL pipeline.*  

---  

**8. Metadata Management**  
8.1 Metadata Types  
- Technical (schema, lineage) vs. business (ownership, SLAs)  

8.2 Amundsen Integration  
- Indexing datasets and dashboards  
- Search relevance tuning (usage-based ranking)  

8.3 Active Metadata  
- Triggering pipelines on metadata changes (e.g., schema updates)  
- Feedback loops (e.g., tagging low-quality datasets)  

*Lab: Build a searchable data catalog for a fintech dataset.*  

---  
