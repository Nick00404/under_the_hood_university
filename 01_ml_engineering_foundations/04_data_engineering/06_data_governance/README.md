### **06_data_governance**  
**1. RBAC & Access Control**  
1.1 Core Principles  
- Least privilege and role inheritance hierarchies  
- Column/row-level security (BigQuery, Snowflake)  

1.2 Dynamic Access Policies  
- Attribute-based access control (ABAC)  
- Just-in-Time (JIT) access for temporary privileges  

1.3 Audit & Compliance  
- Access log analysis (e.g., suspicious IP detection)  
- Automated revocation of stale permissions  

*Lab: Implement column-level masking for sensitive PII in Snowflake.*  

---  

**2. Data Cataloging with Amundsen & Atlan**  
2.1 Metadata Discovery  
- Search relevance tuning (usage frequency, freshness)  
- Business glossary integration (e.g., GDPR data classes)  

2.2 Collaboration Features  
- Dataset annotations and user feedback loops  
- Data stewardship workflows  

2.3 Cost Governance  
- Tagging datasets with ownership/usage costs  
- Deprecating unused resources  

*Lab: Build a searchable catalog for a healthcare dataset with GDPR tagging.*  

---  

**3. Compliance (GDPR, CCPA)**  
3.1 Regulatory Fundamentals  
- Data Subject Access Requests (DSAR) automation  
- Consent management (opt-in/opt-out tracking)  

3.2 Technical Implementation  
- Automated data discovery for PII/PHI  
- Right to erasure workflows  

3.3 Cross-Border Data Transfers  
- SCCs (Standard Contractual Clauses) enforcement  
- Geo-fencing with cloud storage policies  

*Lab: Process a DSAR request end-to-end (identify → redact → confirm).*  

---  

**4. PII Redaction with Presidio**  
4.1 Redaction Strategies  
- Pattern matching (SSN, credit cards)  
- Custom NER models for domain-specific PII  

4.2 Batch vs. Streaming Redaction  
- Tradeoffs in latency vs. completeness  
- Integration with Spark/Flink  

4.3 Validation & Testing  
- Measuring redaction recall/precision  
- False positive analysis  

*Lab: Redact PII from unstructured clinical notes using Presidio.*  

---  

**5. Data Masking & Tokenization**  
5.1 Static vs. Dynamic Masking  
- Prod vs. non-prod environment strategies  
- Format-preserving encryption (FPE)  

5.2 Tokenization Patterns  
- Vaultless tokenization (stateless mapping)  
- PCI-DSS compliance for payment data  

5.3 Re-Identification Risks  
- K-anonymity and l-diversity checks  

*Lab: Tokenize a credit card dataset while preserving format.*  

---  

**6. Data Retention Policies**  
6.1 Legal & Operational Needs  
- Balancing compliance vs. analytics value  
- Legal hold workflows (conflict resolution)  

6.2 Technical Enforcement  
- Lifecycle policies (S3, BigQuery)  
- Immutable backups (WORM compliance)  

6.3 Archival Strategies  
- Cold storage tiering (Glacier, Tape)  
- Metadata retention for deleted datasets  

*Lab: Implement a 7-year retention policy for financial records.*  

---  


**7. Data Security & Encryption**  
7.1 Encryption Strategies  
- BYOK (Bring Your Own Key) vs. cloud-managed keys  
- TLS 1.3 for in-flight data  

7.2 Secrets Management  
- Vault integration (HashiCorp, AWS Secrets Manager)  
- Credential rotation automation  

7.3 Data Residency  
- Geo-redundancy tradeoffs  
- Sovereignty controls (e.g., EU-only storage)  

*Lab: Encrypt a dataset end-to-end (at-rest → in-flight → processing).*  

---  

**8. Data Contracts Specification**  
8.1 Contract Components  
- Schema, SLAs, ownership, lineage  
- Versioning and backward compatibility  

8.2 Automated Enforcement  
- CI/CD validation (Great Expectations, dbt tests)  
- Breaking change communication  

8.3 Cross-Team Collaboration  
- Contract negotiation frameworks  
- Self-service contract templates  

*Lab: Enforce a data contract in CI/CD using GitHub Actions.*  

---  