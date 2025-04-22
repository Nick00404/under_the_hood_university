### **01_data_ingestion**  
**1. Batch Ingestion with NiFi & Luigi**  
1.1 Core Concepts  
- Batch vs. streaming ingestion patterns  
- When to use NiFi (GUI-based workflows) vs. Luigi (Python-centric pipelines)  

1.2 NiFi for Data Flow Automation  
- Processors for file, database, and API ingestion  
- Handling backpressure and prioritization  

1.3 Luigi for Task Orchestration  
- Task dependencies and parameterization  
- Atomic writes and idempotent workflows  

1.4 Real-World Scenarios  
- Incremental batch loading (e.g., daily CSV dumps)  
- Handling large file splits and compression  

*Lab: Build a fault-tolerant batch pipeline for incremental CSV ingestion.*  

---  

**2. Streaming Ingestion with Kafka & Pulsar**  
2.1 Event-Driven Architecture Basics  
- Topics, partitions, and consumer groups  
- Exactly-once semantics vs. at-least-once delivery  

2.2 Kafka for Real-Time Streams  
- Schema registry integration (Avro/Protobuf)  
- Scaling consumers with consumer groups  

2.3 Pulsar for Scalable Event Streaming  
- Tiered storage (hot/warm/cold data)  
- Pulsar Functions for lightweight stream processing  

2.4 Fault Tolerance in Streaming  
- Idempotent producers and consumer offset management  
- Dead-letter queues for poison pills  

*Lab: Ingest real-time sensor data with Kafka and validate schemas.*  

---  

**3. API Ingestion with Requests & FastAPI**  
3.1 RESTful Ingestion Patterns  
- Pagination, rate limiting, and retry logic  
- Handling OAuth2 and API keys  

3.2 Building Ingestion Endpoints with FastAPI  
- Webhooks for push-based data collection  
- Async I/O for high-concurrency workloads  

3.3 Data Validation at Ingestion  
- Schema checks with Pydantic models  
- Sanitizing input data (e.g., SQL injection prevention)  

*Lab: Create a FastAPI endpoint to ingest and validate JSON payloads.*  

---  

**4. Change Data Capture (CDC) with Debezium**  
4.1 CDC Fundamentals  
- Log-based vs. trigger-based CDC  
- Capturing inserts, updates, and deletes  

4.2 Debezium Connectors  
- MySQL, PostgreSQL, and MongoDB setups  
- Schema evolution and backward compatibility  

4.3 Handling Schema Drift  
- Column renames, type changes, and drops  
- Integrating with schema registries  

*Lab: Replicate a PostgreSQL table to Kafka using Debezium.*  

---  

**5. Web Scraping with Scrapy & Selenium**  
5.1 Scrapy for Structured Scraping  
- Spider design and XPath/CSS selectors  
- Middlewares for proxies and user-agent rotation  

5.2 Selenium for Dynamic Content  
- Headless browsers and JavaScript rendering  
- Avoiding bot detection (e.g., CAPTCHAs)  

5.3 Ethical and Legal Considerations  
- Robots.txt compliance and rate limiting  
- Data anonymization for scraped PII  

*Lab: Scrape product data from an e-commerce site (static + dynamic).*  

---  

**6. Cloud Ingestion Patterns**  
6.1 Multi-Cloud Strategies  
- Idempotent writes across cloud storage (S3, GCS, Azure Blob)  
- Hybrid architectures (on-prem + cloud)  

6.2 Fault Tolerance in Cloud Ingestion  
- Retries with exponential backoff  
- Message deduplication (e.g., SQS dedup IDs)  

6.3 Cost Optimization  
- Batch vs. real-time ingestion costs  
- Lifecycle policies for raw data  

*Lab: Ingest data to cloud storage with deduplication and retries.*  

---  

### **Key Features**  
- **Tool Agnostic**: Focuses on *patterns* (e.g., idempotency) over vendor specifics.  
- **Alignment with Data Quality**: Schema validation and PII handling link to `05_data_quality`.  
- **No Deployment Fluff**: Avoids cloud setup/credentials management.  

---
