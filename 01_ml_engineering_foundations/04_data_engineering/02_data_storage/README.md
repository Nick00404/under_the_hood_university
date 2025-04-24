### **02_data_storage**  
**1. Data Lakes with Iceberg & Delta Lake**  
1.1 Modern Data Lake Fundamentals  
- Table formats vs. raw storage (Iceberg/Delta Lake vs. traditional S3/GCS)  
- ACID transactions and time travel capabilities  

1.2 Iceberg for Scalable Metadata  
- Partition evolution and hidden partitioning  
- Schema evolution (in-place vs. snapshot-based changes)  

1.3 Delta Lake for Transactional Guarantees  
- Merge operations for upserts and CDC  
- Optimize and Z-Ordering for query performance  

1.4 Hybrid Architectures  
- Integrating data lakes with warehouses (e.g., Snowflake external tables)  

*Lab: Perform schema evolution on a Delta Lake table while maintaining backward compatibility.*  

---  

**2. Data Warehousing with Snowflake & BigQuery**  
2.1 Warehouse Design Patterns  
- Star vs. Snowflake schemas  
- Materialized views and clustering  

2.2 Snowflake Optimization  
- Virtual warehouses (scaling compute/storage independently)  
- Zero-copy cloning for testing  

2.3 BigQuery Serverless Analytics  
- Partitioning and clustering strategies  
- BI Engine for low-latency queries  

2.4 Cost-Aware Warehousing  
- Storage vs. compute pricing models  
- Query optimization (e.g., avoiding SELECT *)  

*Lab: Compare query performance for partitioned vs. clustered tables in BigQuery.*  

---  

**3. Columnar Storage with Parquet & Avro**  
3.1 Columnar Storage Benefits  
- Predicate pushdown and compression efficiency  
- Vectorized query execution  

3.2 Parquet Deep Dive  
- Dictionary encoding and page indexing  
- Choosing compression codecs (Snappy, Zstd, Gzip)  

3.3 Avro for Schema-Based Storage  
- Schema resolution (reader/writer compatibility)  
- Embedding schemas in Kafka pipelines  

3.4 Tradeoffs in Practice  
- Parquet for analytics vs. Avro for streaming  

*Lab: Convert a CSV dataset to Parquet and analyze storage/query gains.*  

---  

**4. Graph Storage with Neo4j & JanusGraph**  
4.1 Graph Data Modeling  
- Nodes, edges, and properties  
- Cypher (Neo4j) vs. Gremlin (JanusGraph) query languages  

4.2 Neo4j for Real-Time Insights  
- Index-free adjacency and traversal performance  
- Graph algorithms (PageRank, shortest path)  

4.3 JanusGraph for Distributed Graphs  
- Backing storage (Cassandra, HBase)  
- Scaling horizontally with sharding  

4.4 Use Case: Fraud Detection  
- Pattern matching in transactional graphs  

*Lab: Build a fraud detection graph model using Neo4j.*  

---  

**5. Time-Series Storage with InfluxDB & TimescaleDB**  
5.1 Time-Series Data Challenges  
- High write throughput and retention policies  
- Downsampling and continuous aggregates  

5.2 InfluxDB TSM Engine  
- Time-Structured Merge Tree for efficient writes  
- Flux vs. InfluxQL for queries  

5.3 TimescaleDB Hypertables  
- Automated partitioning by time  
- Integrating with PostgreSQL ecosystems  

5.4 Compression Techniques  
- Gorilla encoding for metrics (InfluxDB)  
- Delta-of-Delta for monotonic series (TimescaleDB)  

*Lab: Compare query speeds for compressed vs. raw time-series data.*  

---  

**6. Distributed Storage with HDFS & Ceph**  
6.1 HDFS Architecture  
- NameNode/DataNode roles and fault tolerance  
- Erasure coding vs. replication tradeoffs  

6.2 Ceph Object Storage  
- CRUSH algorithm for data placement  
- RADOS block/object/file interfaces  

6.3 Hybrid Cloud Storage  
- Tiering data between HDFS and S3/GCS  

*Lab: Benchmark read/write performance in HDFS vs. Ceph.*  

---  

**7. Schema Evolution Management**  
7.1 Schema Compatibility  
- Backward, forward, and full compatibility modes  
- Schema registry integration (e.g., Confluent, AWS Glue)  

7.2 Evolution Patterns  
- Additive changes (safe new columns)  
- Type promotions (e.g., INT → BIGINT)  

7.3 Impact on Downstream Systems  
- Versioned API contracts for consumers  
- Breaking change communication strategies  

*Lab: Evolve an Avro schema while maintaining compatibility.*  

---  

**8. Storage Cost Optimization**  
8.1 Tiered Storage Strategies  
- Hot (SSD), warm (HDD), cold (Glacier) data tiers  
- Lifecycle policies for auto-tiering  

8.2 Partitioning for Efficiency  
- Date-based vs. categorical partitioning  
- Dynamic vs. static partitioning  

8.3 Compression & Deduplication  
- Lossless vs. lossy compression (e.g., Parquet vs. JPEG)  
- Hash-based deduplication for redundant datasets  

8.4 Monitoring & Governance  
- Cost anomaly detection (e.g., unexpected S3 spikes)  
- Tagging resources for chargeback/showback  

*Lab: Optimize storage costs for a 10TB dataset using tiering/compression.*  

---  

### **Key Features**  
- **Pattern-First Approach**: Focuses on *when* and *why* to use storage systems, not just *how*.  
- **Cross-Module Links**: Schema evolution (07) ties to data quality (05_data_quality) and governance (06_data_governance).  
- **Real-World Labs**: Emphasizes measurable outcomes (cost reduction, query optimization).  
- **No Vendor Lock-In**: Principles apply across Iceberg/Delta, Snowflake/BigQuery, etc.  

---
