
### **03_data_processing**  
**1. Batch Processing with Spark & Pandas**  
1.1 Core Batch Processing Patterns  
- Distributed processing (Spark) vs. in-memory processing (Pandas)  
- Use cases: ETL workflows, large-scale aggregations  

1.2 Apache Spark Fundamentals  
- RDDs vs. DataFrames vs. Datasets  
- Catalyst optimizer and Tungsten execution  

1.3 Pandas for Scalable Batch Workflows  
- Chunking strategies for memory optimization  
- Parallelism with `swifter` or `modin`  

1.4 Fault Tolerance & Idempotency  
- Spark checkpointing and recovery  
- Ensuring atomic writes in Pandas  

*Lab: Process a 10GB dataset using Spark and Pandas, comparing performance and resource usage.*  

---  

**2. Stream Processing with Flink & ksqlDB**  
2.1 Stream Processing Fundamentals  
- Event time vs. processing time  
- State management and checkpointing  

2.2 Apache Flink for Stateful Processing  
- Windowed aggregations (tumbling, sliding, session)  
- Watermarking and handling late data  

2.3 ksqlDB for Streaming SQL  
- Stream-table joins and materialized views  
- UDFs for custom transformations  

2.4 Fault Tolerance in Practice  
- Exactly-once semantics in Flink  
- Dead-letter topics in ksqlDB  

*Lab: Detect anomalies in a real-time sensor stream using Flink’s CEP library.*  

---  

**3. Data Transformation with dbt & SQLMesh**  
3.1 SQL-Centric Transformation Workflows  
- dbt’s modular Jinja templates vs. SQLMesh’s versioned models  
- Testing and documentation best practices  

3.2 Incremental & Snapshot Models  
- Incremental loads with dbt  
- SQLMesh’s zero-copy cloning for testing  

3.3 Cross-Database Transformations  
- Leveraging dbt adapters (Snowflake, BigQuery)  
- SQLMesh’s federated query support  

*Lab: Build a slowly changing dimension (SCD) Type 2 pipeline using dbt.*  

---  

**4. Distributed Compute with Ray & Dask**  
4.1 Distributed Computing Patterns  
- Task parallelism (Ray) vs. data parallelism (Dask)  
- Shared vs. distributed memory architectures  

4.2 Ray for ML Workloads  
- Distributed hyperparameter tuning  
- Actor model for stateful services  

4.3 Dask for Parallel Data Processing  
- Dask DataFrames for out-of-core computations  
- Integration with Pandas/NumPy APIs  

4.4 Fault Recovery  
- Ray’s object lineage reconstruction  
- Dask’s resilient task graphs  

*Lab: Parallelize a Monte Carlo simulation using Ray and Dask.*  

---  

**5. Geospatial Processing with GeoPandas**  
5.1 Geospatial Data Fundamentals  
- Coordinate reference systems (CRS)  
- Vector vs. raster data handling  

5.2 Spatial Operations in GeoPandas  
- Spatial joins and overlays  
- Indexing with R-trees for query optimization  

5.3 Performance Optimization  
- Partitioning geospatial datasets  
- Leveraging PostGIS for heavy lifting  

5.4 Real-World Use Cases  
- Route optimization  
- Spatial clustering (e.g., retail store placement)  

*Lab: Analyze urban sprawl by processing satellite imagery and census data.*  

---  

**6. Performance Tuning for Spark & Dask**  
6.1 Spark Optimization  
- Caching strategies (`MEMORY_ONLY` vs. `DISK`)  
- Broadcast joins and partition pruning  

6.2 Dask Optimization  
- Task graph visualization and simplification  
- Cluster scaling (adaptive vs. fixed)  

6.3 Resource Management  
- Dynamic allocation in Spark  
- Memory limits and spill-to-disk in Dask  

6.4 Benchmarking & Profiling  
- Spark UI for bottleneck detection  
- Dask’s diagnostic dashboard  

*Lab: Tune a sluggish Spark job to achieve 2x performance gains.*  

---  

### **Key Features**  
- **No Overlaps**: Avoids orchestration/deployment (covered in `04_data_orchestration`).  
- **Tool Agnostic**: Focuses on *patterns* (e.g., windowed aggregations) over vendor specifics.  
- **Cross-Module Links**:  
  - Schema evolution (`02_data_storage`) impacts data transformations.  
  - Validation (covered in `05_data_quality`) is referenced but not duplicated.  
- **Real-World Labs**: Emphasizes measurable outcomes (e.g., 2x performance gains).  

