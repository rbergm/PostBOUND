# PostBOUND

This directory contains the actual (Python) implementation of our PostBOUND framework. The code itself is located in
the `postbound` directory.

## Getting started

```python
from postbound import postbound as pb
from postbound.db import postgres
from postbound.db.systems import systems
from postbound.experiments import workloads
from postbound.optimizer import presets

# Step 1: System setup
presets.apply_standard_system_options()
postgres_instance = postgres.connect()
job_workload = workloads.job()
ues_settings = presets.fetch("ues")

# Step 2: Optimization pipeline setup
optimization_pipeline = pb.OptimizationPipeline(systems.Postgres(postgres_instance))
optimization_pipeline.load_settings(ues_settings)
optimization_pipeline.build()

# Step 3: Query optimization
optimized_query = optimization_pipeline.optimize_query(job_workload["1a"])

# Step 4: Query execution
query_result = postgres_instance.execute_query(optimized_query)
print(query_result)
```
