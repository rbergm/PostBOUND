#!/usr/bin/env python3

# Step 0: imports
# this includes the optimization modules, database modules and possibly workload modules
from postbound import postbound as pb
from postbound.optimizer import presets
from postbound.db import postgres
from postbound.experiments import workloads

# Step 1: System setup
postgres_instance = postgres.connect()
presets.apply_standard_system_options()
job_workload = workloads.job()

# Step 2: Optimization pipeline setup
# If necessary, this step can also include the definition of different optimization strategies
ues_settings = presets.fetch("ues")
optimization_pipeline = pb.TwoStageOptimizationPipeline(postgres_instance)
optimization_pipeline.load_settings(ues_settings)
optimization_pipeline.build()

# Step 3: Query optimization
# If necessary, the optimization step could be merged with Step 4 for integrated benchmarking
input_query = job_workload["1a"]
optimized_query = optimization_pipeline.optimize_query(input_query)

# Step 4: Query execution
# Alternatively, this step could include much more complicated benchmarking logic
query_result = postgres_instance.execute_query(optimized_query)
print(query_result)
