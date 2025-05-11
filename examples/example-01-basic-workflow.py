#!/usr/bin/env python3
#
# This script show the basic steps that are involved in a typical analysis scenario with PostBOUND. Later examples expand on
# these concepts in more detail. The example is focused on the UES optimization algorithm.
#
# Requirements: a running IMDB instance on Postgres with the connect file being set-up correctly. This can be achieved using
# the utilities from db-support/postgres
#

# Step 0: imports
# The main PostBOUND package provides access to the frequently-used parts of the library. For use-cases that do not fit into
# standard scenarios, the sub-packages can be imported directly. This also includes parts of the library that are not
# considered core functionality per se.
import postbound as pb
from postbound.optimizer import presets

# Step 1: System setup
postgres_instance = pb.postgres.connect()
presets.apply_standard_system_options()
job_workload = pb.workloads.job()

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
