#!/usr/bin/env python3
#
# This script demonstrates the usage of different benchmarking tools. It contrasts native optimization by an actual database
# system to a defensive optimization scheme using the UES algorithm.
# Notice that some of the imports can be omitted since the main import postbound as pb already provides shortcuts to them.
# This includes the call to workloads.job() as well as execute_workload and optimize_and_execute_workload.
#
# Requirements: a running IMDB instance on Postgres with the connect file being set-up correctly. This can be achieved using
# the utilities from db-support/postgres.
#

import postbound as pb
from postbound.optimizer import presets

# Setup: we optimize queries from the Join Order Benchmark on a Postgres database
postgres_db = pb.postgres.connect()
job_workload = pb.workloads.job().first(3)

# Configure the optimization pipeline for UES
ues_settings = presets.fetch("ues")
ues_pipeline = (
    pb.MultiStageOptimizationPipeline(postgres_db).load_settings(ues_settings).build()
)

# Execute the benchmarks: each query should be repeated 3 times and each workload should be repeated 3 times as well
# After each workload repetition, the execution order of all queries should be changed. Finally, all queries should be executed
# as COUNT(*) queries
query_preparation = pb.bench.QueryPreparation(count_star=True)

# Benchmark the native workload
native_results = pb.bench.execute_workload(
    job_workload,
    on=postgres_db,
    workload_repetitions=3,
    per_query_repetitions=3,
    shuffled=True,
    query_preparation=query_preparation,
    include_labels=True,
    logger="tqdm",
)

# Benchmark the UES workload
ues_results = pb.bench.execute_workload(
    job_workload,
    on=ues_pipeline,
    workload_repetitions=3,
    per_query_repetitions=3,
    shuffled=True,
    query_preparation=query_preparation,
    include_labels=True,
    logger="tqdm",
)
