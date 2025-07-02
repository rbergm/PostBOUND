#!/usr/bin/env python3
#
# This script demonstrates how a learned optimizer can be used with PostBOUND. More specifically, it demonstrates how to
# obtain training and test splits, and how to use online as well as offline training. The example is focused on the TONIC
# optimization algorithm.
#
# Requirements: a running IMDB instance on Postgres with the connect file being set-up correctly. This can be achieved using
# the utilities from db-support/postgres.
#

import math

import postbound as pb
from postbound.optimizer.strategies import tonic

# Setup: we optimize queries from the Join Order Benchmark on a Postgres database
postgres_db = pb.postgres.connect()
job_workload = pb.workloads.job()

# Obtain a training and test split
n_train_queries = math.floor(0.2 * len(job_workload))
train_queries = job_workload.pick_random(n_train_queries)
test_queries = job_workload - train_queries

# Pre-train the TONIC model
# For optimal results, it would be even better to run an "exploratory" training to explore different physical join operators.
# We leave this step out for simplicity
tonic_recommender = tonic.TonicOperatorSelection()
for train_query in train_queries.queries():
    tonic_recommender.simulate_feedback(train_query)

# Now let's generate the optimization pipeline with our learned optimizer
pipeline = pb.MultiStageOptimizationPipeline(postgres_db)
pipeline.setup_physical_operator_selection(tonic_recommender)
pipeline.build()

# Instead of manually iterating over the test set, we use a pre-defined utility that handles the optimization and benchmarking
# for us. Take a look at the dedicated example for more details on the benchmarking tools.
# Notice that we use the online-learning capabilities of TONIC to update the underlying QEP-S model once an optimized query has
# been executed
result_df = pb.optimize_and_execute_workload(
    test_queries,
    pipeline,
    post_process=lambda res: tonic_recommender.simulate_feedback(res.query),
)

print("Benchmark results:")
print(result_df)

print("Final QEP-S:")
print(tonic_recommender.qeps.inspect())
