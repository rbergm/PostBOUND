from __future__ import annotations

from postbound import postbound as pb
from postbound.db import postgres
from postbound.experiments import workloads, runner
from postbound.optimizer import presets

pg_instance = postgres.connect(config_file=".psycopg_connection_job")
job = workloads.job()

ues_pipeline = pb.TwoStageOptimizationPipeline(pg_instance)
ues_pipeline = ues_pipeline.load_settings(presets.fetch("ues")).build()

query_prep = runner.QueryPreparationService(analyze=True)
ues_results_df = runner.optimize_and_execute_workload(job, ues_pipeline,
                                                      include_labels=True, query_preparation=query_prep)
ues_results_df.to_csv("results/job/job-ues.csv", index=False)
