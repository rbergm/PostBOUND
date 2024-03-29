"""Regression tests to ensure the PostBOUND examples work correctly."""

import unittest

pg_connect_dir = "."


class ReadmeExampleTests(unittest.TestCase):
    def test_example(self) -> None:
        from postbound import postbound as pb
        from postbound.optimizer import presets
        from postbound.db import postgres
        from postbound.experiments import workloads

        postgres_instance = postgres.connect(config_file=f"{pg_connect_dir}/.psycopg_connection_job")
        presets.apply_standard_system_options()
        job_workload = workloads.job()
        ues_settings = presets.fetch("ues")

        optimization_pipeline = pb.TwoStageOptimizationPipeline(postgres_instance)
        optimization_pipeline.load_settings(ues_settings)
        optimization_pipeline.build()

        optimized_query = optimization_pipeline.optimize_query(job_workload["1a"])

        postgres_instance.execute_query(optimized_query)
