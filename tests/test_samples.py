"""Regression tests to ensure the PostBOUND examples work correctly."""

import unittest

from tests import regression_suite

pg_connect_dir = "."


class ReadmeExampleTests(unittest.TestCase):

    @regression_suite.skip_if_no_db(f"{pg_connect_dir}/.psycopg_connection_job")
    def test_example(self) -> None:
        import postbound as pb
        from postbound.optimizer import presets

        postgres_instance = pb.postgres.connect(config_file=f"{pg_connect_dir}/.psycopg_connection_job")
        presets.apply_standard_system_options()
        job_workload = pb.workloads.job()
        ues_settings = presets.fetch("ues")

        optimization_pipeline = pb.TwoStageOptimizationPipeline(postgres_instance)
        optimization_pipeline.load_settings(ues_settings)
        optimization_pipeline.build()

        optimized_query = optimization_pipeline.optimize_query(job_workload["1a"])

        postgres_instance.execute_query(optimized_query)


if __name__ == "__main__":
    unittest.main()
