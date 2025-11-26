from __future__ import annotations

import unittest

import postbound as pb
from postbound.db import postgres
from postbound.experiments import workloads
from postbound.optimizer import tonic
from postbound.qal import transform
from tests import regression_suite

pg_connect_dir = "."


@regression_suite.skip_if_no_db(f"{pg_connect_dir}/.psycopg_connection_job")
class DefaultTonicTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db = postgres.connect(
            config_file=f"{pg_connect_dir}/.psycopg_connection_job"
        )
        self.job = workloads.job()

    def test_optimize_workload(self) -> None:
        tonic_optimizer = tonic.TonicOperatorSelection(False, database=self.db)

        for label, query in self.job.entries():
            with self.subTest("Training", label=label):
                query_plan = self.db.optimizer().query_plan(query)
                tonic_optimizer.integrate_cost(query, query_plan)

        optimization_pipeline = pb.MultiStageOptimizationPipeline(self.db)
        optimization_pipeline.setup_physical_operator_selection(tonic_optimizer)
        optimization_pipeline.build()
        for label, query in self.job.entries():
            with self.subTest("Recommendation", label=label):
                optimized_query = optimization_pipeline.optimize_query(query)
                explain_query = transform.as_explain(optimized_query)
                self.db.execute_query(explain_query, cache_enabled=False)


if __name__ == "__main__":
    unittest.main()
