
from __future__ import annotations

import unittest

from postbound.db import postgres
from postbound.experiments import workloads
from postbound.qal import transform
from postbound import postbound as pb
from postbound.optimizer.strategies import tonic


workloads.workloads_base_dir = "../workloads"
pg_connect_dir = "."


class DefaultTonicTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db = postgres.connect(config_file=f"{pg_connect_dir}/.psycopg_connection_job")
        self.job = workloads.job()

    def test_optimize_workload(self) -> None:
        tonic_optimizer = tonic.TonicOperatorSelection(False, database=self.db)

        for label, query in self.job.entries():
            with self.subTest("Training", label=label):
                query_plan = self.db.optimizer().query_plan(query)
                tonic_optimizer.integrate_cost(query, query_plan)

        optimization_pipeline = pb.OptimizationPipeline(self.db)
        optimization_pipeline.setup_physical_operator_selection(tonic_optimizer)
        optimization_pipeline.build()
        for label, query in self.job.entries():
            with self.subTest("Recommendation", label=label):
                optimized_query = optimization_pipeline.optimize_query(query)
                explain_query = transform.as_explain(optimized_query)
                self.db.execute_query(explain_query, cache_enabled=False)
