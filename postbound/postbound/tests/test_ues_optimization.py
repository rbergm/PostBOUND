from __future__ import annotations

import sys

sys.path.append("../../")

from postbound import postbound as pb
from postbound.db import postgres
from postbound.db.systems import systems
from postbound.experiments import workloads
from postbound.optimizer import presets

from postbound.tests import regression_suite

workloads.workloads_base_dir = "../../../workloads"
pg_connect_file = "../../.psycopg_connection"


class JobWorkloadTests(regression_suite.DatabaseTestCase):
    def setUp(self) -> None:
        self.db = postgres.connect(config_file=pg_connect_file)
        self.db.statistics().emulated = True
        self.job = workloads.job()

    def test_optimize_workload(self) -> None:
        optimization_pipeline = pb.OptimizationPipeline(target_dbs=systems.Postgres(self.db))
        optimization_pipeline.load_settings(presets.fetch("ues"))
        optimization_pipeline.build()
        for label, query in self.job.entries():
            with self.subTest(label, query=query):
                original_result = self.db.execute_query(query, cache_enabled=False)
                optimized_query = optimization_pipeline.optimize_query(query)
                optimized_result = self.db.execute_query(optimized_query, cache_enabled=False)
                self.assertResultSetsEqual(original_result, optimized_result, ordered=query.is_ordered())
