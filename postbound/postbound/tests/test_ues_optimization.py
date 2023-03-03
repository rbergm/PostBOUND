from __future__ import annotations

import sys

sys.path.append("../../")

from postbound import postbound as pb
from postbound.db import postgres
from postbound.db.systems import systems
from postbound.qal import parser
from postbound.experiments import workloads
from postbound.optimizer import presets, validation

from postbound.tests import regression_suite

workloads.workloads_base_dir = "../../../workloads"
pg_connect_dir = "../.."


class JOBWorkloadTests(regression_suite.DatabaseTestCase):
    def setUp(self) -> None:
        self.db = postgres.connect(config_file=f"{pg_connect_dir}/.psycopg_connection_job")
        self.db.statistics().emulated = True
        self.db.statistics().cache_enabled = True
        self.job = workloads.job()

    def test_optimize_workload(self) -> None:
        optimization_pipeline = pb.OptimizationPipeline(target_dbs=systems.Postgres(self.db))
        optimization_pipeline.load_settings(presets.fetch("ues"))
        optimization_pipeline.build()
        for label, query in self.job.entries():
            # JOB is fully supported by UES
            with self.subTest(label=label, query=query):
                original_result = self.db.execute_query(query, cache_enabled=False)
                optimized_query = optimization_pipeline.optimize_query(query)
                optimized_result = self.db.execute_query(optimized_query, cache_enabled=False)
                self.assertResultSetsEqual(original_result, optimized_result, ordered=query.is_ordered())


class SSBWorkloadTests(regression_suite.DatabaseTestCase):
    def setUp(self) -> None:
        self.db = postgres.connect(config_file=f"{pg_connect_dir}/.psycopg_connection_tpch")
        parser.auto_bind_columns = True
        self.db.statistics().emulated = True
        self.db.statistics().cache_enabled = True
        self.ssb = workloads.ssb()

    def test_optimize_workload(self) -> None:
        optimization_pipeline = pb.OptimizationPipeline(target_dbs=systems.Postgres(self.db))
        optimization_pipeline.load_settings(presets.fetch("ues"))
        optimization_pipeline.build()
        for label, query in self.ssb.entries():
            # SSB is fully supported by UES
            with self.subTest(label=label, query=query):
                original_result = self.db.execute_query(query, cache_enabled=False)
                optimized_query = optimization_pipeline.optimize_query(query)
                optimized_result = self.db.execute_query(optimized_query, cache_enabled=False)
                self.assertResultSetsEqual(original_result, optimized_result, ordered=query.is_ordered())


class StackWorkloadTests(regression_suite.DatabaseTestCase):
    def setUp(self) -> None:
        self.db = postgres.connect(config_file=f"{pg_connect_dir}/.psycopg_connection_stack")
        self.db.statistics().emulated = True
        self.db.statistics().cache_enabled = True
        self.ssb = workloads.ssb()

    def test_optimize_workload(self) -> None:
        optimization_pipeline = pb.OptimizationPipeline(target_dbs=systems.Postgres(self.db))
        optimization_pipeline.load_settings(presets.fetch("ues"))
        optimization_pipeline.build()
        for label, query in self.ssb.entries():
            # Stack is only partially supported by UES
            with self.subTest(label=label, query=query):
                try:
                    original_result = self.db.execute_query(query, cache_enabled=False)
                    optimized_query = optimization_pipeline.optimize_query(query)
                    optimized_result = self.db.execute_query(optimized_query, cache_enabled=False)
                    self.assertResultSetsEqual(original_result, optimized_result, ordered=query.is_ordered())
                except validation.UnsupportedQueryError as e:
                    self.skipTest(f"Unsupported query: {e}")
