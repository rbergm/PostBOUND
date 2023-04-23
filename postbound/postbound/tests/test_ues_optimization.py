"""Tests for PostBOUND's implementation of the UES algorithm on various benchmarks.

In order to run these tests, working instances of the various databases are required and connect files have to be
created accordingly.

The tests do not run any performance measurements, but rather only ensure that results between original and optimized
queries remain equal. They act as regression tests in that sense.
"""
from __future__ import annotations

import sys
import unittest

sys.path.append("../../")

from postbound import postbound as pb  # noqa: E402
from postbound.db import postgres  # noqa: E402
from postbound.db.systems import systems  # noqa: E402
from postbound.qal import parser, transform  # noqa: E402
from postbound.experiments import workloads  # noqa: E402
from postbound.optimizer import presets, validation  # noqa: E402

from postbound.tests import regression_suite  # noqa: E402

workloads.workloads_base_dir = "../../../workloads"
pg_connect_dir = "../.."


class JobWorkloadTests(regression_suite.DatabaseTestCase):
    def setUp(self) -> None:
        self.db = postgres.connect(config_file=f"{pg_connect_dir}/.psycopg_connection_job")
        self.db.statistics().emulated = True
        self.db.statistics().cache_enabled = True
        self.job = workloads.job()

    def test_result_set_equivalence(self) -> None:
        optimization_pipeline = pb.OptimizationPipeline(target_dbs=systems.Postgres(self.db))
        optimization_pipeline.load_settings(presets.fetch("ues"))
        optimization_pipeline.build()
        for label, query in self.job.entries():
            # JOB is fully supported by UES
            with self.subTest(label=label, query=query):
                original_result = self.db.execute_query(query, cache_enabled=True)
                optimized_query = optimization_pipeline.optimize_query(query)
                optimized_result = self.db.execute_query(optimized_query, cache_enabled=False)
                self.assertResultSetsEqual(original_result, optimized_result, ordered=query.is_ordered())

    def test_optimize_workload(self) -> None:
        optimization_pipeline = pb.OptimizationPipeline(target_dbs=systems.Postgres(self.db))
        optimization_pipeline.load_settings(presets.fetch("ues"))
        optimization_pipeline.build()
        for label, query in self.job.entries():
            # JOB is fully supported by UES
            with self.subTest(label=label, query=query):
                optimized_query = optimization_pipeline.optimize_query(query)
                explain_query = transform.as_explain(optimized_query)
                self.db.execute_query(explain_query, cache_enabled=False)


class SsbWorkloadTests(regression_suite.DatabaseTestCase):
    def setUp(self) -> None:
        self.db = postgres.connect(config_file=f"{pg_connect_dir}/.psycopg_connection_ssb")
        self.db.statistics().emulated = True
        self.db.statistics().cache_enabled = True
        parser.auto_bind_columns = True
        self.ssb = workloads.ssb()

    def test_result_set_equivalence(self) -> None:
        optimization_pipeline = pb.OptimizationPipeline(target_dbs=systems.Postgres(self.db))
        optimization_pipeline.load_settings(presets.fetch("ues"))
        optimization_pipeline.build()
        for label, query in self.ssb.entries():
            # SSB is fully supported by UES
            with self.subTest(label=label, query=query):
                original_result = self.db.execute_query(query, cache_enabled=True)
                optimized_query = optimization_pipeline.optimize_query(query)
                optimized_result = self.db.execute_query(optimized_query, cache_enabled=False)
                self.assertResultSetsEqual(original_result, optimized_result, ordered=query.is_ordered())

    def test_optimize_workload(self) -> None:
        optimization_pipeline = pb.OptimizationPipeline(target_dbs=systems.Postgres(self.db))
        optimization_pipeline.load_settings(presets.fetch("ues"))
        optimization_pipeline.build()
        for label, query in self.ssb.entries():
            # SSB is fully supported by UES
            with self.subTest(label=label, query=query):
                optimized_query = optimization_pipeline.optimize_query(query)
                explain_query = transform.as_explain(optimized_query)
                self.db.execute_query(explain_query, cache_enabled=False)


class StackWorkloadTests(regression_suite.DatabaseTestCase):
    def setUp(self) -> None:
        self.db = postgres.connect(config_file=f"{pg_connect_dir}/.psycopg_connection_stack")
        self.db.statistics().emulated = True
        self.db.statistics().cache_enabled = True
        parser.auto_bind_columns = True
        self.stack = workloads.stack()

    def test_result_set_equivalence(self) -> None:
        optimization_pipeline = pb.OptimizationPipeline(target_dbs=systems.Postgres(self.db))
        optimization_pipeline.load_settings(presets.fetch("ues"))
        optimization_pipeline.build()
        for label, query in self.stack.entries():
            # Stack is only partially supported by UES
            with self.subTest(label=label, query=query):
                try:
                    optimized_query = optimization_pipeline.optimize_query(query)
                    original_result = self.db.execute_query(query, cache_enabled=True)
                    optimized_result = self.db.execute_query(optimized_query, cache_enabled=False)
                    self.assertResultSetsEqual(original_result, optimized_result, ordered=query.is_ordered())
                except validation.UnsupportedQueryError as e:
                    self.skipTest(f"Unsupported query: {e}")

    def test_optimize_workload(self) -> None:
        optimization_pipeline = pb.OptimizationPipeline(target_dbs=systems.Postgres(self.db))
        optimization_pipeline.load_settings(presets.fetch("ues"))
        optimization_pipeline.build()
        for label, query in self.stack.entries():
            # Stack is only partially supported by UES
            with self.subTest(label=label, query=query):
                try:
                    optimized_query = optimization_pipeline.optimize_query(query)
                    explain_query = transform.as_explain(optimized_query)
                    self.db.execute_query(explain_query, cache_enabled=False)
                except validation.UnsupportedQueryError as e:
                    self.skipTest(f"Unsupported query: {e}")


class RegressionTests(unittest.TestCase):
    # No regressions so far!
    pass
