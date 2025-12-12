"""Tests for PostBOUND's implementation of the UES algorithm on various benchmarks.

In order to run these tests, working instances of the various databases are required and connect files have to be
created accordingly.

The tests do not run any performance measurements, but rather only ensure that results between original and optimized
queries remain equal. They act as regression tests in that sense.
"""

from __future__ import annotations

import os
import unittest

import postbound as pb
from postbound import db, workloads
from postbound.db import postgres
from postbound.opt import presets, ues
from postbound.qal import parser, transform
from tests import regression_suite

pg_connect_dir = "."


@regression_suite.skip_if_no_db(f"{pg_connect_dir}/.psycopg_connection_job")
class JobWorkloadTests(regression_suite.DatabaseTestCase):
    def setUp(self) -> None:
        self.db = postgres.connect(
            config_file=f"{pg_connect_dir}/.psycopg_connection_job"
        )
        self.db.statistics().emulated = False
        self.db.statistics().cache_enabled = True
        self.job = workloads.job()

    @unittest.skipUnless(
        os.environ.get("COMPARE_RESULT_SETS", None),
        "Skipping result set equivalence comparison. Set COMPARE_RESULT_SETS environment variable "
        "to non-empty value to change.",
    )
    def test_result_set_equivalence(self) -> None:
        optimization_pipeline = pb.MultiStageOptimizationPipeline(target_db=self.db)
        optimization_pipeline.load_settings(presets.fetch("ues"))
        optimization_pipeline.build()
        for label, query in self.job.entries():
            # JOB is fully supported by UES
            with self.subTest(label=label, query=query):
                original_result = self.db.execute_query(query, cache_enabled=True)
                optimized_query = optimization_pipeline.optimize_query(query)
                optimized_result = self.db.execute_query(
                    optimized_query, cache_enabled=False
                )
                self.assertResultSetsEqual(
                    original_result, optimized_result, ordered=query.is_ordered()
                )

    def test_optimize_workload(self) -> None:
        optimization_pipeline = pb.MultiStageOptimizationPipeline(target_db=self.db)
        optimization_pipeline.load_settings(presets.fetch("ues"))
        optimization_pipeline.build()
        for label, query in self.job.entries():
            # JOB is fully supported by UES
            with self.subTest(label=label, query=query):
                optimized_query = optimization_pipeline.optimize_query(query)
                explain_query = transform.as_explain(optimized_query)
                self.db.execute_query(explain_query, cache_enabled=False)

    @unittest.skipUnless(
        os.environ.get("CHECK_JOB_SANITY", None),
        "Skipping sanity checks for produced join orders. Set CHECK_JOB_SANITY environment variable to "
        "non-empty value to change. Notice that this test does not fail, but rather prints potential "
        "problems directly to stdout.",
    )
    def test_optimized_join_orders(self) -> None:
        ues_optimizer = ues.UESJoinOrderOptimizer(database=self.db)

        detected_subqueries = 0
        unique_join_orders = set()
        previous_family = ""
        for label, query in self.job.entries():
            current_family = label[:-1]
            if current_family != previous_family:
                if len(unique_join_orders) == 1:
                    print(
                        "All join orders for family",
                        previous_family,
                        "are the same. This could indicate a programming error!",
                    )
                unique_join_orders = set()

            current_join_order = ues_optimizer.optimize_join_order(query)
            unique_join_orders.add(current_join_order)
            contains_subquery = current_join_order.is_bushy()
            if contains_subquery:
                detected_subqueries += 1

            previous_family = current_family

        if not detected_subqueries:
            print(
                "No subqueries have been detected. This could indicate a programming error!"
            )

    def test_basic_behavior(self) -> None:
        optimization_pipeline = (
            pb.MultiStageOptimizationPipeline(target_db=self.db)
            .load_settings(presets.fetch("ues"))
            .build()
        )
        query = self.job["1a"]
        optimized_query = optimization_pipeline.optimize_query(query)
        self.db.optimizer().query_plan(optimized_query)


@regression_suite.skip_if_no_db(f"{pg_connect_dir}/.psycopg_connection_ssb")
class SsbWorkloadTests(regression_suite.DatabaseTestCase):
    def setUp(self) -> None:
        self.db = postgres.connect(
            config_file=f"{pg_connect_dir}/.psycopg_connection_ssb"
        )
        self.db.statistics().emulated = True
        self.db.statistics().cache_enabled = True
        parser.auto_bind_columns = True
        self.ssb = workloads.ssb()

    @unittest.skipUnless(
        os.environ.get("COMPARE_RESULT_SETS", None),
        "Skipping result set equivalence comparison. Set COMPARE_RESULT_SETS environment variable "
        "to non-empty value to change.",
    )
    def test_result_set_equivalence(self) -> None:
        optimization_pipeline = pb.MultiStageOptimizationPipeline(target_db=self.db)
        optimization_pipeline.load_settings(presets.fetch("ues"))
        optimization_pipeline.build()
        for label, query in self.ssb.entries():
            # SSB is fully supported by UES
            with self.subTest(label=label, query=query):
                original_result = self.db.execute_query(query, cache_enabled=True)
                optimized_query = optimization_pipeline.optimize_query(query)
                optimized_result = self.db.execute_query(
                    optimized_query, cache_enabled=False
                )
                self.assertResultSetsEqual(
                    original_result, optimized_result, ordered=query.is_ordered()
                )

    def test_optimize_workload(self) -> None:
        optimization_pipeline = pb.MultiStageOptimizationPipeline(target_db=self.db)
        optimization_pipeline.load_settings(presets.fetch("ues"))
        optimization_pipeline.build()
        for label, query in self.ssb.entries():
            # SSB is fully supported by UES
            with self.subTest(label=label, query=query):
                optimized_query = optimization_pipeline.optimize_query(query)
                explain_query = transform.as_explain(optimized_query)
                self.db.execute_query(explain_query, cache_enabled=False)


@regression_suite.skip_if_no_db(f"{pg_connect_dir}/.psycopg_connection_stack")
class StackWorkloadTests(regression_suite.DatabaseTestCase):
    def setUp(self) -> None:
        self.db = postgres.connect(
            config_file=f"{pg_connect_dir}/.psycopg_connection_stack"
        )
        self.db.statistics().emulated = True
        self.db.statistics().cache_enabled = True
        parser.auto_bind_columns = True
        self.stack = workloads.stack()

    @unittest.skipUnless(
        os.environ.get("COMPARE_RESULT_SETS", None),
        "Skipping result set equivalence comparison. Set COMPARE_RESULT_SETS environment variable "
        "to non-empty value to change.",
    )
    def test_result_set_equivalence(self) -> None:
        optimization_pipeline = pb.MultiStageOptimizationPipeline(target_db=self.db)
        optimization_pipeline.load_settings(presets.fetch("ues"))
        optimization_pipeline.build()
        for label, query in self.stack.entries():
            # Stack is only partially supported by UES
            with self.subTest(label=label, query=query):
                try:
                    optimized_query = optimization_pipeline.optimize_query(query)
                    original_result = self.db.execute_query(query, cache_enabled=True)
                    optimized_result = self.db.execute_query(
                        optimized_query, cache_enabled=False
                    )
                    self.assertResultSetsEqual(
                        original_result, optimized_result, ordered=query.is_ordered()
                    )
                except pb.UnsupportedQueryError as e:
                    self.skipTest(f"Unsupported query: {e}")
                except db.DatabaseServerError as e:
                    self.fail(f"Programming error at query '{label}': {e}")
                except db.DatabaseUserError as e:
                    self.skipTest(f"Internal database error at '{label}': {e}")

    def test_optimize_workload(self) -> None:
        optimization_pipeline = pb.MultiStageOptimizationPipeline(target_db=self.db)
        optimization_pipeline.load_settings(presets.fetch("ues"))
        optimization_pipeline.build()
        for label, query in self.stack.entries():
            # Stack is only partially supported by UES
            with self.subTest(label=label, query=query):
                try:
                    optimized_query = optimization_pipeline.optimize_query(query)
                    explain_query = transform.as_explain(optimized_query)
                    self.db.execute_query(explain_query, cache_enabled=False)
                except pb.UnsupportedQueryError as e:
                    self.skipTest(f"Unsupported query: {e}")


class RegressionTests(unittest.TestCase):
    # No regressions so far!
    pass
