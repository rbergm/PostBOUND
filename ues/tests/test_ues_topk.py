
import unittest
import sys

import pandas as pd

sys.path.append("../")
import regression_suite  # noqa: E402
from transform import db, mosp, ues, util  # noqa: E402, F401


job_workload = regression_suite.load_job_workload()
dbs = db.DBSchema.get_instance()


class JoinGraphTests(unittest.TestCase):
    pass


class BeningQueryOptimizationTests(unittest.TestCase):
    def test_base_query(self):
        query = mosp.MospQuery.parse(job_workload["1a"])
        optimized = ues.optimize_query(query, join_cardinality_estimation="topk", trace=True)  # noqa: F841


class JobWorkloadOptimizationTests(unittest.TestCase):
    def test_workload(self):
        for label, query in job_workload.items():
            try:
                parsed = mosp.MospQuery.parse(query)
                optimized = ues.optimize_query(parsed, join_cardinality_estimation="topk")  # noqa: F841
            except Exception as e:
                self.fail(f"Exception raised on query {label} with exception {e}")


class SnowflaxeQueryOptimizationTests(unittest.TestCase):
    def test_base_query(self):
        query = mosp.MospQuery.parse(job_workload["32a"])
        optimized = ues.optimize_query(query, join_cardinality_estimation="topk")  # noqa: F841


class CrossProductQueryOptimizationTests(unittest.TestCase):
    def test_base_query(self):
        raw_query = """SELECT * FROM info_type it, company_type ct
                       WHERE it.info = 'top 250 rank' AND ct.kind = 'production companies'"""
        query = mosp.MospQuery.parse(raw_query)
        optimized = ues.optimize_query(query, join_cardinality_estimation="topk")  # noqa: F841

    def test_base_with_snowflake(self):
        pass


class WeirdQueriesOptimizationTests(unittest.TestCase):
    def test_no_joins_query(self):
        raw_query = "SELECT * FROM company_type ct WHERE ct.kind = 'production companies'"
        query = mosp.MospQuery.parse(raw_query)
        optimized_query = ues.optimize_query(query, join_cardinality_estimation="topk")  # noqa: F841


class CompoundJoinPredicateOptimizationTests(unittest.TestCase):
    def test_base_query(self):
        raw_query = r"""
            SELECT COUNT(*)
            FROM company_type AS ct,
                info_type AS it,
                movie_companies AS mc,
                movie_info_idx AS mi_idx,
                title AS t
            WHERE ct.kind = 'production companies'
            AND it.info = 'top 250 rank'
            AND mc.note NOT LIKE '%(as Metro-Goldwyn-Mayer Pictures)%'
            AND (mc.note LIKE '%(co-production)%'   OR mc.note LIKE '%(presents)%')
            AND ct.id = mc.company_type_id
            AND (t.id = mc.movie_id AND t.imdb_id = mc.company_id)
            AND t.id = mi_idx.movie_id
            AND mc.movie_id = mi_idx.movie_id
            AND it.id = mi_idx.info_type_id;"""
        query = mosp.MospQuery.parse(raw_query)
        optimized_query = ues.optimize_query(query, join_cardinality_estimation="topk")  # noqa: F841


class UpperBoundTests(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.top1_bounds = pd.read_csv("top1_bounds.csv", index_col="label")

    def test_top1_tighter_ues(self):
        for label, raw_query in job_workload.items():
            query = mosp.MospQuery.parse(raw_query)
            top1_res: ues.OptimizationResult = ues.optimize_query(query, join_cardinality_estimation="topk",
                                                                  topk_list_length=1, introspective=True)
            top1_bound = top1_res.final_bound
            ues_res: ues.OptimizationResult = ues.optimize_query(query, introspective=True)
            ues_bound = ues_res.final_bound
            regression_suite.assert_less_equal(top1_bound, ues_bound,
                                               msg=f"Top-K bound must be less than Top-1 bound at query {label}!")

    def test_top1_is_true_upper_bound(self):
        for label, raw_query in job_workload.items():
            query = mosp.MospQuery.parse(raw_query)
            optimization_res: ues.OptimizationResult = ues.optimize_query(query, join_cardinality_estimation="topk",
                                                                          topk_list_length=1, introspective=True)
            upper_bound = optimization_res.final_bound
            true_cardinality = dbs.execute_query(raw_query)
            self.assertGreaterEqual(upper_bound, true_cardinality,
                                    msg=f"Top-K bound must be a true upper bound at query {label}!")

    def test_top15_tighter_top1(self):
        for label, raw_query in job_workload.items():
            query = mosp.MospQuery.parse(raw_query)
            topk_res: ues.OptimizationResult = ues.optimize_query(query, join_cardinality_estimation="topk",
                                                                  topk_list_length=15, introspective=True)
            topk_bound = topk_res.final_bound
            top1_res: ues.OptimizationResult = ues.optimize_query(query, join_cardinality_estimation="topk",
                                                                  topk_list_length=1, introspective=True)
            top1_bound = top1_res.final_bound
            regression_suite.assert_less_equal(topk_bound, top1_bound,
                                               msg=f"Top-K bound must be less than Top-1 bound at query {label}!")

    def test_top15_is_true_upper_bound(self):
        for label, raw_query in job_workload.items():
            query = mosp.MospQuery.parse(raw_query)
            optimization_res: ues.OptimizationResult = ues.optimize_query(query, join_cardinality_estimation="topk",
                                                                          topk_list_length=15, introspective=True)
            upper_bound = optimization_res.final_bound
            true_cardinality = dbs.execute_query(raw_query)
            self.assertGreaterEqual(upper_bound, true_cardinality,
                                    msg=f"Top-K bound must be a true upper bound at query {label}!")


if "__name__" == "__main__":
    unittest.main()
