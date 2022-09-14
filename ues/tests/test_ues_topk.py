
import collections
import unittest
import sys
import warnings

import numpy as np

sys.path.append("../")
import regression_suite  # noqa: E402
from transform import db, mosp, ues, util  # noqa: E402, F401
from postgres import explain  # noqa: E402


job_workload = regression_suite.load_job_workload()
dbs = db.DBSchema.get_instance()


class JoinGraphTests(unittest.TestCase):
    pass


class BeningQueryOptimizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.trace_enabled = "--trace" in sys.argv
        if self.log_enabled:
            print()
        return super().setUp()

    def test_base_query(self):
        query = mosp.MospQuery.parse(job_workload["1a"])
        optimized = ues.optimize_query(query, join_cardinality_estimation="topk",  # noqa: F841
                                       trace=self.trace_enabled)


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
    def setUp(self) -> None:
        self.log_enabled = "--trace" in sys.argv
        self.fail_eager = "--fail-late" not in sys.argv
        if self.log_enabled:
            print()
        return super().setUp()

    def test_top1_tighter_ues(self):
        for label, raw_query in job_workload.items():
            query = mosp.MospQuery.parse(raw_query)
            top1_res: ues.OptimizationResult = ues.optimize_query(query, join_cardinality_estimation="topk",
                                                                  topk_list_length=1, introspective=True)
            top1_bound = top1_res.final_bound
            ues_res: ues.OptimizationResult = ues.optimize_query(query, introspective=True)
            ues_bound = ues_res.final_bound
            self.assertLessEqual(top1_bound, ues_bound,
                                 msg=f"Top-K bound must be less than Top-1 bound at query {label}!")

    def test_top1_is_true_upper_bound(self):
        for label, raw_query in job_workload.items():
            query = mosp.MospQuery.parse(raw_query)
            optimization_res: ues.OptimizationResult = ues.optimize_query(query, join_cardinality_estimation="topk",
                                                                          topk_list_length=1, introspective=True)
            upper_bound = optimization_res.final_bound
            true_cardinality = dbs.execute_query(raw_query)
            if upper_bound < true_cardinality:
                self._assertFilterEstimationMismatch(query, label=label,
                                                     true_cardinality=true_cardinality, upper_bound=upper_bound)

    def test_tight_increase(self):
        log = util.make_logger(self.log_enabled)
        pretty_log = util.make_logger(self.log_enabled, pretty=True)
        topk_lengths = [5, 10, 20, 50, 100, 500]
        topk_predecessors = [1, 5, 10, 20, 50, 100]
        bound_results = collections.defaultdict(lambda: collections.defaultdict(lambda: np.inf))

        counterexamples = []

        for label, raw_query in job_workload.items():
            query = mosp.MospQuery.parse(raw_query)
            for topk_idx, topk_length in enumerate(topk_lengths):
                predecessor_length = topk_predecessors[topk_idx]
                if raw_query not in bound_results[predecessor_length]:
                    opt_res: ues.OptimizationResult = ues.optimize_query(query, join_cardinality_estimation="topk",
                                                                         topk_list_length=predecessor_length,
                                                                         introspective=True)
                    bound_results[predecessor_length][raw_query] = opt_res.final_bound
                if raw_query not in bound_results[topk_length]:
                    opt_res: ues.OptimizationResult = ues.optimize_query(query, join_cardinality_estimation="topk",
                                                                         topk_list_length=topk_length,
                                                                         introspective=True)
                    bound_results[topk_length][raw_query] = opt_res.final_bound

                predecessor_bound = bound_results[predecessor_length][raw_query]
                tightened_bound = bound_results[topk_length][raw_query]

                err_msg = f"Bound does not decrease at query {label} for k {predecessor_length} -> {topk_length}!"
                if tightened_bound > predecessor_bound:
                    log("Found counterexample: " + err_msg)
                    counterexamples.append({"label": label, "topk_length": topk_length,
                                            "predecessor_bound": predecessor_bound, "new_bound": tightened_bound})
                    if self.fail_eager:
                        break
                else:
                    improvement_pct = round((1 - (tightened_bound / predecessor_bound)) * 100, 2)
                    log(f"Checked label {label} and TopKs {predecessor_length} -> {topk_length} "
                        f"(improv = {improvement_pct}%)")

            if self.fail_eager and counterexamples:
                break

        if counterexamples:
            log("Found ", len(counterexamples), "counterexamples:")
            pretty_log(counterexamples)
            self.fail("Counterexamples exist")

    def test_top15_is_true_upper_bound(self):
        for label, raw_query in job_workload.items():
            query = mosp.MospQuery.parse(raw_query)
            optimization_res: ues.OptimizationResult = ues.optimize_query(query, join_cardinality_estimation="topk",
                                                                          topk_list_length=15, introspective=True)
            upper_bound = optimization_res.final_bound
            true_cardinality = dbs.execute_query(raw_query)
            if upper_bound < true_cardinality:
                self._assertFilterEstimationMismatch(query, label=label,
                                                     true_cardinality=true_cardinality, upper_bound=upper_bound)

    def _assertFilterEstimationMismatch(self, query: mosp.MospQuery, *, label: str = "", dev_treshold: float = 0.3,
                                        true_cardinality: int = np.nan, upper_bound: int = np.nan):
        filter_accuracy_evaluation = explain.evaluate_filter_estimate_accuracy(query)
        max_deviation = explain.max_filter_estimation_deviation(filter_accuracy_evaluation.values())
        if max_deviation < dev_treshold:
            self.fail(f"Not a true upper bound at query '{label}': {true_cardinality} > {upper_bound}")
        else:
            warnings.warn(f"Not a true upper bound at query '{label}', but filter estimates are inaccurate.")


if "__name__" == "__main__":
    unittest.main()
