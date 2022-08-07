
import unittest
import sys

sys.path.append("../")
import regression_suite  # noqa: E402
from transform import db, mosp, ues  # noqa: E402


job_workload = regression_suite.load_job_workload()


class JoinGraphTests(unittest.TestCase):
    pass


class BeningQueryOptimizationTests(unittest.TestCase):
    def test_base_query(self):
        query = mosp.MospQuery.parse(job_workload["1a"])
        optimized = ues.optimize_query(query)


class JobWorkloadOptimizationTests(unittest.TestCase):
    def test_workload(self):
        for label, query in job_workload.items():
            try:
                parsed = mosp.MospQuery.parse(query)
                ues.optimize_query(parsed)
            except Exception as e:
                self.fail(f"Exception raised on query {label} with exception {e}")


class SnowflaxeQueryOptimizationTests(unittest.TestCase):
    def test_base_query(self):
        pass


class CrossProductQueryOptimizationTests(unittest.TestCase):
    def test_base_query(self):
        pass

    def test_base_with_snowflake(self):
        pass


class WeirdQueriesOptimizationTests(unittest.TestCase):
    def test_no_joins_query(self):
        pass


if "__name__" == "__main__":
    unittest.main()
