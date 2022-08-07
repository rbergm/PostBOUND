
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
