
import unittest
import sys

sys.path.append("../")
import regression_suite  # noqa: E402
from transform import mosp  # noqa: E402


class JobWorkloadParsingTests(unittest.TestCase):

    def setUp(self):
        self.workload = regression_suite.load_job_workload()

    def test_can_parse_workload(self):
        for query in self.workload:
            try:
                mosp.MospQuery.parse(query)
            except Exception as e:
                self.fail(f"Exception raised on query {query} with exception {e}")

    def test_can_access_predicates(self):
        for query in self.workload:
            try:
                parsed = mosp.MospQuery.parse(query)
                parsed.predicates()  # TODO: refactor
            except Exception as e:
                self.fail(f"Exception raised on query {query} with exception {e}")


if __name__ == "__main__":
    unittest.main()
