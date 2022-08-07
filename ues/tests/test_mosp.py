
import unittest
import sys

sys.path.append("../")
import regression_suite  # noqa: E402
from transform import db, mosp  # noqa: E402


class JobWorkloadParsingTests(unittest.TestCase):

    def setUp(self):
        self.workload = regression_suite.load_job_workload()

    def test_can_parse_workload(self):
        for query in self.workload.values():
            try:
                mosp.MospQuery.parse(query)
            except Exception as e:
                self.fail(f"Exception raised on query {query} with exception {e}")

    def test_can_access_predicates(self):
        for query in self.workload.values():
            try:
                parsed = mosp.MospQuery.parse(query)
                parsed.predicates().break_conjunction()
            except Exception as e:
                self.fail(f"Exception raised on query {query} with exception {e}")

    def test_predicate_maps(self):
        query = mosp.MospQuery.parse(self.workload["1a"])
        expected_filters = ["ct.kind = 'production companies'",
                            "it.info = 'top 250 rank'",
                            "mc.note NOT LIKE '%(as Metro-Goldwyn-Mayer Pictures)%'",
                            r"mc.note LIKE '%(co-production)%' OR mc.note LIKE '%(presents)%'"]
        expected_joins   = ["ct.id = mc.company_type_id",  # noqa: E221
                            "t.id = mc.movie_id",
                            "t.id = mi_idx.movie_id",
                            "mc.movie_id = mi_idx.movie_id",
                            "it.id = mi_idx.info_type_id"]

        predicate_map = query.predicates().predicate_map()
        for filter in predicate_map.filters:
            if filter.is_base():
                self.assertIn(str(filter), expected_filters, "Join not found!")
            else:
                for child in filter.children:
                    self.assertIn(str(child), expected_filters, "Filter not found!")

        for join in predicate_map.joins:
            self.assertIn(str(join), expected_joins, "Join not found!")

        mc_table = db.TableRef("movie_companies", "mc")
        mc_filters = ["mc.note NOT LIKE '%(as Metro-Goldwyn-Mayer Pictures)%'",
                      r"mc.note LIKE '%(co-production)%' OR mc.note LIKE '%(presents)%'"]
        for filter in predicate_map.filters[mc_table]:
            self.assertIn(str(filter), mc_filters)


class PredicateMospReconstructionTests(unittest.TestCase):
    pass


if __name__ == "__main__":
    unittest.main()
