
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
        optimized = ues.optimize_query(query, trace=True)


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
        query = mosp.MospQuery.parse(job_workload["32a"])
        optimized = ues.optimize_query(query)


class CrossProductQueryOptimizationTests(unittest.TestCase):
    def test_base_query(self):
        raw_query = """SELECT * FROM info_type it, company_type ct
                       WHERE it.info = 'top 250 rank' AND ct.kind = 'production companies'"""
        query = mosp.MospQuery.parse(raw_query)
        ues.optimize_query(query)

    def test_base_with_snowflake(self):
        pass


class WeirdQueriesOptimizationTests(unittest.TestCase):
    def test_no_joins_query(self):
        raw_query = "SELECT * FROM company_type ct WHERE ct.kind = 'production companies'"
        query = mosp.MospQuery.parse(raw_query)
        optimized_query = ues.optimize_query(query)


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
        optimized_query = ues.optimize_query(query)


if "__name__" == "__main__":
    unittest.main()
