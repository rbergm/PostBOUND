
import unittest
import sys

sys.path.append("../")
import regression_suite  # noqa: E402
from transform import db, mosp, ues, util  # noqa: E402, F401


job_workload = regression_suite.load_job_workload()
ssb_workload = regression_suite.load_ssb_workload(simplified=True)


class JoinGraphTests(unittest.TestCase):
    pass


class BeningQueryOptimizationTests(unittest.TestCase):
    def test_base_query(self):
        dbs = db.DBSchema.get_instance(renew=True)
        query = mosp.MospQuery.parse(job_workload["1a"])
        optimized = ues.optimize_query(query, dbs=dbs, trace=True)  # noqa: F841


class JobWorkloadOptimizationTests(unittest.TestCase):
    def test_workload(self):
        dbs = db.DBSchema.get_instance(renew=True)
        for label, query in job_workload.items():
            try:
                parsed = mosp.MospQuery.parse(query)
                optimized = ues.optimize_query(parsed, dbs=dbs)  # noqa: F841
                original_result_set = dbs.execute_query(query)
                optimized_result_set = dbs.execute_query(str(optimized), join_collapse_limit=1, enable_nestloop="off")
                regression_suite.assert_result_sets_equal(original_result_set, optimized_result_set)
            except Exception as e:
                self.fail(f"Exception raised on query {label} with exception {e}")


class SsbWorkloadOptimizationTests(unittest.TestCase):
    def test_workload(self):
        dbs = db.DBSchema.get_instance(postgres_config_file=".psycopg_connection_tpch", renew=True)
        for label, query in ssb_workload.items():
            try:
                parsed = mosp.MospQuery.parse(query)
                optimized = ues.optimize_query(parsed, dbs=dbs)  # noqa: F841
                original_result_set = dbs.execute_query(query)
                optimized_result_set = dbs.execute_query(str(optimized), join_collapse_limit=1, enable_nestloop="off")
                regression_suite.assert_result_sets_equal(original_result_set, optimized_result_set,
                                                          ordered=parsed.is_ordered())
            except Exception as e:
                self.fail(f"Exception raised on query {label} with exception {e}")


class SnowflaxeQueryOptimizationTests(unittest.TestCase):
    def test_base_query(self):
        dbs = db.DBSchema.get_instance(renew=True)
        query = mosp.MospQuery.parse(job_workload["32a"])
        optimized = ues.optimize_query(query, dbs=dbs)  # noqa: F841


class CrossProductQueryOptimizationTests(unittest.TestCase):
    def test_base_query(self):
        dbs = db.DBSchema.get_instance(renew=True)
        raw_query = """SELECT * FROM info_type it, company_type ct
                       WHERE it.info = 'top 250 rank' AND ct.kind = 'production companies'"""
        query = mosp.MospQuery.parse(raw_query)
        optimized = ues.optimize_query(query, dbs=dbs)  # noqa: F841

    def test_base_with_snowflake(self):
        pass


class WeirdQueriesOptimizationTests(unittest.TestCase):
    def test_no_joins_query(self):
        dbs = db.DBSchema.get_instance(renew=True)
        raw_query = "SELECT * FROM company_type ct WHERE ct.kind = 'production companies'"
        query = mosp.MospQuery.parse(raw_query)
        optimized_query = ues.optimize_query(query, dbs=dbs)  # noqa: F841


class CompoundJoinPredicateOptimizationTests(unittest.TestCase):
    def test_base_query(self):
        dbs = db.DBSchema.get_instance(renew=True)
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
        optimized_query = ues.optimize_query(query, dbs=dbs)  # noqa: F841


class BoundsTrackerTests(unittest.TestCase):
    def test_json_serialization(self):
        query = mosp.MospQuery.parse(job_workload["1a"])
        optimized: ues.OptimizationResult = ues.optimize_query(query, introspective=True)
        jsonized = util.to_json(optimized.bounds)  # noqa: F841

    def test_no_gaps(self):
        query = mosp.MospQuery.parse(job_workload["1a"])
        optimized: ues.OptimizationResult = ues.optimize_query(query, introspective=True)
        optimized_query, bounds = optimized.query, optimized.bounds

        current_join_path = [optimized_query.base_table()]
        self.assertIn(current_join_path, bounds)
        for join in optimized_query.joins():
            if join.is_subquery():
                subquery_join_path = [join.base_table()]
                self.assertIn(subquery_join_path, bounds)
                for subquery_join in join.subquery.joins():
                    subquery_join_path.extend(subquery_join.collect_tables())
                    self.assertIn(subquery_join_path, bounds)
            current_join_path.extend(join.collect_tables())
            self.assertIn(current_join_path, bounds)


if "__name__" == "__main__":
    unittest.main()
