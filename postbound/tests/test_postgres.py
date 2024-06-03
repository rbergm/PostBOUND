
from postbound.db import postgres
from postbound.experiments import workloads
from postbound.optimizer import jointree

from tests import regression_suite

workloads.workloads_base_dir = "../workloads"
pg_connect_dir = "."


@regression_suite.skip_if_no_db(f"{pg_connect_dir}/.psycopg_connection_job")
class HintingTests(regression_suite.PlanTestCase):

    def setUp(self) -> None:
        self.pg_instance = postgres.connect(config_file=f"{pg_connect_dir}/.psycopg_connection_job", private=True)
        self.job = workloads.job()

    def test_pglab_backend(self) -> None:
        if self.pg_instance.hinting().backend != "pg_lab":
            self.skipTest("pg_lab is not available")

        for label, query in self.job.entries():
            with self.subTest("Query", label=label):
                self.pg_instance.reset_connection()
                native_plan = self.pg_instance.optimizer().query_plan(query)
                phys_plan = jointree.PhysicalQueryPlan.load_from_query_plan(native_plan, query, operators_only=True)
                hinted_query = self.pg_instance.hinting().generate_hints(query, phys_plan)
                explicit_plan = self.pg_instance.optimizer().query_plan(hinted_query)
                self.assertQueryExecutionPlansEqual(native_plan, explicit_plan)
