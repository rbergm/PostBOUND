from postbound.db import postgres
from postbound.experiments import workloads
from tests import regression_suite

pg_connect_dir = "."


@regression_suite.skip_if_no_db(f"{pg_connect_dir}/.psycopg_connection_job")
class HintingTests(regression_suite.PlanTestCase):
    def setUp(self) -> None:
        self.pg_instance = postgres.connect(
            config_file=f"{pg_connect_dir}/.psycopg_connection_job", private=True
        )
        self.job = workloads.job()

    def test_pglab_backend(self) -> None:
        if self.pg_instance.hinting().backend != "pg_lab":
            self.skipTest("pg_lab is not available")

        for label, query in self.job.entries():
            with self.subTest("Query", label=label):
                self.pg_instance.reset_connection()
                native_plan = self.pg_instance.optimizer().query_plan(query)
                hinted_query = self.pg_instance.hinting().generate_hints(
                    query, native_plan
                )
                explicit_plan = self.pg_instance.optimizer().query_plan(hinted_query)
                self.assertQueryExecutionPlansEqual(
                    native_plan.canonical(), explicit_plan.canonical()
                )
