import postbound as pb
from postbound.postgres import PostgresConfiguration, PostgresSetting
from tests import regression_suite

pg_connect_dir = "."

# pg_hint_plan is more limited in scope than pg_lab. On a practical level, this means that we cannot force certain operators
# with pg_hint_plan reliably, e.g., memoize nodes. The PG optimizer is free to insert such operators as it sees fit. This
# creates problems for our unit tests because the optimizer might (or might not) use a memoize node for the native plan,
# but decide on the opposite for the hinted plan (because pg_hint_plan modifies the internal cost estimates).
# This applies to memoize nodes, materialize nodes, and parallel plans in general.
# In the end, we might end up with different plans and there is nothing we can do aboout it. Therefore, we take a more
# radical approach and disable all problematic PG features beforehand.
PGHintPlanRestrictions = PostgresConfiguration(
    [
        PostgresSetting("max_parallel_workers_per_gather", 0),
        PostgresSetting("enable_material", "off"),
        PostgresSetting("enable_memoize", "off"),
    ]
)


@regression_suite.skip_if_no_db(f"{pg_connect_dir}/.psycopg_connection_job")
class JobHintingTests(regression_suite.PlanTestCase):
    def setUp(self) -> None:
        self.pg_instance = pb.postgres.connect(
            config_file=f"{pg_connect_dir}/.psycopg_connection_job", private=True
        )
        self.job = pb.workloads.job()

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
                self.assertEqual(native_plan, explicit_plan)

    def test_pg_hint_plan_backend(self) -> None:
        if self.pg_instance.hinting().backend != "pg_hint_plan":
            self.skipTest("pg_hint_plan is not available")

        for label, query in self.job.entries():
            with self.subTest("Query", label=label):
                self.pg_instance.reset_connection()
                self.pg_instance.apply_configuration(PGHintPlanRestrictions)
                native_plan = self.pg_instance.optimizer().query_plan(query)
                hinted_query = self.pg_instance.hinting().generate_hints(
                    query, native_plan
                )
                explicit_plan = self.pg_instance.optimizer().query_plan(hinted_query)
                self.assertEqual(native_plan, explicit_plan)


@regression_suite.skip_if_no_db(f"{pg_connect_dir}/.psycopg_connection_stats")
class StatsHintingTests(regression_suite.PlanTestCase):
    def setUp(self) -> None:
        self.pg_instance = pb.postgres.connect(
            config_file=f"{pg_connect_dir}/.psycopg_connection_stats", private=True
        )
        self.stats = pb.workloads.stats()

    def test_pglab_backend(self) -> None:
        if self.pg_instance.hinting().backend != "pg_lab":
            self.skipTest("pg_lab is not available")

        for label, query in self.stats.entries():
            with self.subTest("Query", label=label):
                self.pg_instance.reset_connection()
                native_plan = self.pg_instance.optimizer().query_plan(query)
                hinted_query = self.pg_instance.hinting().generate_hints(
                    query, native_plan
                )
                explicit_plan = self.pg_instance.optimizer().query_plan(hinted_query)
                self.assertEqual(native_plan, explicit_plan)

    def test_pg_hint_plan_backend(self) -> None:
        if self.pg_instance.hinting().backend != "pg_hint_plan":
            self.skipTest("pg_hint_plan is not available")

        for label, query in self.stats.entries():
            with self.subTest("Query", label=label):
                self.pg_instance.reset_connection()
                self.pg_instance.apply_configuration(PGHintPlanRestrictions)
                native_plan = self.pg_instance.optimizer().query_plan(query)
                hinted_query = self.pg_instance.hinting().generate_hints(
                    query, native_plan
                )
                explicit_plan = self.pg_instance.optimizer().query_plan(hinted_query)
                self.assertEqual(native_plan, explicit_plan)
