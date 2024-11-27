"""Tests for PostBOUND's join tree model / interaction.

Requires a working Postgres instance with IMDB set-up.
"""
import textwrap

import unittest

from postbound import db
from postbound.qal import base, parser
from postbound.optimizer import jointree, physops

from tests import regression_suite


pg_connect_dir = "."
imdb_config_file = f"{pg_connect_dir}/.psycopg_connection_job"


class JoinTreeLoadingTests(unittest.TestCase):
    def test_load_from_query_plan(self) -> None:
        title = base.TableReference("title")
        movie_info = base.TableReference("movie_info")
        t_explain_plan = db.QueryExecutionPlan(node_type="Seq Scan", is_join=False, is_scan=True, children=[],
                                               table=title, physical_operator=physops.ScanOperators.SequentialScan)
        mi_explain_plan = db.QueryExecutionPlan(node_type="Seq Scan", is_join=False, is_scan=True, children=[],
                                                table=movie_info, physical_operator=physops.ScanOperators.IndexScan)
        explain_plan = db.QueryExecutionPlan(node_type="Nested Loop", is_join=True, is_scan=False,
                                             physical_operator=physops.JoinOperators.NestedLoopJoin,
                                             children=[t_explain_plan, mi_explain_plan], inner_child=mi_explain_plan)

        phys_plan = jointree.PhysicalQueryPlan.load_from_query_plan(explain_plan)

        self.assertIsInstance(phys_plan.root, jointree.IntermediateJoinNode)
        left_child, right_child = phys_plan.root.left_child, phys_plan.root.right_child
        self.assertIsInstance(left_child, jointree.BaseTableNode)
        self.assertIsInstance(right_child, jointree.BaseTableNode)
        self.assertEqual(left_child.table, title)
        self.assertEqual(right_child.table, movie_info)

    @regression_suite.skip_if_no_db(imdb_config_file)
    def test_load_from_explain(self) -> None:
        pg_db = db.postgres.connect(config_file=imdb_config_file)

        query = parser.parse_query(textwrap.dedent("""SELECT *
                                                   FROM title t
                                                    JOIN movie_companies mc
                                                    ON t.id = mc.movie_id
                                                    JOIN movie_info mi
                                                    ON t.id = mi.movie_id AND mi.movie_id = mc.movie_id"""))

        t = base.TableReference("title", "t")
        mc = base.TableReference("movie_companies", "mc")
        mi = base.TableReference("movie_info", "mi")

        join_order = jointree.LogicalJoinTree.load_from_list([t, mc, mi])
        annotated_query = pg_db.hinting().generate_hints(query, join_order)
        explain_plan = pg_db.optimizer().query_plan(annotated_query)

        reconstructed_join_order = jointree.LogicalJoinTree.load_from_query_plan(explain_plan)

        self.assertEqual(join_order, reconstructed_join_order)


if __name__ == "__main__":
    unittest.main()
