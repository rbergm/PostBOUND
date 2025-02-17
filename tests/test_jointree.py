"""Tests for PostBOUND's join tree model / interaction.

Requires a working Postgres instance with IMDB set-up.
"""
import textwrap

import unittest

from postbound import db, qal, optimizer
from postbound.db import postgres
from postbound.optimizer import _jointree

from tests import regression_suite


pg_connect_dir = "."
imdb_config_file = f"{pg_connect_dir}/.psycopg_connection_job"


class JoinTreeLoadingTests(unittest.TestCase):
    def test_load_from_query_plan(self) -> None:
        title = qal.TableReference("title")
        movie_info = qal.TableReference("movie_info")
        t_explain_plan = db.QueryExecutionPlan(node_type="Seq Scan", is_join=False, is_scan=True, children=[],
                                               table=title, physical_operator=optimizer.ScanOperator.SequentialScan)
        mi_explain_plan = db.QueryExecutionPlan(node_type="Seq Scan", is_join=False, is_scan=True, children=[],
                                                table=movie_info, physical_operator=optimizer.ScanOperator.IndexScan)
        explain_plan = db.QueryExecutionPlan(node_type="Nested Loop", is_join=True, is_scan=False,
                                             physical_operator=optimizer.JoinOperator.NestedLoopJoin,
                                             children=[t_explain_plan, mi_explain_plan], inner_child=mi_explain_plan)

        phys_plan = _jointree.PhysicalQueryPlan.load_from_query_plan(explain_plan)

        self.assertIsInstance(phys_plan.root, _jointree.IntermediateJoinNode)
        left_child, right_child = phys_plan.root.left_child, phys_plan.root.right_child
        self.assertIsInstance(left_child, _jointree.BaseTableNode)
        self.assertIsInstance(right_child, _jointree.BaseTableNode)
        self.assertEqual(left_child.table, title)
        self.assertEqual(right_child.table, movie_info)

    @regression_suite.skip_if_no_db(imdb_config_file)
    def test_load_from_explain(self) -> None:
        pg_db = postgres.connect(config_file=imdb_config_file)

        query = qal.parse_query(textwrap.dedent("""SELECT *
                                                   FROM title t
                                                    JOIN movie_companies mc
                                                    ON t.id = mc.movie_id
                                                    JOIN movie_info mi
                                                    ON t.id = mi.movie_id AND mi.movie_id = mc.movie_id"""))

        t = qal.TableReference("title", "t")
        mc = qal.TableReference("movie_companies", "mc")
        mi = qal.TableReference("movie_info", "mi")

        join_order = _jointree.LogicalJoinTree.load_from_list([t, mc, mi])
        annotated_query = pg_db.hinting().generate_hints(query, join_order=join_order)
        explain_plan = pg_db.optimizer().query_plan(annotated_query)

        reconstructed_join_order = _jointree.LogicalJoinTree.load_from_query_plan(explain_plan)

        self.assertEqual(join_order, reconstructed_join_order)


if __name__ == "__main__":
    unittest.main()
