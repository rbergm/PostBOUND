"""Tests for PostBOUND's join tree model / interaction.

Requires a working Postgres instance with IMDB set-up.
"""
import textwrap

import unittest

from postbound.db import postgres
from postbound.qal import base, parser
from postbound.optimizer import jointree


pg_connect_dir = "."


class JoinTreeLoadingTests(unittest.TestCase):
    def test_load_explain(self) -> None:  # TODO better name
        pg_db = postgres.connect(config_file=f"{pg_connect_dir}/.psycopg_connection_job")

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
