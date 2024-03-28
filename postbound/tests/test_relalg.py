"""Tests for the relalg module, specifically for parsing queries into relational algebra.

Notice that these tests merely check that the queries are parsed without errors. They do not check the relational algebra
DAGs for correctness. This is because right now we do not have a good and manageable way to check the correctness of these
structures.
"""
from __future__ import annotations

import unittest

from postbound.qal import base, expressions, parser, predicates, relalg


class RelalgParserTests(unittest.TestCase):
    def test_q1(self):
        # Query Q1 from Neumann, Kemper: "Unnesting Arbitrary Queries", BTW'15
        q1 = parser.parse_query("""
                                select s.name, e.course
                                from students s, exams e
                                where s.id = e.sid and
                                    e.grade = (select min(e2.grade)
                                                from exams e2
                                                where s.id=e2.sid)
                                """)
        relalg.parse_relalg(q1)

    def test_q2(self):
        # Query Q2 from Neumann, Kemper: "Unnesting Arbitrary Queries", BTW'15
        q2 = parser.parse_query("""
                                select s.name, e.course
                                from students s, exams e
                                where s.id=e.sid and
                                    (s.major = 'CS' or s.major = 'Games Eng') and
                                    e.grade >= (select avg(e2.grade) + 1
                                                from exams e2
                                                where s.id = e2.sid or
                                                    (e2.curriculum = s.major and
                                                    s.year > e2.date))
                                """)
        relalg.parse_relalg(q2)

    def test_tree_modification(self):
        tab_s = base.TableReference("S")
        col_s_a = base.ColumnReference("a", tab_s)
        col_s_b = base.ColumnReference("b", tab_s)
        scan_s = relalg.Relation(base.TableReference("S"), [col_s_a, col_s_b])
        select_s = relalg.Selection(scan_s, predicates.as_predicate(col_s_a, expressions.LogicalSqlOperators.Equal, 42))

        tab_r = base.TableReference("R")
        col_r_a = base.ColumnReference("a", tab_r)
        scan_r = relalg.Relation(base.TableReference("R"), [col_r_a])

        join_node = relalg.ThetaJoin(select_s, scan_r,
                                     predicates.as_predicate(col_s_b, expressions.LogicalSqlOperators.Equal, col_r_a))

        additional_selection = relalg.Selection(select_s.mutate(),
                                                predicates.as_predicate(col_s_b, expressions.LogicalSqlOperators.Equal, 24))
        new_root = join_node.mutate(left_child=additional_selection)

        self.assertNotEqual(join_node, new_root)
        self.assertIsInstance(new_root, relalg.ThetaJoin)

        assert isinstance(new_root, relalg.ThetaJoin)
        self.assertEqual(new_root.left_input, additional_selection)
        self.assertEqual(join_node.left_input, select_s)

    def test_cyclic_tree_modification(self):
        tab_r = base.TableReference("R")
        col_r_a = base.ColumnReference("a", tab_r)
        scan_r = relalg.Relation(base.TableReference("R"), [col_r_a])

        selection_a = relalg.Selection(scan_r, predicates.as_predicate(col_r_a, expressions.LogicalSqlOperators.Less, 42))
        selection_b = relalg.Selection(scan_r, predicates.as_predicate(col_r_a, expressions.LogicalSqlOperators.Greater, 24))

        union_node = relalg.Union(selection_a, selection_b)
        additional_projection = relalg.Projection(selection_b.mutate(), [col_r_a])
        new_root = union_node.mutate(right_child=additional_projection)

        self.assertNotEqual(union_node, new_root)
        self.assertIsInstance(new_root, relalg.Union)

        assert isinstance(new_root, relalg.Union)
        self.assertEqual(new_root.right_input, additional_projection)
        self.assertEqual(union_node.right_input, selection_b)
