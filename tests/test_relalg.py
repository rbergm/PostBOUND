"""Tests for the relalg module, specifically for parsing queries into relational algebra.

Notice that these tests merely check that the queries are parsed without errors. They do not check the relational algebra
DAGs for correctness. This is because right now we do not have a good and manageable way to check the correctness of these
structures.
"""
from __future__ import annotations

import unittest

from postbound import qal
from postbound.qal import relalg, TableReference, ColumnReference


class RelalgParserTests(unittest.TestCase):
    def test_q1(self):
        # Query Q1 from Neumann, Kemper: "Unnesting Arbitrary Queries", BTW'15
        q1 = qal.parse_query("""
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
        q2 = qal.parse_query("""
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
        tab_s = TableReference("S")
        col_s_a = ColumnReference("a", tab_s)
        col_s_b = ColumnReference("b", tab_s)
        scan_s = relalg.Relation(TableReference("S"), [col_s_a, col_s_b])
        select_s = relalg.Selection(scan_s, qal.as_predicate(col_s_a, qal.LogicalSqlOperators.Equal, 42))

        tab_r = TableReference("R")
        col_r_a = ColumnReference("a", tab_r)
        scan_r = relalg.Relation(TableReference("R"), [col_r_a])

        join_node = relalg.ThetaJoin(select_s, scan_r,
                                     qal.as_predicate(col_s_b, qal.LogicalSqlOperators.Equal, col_r_a))

        additional_selection = relalg.Selection(select_s,
                                                qal.as_predicate(col_s_b, qal.LogicalSqlOperators.Equal, 24))
        new_root = join_node.mutate(left_input=additional_selection).root()

        self.assertNotEqual(join_node, new_root)
        self.assertIsInstance(new_root, relalg.ThetaJoin)

        assert isinstance(new_root, relalg.ThetaJoin)
        self.assertEqual(new_root.left_input, additional_selection)
        self.assertEqual(join_node.left_input, select_s)
        self._assert_sound_tree_linkage(new_root)

    def test_cyclic_tree_modification(self):
        tab_r = TableReference("R")
        col_r_a = ColumnReference("a", tab_r)
        scan_r = relalg.Relation(TableReference("R"), [col_r_a])

        selection_a = relalg.Selection(scan_r, qal.as_predicate(col_r_a, qal.LogicalSqlOperators.Less, 42))
        selection_b = relalg.Selection(scan_r, qal.as_predicate(col_r_a, qal.LogicalSqlOperators.Greater, 24))
        union_node = relalg.Union(selection_a, selection_b)

        additional_projection = relalg.Projection(selection_b, [col_r_a])
        new_root = union_node.mutate(right_input=additional_projection).root()

        self.assertNotEqual(union_node, new_root)
        self.assertIsInstance(new_root, relalg.Union)

        assert isinstance(new_root, relalg.Union)
        self.assertEqual(new_root.right_input, additional_projection)
        self.assertEqual(union_node.right_input, selection_b)
        self._assert_sound_tree_linkage(new_root)

    def test_operator_replacement(self):
        tab_r = TableReference("R")
        tab_s = TableReference("S")
        col_r_a = ColumnReference("a", tab_r)
        col_s_a = ColumnReference("a", tab_s)
        join_pred = qal.as_predicate(col_r_a, qal.LogicalSqlOperators.Equal, col_s_a)

        # Our old relalg tree: Projection(Select(CrossProduct(R, S)))
        scan_r = relalg.Relation(tab_r, [col_r_a])
        scan_s = relalg.Relation(tab_s, [col_s_a])
        cross_product = relalg.CrossProduct(scan_r, scan_s)
        selection = relalg.Selection(cross_product, join_pred)
        old_root = relalg.Projection(selection, [col_r_a])

        # Our new relalg tree: Projection(Join(R, S))
        # The root node receives the new join node as input. The join node receives the previous input of the cross product as
        # its input and the selection's predicate becomes the join predicate
        join_node = relalg.ThetaJoin(cross_product.left_input, cross_product.right_input, join_pred)
        new_root: relalg.RelNode = selection.parent_node.mutate(input_node=join_node).root()

        self.assertIsInstance(new_root, type(old_root))
        self.assertEqual(new_root.tables(), old_root.tables())
        self._assert_sound_tree_linkage(new_root)

    def test_operator_reordering(self):
        tab_r = TableReference("R")
        tab_s = TableReference("S")
        col_r_a = ColumnReference("a", tab_r)
        col_s_a = ColumnReference("a", tab_s)
        filter_pred = qal.as_predicate(col_r_a, qal.LogicalSqlOperators.Less, 42)

        # Our old relalg tree: Select(CrossProduct(R, S))
        scan_r = relalg.Relation(tab_r, [col_r_a])
        scan_s = relalg.Relation(tab_s, [col_s_a])
        cross_product = relalg.CrossProduct(scan_r, scan_s)
        selection = relalg.Selection(cross_product, filter_pred)
        old_root = selection.root()

        # in an actual application, the precise types of the nodes would need to be checked and appropriate reordering actions
        # need to be taken based on the types.
        if not isinstance(old_root, relalg.Selection) or not isinstance(old_root.input_node, relalg.CrossProduct):
            self.fail()

        # the selection now receives the previous input of the cross product node as input, this effectively pushes the
        # selection down the tree.
        # Likewise, the new root becomes the previous cross product with the mutated selection as input. Effectively, this
        # pulls the cross product up in the tree.

        # we use an existing node to mutate directly --> no need for cloning
        new_second_node = old_root.mutate(input_node=old_root.input_node.left_input)
        new_root = old_root.input_node.mutate(left_input=new_second_node, as_root=True).root()

        self.assertIsInstance(new_root, relalg.CrossProduct)
        self.assertEqual(old_root.tables(), new_root.tables())
        self._assert_sound_tree_linkage(new_root)

    def test_intermediate_insert(self):
        tab_r = TableReference("R")
        col_r_a = ColumnReference("a", tab_r)
        col_r_b = ColumnReference("b", tab_r)

        # Original structure: Selection1(Projection1(Selection2(R)))
        scan_r = relalg.Relation(tab_r, [col_r_a, col_r_b])
        selection = relalg.Selection(scan_r, qal.as_predicate(col_r_a, qal.LogicalSqlOperators.Less, 42))
        projection = relalg.Projection(selection, [col_r_a])
        old_root = relalg.Selection(projection, qal.as_predicate(col_r_a, qal.LogicalSqlOperators.Greater, 24))

        # Updated structure: Selection1(Projection1(Projection2(Selection2(R)))
        additional_projection = relalg.Projection(selection, [col_r_a, col_r_b])
        new_root = projection.mutate(input_node=additional_projection).root()

        old_node_sequence = list(old_root.dfs_walk())
        new_node_sequence = list(new_root.dfs_walk())

        self.assertNotEqual(old_root, new_root)
        self.assertTrue(len(old_node_sequence) + 1 == len(new_node_sequence))

    def test_rename_operator(self):
        tab_r = TableReference("R")
        col_r_a = ColumnReference("a", tab_r)
        renamed_col = ColumnReference("renamed", tab_r)
        scan_r = relalg.Relation(tab_r, [col_r_a])
        rename = relalg.Rename(scan_r, {col_r_a: renamed_col})

        expected_expressions = frozenset({qal.ColumnExpression(renamed_col)})
        self.assertEqual(rename.provided_expressions(), expected_expressions)
        self.assertTrue(str(rename))

    def test_no_upwards_modification(self):
        tab_r = TableReference("R")
        col_r_a = ColumnReference("a", tab_r)
        filter_pred = qal.as_predicate(col_r_a, qal.LogicalSqlOperators.Greater, 24)

        # Original structure: Projection(R)
        original_relation = relalg.Relation(tab_r, [col_r_a])
        original_root = relalg.Projection(original_relation, [col_r_a])

        # Updated structure: Projection(Selection(R))
        additional_selection = relalg.Selection(original_root.input_node, filter_pred)
        new_root = original_root.mutate(input_node=additional_selection)
        new_relation = new_root.leaf()

        self.assertNotEqual(original_relation.parent_node, new_relation.parent_node)

    def test_inline_reordering(self):
        tab_r = TableReference("R")
        col_r_a = ColumnReference("a", tab_r)
        tab_s = TableReference("S")
        col_s_a = ColumnReference("a", tab_s)
        le_join_pred = qal.as_predicate(col_r_a, qal.LogicalSqlOperators.LessEqual, col_s_a)
        lt_join_pred = qal.as_predicate(col_r_a, qal.LogicalSqlOperators.Less, col_s_a)

        # Original structure: ThetaJoin(R, Projection(ThetaJoin(R, S))
        scan_r = relalg.Relation(tab_r, [col_r_a])
        scan_s = relalg.Relation(tab_s, [col_s_a])

        sideways_join = relalg.ThetaJoin(scan_r, scan_s, le_join_pred)
        s_projection = relalg.Projection(sideways_join, [col_s_a])

        final_join = relalg.ThetaJoin(scan_r, s_projection, lt_join_pred)
        self._assert_sound_tree_linkage(final_join)
        self.assertIs(final_join.left_input, sideways_join.left_input)

        # Updated structure: ThetaJoin(R, ThetaJoin(R, Projection(S)))
        pushed_down_projection = s_projection.mutate(input_node=sideways_join.right_input)
        pulled_up_join = sideways_join.mutate(right_input=pushed_down_projection)
        updated_root = final_join.mutate(right_input=pulled_up_join)
        self._assert_sound_tree_linkage(updated_root)

        self.assertNotEqual(scan_s.parent_node, pushed_down_projection)
        self.assertNotEqual(scan_r.sideways_pass, pulled_up_join.left_input.sideways_pass)
        self.assertIs(updated_root.left_input, updated_root.right_input.left_input)

    def test_parse_subquery_in_predicates(self):
        query = qal.parse_query("SELECT * FROM R WHERE R.a IN (SELECT S.b FROM S WHERE R.a = S.a)")
        relalg.parse_relalg(query)

    def test_parse_subquery_between_predicates(self):
        query = qal.parse_query("""SELECT *
                                   FROM R
                                   WHERE R.a BETWEEN (SELECT min(S.b) FROM S WHERE R.a = S.a)
                                        AND (SELECT max(S.b) FROM S WHERE R.a = S.a)
                                   """)
        relalg.parse_relalg(query)

    def _assert_sound_tree_linkage(self, root: relalg.RelNode):
        if root.parent_node:
            self.assertIn(root, root.parent_node.children())
        for sideways_node in root.sideways_pass:
            self.assertIn(root, sideways_node.children())

        for child in root.children():
            self.assertTrue(root in child.sideways_pass or child.parent_node == root,
                            f"(parent {child.parent_node!r} -> child {child!r}) != {root!r}")
            self._assert_sound_tree_linkage(child)


if __name__ == "__main__":
    unittest.main()
