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

        additional_selection = relalg.Selection(select_s.clone(),  # we re-use an existing node to create a new one --> clone()
                                                predicates.as_predicate(col_s_b, expressions.LogicalSqlOperators.Equal, 24))
        new_root = join_node.mutate(left_input=additional_selection).root()

        self.assertNotEqual(join_node, new_root)
        self.assertIsInstance(new_root, relalg.ThetaJoin)

        assert isinstance(new_root, relalg.ThetaJoin)
        self.assertEqual(new_root.left_input, additional_selection)
        self.assertEqual(join_node.left_input, select_s)
        self._assert_sound_tree_linkage(new_root)

    def test_cyclic_tree_modification(self):
        tab_r = base.TableReference("R")
        col_r_a = base.ColumnReference("a", tab_r)
        scan_r = relalg.Relation(base.TableReference("R"), [col_r_a])

        selection_a = relalg.Selection(scan_r, predicates.as_predicate(col_r_a, expressions.LogicalSqlOperators.Less, 42))
        selection_b = relalg.Selection(scan_r, predicates.as_predicate(col_r_a, expressions.LogicalSqlOperators.Greater, 24))
        union_node = relalg.Union(selection_a, selection_b)

        additional_projection = relalg.Projection(selection_b.clone(), [col_r_a])
        new_root = union_node.mutate(right_input=additional_projection).root()

        self.assertNotEqual(union_node, new_root)
        self.assertIsInstance(new_root, relalg.Union)

        assert isinstance(new_root, relalg.Union)
        self.assertEqual(new_root.right_input, additional_projection)
        self.assertEqual(union_node.right_input, selection_b)
        self._assert_sound_tree_linkage(new_root)

    def test_operator_replacement(self):
        tab_r = base.TableReference("R")
        tab_s = base.TableReference("S")
        col_r_a = base.ColumnReference("a", tab_r)
        col_s_a = base.ColumnReference("a", tab_s)
        join_pred = predicates.as_predicate(col_r_a, expressions.LogicalSqlOperators.Equal, col_s_a)

        # Our old relalg tree: Projection(Select(CrossProduct(R, S)))
        scan_r = relalg.Relation(tab_r, [col_r_a])
        scan_s = relalg.Relation(tab_s, [col_s_a])
        cross_product = relalg.CrossProduct(scan_r, scan_s)
        selection = relalg.Selection(cross_product, join_pred)
        old_root = relalg.Projection(selection, [col_r_a])

        # Our new relalg tree: Projection(Join(R, S))
        # The root node receives the new join node as input. The join node receives the previous input of the cross product as
        # its input and the selection's predicate becomes the join predicate
        join_node = relalg.ThetaJoin(cross_product.left_input.clone(), cross_product.right_input.clone(), join_pred)
        new_root: relalg.RelNode = selection.parent_node.mutate(input_node=join_node).root()

        self.assertIsInstance(new_root, type(old_root))
        self.assertEqual(new_root.tables(), old_root.tables())
        self._assert_sound_tree_linkage(new_root)

    def test_operator_reordering(self):
        tab_r = base.TableReference("R")
        tab_s = base.TableReference("S")
        col_r_a = base.ColumnReference("a", tab_r)
        col_s_a = base.ColumnReference("a", tab_s)
        filter_pred = predicates.as_predicate(col_r_a, expressions.LogicalSqlOperators.Less, 42)

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
        tab_r = base.TableReference("R")
        col_r_a = base.ColumnReference("a", tab_r)
        col_r_b = base.ColumnReference("b", tab_r)

        scan_r = relalg.Relation(tab_r, [col_r_a, col_r_b])
        selection = relalg.Selection(scan_r, predicates.as_predicate(col_r_a, expressions.LogicalSqlOperators.Less, 42))
        projection = relalg.Projection(selection, [col_r_a])
        old_root = relalg.Selection(projection, predicates.as_predicate(col_r_a, expressions.LogicalSqlOperators.Greater, 24))

        additional_projection = relalg.Projection(selection.clone(), [col_r_a, col_r_b])
        new_root = projection.mutate(input_node=additional_projection).root()

        old_node_sequence = list(old_root.dfs_walk())
        new_node_sequence = list(new_root.dfs_walk())

        self.assertNotEqual(old_root, new_root)
        self.assertTrue(len(old_node_sequence) + 1 == len(new_node_sequence))

    def test_rename_operator(self):
        tab_r = base.TableReference("R")
        col_r_a = base.ColumnReference("a", tab_r)
        renamed_col = base.ColumnReference("renamed", tab_r)
        scan_r = relalg.Relation(tab_r, [col_r_a])
        rename = relalg.Rename(scan_r, {col_r_a: renamed_col})

        expected_expressions = frozenset({expressions.ColumnExpression(renamed_col)})
        self.assertEqual(rename.provided_expressions(), expected_expressions)
        self.assertTrue(str(rename))

    def test_no_upwards_modification(self):
        tab_r = base.TableReference("R")
        col_r_a = base.ColumnReference("a", tab_r)
        filter_pred = predicates.as_predicate(col_r_a, expressions.LogicalSqlOperators.Greater, 24)

        original_relation = relalg.Relation(tab_r, [col_r_a])
        original_root = relalg.Projection(original_relation, [col_r_a])

        additional_selection = relalg.Selection(original_root.input_node, filter_pred)
        new_root = original_root.mutate(input_node=additional_selection)
        new_relation = new_root.leaf()

        self.assertNotEqual(original_relation.parent_node, new_relation.parent_node)

    def _assert_sound_tree_linkage(self, root: relalg.RelNode):
        for child in root.children():
            self.assertEqual(child.parent_node, root, f"(parent {child.parent_node!r} -> child {child!r}) != {root!r}")
            self._assert_sound_tree_linkage(child)


if __name__ == "__main__":
    unittest.main()
