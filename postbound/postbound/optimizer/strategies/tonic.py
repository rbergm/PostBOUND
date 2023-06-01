from __future__ import annotations

import collections
import math
from collections.abc import Iterable, Sequence
from typing import Optional

from postbound.qal import qal, base, predicates, transform
from postbound.db import db
from postbound.optimizer import jointree
from postbound.optimizer.physops import selection as opsel, operators as physops
from postbound.util import collections as collection_utils, dicts as dict_utils


def _left_query_plan_child(node: db.QueryExecutionPlan) -> db.QueryExecutionPlan:
    return node.outer_child if node.outer_child else node.children[0]


def _right_query_plan_child(node: db.QueryExecutionPlan) -> db.QueryExecutionPlan:
    return node.inner_child if node.inner_child else node.children[1]


def _iterate_join_tree(current_node: jointree.AbstractJoinTreeNode) -> Sequence[jointree.IntermediateJoinNode]:
    if isinstance(current_node, jointree.BaseTableNode):
        return []
    assert isinstance(current_node, jointree.IntermediateJoinNode)
    left_child, right_child = current_node.left_child, current_node.right_child
    left_child, right_child = ((right_child, left_child) if right_child.tree_depth() < left_child.tree_depth()
                               else (left_child, right_child))
    return list(_iterate_join_tree(right_child)) + [current_node]


def _iterate_query_plan(current_node: db.QueryExecutionPlan) -> Sequence[db.QueryExecutionPlan]:
    if current_node.is_scan:
        return []
    if not current_node.is_join:
        assert len(current_node.children) == 1
        return _iterate_query_plan(current_node.children[0])
    left_child, right_child = _left_query_plan_child(current_node), _right_query_plan_child(current_node)
    left_child, right_child = ((right_child, left_child) if right_child.plan_depth() < left_child.plan_depth()
                               else (left_child, right_child))
    return list(_iterate_query_plan(right_child)) + [current_node]


def _normalize_filter_predicate(tables: base.TableReference | Iterable[base.TableReference],
                                filter_predicate: Optional[predicates.AbstractPredicate]
                                ) -> Optional[predicates.AbstractPredicate]:
    if not filter_predicate:
        return None
    tables: set[base.TableReference] = set(collection_utils.enlist(tables))
    referenced_tables = tables & filter_predicate.tables()
    renamed_tables = {table: table.drop_alias() for table in referenced_tables}
    renamed_columns = {col: base.ColumnReference(col.name, renamed_tables[col.table])
                       for col in filter_predicate.columns() if col.table in renamed_tables}
    return transform.rename_columns_in_predicate(filter_predicate, renamed_columns)


class QepsIdentifier:
    def __init__(self, tables: base.TableReference | Iterable[base.TableReference],
                 filter_predicate: Optional[predicates.AbstractPredicate] = None) -> None:
        if not tables:
            raise ValueError("Tables required")
        self._tables = frozenset(tab.drop_alias() for tab in collection_utils.enlist(tables))
        self._filter_predicate = _normalize_filter_predicate(tables, filter_predicate)
        self._hash_val = hash((self._tables, self._filter_predicate))

    @property
    def table(self) -> Optional[base.TableReference]:
        if not len(self._tables) == 1:
            return None
        return collection_utils.get_any(self._tables)

    @property
    def tables(self) -> frozenset[base.TableReference]:
        return self._tables

    @property
    def filter_predicate(self) -> Optional[predicates.AbstractPredicate]:
        return self._filter_predicate

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, type(self)) and self.tables == other.tables
                and self.filter_predicate == other.filter_predicate)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        table_str = (self.table.identifier() if len(self.tables) == 1
                     else "#" + "#".join(tab.identifier() for tab in self.tables))
        filter_str = f"[{self.filter_predicate}]" if self.filter_predicate else ""
        return table_str + filter_str


class QepsNode:
    def __init__(self, filter_aware: bool, gamma: float, *,
                 identifier: Optional[QepsIdentifier] = None, parent: Optional[QepsNode] = None) -> None:
        self.filter_aware = filter_aware
        self.gamma = gamma
        self.operator_costs: dict[physops.JoinOperators, float] = collections.defaultdict(float)
        self.child_nodes = dict_utils.DynamicDefaultDict(self._init_qeps)
        self._subquery_root: Optional[QepsNode] = None  # only used for subquery nodes
        self._parent = parent
        self._identifier = identifier

    @property
    def subquery_root(self) -> QepsNode:
        if self._subquery_root is None:
            self._subquery_root = QepsNode(self.filter_aware, self.gamma)
        return self._subquery_root

    def path(self) -> Optional[Sequence[QepsIdentifier]]:
        if not self._identifier:
            return None
        parent_path = self._parent.path() if self._parent else []
        return parent_path + [self._identifier] if parent_path else [self._identifier]

    def tables(self) -> frozenset[base.TableReference]:
        qeps_path = self.path()
        if not qeps_path:
            return frozenset()
        return frozenset(collection_utils.set_union(qeps_id.tables for qeps_id in qeps_path))

    def recommend_operators(self, query: qal.SqlQuery, join_order: Sequence[jointree.IntermediateJoinNode],
                            current_assignment: physops.PhysicalOperatorAssignment, *,
                            _skip_first_table: bool = False) -> None:
        if not join_order:
            return

        next_join, *remaining_joins = join_order
        recommendation = self.current_recommendation()
        if recommendation:
            current_assignment.set_join_operator(physops.JoinOperatorAssignment(recommendation, self.tables()))

        if next_join.is_bushy_join():
            first_child, second_child = next_join.left_child, next_join.right_child
            first_child, second_child = ((second_child, first_child)
                                         if second_child.tree_depth() < first_child.tree_depth()
                                         else (first_child, second_child))
            qeps_subquery_id = QepsIdentifier(first_child.tables())
            qeps_subquery_node = self.child_nodes[qeps_subquery_id]
            qeps_subquery_node.subquery_root.recommend_operators(query, _iterate_join_tree(first_child),
                                                                 current_assignment)
            qeps_subquery_node.recommend_operators(query, remaining_joins, current_assignment)
            return
        elif next_join.is_base_join():
            first_table, second_table = next_join.left_child.table, next_join.right_child.table
            first_table, second_table = ((second_table, first_table) if second_table < first_table
                                         else (first_table, second_table))

            if not _skip_first_table:
                qeps_child_id = self._make_identifier(query, first_table)
                qeps_child_node = self.child_nodes[qeps_child_id]
                qeps_child_node.recommend_operators(query, join_order, current_assignment, _skip_first_table=True)
                return
            else:
                next_table = second_table
        else:
            # join between intermediate (our current QEP-S path) and a base table (next node in our QEP-S path)
            next_table = (next_join.left_child.table if next_join.left_child.is_base_table_node()
                          else next_join.right_child.table)

        qeps_child_id = self._make_identifier(query, next_table)
        qeps_child_node = self.child_nodes[qeps_child_id]
        qeps_child_node.recommend_operators(query, remaining_joins, current_assignment)

    def integrate_costs(self, query: qal.SqlQuery, query_plan: Sequence[db.QueryExecutionPlan], *,
                        _skip_first_table: bool = False) -> None:
        if not query_plan:
            return

        next_node, *remaining_nodes = query_plan
        if not next_node.is_join:
            self.integrate_costs(query, remaining_nodes)

        first_child, second_child = _left_query_plan_child(next_node), _right_query_plan_child(next_node)
        if next_node.is_bushy_join():
            first_child, second_child = ((second_child, first_child)
                                         if second_child.plan_depth() < first_child.plan_depth()
                                         else (first_child, second_child))
            qeps_subquery_id = QepsIdentifier(first_child.tables())
            qeps_subquery_node = self.child_nodes[qeps_subquery_id]
            qeps_subquery_node.update_costs(next_node.physical_operator, next_node.cost)
            qeps_subquery_node.subquery_root.integrate_costs(query, _iterate_query_plan(first_child))
            qeps_subquery_node.integrate_costs(query, remaining_nodes)
            return
        elif next_node.is_base_join():
            first_child, second_child = ((second_child, first_child)
                                         if second_child.fetch_base_table() < first_child.fetch_base_table()
                                         else (first_child, second_child))
            if not _skip_first_table:
                qeps_child_id = self._make_identifier(query, first_child.fetch_base_table())
                qeps_child_node = self.child_nodes[qeps_child_id]
                qeps_child_node.integrate_costs(query, query_plan, _skip_first_table=True)
                return
            else:
                child_node = second_child
        else:
            # join between intermediate (our current QEP-S path) and a base table (next node in our QEP-S path)
            child_node = (first_child if first_child.is_scan_branch() else second_child)

        child_table = child_node.fetch_base_table()
        qeps_child_id = self._make_identifier(query, child_table)
        qeps_child_node = self.child_nodes[qeps_child_id]
        qeps_child_node.update_costs(next_node.physical_operator, next_node.cost)
        qeps_child_node.integrate_costs(query, remaining_nodes)

    def current_recommendation(self) -> Optional[physops.JoinOperators]:
        return dict_utils.argmin(self.operator_costs) if len(self.operator_costs) > 1 else None

    def update_costs(self, operator: physops.JoinOperators, cost: float) -> None:
        if not operator or math.isinf(cost):
            raise ValueError("Operator and cost required")
        current_cost = self.operator_costs[operator]
        self.operator_costs[operator] = cost + self.gamma * current_cost

    def inspect(self, *, _current_indentation: int = 0) -> str:
        if not _current_indentation:
            return "[ROOT]\n" + self._child_inspect(2)

        prefix = " " * _current_indentation

        cost_str = prefix + self._cost_str()
        subquery_content = (self.subquery_root.inspect(_current_indentation=_current_indentation + 2)
                            if self._subquery_root else "")
        subquery_str = f"{prefix}[SQ] ->\n{subquery_content}" if subquery_content else ""
        child_content = self._child_inspect(_current_indentation)
        child_str = f"{prefix}[CHILD] ->\n{child_content}" if child_content else f"{prefix}[no children]"

        inspect_entries = [cost_str, subquery_str, child_str]
        return "\n".join(entry for entry in inspect_entries if entry)

    def _init_qeps(self, identifier: QepsIdentifier) -> QepsNode:
        return QepsNode(self.filter_aware, self.gamma, parent=self, identifier=identifier)

    def _make_identifier(self, query: qal.SqlQuery,
                         table: base.TableReference | Iterable[base.TableReference]) -> QepsIdentifier:
        table = collection_utils.simplify(table)
        filter_predicate = query.predicates().filters_for(table) if self.filter_aware else None
        return QepsIdentifier(table, filter_predicate)

    def _child_inspect(self, indentation: int) -> str:
        prefix = " " * indentation
        child_content = []
        for identifier, child_node in self.child_nodes.items():
            child_inspect = child_node.inspect(_current_indentation=indentation + 2)
            child_content.append(f"{prefix}QEP-S node {identifier}\n{child_inspect}")
        return f"\n{prefix}-----\n".join(child for child in child_content)

    def _cost_str(self) -> str:
        cost_content = ", ".join(f"{operator.value}={cost}" for operator, cost in self.operator_costs.items())
        return f"[{cost_content}]" if self.operator_costs else "[no cost]"

    def __bool__(self) -> bool:
        return len(self.child_nodes) > 0 or len(self.operator_costs) > 0

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        qeps_path = self.path()
        identifier = " -> ".join(str(qeps_id) for qeps_id in qeps_path) if qeps_path else "[ROOT]"
        costs = self._cost_str()
        return f"{identifier} {costs}"


class QueryExecutionPlanSynopsis:
    @staticmethod
    def create(filter_aware: bool, gamma: float) -> QueryExecutionPlanSynopsis:
        root = QepsNode(filter_aware, gamma)
        return QueryExecutionPlanSynopsis(root)

    def __init__(self, root: QepsNode) -> None:
        self.root = root

    def recommend_operators(self, query: qal.SqlQuery,
                            join_order: jointree.JoinTree) -> physops.PhysicalOperatorAssignment:
        current_assignment = (join_order.physical_operators() if isinstance(join_order, jointree.PhysicalQueryPlan)
                              else physops.PhysicalOperatorAssignment())
        self.root.recommend_operators(query, _iterate_join_tree(join_order.root), current_assignment)
        return current_assignment

    def integrate_costs(self, query: qal.SqlQuery, query_plan: db.QueryExecutionPlan) -> None:
        self.root.integrate_costs(query, _iterate_query_plan(query_plan.simplify()))

    def reset(self) -> None:
        self.root = QepsNode(self.root.filter_aware, self.root.gamma)

    def inspect(self) -> str:
        return self.root.inspect()


def make_qeps(path: Iterable[base.TableReference], root: Optional[QepsNode] = None, *, gamma: float = 0.8) -> QepsNode:
    current_node = root if root is not None else QepsNode(False, gamma)
    root = current_node
    for table in path:
        current_node = current_node.child_nodes[QepsIdentifier(table)]
    return root


class TonicOperatorSelection(opsel.PhysicalOperatorSelection):

    def __init__(self, filter_aware: bool = False, gamma: float = 0.8, *,
                 database: Optional[db.Database] = None) -> None:
        super().__init__()
        self.filter_aware = filter_aware
        self.gamma = gamma
        self.qeps = QueryExecutionPlanSynopsis.create(filter_aware, gamma)
        self._db = database if database else db.DatabasePool.get_instance().current_database()

    def integrate_cost(self, query: qal.SqlQuery, query_plan: Optional[db.QueryExecutionPlan] = None) -> None:
        query_plan = self._db.optimizer().query_plan(query) if query_plan is None else query_plan
        self.qeps.integrate_costs(query, query_plan)

    def simulate_feedback(self, query: qal.SqlQuery) -> None:
        analyze_plan = self._db.optimizer().analyze_plan(query)
        self.incorporate_feedback(query, analyze_plan)

    def incorporate_feedback(self, query: qal.SqlQuery, analyze_plan: db.QueryExecutionPlan):
        if not analyze_plan.is_analyze():
            raise ValueError("Analyze plan required, but normal plan received")
        physical_qep = jointree.PhysicalQueryPlan.load_from_query_plan(analyze_plan, query)
        hinted_query = self._db.hinting().generate_hints(query, physical_qep)
        self.integrate_cost(hinted_query)

    def reset(self) -> None:
        self.qeps.reset()

    def _apply_selection(self, query: qal.SqlQuery,
                         join_order: Optional[jointree.LogicalJoinTree | jointree.PhysicalQueryPlan]
                         ) -> physops.PhysicalOperatorAssignment:
        if not join_order or join_order.is_empty():
            join_order = self._obtain_native_join_order(query)
        return self.qeps.recommend_operators(query, join_order)

    def _description(self) -> dict:
        return {"name": "tonic", "filter_aware": self.filter_aware, "gamma": self.gamma}

    def _obtain_native_join_order(self, query: qal.SqlQuery) -> jointree.LogicalJoinTree:
        native_plan = self._db.optimizer().query_plan(query)
        return jointree.LogicalJoinTree.load_from_query_plan(native_plan, query)
