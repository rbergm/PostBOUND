"""Implementation of the TONIC algorithm for learned operator selections [1]_.

References
----------

.. [1] A. Hertzschuch et al.: "Turbo-Charging SPJ Query Plans with Learned Physical Join Operator Selections.", VLDB'2022
"""
from __future__ import annotations

import collections
import itertools
import json
import math
import random
from collections.abc import Iterable, Sequence
from typing import Any, Optional

from .. import jointree
from .._hints import PhysicalOperatorAssignment, JoinOperatorAssignment
from ..._core import JoinOperators
from ..._qep import QueryPlan
from ..._stages import PhysicalOperatorSelection
from ... import db, qal, util
from ...qal import parser as query_parser
from ...qal import TableReference, ColumnReference


# TODO: there should be more documentation of the technical design of the QEP-S structure
# More specifically, this documentation should describe the strategies to integrate subquery nodes, and the QEP-S traversal


def _left_query_plan_child(node: QueryPlan) -> QueryPlan:
    """Infers the left child node for a query execution plan.

    Since query execution plans do not carry a notion of directional children directly, this method applies the following rule:
    If the plan node contains an outer child, this is the left child. Otherwise, the first child is returned. If the node does
    not have at least one children,

    Parameters
    ----------
    node : QueryPlan
        The execution plan node for which the children should be found

    Returns
    -------
    QueryPlan
        The child node

    Raises
    ------
    IndexError
        If the node does not contain any children.
    """
    return node.outer_child


def _right_query_plan_child(node: QueryPlan) -> QueryPlan:
    """Infers the right child node for a query execution plan.

    Since query execution plans do not carry a notion of directional children directly, this method applies the following rule:
    If the plan node contains an inner child, this is the right child. Otherwise, the second child is returned.

    Parameters
    ----------
    node : QueryPlan
        The execution plan node for which the children should be found

    Returns
    -------
    QueryPlan
        The child node

    Raises
    ------
    IndexError
        If the node contains less than two children.
    """
    return node.inner_child


def _iterate_query_plan(current_node: QueryPlan) -> Sequence[QueryPlan]:
    """Provides all joins along the deepest join path in the query plan.

    Parameters
    ----------
    current_node : QueryPlan
        The node from which the iteration should start

    Returns
    -------
    Sequence[QueryPlan]
        The join nodes along the deepest path, starting with the deepest nodes.
    """
    if current_node.is_scan():
        return []
    if not current_node.is_join():
        assert current_node.input_node is not None
        return _iterate_query_plan(current_node.input_node)
    left_child, right_child = _left_query_plan_child(current_node), _right_query_plan_child(current_node)
    left_child, right_child = ((right_child, left_child) if right_child.plan_depth() < left_child.plan_depth()
                               else (left_child, right_child))
    return list(_iterate_query_plan(right_child)) + [current_node]


def _iterate_join_tree(current_node: jointree.AbstractJoinTreeNode) -> Sequence[jointree.IntermediateJoinNode]:
    """Provides all joins along the deepest join path in the join tree.

    Parameters
    ----------
    current_node : jointree.AbstractJoinTreeNode
        The node from which the iteration should start

    Returns
    -------
    Sequence[jointree.IntermediateJoinNode]
        The joins along the deepest path, starting with the deepest nodes.
    """
    if isinstance(current_node, jointree.BaseTableNode):
        return []
    assert isinstance(current_node, jointree.IntermediateJoinNode)
    left_child, right_child = current_node.left_child, current_node.right_child
    left_child, right_child = ((right_child, left_child) if right_child.tree_depth() < left_child.tree_depth()
                               else (left_child, right_child))
    return list(_iterate_join_tree(right_child)) + [current_node]


def _normalize_filter_predicate(tables: TableReference | Iterable[TableReference],
                                filter_predicate: Optional[qal.AbstractPredicate]
                                ) -> Optional[qal.AbstractPredicate]:
    """Removes all alias information from a specific set of tables in a predicate.

    Parameters
    ----------
    tables : TableReference | Iterable[TableReference]
        The tables whose alias information should be removed
    filter_predicate : Optional[qal.AbstractPredicate]
        The predicate from which the alias information should be removed. Can be ``None``, in which case no removal is
        performed.

    Returns
    -------
    Optional[qal.AbstractPredicate]
        The normalized predicate or ``None`` if no predicate was given in the first place.
    """
    if not filter_predicate:
        return None
    tables: set[TableReference] = set(util.enlist(tables))
    referenced_tables = tables & filter_predicate.tables()
    renamed_tables = {table: table.drop_alias() for table in referenced_tables}
    renamed_columns = {col: ColumnReference(col.name, renamed_tables[col.table])
                       for col in filter_predicate.columns() if col.table in renamed_tables}
    return qal.transform.rename_columns_in_predicate(filter_predicate, renamed_columns)


def _tables_in_qeps_path(qeps_path: Sequence[QepsIdentifier]) -> frozenset[TableReference]:
    """Extracts all tables along a QEP-S path

    Parameters
    ----------
    qeps_path : Sequence[QepsIdentifier]
        The path to analyze

    Returns
    -------
    frozenset[TableReference]
        All tables in the path
    """
    return util.set_union(identifier.tables() for identifier in qeps_path)


class QepsIdentifier:
    """Models the identifiers of QEP-S nodes.

    Each identifier can either describe a base table node, or an intermediate join node. This depends on the supplied `tables`.
    A single table corresponds to a base table node, whereas multiple tables corresponds to the join of the individual base
    tables. Furthermore, each identifier can optionally be annotated by a filter predicate that can be used to distinguish two
    identifiers over the same tables.

    Identifiers provide efficient hashing and equality comparisons.

    Parameters
    ----------
    tables : TableReference | Iterable[TableReference]
        The tables that constitute the QEP-S node. Subquery nodes consist of multiple tables (the tables in the subquery) and
        scan nodes consist of a single table
    filter_predicate : Optional[qal.AbstractPredicate], optional
        The filter predicate that is used to restrict the allowed tuples of the base table. This does not have any meaning for
        subquery nodes.

    Raises
    ------
    ValueError
        If no table is supplied (either as a ``None`` argument, or as an empty iterable).
    """

    def __init__(self, tables: TableReference | Iterable[TableReference],
                 filter_predicate: Optional[qal.AbstractPredicate] = None) -> None:
        if not tables:
            raise ValueError("Tables required")
        self._tables = frozenset(tab.drop_alias() for tab in util.enlist(tables))
        self._filter_predicate = _normalize_filter_predicate(tables, filter_predicate)
        self._hash_val = hash((self._tables, self._filter_predicate))

    @property
    def table(self) -> Optional[TableReference]:
        """Get the table that is represented by this base table identifier.

        Returns
        -------
        Optional[TableReference]
            The table or ``None`` if this node corresponds to a subquery node.
        """
        if not len(self._tables) == 1:
            return None
        return util.collections.get_any(self._tables)

    @property
    def tables(self) -> frozenset[TableReference]:
        """Get the tables that represent this identifier.

        Returns
        -------
        frozenset[TableReference]
            The tables. This can be a set of just a single table for base table identifiers, but the set will never be empty.
        """
        return self._tables

    @property
    def filter_predicate(self) -> Optional[qal.AbstractPredicate]:
        """Get the filter predicate that is used to describe this identifier.

        Returns
        -------
        Optional[qal.AbstractPredicate]
            The predicate. May be ``None`` if no predicate exists or was specified. For subquery node identifiers, this should
            always be ``None``.
        """
        return self._filter_predicate

    def is_base_table_id(self) -> bool:
        """Checks, whether this identifier describes a normal base table scan.

        Returns
        -------
        bool
            True if its a base table identifier, false otherwise
        """
        return len(self._tables) == 1

    def is_subquery_id(self) -> bool:
        """Checks, whether this identifier describes a subquery (a branch in the join order).

        Returns
        -------
        bool
            True if its a subquery identifier, false otherwise
        """
        return len(self._tables) > 1

    def __json__(self) -> dict:
        return {"tables": self._tables, "filter_predicate": self._filter_predicate}

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
    """Models the a join path with its learned operator costs.

    QEP-S nodes form a tree structure, with each branch corresponding to a different join path. Each node is identified by
    a `QepsIdentifier` that corresponds to the table or subquery that is joined at this point. The join at each QEP-S node can
    be determined by the tables of its predecessor nodes and the table(s) in its identifier.

    Each node maintains the costs of different join operators that it has learned so far.

    Take a look at the fundamental paper on TONIC [1] for more details on the different parameters.

    Parameters
    ----------
    filter_aware : bool
        Whether child nodes should be created for each joined table (not filter aware), or for each pair of joined table,
        filter predicate on that table (filter aware).
    gamma : float
        Controls the balance betwee new cost information and learned costs for the physical operators.
    identifier : Optional[QepsIdentifier], optional
        The identifier of this node. Can be ``None`` for the root node of the entire QEP-S or for subquery nodes. All other
        nodes should have a valid identifier.
    parent : Optional[QepsNode], optional
        The predecessor node of this node. Can be ``None`` for the root node of the entire QEP-S or for subquery nodes. All
        other nodes should have a valid parent.

    Attributes
    ----------
    operator_costs : dict[JoinOperators, float]
        The learned costs of different physical join operators to perform the join of the current path with the identifier
        relation.
    child_nodes : dict[QepsIdentifier, QepsNode]
        The children of the current QEP-S node. Each child corresponds to a different join path and a join of a different
        (potentially intermediate) relation. Children are created automatically as necessary.

    References
    ----------

    .. A. Hertzschuch et al.: "Turbo-Charging SPJ Query Plans with Learned Physical Join Operator Selections.", VLDB'2022
    """
    def __init__(self, filter_aware: bool, gamma: float, *,
                 identifier: Optional[QepsIdentifier] = None, parent: Optional[QepsNode] = None) -> None:
        self.filter_aware = filter_aware
        self.gamma = gamma
        self.operator_costs: dict[JoinOperators, float] = collections.defaultdict(float)
        self.child_nodes = util.dicts.DynamicDefaultDict(self._init_qeps)
        self._subquery_root: Optional[QepsNode] = None  # only used for subquery nodes
        self._parent = parent
        self._identifier = identifier

    @property
    def subquery_root(self) -> QepsNode:
        """The subquery that starts at the current node.

        Accessing this property means that this node is a subquery root. All child nodes are joins that should be executed
        after the subquery.

        If this node has a subquery root, its identifier should be a subquery identifier.

        Returns
        -------
        QepsNode
            The first table in the subquery.
        """
        if self._subquery_root is None:
            self._subquery_root = QepsNode(self.filter_aware, self.gamma)
        return self._subquery_root

    def is_root_node(self) -> bool:
        """Checks, if the current QEP-S node is a root node

        Returns
        -------
        bool
            Whether the node is a root, i.e. a QEP-S node with no predecessor
        """
        return self._parent is None

    def path(self) -> Sequence[QepsIdentifier]:
        """Provides the join path that leads to the current node.

        This includes all identifiers along the path, including the identifier of the current node.

        Returns
        -------
        Optional[Sequence[QepsIdentifier]]
            All identifiers in sequence starting from the root node. For the root node itself, the path is empty.
        """
        if not self._identifier:
            return []
        parent_path = self._parent.path() if self._parent else []
        return parent_path + [self._identifier] if parent_path else [self._identifier]

    def tables(self) -> frozenset[TableReference]:
        """Provides all tables along the join path that leads to the current node.

        Returns
        -------
        frozenset[TableReference]
            All tables of all identifiers along the path. For the root node, the set is empty. Notice that this does only
            include directly designated tables, i.e. tables from filter predicates are neglected.
        """
        return frozenset(util.set_union(qeps_id.tables for qeps_id in self.path()))

    def recommend_operators(self, query: qal.SqlQuery, join_order: Sequence[jointree.IntermediateJoinNode],
                            current_assignment: PhysicalOperatorAssignment, *,
                            _skip_first_table: bool = False) -> None:
        """Inserts the operator with the minimum cost into an operator assignment.

        This method consumes the join order step-by-step, navigating the QEP-S tree along its path. The recommendation
        automatically continues with the next child node.

        In case of an unkown join order, the QEP-S tree is prepared to store costs of that sequence later on.

        Parameters
        ----------
        query : qal.SqlQuery
            The query for which operators should be recommended. This parameter is necessary to infer the applicable filter
            predicates for base tables.
        join_order : Sequence[jointree.IntermediateJoinNode]
            A path to navigate the QEP-S tree. The recommendation logic consumes the next join and supplies all future joins to
            the applicable child node.
        current_assignment : PhysicalOperatorAssignment
            Operators that have already been recommended. This structure is successively inflated with repeated recommendation
            calls.
        _skip_first_table : bool, optional
            Internal parameter that should only be set by the QEP-S implementation. This parameter is required to correctly
            start the path traversal at the bottom join.
        """
        if not join_order:
            return

        next_join, *remaining_joins = join_order
        recommendation = self.current_recommendation()
        if recommendation:
            current_assignment.set_join_operator(JoinOperatorAssignment(recommendation, self.tables()))

        if next_join.is_bushy_join():
            _, subquery_child = next_join.children_by_depth()
            qeps_subquery_id = QepsIdentifier(subquery_child.tables())
            qeps_subquery_node = self.child_nodes[qeps_subquery_id]
            qeps_subquery_node.subquery_root.recommend_operators(query, _iterate_join_tree(subquery_child),
                                                                 current_assignment)
            qeps_subquery_node.recommend_operators(query, remaining_joins, current_assignment)
            return

        if next_join.is_base_join():
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

    def integrate_costs(self, query: qal.SqlQuery, query_plan: Sequence[QueryPlan], *,
                        _skip_first_table: bool = False) -> None:
        """Updates the internal cost model with the costs of the execution plan nodes.

        Notice that the costs of the plan nodes can be calculated using arbitrary strategies and do not need to originate from
        a physical database system. This allows the usage of arbitrary cost models.

        Parameters
        ----------
        query : qal.SqlQuery
            The query which is used to determine new costs. This parameter is necessary to infer the applicable filter
            predicates for base tables.
        query_plan : Sequence[QueryPlan]
            A sequence of join nodes that provide the updated cost information. The update logic consumes the costs of the
            first join and delegates all further updates to the next child node. This requires all plan nodes to contain cost
            information as well as information about the physical join operators.
        _skip_first_table : bool, optional
            Internal parameter that should only be set by the QEP-S implementation. This parameter is required to correctly
            start the path traversal at the bottom join.

        Raises
        ------
        ValueError
            If plan nodes do not contain information about the join costs, or the join operator.

        Notes
        -----
        The implementation of the cost integration uses a "look ahead" approach. This means that each QEP-S node determines the
        next QEP-S node based on the first join in the plan sequence. This node corresponds to a child node of the current
        QEP-S node. If no such child exists, it will be created. Once the next QEP-S node is determined, it is updated with the
        costs of the plan node. Afterwards, the cost integration continues with the next plan node on the next QEP-S node.
        """
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
            qeps_subquery_node.update_costs(next_node.operator, next_node.estimated_cost)
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
        qeps_child_node.update_costs(next_node.operator, next_node.estimated_cost)
        qeps_child_node.integrate_costs(query, remaining_nodes)

    def detect_unknown_costs(self, query: qal.SqlQuery, join_order: Sequence[jointree.IntermediateJoinNode],
                             allowed_operators: frozenset[JoinOperators],
                             unknown_ops: dict[frozenset[TableReference], frozenset[JoinOperators]],
                             _skip_first_table: bool = False) -> None:
        """Collects all joins in the QEP-S that do not have cost information for all possible operators.

        The missing operators are stored in the `unknown_ops` parameter which is inflated as part of the method execution and
        QEP-S traversal, acting as an *output* parameter.

        Parameters
        ----------
        query : qal.SqlQuery
            The query describing the filter predicates to navigate the QEP-S
        join_order : Sequence[jointree.IntermediateJoinNode]
            The join order to navigate the QEP-S
        allowed_operators : frozenset[JoinOperators]
            Operators for which cost information should exist. If the node does not have a cost information for any of the
            operators, this is an unknown cost
        unknown_ops : dict[frozenset[TableReference], frozenset[JoinOperators]]
            The unknown operators that have been detected so far
        _skip_first_table : bool, optional
            Internal parameter that should only be set by the QEP-S implementation. This parameter is required to correctly
            start the path traversal at the bottom join.
        """
        if not join_order:
            return

        if not self.is_root_node() and not self._parent.is_root_node():
            own_unknown_ops = frozenset([operator for operator in allowed_operators if operator not in self.operator_costs])
            unknown_ops[_tables_in_qeps_path(self.path())] = own_unknown_ops

        next_join, *remaining_joins = join_order
        if next_join.is_bushy_join():
            main_child, subquery_child = next_join.children_by_depth()
            qeps_subquery_id = QepsIdentifier(subquery_child.tables())
            qeps_subquery_node = self.child_nodes[qeps_subquery_id]
            qeps_subquery_node.subquery_root.detect_unknown_costs(query, _iterate_join_tree(subquery_child), allowed_operators,
                                                                  unknown_ops)
            qeps_subquery_node.detect_unknown_costs(query, remaining_joins, allowed_operators, unknown_ops)
            return

        if next_join.is_base_join():
            first_table, second_table = next_join.left_child.table, next_join.right_child.table
            first_table, second_table = ((second_table, first_table) if second_table < first_table
                                         else (first_table, second_table))

            if not _skip_first_table:
                qeps_child_id = self._make_identifier(query, first_table)
                qeps_child_node = self.child_nodes[qeps_child_id]
                qeps_child_node.detect_unknown_costs(query, join_order, allowed_operators, unknown_ops, _skip_first_table=True)
                return
            else:
                next_table = second_table
        else:
            # join between intermediate (our current QEP-S path) and a base table (next node in our QEP-S path)
            next_table = (next_join.left_child.table if next_join.left_child.is_base_table_node()
                          else next_join.right_child.table)

        qeps_child_id = self._make_identifier(query, next_table)
        qeps_child_node = self.child_nodes[qeps_child_id]
        qeps_child_node.detect_unknown_costs(query, remaining_joins, allowed_operators, unknown_ops)

    def current_recommendation(self) -> Optional[JoinOperators]:
        """Provides the operator with the minimum cost.

        Returns
        -------
        Optional[JoinOperators]
            The best operator, or ``None`` if not enough information exists to make a good decision.
        """
        return util.argmin(self.operator_costs) if len(self.operator_costs) > 1 else None

    def update_costs(self, operator: JoinOperators, cost: float) -> None:
        """Updates the cost of a specific operator for this node.

        Parameters
        ----------
        operator : JoinOperators
            The operator whose costs should be updated.
        cost : float
            The new cost information.

        Raises
        ------
        ValueError
            If the cost is not a valid number (e.g. NaN or infinity)
        """
        if not operator or math.isinf(cost) or math.isnan(cost):
            raise ValueError("Operator and cost required")
        current_cost = self.operator_costs[operator]
        self.operator_costs[operator] = cost + self.gamma * current_cost

    def inspect(self, *, _current_indentation: int = 0) -> str:
        """Provides a nice hierarchical representation of the QEP-S structure.

        The representation typically spans multiple lines and uses indentation to separate parent nodes from their
        children.

        Parameters
        ----------
        _current_indentation : int, optional
            Internal parameter to the `inspect` function. Should not be modified by the user. Denotes how deeply
            recursed we are in the QEP-S tree. This enables the correct calculation of the current indentation level.
            Defaults to 0 for the root node.

        Returns
        -------
        str
            A string representatio of the QEP-S
        """
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
        """Generates a new QEP-S node with a specific identifier.

        The new node "inherits" configuration settings from the current node. This includes filter awareness and gamma value.
        Likewise, the node is correctly linked up with the current node.

        Parameters
        ----------
        identifier : QepsIdentifier
            The identifier of the new node

        Returns
        -------
        QepsNode
            The new node
        """
        return QepsNode(self.filter_aware, self.gamma, parent=self, identifier=identifier)

    def _make_identifier(self, query: qal.SqlQuery,
                         table: TableReference | Iterable[TableReference]) -> QepsIdentifier:
        """Generates an identifier for a specific table(s).

        The concrete identifier information depends on the configuration of this node, e.g. regarding the filter behavior.

        Parameters
        ----------
        query : qal.SqlQuery
            The query for which the QEP-S identifier should be created. This parameter is necessary to infer filter predicates
            if necessary.
        table : TableReference | Iterable[TableReference]
            The table that should be stored in the identifier. Subquery identifiers will contain multiple tables, but no filter
            predicate.

        Returns
        -------
        QepsIdentifier
            The identifier
        """
        table = util.simplify(table)
        filter_predicate = query.predicates().filters_for(table) if self.filter_aware else None
        return QepsIdentifier(table, filter_predicate)

    def _child_inspect(self, indentation: int) -> str:
        """Worker method to generate the inspection text for child nodes.

        Parameters
        ----------
        indentation : int
            The current indentation level. This parameter will be increased for deeper levels in the QEP-S hierarchy.

        Returns
        -------
        str
            The inspection text
        """
        prefix = " " * indentation
        child_content = []
        for identifier, child_node in self.child_nodes.items():
            child_inspect = child_node.inspect(_current_indentation=indentation + 2)
            child_content.append(f"{prefix}QEP-S node {identifier}\n{child_inspect}")
        return f"\n{prefix}-----\n".join(child for child in child_content)

    def _cost_str(self) -> str:
        """Generates a human-readable string for the cost information in this node.

        Returns
        -------
        str
            The cost information
        """
        cost_content = ", ".join(f"{operator.value}={cost}" for operator, cost in self.operator_costs.items())
        return f"[{cost_content}]" if self.operator_costs else "[no cost]"

    def __json__(self) -> dict:
        cost_json = {operator.value: cost for operator, cost in self.operator_costs.items()}
        children_json = [{"identifier": qeps_id, "node": node} for qeps_id, node in self.child_nodes.items()]
        return {"costs": cost_json, "children": children_json, "subquery": self._subquery_root}

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
    """The plan synopsis maintains a hierarchy of QEP-S nodes, starting at a single root node.

    Most of the methods this synopsis provides simply delegate to the root node.

    Parameters
    ----------
    root : QepsNode
        The root node of the QEP-S tree. This node does not have any predecessor, nor an identifier.

    See Also
    --------
    QepsNode
    """

    @staticmethod
    def create(filter_aware: bool, gamma: float) -> QueryExecutionPlanSynopsis:
        """Generates a new synopsis with specific settings.

        Parameters
        ----------
        filter_aware : bool
            Whether filter predicates should be included in the QEP-S identifiers.
        gamma : float
            The update factor to balance recency and learning of cost information.

        Returns
        -------
        QueryExecutionPlanSynopsis
            The synopsis
        """
        root = QepsNode(filter_aware, gamma)
        return QueryExecutionPlanSynopsis(root)

    def __init__(self, root: QepsNode) -> None:
        self.root = root

    def recommend_operators(self, query: qal.SqlQuery,
                            join_order: jointree.JoinTree) -> PhysicalOperatorAssignment:
        """Provides the optimal operators according to the current QEP-S for a specific join order.

        Parameters
        ----------
        query : qal.SqlQuery
            The query for which the operators should be optimized
        join_order : jointree.JoinTree
            A join order to traverse the QEP-S

        Returns
        -------
        PhysicalOperatorAssignment
            The best operators as learned by the QEP-S
        """
        current_assignment = (join_order.physical_operators() if isinstance(join_order, jointree.PhysicalQueryPlan)
                              else PhysicalOperatorAssignment())
        self.root.recommend_operators(query, _iterate_join_tree(join_order.root), current_assignment)
        return current_assignment

    def integrate_costs(self, query: qal.SqlQuery, query_plan: QueryPlan) -> None:
        """Updates the cost information of the QEP-S with the costs from the query plan.

        Parameters
        ----------
        query : qal.SqlQuery
            The query correponding to the execution plan
        query_plan : QueryPlan
            An execution plan providing the operators and their costs. This information is used for the QEP-S traversal as well
            as the actual update.
        """
        self.root.integrate_costs(query, _iterate_query_plan(query_plan))

    def detect_unknown_costs(self, query: qal.SqlQuery, join_order: jointree.JoinTree,
                             allowed_operators: set[JoinOperators]
                             ) -> dict[frozenset[TableReference], frozenset[JoinOperators]]:
        """Collects all joins in the QEP-S that do not have cost information for all possible operators.

        Parameters
        ----------
        query : qal.SqlQuery
            The query describing the filter predicates to navigate the QEP-S
        join_order : Sequence[jointree.IntermediateJoinNode]
            The join order to navigate the QEP-S
        allowed_operators : frozenset[JoinOperators]
            Operators for which cost information should exist. If the node does not have a cost information for any of the
            operators, this is an unknown cost

        Returns
        -------
        dict[frozenset[TableReference], frozenset[JoinOperators]]
            A mapping from join to the unknown operators at that join. If a join is not contained in the mapping, it is either
            not contained in the `join_order`, or it has cost information for all operators.
        """
        unknown_costs = {}
        self.root.detect_unknown_costs(query, _iterate_join_tree(join_order.root), allowed_operators, unknown_costs)
        return unknown_costs

    def reset(self) -> None:
        """Removes all learned information from the QEP-S.

        This does not only include cost information, but also the tree structure itself.
        """
        self.root = QepsNode(self.root.filter_aware, self.root.gamma)

    def inspect(self) -> str:
        """Provides a nice hierarchical representation of the QEP-S structure.

        The representation typically spans multiple lines and uses indentation to separate parent nodes from their
        children.

        Returns
        -------
        str
            A string representatio of the QEP-S
        """
        return self.root.inspect()

    def __json__(self) -> dict:
        return {"root": self.root, "gamma": self.root.gamma, "filter_aware": self.root.filter_aware}


def _load_qeps_id_from_json(json_data: dict) -> QepsIdentifier:
    """Creates a QEP-S identifier from its JSON representation.

    This is undoes the JSON-serialization via the ``__json__`` method on identifier instances. Whether to create an identifier
    with a filter predicate or a plain identifier is inferred based on the encoded data. The same applies to whether a
    subquery identifier or a normal base table identifier should be created.

    Parameters
    ----------
    json_data : dict
        The encoded identifier

    Returns
    -------
    QepsIdentifier
        The corresponding identifier object

    Raises
    ------
    ValueError
        If the encoding does not contain any tables
    """
    tables = [query_parser.load_table_json(json_table) for json_table in json_data.get("tables", [])]
    filter_pred = query_parser.load_predicate_json(json_data.get("filter_predicate"), {})
    return QepsIdentifier(tables, filter_pred)


def _load_qeps_from_json(json_data: dict, qeps_id: Optional[QepsIdentifier], parent: Optional[QepsNode],
                         filter_aware: bool, gamma: float) -> QepsNode:
    """Creates a QEP-S node from its JSON representation.

    Parameters
    ----------
    json_data : dict
        The encoded node data
    qeps_id : Optional[QepsIdentifier]
        The identifier of the node. Can be ``None`` for root nodes.
    parent : Optional[QepsNode]
        The parent of the node. Can be ``None`` for root nodes.
    filter_aware : bool
        Whether child identifiers should also consider the filter predicates that are applied to base tables.
    gamma : float
        Mediation factor for recent and previous cost information

    Returns
    -------
    QepsNode
        The node instance

    Raises
    ------
    KeyError
        If any of the child node encodings does not contain an identifier
    KeyError
        If any of the child node encodings does not contain an actual node encoding
    """
    node = QepsNode(filter_aware, gamma, identifier=qeps_id, parent=parent)

    cost_info = {JoinOperators(operator_str): cost for operator_str, cost in json_data.get("costs", {}).items()}
    subquery = (_load_qeps_from_json(json_data["subquery"], None, None, filter_aware, gamma)
                if "subquery" in json_data else None)
    children: dict[QepsIdentifier, QepsNode] = {}
    for child_json in json_data.get("children", []):
        child_id = _load_qeps_id_from_json(child_json["identifier"])
        child_node = _load_qeps_from_json(json_data["node"], child_id, node, filter_aware, gamma)
        children[child_id] = child_node

    node.operator_costs = cost_info
    node._subquery_root = subquery
    node.child_nodes = children
    return node


def make_qeps(path: Iterable[TableReference], root: Optional[QepsNode] = None, *, gamma: float = 0.8) -> QepsNode:
    """Generates a QEP-S for the given join path.

    Parameters
    ----------
    path : Iterable[TableReference]
        The join sequence corresponding to the branch in the QEP-S.
    root : Optional[QepsNode], optional
        An optional root node. If this is specified, a branch below that node is inserted. This can be used to construct bushy
        QEP-S via repeated calls to `make_qeps`.
    gamma : float, optional
        The update factor to balance recency and learning of cost information. Defaults to 0.8

    Returns
    -------
    QepsNode
        The QEP-S. The synopsis is not filter-aware.
    """
    current_node = root if root is not None else QepsNode(False, gamma)
    root = current_node
    for table in path:
        current_node = current_node.child_nodes[QepsIdentifier(table)]
    return root


def _obtain_accurate_cost_estimate(query: qal.SqlQuery, database: db.Database) -> QueryPlan:
    """Determines the cost information for a query based on the actual cardinalities of the execution plan.

    This simulates a cost model with perfect input data.

    Parameters
    ----------
    query : qal.SqlQuery
        The query to generate the estimate for. This should be a query with a hint block that describes the physical query
        plan. However, this is not required.
    database : db.Database
        The database which provides the cost model.

    Returns
    -------
    QueryPlan
        The execution plan with cost information
    """
    analyze_plan = database.optimizer().analyze_plan(query)
    physical_qep = jointree.PhysicalQueryPlan.load_from_query_plan(analyze_plan, query)
    query_with_true_hints = database.hinting().generate_hints(query, physical_qep)
    return database.optimizer().query_plan(query_with_true_hints)


def _generate_all_cost_estimates(query: qal.SqlQuery, join_order: jointree.JoinTree,
                                 available_operators: dict[frozenset[TableReference], frozenset[JoinOperators]],
                                 database: db.Database) -> Iterable[QueryPlan]:
    """Provides all cost estimates based on plans with specific operator combinations.

    The cost estimates are based on the true cardinalities of all intermediate results, i.e. the method first determines the
    true cardinalities for each intermediate. Afterwards, the cost model is queried again with the true cardinalities as input
    while fixing the previous execution plan.

    Parameters
    ----------
    query : qal.SqlQuery
        The query for which the cost estimates should be generated
    join_order : jointree.JoinTree
        The join order to use
    available_operators : dict[frozenset[TableReference], frozenset[JoinOperators]]
        A mapping from joins to allowed operators. All possible combinations will be explored.
    database : db.Database
        The database to use for the query execution and cost estimation.

    Returns
    -------
    Iterable[QueryPlan]
        All query plans with the actual costs.
    """
    plans = []
    joins, operators = list(available_operators.keys()), list(available_operators.values())
    for current_operator_selection in itertools.product(*operators):
        current_join_pairs = zip(joins, current_operator_selection)
        current_assignment = PhysicalOperatorAssignment()
        for join, operator in current_join_pairs:
            current_assignment.set_join_operator(JoinOperatorAssignment(operator, join))
        optimized_query = database.hinting().generate_hints(query, join_order, current_assignment)
        plans.append(_obtain_accurate_cost_estimate(optimized_query, database))
    return plans


def _sample_cost_estimates(query: qal.SqlQuery, join_order: jointree.JoinTree,
                           available_operators: dict[frozenset[TableReference], frozenset[JoinOperators]],
                           n_samples: int, database: db.Database) -> Iterable[QueryPlan]:
    """Generates cost estimates based on sampled plans with specific operator combinations.

    The samples are generated based on random operator selections.

    The cost estimates are based on the true cardinalities of all intermediate results, i.e. the method first determines the
    true cardinalities for each intermediate. Afterwards, the cost model is queried again with the true cardinalities as input
    while fixing the previous execution plan.

    Parameters
    ----------
    query : qal.SqlQuery
        The query for which the cost estimates should be generated
    join_order : jointree.JoinTree
        The join order to use
    available_operators : dict[frozenset[TableReference], frozenset[JoinOperators]]
        A mapping from joins to allowed operators. The actual operator assignments will be sampled from this mapping.
    n_samples : int
        The number of samples to generate. If there are less unique plans than samples requested, only the unique plans are
        sampled. Likewise, if the method fails to generate more samples but the requested number of samples is not yet reached,
        (due to bad luck or the number of theoretically available unique plans being close to the number of requested samples),
        the actual number of sampled plans might also be smaller.
    database : db.Database
        The database to use for the query execution and cost estimation.

    Returns
    -------
    Iterable[QueryPlan]
        Query plans with the actual costs
    """
    plans = []
    sampled_assignments = set()
    n_tries = 0
    max_tries = 3 * n_samples
    while len(plans) < n_samples and n_tries < max_tries:
        n_tries += 1
        current_assignment = PhysicalOperatorAssignment()
        for join, operators in available_operators.items():
            selected_operator = random.choice(list(operators))
            current_assignment.set_join_operator(JoinOperatorAssignment(selected_operator, join))
        current_hash = hash(current_assignment)
        if current_hash in sampled_assignments:
            continue
        else:
            sampled_assignments.add(current_hash)
        optimized_query = database.hinting().generate_hints(query, join_order, current_assignment)
        plans.append(_obtain_accurate_cost_estimate(optimized_query, database))
    return plans


class TonicOperatorSelection(PhysicalOperatorSelection):
    """Implementation of the TONIC/QEP-S learned operator recommendation.

    The implementation supports bushy join orders, plain QEP-S and filter-aware QEP-S

    Parameters
    ----------
    filter_aware : bool, optional
        Whether to use the filter-aware QEP-S or the plain QEP-S. Defaults to ``False``, which creates a plain QEP-S.
    gamma : float, optional
        Cost update factor to mediate the bias towards more recent cost information.
    database : Optional[db.Database], optional
        A database to use for the incorporation of native operator costs. If this parameter is omitted, it will be inferred
        from the database pool.

    References
    ----------

    .. [1] A. Hertzschuch et al.: "Turbo-Charging SPJ Query Plans with Learned Physical Join Operator Selections.", VLDB'2022
    """

    @staticmethod
    def load_model(filename: str, database: Optional[db.Database] = None, *,
                   encoding: str = "utf-8") -> TonicOperatorSelection:
        """Re-generates a pre-trained TONIC QEP-S model from disk.

        The model has to be encoded in a JSON file as generated by the jsonize utility

        Parameters
        ----------
        filename : str
            The file that contains the JSON model
        database : Optional[db.Database], optional
            The database that should be used for trainining the model. If omitted, the database is inferred from the
            `DatabasePool`.
        encoding : str, optional
            Enconding of the model file, by default "utf-8"

        Returns
        -------
        TonicOperatorSelection
            The TONIC model
        """
        json_data: dict = {}
        with open(filename, "r", encoding=encoding) as json_file:
            json_data = json.load(json_file)

        filter_aware = json_data.get("filter_aware", False)
        gamma = json_data.get("gamma", 0.8)
        qeps_root = _load_qeps_from_json(json_data["root"], None, None, filter_aware, gamma)
        qeps = QueryExecutionPlanSynopsis(qeps_root)

        tonic_model = TonicOperatorSelection(filter_aware, gamma, database=database)
        tonic_model.qeps = qeps
        return tonic_model

    def __init__(self, filter_aware: bool = False, gamma: float = 0.8, *,
                 database: Optional[db.Database] = None) -> None:
        super().__init__()
        self.filter_aware = filter_aware
        self.gamma = gamma
        self.qeps = QueryExecutionPlanSynopsis.create(filter_aware, gamma)
        self._db = database if database else db.DatabasePool.get_instance().current_database()

    def integrate_cost(self, query: qal.SqlQuery, query_plan: Optional[QueryPlan] = None) -> None:
        """Uses cost information from a query plan to update the QEP-S costs.

        Notice that the costs stored in the query plan do not need to correspond to native costs. Instead, the costs can be
        calculated using arbitrary cost models.

        Parameters
        ----------
        query : qal.SqlQuery
            The query for which the query plan was created.
        query_plan : Optional[QueryPlan], optional
            The query plan which contains the cost information. If this parameter is omitted, the native optimizer of the
            `database` will be queried to obtain the costs of the input query. Notice that is enables the integration of costs
            for arbitrary query plans by setting the hint block of the query.
        """
        query_plan = self._db.optimizer().query_plan(query) if query_plan is None else query_plan
        self.qeps.integrate_costs(query, query_plan)

    def simulate_feedback(self, query: qal.SqlQuery) -> None:
        """Updates the QEP-S cost information with feedback from a specific query.

        This feedback process operates in two stages: in the first stage, the query is executed in *analyze* mode on the native
        optimizer of the database. This results in two crucial sets of information: the actual physical query plan, as well as
        the true cardinalities at each operator. In the second phase, the same input query is enriched with the former plan
        information, as well as the true cardinalities. For this modified query the native optimizer is once again used to
        obtain a query plan. However, this time the cost information is based on the former query plan, but with the true
        cardinalities. Therefore, it resembles the true cost of the query for the database system. Finally, this cost
        information is used to update the QEP-S.

        Parameters
        ----------
        query : qal.SqlQuery
            The query to obtain the cost for
        """
        analyze_plan = self._db.optimizer().analyze_plan(query)
        physical_qep = jointree.PhysicalQueryPlan.load_from_query_plan(analyze_plan, query)
        hinted_query = self._db.hinting().generate_hints(query, physical_qep)
        self.integrate_cost(hinted_query)

    def explore_costs(self, query: qal.SqlQuery, join_order: Optional[jointree.JoinTree] = None, *,
                      allowed_operators: Optional[Iterable[JoinOperators]] = None,
                      max_combinations: Optional[int] = None) -> None:
        """Generates cost information along a specific path in the QEP-S.

        The cost information is generated based on the native optimizer of the database system while using the true
        cardinalities of the intermediate joins.

        For each QEP-S node operators different join operators are selected, independent on the cost information that is
        already available. If the cost information for an operators does already exist, it is updated according to the normal
        updating logic.

        Parameters
        ----------
        query : qal.SqlQuery
            The query to obtain the cost for
        join_order : Optional[jointree.JoinTree], optional
            The QEP-S path along which the cost should be generated. Defaults to ``None``, in which case the join order of the
            native query optimizer of the database system is used.
        allowed_operators : Optional[Iterable[JoinOperators]], optional
            The operators for which cost information should be generated. If a QEP-S node does not have a cost information for
            one of the operators, it is generated. If the node already has a cost information for the operator, this
            information is left as-is. Defaults to all join operators.
        max_combinations : Optional[int], optional
            The maximum number of operator combinations that should be explored. If more combinations are available, a random
            subset of `max_combinations` many samples is explored.
        """
        join_order = join_order if join_order is not None else self._obtain_native_join_order(query)

        allowed_operators = set(allowed_operators) if allowed_operators else set(JoinOperators)
        supported_operators = {join_op for join_op in JoinOperators if self._db.hinting().supports_hint(join_op)}
        allowed_operators = frozenset(allowed_operators & supported_operators)

        unknown_costs = {intermediate.tables(): allowed_operators for intermediate in join_order.join_sequence()}
        total_unknown_combinations = math.prod([len(unknown_ops) for unknown_ops in unknown_costs.values()])

        query_plans = (_sample_cost_estimates(query, join_order, unknown_costs, max_combinations, self._db)
                       if total_unknown_combinations > max_combinations
                       else _generate_all_cost_estimates(query, join_order, unknown_costs, self._db))
        for plan in query_plans:
            self.integrate_cost(query, plan)

    def reset(self) -> None:
        """Generates a brand new QEP-S."""
        self.qeps.reset()

    def select_physical_operators(self, query: qal.SqlQuery,
                                  join_order: Optional[jointree.LogicalJoinTree]) -> PhysicalOperatorAssignment:
        if not join_order or join_order.is_empty():
            join_order = self._obtain_native_join_order(query)
        return self.qeps.recommend_operators(query, join_order)

    def describe(self) -> dict:
        return {"name": "tonic", "filter_aware": self.filter_aware, "gamma": self.gamma}

    def _obtain_native_join_order(self, query: qal.SqlQuery) -> jointree.LogicalJoinTree:
        """Generates the join order for a specific query based on the native database optimizer.

        Parameters
        ----------
        query : qal.SqlQuery
            The query to obtain the join order for

        Returns
        -------
        jointree.LogicalJoinTree
            The join order the database system would use
        """
        native_plan = self._db.optimizer().query_plan(query)
        return jointree.LogicalJoinTree.load_from_query_plan(native_plan, query)

    def __json__(self) -> Any:
        return self.qeps
