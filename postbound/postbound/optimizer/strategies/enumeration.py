"""Enumerative optimization strategies provide all possible plans in an exhaustive manner.

These strategies do not make use of any statistics, etc. to generate "good" plans. Instead, they focus on the structure of the
plans to generate new plans.
"""
from __future__ import annotations

import itertools
from collections.abc import Generator, Iterable
from typing import Optional

import networkx as nx

from postbound.db import db
from postbound.qal import base, qal
from postbound.optimizer import jointree, physops


def _merge_nodes(start: jointree.LogicalJoinTree | base.TableReference,
                 end: jointree.LogicalJoinTree | base.TableReference) -> jointree.LogicalJoinTree:
    """Provides a join tree that combines two specific trees or tables.

    This is a shortcut method to merge arbitrary tables or trees without having to check whether a table-based or tree-based
    merge has to be performed.

    Parameters
    ----------
    start : jointree.LogicalJoinTree | base.TableReference
        The first tree to merge. If this is a base table, it will be treated as a join tree of just a scan of that table.
    end : jointree.LogicalJoinTree | base.TableReference
        The second tree to merge. If this is a base table, it will be treated as a join tree of just a scan of that table.

    Returns
    -------
    jointree.LogicalJoinTree
        A join tree combining the input trees. The `start` node will be the left node of the tree and the `end` node will be
        the right node.
    """
    start = jointree.LogicalJoinTree.for_base_table(start) if isinstance(start, base.TableReference) else start
    end = jointree.LogicalJoinTree.for_base_table(end) if isinstance(end, base.TableReference) else end
    return start.join_with_subtree(end)


def _enumerate_join_graph(join_graph: nx.Graph) -> Generator[jointree.JoinTree]:
    """Provides all possible join trees based on a join graph.

    Parameters
    ----------
    join_graph : nx.Graph
        The join graph that should be "optimized". Due to the recursive nature of the method, this graph is not limited to a
        pure join graph as provided by the *qal* module. Instead, it nodes can already by join trees.

    Yields
    ------
    Generator[jointree.JoinTree]
        A possible join tree of the join graph.

    Warnings
    --------
    This algorithm does not work for join graphs that contain cross products (i.e. multiple connected components).

    Notes
    -----
    This algorithm works in a recursive manner: At each step, two connected nodes are selected. For these nodes, a join is
    simulated. This is done by generating a join tree for the nodes and merging them into a single node for the join tree. The
    recursion stops as soon as the graph only consists of a single node. This node represents the join tree for the entire
    graph. Depending on the order in which the edges are selected, a different join tree is produced. The exhaustive nature of
    this algorithm guarantees that all possible orders are selected.
    """
    if len(join_graph.nodes) == 1:
        node = list(join_graph.nodes)[0]
        yield node
        return

    for edge in join_graph.edges:
        start_node, target_node = edge
        merged_graph = nx.contracted_nodes(join_graph, start_node, target_node, self_loops=False, copy=True)

        start_end_tree = _merge_nodes(start_node, target_node)
        start_end_graph = nx.relabel_nodes(merged_graph, {start_node: start_end_tree}, copy=True)
        yield from _enumerate_join_graph(start_end_graph)

        end_start_tree = _merge_nodes(target_node, start_node)
        end_start_graph = nx.relabel_nodes(merged_graph, {start_node: end_start_tree}, copy=True)
        yield from _enumerate_join_graph(end_start_graph)


class ExhaustiveJoinOrderGenerator:
    """Utility service to provide all possible join trees for an input query.

    The service produces a generator that in turn provides the join orders. This is done in the `all_join_orders_for` method.
    The provided join orders include linear, as well as bushy join trees.

    Warnings
    --------
    For now, the underlying algorithm is limited to queries without cross-products.
    """

    def all_join_orders_for(self, query: qal.SqlQuery) -> Generator[jointree.LogicalJoinTree]:
        """Produces a generator for all possible join trees of a query.

        Parameters
        ----------
        query : qal.SqlQuery
            The query to "optimize"

        Yields
        ------
        Generator[jointree.LogicalJoinTree]
            A generator that produces all possible join orders for the input query, including bushy trees.

        Raises
        ------
        ValueError
            If the query contains cross products.

        Warnings
        --------
        For now, the underlying algorithm is limited to queries without cross-products.
        """
        join_graph = query.predicates().join_graph()
        if len(join_graph.nodes) == 0:
            return
        elif len(join_graph.nodes) == 1:
            base_table = list(join_graph.nodes)[0]
            base_annotation = jointree.LogicalBaseTableMetadata(query.predicates().filters_for(base_table))
            join_tree = jointree.LogicalJoinTree.for_base_table(base_table, base_annotation)
            yield join_tree
            return
        elif not nx.is_connected(join_graph):
            raise ValueError("Cross products are not yet supported for random join order generation!")

        join_order_hashes = set()
        join_order_generator = _enumerate_join_graph(join_graph)
        for join_order in join_order_generator:
            current_hash = hash(join_order)
            if current_hash in join_order_hashes:
                continue

            join_order_hashes.add(current_hash)
            yield join_order


class ExhaustiveOperatorEnumerator:
    """Utility service to generate all possible assignments of physical operators for a join order.

    The service produces a generator that in turn provides the operator assignments. This is done in the
    `all_operator_assignments_for` method. The precise properties of the generated assignments depends on the configuration of
    this service. It can be set up to only use a subset of the available operators or to exclude operators for scans or joins
    completely. By default, the service uses all operators that are supported by the target database system.

    Parameters
    ----------
    scan_operators : Optional[Iterable[physops.ScanOperators]], optional
        The scan operators that can be used in the query plans. If this is ``None`` or empty, all scans supported by the
        `database` are used. Likewise, if the iterable contains an operator that is not supported by the database, it is
        exlcuded from generation.
    join_operators : Optional[Iterable[physops.JoinOperators]], optional
        The join operators that can be used in the query plans. If this is ``None`` or empty, all joins supported by the
        `database` are used. Likewise, if the iterable contains an operator that is not supported by the database, it is
        exlcuded from generation.
    include_scans : bool, optional
        Whether the assignment should contain scan operators at all. By default, this is enabled. However, if scans are
        disabled, this overwrites any supplied operators in the `scan_operators` parameter.
    include_joins : bool, optional
        Whether the assignment should contain join operators at all. By default, this is enabled. However, if joins are
        disabled, this overwrites any supplied operators in the `join_operators` parameter.
    database : Optional[db.Database], optional
        The database that should execute the queries in the end. The database connection is necessary to determine the
        operators that are actually supported by the system. If this parameter is omitted, it is inferred from the
        `DatabasePool`.

    Raises
    ------
    ValueError
        If both scans and joins are disabled
    """

    def __init__(self, scan_operators: Optional[Iterable[physops.ScanOperators]] = None,
                 join_operators: Optional[Iterable[physops.JoinOperators]] = None, *,
                 include_scans: bool = True, include_joins: bool = True,
                 database: Optional[db.Database] = None) -> None:
        if not include_joins and not include_scans:
            raise ValueError("Cannot exclude both join hints and scan hints")
        self._db = database if database is not None else db.DatabasePool.get_instance().current_database()
        self._include_scans = include_scans
        self._include_joins = include_joins
        allowed_scan_ops = scan_operators if scan_operators else physops.ScanOperators
        allowed_join_ops = join_operators if join_operators else physops.JoinOperators
        self.allowed_scan_ops = frozenset(scan_op for scan_op in allowed_scan_ops if self._db.hinting().supports_hint(scan_op))
        self.allowed_join_ops = frozenset(join_op for join_op in allowed_join_ops if self._db.hinting().supports_hint(join_op))

    def all_operator_assignments_for(self, query: qal.SqlQuery, join_order: jointree.JoinTree
                                     ) -> Generator[physops.PhysicalOperatorAssignment, None, None]:
        """Produces a generator for all possible operator assignments of the allowed operators.

        The precise structure of the operator assignments depends on the service configuration. Take a look at the class
        documentation for details.

        Parameters
        ----------
        query : qal.SqlQuery
            The query to "optimize"
        join_order : jointree.JoinTree
            The join sequence to use. This contains all required tables to be scanned and joins to be performed.

        Yields
        ------
        Generator[physops.PhysicalOperatorAssignment, None, None]
            A generator producing all possible operator assignments. The assignments will not contain any cost estimates, nor
            will they specify join directions or parallization data.
        """
        if not self._include_scans:
            return self._all_join_assignments_for(query, join_order)
        elif not self._include_joins:
            return self._all_scan_assignments_for(query)

        tables = list(query.tables())
        scan_ops = [list(self.allowed_scan_ops)] * len(tables)
        joins = [join.tables() for join in join_order.join_sequence()]
        join_ops = [list(self.allowed_join_ops)] * len(joins)

        for scan_selection in itertools.product(*scan_ops):
            current_scan_pairs = zip(scan_ops, scan_selection)
            current_scan_assignment = physops.PhysicalOperatorAssignment()
            for table, operator in current_scan_pairs:
                current_scan_assignment.set_scan_operator(physops.ScanOperatorAssignment(operator, table))

            for join_selection in itertools.product(*join_ops):
                current_join_pairs = zip(joins, join_selection)
                current_total_assignment = current_scan_assignment.clone()
                for join, operator in current_join_pairs:
                    current_total_assignment.set_join_operator(physops.JoinOperatorAssignment(operator, join))

                yield current_total_assignment

    def _all_join_assignments_for(self, query: qal.SqlQuery,
                                  join_order: jointree.JoinTree) -> Generator[physops.PhysicalOperatorAssignment, None, None]:
        """Specialized handler for assignments that only contain join operators.

        Parameters
        ----------
        query : qal.SqlQuery
            The query to "optimize"
        join_order : jointree.JoinTree
            The join sequence to use. This contains all required tables to be scanned and joins to be performed.

        Yields
        ------
        Generator[physops.PhysicalOperatorAssignment, None, None]
            A generator producing all possible operator assignments. The assignments will not contain any cost estimates, nor
            will they specify join directions or parallization data.
        """
        joins = [join.tables() for join in join_order.join_sequence()]
        join_ops = [list(self.allowed_join_ops)] * len(joins)
        for join_selection in itertools.product(*join_ops):
            current_join_pairs = zip(joins, join_selection)
            assignment = physops.PhysicalOperatorAssignment()
            for join, operator in current_join_pairs:
                assignment.set_join_operator(physops.JoinOperatorAssignment(operator, join))
            yield assignment

    def _all_scan_assignments_for(self, query: qal.SqlQuery,
                                  join_order: jointree.JoinTree) -> Generator[physops.PhysicalOperatorAssignment, None, None]:
        """Specialized handler for assignments that only contain scan operators.

        Parameters
        ----------
        query : qal.SqlQuery
            The query to "optimize"
        join_order : jointree.JoinTree
            The join sequence to use. This contains all required tables to be scanned and joins to be performed.

        Yields
        ------
        Generator[physops.PhysicalOperatorAssignment, None, None]
            A generator producing all possible operator assignments. The assignments will not contain any cost estimates, nor
            will they specify join directions or parallization data.        """
        tables = list(query.tables())
        scans = [list(self.allowed_scan_ops)] * len(tables)
        for scan_selection in itertools.product(*scans):
            current_scan_pairs = zip(tables, scan_selection)
            assignment = physops.PhysicalOperatorAssignment()
            for table, operator in current_scan_pairs:
                assignment.set_scan_operator(physops.ScanOperatorAssignment(operator, table))
            yield assignment


class ExhaustivePlanEnumerator:
    """Utility service to provide all possible exection plans fora query.

    This service combines the `ExhaustiveJoinOrderGenerator` and `ExhaustiveOperatorEnumerator` into a single high-level
    service. Therefore, it underlies the same restrictions as these two services. The produced generator can be accessed via
    the `all_plans_for` method.


    Parameters
    ----------
    join_order_args : Optional[dict], optional
        Configuration for the `ExhaustiveJoinOrderGenerator`. This is forwarded to the service's ``__init__`` method.
    operator_args : Optional[dict], optional
        Configuration for the `ExhaustiveOperatorEnumerator`. This is forwarded to the service's ``__init__`` method.

    See Also
    --------
    ExhaustiveJoinOrderGenerator
    ExhaustiveOperatorEnumerator
    """

    def __init__(self, *, join_order_args: Optional[dict] = None,
                 operator_args: Optional[dict] = None) -> None:
        join_order_args = join_order_args if join_order_args else {}
        operator_args = operator_args if operator_args else {}

        self._join_order_generator = ExhaustiveJoinOrderGenerator(**join_order_args)
        self._operator_generator = ExhaustiveOperatorEnumerator(**operator_args)

    def all_plans_for(self, query: qal.SqlQuery) -> Generator[jointree.PhysicalQueryPlan, None, None]:
        """Produces a generator for all possible query plans of an input query.

        The structure of the provided plans can be restricted by configuring the underlying services. Consult the class-level
        documentation for details.

        Parameters
        ----------
        query : qal.SqlQuery
            The query to "optimize"

        Yields
        ------
        Generator[jointree.PhysicalQueryPlan, None, None]
            A generator producing all possible query plans
        """
        for join_order in self._join_order_generator.all_join_orders_for(query):
            for operator_assignment in self._operator_generator.all_operator_assignments_for(query, join_order):
                return jointree.PhysicalQueryPlan.load_from_logical_order(join_order, operator_assignment)
