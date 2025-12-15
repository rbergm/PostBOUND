"""Provides a collection of utilities related to query optimization."""

from __future__ import annotations

import collections
import math
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any, Literal, Optional

import Levenshtein

from .. import parser, transform, util
from .._core import ColumnReference, PhysicalOperator, TableReference
from .._hints import JoinTree
from .._qep import QueryPlan
from ..db import Database, DatabasePool
from ..qal import (
    AbstractPredicate,
    BinaryPredicate,
    ColumnExpression,
    CompoundPredicate,
    SqlQuery,
    StaticValueExpression,
    Where,
)
from ..util import StateError


def possible_plans_bound(
    query: SqlQuery,
    *,
    join_operators: set[str] = {"nested-loop join", "hash join", "sort-merge join"},
    scan_operators: set[str] = {"sequential scan", "index scan"},
) -> int:
    """Computes a quick upper bound on the maximum number of possible query execution plans for a given query.

    This upper bound is a very coarse one, based on three assumptions:

    1. any join sequence (even involving cross-products) of any form (i.e. right-deep, bushy, ...) is allowed
    2. the choice of scan operators and join operators can be varied freely
    3. each table can be scanned using arbitrary operators

    The number of real-world query execution plans will typically be much smaller, because cross-products are only
    used if really necessary and the selected join operator influences the scan operators and vice-versa.

    Parameters
    ----------
    query : SqlQuery
        The query for which the bound should be computed
    join_operators : set[str], optional
        The allowed join operators, by default {"nested-loop join", "hash join", "sort-merge join"}
    scan_operators : set[str], optional
        The allowed scan operators, by default {"sequential scan", "index scan"}

    Returns
    -------
    int
        An upper bound on the number of possible query execution plans
    """
    n_tables = len(query.tables())

    join_orders = util.stats.catalan_number(n_tables)
    joins = (n_tables - 1) * len(join_operators)
    scans = n_tables * len(scan_operators)

    return join_orders * joins * scans


def actual_plan_cost(
    query: SqlQuery, analyze_plan: QueryPlan, *, database: Optional[Database] = None
) -> float:
    """Utility to compute the true cost of a query plan based on the actual cardinalities.

    Parameters
    ----------
    query : SqlQuery
        The query to analyze
    analyze_plan : QueryPlan
        The executed query which also contains the true cardinalities
    database : Optional[Database], optional
        The database providing the cost model. If omitted, the database is inferred from the database pool.

    Returns
    -------
    float
        _description_
    """
    if not analyze_plan.is_analyze():
        raise ValueError("The provided plan is not an ANALYZE plan")
    database = database if database is not None else DatabasePool().get_instance()
    hinted_query = database.hinting().generate_hints(
        query, analyze_plan.with_actual_card()
    )
    return database.optimizer().cost_estimate(hinted_query)


def text_diff(left: str, right: str, *, sep: str = " | ") -> str:
    """Merges two text snippets to allow for a comparison on a per-line basis.

    The two snippets are split into their individual lines and then merged back together.

    Parameters
    ----------
    left : str
        The text snippet to display on the left-hand side.
    right : str
        The text snippet to display on the right-hand side.
    sep : str, optional
        The separator to use between the left and right text snippets, by default `` | ``.

    Returns
    -------
    str
        The combined text snippet
    """
    left_lines = left.splitlines()
    right_lines = right.splitlines()

    max_left_len = max(len(line) for line in left_lines)
    left_lines_padded = [line.ljust(max_left_len) for line in left_lines]

    merged_lines = [
        f"{left_line}{sep}{right_line}"
        for left_line, right_line in zip(left_lines_padded, right_lines)
    ]
    return "\n".join(merged_lines)


def star_query_cardinality(
    query: SqlQuery,
    fact_table_pk_column: ColumnReference,
    *,
    database: Optional[Database] = None,
    verbose: bool = False,
) -> int:
    """Utility function to manually compute the cardinality of a star query.

    This function is intended for situations where the database is unable to compute the cardinality because the intermediates
    involved in the query become to large or the query plans are simply too bad. It operates by manually computing the number
    of output tuples for each of the entries in the fact table by sequentially joining the fact table with each dimension
    table.

    Parameters
    ----------
    query : SqlQuery
        The query to compute the cardinality for. This is assumed to be a **SELECT \\*** query and the actual **SELECT** clause
        is ignored completely.
    fact_table_pk_column : ColumnReference
        The fact table's primary key column. All dimension tables must perform an equi-join on this column.
    database : Optional[Database], optional
        The actual database. If this is omitted, the current database from the database pool is used.
    verbose : bool, optional
        Whether progress information should be printed during the computation. If this is enabled, the function will report
        every 1000th value processed.

    Returns
    -------
    int
        The cardinality (i.e. number of output tuples) of the query

    Warnings
    --------
    Currently, this function works well for simple SPJ-based queries, more complicated features might lead to wrong results.
    Similarly, only pure star queries are supported, i.e. there has to be one central fact table and each dimension table
    performs exactly one equi-join with the fact table's primary key. There may not be additional joins on the dimension
    tables. If such additional dimension joins exist, they have to be pre-processed (e.g. by introducing materialized views)
    and the query has to be rewritten to operate on the views instead.
    It is the user's responsibility to ensure that the query is well-formed in these regards.
    """
    logger = util.make_logger(verbose, prefix=util.timestamp)
    database = (
        DatabasePool().get_instance().current_database()
        if database is None
        else database
    )
    fact_table = (
        fact_table_pk_column.table
        if fact_table_pk_column.is_bound()
        else database.schema().lookup_column(fact_table_pk_column, query.tables())
    )
    if fact_table is None:
        raise ValueError(
            f"Cannot infer fact table from column '{fact_table_pk_column}'"
        )
    fact_table_pk_column = fact_table_pk_column.bind_to(fact_table)

    id_vals_query = parser.parse_query(f"""
                                    SELECT {fact_table_pk_column}, COUNT(*) AS card
                                    FROM {fact_table}
                                    GROUP BY {fact_table_pk_column}""")
    if query.predicates().filters_for(fact_table):
        filter_clause = Where(query.predicates().filters_for(fact_table))
        id_vals_query = transform.add_clause(id_vals_query, filter_clause)
    id_vals: list[tuple[Any, int]] = database.execute_query(id_vals_query)

    base_query_fragments: dict[AbstractPredicate, SqlQuery] = {}
    for join_pred in query.predicates().joins_for(fact_table):
        join_partner = join_pred.join_partners_of(fact_table)
        if not len(join_partner) == 1:
            raise ValueError("Currently only singular joins are supported")

        partner_table: ColumnReference = util.simplify(join_partner).table
        query_fragment = transform.extract_query_fragment(
            query, [fact_table, partner_table]
        )
        base_query_fragments[join_pred] = transform.as_count_star_query(query_fragment)

    total_cardinality = 0
    total_ids = len(id_vals)
    for value_idx, (id_value, current_card) in enumerate(id_vals):
        if value_idx % 1000 == 0:
            logger("--", value_idx, "out of", total_ids, "values processed")

        id_filter = BinaryPredicate.equal(
            ColumnExpression(fact_table_pk_column),
            StaticValueExpression(id_value),
        )

        for join_pred, base_query in base_query_fragments.items():
            if current_card == 0:
                break

            expanded_predicate = CompoundPredicate.create_and(
                [base_query.where_clause.predicate, id_filter]
            )
            expanded_where_clause = Where(expanded_predicate)

            dimension_query = transform.replace_clause(
                base_query, expanded_where_clause
            )
            dimension_card = database.execute_query(dimension_query)

            current_card *= dimension_card

        total_cardinality += current_card

    return total_cardinality


def jointree_similarity_topdown(
    a: JoinTree, b: JoinTree, *, symmetric: bool = False, gamma: float = 1.1
) -> float:
    """Computes the similarity of two join trees using a top-down approach.

    Parameters
    ----------
    a : JoinTree
        The first join tree
    b : JoinTree
        The second join tree
    symmetric : bool, optional
        Whether the calculation should be symmetric. If true, the occurence of joins in different branches is not
        penalized. See Notes for details.
    gamma : float, optional
        The reinforcement factor to prioritize similarity of earlier (i.e. deeper) joins. The higher the value, the
        stronger the amplification, by default 1.1

    Returns
    -------
    float
        An artificial similarity score in [0, 1]. Higher values indicate larger similarity.

    Notes
    -----
    TODO: add discussion of the algorithm
    """
    tables_a, tables_b = a.tables(), b.tables()
    total_n_tables = len(tables_a | tables_b)
    normalization_factor = 1 / total_n_tables

    # similarity between two leaf nodes
    if len(tables_a) == 1 and len(tables_b) == 1:
        return 1 if tables_a == tables_b else 0

    # similarity between leaf node and intermediate node
    if len(tables_a) == 1 or len(tables_b) == 1:
        leaf_tree = a if len(tables_a) == 1 else b
        intermediate_tree = b if leaf_tree == a else a

        inner_score = util.jaccard(
            leaf_tree.tables(), intermediate_tree.inner_child.tables()
        )
        outer_score = util.jaccard(
            leaf_tree.tables(), intermediate_tree.outer_child.tables()
        )

        return normalization_factor * max(inner_score, outer_score)

    # similarity between two intermediate nodes
    a_inner, a_outer = a.inner_child, a.outer_child
    b_inner, b_outer = b.inner_child, b.outer_child

    symmetric_score = util.jaccard(a_inner.tables(), b_inner.tables()) + util.jaccard(
        a_outer.tables(), b_outer.tables()
    )
    crossover_score = (
        util.jaccard(a_inner.tables(), b_outer.tables())
        + util.jaccard(a_outer.tables(), b_inner.tables())
        if symmetric
        else 0
    )
    node_score = normalization_factor * max(symmetric_score, crossover_score)

    if symmetric and crossover_score > symmetric_score:
        child_score = jointree_similarity_topdown(
            a_inner, b_outer, symmetric=symmetric, gamma=gamma
        ) + jointree_similarity_topdown(
            a_outer, b_inner, symmetric=symmetric, gamma=gamma
        )
    else:
        child_score = jointree_similarity_topdown(
            a_inner, b_inner, symmetric=symmetric, gamma=gamma
        ) + jointree_similarity_topdown(
            a_outer, b_outer, symmetric=symmetric, gamma=gamma
        )

    return node_score + gamma * child_score


def jointree_similarity_bottomup(a: JoinTree, b: JoinTree) -> float:
    """Computes the similarity of two join trees based on a bottom-up approach.

    Parameters
    ----------
    a : JoinTree
        The first join tree to compare
    b : JoinTree
        The second join tree to compare

    Returns
    -------
    float
        An artificial similarity score in [0, 1]. Higher values indicate larger similarity.

    Notes
    -----
    TODO: add discussion of the algorithm
    """
    a_subtrees = {join.tables() for join in a.iterjoins()}
    b_subtrees = {join.tables() for join in b.iterjoins()}
    return util.jaccard(a_subtrees, b_subtrees)


def linearized_levenshtein_distance(a: JoinTree, b: JoinTree) -> int:
    """Computes the levenshtein distance of the table sequences of two join trees.

    Parameters
    ----------
    a : JoinTree
        The first join tree to compare
    b : JoinTree
        The second join tree to compare

    Returns
    -------
    int
        The distance score. Higher values indicate larger distance.

    References
    ----------

    .. Levenshtein distance: https://en.wikipedia.org/wiki/Levenshtein_distance
    """
    return Levenshtein.distance(a.itertables(), b.itertables())


_DepthState = collections.namedtuple("_DepthState", ["current_level", "depths"])
"""Keeps track of the current calculated depths of different base tables."""


def _traverse_join_tree_depth(
    current_node: JoinTree, current_depth: _DepthState
) -> _DepthState:
    """Calculates a new depth state for the current join tree node based on the current depth.

    This is the handler method for `join_depth`.

    Depending on the specific node, different calculations are applied:

    - for base tables, a new entry of depth one is inserted into the depth state
    - for intermediate nodes, the children are visited to integrate their depth states. Afterwards, their depth is
      increase to incoporate the join

    Parameters
    ----------
    current_node : JoinTree
        The node whose depth information should be integrated
    current_depth : _DepthState
        The current depth state

    Returns
    -------
    _DepthState
        The updated depth state

    Raises
    ------
    TypeError
        If the node is neither a base table node, nor an intermediate join node. This indicates that the class
        hierarchy of join tree nodes was expanded, and this method was not updated properly.
    """
    if current_node.is_scan():
        return _DepthState(1, current_depth.depths | {current_node.base_table: 1})

    if current_node.is_join():
        raise TypeError("Unknown current node type: " + str(current_node))

    inner_child, outer_child = current_node.inner_child, current_node.outer_child
    if current_node.is_base_join():
        return _DepthState(
            1,
            current_depth.depths
            | {inner_child.base_table: 1, outer_child.base_table: 1},
        )
    elif inner_child.is_scan():
        outer_depth = _traverse_join_tree_depth(outer_child, current_depth)
        updated_depth = outer_depth.current_level + 1
        return _DepthState(
            updated_depth, outer_depth.depths | {inner_child.base_table: updated_depth}
        )
    elif outer_child.is_scan():
        inner_depth = _traverse_join_tree_depth(inner_child, current_depth)
        updated_depth = inner_depth.current_level + 1
        return _DepthState(
            updated_depth, inner_depth.depths | {outer_child.table: updated_depth}
        )
    else:
        inner_depth = _traverse_join_tree_depth(inner_child, current_depth)
        outer_depth = _traverse_join_tree_depth(outer_child, current_depth)
        updated_depth = max(inner_depth.current_level, outer_depth.current_level) + 1
        return _DepthState(updated_depth, inner_depth.depths | outer_depth.depths)


def join_depth(join_tree: JoinTree) -> dict[TableReference, int]:
    """Calculates for each base table in a join tree the join index when it was integrated into an intermediate result.

    For joins of two base tables, the depth value is 1. If a table is joined with the intermediate result of the base
    table join, its depth is 2. Generally speaking, the depth of each table is 1 plus the maximum depth of any table
    in the intermediate result that the new table is joined with.

    Parameters
    ----------
    join_tree : JoinTree
        The join tree for which the depths should be calculated.

    Returns
    -------
    dict[TableReference, int]
        A mapping from tables to their depth values.

    Examples
    --------
    TODO add examples
    """
    if join_tree.is_empty():
        return {}
    return _traverse_join_tree_depth(join_tree, _DepthState(0, {})).depths


@dataclass
class PlanChangeEntry:
    """Models a single diff between two join trees.

    The compared join trees are referred two as the left tree and the right tree, respectively.

    Attributes
    ----------
    change_type : Literal["tree-structure", "join-direction", "physical-op", "card-est"]
        Describes the precise difference between the trees. *tree-structure* indicates that the two trees are fundamentally
        different. This occurs when the join orders are not the same. *join-direction* means that albeit the join orders are
        the same, the roles in a specific join are reversed: the inner relation of one tree acts as the outer relation in the
        other one and vice-versa. *physical-op* means that two structurally identical nodes (i.e. same join or base table)
        differ in the assigned physical operator. *card-est* indicates that two structurally identifcal nodes (i.e. same join
        or base table) differ in the estimated cardinality, while *cost-est* does the same, just for the estimated cost.
    left_state : frozenset[TableReference] | PhysicalOperator | float
        Depending on the `change_type` this attribute describes the left tree. For example, for different tree structures,
        these are the tables in the left subtree, for different physical operators, this is the operator assigned to the node
        in the left tree and so on. For different join directions, this is the entire join node
    right_state : frozenset[TableReference] | PhysicalOperator | float
        Equivalent attribute to `left_state`, just for the right tree.
    context : Optional[frozenset[TableReference]], optional
        For different physical operators or cardinality estimates, this describes the intermediate that is different. This
        attribute is unset by default.
    """

    change_type: Literal[
        "tree-structure",
        "join-direction",
        "physical-op",
        "card-est",
        "cost-est",
        "actual-card",
    ]
    left_state: frozenset[TableReference] | PhysicalOperator | float
    right_state: frozenset[TableReference] | PhysicalOperator | float
    context: Optional[frozenset[TableReference]] = None

    def inspect(self) -> str:
        """Provides a human-readable string of the diff.

        Returns
        -------
        str
            The diff
        """
        match self.change_type:
            case "tree-structure":
                left_str = [tab.identifier() for tab in self.left_state]
                right_str = [tab.identifier() for tab in self.right_state]
                return f"Different subtrees: left={left_str} right={right_str}"
            case "join-direction":
                left_str = [tab.identifier() for tab in self.left_state]
                right_str = [tab.identifier() for tab in self.right_state]
                return f"Swapped join direction: left={left_str} right={right_str}"
            case "physical-op":
                return f"Different physical operators on node {self.context}: left={self.left_state} right={self.right_state}"
            case "card-est":
                return (
                    f"Different cardinality estimates on node {self.context}: "
                    f"left={self.left_state} right={self.right_state}"
                )
            case "cost-est":
                return (
                    f"Different cost estimates on node {self.context}: "
                    f"left={self.left_state} right={self.right_state}"
                )
            case "actual-card":
                return (
                    f"Different actual cardinality on node {self.context}: "
                    f"left={self.left_state} right={self.right_state}"
                )
            case _:
                raise StateError(f"Unknown change type '{self.change_type}'")


@dataclass
class PlanChangeset:
    """Captures an arbitrary amount of join tree diffs.

    Attributes
    ----------
    changes : Collection[JointreeChangeEntry]
        The diffs
    """

    changes: Collection[PlanChangeEntry]

    def inspect(self) -> str:
        """Provides a human-readable string of the entire diff.

        The diff will typically contain newlines to separate individual entries.

        Returns
        -------
        str
            The diff
        """
        return "\n".join(entry.inspect() for entry in self.changes)


def compare_query_plans(left: QueryPlan, right: QueryPlan) -> PlanChangeset:
    """Computes differences between two query execution plans.

    Parameters
    ----------
    left : QueryPlan
        The first plan to compare
    right : QueryPlan
        The second plan to compare

    Returns
    -------
    JointreeChangeset
        A diff between the two join trees
    """
    # FIXME: query plans might contain auxiliary nodes that are currently not handled/recognized
    if left.find_first_node(lambda node: node.is_auxiliary()) or right.find_first_node(
        lambda node: node.is_auxiliary()
    ):
        raise ValueError(
            "Comparison of query plans with auxiliary (i.e. non-join and non-scan) operators "
            "is currently not supported"
        )

    if left.tables() != right.tables():
        changeset = [
            PlanChangeEntry(
                "tree-structure", left_state=left.tables(), right_state=right.tables()
            )
        ]
        return PlanChangeset(changeset)

    changes: list[PlanChangeEntry] = []

    left_card_est, right_card_est = (
        left.estimated_cardinality,
        right.estimated_cardinality,
    )
    left_card_actual, right_card_actual = (
        left.actual_cardinality,
        right.actual_cardinality,
    )
    left_cost, right_cost = left.estimated_cost, right.estimated_cost
    if left_card_est != right_card_est and not (
        math.isnan(left_card_est) and math.isnan(right_card_est)
    ):
        changes.append(
            PlanChangeEntry(
                "card-est",
                left_state=left_card_est,
                right_state=right_card_est,
                context=left.tables(),
            )
        )
    if left_card_actual != right_card_actual and not (
        math.isnan(left_card_actual) and math.isnan(right_card_actual)
    ):
        changes.append(
            PlanChangeEntry(
                "actual-card",
                left_state=left_card_actual,
                right_state=right_card_actual,
                context=left.tables(),
            )
        )
    if left_cost != right_cost and not (
        math.isnan(left_cost) and math.isnan(left_cost)
    ):
        changes.append(
            PlanChangeEntry(
                "cost-est",
                left_state=left_cost,
                right_state=right_cost,
                context=left.tables(),
            )
        )

    left_op, right_op = left.node_type, right.node_type
    if left_op != right_op:
        changes.append(
            PlanChangeEntry(
                "physical-op",
                left_state=left_op,
                right_state=right_op,
                context=left.tables(),
            )
        )

    if left.is_join():
        # we can also assume that right is an intermediate node since we know both nodes have the same tables and the left tree
        # is an intermediate node

        join_direction_swap = left.inner_child.tables() == right.outer_child.tables()
        if join_direction_swap:
            changes.append(
                PlanChangeEntry("join-direction", left_state=left, right_state=right)
            )
            changes.extend(
                compare_query_plans(left.inner_child, right.outer_child).changes
            )
            changes.extend(
                compare_query_plans(left.outer_child, right.inner_child).changes
            )
        else:
            changes.extend(
                compare_query_plans(left.inner_child, right.inner_child).changes
            )
            changes.extend(
                compare_query_plans(left.outer_child, right.inner_child).changes
            )

    return PlanChangeset(changes)
