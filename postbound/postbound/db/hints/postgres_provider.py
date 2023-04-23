"""PostgreSQL-specific hint generation and query transformation."""
from __future__ import annotations

import collections
import copy
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from postbound.db.hints import provider
from postbound.qal import qal, base, clauses, joins, predicates, transform
from postbound.util import collections as collection_utils
from postbound.util.typing import deprecated
from postbound.optimizer import data
from postbound.optimizer.physops import operators
from postbound.optimizer.planmeta import hints as plan_param


@dataclass
class HintParts:
    """Captures the different kinds of Postgres-hints to collect them more easily."""
    settings: list[str]
    hints: list[str]

    @staticmethod
    def empty() -> HintParts:
        """An empty hint parts object, i.e. no hints have been specified, yet."""
        return HintParts([], [])

    def merge_with(self, other: HintParts) -> HintParts:
        """Combines the hints that are contained in this hint parts object with all hints in the other object.

        This construct new hint parts and leaves the current object unmodified.
        """
        merged_settings = self.settings + [setting for setting in other.settings if setting not in self.settings]
        merged_hints = self.hints + [hint for hint in other.hints if hint not in self.hints]
        return HintParts(merged_settings, merged_hints)


def _build_subquery_alias(tables: Iterable[base.TableReference]) -> str:
    """Generates a unified name for a subquery joining the specified tables."""
    return "_".join(tab.identifier() for tab in sorted(tables))


def _build_column_alias(column: base.ColumnReference) -> str:
    """Generates a unified name for columns that are exported from a subquery.

    This prevents name clashes with other columns of similar name that are processed outside of the subquery.
    """
    return f"{column.table.identifier()}_{column.name}"


@deprecated
class PostgresRightDeepJoinClauseBuilder:
    """Service to generate an explicit from clause for a given join order.

    The join order is traversed in right-deep manner. Any non-base table joins are inserted as subqueries.
    """

    def __init__(self, query: qal.ImplicitSqlQuery) -> None:
        self.query = query
        self.joined_tables = set()
        self.available_joins: dict[base.TableReference, list[predicates.AbstractPredicate]] = {}
        self.renamed_columns: dict[base.ColumnReference, base.ColumnReference] = {}
        self.original_column_names: dict[base.ColumnReference, str] = {}

    def for_join_tree(self, join_tree: data.JoinTree) -> clauses.ExplicitFromClause:
        """Generates the FROM clause based on the join tree. Subqueries are inserted as-needed.

        If a subquery exports specific columns that are used at other places in the query, these columns will have
        been renamed to prevent name clashes with columns from tables that are joined outside of the subquery. While
        this method performs the appropriate renaming within the FROM clause, the renamed attributes have to be applied
        to other clauses such as GROUP BY as well. In order to do so, the performed renamings can be looked up in the
        `renamed_columns` attribute after the FROM clause has been build. Likewise, if a renamed attribute should be
        exported in the SELECT clause, the renaming has to undone to ensure compatibility with other code that uses the
        optimized query (and must refer to the columns by name). The `original_column_names` attribute contains the
        reverse mapping from renamed column to original name and can be used to select the columns AS their true name.

        The linear structure of a FROM clause means that circular joins have to be broken up in some way. In order to
        do so, this service applies the following strategy: each time a new table is joined, all possible join
        predicates on that table are looked up. If one of these predicates only references tables that have already
        been joined as well, that predicate is included in the final join predicate for the table. Likewise, filters
        on the table are included in the ON clause, too.
        """
        self._setup()
        base_table, *joined_tables = self._build_from_clause(join_tree.root)
        return clauses.ExplicitFromClause(base_table, joined_tables)

    def _setup(self) -> None:
        """Initializes all necessary attributes."""
        self.joined_tables = set()
        self.available_joins = collections.defaultdict(list)
        self.renamed_columns = {}
        self.original_column_names = {}

    def _build_from_clause(self, join_node: data.JoinTreeNode) -> list[base.TableReference | joins.Join]:
        """Constructs the actual FROM clause and updates all renamings and available joins as needed."""
        if isinstance(join_node, data.BaseTableNode):
            self._mark_tables_joined(join_node.table)
            return [join_node.table]

        if not isinstance(join_node, data.JoinNode):
            raise ValueError("Unknown join node type: " + str(join_node))

        right_joins = self._build_from_clause(join_node.right_child)
        if isinstance(join_node.left_child, data.JoinNode):
            subquery_join = self._build_subquery_join(join_node.left_child, join_node.join_condition)
            return right_joins + [subquery_join]
        elif isinstance(join_node.left_child, data.BaseTableNode):
            base_table_join = self._build_base_table_join(join_node.left_child, join_node.join_condition)
            return right_joins + [base_table_join]
        else:
            raise ValueError("Unknown join node type: " + str(join_node.left_child))

    def _build_base_table_join(self, base_table_node: data.BaseTableNode,
                               join_condition: predicates.AbstractPredicate) -> joins.TableJoin:
        """Constructs the join statement for the given table.

        The ON clause is build according to the following rules:

        1. all filters on the table are added to the ON clause
        2. the actual join predicate is included in the ON clause
        3. all join predicates that join tables that are already part of the FROM clause are included in the ON clause

        If some of the join columns have to be renamed, this is performed here as well.

        Lastly, this method also updates all the renamings and available joins.
        """
        base_table = base_table_node.table
        table_filters = self._fetch_filters(base_table)
        transitive_join_predicates = self._fetch_join_predicate(base_table)

        all_predicates = set(predicate for predicate in [table_filters, transitive_join_predicates, join_condition]
                             if predicate)
        merged_join_predicate = predicates.CompoundPredicate.create_and(all_predicates)
        merged_join_predicate = transform.flatten_and_predicate(merged_join_predicate)

        self._perform_column_renaming(merged_join_predicate)
        self._mark_tables_joined(base_table)
        return joins.TableJoin.inner(base_table, merged_join_predicate)

    def _build_subquery_join(self, join_node: data.JoinNode,
                             join_condition: predicates.AbstractPredicate) -> joins.SubqueryJoin:
        """Constructs the subquery join statement for the given table.

        The subquery will have the following structure:

        - all columns that are referenced in other parts of the query are exported in the SELECT clause. They will be
        renamed to include their original table name to prevent any name clashes with columns of the same name that
        are exported by other tables in the query
        - the subquery contains all tables that are joined in the current join node branch in its FROM clause
        - the FROM clause itself is structured in an explicit manner once again and the same rules for normal table
        joins apply (see `_build_base_table_join`)

        The subquery does not contain a WHERE clause, GROUP BY, etc. Such operations are part of the outer query.

        If the current join branch contains further subqueries, these are generated in a recursive manner, using the
        same rules.
        """
        subquery_tables = set(join_node.tables())
        subquery_export_name = _build_subquery_alias(subquery_tables)

        exported_columns = self._collect_exported_columns(subquery_tables)
        renamed_columns = [clauses.BaseProjection.column(col, _build_column_alias(col)) for col in exported_columns]

        subquery_joins = self._build_from_clause(join_node)
        base_table, *additional_joins = subquery_joins

        select_clause = clauses.Select(renamed_columns)
        from_clause = clauses.ExplicitFromClause(base_table, additional_joins)
        subquery = qal.ExplicitSqlQuery(select_clause=select_clause, from_clause=from_clause)

        self._update_renamed_columns(exported_columns, base.TableReference.create_virtual(subquery_export_name))
        self._mark_tables_joined(list(subquery_tables))
        self._perform_column_renaming(join_condition)
        return joins.SubqueryJoin.inner(subquery, subquery_export_name, join_condition)

    def _fetch_filters(self, table: base.TableReference) -> predicates.AbstractPredicate | None:
        """Provides all the filter predicates specified on the current table.

        It only a single table has been placed in the FROM clause so far, this also includes all filters on that table.
        """
        table_filters = list(self.query.predicates().filters_for(table))
        if len(self.joined_tables) == 1:
            base_table = collection_utils.simplify(self.joined_tables)
            base_table_filters = list(self.query.predicates().filters_for(base_table))
        else:
            base_table_filters = []
        all_filters = table_filters + base_table_filters
        all_filters = [filter_pred for filter_pred in all_filters if self._can_include_predicate(filter_pred, table)]
        return predicates.CompoundPredicate.create_and(all_filters) if all_filters else None

    def _fetch_join_predicate(self, tables: base.TableReference | list[base.TableReference]
                              ) -> predicates.AbstractPredicate | None:
        """Provides all available join predicates on the given tables."""
        tables = collection_utils.enlist(tables)
        join_predicates = []
        for table in tables:
            join_predicates.extend(self.available_joins[table])
        join_predicates = [join_pred for join_pred in join_predicates
                           if self._can_include_predicate(join_pred, tables)]
        return predicates.CompoundPredicate.create_and(join_predicates) if join_predicates else None

    def _collect_exported_columns(self, tables: set[base.TableReference]) -> set[base.ColumnReference]:
        """Provides all columns from the given tables that are used at some other place in the query."""
        columns_in_predicates = set()
        for table in tables:
            join_predicates = self.query.predicates().joins_for(table)
            for join_predicate in join_predicates:
                join_partners = set(column.table for column in join_predicate.join_partners_of(table))
                if not join_partners <= tables:
                    columns_in_predicates |= join_predicate.columns_of(table)

        columns_in_select_clause = set()
        for column in self.query.select_clause.columns():
            if column.table in tables:
                columns_in_select_clause.add(column)

        columns_in_grouping = set()
        if self.query.groupby_clause:
            for group_expression in self.query.groupby_clause.group_columns:
                columns_in_grouping |= {column for column in group_expression.columns() if column.table in tables}
        if self.query.having_clause:
            having_columns = self.query.having_clause.condition.columns()
            columns_in_grouping |= {column for column in having_columns if column.table in tables}

        columns_in_ordering = set()
        if self.query.orderby_clause:
            for order_expression in self.query.orderby_clause.expressions:
                columns_in_grouping |= {col for col in order_expression.column.columns() if col.table in tables}

        return columns_in_select_clause | columns_in_predicates | columns_in_grouping | columns_in_ordering

    def _mark_tables_joined(self, tables: base.TableReference | list[base.TableReference]) -> None:
        """Marks the given tables as joined, potentially making new join predicates to other tables available."""
        tables = collection_utils.enlist(tables)
        for table in tables:
            all_join_predicates = self.query.predicates().joins_for(table)
            for predicate in all_join_predicates:
                join_partner = set(column.table for column in predicate.join_partners_of(table))
                predicate_was_joined = len(join_partner & self.joined_tables) > 0
                predicate_is_joined_now = len(join_partner & set(tables)) > 0
                if not predicate_was_joined and not predicate_is_joined_now:
                    join_partner = collection_utils.simplify(join_partner)
                    self.available_joins[join_partner].append(predicate)
            self.joined_tables.add(table)

    def _perform_column_renaming(self, predicate: predicates.AbstractPredicate) -> None:
        """Renames all columns in the predicate according to the currently available renamings."""
        for column in predicate.columns():
            if column in self.renamed_columns:
                renamed_column = self.renamed_columns[column]
                column.name = renamed_column.name
                column.table = renamed_column.table

    def _update_renamed_columns(self, columns: set[base.ColumnReference], target_table: base.TableReference) -> None:
        """Renames all of the given columns to refer to the target table (i.e. the virtual subquery table) instead."""
        for column in columns:
            export_name = _build_column_alias(column)
            renamed_column = copy.copy(column)
            renamed_column.name = export_name
            renamed_column.table = target_table
            self.renamed_columns[column] = renamed_column
            self.original_column_names[renamed_column] = column.name

    def _can_include_predicate(self, predicate: predicates.AbstractPredicate,
                               joined_tables: base.TableReference | Iterable[base.TableReference]) -> bool:
        """Checks, whether the given predicate can already be included in the join tree.

        This asserts that all the tables that are required by the predicate are either already joined, or being
        joined right now.
        """
        joined_tables = set(collection_utils.enlist(joined_tables))
        return predicate.required_tables() < self.joined_tables | joined_tables


@deprecated
def _enforce_pg_join_order(query: qal.ImplicitSqlQuery,
                           join_order: data.JoinTree) -> tuple[qal.ExplicitSqlQuery, Optional[HintParts]]:
    """Generates the explicit join order for the given query and performs all necessary renaming operations."""
    join_order_builder = PostgresRightDeepJoinClauseBuilder(query)

    from_clause = join_order_builder.for_join_tree(join_order)
    select_clause = copy.deepcopy(query.select_clause)
    for projection in select_clause.targets:
        if projection.target_name:
            continue
        for column in projection.columns():
            if column in join_order_builder.original_column_names:
                projection.target_name = join_order_builder.original_column_names[column]
                break

    where_clause = None

    groupby_clause = copy.deepcopy(query.groupby_clause)
    if groupby_clause:
        transform.rename_columns_in_clause(groupby_clause, join_order_builder.renamed_columns)
    having_clause = copy.deepcopy(query.having_clause)
    if having_clause:
        transform.rename_columns_in_clause(having_clause, join_order_builder.renamed_columns)
    orderby_clause = copy.deepcopy(query.orderby_clause)
    if orderby_clause:
        transform.rename_columns_in_clause(orderby_clause, join_order_builder.renamed_columns)

    reordered_query = qal.ExplicitSqlQuery(select_clause=select_clause,
                                           from_clause=from_clause,
                                           where_clause=where_clause,
                                           groupby_clause=groupby_clause,
                                           having_clause=having_clause,
                                           orderby_clause=orderby_clause,
                                           limit_clause=copy.deepcopy(query.limit_clause))
    return reordered_query, None


def _is_hash_join(join_tree_node: data.JoinTreeNode,
                  operator_assignment: Optional[operators.PhysicalOperatorAssignment]) -> bool:
    """Checks, whether the given node should be executed as a hash join.

    Fails gracefully for base table and unspecified operator assignments (by returning False).
    """
    return operator_assignment and operator_assignment[join_tree_node.tables()] == operators.JoinOperators.HashJoin


def _generate_leading_hint_content(join_tree_node: data.JoinTreeNode,
                                   operator_assignment: Optional[operators.PhysicalOperatorAssignment] = None) -> str:
    """Builds part of the Leading hint to enforce join order and join direction for the given join node."""
    if isinstance(join_tree_node, data.JoinNode):
        left, right = join_tree_node.left_child, join_tree_node.right_child
        left_hint = _generate_leading_hint_content(left, operator_assignment)
        right_hint = _generate_leading_hint_content(right, operator_assignment)
        left_bound = left.upper_bound if left.upper_bound and not np.isnan(left.upper_bound) else -np.inf
        right_bound = right.upper_bound if right.upper_bound and not np.isnan(right.upper_bound) else np.inf

        # for Postgres, the inner relation of a Hash join is the one that gets the hash table and the outer relation is
        # the one being probed. For all other joins, the inner/outer relation actually is the inner/outer relation
        # Therefore, we want to have the smaller relation as the inner relation for hash joins and the other way around
        # for all other joins

        if _is_hash_join(join_tree_node, operator_assignment):
            left_hint, right_hint = (left_hint, right_hint) if right_bound > left_bound else (right_hint, left_hint)
        elif left_bound > right_bound:
            left_hint, right_hint = right_hint, left_hint

        return f"({left_hint} {right_hint})"
    elif isinstance(join_tree_node, data.BaseTableNode):
        return join_tree_node.table.identifier()
    else:
        raise ValueError(f"Unknown join tree node: {join_tree_node}")


def _generate_pg_join_order_hint(query: qal.SqlQuery, join_order: data.JoinTree,
                                 operator_assignment: Optional[operators.PhysicalOperatorAssignment] = None
                                 ) -> tuple[qal.SqlQuery, Optional[HintParts]]:
    """Generates the Leading hint to enforce join order and join direction for the given query.

    This function needs access to the operator assignment in addition to the join tree, because the actual join
    directions in the leading hint depend on the selected join operators.

    More specifically, the join tree assumes that the left join partner of a join node acts as the outer relation
    whereas the right partner acts as the inner relation. For hash joins this means that the inner relation should be
    probed whereas the hash table is created for the outer relation. However, Postgres denotes the directions
    exactly the other way around. Therefore, the direction has to be swapped for hash joins.
    """
    if not join_order.root or len(join_order) < 2:
        return query, None
    leading_hint = _generate_leading_hint_content(join_order.root, operator_assignment)
    leading_hint = f"Leading({leading_hint})"
    hints = HintParts([], [leading_hint])
    return query, hints


PG_OPTIMIZER_SETTINGS = {
    operators.JoinOperators.NestedLoopJoin: "enable_nestloop",
    operators.JoinOperators.HashJoin: "enable_hashjoin",
    operators.JoinOperators.SortMergeJoin: "enable_mergejoin",
    operators.ScanOperators.SequentialScan: "enable_seqscan",
    operators.ScanOperators.IndexScan: "enable_indexscan",
    operators.ScanOperators.IndexOnlyScan: "enable_indexonlyscan"
}
"""Denotes all (session-global) optimizer settings that modify the allowed physical operators."""

# based on PG_HINT_PLAN extension (https://github.com/ossc-db/pg_hint_plan)
# see https://github.com/ossc-db/pg_hint_plan#hints-list for details
PG_OPTIMIZER_HINTS = {
    operators.JoinOperators.NestedLoopJoin: "NestLoop",
    operators.JoinOperators.HashJoin: "HashJoin",
    operators.JoinOperators.SortMergeJoin: "MergeJoin",
    operators.ScanOperators.SequentialScan: "SeqScan",
    operators.ScanOperators.IndexScan: "IndexOnlyScan",
    operators.ScanOperators.IndexOnlyScan: "IndexOnlyScan"
}
"""Denotes all physical operators that can be enforced for individual parts of a query.

These settings overwrite the session-global optimizer settings.
"""


def _generate_join_key(tables: Iterable[base.TableReference]) -> str:
    """Builds a PG_HINT_PLAN-compatible identifier for the join consisting of the given tables."""
    return " ".join(tab.identifier() for tab in tables)


def _escape_setting(setting) -> str:
    """Transforms the setting variable into a string that can be used in an SQL query."""
    if isinstance(setting, float) or isinstance(setting, int):
        return str(setting)
    elif isinstance(setting, bool):
        return "TRUE" if setting else "FALSE"
    return f"'{setting}'"


def _generate_pg_operator_hints(physical_operators: operators.PhysicalOperatorAssignment) -> HintParts:
    """Generates the hints and preparatory statements to enforce the selected optimization in Postgres."""
    settings = []
    for operator, enabled in physical_operators.global_settings.items():
        setting = "on" if enabled else "off"
        operator_key = PG_OPTIMIZER_SETTINGS[operator]
        settings.append(f"SET {operator_key} = '{setting}';")
    for operator, setting in physical_operators.system_specific_settings.items():
        setting = _escape_setting(setting)
        settings.append(f"SET {operator} = {setting};")

    hints = []
    for table, scan_operator in physical_operators.scan_operators.items():
        table_key = table.identifier()
        scan_operator = PG_OPTIMIZER_HINTS[scan_operator]
        hints.append(f"{scan_operator}({table_key})")

    if hints:
        hints.append("")
    for join, join_operator in physical_operators.join_operators.items():
        join_key = _generate_join_key(join)
        join_operator = PG_OPTIMIZER_HINTS[join_operator]
        hints.append(f"{join_operator}({join_key})")

    if not settings and not hints:
        return HintParts.empty()

    return HintParts(settings, hints)


def _generate_pg_parameter_hints(plan_parameters: plan_param.PlanParameterization) -> HintParts:
    """Produces the cardinality and parallelization hints for Postgres."""
    hints = []
    for join, cardinality_hint in plan_parameters.cardinality_hints.items():
        if len(join) < 2:
            # pg_hint_plan can only generate cardinality hints for joins
            continue
        join_key = _generate_join_key(join)
        hints.append(f"Rows({join_key} #{cardinality_hint})")

    for join, num_workers in plan_parameters.parallel_worker_hints.items():
        if len(join) != 1:
            # pg_hint_plan can only generate parallelization hints for single tables
            continue
        table: base.TableReference = collection_utils.simplify(join)
        hints.append(f"Parallel({table.identifier()} {num_workers} hard)")
    return HintParts([], hints)


def _generate_hint_block(parts: HintParts) -> Optional[clauses.Hint]:
    """Constructs the hint block for the given hint parts"""
    settings, hints = parts.settings, parts.hints
    if not settings and not hints:
        return None
    settings_block = "\n".join(settings)
    hints_block = "\n".join(["/*+"] + ["  " + hint for hint in hints] + ["*/"]) if hints else ""
    return clauses.Hint(settings_block, hints_block)


def _apply_hint_block_to_query(query: qal.SqlQuery, hint_block: Optional[clauses.Hint]) -> qal.SqlQuery:
    """Generates a new query with the given hint block."""
    return transform.replace_clause(query, hint_block) if hint_block else query


class PostgresHintProvider(provider.HintProvider):
    """PostgreSQL implementation of the `HintProvider`.

    The query transformation works as follows:

    - the join order is enforced by transforming the implicit SQL query into a query that uses the JOIN ON syntax
    - the Postgres optimizer is forced to adhere to that order via the `join_collapse_limit` setting / hint
    - all operator hints are enforced via the PG_HINT_PLAN Postgres extension and receive hints of the appropriate
    syntax
    """

    def adapt_query(self, query: qal.SqlQuery, *, join_order: data.JoinTree | None = None,
                    physical_operators: operators.PhysicalOperatorAssignment | None = None,
                    plan_parameters: plan_param.PlanParameterization | None = None) -> qal.SqlQuery:
        adapted_query = query
        hint_parts = None

        if join_order:
            # not needed any more, un-comment if old join order strategy should become necessary again:
            # >>> physical_operators.set_system_settings(join_collapse_limit=1)
            adapted_query, hint_parts = _generate_pg_join_order_hint(adapted_query, join_order, physical_operators)

        hint_parts = hint_parts if hint_parts else HintParts.empty()
        if physical_operators:
            operator_hints = _generate_pg_operator_hints(physical_operators)
            hint_parts = hint_parts.merge_with(operator_hints)

        if plan_parameters:
            plan_hints = _generate_pg_parameter_hints(plan_parameters)
            hint_parts = hint_parts.merge_with(plan_hints)

        hint_block = _generate_hint_block(hint_parts)
        adapted_query = _apply_hint_block_to_query(adapted_query, hint_block)
        return adapted_query
