from __future__ import annotations

import collections
import copy
from typing import Iterable

from postbound.db.hints import provider
from postbound.qal import qal, base, clauses, joins, predicates, transform
from postbound.util import collections as collection_utils
from postbound.optimizer import data
from postbound.optimizer.physops import operators


def _build_subquery_alias(tables: Iterable[base.TableReference]) -> str:
    return "_".join(tab.identifier() for tab in sorted(tables))


def _build_column_alias(column: base.ColumnReference) -> str:
    return f"{column.table.identifier()}_{column.name}"


class PostgresRightDeepJoinClauseBuilder:
    def __init__(self, query: qal.ImplicitSqlQuery) -> None:
        self.query = query
        self.joined_tables = set()
        self.available_joins: dict[base.TableReference, list[predicates.AbstractPredicate]] = {}
        self.renamed_columns: dict[base.ColumnReference, base.ColumnReference] = {}
        self.original_column_names: dict[base.ColumnReference, str] = {}

    def for_join_tree(self, join_tree: data.JoinTree) -> clauses.ExplicitFromClause:
        self._setup()
        base_table, *joined_tables = self._build_from_clause(join_tree.root)
        return clauses.ExplicitFromClause(base_table, joined_tables)

    def _setup(self) -> None:
        self.joined_tables = set()
        self.available_joins = collections.defaultdict(list)
        self.renamed_columns = {}
        self.original_column_names = {}

    def _build_from_clause(self, join_node: data.JoinTreeNode) -> list[base.TableReference | joins.Join]:
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
        self._mark_tables_joined(subquery_tables)
        self._perform_column_renaming(join_condition)
        return joins.SubqueryJoin.inner(subquery, subquery_export_name, join_condition)

    def _fetch_filters(self, table: base.TableReference) -> predicates.AbstractPredicate | None:
        table_filters = list(self.query.predicates().filters_for(table))
        if len(self.joined_tables) == 1:
            base_table = collection_utils.simplify(self.joined_tables)
            base_table_filters = list(self.query.predicates().filters_for(base_table))
        else:
            base_table_filters = []
        all_filters = table_filters + base_table_filters
        return predicates.CompoundPredicate.create_and(all_filters) if all_filters else None

    def _fetch_join_predicate(self, tables: base.TableReference | list[base.TableReference]
                              ) -> predicates.AbstractPredicate | None:
        tables = collection_utils.enlist(tables)
        join_predicates = []
        for table in tables:
            join_predicates.extend(self.available_joins[table])
        return predicates.CompoundPredicate.create_and(join_predicates) if join_predicates else None

    def _collect_exported_columns(self, tables: set[base.TableReference]) -> set[base.ColumnReference]:
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

    def _mark_tables_joined(self, tables: base.TableReference | Iterable[base.TableReference]) -> None:
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
        for column in predicate.columns():
            if column in self.renamed_columns:
                renamed_column = self.renamed_columns[column]
                column.name = renamed_column.name
                column.table = renamed_column.table

    def _update_renamed_columns(self, columns: set[base.ColumnReference], target_table: base.TableReference) -> None:
        for column in columns:
            export_name = _build_column_alias(column)
            renamed_column = copy.copy(column)
            renamed_column.name = export_name
            renamed_column.table = target_table
            self.renamed_columns[column] = renamed_column
            self.original_column_names[renamed_column] = column.name


def _enforce_pg_join_order(query: qal.ImplicitSqlQuery, join_order: data.JoinTree) -> qal.ExplicitSqlQuery:
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
    return reordered_query


PG_OPTIMIZER_SETTINGS = {
    operators.JoinOperators.NestedLoopJoin: "enable_nestloop",
    operators.JoinOperators.HashJoin: "enable_hashjoin",
    operators.JoinOperators.SortMergeJoin: "enable_mergejoin",
    operators.ScanOperators.SequentialScan: "enable_seqscan",
    operators.ScanOperators.IndexScan: "enable_indexscan",
    operators.ScanOperators.IndexOnlyScan: "enable_indexonlyscan"
}

PG_OPTIMIZER_HINTS = {
    operators.JoinOperators.NestedLoopJoin: "NestLoop",
    operators.JoinOperators.HashJoin: "HashJoin",
    operators.JoinOperators.SortMergeJoin: "MergeJoin",
    operators.ScanOperators.SequentialScan: "SeqScan",
    operators.ScanOperators.IndexScan: "IndexOnlyScan",
    operators.ScanOperators.IndexOnlyScan: "IndexOnlyScan"
}


def _generate_join_key(tables: Iterable[base.TableReference]) -> str:
    return " ".join(tab.identifier() for tab in tables)


def _escape_setting(setting) -> str:
    if isinstance(setting, float) or isinstance(setting, int):
        return str(setting)
    return f"'{setting}'"


def _generate_pg_operator_hints(query: qal.SqlQuery, join_order: data.JoinTree,
                                physical_operators: operators.PhysicalOperatorAssignment) -> qal.SqlQuery:
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
        return query

    settings_block = clauses.HintBlock("\n".join(settings), True) if settings else None
    hints_block = clauses.HintBlock("\n".join(["/*+"] + hints + ["*/"])) if hints else None
    all_hints = []
    if settings_block:
        all_hints.append(settings_block)
    if hints_block:
        all_hints.append(hints_block)

    hinted_query = copy.deepcopy(query)
    hinted_query.hints = clauses.Hint(all_hints)
    return hinted_query


class PostgresHintProvider(provider.HintProvider):
    def adapt_query(self, query: qal.ImplicitSqlQuery, join_order: data.JoinTree | None,
                    physical_operators: operators.PhysicalOperatorAssignment | None) -> qal.SqlQuery:
        adapted_query = query
        if join_order:
            physical_operators.set_system_settings(join_collapse_limit=1)
            adapted_query = _enforce_pg_join_order(adapted_query, join_order)
        if physical_operators:
            adapted_query = _generate_pg_operator_hints(adapted_query, join_order, physical_operators)
        return adapted_query
