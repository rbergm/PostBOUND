"""`transform` provides utilities to generate SQL queries from other queries."""

from __future__ import annotations

import copy
import typing
from typing import Iterable

from postbound.db import db
from postbound.qal import qal, base, clauses, expressions as expr, joins, predicates as preds
from postbound.util import collections as collection_utils

_Q = typing.TypeVar("_Q", bound=qal.SqlQuery)


def flatten_and_predicate(predicate: preds.AbstractPredicate) -> preds.AbstractPredicate:
    if not isinstance(predicate, preds.CompoundPredicate):
        return predicate

    compound_predicate: preds.CompoundPredicate = predicate
    not_operation = compound_predicate.operation == expr.LogicalSqlCompoundOperators.Not
    or_operation = compound_predicate.operation == expr.LogicalSqlCompoundOperators.Or
    if not_operation or or_operation:
        return compound_predicate

    flattened_children = set()
    for child in compound_predicate.children:
        if child.is_compound() and child.operation == expr.LogicalSqlCompoundOperators.And:
            compound_child: preds.CompoundPredicate = child
            flattened_child = flatten_and_predicate(compound_child)
            if isinstance(flattened_child, preds.CompoundPredicate):
                flattened_children |= set(flattened_child.children)
            else:
                flattened_children.add(flattened_child)
        else:
            flattened_children.add(child)

    if len(flattened_children) == 1:
        return collection_utils.simplify(flattened_children)

    return preds.CompoundPredicate.create_and(flattened_children)


def explicit_to_implicit(source_query: qal.ExplicitSqlQuery) -> qal.ImplicitSqlQuery:
    original_from_clause: clauses.ExplicitFromClause = source_query.from_clause
    additional_predicates = []
    complete_from_tables = [original_from_clause.base_table]

    for joined_table in original_from_clause.joined_tables:
        if joined_table.is_subquery_join():
            raise ValueError("Transforming joined subqueries to implicit table references is not support yet")
        table_join: joins.TableJoin = joined_table
        complete_from_tables.append(table_join.joined_table)
        additional_predicates.append(table_join.join_condition)

    final_from_clause = clauses.ImplicitFromClause(complete_from_tables)

    if source_query.where_clause:
        final_predicate = preds.CompoundPredicate.create_and([source_query.where_clause.predicate]
                                                             + additional_predicates)
    else:
        final_predicate = preds.CompoundPredicate.create_and(additional_predicates)

    final_predicate = flatten_and_predicate(final_predicate)
    final_where_clause = clauses.Where(final_predicate)

    return qal.ImplicitSqlQuery(select_clause=source_query.select_clause, from_clause=final_from_clause,
                                where_clause=final_where_clause,
                                groupby_clause=source_query.groupby_clause, having_clause=source_query.having_clause,
                                orderby_clause=source_query.orderby_clause, limit_clause=source_query.limit_clause)


def query_to_mosp(source_query: qal.SqlQuery) -> dict:
    pass


def _get_predicate_fragment(predicate: preds.AbstractPredicate,
                            referenced_tables: set[base.TableReference]) -> preds.AbstractPredicate | None:
    if not isinstance(predicate, preds.CompoundPredicate):
        return predicate if predicate.tables().issubset(referenced_tables) else None

    compound_predicate: preds.CompoundPredicate = predicate
    child_fragments = [_get_predicate_fragment(child, referenced_tables) for child in compound_predicate.children]
    child_fragments = [fragment for fragment in child_fragments if fragment]
    return preds.CompoundPredicate(compound_predicate.operation, child_fragments) if child_fragments else None


def extract_query_fragment(source_query: qal.ImplicitSqlQuery,
                           referenced_tables: Iterable[base.TableReference]) -> qal.ImplicitSqlQuery | None:
    referenced_tables = set(referenced_tables)
    if not referenced_tables.issubset(source_query.tables()):
        return None

    select_fragment = []
    for target in source_query.select_clause.targets:
        if target.tables() == referenced_tables or not target.columns():
            select_fragment.append(target)

    if select_fragment:
        select_clause = clauses.Select(select_fragment, source_query.select_clause.projection_type)
    else:
        select_clause = clauses.Select.star()

    if source_query.from_clause:
        from_clause = clauses.ImplicitFromClause([tab for tab in source_query.tables() if tab in referenced_tables])
    else:
        from_clause = None

    if source_query.where_clause:
        predicate_fragment = _get_predicate_fragment(source_query.where_clause.predicate, referenced_tables)
        where_clause = clauses.Where(predicate_fragment) if predicate_fragment else None
    else:
        where_clause = None

    if source_query.groupby_clause:
        group_column_fragment = [col for col in source_query.groupby_clause.group_columns
                                 if col.tables().issubset(referenced_tables)]
        if group_column_fragment:
            groupby_clause = clauses.GroupBy(group_column_fragment, source_query.groupby_clause.distinct)
        else:
            groupby_clause = None
    else:
        groupby_clause = None

    if source_query.having_clause:
        having_fragment = _get_predicate_fragment(source_query.having_clause.condition, referenced_tables)
        having_clause = clauses.Having(having_fragment) if having_fragment else None
    else:
        having_clause = None

    if source_query.orderby_clause:
        order_fragment = [order for order in source_query.orderby_clause.expressions
                          if order.column.tables().issubset(referenced_tables)]
        orderby_clause = clauses.OrderBy(order_fragment) if order_fragment else None
    else:
        orderby_clause = None

    return qal.ImplicitSqlQuery(select_clause=select_clause, from_clause=from_clause, where_clause=where_clause,
                                groupby_clause=groupby_clause, having_clause=having_clause,
                                orderby_clause=orderby_clause, limit_clause=source_query.limit_clause)


def as_count_star_query(source_query: qal.SqlQuery) -> qal.SqlQuery:
    # TODO: how to work with column aliases from the SELECT clause that are referenced later on?
    # E.g. SELECT SUM(foo) AS f FROM bar ORDER BY f
    target_query = copy.copy(source_query)
    target_query.select_clause = clauses.Select.count_star()
    return target_query


def drop_hints(query: qal.SqlQuery, preparatory_statements_only: bool = False) -> qal.SqlQuery:
    query_without_hints = copy.copy(query)
    if preparatory_statements_only and query_without_hints.hints:
        new_hints = copy.copy(query_without_hints.hints)
        new_hints.preparatory_statements = ""
        query_without_hints.hints = new_hints
    else:
        query_without_hints.hints = None
    return query_without_hints


def as_explain(query: qal.SqlQuery, explain: clauses.Explain) -> qal.SqlQuery:
    explain_query = copy.copy(query)
    explain_query.explain = explain
    return explain_query


def rename_table(source_query: qal.SqlQuery, from_table: base.TableReference, target_table: base.TableReference, *,
                 prefix_column_names: bool = False) -> qal.SqlQuery:
    target_query = copy.deepcopy(source_query)

    def _update_column_name(col: base.ColumnReference):
        if col.table == from_table:
            col.table = target_table
        if prefix_column_names and col.table == target_table:
            col.name = f"{from_table.alias}_{col.name}"

    for column in target_query.select_clause.itercolumns():
        _update_column_name(column)

    for table in target_query.tables():
        if table == from_table:
            table.full_name = target_table.full_name
            table.alias = target_table.alias

    if source_query.predicates():
        for column in target_query.predicates().root().itercolumns():
            _update_column_name(column)

    if source_query.groupby_clause:
        for expression in target_query.groupby_clause.group_columns:
            for column in expression.itercolumns():
                _update_column_name(column)

    if source_query.having_clause:
        for column in target_query.having_clause.condition.itercolumns():
            _update_column_name(column)

    if source_query.orderby_clause:
        for expression in target_query.orderby_clause.expressions:
            for column in expression.column.itercolumns():
                _update_column_name(column)

    return target_query


def rename_columns_in_clause(clause: clauses.GroupBy | clauses.Having | clauses.OrderBy,
                             available_renamings: dict[base.ColumnReference, base.ColumnReference]) -> None:
    def _perform_renaming(col: base.ColumnReference):
        if col in available_renamings:
            renamed_column = available_renamings[col]
            col.name = renamed_column.name
            col.table = renamed_column.table

    if isinstance(clause, clauses.GroupBy):
        for grouping in clause.group_columns:
            for column in grouping.columns():
                _perform_renaming(column)

    elif isinstance(clause, clauses.Having):
        for column in clause.condition.columns():
            _perform_renaming(column)
    elif isinstance(clause, clauses.OrderBy):
        for ordering in clause.expressions:
            for column in ordering.column.columns():
                _perform_renaming(column)
    else:
        raise TypeError("Unknown clause type: " + str(clause))


def bind_columns(query: qal.SqlQuery, *, with_schema: bool = True, db_schema: db.DatabaseSchema | None = None) -> None:
    """Queries the table metadata to obtain additional information about the referenced columns.

    The retrieved information includes type information for all columns and the tables that contain the columns.
    """
    # TODO: should this process also create redirections for renamed columns?

    alias_map = {table.alias: table for table in query.tables() if table.alias and table.full_name}
    unbound_tables = [table for table in query.tables() if not table.alias]
    unbound_columns = []

    for table in query.tables():
        if table.alias in alias_map and not table.full_name:
            table.full_name = alias_map[table.alias].full_name

    def _update_column_binding(col: base.ColumnReference) -> None:
        if not col.table:
            unbound_columns.append(col)
        elif not col.table.full_name and col.table.alias in alias_map:
            col.table.full_name = alias_map[col.table.alias].full_name
        elif col.table and not col.table.full_name:
            col.table.full_name = col.table.alias
            col.table.alias = ""

    for column in query.select_clause.itercolumns():
        _update_column_binding(column)

    if query.predicates():
        for column in query.predicates().root().itercolumns():
            _update_column_binding(column)

    column_output_names = query.select_clause.output_names()

    if query.groupby_clause:
        for expression in query.groupby_clause.group_columns:
            for column in expression.itercolumns():
                _update_column_binding(column)
                if column.name in column_output_names:
                    column.redirect = column_output_names[column.name]

    if query.having_clause:
        for column in query.having_clause.condition.itercolumns():
            _update_column_binding(column)
            if column.name in column_output_names:
                column.redirect = column_output_names[column.name]

    if query.orderby_clause:
        for expression in query.orderby_clause.expressions:
            for column in expression.column.itercolumns():
                _update_column_binding(column)
                if column.name in column_output_names:
                    column.redirect = column_output_names[column.name]

    if with_schema:
        db_schema = db_schema if db_schema else db.DatabasePool.get_instance().current_database().schema()
        for column in unbound_columns:
            try:
                column.table = db_schema.lookup_column(column, unbound_tables)
            except ValueError:
                # A ValueError is raised if the column is not found in any of the tables. However, this can still be
                # a valid query, e.g. a dependent subquery. Therefore, we simply ignore this error and leave the column
                # unbound.
                pass
