"""`transform` provides utilities to generate SQL queries from other queries and to modify existing queries."""
from __future__ import annotations

import typing
from collections.abc import Callable, Iterable
from typing import Optional

from postbound.db import db
from postbound.qal import qal, base, clauses, expressions as expr, predicates as preds
from postbound.util import collections as collection_utils

# TODO: at a later point in time, the entire query traversal/modification logic could be refactored to use unified
# access instead of implementing the same pattern matching and traversal logic all over again

QueryType = typing.TypeVar("QueryType", bound=qal.SqlQuery)
ClauseType = typing.TypeVar("ClauseType", bound=clauses.BaseClause)
PredicateType = typing.TypeVar("PredicateType", bound=preds.AbstractPredicate)


def flatten_and_predicate(predicate: preds.AbstractPredicate) -> preds.AbstractPredicate:
    """Simplifies the predicate structure by moving all nested AND predicates their parent AND predicate.

    For example, consider the predicate `(R.a = S.b AND R.a = 42) AND S.b = 24`. This is transformed into a flat
    conjunction, i.e. `R.a = S.b AND R.a = 42 AND S.b = 24`.

    This procedure continues in a recursive manner, until the first disjunction or negation is encountered. All
    predicates below that are left as-is.
    """
    if not isinstance(predicate, preds.CompoundPredicate):
        return predicate

    not_operation = predicate.operation == expr.LogicalSqlCompoundOperators.Not
    or_operation = predicate.operation == expr.LogicalSqlCompoundOperators.Or
    if not_operation or or_operation:
        return predicate

    flattened_children = set()
    for child in predicate.children:
        if isinstance(child, preds.CompoundPredicate) and child.operation == expr.LogicalSqlCompoundOperators.And:
            flattened_child = flatten_and_predicate(child)
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
    """Transforms a query with an explicit FROM clause to a query with an implicit FROM clause.

    Currently, this process is only supported for explicit queries that do not contain subqueries in their FROM clause.
    """
    original_from_clause: clauses.ExplicitFromClause = source_query.from_clause
    additional_predicates = []
    complete_from_tables: list[base.TableReference] = [original_from_clause.base_table.table]

    for joined_table in original_from_clause.joined_tables:
        table_source = joined_table.source
        if not isinstance(table_source, clauses.DirectTableSource):
            raise ValueError("Transforming joined subqueries to implicit table references is not support yet")
        complete_from_tables.append(table_source.table)
        additional_predicates.append(joined_table.join_condition)

    final_from_clause = clauses.ImplicitFromClause.create_for(complete_from_tables)

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


def _get_predicate_fragment(predicate: preds.AbstractPredicate,
                            referenced_tables: set[base.TableReference]) -> preds.AbstractPredicate | None:
    """Filters the predicate hierarchy to include only exactly those base predicates that reference the given tables.

    This applies all simplifications as necessary.
    """
    if not isinstance(predicate, preds.CompoundPredicate):
        return predicate if predicate.tables().issubset(referenced_tables) else None

    compound_predicate: preds.CompoundPredicate = predicate
    child_fragments = [_get_predicate_fragment(child, referenced_tables) for child in compound_predicate.children]
    child_fragments = [fragment for fragment in child_fragments if fragment]
    if not child_fragments:
        return None
    elif len(child_fragments) == 1 and compound_predicate.operation != expr.LogicalSqlCompoundOperators.Not:
        return child_fragments[0]
    else:
        return preds.CompoundPredicate(compound_predicate.operation, child_fragments)


def extract_query_fragment(source_query: qal.ImplicitSqlQuery,
                           referenced_tables: Iterable[base.TableReference]) -> qal.ImplicitSqlQuery | None:
    """Filters the `source_query` to only include the given tables.

    This constructs a new query from the given query that contains exactly those parts of the original query's clauses
    that reference only the given tables.

    For example, consider the query `SELECT * FROM R, S, T WHERE R.a = S.b AND S.c = T.d AND R.a = 42 ORDER BY S.b`
    the query fragment for tables `R` and `S` would look like this:
    `SELECT * FROM R, S WHERE R.a = S.b AND R.a = 42 ORDER BY S.b`, whereas the query fragment for table `S` would
    look like `SELECT * FROM S ORDER BY S.b`.
    """
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
        from_clause = clauses.ImplicitFromClause([clauses.DirectTableSource(tab) for tab in source_query.tables()
                                                  if tab in referenced_tables])
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


def as_count_star_query(source_query: QueryType) -> QueryType:
    """Replaces the SELECT clause of the given query with a COUNT(*) statement."""
    # TODO: how to work with column aliases from the SELECT clause that are referenced later on?
    # E.g. SELECT SUM(foo) AS f FROM bar ORDER BY f
    select = clauses.Select.count_star()
    query_clauses = [clause for clause in source_query.clauses() if not isinstance(clause, clauses.Select)]
    return qal.build_query(query_clauses + [select])


def drop_hints(query: QueryType, preparatory_statements_only: bool = False) -> QueryType:
    """Removes the hint clause from the given query."""
    new_hints = clauses.Hint("", query.hints.query_hints) if preparatory_statements_only and query.hints else None
    query_clauses = [clause for clause in query.clauses() if not isinstance(clause, clauses.Hint)]
    return qal.build_query(query_clauses + [new_hints])


def as_explain(query: QueryType, explain: clauses.Explain = clauses.Explain.plan()) -> QueryType:
    """Transforms the given query into a query that uses the provided EXPLAIN clause."""
    query_clauses = [clause for clause in query.clauses() if not isinstance(clause, clauses.Explain)]
    return qal.build_query(query_clauses + [explain])


def as_explain_analyze(query: QueryType) -> QueryType:
    """Transforms the given query into an EXPLAIN ANALYZE query."""
    return as_explain(query, clauses.Explain.explain_analyze())


def add_clause(query: qal.SqlQuery, clauses_to_add: clauses.BaseClause | Iterable[clauses.BaseClause]) -> qal.SqlQuery:
    """Creates a new SQL query with the given additional clauses.

    No validation is performed. Conflicts are resolved according to the rules of `qal.build_query`
    This can potentially switch an implicit query to an explicit one and vice-versa.
    """
    clauses_to_add = collection_utils.enlist(clauses_to_add)
    new_clause_types = {type(clause) for clause in clauses_to_add}
    remaining_clauses = [clause for clause in query.clauses() if type(clause) not in new_clause_types]
    return qal.build_query(remaining_clauses + list(clauses_to_add))


def drop_clause(query: qal.SqlQuery, clauses_to_drop: typing.Type | Iterable[typing.Type]) -> qal.SqlQuery:
    """Creates a new SQL query without all clauses of the indicated types. No validation is performed."""
    clauses_to_drop = set(collection_utils.enlist(clauses_to_drop))
    remaining_clauses = [clause for clause in query.clauses() if not type(clause) in clauses_to_drop]
    return qal.build_query(remaining_clauses)


def replace_clause(query: QueryType, replacements: clauses.BaseClause | Iterable[clauses.BaseClause]) -> QueryType:
    """Creates a new SQL query with the replacements being used instead of the original clauses of the same type.

    This function does not switch a query from implicit to explicit or vice-versa. Use a combination of `drop_clause`
    and `add_clause` for that. No validation is performed.
    """
    replacements = collection_utils.enlist(replacements)
    clauses_to_replace = {type(clause): clause for clause in replacements}
    replaced_clauses = [clauses_to_replace.get(type(current_clause), current_clause)
                        for current_clause in query.clauses()]
    return qal.build_query(replaced_clauses)


def _replace_expression_in_predicate(predicate: PredicateType,
                                     replacement: Callable[[expr.SqlExpression], expr.SqlExpression]
                                     ) -> Optional[PredicateType]:
    if not predicate:
        return None

    if isinstance(predicate, preds.BinaryPredicate):
        renamed_first_arg = replacement(predicate.first_argument)
        renamed_second_arg = replacement(predicate.second_argument)
        return preds.BinaryPredicate(predicate.operation, renamed_first_arg, renamed_second_arg)
    elif isinstance(predicate, preds.BetweenPredicate):
        renamed_col = replacement(predicate.column)
        renamed_interval_start = replacement(predicate.interval_start)
        renamed_interval_end = replacement(predicate.interval_end)
        return preds.BetweenPredicate(renamed_col, (renamed_interval_start, renamed_interval_end))
    elif isinstance(predicate, preds.InPredicate):
        renamed_col = replacement(predicate.column)
        renamed_vals = [replacement(val) for val in predicate.values]
        return preds.InPredicate(renamed_col, renamed_vals)
    elif isinstance(predicate, preds.UnaryPredicate):
        return preds.UnaryPredicate(replacement(predicate.column), predicate.operation)
    elif isinstance(predicate, preds.CompoundPredicate):
        if predicate.operation == expr.LogicalSqlCompoundOperators.Not:
            renamed_children = [_replace_expression_in_predicate(predicate.children, replacement)]
        else:
            renamed_children = [_replace_expression_in_predicate(child, replacement) for child in predicate.children]
        return preds.CompoundPredicate(predicate.operation, renamed_children)
    else:
        raise ValueError("Unknown predicate type: " + str(predicate))


def _replace_expression_in_table_source(table_source: clauses.TableSource,
                                        replacement: Callable[[expr.SqlExpression], expr.SqlExpression]
                                        ) -> Optional[clauses.TableSource]:
    if table_source is None:
        return None
    if isinstance(table_source, clauses.DirectTableSource):
        return table_source
    elif isinstance(table_source, clauses.SubqueryTableSource):
        replaced_subquery = replace_expressions(table_source.query, replacement)
        return clauses.SubqueryTableSource(replaced_subquery, table_source.target_name)
    elif isinstance(table_source, clauses.JoinTableSource):
        replaced_source = _replace_expression_in_table_source(table_source.source, replacement)
        replaced_condition = _replace_expression_in_predicate(table_source.join_condition, replacement)
        return clauses.JoinTableSource(replaced_source, replaced_condition, join_type=table_source.join_type)
    else:
        raise TypeError("Unknown table source type: " + str(table_source))

def _replace_expressions_in_clause(clause: ClauseType, replacement: Callable[[expr.SqlExpression], expr.SqlExpression]
                                   ) -> Optional[ClauseType]:
    if not clause:
        return None

    if isinstance(clause, clauses.Hint) or isinstance(clause, clauses.Explain):
        return clause
    if isinstance(clause, clauses.Select):
        replaced_targets = [clauses.BaseProjection(replacement(proj.expression), proj.target_name)
                           for proj in clause.targets]
        return clauses.Select(replaced_targets, clause.projection_type)
    elif isinstance(clause, clauses.ImplicitFromClause):
        return clause
    elif isinstance(clause, clauses.ExplicitFromClause):
        replaced_joins = [_replace_expression_in_table_source(join, replacement) for join in clause.joined_tables]
        return clauses.ExplicitFromClause(clause.base_table, replaced_joins)
    elif isinstance(clause, clauses.From):
        replaced_contents = [_replace_expression_in_table_source(target, replacement) for target in clause.contents]
        return clauses.From(replaced_contents)
    elif isinstance(clause, clauses.Where):
        return clauses.Where(_replace_expression_in_predicate(clause.predicate, replacement))
    elif isinstance(clause, clauses.GroupBy):
        replaced_cols = [replacement(col) for col in clause.group_columns]
        return clauses.GroupBy(replaced_cols, clause.distinct)
    elif isinstance(clause, clauses.Having):
        return clauses.Having(_replace_expression_in_predicate(clause.condition, replacement))
    elif isinstance(clause, clauses.OrderBy):
        replaced_cols = [clauses.OrderByExpression(replacement(col.column), col.ascending, col.nulls_first)
                        for col in clause.expressions]
        return clauses.OrderBy(replaced_cols)
    elif isinstance(clause, clauses.Limit):
        return clause
    else:
        raise ValueError("Unknown clause: " + str(clause))


def replace_expressions(query: QueryType,
                        replacement: Callable[[expr.SqlExpression], expr.SqlExpression]) -> QueryType:
    replaced_clauses = [_replace_expressions_in_clause(clause, replacement) for clause in query.clauses()]
    return qal.build_query(replaced_clauses)


def _perform_predicate_replacement(current_predicate: preds.AbstractPredicate,
                                   target_predicate: preds.AbstractPredicate,
                                   new_predicate: preds.AbstractPredicate) -> preds.AbstractPredicate:
    """Performs the designated predicate replacement, moving deeper into the predicate tree as necessary."""
    if current_predicate == target_predicate:
        return new_predicate

    if isinstance(current_predicate, preds.CompoundPredicate):
        if current_predicate.operation == expr.LogicalSqlCompoundOperators.Not:
            replaced_children = [_perform_predicate_replacement(current_predicate.children,
                                                                target_predicate, new_predicate)]
        else:
            replaced_children = [_perform_predicate_replacement(child_pred, target_predicate, new_predicate)
                                 for child_pred in current_predicate.children]
        return preds.CompoundPredicate(current_predicate.operation, replaced_children)
    else:
        return current_predicate


def replace_predicate(query: qal.ImplicitSqlQuery, predicate_to_replace: preds.AbstractPredicate,
                      new_predicate: preds.AbstractPredicate) -> qal.ImplicitSqlQuery:
    """Rewrites the given query to use `new_predicate` in all occurrences of the other predicate.

    In the current implementation this does only work for top-level predicates, i.e. subqueries are not considered.
    Furthermore, only the WHERE clause and the HAVING clause are modified.

    If the predicate to replace is not found, nothing happens.
    """
    # TODO: also allow replacement in explicit SQL queries
    # TODO: allow predicate replacement in subqueries
    if not query.where_clause and not query.having_clause:
        return query

    if query.where_clause:
        replaced_predicate = _perform_predicate_replacement(query.where_clause.predicate, predicate_to_replace,
                                                            new_predicate)
        replaced_where = clauses.Where(replaced_predicate)
    else:
        replaced_where = None

    if query.having_clause:
        replaced_predicate = _perform_predicate_replacement(query.having_clause.condition, predicate_to_replace,
                                                            new_predicate)
        replaced_having = clauses.Having(replaced_predicate)
    else:
        replaced_having = None

    return replace_clause(query, [clause for clause in (replaced_where, replaced_having) if clause])


def _rename_columns_in_query(query: QueryType,
                             available_renamings: dict[base.ColumnReference, base.ColumnReference]) -> QueryType:
    """Renames all columns in the query predicate according to the available renamings.

    A renaming maps the current column to the column that should be used instead.
    """
    renamed_select = rename_columns_in_clause(query.select_clause, available_renamings)
    renamed_from = rename_columns_in_clause(query.from_clause, available_renamings)
    renamed_where = rename_columns_in_clause(query.where_clause, available_renamings)
    renamed_groupby = rename_columns_in_clause(query.groupby_clause, available_renamings)
    renamed_having = rename_columns_in_clause(query.having_clause, available_renamings)
    renamed_orderby = rename_columns_in_clause(query.orderby_clause, available_renamings)

    if isinstance(query, qal.ImplicitSqlQuery):
        return qal.ImplicitSqlQuery(select_clause=renamed_select, from_clause=renamed_from, where_clause=renamed_where,
                                    groupby_clause=renamed_groupby, having_clause=renamed_having,
                                    orderby_clause=renamed_orderby, limit_clause=query.limit_clause,
                                    hints=query.hints, explain_clause=query.explain)
    elif isinstance(query, qal.ExplicitSqlQuery):
        return qal.ExplicitSqlQuery(select_clause=renamed_select, from_clause=renamed_from, where_clause=renamed_where,
                                    groupby_clause=renamed_groupby, having_clause=renamed_having,
                                    orderby_clause=renamed_orderby, limit_clause=query.limit_clause,
                                    hints=query.hints, explain_clause=query.explain)
    elif isinstance(query, qal.MixedSqlQuery):
        return qal.MixedSqlQuery(select_clause=renamed_select, from_clause=renamed_from, where_clause=renamed_where,
                                    groupby_clause=renamed_groupby, having_clause=renamed_having,
                                    orderby_clause=renamed_orderby, limit_clause=query.limit_clause,
                                    hints=query.hints, explain_clause=query.explain)
    else:
        raise TypeError("Unknown query type: " + str(query))


def _rename_columns_in_expression(expression: Optional[expr.SqlExpression],
                                  available_renamings: dict[base.ColumnReference, base.ColumnReference]
                                  ) -> Optional[expr.SqlExpression]:
    """Renames all columns in the given expression according to the available renamings.

    A renaming maps the current column to the column that should be used instead.
    """
    if expression is None:
        return None

    if isinstance(expression, expr.StaticValueExpression) or isinstance(expression, expr.StarExpression):
        return expression
    elif isinstance(expression, expr.ColumnExpression):
        return (expr.ColumnExpression(available_renamings[expression.column])
                if expression.column in available_renamings else expression)
    elif isinstance(expression, expr.CastExpression):
        renamed_child = _rename_columns_in_expression(expression.casted_expression, available_renamings)
        return expr.CastExpression(renamed_child, expression.target_type)
    elif isinstance(expression, expr.MathematicalExpression):
        renamed_first_arg = _rename_columns_in_expression(expression.first_arg, available_renamings)
        renamed_second_arg = _rename_columns_in_expression(expression.second_arg, available_renamings)
        return expr.MathematicalExpression(expression.operator, renamed_first_arg, renamed_second_arg)
    elif isinstance(expression, expr.FunctionExpression):
        renamed_arguments = [_rename_columns_in_expression(arg, available_renamings)
                             for arg in expression.arguments]
        return expr.FunctionExpression(expression.function, renamed_arguments, distinct=expression.distinct)
    elif isinstance(expression, expr.SubqueryExpression):
        return expr.SubqueryExpression(_rename_columns_in_query(expression.query, available_renamings))
    else:
        raise ValueError("Unknown expression type: " + str(expression))


def rename_columns_in_predicate(predicate: Optional[preds.AbstractPredicate],
                                available_renamings: dict[base.ColumnReference, base.ColumnReference]
                                ) -> Optional[preds.AbstractPredicate]:
    """Renames all columns in the given predicate according to the available renamings.

    A renaming maps the current column to the column that should be used instead.
    """
    if not predicate:
        return None

    if isinstance(predicate, preds.BinaryPredicate):
        renamed_first_arg = _rename_columns_in_expression(predicate.first_argument, available_renamings)
        renamed_second_arg = _rename_columns_in_expression(predicate.second_argument, available_renamings)
        return preds.BinaryPredicate(predicate.operation, renamed_first_arg, renamed_second_arg)
    elif isinstance(predicate, preds.BetweenPredicate):
        renamed_col = _rename_columns_in_expression(predicate.column, available_renamings)
        renamed_interval_start = _rename_columns_in_expression(predicate.interval_start, available_renamings)
        renamed_interval_end = _rename_columns_in_expression(predicate.interval_end, available_renamings)
        return preds.BetweenPredicate(renamed_col, (renamed_interval_start, renamed_interval_end))
    elif isinstance(predicate, preds.InPredicate):
        renamed_col = _rename_columns_in_expression(predicate.column, available_renamings)
        renamed_vals = [_rename_columns_in_expression(val, available_renamings)
                        for val in predicate.values]
        return preds.InPredicate(renamed_col, renamed_vals)
    elif isinstance(predicate, preds.UnaryPredicate):
        return preds.UnaryPredicate(_rename_columns_in_expression(predicate.column, available_renamings),
                                    predicate.operation)
    elif isinstance(predicate, preds.CompoundPredicate):
        renamed_children = ([rename_columns_in_predicate(predicate.children, available_renamings)]
                            if predicate.operation == expr.LogicalSqlCompoundOperators.Not
                            else [rename_columns_in_predicate(child, available_renamings)
                                  for child in predicate.children])
        return preds.CompoundPredicate(predicate.operation, renamed_children)
    else:
        raise ValueError("Unknown predicate type: " + str(predicate))


def _rename_columns_in_table_source(table_source: clauses.TableSource,
                                    available_renamings: dict[base.ColumnReference, base.ColumnReference]
                                    ) -> Optional[clauses.TableSource]:
    if table_source is None:
        return None
    if isinstance(table_source, clauses.DirectTableSource):
        return table_source
    elif isinstance(table_source, clauses.SubqueryTableSource):
        renamed_subquery = _rename_columns_in_query(table_source.query, available_renamings)
        return clauses.SubqueryTableSource(renamed_subquery, table_source.target_name)
    elif isinstance(table_source, clauses.JoinTableSource):
        renamed_source = _rename_columns_in_table_source(table_source.source, available_renamings)
        renamed_condition = rename_columns_in_predicate(table_source.join_condition, available_renamings)
        return clauses.JoinTableSource(renamed_source, renamed_condition, join_type=table_source.join_type)
    else:
        raise TypeError("Unknown table source type: " + str(table_source))


def rename_columns_in_clause(clause: Optional[ClauseType],
                             available_renamings: dict[base.ColumnReference, base.ColumnReference]
                             ) -> Optional[ClauseType]:
    """Renames all columns in the given clause according to the available renamings.

    A renaming maps the current column to the column that should be used instead.
    """
    if not clause:
        return None

    if isinstance(clause, clauses.Hint) or isinstance(clause, clauses.Explain):
        return clause
    if isinstance(clause, clauses.Select):
        renamed_targets = [clauses.BaseProjection(_rename_columns_in_expression(proj.expression, available_renamings),
                                                  proj.target_name)
                           for proj in clause.targets]
        return clauses.Select(renamed_targets, clause.projection_type)
    elif isinstance(clause, clauses.ImplicitFromClause):
        return clause
    elif isinstance(clause, clauses.ExplicitFromClause):
        renamed_joins = [_rename_columns_in_table_source(join, available_renamings) for join in clause.joined_tables]
        return clauses.ExplicitFromClause(clause.base_table, renamed_joins)
    elif isinstance(clause, clauses.From):
        renamed_sources = [_rename_columns_in_table_source(table_source, available_renamings)
                           for table_source in clause.contents]
        return clauses.From(renamed_sources)
    elif isinstance(clause, clauses.Where):
        return clauses.Where(rename_columns_in_predicate(clause.predicate, available_renamings))
    elif isinstance(clause, clauses.GroupBy):
        renamed_cols = [_rename_columns_in_expression(col, available_renamings) for col in clause.group_columns]
        return clauses.GroupBy(renamed_cols, clause.distinct)
    elif isinstance(clause, clauses.Having):
        return clauses.Having(rename_columns_in_predicate(clause.condition, available_renamings))
    elif isinstance(clause, clauses.OrderBy):
        renamed_cols = [clauses.OrderByExpression(_rename_columns_in_expression(col.column, available_renamings),
                                                  col.ascending, col.nulls_first)
                        for col in clause.expressions]
        return clauses.OrderBy(renamed_cols)
    elif isinstance(clause, clauses.Limit):
        return clause
    else:
        raise ValueError("Unknown clause: " + str(clause))


def rename_table(source_query: QueryType, from_table: base.TableReference, target_table: base.TableReference, *,
                 prefix_column_names: bool = False) -> QueryType:
    """Changes all occurrences of the `from_table` in the `source_query` to use the `target_table` instead.

    If `prefix_column_names` is set to `True`, all renamed columns will also have their name changed to include the
    original table name. This is mostly used in outer queries where a subquery had the names of the exported columns
    changed.
    """
    necessary_renamings: dict[base.ColumnReference, base.ColumnReference] = {}
    for column in filter(lambda col: col.table == from_table, source_query.columns()):
        new_column_name = f"{column.table.alias}_{column.name}" if prefix_column_names else column.name
        necessary_renamings[column] = base.ColumnReference(new_column_name, target_table)
    return _rename_columns_in_query(source_query, necessary_renamings)


def bind_columns(query: QueryType, *, with_schema: bool = True,
                 db_schema: Optional[db.DatabaseSchema] = None) -> QueryType:
    """Adds additional metadata to all column references that appear in the given query.

    This sets the `table` reference of all column objects to the actual tables that provide the column. If
    `with_schema` is `False`, this process only uses the names and aliases of the tables themselves. Otherwise, the
    database schema (falling back to the schema provided by the `DatabasePool` if necessary) is used to retrieve the
    tables for all columns where a simple name-based binding does not work.
    """

    table_alias_map: dict[str, base.TableReference] = {table.identifier(): table for table in query.tables()
                                                       if table.full_name}
    unbound_columns: list[base.ColumnReference] = []
    necessary_renamings: dict[base.ColumnReference, base.ColumnReference] = {}
    for column in query.columns():
        if not column.table:
            unbound_columns.append(column)
        elif column.table.identifier() in table_alias_map:
            bound_column = base.ColumnReference(column.name, table_alias_map[column.table.identifier()])
            necessary_renamings[column] = bound_column

    partially_bound_query = _rename_columns_in_query(query, necessary_renamings)
    if not with_schema:
        return partially_bound_query

    db_schema = db_schema if db_schema else db.DatabasePool().get_instance().current_database().schema()
    candidate_tables = [table for table in query.tables() if table.full_name]
    unbound_renamings: dict[base.ColumnReference, base.ColumnReference] = {}
    for column in unbound_columns:
        try:
            target_table = db_schema.lookup_column(column, candidate_tables)
            bound_column = base.ColumnReference(column.name, target_table)
            unbound_renamings[column] = bound_column
        except ValueError:
            # A ValueError is raised if the column is not found in any of the tables. However, this can still be
            # a valid query, e.g. a dependent subquery. Therefore, we simply ignore this error and leave the column
            # unbound.
            pass
    return _rename_columns_in_query(partially_bound_query, unbound_renamings)
