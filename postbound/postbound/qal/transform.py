"""This module provides tools to modify the contents of existing `SqlQuery` instances.

Since queries are designed as immutable data objects, these transformations operate by implementing new query instances.

The tools differ in their granularity, ranging from utilities that swap out individual expressions and predicates, to tools
that change the entire structure of the query.
"""
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
"""The concrete class of a query.

This generic type is used for transformations that do not change the type of a query and operate on all the different query
types.
"""

ClauseType = typing.TypeVar("ClauseType", bound=clauses.BaseClause)
"""The concrete class of a clause.

This generic type is used for transformations that do not change the type of a clause and operate on all the different clause
types.
"""

PredicateType = typing.TypeVar("PredicateType", bound=preds.AbstractPredicate)
"""The concrete type of a predicate.

This generic type is used for transformations that do not change the type of a predicate and operate on all the different
predicate types.
"""


def flatten_and_predicate(predicate: preds.AbstractPredicate) -> preds.AbstractPredicate:
    """Simplifies the predicate structure by moving all nested ``AND`` predicates to their parent ``AND`` predicate.

    For example, consider the predicate ``(R.a = S.b AND R.a = 42) AND S.b = 24``. This is transformed into the flattened
    equivalent conjunction ``R.a = S.b AND R.a = 42 AND S.b = 24``.

    This procedure continues in a recursive manner, until the first disjunction or negation is encountered. All predicates
    below that point are left as-is for the current branch of the predicate hierarchy.

    Parameters
    ----------
    predicate : preds.AbstractPredicate
        The predicate to simplified

    Returns
    -------
    preds.AbstractPredicate
        An equivalent version of the given `predicate`, with all conjunctions unnested
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
    """Transforms a query with an explicit ``FROM`` clause to a query with an implicit ``FROM`` clause.

    Currently, this process is only supported for explicit queries that do not contain subqueries in their ``FROM`` clause.

    Parameters
    ----------
    source_query : qal.ExplicitSqlQuery
        The query that should be transformed

    Returns
    -------
    qal.ImplicitSqlQuery
        An equivalent version of the given query, using an implicit ``FROM`` clause

    Raises
    ------
    ValueError
        If the `source_query` contains subquery table sources
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
                                orderby_clause=source_query.orderby_clause, limit_clause=source_query.limit_clause,
                                cte_clause=source_query.cte_clause)


def _get_predicate_fragment(predicate: preds.AbstractPredicate,
                            referenced_tables: set[base.TableReference]) -> preds.AbstractPredicate | None:
    """Filters the predicate hierarchy to include only those base predicates that reference the given tables.

    The referenced tables operate as a superset - parts of the predicate are retained if the tables that they reference are a
    subset of the target tables.

    In the general case, the resulting predicate is no longer equivalent to the original predicate, since large portions of
    the predicate hierarchy are pruned (exactly those that do not touch the given tables). Simplifications will be applied to
    the predicate as necessary. For example, if only a single child predicate of a conjunction references the given tables, the
    conjunction is removed and the child predicate is inserted instead.

    Notice that no logical simplifications are applied. For example, if the resulting predicate fragment is the conjunction
    ``R.a < 42 AND R.a < 84``, this is not simplified into ``R.a < 42``.

    Parameters
    ----------
    predicate : preds.AbstractPredicate
        The predicate to filter
    referenced_tables : set[base.TableReference]
        The superset of all allowed tables. Those parts of the `predicate` are pruned, whose tables are not a subset of the

    Returns
    -------
    preds.AbstractPredicate | None
        The largest fragment of the original predicate that references only a subset of the `referenced_tables`. If the entire
        predicate was pruned, ``None`` is returned.

    Examples
    --------
    Consider the predicate ``R.a > 100 AND (S.b = 42 OR R.a = 42)``. The predicate fragment for table ``R`` would be
    ``R.a > 100 AND R.a = 42``.
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
                           referenced_tables: Iterable[base.TableReference]) -> Optional[qal.ImplicitSqlQuery]:
    """Filters a query to only include parts that reference specific tables.

    This builds a new query from the given query that contains exactly those parts of the original query's clauses that
    reference only the given tables or a subset of them.

    For example, consider the query ``SELECT * FROM R, S, T WHERE R.a = S.b AND S.c = T.d AND R.a = 42 ORDER BY S.b``
    the query fragment for tables ``R`` and ``S`` would look like this:
    ``SELECT * FROM R, S WHERE R.a = S.b AND R.a = 42 ORDER BY S.b``, whereas the query fragment for table ``S`` would
    look like ``SELECT * FROM S ORDER BY S.b``.

    Notice that this can break disjunctions: the fragment for table ``R`` of query
    ``SELECT * FROM R, S, WHERE R.a < 100 AND (R.a = 42 OR S.b = 42)`` is ``SELECT * FROM R WHERE R.a < 100 AND R.a = 42``.
    This also indicates that the fragment extraction does not perform any logical pruning of superflous predicates.

    Parameters
    ----------
    source_query : qal.ImplicitSqlQuery
        The query that should be transformed
    referenced_tables : Iterable[base.TableReference]
        The tables that should be extracted

    Returns
    -------
    Optional[qal.ImplicitSqlQuery]
        A query that only consists of those parts of the `source_query`, that reference (a subset of) the `referenced_tables`.
        If there is no such subset, ``None`` is returned.
    """
    referenced_tables = set(referenced_tables)
    if not referenced_tables.issubset(source_query.tables()):
        return None

    cte_fragment = ([with_query for with_query
                     in source_query.cte_clause.queries if with_query.target_table in referenced_tables]
                    if source_query.cte_clause else [])
    cte_clause = clauses.CommonTableExpression(cte_fragment) if cte_fragment else None

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
                                orderby_clause=orderby_clause, limit_clause=source_query.limit_clause,
                                cte_clause=cte_clause)


def _default_subquery_name(tables: Iterable[base.TableReference]) -> str:
    """Constructs a valid SQL table name for a subquery consisting of specific tables.

    Parameters
    ----------
    tables : Iterable[base.TableReference]
        The tables that should be represented by a subquery.

    Returns
    -------
    str
        The target name of the subquery
    """
    return "_".join(table.identifier() for table in tables)


def move_into_subquery(query: qal.SqlQuery, tables: Iterable[base.TableReference], subquery_name: str = "") -> qal.SqlQuery:
    """Transforms a specific query by moving some of its tables into a subquery.

    This transformation renames all usages of columns that are now produced by the subquery to references to the virtual
    subquery table instead.

    Notice that this transformation currently only really works for "good natured" queries, i.e. mostly implicit SPJ+ queries.
    Notably, the transformation is only supported for queries that do not already contain subqueries, since moving tables
    between subqueries is a quite tricky process. Likewise, the renaming is only applied at a table-level. If the tables export
    columns of the same name, these are not renamed and the transformation fails as well. If in doubt, you should definitely
    check the output of this method for your complicated queries to prevent bad surprises!

    Parameters
    ----------
    query : qal.SqlQuery
        The query to transform
    tables : Iterable[base.TableReference]
        The tables that should be placed into a subquery
    subquery_name : str, optional
        The target name of the virtual subquery table. If empty, a default name (consisting of all the subquery tables) is
        generated

    Returns
    -------
    qal.SqlQuery
        The transformed query

    Raises
    ------
    ValueError
        If the `query` does not contain a ``FROM`` clause.
    ValueError
        If the query contains virtual tables
    ValueError
        If `tables` contains less than 2 entries. In this case, using a subquery is completely pointless.
    ValueError
        If the tables that should become part of the subquery both provide columns of the same name, and these columns are used
        in the rest of the query. This level of renaming is currently not accounted for.
    """
    if not query.from_clause:
        raise ValueError("Cannot create a subquery for a query without a FROM clause")
    if any(table.virtual for table in query.tables()):
        raise ValueError("Cannot move into subquery for queries with virtual tables")

    tables = set(tables)

    # deleted CTE check: this was not necessary because a CTE produces a virtual table. This already fails the previous test
    if len(tables) < 2:
        raise ValueError("At least two tables required")

    predicates = query.predicates()
    all_referenced_columns = collection_utils.set_union(clause.columns() for clause in query.clauses()
                                                        if not isinstance(clause, clauses.From))
    columns_from_subquery_tables = {column for column in all_referenced_columns if column.table in tables}
    if len({column.name for column in columns_from_subquery_tables}) < len(columns_from_subquery_tables):
        raise ValueError("Cannot create subquery: subquery tables export columns of the same name")

    subquery_name = subquery_name if subquery_name else _default_subquery_name(tables)
    subquery_table = base.TableReference.create_virtual(subquery_name)
    renamed_columns = {column: base.ColumnReference(column.name, subquery_table)
                       for column in columns_from_subquery_tables}

    subquery_predicates: list[preds.AbstractPredicate] = []
    for table in tables:
        filter_predicate = predicates.filters_for(table)
        if not filter_predicate:
            continue
        subquery_predicates.append(filter_predicate)
    join_predicates = predicates.joins_between(tables, tables)
    if join_predicates:
        subquery_predicates.append(join_predicates)

    subquery_select = clauses.Select.create_for(columns_from_subquery_tables)
    subquery_from = clauses.ImplicitFromClause.create_for(tables)
    subquery_where = (clauses.Where(preds.CompoundPredicate.create_and(subquery_predicates))
                      if subquery_predicates else None)
    subquery_clauses = [subquery_select, subquery_from, subquery_where]
    subquery = qal.build_query(clause for clause in subquery_clauses if clause)
    subquery_table_source = clauses.SubqueryTableSource(subquery, subquery_name)

    updated_from_sources = [table_source for table_source in query.from_clause.contents
                            if not table_source.tables() < tables]
    update_from_clause = clauses.ImplicitFromClause.create_for(updated_from_sources + [subquery_table_source])
    updated_predicate = query.where_clause.predicate if query.where_clause else None
    for predicate in subquery_predicates:
        updated_predicate = remove_predicate(updated_predicate, predicate)
    updated_where_clause = clauses.Where(updated_predicate) if updated_predicate else None

    updated_query = drop_clause(query, [clauses.From, clauses.Where])
    updated_query = add_clause(updated_query, [update_from_clause, updated_where_clause])

    updated_other_clauses: list[clauses.BaseClause] = []
    for clause in updated_query.clauses():
        if isinstance(clause, clauses.From):
            continue
        renamed_clause = rename_columns_in_clause(clause, renamed_columns)
        updated_other_clauses.append(renamed_clause)
    final_query = replace_clause(updated_query, updated_other_clauses)
    return final_query


def as_star_query(source_query: QueryType) -> QueryType:
    """Transforms a specific query to use a ``SELECT *`` projection instead.

    Notice that this can break certain queries where a renamed column from the ``SELECT`` clause is used in other parts of
    the query, such as ``ORDER BY`` clauses (e.g. ``SELECT SUM(foo) AS f FROM bar ORDER BY f``). We currently do not undo such
    a renaming.

    Parameters
    ----------
    source_query : QueryType
        The query to transform

    Returns
    -------
    QueryType
        A variant of the input query that uses a ``SELECT *`` projection.
    """
    select = clauses.Select.star()
    query_clauses = [clause for clause in source_query.clauses() if not isinstance(clause, clauses.Select)]
    return qal.build_query(query_clauses + [select])


def as_count_star_query(source_query: QueryType) -> QueryType:
    """Transforms a specific query to use a ``SELECT COUNT(*)`` projection instead.

    Notice that this can break certain queries where a renamed column from the ``SELECT`` clause is used in other parts of
    the query, such as ``ORDER BY`` clauses (e.g. ``SELECT SUM(foo) AS f FROM bar ORDER BY f``). We currently do not undo such
    a renaming.

    Parameters
    ----------
    source_query : QueryType
        The query to transform

    Returns
    -------
    QueryType
        A variant of the input query that uses a ``SELECT COUNT(*)`` projection.
    """
    select = clauses.Select.count_star()
    query_clauses = [clause for clause in source_query.clauses() if not isinstance(clause, clauses.Select)]
    return qal.build_query(query_clauses + [select])


def drop_hints(query: QueryType, preparatory_statements_only: bool = False) -> QueryType:
    """Removes the hint clause from a specific query.

    Parameters
    ----------
    query : QueryType
        The query to transform
    preparatory_statements_only : bool, optional
        Whether only the preparatory statements from the hint block should be removed. This would retain the actual hints.
        Defaults to ``False``, which removes the entire block, no matter its contents.

    Returns
    -------
    QueryType
        The query without the hint block
    """
    new_hints = clauses.Hint("", query.hints.query_hints) if preparatory_statements_only and query.hints else None
    query_clauses = [clause for clause in query.clauses() if not isinstance(clause, clauses.Hint)]
    return qal.build_query(query_clauses + [new_hints])


def as_explain(query: QueryType, explain: clauses.Explain = clauses.Explain.plan()) -> QueryType:
    """Transforms a specific query into an ``EXPLAIN`` query.

    Parameters
    ----------
    query : QueryType
        The query to transform
    explain : clauses.Explain, optional
        The ``EXPLAIN`` block to use. Defaults to a standard ``Explain.plan()`` block.

    Returns
    -------
    QueryType
        The transformed query
    """
    query_clauses = [clause for clause in query.clauses() if not isinstance(clause, clauses.Explain)]
    return qal.build_query(query_clauses + [explain])


def as_explain_analyze(query: QueryType) -> QueryType:
    """Transforms a specific query into an ``EXPLAIN ANALYZE`` query.

    Parameters
    ----------
    query : QueryType
        The query to transform

    Returns
    -------
    QueryType
        The transformed query. It uses an ``EXPLAIN ANALYZE`` block with the default output format. If this is not desired,
        the `as_explain` transformation has to be used and the target ``EXPLAIN`` block has to be given explicitly.
    """
    return as_explain(query, clauses.Explain.explain_analyze())


def remove_predicate(predicate: Optional[preds.AbstractPredicate],
                     predicate_to_remove: preds.AbstractPredicate) -> Optional[preds.AbstractPredicate]:
    """Drops a specific predicate from the predicate hierarchy.

    If necessary, the hierarchy will be simplified. For example, if the `predicate_to_remove` is one of two childs of a
    conjunction, the removal would leave a conjunction of just a single predicate. In this case, the conjunction can be dropped
    altogether, leaving just the other child predicate. The same also applies to disjunctions and negations.

    Parameters
    ----------
    predicate : Optional[preds.AbstractPredicate]
        The predicate hierarchy from which should removed. If this is ``None``, no removal is attempted.
    predicate_to_remove : preds.AbstractPredicate
        The predicate that should be removed.

    Returns
    -------
    Optional[preds.AbstractPredicate]
        The resulting (simplified) predicate hierarchy. Will be ``None`` if there are no meaningful predicates left after
        removal, or if the `predicate` equals the `predicate_to_remove`.
    """
    if not predicate or predicate == predicate_to_remove:
        return None
    if not isinstance(predicate, preds.CompoundPredicate):
        return predicate

    if predicate.operation == expr.LogicalSqlCompoundOperators.Not:
        updated_child = remove_predicate(predicate.children, predicate_to_remove)
        return preds.CompoundPredicate.create_not(updated_child) if updated_child else None

    updated_children = [remove_predicate(child_pred, predicate_to_remove) for child_pred in predicate.children]
    updated_children = [child_pred for child_pred in updated_children if child_pred]
    if not updated_children:
        return None
    elif len(updated_children) == 1:
        return updated_children[0]
    else:
        return preds.CompoundPredicate(predicate.operation, updated_children)


def add_clause(query: qal.SqlQuery, clauses_to_add: clauses.BaseClause | Iterable[clauses.BaseClause]) -> qal.SqlQuery:
    """Creates a new SQL query, potentailly with additional clauses.

    No validation is performed. Conflicts are resolved according to the rules of `qal.build_query`. This means that the query
    can potentially be switched from an implicit query to an explicit one and vice-versa.

    Parameters
    ----------
    query : qal.SqlQuery
        The query to which the clause(s) should be added
    clauses_to_add : clauses.BaseClause | Iterable[clauses.BaseClause]
        The new clauses

    Returns
    -------
    qal.SqlQuery
        A new clauses consisting of the old query's clauses and the `clauses_to_add`. Duplicate clauses are overwritten by
        the `clauses_to_add`.
    """
    clauses_to_add = collection_utils.enlist(clauses_to_add)
    new_clause_types = {type(clause) for clause in clauses_to_add}
    remaining_clauses = [clause for clause in query.clauses() if type(clause) not in new_clause_types]
    return qal.build_query(remaining_clauses + list(clauses_to_add))


ClauseDescription = typing.Union[typing.Type, clauses.BaseClause, Iterable[typing.Type | clauses.BaseClause]]
"""Denotes different ways clauses to remove can be denoted.

See Also
--------
drop_clause
"""


def drop_clause(query: qal.SqlQuery, clauses_to_drop: ClauseDescription) -> qal.SqlQuery:
    """Removes specific clauses from a query.

    The clauses can be denoted in two different ways: either as the raw type of the clause, or as an instance of the same
    clause type as the one that should be removed. Notice that the instance of the clause does not need to be equal to the
    clause of the query. It just needs to be the same type of clause.

    This method does not perform any validation, other than the rules described in `qal.build_query`.

    Parameters
    ----------
    query : qal.SqlQuery
        The query to remove clauses from
    clauses_to_drop : ClauseDescription
        The clause(s) to remove. This can be a single clause type or clause instance, or an iterable of clauses types,
        intermixed with clause instances. In either way clauses of the desired types are dropped from the query.

    Returns
    -------
    qal.SqlQuery
        A query without the specified clauses

    Examples
    --------

    The following two calls achieve exactly the same thing: getting rid of the ``LIMIT`` clause.

    .. code-block:: python

        drop_clause(query, clauses.Limit)
        drop_clause(query, query.limit_clause)
    """
    clauses_to_drop = set(collection_utils.enlist(clauses_to_drop))
    clauses_to_drop = {drop if isinstance(drop, typing.Type) else type(drop) for drop in clauses_to_drop}
    remaining_clauses = [clause for clause in query.clauses() if not type(clause) in clauses_to_drop]
    return qal.build_query(remaining_clauses)


def replace_clause(query: QueryType, replacements: clauses.BaseClause | Iterable[clauses.BaseClause]) -> QueryType:
    """Creates a new SQL query with the replacements being used instead of the original clauses.

    Clauses are matched on a per-type bassis. Therefore, this function does not switch a query from implicit to explicit or
    vice-versa. Use a combination of `drop_clause` and `add_clause` for that. If a replacement is not present in the original
    query, it is simply ignored.

    No validation other than the rules of `qal.build_query` is performed.

    Parameters
    ----------
    query : QueryType
        The query to update
    replacements : clauses.BaseClause | Iterable[clauses.BaseClause]
        The new clause instances that should be used instead of the old ones.

    Returns
    -------
    QueryType
        An updated query where the matching `replacements` clauses are used in place of the clause instances that were
        originally present in the query
    """
    replacements = collection_utils.enlist(replacements)
    clauses_to_replace = {type(clause): clause for clause in replacements}
    replaced_clauses = [clauses_to_replace.get(type(current_clause), current_clause)
                        for current_clause in query.clauses()]
    return qal.build_query(replaced_clauses)


def _replace_expression_in_predicate(predicate: Optional[PredicateType],
                                     replacement: Callable[[expr.SqlExpression], expr.SqlExpression]
                                     ) -> Optional[PredicateType]:
    """Handler to update all expressions in a specific predicate.

    This method does not perform any sanity checks on the new predicate.


    Parameters
    ----------
    predicate : PredicateType
        The predicate to update. Can be ``None``, in which case no replacement is performed.
    replacement : Callable[[expr.SqlExpression], expr.SqlExpression]
        A function mapping each expression to a (potentially updated) expression

    Returns
    -------
    Optional[PredicateType]
        The updated predicate. Can be ``None``, if `predicate` already was.

    Raises
    ------
    ValueError
        If the predicate is of no known type. This indicates that this method is missing a handler for a specific predicate
        type that was added later on.
    """
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


def _replace_expression_in_table_source(table_source: Optional[clauses.TableSource],
                                        replacement: Callable[[expr.SqlExpression], expr.SqlExpression]
                                        ) -> Optional[clauses.TableSource]:
    """Handler to update all expressions in a table source.

    This method does not perform any sanity checks on the updated sources.

    Parameters
    ----------
    table_source : clauses.TableSource
        The source to update. Can be ``None``, in which case no replacement is performed.
    replacement : Callable[[expr.SqlExpression], expr.SqlExpression]
        A function mapping each expression to a (potentially updated) expression

    Returns
    -------
    Optional[clauses.TableSource]
        The updated table source. Can be ``None``, if `table_source` already was.

    Raises
    ------
    ValueError
        If the table source is of no known type. This indicates that this method is missing a handler for a specific source
        type that was added later on.
    """
    if table_source is None:
        return None
    if isinstance(table_source, clauses.DirectTableSource):
        return table_source
    elif isinstance(table_source, clauses.SubqueryTableSource):
        replaced_subquery = replacement(table_source.expression)
        assert isinstance(replaced_subquery, expr.SubqueryExpression)
        replaced_subquery = replace_expressions(replaced_subquery.query, replacement)
        return clauses.SubqueryTableSource(replaced_subquery, table_source.target_name)
    elif isinstance(table_source, clauses.JoinTableSource):
        replaced_source = _replace_expression_in_table_source(table_source.source, replacement)
        replaced_condition = _replace_expression_in_predicate(table_source.join_condition, replacement)
        return clauses.JoinTableSource(replaced_source, replaced_condition, join_type=table_source.join_type)
    else:
        raise TypeError("Unknown table source type: " + str(table_source))


def _replace_expressions_in_clause(clause: Optional[ClauseType],
                                   replacement: Callable[[expr.SqlExpression], expr.SqlExpression]) -> Optional[ClauseType]:
    """Handler to update all expressions in a clause.

    This method does not perform any sanity checks on the updated clauses.

    Parameters
    ----------
    clause : ClauseType
        The clause to update. Can be ``None``, in which case no replacement is performed.
    replacement : Callable[[expr.SqlExpression], expr.SqlExpression]
        A function mapping each expression to a (potentially updated) expression

    Returns
    -------
    Optional[ClauseType]
        The updated clause. Can be ``None``, if `clause` already was.

    Raises
    ------
    ValueError
        If the clause is of no known type. This indicates that this method is missing a handler for a specific clause type that
        was added later on.
    """
    if not clause:
        return None

    if isinstance(clause, clauses.Hint) or isinstance(clause, clauses.Explain):
        return clause
    if isinstance(clause, clauses.CommonTableExpression):
        replaced_queries = [clauses.WithQuery(replace_expressions(cte.query, replacement), cte.target_name)
                            for cte in clause.queries]
        return clauses.CommonTableExpression(replaced_queries)
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
        replaced_contents = [_replace_expression_in_table_source(target, replacement) for target in clause.items]
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
    """Updates all expressions in a query.

    The replacement handler can either produce entirely new expressions, or simply return the current expression instance if
    no update should be performed. Be very careful with this method since no sanity checks are performed, other than the rules
    of `qal.build_query`.

    Parameters
    ----------
    query : QueryType
        The query to update
    replacement : Callable[[expr.SqlExpression], expr.SqlExpression]
        A function mapping each of the current expressions in the `query` to potentially updated expressions.

    Returns
    -------
    QueryType
        The updated query
    """
    replaced_clauses = [_replace_expressions_in_clause(clause, replacement) for clause in query.clauses()]
    return qal.build_query(replaced_clauses)


def _perform_predicate_replacement(current_predicate: preds.AbstractPredicate,
                                   target_predicate: preds.AbstractPredicate,
                                   new_predicate: preds.AbstractPredicate) -> preds.AbstractPredicate:
    """Handler to change specific predicates in a predicate hierarchy to other predicates.

    This does not perform any sanity checks on the updated predicate hierarchy, nor is the hierarchy simplified.

    Parameters
    ----------
    current_predicate : preds.AbstractPredicate
        The predicate hierarchy in which the updates should occur
    target_predicate : preds.AbstractPredicate
        The predicate that should be replaced
    new_predicate : preds.AbstractPredicate
        The new predicate that should be used instead of the `target_predicate`

    Returns
    -------
    preds.AbstractPredicate
        The updated predicate
    """
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
    """Rewrites a specific query to use a new predicate in place of an old one.

    In the current implementation this does only work for top-level predicates, i.e. subqueries and CTEs are not considered.
    Furthermore, only the ``WHERE`` clause and the ``HAVING`` clause are modified, since these should be the only ones that
    contain predicates.

    If the predicate to replace is not found, nothing happens. In the same vein, no sanity checks are performed on the updated
    query.

    Parameters
    ----------
    query : qal.ImplicitSqlQuery
        The query update
    predicate_to_replace : preds.AbstractPredicate
        The old predicate that should be dropped
    new_predicate : preds.AbstractPredicate
        The predicate that should be used in place of `predicate_to_replace`. This can be an entirely different type of
        predicate, e.g. a conjunction of join conditions that replace a single join predicate.

    Returns
    -------
    qal.ImplicitSqlQuery
        The updated query
    """
    # TODO: also allow replacement in explicit SQL queries
    # TODO: allow predicate replacement in subqueries / CTEs
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
    """Handler method to replace specific column references by new references for an entire query.

    Parameters
    ----------
    query : QueryType
        The query to update
    available_renamings : dict[base.ColumnReference, base.ColumnReference]
        A dictionary mapping each of the old column values to the values that should be used instead.

    Returns
    -------
    QueryType
        The updated query

    Raises
    ------
    TypeError
        If the query is of no known type. This indicates that this method is missing a handler for a specific query type that
        was added later on.
    """
    renamed_cte = rename_columns_in_clause(query.cte_clause, available_renamings)
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
                                    cte_clause=renamed_cte,
                                    hints=query.hints, explain_clause=query.explain)
    elif isinstance(query, qal.ExplicitSqlQuery):
        return qal.ExplicitSqlQuery(select_clause=renamed_select, from_clause=renamed_from, where_clause=renamed_where,
                                    groupby_clause=renamed_groupby, having_clause=renamed_having,
                                    orderby_clause=renamed_orderby, limit_clause=query.limit_clause,
                                    cte_clause=renamed_cte,
                                    hints=query.hints, explain_clause=query.explain)
    elif isinstance(query, qal.MixedSqlQuery):
        return qal.MixedSqlQuery(select_clause=renamed_select, from_clause=renamed_from, where_clause=renamed_where,
                                 groupby_clause=renamed_groupby, having_clause=renamed_having,
                                 orderby_clause=renamed_orderby, limit_clause=query.limit_clause,
                                 cte_clause=renamed_cte,
                                 hints=query.hints, explain_clause=query.explain)
    else:
        raise TypeError("Unknown query type: " + str(query))


def _rename_columns_in_expression(expression: Optional[expr.SqlExpression],
                                  available_renamings: dict[base.ColumnReference, base.ColumnReference]
                                  ) -> Optional[expr.SqlExpression]:
    """Handler method to replace specific column references by new references in an expression.

    Parameters
    ----------
    expression : Optional[expr.SqlExpression]
        The expression to update. If ``None``, no renaming is performed.
    available_renamings : dict[base.ColumnReference, base.ColumnReference]
        A dictionary mapping each of the old column values to the values that should be used instead.

    Returns
    -------
    Optional[expr.SqlExpression]
        The updated expression. Can be ``None``, if `expression` already was.

    Raises
    ------
    ValueError
        If the expression is of no known type. This indicates that this method is missing a handler for a specific expressoin
        type that was added later on.
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
    """Replaces all references to specific columns in a predicate by new references.

    Parameters
    ----------
    predicate : Optional[preds.AbstractPredicate]
        The predicate to update. Can be ``None``, in which case no update is performed.
    available_renamings : dict[base.ColumnReference, base.ColumnReference]
        A dictionary mapping each of the old column values to the values that should be used instead.

    Returns
    -------
    Optional[preds.AbstractPredicate]
        The updated predicate. Can be ``None``, if `predicate` already was.

    Raises
    ------
    ValueError
        If the query is of no known type. This indicates that this method is missing a handler for a specific query type that
        was added later on.
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
    """Handler method to replace all references to specific columns by new columns.

    Parameters
    ----------
    table_source : clauses.TableSource
        The source that should be updated
    available_renamings : dict[base.ColumnReference, base.ColumnReference]
        A dictionary mapping each of the old column values to the values that should be used instead.

    Returns
    -------
    Optional[clauses.TableSource]
        The updated source. Can be ``None``, if `table_source` already was.

    Raises
    ------
    TypeError
        If the source is of no known type. This indicates that this method is missing a handler for a specific source type that
        was added later on.
    """
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
    """Replaces all references to specific columns in a clause by new columns.

    Parameters
    ----------
    clause : Optional[ClauseType]
        The clause to update. Can be ``None``, in which case no update is performed.
    available_renamings : dict[base.ColumnReference, base.ColumnReference]
        A dictionary mapping each of the old column values to the values that should be used instead.

    Returns
    -------
    Optional[ClauseType]
        The updated clause. Can be ``None``, if `clause` already was.

    Raises
    ------
    ValueError
        If the clause is of no known type. This indicates that this method is missing a handler for a specific clause type that
        was added later on.
    """
    if not clause:
        return None

    if isinstance(clause, clauses.Hint) or isinstance(clause, clauses.Explain):
        return clause
    if isinstance(clause, clauses.CommonTableExpression):
        renamed_ctes = [clauses.WithQuery(_rename_columns_in_query(cte.query, available_renamings), cte.target_name)
                        for cte in clause.queries]
        return clauses.CommonTableExpression(renamed_ctes)
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
                           for table_source in clause.items]
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
    """Changes all references to a specific table to refer to another table instead.

    Parameters
    ----------
    source_query : QueryType
        The query that should be updated
    from_table : base.TableReference
        The table that should be replaced
    target_table : base.TableReference
        The table that should be used instead
    prefix_column_names : bool, optional
        Whether a prefix should be added to column names. If this is ``True``, column references will be changed in two ways:

        1. if they belonged to the `from_table`, they will now belong to the `target_table` after the renaming
        2. The column names will be changed to include the identifier of the `from_table` as a prefix.

    Returns
    -------
    QueryType
        The updated query
    """
    necessary_renamings: dict[base.ColumnReference, base.ColumnReference] = {}
    for column in filter(lambda col: col.table == from_table, source_query.columns()):
        new_column_name = f"{column.table.identifier()}_{column.name}" if prefix_column_names else column.name
        necessary_renamings[column] = base.ColumnReference(new_column_name, target_table)
    return _rename_columns_in_query(source_query, necessary_renamings)


def bind_columns(query: QueryType, *, with_schema: bool = True,
                 db_schema: Optional[db.DatabaseSchema] = None) -> QueryType:
    """Determines the tables that each column belongs to and sets the appropriate references.

    This binding of columns to their tables happens in two phases: During the first phase, a *syntactic* binding is performed.
    This operates on column names of the form ``<alias>.<column name>``, where ``<alias>`` is either an actual alias of a table
    from the ``FROM`` clause, or the full name of such a table. For all such names, the reference is set up directly.
    During the second phase, a *schema* binding is performed. This is applied to all columns that could not be bound during the
    first phase and involves querying the schema catalog of a live database. It determines which of the tables from the
    ``FROM`` clause contain a column with a name similar to the name of the unbound column and sets up the corresponding table
    reference. If multiple tables contain a specific column, any of them might be chosen. The second phase is entirely
    optional and can be skipped altogether. In this case, some columns might end up without a valid table reference, however.
    This in turn might break some applications.

    Parameters
    ----------
    query : QueryType
        The query whose columns should be bound
    with_schema : bool, optional
        Whether the second binding phase based on the schema catalog of a live database should be performed. This is enabled by
        default
    db_schema : Optional[db.DatabaseSchema], optional
        The schema to use for the second binding phase. If `with_schema` is enabled, but this parameter is ``None``, the schema
        is inferred based on the current database of the `DatabasePool`. This defaults to ``None``.

    Returns
    -------
    QueryType
        The updated query. Notice that some columns might still remain unbound if none of the phases was able to find a table.
    """

    table_alias_map: dict[str, base.TableReference] = {table.identifier(): table for table in query.tables()
                                                       if table.full_name or table.virtual}
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
