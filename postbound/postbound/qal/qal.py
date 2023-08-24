"""*qal* describes the query abstraction layer of PostBOUND. It offers interfaces to operate with SQL queries on a high level.

The most important type of our query abstraction is the `SqlQuery` class, which is defined in this module. It focuses on
modelling an entire SQL query with all important concepts. Notice that the focus here really in on modelling - nearly no
interactive functionality, no input/output capabilities and no modification tools are provided. These are handled by
dedicated modules (e.g. the `parser` module for reading queries from text, or the `transform` module for changing existing
query objects).

In addition to the pure `SqlQuery`, a number of subclasses exist. These model queries with specific ``FROM`` clauses. For
example, the `ImplicitSqlQuery` provides an `ImplicitFromClause` that restricts how tables can be referenced in this clause.
For some use-cases, these might be easier to work with than the more general `SqlQuery` class, where much more diverse ``FROM``
clauses are permitted.

In line with the other parts of the query abstraction layer, queries are designed as read-only data objects. Any forced
modifications on queries will break the entire qal and result in unpredictable behaviour.
"""
from __future__ import annotations

import abc
import functools
import typing
from collections.abc import Collection, Iterable, Sequence
from typing import Generic, Optional

from postbound.qal import base, clauses, expressions as expr, predicates as preds
from postbound.util import collections as collection_utils


def _stringify_clause(clause: clauses.BaseClause) -> str:
    """Handler method to provide a refined string for a specific given clause, to be used by the `SqlQuery` ``__str__`` method.

    This method is slightly smarter than calling ``__str__`` directly, because it inserts newlines at sensible places in a
    query, e.g. after the hint block.

    Parameters
    ----------
    clause : clauses.BaseClause
        The clause to build the string representation

    Returns
    -------
    str
        The string representation of the clause
    """
    if isinstance(clause, clauses.Hint):
        return str(clause) + "\n"
    return str(clause) + " "


def _collect_subqueries_in_expression(expression: expr.SqlExpression) -> set[SqlQuery]:
    """Handler method to provide all the subqueries that are contained in a specific expression.

    Parameters
    ----------
    expression : expr.SqlExpression
        The expression to analyze

    Returns
    -------
    set[SqlQuery]
        The subqueries from the `expression`

    See Also
    --------
    SqlQuery.subqueries
    """
    if isinstance(expression, expr.SubqueryExpression):
        return {expression.query}
    return collection_utils.set_union(_collect_subqueries_in_expression(child_expr)
                                      for child_expr in expression.iterchildren())


def _collect_subqueries_in_table_source(table_source: clauses.TableSource) -> set[SqlQuery]:
    """Handler method to provide all subqueries that are contained in a specific table.

    This does not collect the subqueries in a recursive manner: once a subquery has been found, the collection stops.
    Therefore, subqueries that are contained in subqueries are not collected.

    Parameters
    ----------
    table_source : clauses.TableSource
        The table to analyze

    Returns
    -------
    set[SqlQuery]
        The subqueries that were found

    See Also
    --------
    SqlQuery.subqueries
    """
    if isinstance(table_source, clauses.SubqueryTableSource):
        return {table_source.query}
    elif isinstance(table_source, clauses.JoinTableSource):
        source_subqueries = _collect_subqueries_in_table_source(table_source.source)
        condition_subqueries = (collection_utils.set_union(_collect_subqueries_in_expression(cond_expr) for cond_expr
                                                           in table_source.join_condition.iterexpressions())
                                if table_source.join_condition else set())
        return source_subqueries | condition_subqueries
    else:
        return set()


def _collect_subqueries(clause: clauses.BaseClause) -> set[SqlQuery]:
    """Handler method to provide all the subqueries that are contained in a specific clause.

    Following the definitions of `SqlQuery.subqueries`, this completely ignores CTEs. Therefore, subqueries that are defined
    within CTEs are not detected.

    Parameters
    ----------
    clause : clauses.BaseClause
        The clause to check

    Returns
    -------
    set[SqlQuery]
        All subqueries that have been found in the clause.

    Raises
    ------
    ValueError
        If the given clause is unknown. This indicates that this method is missing a handler for a specific clause type that
        was added later on.

    See Also
    --------
    SqlQuery.subqueries
    """
    if isinstance(clause, clauses.Hint) or isinstance(clause, clauses.Limit) or isinstance(clause, clauses.Explain):
        return set()

    if isinstance(clause, clauses.CommonTableExpression):
        return set()
    elif isinstance(clause, clauses.Select):
        return collection_utils.set_union(_collect_subqueries_in_expression(target.expression)
                                          for target in clause.targets)
    elif isinstance(clause, clauses.ImplicitFromClause):
        return set()
    elif isinstance(clause, clauses.From):
        return collection_utils.set_union(_collect_subqueries_in_table_source(src) for src in clause.items)
    elif isinstance(clause, clauses.Where):
        where_predicate = clause.predicate
        return collection_utils.set_union(_collect_subqueries_in_expression(expression)
                                          for expression in where_predicate.iterexpressions())
    elif isinstance(clause, clauses.GroupBy):
        return collection_utils.set_union(_collect_subqueries_in_expression(column) for column in clause.group_columns)
    elif isinstance(clause, clauses.Having):
        having_predicate = clause.condition
        return collection_utils.set_union(_collect_subqueries_in_expression(expression)
                                          for expression in having_predicate.iterexpressions())
    elif isinstance(clause, clauses.OrderBy):
        return collection_utils.set_union(_collect_subqueries_in_expression(expression.column)
                                          for expression in clause.expressions)
    else:
        raise ValueError(f"Unknown clause type: {clause}")


def _collect_bound_tables_from_source(table_source: clauses.TableSource) -> set[base.TableReference]:
    """Handler method to provide all tables that are "produced" by a table source.

    "Produced" tables are tables that are either directly referenced in the ``FROM`` clause (e.g. ``FROM R``), referenced as
    part of joins (e.g. ``FROM R JOIN S ON ...``), or part of the ``FROM`` clauses of subqueries. In contrast, an unbound table
    is one that has to be provided by "context", such as the dependent table in a dependent subquery.

    Parameters
    ----------
    table_source : clauses.TableSource
        The table to check

    Returns
    -------
    set[base.TableReference]
        The "produced" tables.

    See Also
    --------
    SqlQuery.bound_tables
    """
    if isinstance(table_source, clauses.DirectTableSource):
        return {table_source.table}
    elif isinstance(table_source, clauses.SubqueryTableSource):
        return _collect_bound_tables(table_source.query.from_clause)
    elif isinstance(table_source, clauses.JoinTableSource):
        return _collect_bound_tables_from_source(table_source.source)


def _collect_bound_tables(from_clause: clauses.From) -> set[base.TableReference]:
    """Handler method to provide all tables that are "produced" in the given clause.

    "Produced" tables are tables that are either directly referenced in the ``FROM`` clause (e.g. ``FROM R``), referenced as
    part of joins (e.g. ``FROM R JOIN S ON ...``), or part of the ``FROM`` clauses of subqueries. In contrast, an unbound table
    is one that has to be provided by "context", such as the dependent table in a dependent subquery.

    Parameters
    ----------
    from_clause : clauses.From
        The clause to check

    Returns
    -------
    set[base.TableReference]
        The "produced" tables

    See Also
    --------
    SqlQuery.bound_tables
    """
    if isinstance(from_clause, clauses.ImplicitFromClause):
        return from_clause.tables()
    else:
        return collection_utils.set_union(_collect_bound_tables_from_source(src) for src in from_clause.items)


FromClauseType = typing.TypeVar("FromClauseType", bound=clauses.From)


class SqlQuery(Generic[FromClauseType], abc.ABC):
    """Represents an arbitrary SQL query, providing direct access to the different clauses in the query.

    At a basic level, PostBOUND differentiates between two types of queries:

    - implicit SQL queries specify all referenced tables in the ``FROM`` clause and the join predicates in the ``WHERE``
      clause, e.g. ``SELECT * FROM R, S WHERE R.a = S.b AND R.c = 42``. This is the traditional way of writing SQL queries.
    - explicit SQL queries use the ``JOIN ON`` syntax to reference tables, e.g.
      ``SELECT * FROM R JOIN S ON R.a = S.b WHERE R.c = 42``. This is the more "modern" way of writing SQL queries.

    There is also a third possibility of mixing the implicit and explicit syntax. For each of these cases, designated
    subqueries exist. They all provide the same functionality and only differ in the (sub-)types of their ``FROM`` clauses.
    Therefore, these classes can be considered as "marker" types to indicate that at a certain point of a computation, a
    specific kind of query is required. The `SqlQuery` class acts as a superclass that specifies the general behaviour of all
    query instances and can act as the most general type of query.

    The clauses of each query can be accessed via properties. If a clause is optional, the absence of the clause is indicated
    through a ``None`` value. All additional behaviour of the queries is provided by the different methods. These are mostly
    focused on an easy introspection of the query's structure.

    Notice that PostBOUND does not enforce any semantics on the queries (e.g. regarding data types, access to values, the
    cardinality of subquery results, or the connection between different clauses). This has to be done by the user, or by the
    actual database system.

    Parameters
    ----------
    select_clause : clauses.Select
        The ``SELECT`` part of the query. This is the only required part of a query. Notice however, that some database systems
        do not allow queries without a ``FROM`` clause.
    from_clause : Optional[FromClauseType], optional
        The ``FROM`` part of the query, by default ``None``
    where_clause : Optional[clauses.Where], optional
        The ``WHERE`` part of the query, by default ``None``
    groupby_clause : Optional[clauses.GroupBy], optional
        The ``GROUP BY`` part of the query, by default ``None``
    having_clause : Optional[clauses.Having], optional
        The ``HAVING`` part of the query, by default ``None``.
    orderby_clause : Optional[clauses.OrderBy], optional
        The ``ORDER BY`` part of the query, by default ``None``
    limit_clause : Optional[clauses.Limit], optional
        The ``LIMIT`` and ``OFFSET`` part of the query. In standard SQL, this is designated using the ``FETCH FIRST`` syntax.
        Defaults to ``None``.
    cte_clause : Optional[clauses.CommonTableExpression], optional
        The ``WITH`` part of the query, by default ``None``
    hints : Optional[clauses.Hint], optional
        The hint block of the query. Hints are not part of standard SQL and follow a completely system-specific syntax. Even
        their placement in within the query varies from system to system and from extension to extension. Defaults to ``None``.
    explain : Optional[clauses.Explain], optional
        The ``EXPLAIN`` part of the query. Like hints, this is not part of standard SQL. However, most systems provide
        ``EXPLAIN`` functionality. The specific features and syntax are quite similar, but still system specific. Defaults to
        ``None``.
    """

    def __init__(self, *, select_clause: clauses.Select,
                 from_clause: Optional[FromClauseType] = None, where_clause: Optional[clauses.Where] = None,
                 groupby_clause: Optional[clauses.GroupBy] = None, having_clause: Optional[clauses.Having] = None,
                 orderby_clause: Optional[clauses.OrderBy] = None, limit_clause: Optional[clauses.Limit] = None,
                 cte_clause: Optional[clauses.CommonTableExpression] = None,
                 hints: Optional[clauses.Hint] = None, explain: Optional[clauses.Explain] = None) -> None:
        self._cte_clause = cte_clause
        self._select_clause = select_clause
        self._from_clause = from_clause
        self._where_clause = where_clause
        self._groupby_clause = groupby_clause
        self._having_clause = having_clause
        self._orderby_clause = orderby_clause
        self._limit_clause = limit_clause
        self._hints = hints
        self._explain = explain

        self._hash_val = hash((self._hints, self._explain,
                               self._cte_clause,
                               self._select_clause, self._from_clause, self._where_clause,
                               self._groupby_clause, self._having_clause,
                               self._orderby_clause, self._limit_clause))

    @property
    def cte_clause(self) -> Optional[clauses.CommonTableExpression]:
        """Get the ``WITH`` clause of the query.

        Returns
        -------
        Optional[clauses.CommonTableExpression]
            The ``WITH`` clause if it was specified, or ``None`` otherwise.
        """
        return self._cte_clause

    @property
    def select_clause(self) -> clauses.Select:
        """Get the ``SELECT`` clause of the query. Will always be set.

        Returns
        -------
        clauses.Select
            The ``SELECT`` clause
        """
        return self._select_clause

    @property
    def from_clause(self) -> Optional[FromClauseType]:
        """Get the ``FROM`` clause of the query.

        Returns
        -------
        Optional[FromClauseType]
            The ``FROM`` clause if it was specified, or ``None`` otherwise.
        """
        return self._from_clause

    @property
    def where_clause(self) -> Optional[clauses.Where]:
        """Get the ``WHERE`` clause of the query.

        Returns
        -------
        Optional[clauses.Where]
            The ``WHERE`` clause if it was specified, or ``None`` otherwise.
        """
        return self._where_clause

    @property
    def groupby_clause(self) -> Optional[clauses.GroupBy]:
        """Get the ``GROUP BY`` clause of the query.

        Returns
        -------
        Optional[clauses.GroupBy]
            The ``GROUP BY`` clause if it was specified, or ``None`` otherwise.
        """
        return self._groupby_clause

    @property
    def having_clause(self) -> Optional[clauses.Having]:
        """Get the ``HAVING`` clause of the query.

        Returns
        -------
        Optional[clauses.Having]
            The ``HAVING`` clause if it was specified, or ``None`` otherwise.
        """
        return self._having_clause

    @property
    def orderby_clause(self) -> Optional[clauses.OrderBy]:
        """Get the ``ORDER BY`` clause of the query.

        Returns
        -------
        Optional[clauses.OrderBy]
            The ``ORDER BY`` clause if it was specified, or ``None`` otherwise.
        """
        return self._orderby_clause

    @property
    def limit_clause(self) -> Optional[clauses.Limit]:
        """Get the combined ``LIMIT`` and ``OFFSET`` clauses of the query.

        According to the SQL standard, these clauses should use the ``FETCH FIRST`` syntax. However, many systems use
        ``OFFSET`` and ``LIMIT`` instead.

        Returns
        -------
        Optional[clauses.Limit]
            The ``FETCH FIRST`` clause if it was specified, or ``None`` otherwise.
        """
        return self._limit_clause

    @property
    def hints(self) -> Optional[clauses.Hint]:
        """Get the hint block of the query.

        The hints can specify preparatory statements that have to be executed before the actual query is run in addition to the
        hints themselves.

        Returns
        -------
        Optional[clauses.Hint]
            The hint block if it was specified, or ``None`` otherwise.
        """
        return self._hints

    @property
    def explain(self) -> Optional[clauses.Explain]:
        """Get the ``EXPLAIN`` block of the query.

        Returns
        -------
        Optional[clauses.Explain]
            The ``EXPLAIN`` settings if specified, or ``None`` otherwise.
        """
        return self._explain

    @abc.abstractmethod
    def is_implicit(self) -> bool:
        """Checks, whether this query has an implicit ``FROM`` clause.

        The implicit ``FROM`` clause only consists of the source tables that should be scanned for the query. No subqueries or
        joins are contained in the clause. All join predicates must be part of the ``WHERE`` clause.

        Returns
        -------
        bool
            Whether the query is implicit

        See Also
        --------
        ImplicitSqlQuery
        """
        raise NotImplementedError

    @abc.abstractmethod
    def is_explicit(self) -> bool:
        """Checks, whether this query has an explicit ``FROM`` clause.

        The explicit ``FROM`` clause exclusively makes use of the ``JOIN ON`` syntax to denote both the tables that should be
        scanned, and the predicates that should be used to join the tables together. Therefore, the ``WHERE`` clause should
        only consist of filter predicates on the base tables. However, this is not enforced and the contents of the ``ON``
        conditions as well as the ``WHERE`` clause can be arbitrary predicates.

        Returns
        -------
        bool
            Whether the query is explicit

        See Also
        --------
        ExplicitSqlQuery
        """
        raise NotImplementedError

    def is_explain(self) -> bool:
        """Checks, whether this query is an ``EXPLAIN`` query rather than a normal SQL query.

        An ``EXPLAIN`` query is not executed like a normal ``SELECT ...`` query. Instead of actually calculating a result set,
        the database system only provides a query plan. This plan is the execution plan that would be used, had the query been
        entered as a normal SQL query.

        Returns
        -------
        bool
            Whether this query should be explained, rather than executed.
        """
        return self.explain is not None

    @functools.cache
    def tables(self) -> set[base.TableReference]:
        """Provides all tables that are referenced at any point in the query.

        This includes tables from all clauses. Virtual tables will be included and tables that are only scanned within
        subqueries are included as well. Notice however, that some database systems might not support subqueries to be put
        at arbitrary positions in the query (e.g. ``GROUP BY`` clause).

        Returns
        -------
        set[base.TableReference]
            All tables that are referenced in the query.
        """
        return collection_utils.set_union(clause.tables() for clause in self.clauses())

    def columns(self) -> set[base.ColumnReference]:
        """Provides all columns that are referenced at any point in the query.

        This includes columns from all clauses and does not account for renamed columns from subqueries. For example, consider
        the query ``SELECT R.a, my_sq.b FROM R JOIN (SELECT b FROM S) my_sq ON R.a < my_sq.b``. `columns` would return the
        following set: ``{R.a, S.b, my_sq.b}``, even though ``my_sq.b`` can be considered as just an alias for ``S.b``.

        Returns
        -------
        set[base.ColumnReference]
            All columns that are referenced in the query.
        """
        return collection_utils.set_union(clause.columns() for clause in self.clauses())

    @functools.cache
    def predicates(self) -> preds.QueryPredicates:
        """Provides all predicates in this query.

        *All* predicates really means *all* predicates: this includes predicates that appear in the ``FROM`` clause, the
        ``WHERE`` clause, as well as any predicates from CTEs.

        Returns
        -------
        preds.QueryPredicates
            A predicates wrapper around the conjunction of all individual predicates.
        """
        predicate_handler = preds.DefaultPredicateHandler
        current_predicate = predicate_handler.empty_predicate()

        if self.cte_clause:
            for with_query in self.cte_clause.queries:
                current_predicate = current_predicate.and_(with_query.query.predicates())

        if self.where_clause:
            current_predicate = current_predicate.and_(self.where_clause.predicate)

        from_predicates = self.from_clause.predicates()
        if from_predicates:
            current_predicate = current_predicate.and_(from_predicates)

        return current_predicate

    def subqueries(self) -> Collection[SqlQuery]:
        """Provides all subqueries that are referenced in this query.

        Notice that CTEs are ignored by this method, since they can be accessed directly via the `cte_clause` property.

        Returns
        -------
        Collection[SqlQuery]
            All subqueries that appear in any of the "inner" clauses of the query
        """
        return collection_utils.set_union(_collect_subqueries(clause) for clause in self.clauses())

    def clauses(self) -> Sequence[clauses.BaseClause]:
        """Provides all the clauses that are defined (i.e. not ``None``) in this query.

        Returns
        -------
        Sequence[clauses.BaseClause]
            The clauses. The current order of the clauses is as follows: hints, explain, cte, select, from, where, group by,
            having, order by, limit. Notice however, that this order is not strictly standardized and may change in the future.
            All clauses that are not specified on the query will be skipped.
        """
        all_clauses = [self.hints, self.explain, self.cte_clause,
                       self.select_clause, self.from_clause, self.where_clause,
                       self.groupby_clause, self.having_clause,
                       self.orderby_clause, self.limit_clause]
        return [clause for clause in all_clauses if clause is not None]

    def bound_tables(self) -> set[base.TableReference]:
        """Provides all tables that can be assigned to a physical or virtual table reference in this query.

        Bound tables are those tables, that are selected in the ``FROM`` clause of the query, or a subquery. Conversely,
        unbound tables are those that have to be "injected" by an outer query, as is the case for dependent subqueries.

        For example, the query ``SELECT * FROM R, S WHERE R.a = S.b`` has two bound tables: ``R`` and ``S``.
        On the other hand, the query ``SELECT * FROM R WHERE R.a = S.b`` has only bound ``R``, whereas ``S`` has to be bound in
        a surrounding query.

        Returns
        -------
        set[base.TableReference]
            All tables that are bound (i.e. listed in the ``FROM`` clause) of the query.
        """
        subquery_produced_tables = collection_utils.set_union(subquery.bound_tables()
                                                              for subquery in self.subqueries())
        cte_produced_tables = self.cte_clause.tables() if self.cte_clause else set()
        own_produced_tables = _collect_bound_tables(self.from_clause)
        return own_produced_tables | subquery_produced_tables | cte_produced_tables

    def unbound_tables(self) -> set[base.TableReference]:
        """Provides all tables that are referenced in this query but not bound.

        While `tables()` provides all tables that are referenced in this query in any way, `bound_tables` restricts
        these tables. This method provides the complementary set to `bound_tables` i.e.
        ``tables = bound_tables ⊕ unbound_tables``.

        Returns
        -------
        set[base.TableReference]
            The unbound tables that have to be supplied as part of an outer query
        """
        if self.from_clause:
            virtual_subquery_targets = {subquery_source.target_table for subquery_source in self.from_clause.items
                                        if isinstance(subquery_source, clauses.SubqueryTableSource)}
        else:
            virtual_subquery_targets = set()

        if self.cte_clause:
            virtual_cte_targets = {with_query.target_table for with_query in self.cte_clause.queries}
        else:
            virtual_cte_targets = set()

        return self.tables() - self.bound_tables() - virtual_subquery_targets - virtual_cte_targets

    def is_ordered(self) -> bool:
        """Checks, whether this query produces its result tuples in order.

        Returns
        -------
        bool
            Whether a valid ``ORDER BY`` clause was specified on the query.
        """
        return self.orderby_clause is not None

    def is_dependent(self) -> bool:
        """Checks, whether all columns that are referenced in this query are provided by the tables from this query.

        In order for this check to work, all columns have to be bound to actual tables, i.e. the `tables` attribute of all
        column references have to be set to a valid object.

        Returns
        -------
        bool
            Whether all columns belong to tables that are bound by this query
        """
        return not (self.tables() <= self.bound_tables())

    def iterexpressions(self) -> Iterable[expr.SqlExpression]:
        """Provides access to all expressions that are directly contained in this query.

        Nested expressions can be accessed from these expressions in a recursive manner (see the `SqlExpression`
        interface for details).

        Returns
        -------
        Iterable[SqlExpression]
            The expressions
        """
        return collection_utils.flatten(clause.iterexpressions() for clause in self.clauses())

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        """Provides access to all column in this query.

        In contrast to the `columns` method, duplicates are returned multiple times, i.e. if a column is referenced *n* times
        in this query, it will also be returned *n* times by this method. Furthermore, the order in which columns are provided
        by the iterable matches the order in which they appear in this query.

        Returns
        -------
        Iterable[base.ColumnReference]
            The columns
        """
        return collection_utils.flatten(clause.itercolumns() for clause in self.clauses())

    def stringify(self, *, trailing_delimiter: bool = True) -> str:
        """Provides a string representation of this query.

        The only difference to calling `str` directly, is that the `stringify` method provides control over whether a trailing
        delimiter should be appended to the query.

        Parameters
        ----------
        trailing_delimiter : bool, optional
            Whether a delimiter should be appended to the query. Defaults to ``True``.

        Returns
        -------
        str
            A string representation of this query
        """
        delim = ";" if trailing_delimiter else ""
        return "".join(_stringify_clause(clause) for clause in self.clauses()).rstrip() + delim

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self.clauses() == other.clauses()

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return self.stringify(trailing_delimiter=True)


class ImplicitSqlQuery(SqlQuery[clauses.ImplicitFromClause]):
    """An implicit query restricts the constructs that may appear in the ``FROM`` clause.

    For implicit queries, the ``FROM`` clause may only consist of simple table sources. All join conditions have to be put in
    the ``WHERE`` clause. Notice that this does not restrict the structure of other clauses. For example, the ``WHERE`` clause
    can still contain subqueries. As a special case, queries without a ``FROM`` clause are also considered implicit.

    The attributes and parameters for this query type are the same as for `SqlQuery`, only the type of the `From` clause is
    restricted.

    See Also
    --------
    clauses.ImplicitFromClause
    ExplicitSqlQuery

    Examples
    --------
    The following queries are considered as implicit queries:

    .. code-block:: sql

        SELECT *
        FROM R, S, T
        WHERE R.a = S.b
            AND S.b = T.c
            AND R.a < 42

    .. code-block:: sql

        SELECT *
        FROM R, S, T
        WHERE R.a = S.b
            AND S.b = T.c
            AND R.a = (SELECT MIN(R.a) FROM R)
    """

    def __init__(self, *, select_clause: clauses.Select,
                 from_clause: Optional[clauses.ImplicitFromClause] = None,
                 where_clause: Optional[clauses.Where] = None,
                 groupby_clause: Optional[clauses.GroupBy] = None,
                 having_clause: Optional[clauses.Having] = None,
                 orderby_clause: Optional[clauses.OrderBy] = None,
                 limit_clause: Optional[clauses.Limit] = None,
                 cte_clause: Optional[clauses.CommonTableExpression] = None,
                 explain_clause: Optional[clauses.Explain] = None,
                 hints: Optional[clauses.Hint] = None) -> None:
        super().__init__(select_clause=select_clause, from_clause=from_clause, where_clause=where_clause,
                         groupby_clause=groupby_clause, having_clause=having_clause,
                         orderby_clause=orderby_clause, limit_clause=limit_clause,
                         cte_clause=cte_clause,
                         explain=explain_clause, hints=hints)

    def is_implicit(self) -> bool:
        return True

    def is_explicit(self) -> bool:
        return False


class ExplicitSqlQuery(SqlQuery[clauses.ExplicitFromClause]):
    """An explicit query restricts the constructs that may appear in the ``FROM`` clause.

    For explicit queries, the ``FROM`` clause must utilize the ``JOIN ON`` syntax for all tables. The join conditions should
    be put into the ``ON`` blocks. Notice however, that PostBOUND does not perform any sanity checks here. Therefore, it is
    possible to put mix joins and filters in the ``ON`` blocks, move all joins to the ``WHERE`` clause or scatter the join
    conditions between the two clauses. Whether this is good style is up for debate, but at least PostBOUND does allow it. In
    contrast to the implicit query, subqueries are also allowed as table sources.

    Notice that each explicit query must join at least two tables in its ``FROM`` clause.

    The attributes and parameters for this query type are the same as for `SqlQuery`, only the type of the `From` clause is
    restricted.

    See Also
    --------
    clauses.ExplicitFromClause
    ImplicitSqlQuery

    Examples
    --------
    The following queries are considered as explicit queries:

    .. code-block:: sql

        SELECT *
        FROM R
            JOIN S ON R.a = S.b
            JOIN T ON S.b = T.c
        WHERE R.a < 42

    .. code-block:: sql

        SELECT *
        FROM R
            JOIN S ON R.a = S.b AND R.a = (SELECT MIN(R.a) FROM R)
            JOIN T ON S.b = T.c
    """

    def __init__(self, *, select_clause: clauses.Select,
                 from_clause: Optional[clauses.ExplicitFromClause] = None,
                 where_clause: Optional[clauses.Where] = None,
                 groupby_clause: Optional[clauses.GroupBy] = None,
                 having_clause: Optional[clauses.Having] = None,
                 orderby_clause: Optional[clauses.OrderBy] = None,
                 limit_clause: Optional[clauses.Limit] = None,
                 cte_clause: Optional[clauses.CommonTableExpression] = None,
                 explain_clause: Optional[clauses.Explain] = None,
                 hints: Optional[clauses.Hint] = None) -> None:
        super().__init__(select_clause=select_clause, from_clause=from_clause, where_clause=where_clause,
                         groupby_clause=groupby_clause, having_clause=having_clause,
                         orderby_clause=orderby_clause, limit_clause=limit_clause,
                         cte_clause=cte_clause,
                         explain=explain_clause, hints=hints)

    def is_implicit(self) -> bool:
        return False

    def is_explicit(self) -> bool:
        return True


class MixedSqlQuery(SqlQuery[clauses.From]):
    """A mixed query allows for both the explicit as well as the implicit syntax to be used within the same ``FROM`` clause.

    The mixed query complements `ImplicitSqlQuery` and `ExplicitSqlQuery` by removing the "purity" restriction: the tables that
    appear in the ``FROM`` clause can be described using either plain references or subqueries and they are free to use the
    ``JOIN ON`` syntax. The only thing that is not allowed as a ``FROM`` clause is an instance of `ImplicitFromClause` or an
    instance of `ExplicitFromClause`, since those cases are already covered by their respective query classes.

    Notice however, that we currently do not enforce the `From` clause to not be a valid explicit or implicit clause. All
    checks happen on a type level. If the contents of a general `From` clause just happen to also be a valid
    `ImplicitFromClause`, this is fine.

    The attributes and parameters for this query type are the same as for `SqlQuery`, only the type of the `From` clause is
    restricted.

    Raises
    ------
    ValueError
        If the given `from_clause` is either an implicit ``FROM`` clause or an explicit one.
    """
    def __init__(self, *, select_clause: clauses.Select,
                 from_clause: Optional[clauses.From] = None,
                 where_clause: Optional[clauses.Where] = None,
                 groupby_clause: Optional[clauses.GroupBy] = None,
                 having_clause: Optional[clauses.Having] = None,
                 orderby_clause: Optional[clauses.OrderBy] = None,
                 limit_clause: Optional[clauses.Limit] = None,
                 cte_clause: Optional[clauses.CommonTableExpression] = None,
                 explain_clause: Optional[clauses.Explain] = None,
                 hints: Optional[clauses.Hint] = None) -> None:
        if isinstance(from_clause, clauses.ExplicitFromClause) or isinstance(from_clause, clauses.ImplicitFromClause):
            raise ValueError("MixedSqlQuery cannot be combined with explicit/implicit FROM clause")
        super().__init__(select_clause=select_clause, from_clause=from_clause, where_clause=where_clause,
                         groupby_clause=groupby_clause, having_clause=having_clause,
                         orderby_clause=orderby_clause, limit_clause=limit_clause,
                         cte_clause=cte_clause,
                         explain=explain_clause, hints=hints)

    def is_implicit(self) -> bool:
        return False

    def is_explicit(self) -> bool:
        return False


def build_query(query_clauses: Iterable[clauses.BaseClause]) -> SqlQuery:
    """Constructs an SQL query based on specific clauses.

    No validation is performed. If clauses appear multiple times, later clauses overwrite former ones. The specific
    type of query (i.e. implicit, explicit or mixed) is inferred from the clauses (i.e. occurrence of an implicit ``FROM``
    clause enforces an `ImplicitSqlQuery` and vice-versa). The overwriting rules apply here as well: a later `From` clause
    overwrites a former one and can change the type of the produced query.

    Parameters
    ----------
    query_clauses : Iterable[clauses.BaseClause]
        The clauses that should be used to construct the query

    Returns
    -------
    SqlQuery
        A query consisting of the specified clauses

    Raises
    ------
    ValueError
        If `query_clauses` does not contain a `Select` clause
    ValueError
        If any of the clause types is unknown. This indicates that this method is missing a handler for a specific clause type
        that was added later on.
    """
    build_implicit_query, build_explicit_query = True, True

    cte_clause = None
    select_clause, from_clause, where_clause = None, None, None
    groupby_clause, having_clause = None, None
    orderby_clause, limit_clause = None, None
    explain_clause, hints_clause = None, None
    for clause in query_clauses:
        if not clause:
            continue

        if isinstance(clause, clauses.CommonTableExpression):
            cte_clause = clause
        elif isinstance(clause, clauses.Select):
            select_clause = clause
        elif isinstance(clause, clauses.ImplicitFromClause):
            from_clause = clause
            build_implicit_query, build_explicit_query = True, False
        elif isinstance(clause, clauses.ExplicitFromClause):
            from_clause = clause
            build_implicit_query, build_explicit_query = False, True
        elif isinstance(clause, clauses.From):
            from_clause = clause
            build_implicit_query, build_explicit_query = False, False
        elif isinstance(clause, clauses.Where):
            where_clause = clause
        elif isinstance(clause, clauses.GroupBy):
            groupby_clause = clause
        elif isinstance(clause, clauses.Having):
            having_clause = clause
        elif isinstance(clause, clauses.OrderBy):
            orderby_clause = clause
        elif isinstance(clause, clauses.Limit):
            limit_clause = clause
        elif isinstance(clause, clauses.Explain):
            explain_clause = clause
        elif isinstance(clause, clauses.Hint):
            hints_clause = clause
        else:
            raise ValueError("Unknown clause type: " + str(clause))

    if select_clause is None:
        raise ValueError("No SELECT clause detected")

    if build_implicit_query:
        return ImplicitSqlQuery(select_clause=select_clause, from_clause=from_clause, where_clause=where_clause,
                                groupby_clause=groupby_clause, having_clause=having_clause,
                                orderby_clause=orderby_clause, limit_clause=limit_clause,
                                cte_clause=cte_clause,
                                hints=hints_clause, explain_clause=explain_clause)
    elif build_explicit_query:
        return ExplicitSqlQuery(select_clause=select_clause, from_clause=from_clause, where_clause=where_clause,
                                groupby_clause=groupby_clause, having_clause=having_clause,
                                orderby_clause=orderby_clause, limit_clause=limit_clause,
                                cte_clause=cte_clause,
                                hints=hints_clause, explain_clause=explain_clause)
    else:
        return MixedSqlQuery(select_clause=select_clause, from_clause=from_clause, where_clause=where_clause,
                             groupby_clause=groupby_clause, having_clause=having_clause,
                             orderby_clause=orderby_clause, limit_clause=limit_clause,
                             cte_clause=cte_clause,
                             hints=hints_clause, explain_clause=explain_clause)
