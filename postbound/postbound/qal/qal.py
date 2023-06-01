"""`qal` is the query abstraction layer of PostBOUND. It offers interfaces to operate with SQL queries on a high level.

The most important type of `qal` is `SqlQuery` along with the classes that inherit from it. Notice that this package
focuses on the representation of queries. Other packages of the `qal` provide methods to transform SQL queries,
generating new queries from existing ones or formatting the query strings.
"""
from __future__ import annotations

import abc
import functools
import typing
from collections.abc import Collection, Iterable, Sequence
from typing import Generic, Optional

from postbound.qal import base, clauses, expressions as expr, predicates as preds
from postbound.util import collections as collection_utils


# TODO: add support for CTEs. This _should_ be pretty straightforward. Just remember to output the columns/tables at
# the appropriate places

def _stringify_clause(clause: clauses.BaseClause) -> str:
    """Provides a refined string for the given clause, to be used by the SqlQuery __str__ method."""
    return str(clause) + "\n" if isinstance(clause, clauses.Hint) else str(clause) + " "


def _collect_subqueries_in_expression(expression: expr.SqlExpression) -> set[SqlQuery]:
    """Provides all the subqueries that are contained in the expression."""
    if isinstance(expression, expr.SubqueryExpression):
        return {expression.query}
    return collection_utils.set_union(_collect_subqueries_in_expression(child_expr)
                                      for child_expr in expression.iterchildren())


def _collect_subqueries_in_table_source(table_source: clauses.TableSource) -> set[SqlQuery]:
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
    """Provides all the subqueries that are contained in the given clause."""
    if isinstance(clause, clauses.Hint) or isinstance(clause, clauses.Limit) or isinstance(clause, clauses.Explain):
        return set()

    if isinstance(clause, clauses.Select):
        return collection_utils.set_union(_collect_subqueries_in_expression(target.expression)
                                          for target in clause.targets)
    elif isinstance(clause, clauses.ImplicitFromClause):
        return set()
    elif isinstance(clause, clauses.From):
        return collection_utils.set_union(_collect_subqueries_in_table_source(src) for src in clause.contents)
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
    if isinstance(table_source, clauses.DirectTableSource):
        return {table_source.table}
    elif isinstance(table_source, clauses.SubqueryTableSource):
        return _collect_bound_tables(table_source.query.from_clause)
    elif isinstance(table_source, clauses.JoinTableSource):
        return _collect_bound_tables_from_source(table_source.source)

def _collect_bound_tables(from_clause: clauses.From) -> set[base.TableReference]:
    """Provides all tables that are bound in the given clause. See `bound_tables` in SqlQuery for details."""
    if isinstance(from_clause, clauses.ImplicitFromClause):
        return from_clause.tables()
    else:
        return collection_utils.set_union(_collect_bound_tables_from_source(src) for src in from_clause.contents)


FromClauseType = typing.TypeVar("FromClauseType", bound=clauses.From)


class SqlQuery(Generic[FromClauseType], abc.ABC):
    """Represents an arbitrary SQL query, providing convenient access to the different clauses in the query.

    At a basic level, PostBOUND differentiates between two types of queries:

    - implicit SQL queries specify all referenced tables in the FROM clause and the join predicates in the WHERE
    clause, e.g. SELECT * FROM R, S WHERE R.a = S.b AND R.c = 42
    - explicit SQL queries use the JOIN ON syntax to reference tables, e.g.
    SELECT * FROM R JOIN S ON R.a = S.b WHERE R.c = 42

    Currently, a mixed syntax for queries with both join types cannot be represented in PostBOUND (due to technical
    reasons). Implicit and explicit queries are represented through their own dedicated classes (ImplicitSqlQuery and
    ExplicitSqlQuery).
    """

    def __init__(self, *, select_clause: clauses.Select,
                 from_clause: Optional[FromClauseType] = None, where_clause: Optional[clauses.Where] = None,
                 groupby_clause: Optional[clauses.GroupBy] = None, having_clause: Optional[clauses.Having] = None,
                 orderby_clause: Optional[clauses.OrderBy] = None, limit_clause: Optional[clauses.Limit] = None,
                 hints: Optional[clauses.Hint] = None, explain: Optional[clauses.Explain] = None) -> None:
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
                               self._select_clause, self._from_clause, self._where_clause,
                               self._groupby_clause, self._having_clause,
                               self._orderby_clause, self._limit_clause))

    @property
    def select_clause(self) -> clauses.Select:
        """Get the SELECT clause of the query. Will always be set."""
        return self._select_clause

    @property
    def from_clause(self) -> Optional[FromClauseType]:
        """Get the FROM clause of the query, if specified."""
        return self._from_clause

    @property
    def where_clause(self) -> Optional[clauses.Where]:
        """Get the WHERE clause of the query, if specified."""
        return self._where_clause

    @property
    def groupby_clause(self) -> Optional[clauses.GroupBy]:
        """Get the GROUP BY clause of the query, if specified."""
        return self._groupby_clause

    @property
    def having_clause(self) -> Optional[clauses.Having]:
        """Get the HAVING clause of the query, if specified."""
        return self._having_clause

    @property
    def orderby_clause(self) -> Optional[clauses.OrderBy]:
        """Get the ORDER BY clause of the query, if specified."""
        return self._orderby_clause

    @property
    def limit_clause(self) -> Optional[clauses.Limit]:
        """Get the combined LIMIT and OFFSET clauses of the query, if specified."""
        return self._limit_clause

    @property
    def hints(self) -> Optional[clauses.Hint]:
        """Get the hint block of the query, if specified.

        Can consist of preparatory statements as well as query hints.
        """
        return self._hints

    @property
    def explain(self) -> Optional[clauses.Explain]:
        """Get the EXPLAIN block of the query, if specified."""
        return self._explain

    @abc.abstractmethod
    def is_implicit(self) -> bool:
        """Checks, whether this query is in implicit form."""
        raise NotImplementedError

    @abc.abstractmethod
    def is_explicit(self) -> bool:
        """Checks, whether this query is in explicit form."""
        raise NotImplementedError

    @functools.cache
    def tables(self) -> set[base.TableReference]:
        """Provides all tables that are referenced at any place in the query.

        This really means at any place, i.e. if a table only appears in the SELECT clause, it will still be returned
        here.
        """
        select_tables = self.select_clause.tables()
        from_tables = self.from_clause.tables() if self.from_clause else set()
        where_tables = self.where_clause.predicate.tables() if self.where_clause else set()
        return select_tables | from_tables | where_tables

    def columns(self) -> set[base.ColumnReference]:
        """Provides all columns that are referenced at any place in the query."""
        return collection_utils.set_union(clause.columns() for clause in self.clauses())

    @functools.cache
    def predicates(self) -> preds.QueryPredicates:
        """Provides all predicates in this query.

        This includes predicates in the FROM clause, as well as the WHERE clause.
        """
        predicate_handler = preds.DefaultPredicateHandler
        where_predicates = (predicate_handler(self.where_clause.predicate) if self.where_clause is not None
                            else predicate_handler.empty_predicate())
        from_predicate = self.from_clause.predicates()
        if from_predicate:
            return where_predicates.and_(from_predicate)
        else:
            return where_predicates

    def subqueries(self) -> Collection[SqlQuery]:
        """Provides all subqueries that are referenced in this query."""
        return collection_utils.set_union(_collect_subqueries(clause) for clause in self.clauses())

    def clauses(self) -> Sequence[clauses.BaseClause]:
        """Provides all the clauses that are present in this query.

        To distinguish the individual clauses, type checks are necessary.
        """
        all_clauses = [self.hints, self.explain,
                       self.select_clause, self.from_clause, self.where_clause,
                       self.groupby_clause, self.having_clause,
                       self.orderby_clause, self.limit_clause]
        return [clause for clause in all_clauses if clause is not None]

    def bound_tables(self) -> set[base.TableReference]:
        """Provides all tables that can be assigned to a physical/virtual table reference in this query.

        For example, the query SELECT * FROM R, S WHERE R.a = S.b has two bound tables: R and S.
        On the other hand, the query SELECT * FROM R WHERE R.a = S.b has only bound R, whereas S has to be bound in
        a surrounding query.
        """
        subquery_produced_tables = collection_utils.set_union(subquery.bound_tables()
                                                              for subquery in self.subqueries())
        own_produced_tables = _collect_bound_tables(self.from_clause)
        return own_produced_tables | subquery_produced_tables

    def unbound_tables(self) -> set[base.TableReference]:
        """Provides all tables that are referenced in this query but not bound.

        While `tables()` provides all tables that are referenced in this query in any way, `bound_tables` restricts
        these tables. This method provides the complementary set to `bound_tables` i.e.
        `tables = bound_tables ⊕ unbound_tables`.
        """
        return self.tables() - self.bound_tables()

    def is_ordered(self) -> bool:
        """Checks, whether this query produces its result tuples in order."""
        return self.orderby_clause is not None

    def is_dependent(self) -> bool:
        """Checks, whether all columns that are referenced in this query are provided by the tables from this query.

        In order for this check to work, all columns have to be bound to actual tables.
        """
        return not (self.tables() <= self.bound_tables())

    def iterexpressions(self) -> Iterable[expr.SqlExpression]:
        """Provides access to all directly contained expressions in this query.

        Nested expressions can be accessed from these expressions in a recursive manner (see the `SqlExpression`
        interface for details).
        """
        return collection_utils.flatten(clause.iterexpressions() for clause in self.clauses())

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        """Provides access to all column in this query.

        In contrast to the `columns` method, duplicates are returned multiple times, i.e. if a column is referenced `n`
        times in this query, it will also be returned `n` times by this method. Furthermore, the order in which
        columns are provided by the iterable matches the order in which they appear in this query.
        """
        return collection_utils.flatten(clause.itercolumns() for clause in self.clauses())

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self.clauses() == other.clauses()

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return "".join(_stringify_clause(clause) for clause in self.clauses()).rstrip() + ";"


class ImplicitSqlQuery(SqlQuery[clauses.ImplicitFromClause]):
    """Represents an implicit SQL query.

    A SQL query is implicit, if the FROM clause only lists the referenced tables and all joins are specified in the
    WHERE clause.

    For example, the following query is considered implicit:

    SELECT *
    FROM R, S, T
    WHERE R.a = S.b AND R.a = T.c

    As a special case, queries without a FROM clause or a WHERE clause are still considered as implicit.
    """

    def __init__(self, *, select_clause: clauses.Select,
                 from_clause: Optional[clauses.ImplicitFromClause] = None,
                 where_clause: Optional[clauses.Where] = None,
                 groupby_clause: Optional[clauses.GroupBy] = None,
                 having_clause: Optional[clauses.Having] = None,
                 orderby_clause: Optional[clauses.OrderBy] = None,
                 limit_clause: Optional[clauses.Limit] = None,
                 explain_clause: Optional[clauses.Explain] = None,
                 hints: Optional[clauses.Hint] = None) -> None:
        super().__init__(select_clause=select_clause, from_clause=from_clause, where_clause=where_clause,
                         groupby_clause=groupby_clause, having_clause=having_clause,
                         orderby_clause=orderby_clause, limit_clause=limit_clause,
                         explain=explain_clause, hints=hints)

    def is_implicit(self) -> bool:
        return True

    def is_explicit(self) -> bool:
        return False


class ExplicitSqlQuery(SqlQuery[clauses.ExplicitFromClause]):
    """Represents an explicit SQL query.

        A SQL query is explicit, if the FROM clause only lists all referenced tables along with the join predicates
        as JOIN ON statements.

        For example, the following query is considered explicit:

        SELECT *
        FROM R
            JOIN S ON R.a = S.b
            JOIN T ON R.a = T.c
        """

    def __init__(self, *, select_clause: clauses.Select,
                 from_clause: Optional[clauses.ExplicitFromClause] = None,
                 where_clause: Optional[clauses.Where] = None,
                 groupby_clause: Optional[clauses.GroupBy] = None,
                 having_clause: Optional[clauses.Having] = None,
                 orderby_clause: Optional[clauses.OrderBy] = None,
                 limit_clause: Optional[clauses.Limit] = None,
                 explain_clause: Optional[clauses.Explain] = None,
                 hints: Optional[clauses.Hint] = None) -> None:
        super().__init__(select_clause=select_clause, from_clause=from_clause, where_clause=where_clause,
                         groupby_clause=groupby_clause, having_clause=having_clause,
                         orderby_clause=orderby_clause, limit_clause=limit_clause,
                         explain=explain_clause, hints=hints)

    def is_implicit(self) -> bool:
        return False

    def is_explicit(self) -> bool:
        return True


class MixedSqlQuery(SqlQuery[clauses.From]):
    def __init__(self, *, select_clause: clauses.Select,
                 from_clause: Optional[clauses.From] = None,
                 where_clause: Optional[clauses.Where] = None,
                 groupby_clause: Optional[clauses.GroupBy] = None,
                 having_clause: Optional[clauses.Having] = None,
                 orderby_clause: Optional[clauses.OrderBy] = None,
                 limit_clause: Optional[clauses.Limit] = None,
                 explain_clause: Optional[clauses.Explain] = None,
                 hints: Optional[clauses.Hint] = None) -> None:
        if isinstance(from_clause, clauses.ExplicitFromClause) or isinstance(from_clause, clauses.ImplicitFromClause):
            raise ValueError("MixedSqlQuery cannot be combined with explicit/implicit FROM clause")
        super().__init__(select_clause=select_clause, from_clause=from_clause, where_clause=where_clause,
                         groupby_clause=groupby_clause, having_clause=having_clause,
                         orderby_clause=orderby_clause, limit_clause=limit_clause,
                         explain=explain_clause, hints=hints)

    def is_implicit(self) -> bool:
        return False

    def is_explicit(self) -> bool:
        return False


def build_query(query_clauses: Iterable[clauses.BaseClause]) -> SqlQuery:
    """Constructs an SQL query based on the given clauses.

    No validation is performed. If clauses appear multiple times, later clauses overwrite former ones. The specific
    type of query (i.e. implicit or explicit) is inferred from the clauses (i.e. occurrence of an implicit FROM clause
    enforces an ImplicitSqlQuery and vice-versa). The overwriting rules apply here as well.
    """
    build_implicit_query = True

    select_clause, from_clause, where_clause = None, None, None
    groupby_clause, having_clause = None, None
    orderby_clause, limit_clause = None, None
    explain_clause, hints_clause = None, None
    for clause in query_clauses:
        if not clause:
            continue

        if isinstance(clause, clauses.Select):
            select_clause = clause
        elif isinstance(clause, clauses.ImplicitFromClause):
            from_clause = clause
            build_implicit_query = True
        elif isinstance(clause, clauses.ExplicitFromClause):
            from_clause = clause
            build_implicit_query = False
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

    if build_implicit_query:
        return ImplicitSqlQuery(select_clause=select_clause, from_clause=from_clause, where_clause=where_clause,
                                groupby_clause=groupby_clause, having_clause=having_clause,
                                orderby_clause=orderby_clause, limit_clause=limit_clause,
                                hints=hints_clause, explain_clause=explain_clause)
    else:
        return ExplicitSqlQuery(select_clause=select_clause, from_clause=from_clause, where_clause=where_clause,
                                groupby_clause=groupby_clause, having_clause=having_clause,
                                orderby_clause=orderby_clause, limit_clause=limit_clause,
                                hints=hints_clause, explain_clause=explain_clause)
