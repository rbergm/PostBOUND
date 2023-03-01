"""`qal` is the query abstraction layer of PostBOUND. It offers interfaces to operate with SQL queries on a high level.

The most important type of `qal` is `SqlQuery` along with the classes that inherit from it. Notice that this package
focuses on the representation of queries. Other packages of the `qal` provide methods to transform SQL queries,
generating new queries from existing ones or formatting the query strings.
"""

from __future__ import annotations

import abc

from postbound.qal import base, clauses, predicates as preds


# TODO: add support for CTEs. This _should_ be pretty straightforward. Just remember to output the columns/tables at
# the appropriate places

class SqlQuery(abc.ABC):
    """Represents an arbitrary SQL query, providing convenient access to the different clauses in the query.

    At a basic level, PostBOUND differentiates between two types of queries:

    - implicit SQL queries specify all referenced tables in the FROM clause and the join predicates in the WHERE clause,
    e.g. SELECT * FROM R, S WHERE R.a = S.b AND R.c = 42
    - explicit SQL queries use the JOIN ON syntax to reference tables, e.g.
    SELECT * FROM R JOIN S ON R.a = S.b WHERE R.c = 42

    Currently, a mixed syntax for queries with both join types cannot be represented in PostBOUND (due to technical
    reasons). Implicit and explicit queries are represented through their own dedicated classes (ImplicitSqlQuery and
    ExplicitSqlQuery).
    """

    def __init__(self, *, select_clause: clauses.Select,
                 from_clause: clauses.From | None = None, where_clause: clauses.Where | None = None,
                 groupby_clause: clauses.GroupBy | None = None, having_clause: clauses.Having | None = None,
                 orderby_clause: clauses.OrderBy | None = None, limit_clause: clauses.Limit | None = None,
                 hints: clauses.Hint | None = None, explain: clauses.Explain | None = None) -> None:
        self.select_clause = select_clause
        self.from_clause = from_clause
        self.where_clause = where_clause
        self.groupby_clause = groupby_clause
        self.having_clause = having_clause
        self.orderby_clause = orderby_clause
        self.limit_clause = limit_clause
        self.hints = hints
        self.explain = explain

    @abc.abstractmethod
    def is_implicit(self) -> bool:
        """Checks, whether this query is in implicit form."""
        raise NotImplementedError()

    def is_explicit(self) -> bool:
        """Checks, whether this query is in explicit form."""
        return not self.is_implicit()

    def tables(self) -> set[base.TableReference]:
        """Provides all tables that are referenced at any place in the query.

        This really means at any place, i.e. if a table only appears in the SELECT clause, it will still be returned
        here.
        """
        select_tables = self.select_clause.tables()
        from_tables = self.from_clause.tables() if self.from_clause else set()
        where_tables = self.where_clause.predicate.tables() if self.where_clause else set()
        return select_tables | from_tables | where_tables

    def predicates(self) -> preds.QueryPredicates | None:
        """Provides all predicates in this query.

        This includes predicates in the FROM clause, as well as the WHERE clause.
        """
        where_predicates = (preds.QueryPredicates(self.where_clause.predicate) if self.where_clause is not None
                            else preds.QueryPredicates.empty_predicate())
        from_predicate = self.from_clause.predicates()
        if from_predicate:
            return where_predicates.and_(from_predicate)
        elif where_predicates.is_empty():
            return None
        else:
            return where_predicates

    def clauses(self) -> list:
        """Provides all the clauses that are present in this query.

        To distinguish the individual clauses, type checks are necessary.
        """
        all_clauses = [self.hints, self.explain,
                       self.select_clause, self.from_clause, self.where_clause,
                       self.orderby_clause, self.having_clause,
                       self.orderby_clause, self.limit_clause]
        return [clause for clause in all_clauses if clause is not None]

    def is_ordered(self) -> bool:
        """Checks, whether this query produces its result tuples in order."""
        return self.orderby_clause is not None

    def is_dependent(self) -> bool:
        """Checks, whether all columns that are referenced in this query are provided by the tables from this query.

        In order for this check to work, all columns have to be bound to actual tables.
        """
        return any(not tab.full_name and not tab.virtual for tab in self.tables())

    def __hash__(self) -> int:
        return hash(tuple(self.clauses()))

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self.clauses() == other.clauses()

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return " ".join(str(clause) for clause in self.clauses()) + ";"


class ImplicitSqlQuery(SqlQuery):
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
                 from_clause: clauses.ImplicitFromClause | None = None,
                 where_clause: clauses.Where | None = None,
                 groupby_clause: clauses.GroupBy | None = None,
                 having_clause: clauses.Having | None = None,
                 orderby_clause: clauses.OrderBy | None = None,
                 limit_clause: clauses.Limit | None = None,
                 explain_clause: clauses.Explain | None = None) -> None:
        super().__init__(select_clause=select_clause, from_clause=from_clause, where_clause=where_clause,
                         groupby_clause=groupby_clause, having_clause=having_clause,
                         orderby_clause=orderby_clause, limit_clause=limit_clause,
                         explain=explain_clause)

    def is_implicit(self) -> bool:
        return True


class ExplicitSqlQuery(SqlQuery):
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
                 from_clause: clauses.ExplicitFromClause,
                 where_clause: clauses.Where | None = None,
                 groupby_clause: clauses.GroupBy | None = None,
                 having_clause: clauses.Having | None = None,
                 orderby_clause: clauses.OrderBy | None = None,
                 limit_clause: clauses.Limit | None = None,
                 explain_clause: clauses.Explain | None = None) -> None:
        super().__init__(select_clause=select_clause, from_clause=from_clause, where_clause=where_clause,
                         groupby_clause=groupby_clause, having_clause=having_clause,
                         orderby_clause=orderby_clause, limit_clause=limit_clause,
                         explain=explain_clause)

    def is_implicit(self) -> bool:
        return False
