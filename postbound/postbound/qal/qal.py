"""`qal` is the query abstraction layer.

It contains classes and methods to conveniently work with different parts of SQL queries.
"""

from __future__ import annotations

import abc
from collections.abc import Iterable

from postbound.qal import base, clauses, predicates as preds


class SqlQuery(abc.ABC):
    def __init__(self, *, select_clause: clauses.Select,
                 from_clause: clauses.From | None = None, where_clause: clauses.Where | None = None,
                 groupby_clause: clauses.GroupBy | None = None, having_clause: clauses.Having | None = None,
                 orderby_clause: clauses.OrderBy | None = None, limit_clause: clauses.Limit | None = None,
                 hints: clauses.Hint | None = None) -> None:
        self.select_clause = select_clause
        self.from_clause = from_clause
        self.where_clause = where_clause
        self.groupby_clause = groupby_clause
        self.having_clause = having_clause
        self.orderby_clause = orderby_clause
        self.limit_clause = limit_clause
        self.hints = hints

    @abc.abstractmethod
    def is_implicit(self) -> bool:
        raise NotImplementedError()

    def is_explicit(self) -> bool:
        return not self.is_implicit()

    def tables(self) -> Iterable[base.TableReference]:
        """Provides all tables that are referenced in the query."""
        from_tables = self.from_clause.tables() if self.from_clause else set()
        where_tables = self.where_clause.predicate.tables() if self.where_clause else set()
        return from_tables | where_tables

    def predicates(self) -> preds.QueryPredicates | None:
        """Provides all predicates in this query."""
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
        all_clauses = [self.hints,
                       self.select_clause, self.from_clause, self.where_clause,
                       self.orderby_clause, self.having_clause,
                       self.orderby_clause, self.limit_clause]
        return [clause for clause in all_clauses if clause is not None]

    def __hash__(self) -> int:
        return hash(tuple(self.clauses()))

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self.clauses() == other.clauses()

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return " ".join(str(clause) for clause in self.clauses())


class ImplicitSqlQuery(SqlQuery):
    def __init__(self, *, select_clause: clauses.Select,
                 from_clause: clauses.ImplicitFromClause | None = None,
                 where_clause: clauses.Where | None = None,
                 groupby_clause: clauses.GroupBy | None = None,
                 having_clause: clauses.Having | None = None,
                 orderby_clause: clauses.OrderBy | None = None,
                 limit_clause: clauses.Limit | None = None
                 ) -> None:
        super().__init__(select_clause=select_clause, from_clause=from_clause, where_clause=where_clause,
                         groupby_clause=groupby_clause, having_clause=having_clause,
                         orderby_clause=orderby_clause, limit_clause=limit_clause)

    def is_implicit(self) -> bool:
        return True


class ExplicitSqlQuery(SqlQuery):
    def __init__(self, *, select_clause: clauses.Select,
                 from_clause: clauses.ExplicitFromClause,
                 where_clause: clauses.Where | None = None,
                 groupby_clause: clauses.GroupBy | None = None,
                 having_clause: clauses.Having | None = None,
                 orderby_clause: clauses.OrderBy | None = None,
                 limit_clause: clauses.Limit | None = None
                 ) -> None:
        super().__init__(select_clause=select_clause, from_clause=from_clause, where_clause=where_clause,
                         groupby_clause=groupby_clause, having_clause=having_clause,
                         orderby_clause=orderby_clause, limit_clause=limit_clause)

    def is_implicit(self) -> bool:
        return False
