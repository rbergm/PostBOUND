"""`qal` is the query abstraction layer.

It contains classes and methods to conveniently work with different parts of SQL queries.
"""

from __future__ import annotations

import abc
from collections.abc import Iterable

from postbound.qal import base, joins, predicates as preds, projection as proj


class SqlQuery(abc.ABC):
    def __init__(self, mosp_data: dict) -> None:
        self._mosp_data = mosp_data

    @abc.abstractmethod
    def is_implicit(self) -> bool:
        raise NotImplementedError()

    def is_explicit(self) -> bool:
        return not self.is_implicit()

    @abc.abstractmethod
    def tables(self) -> Iterable[base.TableReference]:
        """Provides all tables that are referenced in the query."""
        raise NotImplementedError

    @abc.abstractmethod
    def predicates(self) -> preds.QueryPredicates | None:
        """Provides all predicates in this query."""
        raise NotImplementedError


class ImplicitSqlQuery(SqlQuery):
    def __init__(self, mosp_data: dict, *,
                 select_clause: proj.QueryProjection,
                 from_clause: list[base.TableReference] | None = None,
                 where_clause: preds.QueryPredicates | None = None) -> None:
        super().__init__(mosp_data)
        self.select_clause = select_clause
        self.from_clause = from_clause
        self.where_clause = where_clause

    def is_implicit(self) -> bool:
        return True

    def tables(self) -> Iterable[base.TableReference]:
        return self.from_clause

    def predicates(self) -> preds.QueryPredicates | None:
        return self.where_clause


class ExplicitSqlQuery(SqlQuery):
    def __init__(self, mosp_data: dict, *,
                 select_clause: proj.QueryProjection,
                 from_clause: tuple[base.TableReference, list[joins.Join]] | None = None,
                 where_clause: preds.QueryPredicates | None = None) -> None:
        super().__init__(mosp_data)
        self.select_clause = select_clause
        base_table, joined_tables = from_clause
        self.base_table: base.TableReference = base_table
        self.joined_tables: list[joins.Join] = joined_tables
        self.where_clause = where_clause

    def is_implicit(self) -> bool:
        return False

    def tables(self) -> Iterable[base.TableReference]:
        all_tables = [self.base_table]
        for join in self.joined_tables:
            all_tables.extend(join.tables())
        return all_tables

    def predicates(self) -> preds.QueryPredicates | None:
        all_predicates = self.where_clause if self.where_clause else preds.QueryPredicates.empty_predicate()
        for join in self.joined_tables:
            if not isinstance(join, joins.SubqueryJoin):
                continue
            subquery_join: joins.SubqueryJoin = join

            subquery_predicates = subquery_join.subquery.predicates()
            if subquery_predicates:
                all_predicates = all_predicates.and_(subquery_predicates)
            if subquery_join.join_condition:
                all_predicates = all_predicates.and_(subquery_join.join_condition)

        return all_predicates
