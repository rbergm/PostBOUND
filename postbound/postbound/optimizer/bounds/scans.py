from __future__ import annotations

import abc

from postbound.db import db
from postbound.qal import base, clauses, predicates, qal


class BaseTableCardinalityEstimator(abc.ABC):

    def __init__(self, name: str) -> None:
        self.name = name

    @abc.abstractmethod
    def setup_for_query(self, query: qal.ImplicitSqlQuery) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def estimate_for(self, table: base.TableReference) -> int:
        """

        This method falls back to `estimate_total_rows` if the given table is not filtered.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def estimate_total_rows(self, table: base.TableReference) -> int:
        raise NotImplementedError

    def __getitem__(self, item: base.TableReference) -> int:
        return self.estimate_for(item)


class DBCardinalityEstimator(BaseTableCardinalityEstimator):
    def __init__(self, database: db.Database) -> None:
        super().__init__("Native optimizer")
        self.database = database
        self.query: qal.ImplicitSqlQuery | None = None

    def setup_for_query(self, query: qal.ImplicitSqlQuery) -> None:
        self.query = query

    def estimate_for(self, table: base.TableReference) -> int:
        filters = list(self.query.predicates().filters_for(table))
        if not filters:
            return self.estimate_total_rows(table)

        select_clause = clauses.Select(clauses.BaseProjection.star())
        from_clause = clauses.ImplicitFromClause(table)
        where_clause = clauses.Where(predicates.CompoundPredicate.create_and(filters))

        emulated_query = qal.ImplicitSqlQuery(select_clause=select_clause,
                                              from_clause=from_clause,
                                              where_clause=where_clause)
        return self.database.cardinality_estimate(emulated_query)

    def estimate_total_rows(self, table: base.TableReference) -> int:
        return self.database.statistics().total_rows(table, emulated=True)


class PreciseCardinalityEstimator(BaseTableCardinalityEstimator):
    def __init__(self, database: db.Database) -> None:
        super().__init__("Precise estimator")
        self.database = database
        self.query: qal.ImplicitSqlQuery | None = None

    def setup_for_query(self, query: qal.ImplicitSqlQuery) -> None:
        self.query = query

    def estimate_for(self, table: base.TableReference) -> int:
        select_clause = clauses.Select(clauses.BaseProjection.count_star())
        from_clause = clauses.ImplicitFromClause(table)

        filters = self.query.predicates().filters_for(table)
        where_clause = clauses.Where(predicates.CompoundPredicate.create_and(filters))

        emulated_query = qal.ImplicitSqlQuery(select_clause=select_clause,
                                              from_clause=from_clause,
                                              where_clause=where_clause)
        return self.database.execute_query(emulated_query)

    def estimate_total_rows(self, table: base.TableReference) -> int:
        return self.database.statistics().total_rows(table, emulated=False)
