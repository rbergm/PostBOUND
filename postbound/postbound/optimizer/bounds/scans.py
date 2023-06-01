"""Basic interface to determine cardinality estimates or upper bounds for filtered base tables."""
from __future__ import annotations

import abc
from typing import Optional

from postbound.db import db
from postbound.qal import base, clauses, qal
from postbound.optimizer import validation


class BaseTableCardinalityEstimator(abc.ABC):
    """The base table estimator calculates cardinality estimates for filtered base tables

    This can be considered a meta-strategy that is used by the actual join enumerator or operator selector.

    Implementations could for example use direct computation based on advanced statistics, sampling strategies or
    machine learning-based approaches.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abc.abstractmethod
    def setup_for_query(self, query: qal.SqlQuery) -> None:
        """
        The setup functionality allows the estimator to prepare internal data structures such that the input query can
        be optimized.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def estimate_for(self, table: base.TableReference) -> int:
        """Calculates the estimate for the given (potentially filtered) base table.

        The table can be assumed to not be part of any intermediate result so far.

        At this point, the appropriate filter predicates could have determined during the query-specific estimator
        setup.

        This method falls back to `estimate_total_rows` if the given table is not filtered.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def estimate_total_rows(self, table: base.TableReference) -> int:
        """Calculates an estimate of the number of rows in the table, ignoring all filter predicates."""
        raise NotImplementedError

    @abc.abstractmethod
    def describe(self) -> dict:
        """Provides a representation of the selected cardinality estimation strategy."""
        raise NotImplementedError

    def pre_check(self) -> Optional[validation.OptimizationPreCheck]:
        """Provides requirements that an input query has to satisfy in order for the estimator to work properly."""
        return None

    def __getitem__(self, item: base.TableReference) -> int:
        return self.estimate_for(item)


class NativeCardinalityEstimator(BaseTableCardinalityEstimator):
    """Delegates the estimation process to the native optimizer of the database system."""

    def __init__(self, database: db.Database) -> None:
        super().__init__("Native optimizer")
        self.database = database
        self.query: qal.SqlQuery | None = None

    def setup_for_query(self, query: qal.SqlQuery) -> None:
        self.query = query

    def estimate_for(self, table: base.TableReference) -> int:
        filters = self.query.predicates().filters_for(table)
        if not filters:
            return self.estimate_total_rows(table)

        select_clause = clauses.Select(clauses.BaseProjection.star())
        from_clause = clauses.ImplicitFromClause.create_for(table)
        where_clause = clauses.Where(filters) if filters else None

        emulated_query = qal.ImplicitSqlQuery(select_clause=select_clause,
                                              from_clause=from_clause,
                                              where_clause=where_clause)
        return self.database.optimizer().cardinality_estimate(emulated_query)

    def estimate_total_rows(self, table: base.TableReference) -> int:
        return self.database.statistics().total_rows(table, emulated=True)

    def describe(self) -> dict:
        return {"name": "native"}


class PreciseCardinalityEstimator(BaseTableCardinalityEstimator):
    """Obtains the true cardinality counts by executing COUNT queries against the database system.

    This strategy provides a better reproducibility than the native estimates, but can be more compute-intense if
    caching is disabled.

    The executed COUNT queries account for all filters on the base table.
    """

    def __init__(self, database: db.Database) -> None:
        super().__init__("Precise estimator")
        self.database = database
        self.query: qal.SqlQuery | None = None

    def setup_for_query(self, query: qal.SqlQuery) -> None:
        self.query = query

    def estimate_for(self, table: base.TableReference) -> int:
        select_clause = clauses.Select(clauses.BaseProjection.count_star())
        from_clause = clauses.ImplicitFromClause.create_for(table)

        filters = self.query.predicates().filters_for(table)
        where_clause = clauses.Where(filters) if filters else None

        emulated_query = qal.ImplicitSqlQuery(select_clause=select_clause,
                                              from_clause=from_clause,
                                              where_clause=where_clause)

        cache_enabled = self.database.statistics().cache_enabled  # this should be treated like a statistics query
        return self.database.execute_query(emulated_query, cache_enabled=cache_enabled)

    def estimate_total_rows(self, table: base.TableReference) -> int:
        return self.database.statistics().total_rows(table, emulated=False)

    def describe(self) -> dict:
        return {"name": "precise"}
