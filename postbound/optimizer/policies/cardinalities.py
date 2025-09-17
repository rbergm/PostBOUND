"""Contains different policies to estimate base table and join cardinalities.

The policies can be divided into two different categories:

1. Policies that can be used within an optimization framework. These services take care of predicting cardinalities for
   individual base tables or joins. The `BaseTableCardinalityEstimator` and `JoinCardinalityEstimator` fall into this category.
2. End-to-end policies that predict all cardinalities for an entire query. These can be used to test different cardinality
   estimators within an existing optimizer of an actual database system. For example, this allows to conveniently test a novel
   machine learning-based optimizer for Postgres. The implementations of `CardinalityGenerator` are designed for this
   use-case.
"""

from __future__ import annotations

import abc

from ... import qal
from ..._core import Cardinality, TableReference
from ..._stages import EmptyPreCheck, OptimizationPreCheck
from ...db._db import Database
from ...optimizer._joingraph import JoinGraph


class BaseTableCardinalityEstimator(abc.ABC):
    """The base table estimator calculates cardinality estimates for filtered base tables.

    Implementations could for example use direct computation based on advanced statistics, sampling strategies or
    machine learning-based approaches.

    Each strategy provides dict-like access to the estimates: ``estimator[my_table]`` works as expected.

    Parameters
    ----------
    name : str
        The name of the actual estimation strategy.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abc.abstractmethod
    def setup_for_query(self, query: qal.SqlQuery) -> None:
        """Enables the estimator to prepare internal data structures.

        Parameters
        ----------
        query : qal.SqlQuery
            The query for which cardinalities should be estimated.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def estimate_for(self, table: TableReference) -> Cardinality:
        """Calculates the cardinality for an arbitrary base table of the query.

        If the query is not filtered, this method should fall back to `estimate_total_rows`. Furthermore, the table can be
        assumed to not be part of any intermediate result, yet.

        Parameters
        ----------
        table : TableReference
            The table to estimate.

        Returns
        -------
        Cardinality
            The estimated number of rows
        """
        raise NotImplementedError

    @abc.abstractmethod
    def estimate_total_rows(self, table: TableReference) -> Cardinality:
        """Calculates an estimate of the number of rows in the table, ignoring all filter predicates.

        Parameters
        ----------
        table : TableReference
            The table to estimate.

        Returns
        -------
        Cardinality
            The estimated number of rows
        """
        raise NotImplementedError

    @abc.abstractmethod
    def describe(self) -> dict:
        """Provides a JSON-serializable representation of the selected cardinality estimation strategy.

        Returns
        -------
        dict
            The representation
        """
        raise NotImplementedError

    def pre_check(self) -> OptimizationPreCheck:
        """Provides requirements that an input query has to satisfy in order for the estimator to work properly.

        Returns
        -------
        OptimizationPreCheck
            The requirements check
        """
        return EmptyPreCheck()

    def __getitem__(self, item: TableReference) -> int:
        return self.estimate_for(item)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"BaseTableCardinalityEstimator[{self.name}]"


class NativeCardinalityEstimator(BaseTableCardinalityEstimator):
    """Provides cardinality estimates for base tables using the optimizer of some database system.

    Parameters
    ----------
    database : Database
        The database system that should be used to obtain the estimates
    """

    def __init__(self, database: Database) -> None:
        super().__init__("native_optimizer")
        self.database = database
        self.query: qal.SqlQuery | None = None

    def setup_for_query(self, query: qal.SqlQuery) -> None:
        self.query = query

    def estimate_for(self, table: TableReference) -> Cardinality:
        filters = self.query.predicates().filters_for(table)
        if not filters:
            return self.estimate_total_rows(table)

        select_clause = qal.Select(qal.BaseProjection.star())
        from_clause = qal.ImplicitFromClause.create_for(table)
        where_clause = qal.Where(filters) if filters else None

        emulated_query = qal.ImplicitSqlQuery(
            select_clause=select_clause,
            from_clause=from_clause,
            where_clause=where_clause,
        )
        return self.database.optimizer().cardinality_estimate(emulated_query)

    def estimate_total_rows(self, table: TableReference) -> Cardinality:
        return Cardinality(self.database.statistics().total_rows(table, emulated=True))

    def describe(self) -> dict:
        return {"name": "native", "database": self.database.describe()}


class PreciseCardinalityEstimator(BaseTableCardinalityEstimator):
    """Obtains true cardinality counts by executing COUNT queries against a database system.

    This strategy provides a better reproducibility than the native estimates, but can be more compute-intense if caching is
    disabled.

    The executed COUNT queries account for all filters on the base table.

    Parameters
    ----------
    database : Database
        The database system that should be used to obtain the estimates
    """

    def __init__(self, database: Database) -> None:
        super().__init__("precise_estimates")
        self.database = database
        self.query: qal.SqlQuery | None = None

    def setup_for_query(self, query: qal.SqlQuery) -> None:
        self.query = query

    def estimate_for(self, table: TableReference) -> Cardinality:
        select_clause = qal.Select(qal.BaseProjection.count_star())
        from_clause = qal.ImplicitFromClause.create_for(table)

        filters = self.query.predicates().filters_for(table)
        where_clause = qal.Where(filters) if filters else None

        emulated_query = qal.ImplicitSqlQuery(
            select_clause=select_clause,
            from_clause=from_clause,
            where_clause=where_clause,
        )

        cache_enabled = (
            self.database.statistics().cache_enabled
        )  # this should be treated like a statistics query
        return Cardinality(
            self.database.execute_query(emulated_query, cache_enabled=cache_enabled)
        )

    def estimate_total_rows(self, table: TableReference) -> Cardinality:
        return Cardinality(self.database.statistics().total_rows(table, emulated=False))

    def describe(self) -> dict:
        return {"name": "precise", "database": self.database.describe()}


class JoinCardinalityEstimator(abc.ABC):
    """The join cardinality estimator calculates cardinality estimates for arbitrary n-ary joins.

    Implementations could for example use direct computation based on advanced statistics, sampling strategies or
    machine learning-based approaches.

    Parameters
    ----------
    name : str
        The name of the actual estimation strategy.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abc.abstractmethod
    def setup_for_query(self, query: qal.SqlQuery) -> None:
        """Enables the estimator to prepare internal data structures.

        Parameters
        ----------
        query : qal.SqlQuery
            The query for which cardinalities should be estimated.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def estimate_for(
        self, join_edge: qal.AbstractPredicate, join_graph: JoinGraph
    ) -> Cardinality:
        """Calculates the cardinality estimate for a specific join predicate, given the current state in the join graph.

        Parameters
        ----------
        join_edge : qal.AbstractPredicate
            The predicate that should be estimated.
        join_graph : JoinGraph
            A graph describing the currently joined relations as well as the join types (e.g. primary key/foreign key or n:m
            joins).

        Returns
        -------
        int
            The estimated join cardinality
        """
        raise NotImplementedError

    @abc.abstractmethod
    def describe(self) -> dict:
        """Provides a JSON-serializable representation of the selected cardinality estimation strategy.

        Returns
        -------
        dict
            The representation
        """
        raise NotImplementedError

    def pre_check(self) -> OptimizationPreCheck:
        """Provides requirements that an input query has to satisfy in order for the estimator to work properly.

        Returns
        -------
        OptimizationPreCheck
            The requirements check
        """
        return EmptyPreCheck()

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"JoinCardinalityEstimator[{self.name}]"
