"""Contains different policies to estimate base table and join cardinalities.

The policies can be divided into two different categories:

1. Policies that can be used within an optimization framework. These services take care of predicting cardinalities for
   individual base tables or joins. The `BaseTableCardinalityEstimator` and `JoinCardinalityEstimator` fall into this category.
2. End-to-end policies that predict all cardinalities for an entire query. These can be used to test different cardinality
   estimators within an existing optimizer of an actual database system. For example, this allows to conveniently test a novel
   machine learning-based optimizer for Postgres. The `CardinalityHintsGenerator` is designed for this use-case.
"""
from __future__ import annotations

import abc
import random
from collections.abc import Generator
from typing import Literal, Optional

from postbound.db import db
from postbound.qal import base, clauses, predicates, qal, transform
from postbound.optimizer import joingraph, jointree, physops, validation, planparams, stages
from postbound.util import collections as collection_utils


class CardinalityHintsGenerator(stages.ParameterGeneration, abc.ABC):
    """End-to-end cardinality estimator.

    Implementations of this service calculate cardinalities for all relevant intermediate results of a query. In turn, these
    cardinalities can be used by the optimizer of an actual database system to overwrite the native estimates.

    The default implementations of all methods either request cardinality estimates for all possible intermediate results (in
    the `estimate_cardinalities` method), or for exactly those intermediates that are defined in a specific join order (in the
    `generate_plan_parameters` method that implements the protocol of the `ParameterGeneration` class). Therefore, developers
    working on their own cardinality estimation algorithm only need to implement the `calculate_estimate` method. All related
    processes are provided by the generator with reasonable default strategies.

    However, special care is required when considering cross products: depending on the setting intermediates can either allow
    cross products at all stages (by passing ``allow_cross_products=True`` during instantiation), or to disallow them entirely.
    Therefore, the `calculate_estimate` method should act accordingly. Implementations of this class should pass the
    appropriate parameter value to the super *__init__* method. If they support both scenarios, the parameter can also be
    exposed to the client.

    Notice that this strategies fails for queries which contain actual cross products. That is why the `pre_check` only
    accepts queries without cross products. Developers should overwrite the relevant methods as needed. See *Warnings* for more
    details.

    Parameters
    ----------
    allow_cross_products : bool
        Whether the default intermediate generation is allowed to emit cross products between arbitrary tables in the input
        query.

    Warnings
    --------
    The default implementation of this service does not work well for queries that naturally contain cross products. If you
    intend to use if for workloads that contain cross products, you should overwrite the `generate_intermediates` method to
    produce exactly those (partial) joins that you want to allow.
    """
    def __init__(self, allow_cross_products: bool) -> None:
        super().__init__()
        self.allow_cross_products = allow_cross_products

    @abc.abstractmethod
    def calculate_estimate(self, query: qal.SqlQuery, tables: frozenset[base.TableReference]) -> int:
        """Determines the cardinality estimate for a specific intermediate result.

        Ideally this is the only functionality-related method that needs to be implemented by developers using the cardinality
        generator.

        Parameters
        ----------
        query : qal.SqlQuery
            The query to optimize
        tables : frozenset[base.TableReference]
            The intermediate which should be estimated. The intermediate is described by its tables. It should be assumed that
            all filters and join predicates have been pushed down as far as possible.

        Returns
        -------
        int
            The estimated cardinality
        """
        raise NotImplementedError

    def generate_intermediates(self, query: qal.SqlQuery) -> Generator[frozenset[base.TableReference], None, None]:
        """Provides all intermediate results of a query.

        The inclusion of cross-products between arbitrary tables can be configured via the `allow_cross_products` attribute.

        Parameters
        ----------
        query : qal.SqlQuery
            The query for which to generate the intermediates

        Yields
        ------
        Generator[frozenset[base.TableReference], None, None]
            The intermediates

        Warnings
        --------
        The default implementation of this method does not work for queries that naturally contain cross products. If such a
        query is passed, no intermediates with tables from different partitions of the join graph are yielded.
        """
        for candidate_join in collection_utils.powerset(query.tables()):
            if not candidate_join:  # skip empty set (which is an artefact of the powerset method)
                continue
            if not self.allow_cross_products and not query.predicates().joins_tables(candidate_join):
                continue
            yield frozenset(candidate_join)

    def estimate_cardinalities(self, query: qal.SqlQuery) -> planparams.PlanParameterization:
        """Produces all cardinality estimates for a specific query.

        The default implementation of this method delegates the actual estimation to the `calculate_estimate` method. It is
        called for each intermediate produced by `generate_intermediates`.

        Parameters
        ----------
        query : qal.SqlQuery
            The query to optimize

        Returns
        ------
        planparams.PlanParameterization
            A parameterization containing cardinality hints for all intermediates. Other attributes of the parameterization are
            not modified.
        """
        parameterization = planparams.PlanParameterization()
        for join in self.generate_intermediates(query):
            estimate = self.calculate_estimate(query, join)
            parameterization.add_cardinality_hint(join, estimate)
        return parameterization

    def generate_plan_parameters(self, query: qal.SqlQuery,
                                 join_order: Optional[jointree.LogicalJoinTree | jointree.PhysicalQueryPlan],
                                 operator_assignment: Optional[physops.PhysicalOperatorAssignment]
                                 ) -> planparams.PlanParameterization:
        if join_order is None:
            return self.estimate_cardinalities(query)

        parameterization = planparams.PlanParameterization()
        for base_table in join_order.table_sequence():
            estimate = self.calculate_estimate(base_table.tables())
            parameterization.add_cardinality_hint(base_table.tables(), estimate)

        for join in join_order.join_sequence():
            estimate = self.calculate_estimate(join.tables())
            parameterization.add_cardinality_hint(join.tables(), estimate)

        return parameterization

    def pre_check(self) -> validation.OptimizationPreCheck:
        return validation.CrossProductPreCheck()


class NativeCardinalityHintGenerator(CardinalityHintsGenerator):
    """Cardinality estimator that delegates all estiamtes to an actual database system.

    Parameters
    ----------
    database : Optional[db.Database], optional
        The database system exposing an optimizer along with its cardinality estimates. If omitted, the database system is
        inferred from the database pool.
    allow_cross_products : bool, optional
        Whether cardinality estimates for arbitrary cross products should be included.
    """
    def __init__(self, database: Optional[db.Database] = None, *, allow_cross_products: bool = False) -> None:
        super().__init__(allow_cross_products)
        self.database = database if database is not None else db.DatabasePool.get_instance().current_database()

    def describe(self) -> dict:
        return {"name": "native-cards", "database": self.database.describe()}

    def calculate_estimate(self, query: qal.SqlQuery, tables: frozenset[base.TableReference]) -> int:
        partial_query = transform.as_star_query(transform.extract_query_fragment(query, tables))
        return self.database.optimizer().cardinality_estimate(partial_query)


class PreciseCardinalityHintGenerator(CardinalityHintsGenerator):
    """Cardinality "estimator" that calculates exact cardinalities.

    These cardinalities are determined by actually executing the intermediate query plan and counting the number of result
    tuples. To speed up this potentially very costly computation, the estimator can store already calculated cardinalities in
    an intermediate cache. Notice that this cache is different from the query cache provided by the `Database` interface. The
    reason for this distinction is simple: the query result cache assumes static databases. If it connects to the same logical
    database at two different points in time (potentially after a data shift), the cached results will be out-of-date. On the
    other hand, the cardinality cache is transient and local to each estimator. Therefore, it will always calculate the current
    results, even when a data shift is simulated. Even when the same estimator is used while simulating a data shift, the cache
    can be reset manually without impacting caching of all other queries.

    Parameters
    ----------
    database : Optional[db.Database], optional
        The database for which the estimates should be calculated. If omitted, the database system is inferred from the
        database pool.
    enable_cache : bool, optional
        Whether cardinalities of intermediates should be cached *for the lifetime of the estimator object*. Defaults to
        *False*.
    allow_cross_products : bool, optional
        Whether cardinality estimates for arbitrary cross products should be included.
    """

    def __init__(self, database: Optional[db.Database] = None, *, enable_cache: bool = False,
                 allow_cross_products: bool = False) -> None:
        super().__init__(allow_cross_products)
        self.database = database if database is not None else db.DatabasePool.get_instance().current_database()
        self.cache_enabled = enable_cache
        self._cardinality_cache: dict[qal.SqlQuery, int] = {}

    def describe(self) -> dict:
        return {"name": "true-cards", "database": self.database.describe()}

    def calculate_estimate(self, query: qal.SqlQuery, tables: frozenset[base.TableReference]) -> int:
        partial_query = transform.as_count_star_query(transform.extract_query_fragment(query, tables))
        if partial_query in self._cardinality_cache:
            return self._cardinality_cache[partial_query]
        cardinality = self.database.execute_query(partial_query)
        self._cardinality_cache[partial_query] = cardinality
        return cardinality

    def reset_cache(self) -> None:
        self._cardinality_cache.clear()


class CardinalityDistortion(CardinalityHintsGenerator):
    """Decorator to simulate errors during cardinality estimation.

    The distortion service uses cardinality estimates produced by an actual estimator and mofifies its estimations to simulate
    the effect of deviations and misestimates.

    Behavior regarding cross products is inferred based on the behavior of the actual estimator.

    Parameters
    ----------
    estimator : CardinalityHintsGenerator
        The actual estimator that calculates the "correct" cardinalities.
    distortion_factor : float
        How much the cardinalities are allowed to deviate from the original estimations. Values > 1 simulate overestimation
        whereas values < 1 simulate underestimation. For example, a distortion factor of 0.5 means that the final estimates can
        deviate at most half of the original cardinalities, pr a factor of 1.3 allows an overestimation of up to 30%.
    distortion_strategy : Literal["fixed", "random"], optional
        How the estimation errors should be calculated. The default *fixed* strategy always applies the exact distrotion factor
        to the cardinalities. For example, an estimate of 1000 tuples would always become 1300 tuples with a distrotion factor
        of 1.3. On the other hand the *random* strategy allows any error between 1 and the desired factor and selects the
        specific distortion at random. For example, an estimate of 100 could become any cardinality between 50 and 100 tuples
        with a distortion factor of 0.5.
    """
    def __init__(self, estimator: CardinalityHintsGenerator, distortion_factor: float, *,
                 distortion_strategy: Literal["fixed", "random"] = "fixed") -> None:
        super().__init__(estimator.allow_cross_products)
        self.estimator = estimator
        self.distortion_factor = distortion_factor
        self.distortion_strategy = distortion_strategy

    def describe(self) -> dict:
        return {"name": "cardinality-distortion", "estimator": "distortion",
                "distortion_factor": self.distortion_factor, "distortion_strategy": self.distortion_strategy}

    def calculate_estimate(self, query: qal.SqlQuery, tables: frozenset[base.TableReference]) -> int:
        card_est = self.estimator.calculate_estimate(tables)
        if self.distortion_strategy == "fixed":
            card_est *= self.distortion_factor
        elif self.distortion_strategy == "random":
            distortion_factor = random.uniform(min(self.distortion_factor, 1.0), max(self.distortion_factor, 1.0))
            card_est *= distortion_factor
        return card_est


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
    def estimate_for(self, table: base.TableReference) -> int:
        """Calculates the cardinality for an arbitrary base table of the query.

        If the query is not filtered, this method should fall back to `estimate_total_rows`. Furthermore, the table can be
        assumed to not be part of any intermediate result, yet.

        Parameters
        ----------
        table : base.TableReference
            The table to estimate.

        Returns
        -------
        int
            The estimated number of rows
        """
        raise NotImplementedError

    @abc.abstractmethod
    def estimate_total_rows(self, table: base.TableReference) -> int:
        """Calculates an estimate of the number of rows in the table, ignoring all filter predicates.

        Parameters
        ----------
        table : base.TableReference
            The table to estimate.

        Returns
        -------
        int
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

    def pre_check(self) -> validation.OptimizationPreCheck:
        """Provides requirements that an input query has to satisfy in order for the estimator to work properly.

        Returns
        -------
        validation.OptimizationPreCheck
            The requirements check
        """
        return validation.EmptyPreCheck()

    def __getitem__(self, item: base.TableReference) -> int:
        return self.estimate_for(item)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"BaseTableCardinalityEstimator[{self.name}]"


class NativeCardinalityEstimator(BaseTableCardinalityEstimator):
    """Provides cardinality estimates for base tables using the optimizer of some database system.

    Parameters
    ----------
    database : db.Database
        The database system that should be used to obtain the estimates
    """

    def __init__(self, database: db.Database) -> None:
        super().__init__("native_optimizer")
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
        return {"name": "native", "database": self.database.describe()}


class PreciseCardinalityEstimator(BaseTableCardinalityEstimator):
    """Obtains true cardinality counts by executing COUNT queries against a database system.

    This strategy provides a better reproducibility than the native estimates, but can be more compute-intense if caching is
    disabled.

    The executed COUNT queries account for all filters on the base table.

    Parameters
    ----------
    database : db.Database
        The database system that should be used to obtain the estimates
    """

    def __init__(self, database: db.Database) -> None:
        super().__init__("precise_estimates")
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
    def estimate_for(self, join_edge: predicates.AbstractPredicate, join_graph: joingraph.JoinGraph) -> int:
        """Calculates the cardinality estimate for a specific join predicate, given the current state in the join graph.

        Parameters
        ----------
        join_edge : predicates.AbstractPredicate
            The predicate that should be estimated.
        join_graph : joingraph.JoinGraph
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

    def pre_check(self) -> validation.OptimizationPreCheck:
        """Provides requirements that an input query has to satisfy in order for the estimator to work properly.

        Returns
        -------
        validation.OptimizationPreCheck
            The requirements check
        """
        return validation.EmptyPreCheck()

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"JoinCardinalityEstimator[{self.name}]"
