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
import json
import random
from collections.abc import Generator
from typing import Literal, Optional

import pandas as pd

from .. import joingraph, jointree, physops, validation, planparams, stages
from ... import db, qal, util
from ...qal import parser, TableReference
from ...experiments import workloads


class CardinalityHintsGenerator(stages.ParameterGeneration, stages.CardinalityEstimator, abc.ABC):
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
    def calculate_estimate(self, query: qal.SqlQuery, tables: frozenset[TableReference]) -> Optional[int]:
        """Determines the cardinality estimate for a specific intermediate result.

        Ideally this is the only functionality-related method that needs to be implemented by developers using the cardinality
        generator.

        Parameters
        ----------
        query : qal.SqlQuery
            The query to optimize
        tables : frozenset[TableReference]
            The intermediate which should be estimated. The intermediate is described by its tables. It should be assumed that
            all filters and join predicates have been pushed down as far as possible.

        Returns
        -------
        Optional[int]
            The estimated cardinality if it could be computed, *None* otherwise.
        """
        raise NotImplementedError

    def generate_intermediates(self, query: qal.SqlQuery) -> Generator[frozenset[TableReference], None, None]:
        """Provides all intermediate results of a query.

        The inclusion of cross-products between arbitrary tables can be configured via the `allow_cross_products` attribute.

        Parameters
        ----------
        query : qal.SqlQuery
            The query for which to generate the intermediates

        Yields
        ------
        Generator[frozenset[TableReference], None, None]
            The intermediates

        Warnings
        --------
        The default implementation of this method does not work for queries that naturally contain cross products. If such a
        query is passed, no intermediates with tables from different partitions of the join graph are yielded.
        """
        for candidate_join in util.powerset(query.tables()):
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
            if estimate is not None:
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
            estimate = self.calculate_estimate(query, base_table.tables())
            if estimate is not None:
                parameterization.add_cardinality_hint(base_table.tables(), estimate)

        for join in join_order.join_sequence():
            estimate = self.calculate_estimate(query, join.tables())
            if estimate is not None:
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

    def calculate_estimate(self, query: qal.SqlQuery, tables: frozenset[TableReference]) -> int:
        partial_query = qal.transform.as_star_query(qal.transform.extract_query_fragment(query, tables))
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

    def calculate_estimate(self, query: qal.SqlQuery, tables: frozenset[TableReference]) -> int:
        partial_query = qal.transform.as_count_star_query(qal.transform.extract_query_fragment(query, tables))
        if partial_query in self._cardinality_cache:
            return self._cardinality_cache[partial_query]
        cardinality = self.database.execute_query(partial_query)
        self._cardinality_cache[partial_query] = cardinality
        return cardinality

    def reset_cache(self) -> None:
        self._cardinality_cache.clear()


def _parse_tables(tabs: str) -> set[TableReference]:
    """Utility to load tables from their JSON representation.

    Parameters
    ----------
    tabs : str
        The raw JSON data

    Returns
    -------
    set[TableReference]
        The corresponding tables
    """
    return {parser.load_table_json(t) for t in json.loads(tabs)}


class PreComputedCardinalities(CardinalityHintsGenerator):
    """Re-uses existing cardinalities from an external data source.

    The cardinalities have to be stored in a CSV file which follows a certain structure. Some details can be customized (e.g.
    column names). Most importantly, queries have to be identified via their labels. See parameters for details.

    Parameters
    ----------
    workload : workloads.Workload
        The workload which was used to calculate the cardinalities. This is required to determine the query label based on an
        input query. Each hint generator can only support a specific workload.
    lookup_table_path : str
        The file path to the CSV file containing the cardinalities.
    include_cross_products : bool, optional
        Whether cardinality estimates for arbitrary cross products are contained in the CSV file and hence can be used during
        estimation. By default this is disabled.
    default_cardinality : Optional[int], optional
        In case no cardinality estimate exists for a specific intermediate, a default cardinality can be used instead. In case
        no default value has been specified, an error would be raised. Notice that a ``None`` value unsets the default. If the
        client should handle this situation instead, another value (e.g. ``math.nan`` has to be used).
    label_col : str, optional
        The column in the CSV file that contains the query labels. Defaults to *label*.
    tables_col : str, optional
        The column in the CSV file that contains the (JSON serialized) tables that form the current intermediate result of the
        current query. Defaults to *tables*.
    cardinality_col : str, optional
        The column in the CSV file that contains the actual cardinalities. Defaults to *cardinality*.
    live_fallback : bool, optional
        Whether to fall back to a live database in case no cardinality estimate is found in the CSV file. This is off by
        default.
    error_on_missing_card : bool, optional
        If live fallback is disabled and we did not find a cardinality estimate for a specific intermediate, we will raise an
        error by default. If this is not desired and missing values can be handled by the client, this behavior can be disabled
        with this parameter.
    live_fallback_style : Literal["actual", "estimated"], optional
        In case the fallback is enabled, this customizes the calculation strategy. "actual" will calculate the true cardinality
        of the intermediate in question, whereas "estimated" (the default) will use the native optimizer to estimate the
        cardinality.
    live_db : Optional[db.Database], optional
        The database system that should be used in case of a live fallback. If omitted, the database system is inferred from
        the database pool.
    save_live_fallback_results : bool, optional
        Whether the cardinalities computed by the live fallback should be stored in the original file containing the lookup
        table. This is only used if live fallback is active and enabled by default.
    """
    def __init__(self, workload: workloads.Workload, lookup_table_path: str, *,
                 include_cross_products: bool = False, default_cardinality: Optional[int] = None,
                 label_col: str = "label", tables_col: str = "tables", cardinality_col: str = "cardinality",
                 live_fallback: bool = False, error_on_missing_card: bool = True,
                 live_db: Optional[db.Database] = None,
                 live_fallback_style: Literal["actual", "estimated"] = "estimated",
                 save_live_fallback_results: bool = True) -> None:
        super().__init__(include_cross_products)
        self._workload = workload
        self._label_col = label_col
        self._tables_col = tables_col
        self._card_col = cardinality_col
        self._default_card = default_cardinality
        self._lookup_df_path = lookup_table_path

        self._error_on_missing_card = error_on_missing_card
        self._live_db: Optional[db.Database] = None
        if live_fallback:
            self._live_db = db.DatabasePool.get_instance().current_database() if live_db is None else live_db
        else:
            self._live_db = None
        self._live_fallback_style = live_fallback_style
        self._save_life_fallback = save_live_fallback_results

        self._true_card_df = pd.read_csv(lookup_table_path, converters={tables_col: _parse_tables})

    def calculate_estimate(self, query: qal.SqlQuery, tables: frozenset[TableReference]) -> Optional[int]:
        label = self._workload.label_of(query)
        relevant_samples = self._true_card_df[self._true_card_df[self._label_col] == label]
        cardinality_sample = relevant_samples[relevant_samples[self._tables_col] == tables]

        tables_debug = "(" + ", ".join(tab.identifier() for tab in tables) + ")"
        n_samples = len(cardinality_sample)
        if n_samples == 1:
            cardinality = cardinality_sample.iloc[0][self._card_col]
            return int(cardinality)
        elif n_samples > 1:
            raise ValueError(f"{n_samples} samples found for join {tables_debug} in query {label}. Expected 1.")

        fallback_value = self._attempt_fallback_estimate(n_samples, query, tables)
        if fallback_value is None and self._error_on_missing_card:
            raise ValueError(f"No matching sample found for join {tables_debug} in query {label}")
        return fallback_value

    def describe(self) -> dict:
        return {"name": "pre-computed-cards", "location": self._lookup_df_path, "workload": self._workload.name}

    def _attempt_fallback_estimate(self, n_samples: int, query: qal.SqlQuery,
                                   tables: frozenset[TableReference]) -> Optional[int]:
        """Tries to infer the fallback value for a specific estimate, if this is necessary.

        The inference strategy applies the following rules:

        1. If exactly one sample was found, no fallback is necessary.
        2. If no sample was found, but we specified a static fallback value, this value is used.
        3. If a live fallback is available, the cardinality is calculated according to the `live_fallback_style`.
        4. Otherwise no fallback is possible.

        Parameters
        ----------
        n_samples : int
            The number of samples found for the current intermediate
        query : qal.SqlQuery
            The query for which the cardinality should be estimated
        tables : frozenset[TableReference]
            The joins that form the current intermediate

        Returns
        -------
        Optional[int]
            The fallback value if it could be inferred, otherwise *None*.
        """
        if n_samples == 1:
            # If we found exactly one sample, we do not need to fall back at all
            return None

        if self._default_card is not None:
            return self._default_card
        if self._live_db is None:
            return None

        query_fragment = qal.transform.extract_query_fragment(query, tables)
        if not query_fragment:
            return None

        if self._live_fallback_style == "actual":
            true_card_query = qal.transform.as_count_star_query(query_fragment)
            cardinality = self._live_db.execute_query(true_card_query)
        elif self._live_fallback_style == "estimated":
            cardinality = self._live_db.optimizer().cardinality_estimate(query_fragment)
        else:
            raise ValueError(f"Unknown fallback style: '{self._live_fallback_style}'")

        if self._save_life_fallback:
            self._dump_fallback_estimate(query, tables, cardinality)
        return cardinality

    def _dump_fallback_estimate(self, query: qal.SqlQuery, tables: frozenset[TableReference], cardinality: int) -> None:
        """Stores a newly computed cardinality estimate in the lookup table.

        Parameters
        ----------
        query : qal.SqlQuery
            The query for which the cardinality was estimated
        tables : frozenset[TableReference]
            The tables that form the current intermediate
        cardinality : int
            The computed cardinality
        """
        result_row = {}
        result_row[self._label_col] = [self._workload.label_of(query)]
        result_row[self._tables_col] = [util.to_json(tables)]

        if "query" in self._true_card_df.columns:
            result_row["query"] = [str(query)]
        if "query_fragment" in self._true_card_df.columns:
            result_row["query_fragment"] = [str(qal.transform.extract_query_fragment(query, tables))]

        result_row[self._card_col] = [cardinality]
        result_df = pd.DataFrame(result_row)

        self._true_card_df = pd.concat([self._true_card_df, result_df], ignore_index=True)
        self._true_card_df.to_csv(self._lookup_df_path, index=False)


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

    def calculate_estimate(self, query: qal.SqlQuery, tables: frozenset[TableReference]) -> Optional[int]:
        card_est = self.estimator.calculate_estimate(query, tables)
        if not card_est:
            return None
        if self.distortion_strategy == "fixed":
            distortion_factor = self.distortion_factor
        elif self.distortion_strategy == "random":
            distortion_factor = random.uniform(min(self.distortion_factor, 1.0), max(self.distortion_factor, 1.0))
        else:
            raise ValueError(f"Unknown distortion strategy: '{self.distortion_strategy}'")
        return card_est * distortion_factor


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
    def estimate_for(self, table: TableReference) -> int:
        """Calculates the cardinality for an arbitrary base table of the query.

        If the query is not filtered, this method should fall back to `estimate_total_rows`. Furthermore, the table can be
        assumed to not be part of any intermediate result, yet.

        Parameters
        ----------
        table : TableReference
            The table to estimate.

        Returns
        -------
        int
            The estimated number of rows
        """
        raise NotImplementedError

    @abc.abstractmethod
    def estimate_total_rows(self, table: TableReference) -> int:
        """Calculates an estimate of the number of rows in the table, ignoring all filter predicates.

        Parameters
        ----------
        table : TableReference
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
    database : db.Database
        The database system that should be used to obtain the estimates
    """

    def __init__(self, database: db.Database) -> None:
        super().__init__("native_optimizer")
        self.database = database
        self.query: qal.SqlQuery | None = None

    def setup_for_query(self, query: qal.SqlQuery) -> None:
        self.query = query

    def estimate_for(self, table: TableReference) -> int:
        filters = self.query.predicates().filters_for(table)
        if not filters:
            return self.estimate_total_rows(table)

        select_clause = qal.Select(qal.BaseProjection.star())
        from_clause = qal.ImplicitFromClause.create_for(table)
        where_clause = qal.Where(filters) if filters else None

        emulated_query = qal.ImplicitSqlQuery(select_clause=select_clause,
                                              from_clause=from_clause,
                                              where_clause=where_clause)
        return self.database.optimizer().cardinality_estimate(emulated_query)

    def estimate_total_rows(self, table: TableReference) -> int:
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

    def estimate_for(self, table: TableReference) -> int:
        select_clause = qal.Select(qal.BaseProjection.count_star())
        from_clause = qal.ImplicitFromClause.create_for(table)

        filters = self.query.predicates().filters_for(table)
        where_clause = qal.Where(filters) if filters else None

        emulated_query = qal.ImplicitSqlQuery(select_clause=select_clause,
                                              from_clause=from_clause,
                                              where_clause=where_clause)

        cache_enabled = self.database.statistics().cache_enabled  # this should be treated like a statistics query
        return self.database.execute_query(emulated_query, cache_enabled=cache_enabled)

    def estimate_total_rows(self, table: TableReference) -> int:
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
    def estimate_for(self, join_edge: qal.AbstractPredicate, join_graph: joingraph.JoinGraph) -> int:
        """Calculates the cardinality estimate for a specific join predicate, given the current state in the join graph.

        Parameters
        ----------
        join_edge : qal.AbstractPredicate
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
