from __future__ import annotations

import abc
import json
import math
import random
from collections.abc import Generator, Iterable
from typing import Literal, Optional

import pandas as pd

from .. import util
from .._core import Cardinality, TableReference
from .._hints import PhysicalOperatorAssignment
from .._jointree import JoinTree
from .._stages import (
    CardinalityEstimator,
    OptimizationPreCheck,
    ParameterGeneration,
    PlanParameterization,
)
from ..db._db import Database, DatabasePool
from ..experiments.workloads import Workload
from ..qal import parser, transform
from ..qal._qal import SqlQuery
from . import validation


class CardinalityGenerator(ParameterGeneration, CardinalityEstimator, abc.ABC):
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
    def calculate_estimate(
        self, query: SqlQuery, tables: TableReference | Iterable[TableReference]
    ) -> Cardinality:
        """Determines the cardinality estimate for a specific intermediate result.

        Ideally this is the only functionality-related method that needs to be implemented by developers using the cardinality
        generator.

        Parameters
        ----------
        query : SqlQuery
            The query to optimize
        tables : TableReference | Iterable[TableReference]
            The intermediate which should be estimated. The intermediate is described by its tables. It should be assumed that
            all filters and join predicates have been pushed down as far as possible.

        Returns
        -------
        Cardinality
            The estimated cardinality if it could be computed, *NaN* otherwise.
        """
        raise NotImplementedError

    def generate_intermediates(
        self, query: SqlQuery
    ) -> Generator[frozenset[TableReference], None, None]:
        """Provides all intermediate results of a query.

        The inclusion of cross-products between arbitrary tables can be configured via the `allow_cross_products` attribute.

        Parameters
        ----------
        query : SqlQuery
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
            if (
                not candidate_join
            ):  # skip empty set (which is an artefact of the powerset method)
                continue
            if not self.allow_cross_products and not query.predicates().joins_tables(
                candidate_join
            ):
                continue
            yield frozenset(candidate_join)

    def estimate_cardinalities(self, query: SqlQuery) -> PlanParameterization:
        """Produces all cardinality estimates for a specific query.

        The default implementation of this method delegates the actual estimation to the `calculate_estimate` method. It is
        called for each intermediate produced by `generate_intermediates`.

        Parameters
        ----------
        query : SqlQuery
            The query to optimize

        Returns
        ------
        PlanParameterization
            A parameterization containing cardinality hints for all intermediates. Other attributes of the parameterization are
            not modified.
        """
        parameterization = PlanParameterization()
        for join in self.generate_intermediates(query):
            estimate = self.calculate_estimate(query, join)
            if not math.isnan(estimate):
                parameterization.add_cardinality(join, estimate)
        return parameterization

    def generate_plan_parameters(
        self,
        query: SqlQuery,
        join_order: Optional[JoinTree],
        operator_assignment: Optional[PhysicalOperatorAssignment],
    ) -> PlanParameterization:
        if join_order is None:
            return self.estimate_cardinalities(query)

        parameterization = PlanParameterization()
        for intermediate in join_order.iternodes():
            estimate = self.calculate_estimate(query, intermediate.tables())
            if not math.isnan(estimate):
                parameterization.add_cardinality(intermediate.tables(), estimate)

        return parameterization

    def pre_check(self) -> OptimizationPreCheck:
        return validation.CrossProductPreCheck()


class NativeCardinalityHintGenerator(CardinalityGenerator):
    """Cardinality estimator that delegates all estiamtes to an actual database system.

    Parameters
    ----------
    database : Optional[Database], optional
        The database system exposing an optimizer along with its cardinality estimates. If omitted, the database system is
        inferred from the database pool.
    allow_cross_products : bool, optional
        Whether cardinality estimates for arbitrary cross products should be included.
    """

    def __init__(
        self,
        database: Optional[Database] = None,
        *,
        allow_cross_products: bool = False,
    ) -> None:
        super().__init__(allow_cross_products)
        self.database = (
            database
            if database is not None
            else DatabasePool.get_instance().current_database()
        )

    def describe(self) -> dict:
        return {"name": "native-cards", "database": self.database.describe()}

    def calculate_estimate(
        self, query: SqlQuery, tables: TableReference | Iterable[TableReference]
    ) -> Cardinality:
        tables = util.enlist(tables)
        partial_query = transform.as_star_query(
            transform.extract_query_fragment(query, tables)
        )
        return self.database.optimizer().cardinality_estimate(partial_query)


class PreciseCardinalityHintGenerator(CardinalityGenerator):
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
    database : Optional[Database], optional
        The database for which the estimates should be calculated. If omitted, the database system is inferred from the
        database pool.
    enable_cache : bool, optional
        Whether cardinalities of intermediates should be cached *for the lifetime of the estimator object*. Defaults to
        *False*.
    allow_cross_products : bool, optional
        Whether cardinality estimates for arbitrary cross products should be included.
    """

    def __init__(
        self,
        database: Optional[Database] = None,
        *,
        enable_cache: bool = False,
        allow_cross_products: bool = False,
    ) -> None:
        super().__init__(allow_cross_products)
        self.database = (
            database
            if database is not None
            else DatabasePool.get_instance().current_database()
        )
        self.cache_enabled = enable_cache
        self._cardinality_cache: dict[SqlQuery, int] = {}

    def describe(self) -> dict:
        return {"name": "true-cards", "database": self.database.describe()}

    def calculate_estimate(
        self, query: SqlQuery, tables: TableReference | Iterable[TableReference]
    ) -> Cardinality:
        tables = util.enlist(tables)
        partial_query = transform.as_count_star_query(
            transform.extract_query_fragment(query, tables)
        )
        if partial_query in self._cardinality_cache:
            return self._cardinality_cache[partial_query]
        cardinality = Cardinality(self.database.execute_query(partial_query))
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


class PreComputedCardinalities(CardinalityGenerator):
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
    default_cardinality : Optional[Cardinality], optional
        In case no cardinality estimate exists for a specific intermediate, a default cardinality can be used instead. In case
        no default value has been specified, an error would be raised. Notice that a ``None`` value unsets the default. If the
        client should handle this situation instead, another value (e.g. ``Cardinality.unknown()`` has to be used).
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
    live_db : Optional[Database], optional
        The database system that should be used in case of a live fallback. If omitted, the database system is inferred from
        the database pool.
    save_live_fallback_results : bool, optional
        Whether the cardinalities computed by the live fallback should be stored in the original file containing the lookup
        table. This is only used if live fallback is active and enabled by default.
    """

    def __init__(
        self,
        workload: Workload,
        lookup_table_path: str,
        *,
        include_cross_products: bool = False,
        default_cardinality: Optional[Cardinality] = None,
        label_col: str = "label",
        tables_col: str = "tables",
        cardinality_col: str = "cardinality",
        live_fallback: bool = False,
        error_on_missing_card: bool = True,
        live_db: Optional[Database] = None,
        live_fallback_style: Literal["actual", "estimated"] = "estimated",
        save_live_fallback_results: bool = True,
    ) -> None:
        super().__init__(include_cross_products)
        self._workload = workload
        self._label_col = label_col
        self._tables_col = tables_col
        self._card_col = cardinality_col
        self._default_card = default_cardinality
        self._lookup_df_path = lookup_table_path

        self._error_on_missing_card = error_on_missing_card
        self._live_db: Optional[Database] = None
        if live_fallback:
            self._live_db = (
                DatabasePool.get_instance().current_database()
                if live_db is None
                else live_db
            )
        else:
            self._live_db = None
        self._live_fallback_style = live_fallback_style
        self._save_life_fallback = save_live_fallback_results

        self._true_card_df = pd.read_csv(
            lookup_table_path, converters={tables_col: _parse_tables}
        )

    def calculate_estimate(
        self, query: SqlQuery, tables: TableReference | Iterable[TableReference]
    ) -> Cardinality:
        tables = util.enlist(tables)
        label = self._workload.label_of(query)
        relevant_samples = self._true_card_df[
            self._true_card_df[self._label_col] == label
        ]
        cardinality_sample = relevant_samples[
            relevant_samples[self._tables_col] == tables
        ]

        tables_debug = "(" + ", ".join(tab.identifier() for tab in tables) + ")"
        n_samples = len(cardinality_sample)
        if n_samples == 1:
            cardinality = Cardinality(cardinality_sample.iloc[0][self._card_col])
            return cardinality
        elif n_samples > 1:
            raise ValueError(
                f"{n_samples} samples found for join {tables_debug} in query {label}. Expected 1."
            )

        fallback_value = self._attempt_fallback_estimate(n_samples, query, tables)
        if fallback_value is None and self._error_on_missing_card:
            raise ValueError(
                f"No matching sample found for join {tables_debug} in query {label}"
            )
        return fallback_value

    def describe(self) -> dict:
        return {
            "name": "pre-computed-cards",
            "location": self._lookup_df_path,
            "workload": self._workload.name,
        }

    def _attempt_fallback_estimate(
        self, n_samples: int, query: SqlQuery, tables: frozenset[TableReference]
    ) -> Cardinality:
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
        query : SqlQuery
            The query for which the cardinality should be estimated
        tables : frozenset[TableReference]
            The joins that form the current intermediate

        Returns
        -------
        Cardinality
            The fallback value if it could be inferred, otherwise *NaN*.
        """
        if n_samples == 1:
            # If we found exactly one sample, we did not need to fall back at all
            return Cardinality.unknown()

        if self._default_card is not None:
            return self._default_card
        if self._live_db is None:
            return Cardinality.unknown()

        query_fragment = transform.extract_query_fragment(query, tables)
        if not query_fragment:
            return Cardinality.unknown()

        if self._live_fallback_style == "actual":
            true_card_query = transform.as_count_star_query(query_fragment)
            cardinality = Cardinality(self._live_db.execute_query(true_card_query))
        elif self._live_fallback_style == "estimated":
            cardinality = self._live_db.optimizer().cardinality_estimate(query_fragment)
        else:
            raise ValueError(f"Unknown fallback style: '{self._live_fallback_style}'")

        if self._save_life_fallback:
            self._dump_fallback_estimate(query, tables, cardinality)
        return cardinality

    def _dump_fallback_estimate(
        self,
        query: SqlQuery,
        tables: frozenset[TableReference],
        cardinality: Cardinality,
    ) -> None:
        """Stores a newly computed cardinality estimate in the lookup table.

        Parameters
        ----------
        query : SqlQuery
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
            result_row["query_fragment"] = [
                str(transform.extract_query_fragment(query, tables))
            ]

        result_row[self._card_col] = [cardinality]
        result_df = pd.DataFrame(result_row)

        self._true_card_df = pd.concat(
            [self._true_card_df, result_df], ignore_index=True
        )
        self._true_card_df.to_csv(self._lookup_df_path, index=False)


class CardinalityDistortion(CardinalityGenerator):
    """Decorator to simulate errors during cardinality estimation.

    The distortion service uses cardinality estimates produced by an actual estimator and mofifies its estimations to simulate
    the effect of deviations and misestimates.

    Behavior regarding cross products is inferred based on the behavior of the actual estimator.

    Parameters
    ----------
    estimator : CardinalityGenerator
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

    def __init__(
        self,
        estimator: CardinalityGenerator,
        distortion_factor: float,
        *,
        distortion_strategy: Literal["fixed", "random"] = "fixed",
    ) -> None:
        super().__init__(estimator.allow_cross_products)
        self.estimator = estimator
        self.distortion_factor = distortion_factor
        self.distortion_strategy = distortion_strategy

    def describe(self) -> dict:
        return {
            "name": "cardinality-distortion",
            "estimator": "distortion",
            "distortion_factor": self.distortion_factor,
            "distortion_strategy": self.distortion_strategy,
        }

    def calculate_estimate(
        self, query: SqlQuery, tables: TableReference | Iterable[TableReference]
    ) -> Cardinality:
        tables = util.enlist(tables)
        card_est = self.estimator.calculate_estimate(query, tables)
        if not card_est.is_valid():
            return Cardinality.unknown()
        if self.distortion_strategy == "fixed":
            distortion_factor = self.distortion_factor
        elif self.distortion_strategy == "random":
            distortion_factor = random.uniform(
                min(self.distortion_factor, 1.0), max(self.distortion_factor, 1.0)
            )
        else:
            raise ValueError(
                f"Unknown distortion strategy: '{self.distortion_strategy}'"
            )
        return round(card_est * distortion_factor)
