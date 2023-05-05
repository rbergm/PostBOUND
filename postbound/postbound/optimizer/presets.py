"""Presets allow to set up optimization pipelines quickly by providing combinations of different algorithms."""
from __future__ import annotations

import abc
from typing import Optional

from postbound.db import db
from postbound.db.systems import systems
from postbound.qal import parser
from postbound.optimizer import validation
from postbound.optimizer.bounds import scans
from postbound.optimizer.joinorder import enumeration
from postbound.optimizer.physops import selection
from postbound.optimizer.strategies import ues
from postbound.optimizer.planmeta import parameterization as plan_param


def apply_standard_system_options(database: Optional[db.Database] = None) -> None:
    """Configures a number of typically used settings for the query optimization process.

    This method requires that a working database connection has been set up. If it is not supplied directly, then it
    is retrieved from the `DatabasePool`.

    Currently, these settings include:

    - disabling cached query execution for the current database
    - enabling cached query execution for all statistics-related queries in the database
    - using emulated statistics instead of the native database statistics for better reproducibility (this is why we
    need cached query execution for the statistics queries)
    - enabling auto-binding of columns when parsing queries since we have a working database connection anyway
    """
    database = database if database else db.DatabasePool.get_instance().current_database()
    database.cache_enabled = False
    database.statistics().emulated = True
    database.statistics().cache_enabled = True
    parser.auto_bind_columns = True


class OptimizationSettings(abc.ABC):
    """Captures related settings for the `OptimizationPipeline` to make them more easily accessible.

    All components are optional, depending on the specific optimization scenario/approach.
    """

    @abc.abstractmethod
    def query_pre_check(self) -> validation.OptimizationPreCheck | None:
        """The required query pre-check."""
        raise NotImplementedError

    @abc.abstractmethod
    def build_join_order_optimizer(self) -> enumeration.JoinOrderOptimizer | None:
        """The algorithm that is used to obtain the optimized join order."""
        raise NotImplementedError

    @abc.abstractmethod
    def build_physical_operator_selection(self) -> selection.PhysicalOperatorSelection | None:
        """The algorithm that is used to determine the physical operators."""
        raise NotImplementedError

    @abc.abstractmethod
    def build_plan_parameterization(self) -> plan_param.ParameterGeneration | None:
        """The algorithm that is used to further parameterize the query plan."""
        raise NotImplementedError


class UESOptimizationSettings(OptimizationSettings):
    """Provides the optimization settings that are used for the UES query optimizer.

    See Hertzschuch et al.: "Simplicity Done Right for Join Ordering", CIDR'2021 for details.
    """

    def __init__(self, database: Optional[db.Database] = None):
        self.database = database if database else db.DatabasePool.get_instance().current_database()

    def query_pre_check(self) -> validation.OptimizationPreCheck | None:
        return validation.UESOptimizationPreCheck()

    def build_join_order_optimizer(self) -> enumeration.JoinOrderOptimizer | None:
        base_table_estimator = scans.NativeCardinalityEstimator(self.database)
        join_cardinality_estimator = ues.UESJoinBoundEstimator()
        subquery_policy = ues.UESSubqueryGenerationPolicy()
        stats_container = ues.MaxFrequencyStatsContainer(self.database.statistics())
        enumerator = ues.UESJoinOrderOptimizer(base_table_estimation=base_table_estimator,
                                               join_estimation=join_cardinality_estimator,
                                               subquery_policy=subquery_policy,
                                               stats_container=stats_container,
                                               database=self.database)
        return enumerator

    def build_physical_operator_selection(self) -> selection.PhysicalOperatorSelection | None:
        return ues.UESOperatorSelection(systems.DatabaseSystemRegistry.load_system_for(self.database))

    def build_plan_parameterization(self) -> plan_param.ParameterGeneration | None:
        return None


def fetch(key: str, *, database: Optional[db.Database] = None) -> OptimizationSettings:
    """Provides the optimization settings registered under the given key. Keys are case-insensitive.

    Currently supported settings are:

    - `UESOptimizationSettings`, available under key "ues"
    """
    if key.upper() == "UES":
        return UESOptimizationSettings(database)
    else:
        raise ValueError(f"Unknown presets for key '{key}'")
