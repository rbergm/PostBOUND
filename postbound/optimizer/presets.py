"""Presets allow to set up optimization pipelines quickly by providing pre-defined combinations of different algorithms.

The current design of the presets is targeted at the `TwoStageOptimizationPipeline`, since this one requires the most setup.
"""
from __future__ import annotations

from typing import Literal, Optional

from postbound.qal import parser
from ._pipelines import (
    OptimizationSettings,
    JoinOrderOptimization, PhysicalOperatorSelection,
    OptimizationPreCheck
)
from .policies import cardinalities
from .strategies import ues, native
from .. import db


def apply_standard_system_options(database: Optional[db.Database] = None) -> None:
    """Configures a number of typically used settings for the query optimization process.

    This method requires that a working database connection has been set up. If it is not supplied directly, it is retrieved
    from the `DatabasePool`.

    Currently, the applied settings include:

    - disabling cached query execution for the current database
    - enabling cached query execution for all statistics-related queries in the database
    - using emulated statistics instead of the native database statistics for better reproducibility (this is why we
      need cached query execution for the statistics queries)
    - enabling auto-binding of columns when parsing queries since we have a working database connection anyway

    Parameters
    ----------
    database : Optional[db.Database], optional
        The database that should be configured. Defaults to ``None``, in which case the system is loaded from the
        `DatabasePool`.
    """
    database = database if database else db.DatabasePool.get_instance().current_database()
    database.cache_enabled = False
    database.statistics().emulated = True
    database.statistics().cache_enabled = True
    parser.auto_bind_columns = True


class UESOptimizationSettings(OptimizationSettings):
    """Provides the optimization settings that are used for the UES query optimizer.

    Parameters
    ----------
    database : Optional[db.Database], optional
        The database for which the optimized queries should be executed. This is necessary to initialize the optimization
        strategies correctly. Defaults to ``None``, in which case the database will be inferred from the `DatabasePool`.

    References
    ----------

    .. Hertzschuch et al.: "Simplicity Done Right for Join Ordering", CIDR'2021
    """

    def __init__(self, database: Optional[db.Database] = None):
        self.database = database if database else db.DatabasePool.get_instance().current_database()

    def query_pre_check(self) -> Optional[OptimizationPreCheck]:
        return ues.UESOptimizationPreCheck

    def build_join_order_optimizer(self) -> Optional[JoinOrderOptimization]:
        base_table_estimator = cardinalities.NativeCardinalityEstimator(self.database)
        join_cardinality_estimator = ues.UESJoinBoundEstimator()
        subquery_policy = ues.UESSubqueryGenerationPolicy()
        stats_container = ues.MaxFrequencyStatsContainer(self.database.statistics())
        enumerator = ues.UESJoinOrderOptimizer(base_table_estimation=base_table_estimator,
                                               join_estimation=join_cardinality_estimator,
                                               subquery_policy=subquery_policy,
                                               stats_container=stats_container,
                                               database=self.database)
        return enumerator

    def build_physical_operator_selection(self) -> Optional[PhysicalOperatorSelection]:
        return ues.UESOperatorSelection(self.database)


class NativeOptimizationSettings(OptimizationSettings):
    """Provides the optimization settings to use plans from the native optimizer of a database system.

    Parameters
    ----------
    database : Optional[db.Database], optional
        The database from which the query plans should be retrieved. Defaults to ``None``, in which case the database will be
        inferred from the `DatabasePool`.
    """
    def __init__(self, database: Optional[db.Database] = None) -> None:
        self.database = database

    def build_join_order_optimizer(self) -> Optional[JoinOrderOptimization]:
        return native.NativeJoinOrderOptimizer(self.database)

    def build_physical_operator_selection(self) -> Optional[PhysicalOperatorSelection]:
        return native.NativePhysicalOperatorSelection(self.database)


def fetch(key: Literal["ues", "native"], *, database: Optional[db.Database] = None) -> OptimizationSettings:
    """Provides the optimization settings registered under a specific key.

    Currently supported settings are:

    - `UESOptimizationSettings`, available under key ``"ues"``
    - `NativeOptimizationSettings`, available under key ``"native"``

    All registration happens statically and cannot be changed at runtime.

    Parameters
    ----------
    key : Literal["ues"]
        The key which was used to register the optimization strategy. The comparison happens case-insensitively. Therefore, the
        key can be written unsing any casing.
    database : Optional[db.Database], optional
        The database that is used to optimize and/or execute the optimized queries. The precise usage of this parameter
        depends on the specific optimization strategy and should be documented there. There could also be optimization
        strategies that do not use the this parameter at all. Defaults to ``None``, in which case the behavior once again
        depends on the selected optimization strategy. Typically, the database is inferred from the `DatabasePool` then.

    Returns
    -------
    OptimizationSettings
        The optimization settings that were registered under the given key

    Raises
    ------
    ValueError
        If the key is none of of the allowed values.
    """
    if key.upper() == "UES":
        return UESOptimizationSettings(database)
    elif key == "native":
        return NativeOptimizationSettings(database)
    else:
        raise ValueError(f"Unknown presets for key '{key}'")
