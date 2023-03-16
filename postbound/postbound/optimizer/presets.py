from __future__ import annotations

import abc

from postbound.db import db
from postbound.db.systems import systems
from postbound.qal import parser
from postbound.optimizer import validation
from postbound.optimizer.bounds import scans, joins, stats
from postbound.optimizer.joinorder import enumeration, subqueries
from postbound.optimizer.physops import selection
from postbound.optimizer.planmeta import hints as plan_param


def apply_standard_system_options() -> None:
    database = db.DatabasePool.get_instance().current_database()
    database.statistics().emulated = True
    database.statistics().cache_enabled = True
    parser.auto_bind_columns = True


class OptimizationSettings(abc.ABC):

    @abc.abstractmethod
    def query_pre_check(self) -> validation.OptimizationPreCheck | None:
        raise NotImplementedError

    @abc.abstractmethod
    def build_join_order_optimizer(self) -> enumeration.JoinOrderOptimizer | None:
        raise NotImplementedError

    @abc.abstractmethod
    def build_physical_operator_selection(self) -> selection.PhysicalOperatorSelection | None:
        raise NotImplementedError

    @abc.abstractmethod
    def build_plan_parameterization(self) -> plan_param.PlanParameterization | None:
        raise NotImplementedError


class UESOptimizationSettings(OptimizationSettings):

    def __init__(self, database: db.Database | None = None):
        self.database = database if database else db.DatabasePool.get_instance().current_database()

    def query_pre_check(self) -> validation.OptimizationPreCheck | None:
        return validation.UESOptimizationPreCheck()

    def build_join_order_optimizer(self) -> enumeration.JoinOrderOptimizer | None:
        base_table_estimator = scans.DBCardinalityEstimator(self.database)
        join_cardinality_estimator = joins.UESJoinBoundEstimator()
        subquery_policy = subqueries.UESSubqueryGenerationPolicy()
        stats_container = stats.MaxFrequencyStatsContainer(self.database.statistics())
        enumerator = enumeration.UESJoinOrderOptimizer(base_table_estimation=base_table_estimator,
                                                       join_estimation=join_cardinality_estimator,
                                                       subquery_policy=subquery_policy,
                                                       stats_container=stats_container,
                                                       database=self.database)
        return enumerator

    def build_physical_operator_selection(self) -> selection.PhysicalOperatorSelection | None:
        return selection.UESOperatorSelection(systems.DatabaseSystemRegistry.load_system_for(self.database))

    def build_plan_parameterization(self) -> plan_param.PlanParameterization | None:
        return None


def fetch(key: str) -> OptimizationSettings:
    if key.upper() == "UES":
        return UESOptimizationSettings()
    else:
        raise ValueError(f"Unknown presets for key '{key}'")
