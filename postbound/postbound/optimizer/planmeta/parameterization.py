from __future__ import annotations

import abc

from typing import Optional

from postbound.qal import qal
from postbound.optimizer import data
from postbound.optimizer.physops import operators as physops
from postbound.optimizer.planmeta import hints as plan_params


class ParameterGeneration(abc.ABC):

    @abc.abstractmethod
    def generate_plan_parameters(self, query: qal.SqlQuery, join_order: Optional[data.JoinTree],
                                 operator_assignment: Optional[physops.PhysicalOperatorAssignment]
                                 ) -> Optional[plan_params.PlanParameterization]:
        raise NotImplementedError


class EmptyParameterization(ParameterGeneration):
    def generate_plan_parameters(self, query: qal.SqlQuery, join_order: Optional[data.JoinTree],
                                 operator_assignment: Optional[physops.PhysicalOperatorAssignment]
                                 ) -> Optional[plan_params.PlanParameterization]:
        return None
