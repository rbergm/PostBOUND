
import random
from typing import Optional

from postbound.experiments import workloads
from postbound.qal import qal
from postbound.optimizer import jointree, physops, stages


def hints_to_operators(hints) -> physops.PhysicalOperatorAssignment:
    pass


def determine_model_file(workload: workloads.Workload) -> str:
    return f"fastgres_model_{workload.name}.model"


def load_model_from_path(path: str):
    pass


class FastgresOperatorSelection(stages.PhysicalOperatorSelection):

    def __init__(self, workload: workloads.Workload, *, rand_seed: float = random.random()) -> None:
        # TODO: fastgres initialization, model loading, sanity checks
        self.model = None
        random.seed(rand_seed)
        super().__init__()

    def select_physical_operators(self, query: qal.SqlQuery,
                                  join_order: Optional[jointree.LogicalJoinTree | jointree.PhysicalQueryPlan]
                                  ) -> physops.PhysicalOperatorAssignment:
        hints = self.model.predict(str(query))
        assignment = hints_to_operators(hints)
        return assignment
