
import random
import warnings
from typing import Optional

from .._hints import PhysicalOperatorAssignment
from ..jointree import LogicalJoinTree
from ..._stages import PhysicalOperatorSelection
from ... import qal
from ...experiments import workloads


warnings.warn("FASTgres implementation is not yet functional", FutureWarning)


def hints_to_operators(hints) -> PhysicalOperatorAssignment:
    pass


def determine_model_file(workload: workloads.Workload) -> str:
    return f"fastgres_model_{workload.name}.model"


def load_model_from_path(path: str):
    pass


class FastgresOperatorSelection(PhysicalOperatorSelection):

    def __init__(self, workload: workloads.Workload, *, rand_seed: float = random.random()) -> None:
        # TODO: fastgres initialization, model loading, sanity checks
        self.model = None
        random.seed(rand_seed)
        super().__init__()

    def select_physical_operators(self, query: qal.SqlQuery,
                                  join_order: Optional[LogicalJoinTree]) -> PhysicalOperatorAssignment:
        hints = self.model.predict(str(query))
        assignment = hints_to_operators(hints)
        return assignment
