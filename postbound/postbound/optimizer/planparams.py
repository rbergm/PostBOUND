"""Provides the central data structure that stores the actual plan parameters."""
from __future__ import annotations

import enum
from typing import Any, Iterable

from postbound.qal import base


class PlanParameterization:
    """The `PlanParameterization` stores the parameters that are assigned to different parts of the plan.

    Currently, three types of parameters are supported:

    - `cardinality_hints` provide specific cardinality estimates for individual joins or tables that overwrite the
    estimation of the native database system
    - `parallel_worker_hints` indicate how many worker processes should be used to execute individual joins or table
    scans (assuming that the selected operator can be parallelized)
    - `system_specific_settings` do not fit in any of the above categories and only work for a known target database
    system. These should be used sparingly since they defeat the purpose of optimization algorithms that are
    independent of specific database systems. For example, they could modify the assignment strategy of the native
    database system. Their value also depends on the specific setting.
    """

    def __init__(self) -> None:
        self.cardinality_hints: dict[frozenset[base.TableReference], int | float] = {}
        self.parallel_worker_hints: dict[frozenset[base.TableReference], int] = {}
        self.system_specific_settings: dict[str, Any] = {}

    def add_cardinality_hint(self, tables: Iterable[base.TableReference], cardinality: int | float) -> None:
        """Assigns the given cardinality hint to the (join of) tables."""
        self.cardinality_hints[frozenset(tables)] = cardinality

    def add_parallelization_hint(self, tables: Iterable[base.TableReference], num_workers: int) -> None:
        """Assigns the given number of parallel works to the (join of) tables."""
        self.parallel_worker_hints[frozenset(tables)] = num_workers

    def set_system_settings(self, setting_name: str = "", setting_value: Any = None, **kwargs) -> None:
        """Stores the given system setting.

        This may happen in one of two ways: giving the setting name and value as two different parameters, or combining
        their assignment in the keyword parameters. While the first is limited to a single parameter, the second can
        be used to assign an arbitrary number of settings. However, this is limited to setting names that form valid
        keyword names.

        Example usage with separate setting name and value: `set_system_settings("join_collapse_limit", 1)`
        Example usage with kwargs: `set_system_settings(join_collapse_limit=1, jit=False)`
        Both examples are specific to Postgres (see https://www.postgresql.org/docs/current/runtime-config-query.html).
        """
        # TODO: system settings should be moved to the plan parameters
        if setting_name and kwargs:
            raise ValueError("Only setting or kwargs can be supplied")
        elif not setting_name and not kwargs:
            raise ValueError("setting_name or kwargs required!")

        if setting_name:
            self.system_specific_settings[setting_name] = setting_value
        else:
            self.system_specific_settings |= kwargs

    def merge_with(self, other_parameters: PlanParameterization) -> PlanParameterization:
        """Combines the plan parameters with the settings from the `other_parameters`.

        In case of contradicting parameters, the `other_parameters` take precedence.
        """
        merged_params = PlanParameterization()
        merged_params.cardinality_hints = self.cardinality_hints | other_parameters.cardinality_hints
        merged_params.parallel_worker_hints = self.parallel_worker_hints | other_parameters.parallel_worker_hints
        merged_params.system_specific_settings = (self.system_specific_settings
                                                  | other_parameters.system_specific_settings)
        return merged_params


class HintType(enum.Enum):
    """Contains all hint types that are supported by PostBOUND (or at least should be supported in the future)."""
    JoinOrderHint = "Join order"
    JoinDirectionHint = "Join direction"
    JoinSubqueryHint = "Bushy join order"
    ParallelizationHint = "Par. workers"
    CardinalityHint = "Cardinality"
