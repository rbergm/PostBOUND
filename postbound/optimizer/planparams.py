"""Models metadata about query plans that is not directly concerned with join orders and physical operators."""
from __future__ import annotations

import enum
from typing import Any, Iterable

from postbound.qal import base


class PlanParameterization:
    """The plan parameterization stores metadata that is assigned to different parts of the plan.

    Currently, three types of parameters are supported:

    - `cardinality_hints` provide specific cardinality estimates for individual joins or tables. These can be used to overwrite
      the estimation of the native database system
    - `parallel_worker_hints` indicate how many worker processes should be used to execute individual joins or table
      scans (assuming that the selected operator can be parallelized). Notice that this can also be indicated as part of the
      `physops` module
    - `system_specific_settings` can be used to enable or disable specific optimization or execution features of the target
      database. For example, they can be used to disable parallel execution or switch to another cardinality estimation method.
      Such settings should be used sparingly since they defeat the purpose of optimization algorithms that are independent of
      specific database systems.

    Although it is allowed to modify the different dictionaries directly, the more high-level methods should be used instead.
    This ensures that all potential (future) invariants are maintained.

    Attributes
    ----------
    cardinality_hints : dict[frozenset[base.TableReference], int | float]
        Contains the cardinalities for individual joins and scans. This is always the cardinality that is emitted by a specific
        operator. All joins are identified by the base tables that they combine. Keys of single tables correpond to scans.
    paralell_worker_hints : dict[frozenset[base.TableReference], int]
        Contains the number of parallel processes that should be used to execute a join or scan. All joins are identified by
        the base tables that they combine. Keys of single tables correpond to scans. "Processes" does not necessarily mean
        "system processes". The database system can also choose to use threads or other means of parallelization. This is not
        restricted by the join assignment.
    system_specific_settings : dict[str, Any]
        Contains the settings for the target database system. The keys and values, as well as their usage depend entirely on
        the system. For example, in Postgres a setting like *enable_geqo = 'off'* can be used to disable the genetic optimizer.
        During query execution, this is applied as preparatory statement before the actual query is executed.
    """

    def __init__(self) -> None:
        self.cardinality_hints: dict[frozenset[base.TableReference], int | float] = {}
        self.parallel_worker_hints: dict[frozenset[base.TableReference], int] = {}
        self.system_specific_settings: dict[str, Any] = {}

    def add_cardinality_hint(self, tables: Iterable[base.TableReference], cardinality: int | float) -> None:
        """Assigns a specific cardinality hint to a (join of) tables.

        Parameters
        ----------
        tables : Iterable[base.TableReference]
            The tables for which the hint is generated. This can be an iterable of a single table, which denotes a scan hint.
        cardinality : int | float
            The estimated or known cardinality.
        """
        self.cardinality_hints[frozenset(tables)] = cardinality

    def add_parallelization_hint(self, tables: Iterable[base.TableReference], num_workers: int) -> None:
        """Assigns a specific number of parallel workers to a (join of) tables.

        How these workers are implemented depends on the database system. They could become actual system processes, threads,
        etc.

        Parameters
        ----------
        tables : Iterable[base.TableReference]
            The tables for which the hint is generated. This can be an iterable of a single table, which denotes a scan hint.
        num_workers : int
            The desired number of worker processes. This denotes the total number of processes, not an additional amount. For
            some database systems this is an important distinction since one operator node will always be created. This node
            is then responsible for spawning the workers, but can also take part in the actual calculation. To prevent one-off
            errors, we standardize this number to denote the total number of workers that take part in the calculation.
        """
        self.parallel_worker_hints[frozenset(tables)] = num_workers

    def set_system_settings(self, setting_name: str = "", setting_value: Any = None, **kwargs) -> None:
        """Stores a specific system setting.

        This may happen in one of two ways: giving the setting name and value as two different parameters, or combining their
        assignment in the keyword parameters. While the first is limited to a single parameter, the second can be used to
        assign an arbitrary number of settings. However, this is limited to setting names that form valid keyword names.

        Parameters
        ----------
        setting_name : str, optional
            The name of the setting when using the separate key/value assignment mode. Defaults to an empty string to enable
            the integrated keyword parameter mode.
        setting_value : Any, optional
            The setting's value when using the separate key/value assignment mode. Defaults to ``None`` to enable the
            integrated keyword parameter mode.
        **kwargs
            The key/value pairs in the integrated keyword parameter mode.

        Raises
        ------
        ValueError
            If both the `setting_name` as well as keyword arguments are given
        ValueError
            If neither the `setting_name` nor keyword arguments are given

        Examples
        --------
        Using the separate setting name and value syntax: ``set_system_settings("join_collapse_limit", 1)``
        Using the kwargs syntax: ``set_system_settings(join_collapse_limit=1, jit=False)``
        Both examples are specific to Postgres (see https://www.postgresql.org/docs/current/runtime-config-query.html).
        """
        if setting_name and kwargs:
            raise ValueError("Only setting or kwargs can be supplied")
        elif not setting_name and not kwargs:
            raise ValueError("setting_name or kwargs required!")

        if setting_name:
            self.system_specific_settings[setting_name] = setting_value
        else:
            self.system_specific_settings |= kwargs

    def merge_with(self, other_parameters: PlanParameterization) -> PlanParameterization:
        """Combines the current parameters with additional hints.

        In case of assignments to the same hints, the values from the other parameters take precedence. None of the input
        parameterizations are modified.

        Parameters
        ----------
        other_parameters : PlanParameterization
            The parameterization to combine with the current parameterization

        Returns
        -------
        PlanParameterization
            The merged parameters
        """
        merged_params = PlanParameterization()
        merged_params.cardinality_hints = self.cardinality_hints | other_parameters.cardinality_hints
        merged_params.parallel_worker_hints = self.parallel_worker_hints | other_parameters.parallel_worker_hints
        merged_params.system_specific_settings = (self.system_specific_settings
                                                  | other_parameters.system_specific_settings)
        return merged_params

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return (f"PlanParams(cards={self.cardinality_hints}, "
                f"system specific={self.system_specific_settings}, par workers={self.parallel_worker_hints})")


class HintType(enum.Enum):
    """Contains all hint types that are supported by PostBOUND.

    Notice that not all of these hints need to be represented in the `PlanParameterization`, since some of them concern other
    aspects such as the join order. Furthermore, not all database systems will support all operators. The availability of
    certain hints can be checked on the database system interface and should be handled as part of the optimization pre-checks.
    """
    JoinOrderHint = "Join order"
    JoinDirectionHint = "Join direction"
    JoinSubqueryHint = "Bushy join order"
    OperatorHint = "Physical operators"
    ParallelizationHint = "Par. workers"
    CardinalityHint = "Cardinality"
