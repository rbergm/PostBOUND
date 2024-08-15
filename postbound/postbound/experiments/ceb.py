"""
Implementation of the Cardinality Estimation Benchmark (CEB) algorithm to automatically generate workload queries based on
different templates.

References
----------
[1] Parimarjan Negi et al.: "Flow-Loss: Learning Cardinality Estimates That Matter" (PVLDB 2021)
"""

from __future__ import annotations

import pathlib
import random
from collections.abc import Iterable, Sequence
from typing import Any, Literal, Optional

import tomli

from .workloads import Workload
from .. db.db import Database
from ..qal import parser
from .. qal.qal import SqlQuery


class PredicateGenerator:

    @staticmethod
    def load_from_toml(contents: dict) -> PredicateGenerator:
        pass

    def __init__(self, value_type: Literal["sql", "list"], *,
                 db_connection: Database,
                 query_column: str,
                 template_key: str,
                 pred_type: Literal["like", "ilike", "=", "!=", "<", ">", "<=", ">="],
                 sampling_method: Literal["uniform", "quantile"],
                 sql_query: Optional[Sequence[SqlQuery]] = None,
                 list_options: Optional[Sequence[Any]] = None,
                 num_quantiles: Optional[int] = None) -> None:
        self.db_connection = db_connection
        self.value_type = value_type
        self.query_column = query_column
        self.template_key = template_key
        self.pred_type = pred_type
        self.sampling_method = sampling_method
        self.sql_query = sql_query
        self.list_options = list_options
        self.num_quantiles = num_quantiles

    def full_predicate_text(self) -> str:
        candidate_values = self._obtain_candidate_values()
        selected_value = self._choose_value(candidate_values)
        return f"{self.query_column} {self.pred_type} {selected_value}"

    def _obtain_candidate_values(self) -> Sequence[Any]:
        if self.value_type == "list":
            candidate_values = self.list_options
        elif self.value_type == "sql":
            # we utilize the query simplifcation rules here:
            # this should already be the list of possible values, no further unwrapping required.
            candidate_values = self.db_connection.execute_query(self.sql_query)
        else:
            raise ValueError(f"Unknown value type: '{self.value_type}'")

        return candidate_values

    def _choose_value(self, candidate_values: Sequence[Any]) -> Any:
        if self.sampling_method == "uniform":
            candidate_values = list(set(candidate_values))
            return random.choice(candidate_values)

        if self.sampling_method != "quantile":
            raise ValueError(f"Unknown sampling method: '{self.sampling_method}'")
        if self.num_quantiles is None:
            raise ValueError("Number of quantiles is required")

        candidate_values = sorted(candidate_values)
        quantile_size = len(candidate_values) // self.num_quantiles
        quantile_idx = random.randint(0, self.num_quantiles - 1)
        lower_candidate_idx = quantile_idx * quantile_size
        upper_candidate_idx = (quantile_idx + 1) * quantile_size
        selected_quantile = candidate_values[lower_candidate_idx:upper_candidate_idx]

        return random.choice(selected_quantile)


class QueryTemplate:

    @staticmethod
    def load_from_toml(contents: dict) -> QueryTemplate:
        pass

    def __init__(self, base_query: str, *, label: str, predicates: Iterable[PredicateGenerator]) -> None:
        self.label = label
        self.base_query = base_query
        self.predicates = predicates

    def generate_query(self) -> str:
        query = self.base_query
        for pred in self.predicates:
            query = query.replace(pred.template_key, pred.full_predicate_text())
        return parser.parse_query(query)


def generate_workload(path: str | pathlib.Path, *, queries_per_template: int,
                      template_pattern: str = "*.toml") -> Workload[str]:
    pass


def persist_workload(path: str | pathlib.Path, workload: Workload[str]) -> None:
    pass
